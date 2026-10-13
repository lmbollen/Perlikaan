"""Playing together through the relay, with an in-memory stand-in for Ably
(and, when ABLY_KEY is set, against Ably itself)."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from perudo_app import relay                                           # noqa: E402
from perudo_app.multiplayer import LobbyClient, LobbyHost, join_lobby  # noqa: E402

from test_multiplayer import _play_people, key, registry, wait         # noqa: E402


LIVE = os.environ.get("PERUDO_RELAY") == "ably"     # run these against Ably itself


@pytest.fixture
def backend():
    if LIVE:
        if not relay.ably_key():
            pytest.skip("PERUDO_RELAY=ably needs ABLY_KEY")
        return relay.AblyBackend(relay.ably_key())
    return relay.MemoryBackend(delay=0.002)


@pytest.fixture
def host(backend):
    h = LobbyHost("Anna", registry(), guest_bot_limit=2, relay=backend)
    yield h
    h.close()


def test_a_link_batches_and_keeps_order():
    backend = relay.MemoryBackend(delay=0.002)          # counts requests; Ably can't
    got = []
    sub = backend.subscribe("c", lambda name, data: got.extend(relay._parse(data)), None)
    link = relay.Link(backend, "c", "me", lambda *_: None, lambda *_: None)
    for i in range(300):
        link.send({"t": "n", "i": i})
    assert wait(lambda: len([m for m in got if m.get("t") == "n"]) == 300)
    assert [m["i"] for m in got if m.get("t") == "n"] == list(range(300))
    assert backend.published < 300                      # many messages, fewer requests
    link.close()
    sub.close()


def test_a_big_burst_is_split_under_the_message_limit():
    batch = [{"t": "x", "pad": "y" * 1000} for _ in range(200)]
    chunks = relay._chunks(batch)
    assert len(chunks) > 1 and all(len(c) <= relay.MAX_BATCH + 2 for c in chunks)
    assert sum(len(json.loads(c)) for c in chunks) == 200


def test_a_code_in_use_is_not_handed_out_twice(backend):
    first = relay.RelayHost(backend, lambda *_: None, lambda *_: None, code="ABCD", claim_wait=0.3)
    try:
        with pytest.raises(ConnectionError):
            relay.RelayHost(backend, lambda *_: None, lambda *_: None, code="ABCD", claim_wait=0.3)
        other = relay.RelayHost(backend, lambda *_: None, lambda *_: None, claim_wait=0.3)
        assert other.code != "ABCD"
        other.close()
    finally:
        first.close()


def test_a_game_over_the_internet(host, backend):
    reg_ben = registry()
    ben = LobbyClient("Ben", reg_ben, host.code, relay=backend)
    try:
        assert wait(lambda: sorted(m["name"] for m in host.state()["members"]) == ["Anna", "Ben"])
        ben.add_bot(key(reg_ben, "MinRaiseBot"), "MinRaise Bot")
        host.add_bot(key(host.registry, "RandomBot"), "Random Bot")
        assert wait(lambda: len(host.state()["seats"]) == 2)
        session = host.start_game(delay=0.0)
        assert wait(lambda: ben.session is not None)
        _play_people(host, [ben], until=lambda: session.snapshot().finished)
        final = session.snapshot()
        assert final.finished and final.error is None
        assert wait(lambda: ben.session.snapshot().finished)
        assert ben.session.snapshot().standings == final.standings
        moved = {f.player for f in final.feed if f.kind in ("bid", "dudo", "calza")}
        assert "MinRaise Bot (Ben)" in moved
    finally:
        ben.close()
    assert wait(lambda: len(host.state()["members"]) == 1)


def test_wrong_codes_and_leaving(host, backend):
    with pytest.raises(ConnectionError):
        LobbyClient("Ben", registry(), "ZZZZ" if host.code != "ZZZZ" else "YYYY", relay=backend, timeout=1.0)
    ben = LobbyClient("Ben", registry(), host.code, relay=backend)
    host.close()
    assert wait(lambda: ben.closed and "host left" in (ben.error or ""))


def test_a_guest_that_goes_silent_has_left(host, backend, monkeypatch):
    monkeypatch.setattr(relay, "GONE_AFTER", 1.0)
    ben = LobbyClient("Ben", registry(), host.code, relay=backend)
    assert wait(lambda: len(host.state()["members"]) == 2)
    ben.conn.closed = True                              # the phone vanishes: no bye, no pings
    assert wait(lambda: len(host.state()["members"]) == 1, timeout=10)


def test_joining_finds_the_lobby_where_it_is(host, backend):
    local = LobbyHost("Carla", registry(), ip="127.0.0.1")
    try:
        online = join_lobby("Ben", registry(), host.code, relay=backend, local_ips=["127.0.0.1"])
        assert online.online
        nearby = join_lobby("Dirk", registry(), local.code, relay=backend, local_ips=["127.0.0.1"], timeout=1.5)
        assert not nearby.online
        online.close()
        nearby.close()
        with pytest.raises(ConnectionError, match="No lobby"):
            join_lobby("Eve", registry(), "QQQQ", relay=backend, local_ips=["127.0.0.1"], timeout=1.0)
    finally:
        local.close()


@pytest.mark.skipif(not relay.ably_key(), reason="ABLY_KEY is not set")
def test_through_ably_itself():
    backend = relay.AblyBackend(relay.ably_key())
    host = LobbyHost("Anna", registry(), relay=backend)
    try:
        ben = LobbyClient("Ben", registry(), host.code, relay=backend, timeout=15)
        assert wait(lambda: len(host.state()["members"]) == 2, timeout=15)
        ben.rename("Bennie")
        assert wait(lambda: any(m["name"] == "Bennie" for m in host.state()["members"]), timeout=15)
        ben.close()
        assert wait(lambda: len(host.state()["members"]) == 1, timeout=15)
    finally:
        host.close()


def test_bot_talk_is_said_on_every_phone(host, backend, monkeypatch):
    """A line any bot says (the host's, or a guest's) is shown and said on
    the host and on every guest."""
    from perudo_app import core
    monkeypatch.setattr(core.GameSession, "TALK_CHANCE", 1.0)
    host_heard, ben_heard, carla_heard = [], [], []
    monkeypatch.setattr(core.ROUTER, "speak", lambda text, lang, seed: host_heard.append((seed, text)))
    # One process plays every phone here, and the talk router is one per
    # process: keep it with the host's game (on a phone, each has its own).
    attach = core.ROUTER.attach
    monkeypatch.setattr(core.ROUTER, "attach", lambda listener, thread: None
                        if isinstance(getattr(listener, "__self__", None), LobbyClient) else attach(listener, thread))
    ben = LobbyClient("Ben", registry(), host.code, relay=backend,
                      speak=lambda text, lang, seed: ben_heard.append((seed, text)))
    carla = LobbyClient("Carla", registry(), host.code, relay=backend,
                        speak=lambda text, lang, seed: carla_heard.append((seed, text)))
    try:
        host.add_bot(key(host.registry, "RandomBot"), "Random Bot")
        assert wait(lambda: len(host.state()["seats"]) == 1 and len(host.state()["members"]) == 3)
        session = host.start_game(delay=0.0)
        assert wait(lambda: ben.session is not None and carla.session is not None)
        ben._on_bot_talk("Ben's Bot", "Hallo allemaal", "nl")        # a guest's bot says something
        _play_people(host, [ben, carla], until=lambda: session.snapshot().finished)
        assert wait(lambda: ("Ben's Bot", "Hallo allemaal") in carla_heard)
        for heard in (host_heard, ben_heard, carla_heard):
            assert ("Ben's Bot", "Hallo allemaal") in heard
            assert any(seed == "Random Bot" for seed, _ in heard)    # the host's bot, heard everywhere
    finally:
        ben.close()
        carla.close()
