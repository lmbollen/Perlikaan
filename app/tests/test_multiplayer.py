"""Playing together, with a host and guests in one process over real sockets."""

import json
import random
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from perudo.game import PerudoGame                                     # noqa: E402
from perudo.models import Bid, Dudo                                    # noqa: E402
from perudo_app import net                                             # noqa: E402
from perudo_app.core import BotRegistry                                # noqa: E402
from perudo_app.multiplayer import LobbyClient, LobbyHost, fallback_action   # noqa: E402

LOCAL = ["127.0.0.1"]


def wait(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def registry():
    reg = BotRegistry()
    reg.scan_bundled()
    return reg


def key(reg, class_name):
    return next(reg.key(e) for e in reg.entries if e.class_name == class_name)


@pytest.fixture
def host():
    h = LobbyHost("Anna", registry(), guest_bot_limit=2, ip="127.0.0.1")
    yield h
    h.close()


# ---------------------------------------------------------------------------

def test_codes_say_where_the_host_is():
    for ip in ("192.168.1.23", "10.0.244.7", "172.20.3.255", "127.0.0.1"):
        for slot in (0, 7, 15):
            code = net.make_code(ip, slot)
            assert len(code) == 4 and all(ch in net.ALPHABET for ch in code)
            c, d, s = net.read_code(code.lower())
            assert (c, d, s) == (int(ip.split(".")[2]), int(ip.split(".")[3]), slot)
    code = net.make_code("192.168.1.23", 3)
    assert net.candidates(code, ["192.168.1.40", "10.1.2.3"]) == ["192.168.1.23", "10.1.1.23"]
    with pytest.raises(ValueError):
        net.read_code("AB0")


def test_every_observation_and_event_survives_the_trip():
    reg = registry()
    seen = []
    bots = reg.make([e for e in reg.entries if e.class_name in ("RandomBot", "MinRaiseBot", "ExampleBot")])
    PerudoGame(bots=bots, seed=4, before_turn=seen.append, listeners=[seen.append]).play()
    assert len(seen) > 50
    for thing in seen:
        assert net.decode(json.loads(json.dumps(net.encode(thing)))) == thing


def test_the_lobby_names_bots_and_rules(host):
    reg = registry()
    ben = LobbyClient("Ben", reg, host.code, local_ips=LOCAL)
    ben2 = LobbyClient("Ben", reg, host.code, local_ips=LOCAL)
    try:
        assert wait(lambda: len(host.state()["members"]) == 3)
        assert sorted(m["name"] for m in host.state()["members"]) == ["Anna", "Ben", "Ben 2"]
        ben2.rename("Carla")
        assert wait(lambda: any(m["name"] == "Carla" for m in ben.lobby.get("members", [])))
        for _ in range(3):
            ben.add_bot(key(reg, "RandomBot"), "Random Bot")
        assert wait(lambda: ben.notice is not None)
        assert "at most 2" in ben.notice
        assert sum(1 for s in host.state()["seats"] if s["owner_name"] == "Ben") == 2
        host.set_guest_bots(False, 2)
        carla_seats = lambda: [s for s in host.state()["seats"] if s["owner_name"] == "Carla"]   # noqa: E731
        ben2.add_bot(key(reg, "MinRaiseBot"), "MinRaise Bot")
        assert wait(lambda: ben2.notice is not None) and carla_seats() == []
        seat = host.state()["seats"][0]["id"]
        host.set_plays(seat, False)
        assert wait(lambda: not next(s for s in ben.lobby["seats"] if s["id"] == seat)["plays"])
        ben.remove_bot(seat)
        assert wait(lambda: all(s["id"] != seat for s in host.state()["seats"]))
    finally:
        ben.close()
        ben2.close()
    assert wait(lambda: len(host.state()["members"]) == 1)
    assert host.state()["seats"] == []                     # a guest who leaves takes their bots along


def _play_people(host, clients, until, timeout=60):
    """Play every person's turns (host and guests) and the host's Next round."""
    end = time.time() + timeout
    rng = random.Random(1)
    looks = []
    while time.time() < end and not until():
        for session in [host.session] + [c.session for c in clients]:
            if session is None:
                continue
            snap = session.snapshot()
            if snap.pending is not None:
                state = snap.pending
                dudo = next((a for a in state.legal_actions if isinstance(a, Dudo)), None)
                if dudo is not None and (state.current_bid.quantity >= 4 or rng.random() < 0.2):
                    session.submit(dudo)
                else:
                    session.submit(fallback_action(state))
                looks.append(snap)
        if host.session is not None and host.session.snapshot().awaiting_next_round:
            host.session.next_round()
        time.sleep(0.005)
    return looks


def test_a_game_with_guests_and_their_bots(host):
    reg_ben, reg_carla = registry(), registry()
    ben = LobbyClient("Ben", reg_ben, host.code, local_ips=LOCAL)
    carla = LobbyClient("Carla", reg_carla, host.code, local_ips=LOCAL)
    try:
        ben.add_bot(key(reg_ben, "MinRaiseBot"), "MinRaise Bot")       # seats keep the order they came in
        assert wait(lambda: len(host.state()["seats"]) == 1)
        carla.add_bot(key(reg_carla, "ExampleBot"), "Example Bot")
        assert wait(lambda: len(host.state()["seats"]) == 2)
        host.add_bot(key(host.registry, "RandomBot"), "Random Bot")
        assert wait(lambda: len(host.state()["seats"]) == 3)
        session = host.start_game(delay=0.0)
        assert wait(lambda: ben.session is not None and carla.session is not None)
        names = [b.name for b in session.bots]
        assert names == ["Anna", "Ben", "Carla", "MinRaise Bot (Ben)", "Example Bot (Carla)", "Random Bot"]
        # a new player cannot walk into a game going on
        with pytest.raises(ConnectionError):
            LobbyClient("Dirk", registry(), host.code, local_ips=LOCAL)

        def guest_sees_only_own_dice():
            snap = ben.session.snapshot()
            if not snap.players or snap.reveal is not None:
                return False
            own = [p for p in snap.players if p.is_human]
            others = [p for p in snap.players if not p.is_human]
            return (len(own) == 1 and own[0].name == "Ben" and len(own[0].dice) == own[0].dice_count
                    and all(p.dice == () for p in others))

        assert wait(guest_sees_only_own_dice)
        _play_people(host, [ben, carla], until=lambda: session.snapshot().finished)
        final = session.snapshot()
        assert final.finished and final.error is None
        assert wait(lambda: ben.session.snapshot().finished and carla.session.snapshot().finished)
        assert ben.session.snapshot().standings == final.standings == carla.session.snapshot().standings
        assert ben.session.snapshot().places == final.places                # who went out where, on every phone
        moved = {f.player for f in final.feed if f.kind in ("bid", "dudo", "calza")}
        assert {"MinRaise Bot (Ben)", "Example Bot (Carla)"} <= moved    # the guests' bots really played
        assert type(ben._bots[next(iter(ben._bots))]).__name__ == "MinRaiseBot"   # ... on the guest's side
        # a guest scores the reveals it saw itself, the same as the host did
        host_hands = {(h.round, h.name): (h.score, h.lost) for h in final.hands}
        guest_hands = ben.session.snapshot().hands
        assert guest_hands and all(host_hands[(h.round, h.name)] == (h.score, h.lost) for h in guest_hands)
        guest_feed = ben.session.snapshot().feed
        assert "win" in [f.kind for f in guest_feed][-4:]      # a bot may still react after the win
        host.end_game()
        assert wait(lambda: ben.lobby.get("phase") == "lobby")
    finally:
        ben.close()
        carla.close()


def test_a_guest_who_leaves_mid_game_does_not_stall_it(host):
    reg = registry()
    ben = LobbyClient("Ben", reg, host.code, local_ips=LOCAL)
    ben.add_bot(key(reg, "RandomBot"), "Random Bot")
    host.add_bot(key(host.registry, "MinRaiseBot"), "MinRaise Bot")
    assert wait(lambda: len(host.state()["seats"]) == 2)
    session = host.start_game(delay=0.0)
    assert wait(lambda: ben.session is not None)
    time.sleep(0.3)
    ben.close()
    _play_people(host, [], until=lambda: session.snapshot().finished)
    assert session.snapshot().finished and session.snapshot().error is None
    # gone mid-game: Ben and his bot lost at once
    assert set(session.snapshot().standings[-2:]) == {"Ben", "Random Bot (Ben)"}


def _out(snap):
    return {f.player: f.text for f in snap.feed if f.kind == "out"}


def test_a_guest_who_walks_away_loses_with_their_bots_and_the_rest_play_on(host):
    reg_ben, reg_carla = registry(), registry()
    ben = LobbyClient("Ben", reg_ben, host.code, local_ips=LOCAL)
    carla = LobbyClient("Carla", reg_carla, host.code, local_ips=LOCAL)
    try:
        ben.add_bot(key(reg_ben, "MinRaiseBot"), "MinRaise Bot")
        host.add_bot(key(host.registry, "RandomBot"), "Random Bot")
        assert wait(lambda: len(host.state()["seats"]) == 2)
        session = host.start_game(delay=0.0)
        assert wait(lambda: ben.session is not None and carla.session is not None)
        assert ben.session.mine == ["Ben", "MinRaise Bot (Ben)"]
        assert host.seats_of(0) == ["Anna", "Random Bot"]
        # everyone else plays until it is Ben's turn (or a round has gone by)
        _play_people(host, [carla], until=lambda: ben.session.snapshot().pending is not None
                     or session.snapshot().round_number >= 2, timeout=20)
        ben.forfeit()                                   # possibly while it is Ben's turn
        assert ben.session.snapshot().pending is None
        assert wait(lambda: {"Ben", "MinRaise Bot (Ben)"} <= set(_out(session.snapshot())))
        assert _out(session.snapshot())["Ben"] == "left the game"
        _play_people(host, [carla], until=lambda: session.snapshot().finished)
        final = session.snapshot()
        assert final.finished and final.error is None
        assert set(final.standings[-2:]) == {"Ben", "MinRaise Bot (Ben)"}
        assert wait(lambda: ben.session.snapshot().finished)    # Ben can still watch it end
        assert not ben.closed
    finally:
        ben.close()
        carla.close()


def test_the_host_can_walk_away_and_the_guests_play_on(host):
    reg = registry()
    ben = LobbyClient("Ben", reg, host.code, local_ips=LOCAL)
    carla = LobbyClient("Carla", registry(), host.code, local_ips=LOCAL)
    try:
        host.add_bot(key(host.registry, "MinRaiseBot"), "MinRaise Bot")
        ben.add_bot(key(reg, "RandomBot"), "Random Bot")
        assert wait(lambda: len(host.state()["seats"]) == 2)
        session = host.start_game(delay=0.0)
        assert wait(lambda: ben.session is not None and carla.session is not None)
        session.hold_timeout = 0.05                     # nobody at the host's table presses Next round
        host.forfeit()
        assert wait(lambda: {"Anna", "MinRaise Bot"} <= set(_out(session.snapshot())))
        end = time.time() + 60
        while time.time() < end and not session.snapshot().finished:
            for c in (ben, carla):
                snap = c.session.snapshot()
                if snap.pending is not None:
                    c.session.submit(fallback_action(snap.pending))
            time.sleep(0.005)
        final = session.snapshot()
        assert final.finished and final.error is None
        assert set(final.standings[-2:]) == {"Anna", "MinRaise Bot"}
    finally:
        ben.close()
        carla.close()


def test_the_host_leaving_closes_the_lobby_for_guests():
    h = LobbyHost("Anna", registry(), ip="127.0.0.1")
    ben = LobbyClient("Ben", registry(), h.code, local_ips=LOCAL)
    h.close()
    assert wait(lambda: ben.closed and ben.error is not None)
    assert "host left" in ben.error


def test_the_bots_learn_on_both_sides_of_a_lobby(tmp_path):
    """The host's bots learn about the guests, a guest's bots (on the
    guest's phone) about the host and the others -- and keep it, by id."""
    from perudo_app.core import BotMemory
    host_memory, ben_memory = BotMemory(str(tmp_path / "anna")), BotMemory(str(tmp_path / "ben"))
    host = LobbyHost("Anna", registry(), ip="127.0.0.1", memory=host_memory, player_id="anna-id")
    reg_ben = registry()
    ben = LobbyClient("Ben", reg_ben, host.code, local_ips=LOCAL, player_id="ben-id", memory=ben_memory)
    try:
        assert wait(lambda: len(host.state()["members"]) == 2)
        host.add_bot(key(host.registry, "CarlosBot"), "Carlos")
        ben.add_bot(key(reg_ben, "PokerfaceBot"), "Pokerface")
        assert wait(lambda: len(host.state()["seats"]) == 2)
        session = host.start_game(delay=0.0)
        assert wait(lambda: ben.session is not None)
        _play_people(host, [ben], until=lambda: session.snapshot().finished)
        # the host's Carlos: about Ben, and about Ben's bot
        assert wait(lambda: host_memory.who_remembers("ben-id") == ["Carlos"])
        # Ben's Pokerface, on Ben's phone: about Anna (and kept under its own name)
        assert wait(lambda: ben_memory.who_remembers("anna-id") == ["Pokerface"])
        saved = list((tmp_path / "ben").glob("*.json"))
        assert len(saved) == 1 and saved[0].name.endswith("_Pokerface.json")   # the same memory as at home
        host.end_game()
    finally:
        ben.close()
        host.close()


def test_people_bring_who_they_are_to_the_table(tmp_path):
    from perudo_app.core import BotMemory
    memory = BotMemory(str(tmp_path / "host-memory"))
    host = LobbyHost("Anna", registry(), ip="127.0.0.1", memory=memory, player_id="anna-id")
    ben = LobbyClient("Ben", registry(), host.code, local_ips=LOCAL, player_id="ben-id",
                      memory=BotMemory(str(tmp_path / "ben-memory")))
    old = LobbyClient("Old", registry(), host.code, local_ips=LOCAL)           # an app without ids
    try:
        assert wait(lambda: len(host.state()["members"]) == 3)
        host.add_bot(key(host.registry, "CarlosBot"), "Carlos")
        assert wait(lambda: len(host.state()["seats"]) == 1)
        session = host.start_game(delay=0.0)
        assert session.people == {"anna-id": "Anna", "ben-id": "Ben"}
        assert wait(lambda: ben.session is not None and ben._people == session.people)
        _play_people(host, [ben, old], until=lambda: session.snapshot().finished)
        assert wait(lambda: memory.who_remembers("ben-id") == ["Carlos"])  # by who he is
        host.end_game()
    finally:
        ben.close()
        old.close()
        host.close()
