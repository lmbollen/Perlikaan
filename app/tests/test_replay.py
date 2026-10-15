"""Recording games and playing them again."""

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from perudo.models import Dudo                                     # noqa: E402
from perudo_app.core import BotRegistry, GameSession, HumanSeat, ReplaySession   # noqa: E402
from perudo_app.multiplayer import fallback_action                 # noqa: E402
from perudo_app.replay import ReplayStore                          # noqa: E402


def wait(cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def table(*names):
    reg = BotRegistry()
    reg.scan_bundled()
    picks = [e for e in reg.entries if e.class_name in names]
    for e in picks:
        e.copies = 1
    return reg.make(picks)


def play_as_human(session, until, leave_after=None, rng=None):
    """Play the person's turns (Dudo now and then); leave after that many turns."""
    rng = rng or random.Random(4)
    turns = 0
    end = time.time() + 60
    while time.time() < end and not until():
        snap = session.snapshot()
        if snap.awaiting_next_round:
            session.next_round()
        state = snap.pending
        if state is not None:
            if leave_after is not None and turns >= leave_after:
                session.forfeit([session.human.name])
                return
            dudo = next((a for a in state.legal_actions if isinstance(a, Dudo)), None)
            session.submit(dudo if dudo and rng.random() < 0.3 else fallback_action(state))
            turns += 1
        time.sleep(0.002)


def story(snap):
    """Everything that happened, as the feed tells it (no talk)."""
    return [(f.kind, f.player, f.text, repr(f.bid), f.loser) for f in snap.feed if f.kind != "talk"]


def replay(data):
    session = ReplaySession(json.loads(json.dumps(data)), delay=0.0)
    session.set_auto(True, 0.0)
    session.start()
    assert wait(lambda: session.snapshot().finished, 60)
    return session.snapshot()


def test_a_game_played_again_goes_exactly_as_it_went(tmp_path):
    store = ReplayStore(str(tmp_path))
    bots = table("CarlosBot", "SniperBot", "RandomBot") + [HumanSeat("Me")]
    session = GameSession(bots, step=False, delay=0.0, hold_reveals=True, replays=store)
    session.TALK_CHANCE = 1.0
    session.start()
    play_as_human(session, until=lambda: session.snapshot().finished)
    assert wait(lambda: session.snapshot().finished)
    original = session.snapshot()
    assert wait(lambda: store.games())
    meta = store.games()[0]
    assert meta["finished"] and meta["winner"] == original.standings[0]
    assert meta["place"] == original.standings.index("Me") + 1 and meta["players"] == 4

    again = replay(store.load(meta["id"]))
    assert again.error is None and again.note is None
    assert again.standings == original.standings
    assert story(again) == story(original)
    said = [(f.player, f.text) for f in original.feed if f.kind == "talk"]
    assert said and [(f.player, f.text) for f in again.feed if f.kind == "talk"] == said
    # the hands are scored again at every reveal, the same
    assert [(h.round, h.name, h.score, h.lost) for h in again.hands] == \
           [(h.round, h.name, h.score, h.lost) for h in original.hands]


def test_leaving_mid_game_is_played_again_too(tmp_path):
    store = ReplayStore(str(tmp_path))
    bots = table("PokerfaceBot", "MinRaiseBot", "ExampleBot") + [HumanSeat("Me")]
    session = GameSession(bots, step=False, delay=0.0, hold_reveals=True, replays=store)
    session.start()
    play_as_human(session, until=lambda: session.snapshot().finished, leave_after=3)
    # the bots play on to the end
    end = time.time() + 60
    while time.time() < end and not session.snapshot().finished:
        if session.snapshot().awaiting_next_round:
            session.next_round()
        time.sleep(0.002)
    original = session.snapshot()
    assert original.finished and original.standings[-1] == "Me"
    assert wait(lambda: store.games())
    again = replay(store.load(store.games()[0]["id"]))
    assert again.standings == original.standings
    assert [s for s in story(again) if s[0] != "round"] == [s for s in story(original) if s[0] != "round"]


def test_a_recording_that_stops_early_ends_quietly(tmp_path):
    store = ReplayStore(str(tmp_path))
    bots = table("SniperBot", "RandomBot") + [HumanSeat("Me")]
    session = GameSession(bots, step=False, delay=0.0, hold_reveals=True, replays=store)
    session.start()
    turns = {"n": 0}

    def three_turns():
        return session.snapshot().round_number >= 3

    play_as_human(session, until=three_turns)
    session.stop()                                  # back to the menu mid-game
    assert wait(lambda: not session.running)
    assert wait(lambda: store.games())
    meta = store.games()[0]
    assert not meta["finished"] and meta["winner"] is None
    again = replay(store.load(meta["id"]))
    assert again.error is None and again.note == "End of the recording" and not again.standings


def test_only_the_newest_recordings_are_kept(tmp_path):
    store = ReplayStore(str(tmp_path), keep=2)
    for _ in range(3):
        bots = table("SniperBot", "RandomBot") + [HumanSeat("Me")]
        session = GameSession(bots, step=False, delay=0.0, hold_reveals=True, replays=store)
        session.start()
        play_as_human(session, until=lambda: session.snapshot().finished)
        assert wait(lambda: not session.running)
    assert len(store.games()) == 2
