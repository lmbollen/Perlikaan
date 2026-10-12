"""The app's model, run on the desktop without Kivy."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from perudo.models import Bid, Calza, Dudo                     # noqa: E402
from perudo_app.core import (                                   # noqa: E402
    BotRegistry, GameSession, HumanSeat, Settings, Simulation, can, lowest_legal,
)


def wait(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_registry_scans_bundled_bots_and_loads_a_file(tmp_path):
    reg = BotRegistry()
    reg.scan_bundled()
    names = {e.class_name for e in reg.entries}
    assert {"CarlosDosBot", "Trihard", "PokerfaceBot"} <= names
    assert all(e.source == "bundled" for e in reg.entries)
    bot_file = tmp_path / "mybot.py"
    bot_file.write_text(
        "from perudo.models import Bid, Dudo\n"
        "class MyBot:\n"
        "    name = 'My Bot'\n"
        "    def play(self, state):\n"
        "        bids = [a for a in state.legal_actions if isinstance(a, Bid)]\n"
        "        return bids[0] if bids else Dudo()\n"
    )
    added = reg.add_file(str(bot_file))
    assert [e.display_name for e in added] == ["My Bot"]
    assert added[0].source == str(bot_file)
    bots = reg.make(added)
    assert bots[0].name == "My Bot"


def test_registry_reports_a_broken_file(tmp_path):
    reg = BotRegistry()
    bad = tmp_path / "bad.py"
    bad.write_text("this is not python\n")
    assert reg.add_file(str(bad)) == []
    assert reg.errors and "bad.py" in reg.errors[0]


def test_copies_are_numbered():
    reg = BotRegistry()
    reg.scan_bundled()
    entry = next(e for e in reg.entries if e.class_name == "RandomBot")
    entry.copies = 3
    bots = reg.make([entry])
    assert [b.name for b in bots] == [bots[0].name, f"{bots[0].name} 2", f"{bots[0].name} 3"]


def test_session_steps_through_a_game_and_shows_all_dice():
    reg = BotRegistry()
    reg.scan_bundled()
    picks = [e for e in reg.entries if e.class_name in ("RandomBot", "MinRaiseBot", "ExampleBot")]
    for e in picks:
        e.copies = 1
    session = GameSession(reg.make(picks), seed=7, step=True)
    session.start()
    assert wait(lambda: session.snapshot().to_act is not None)
    snap = session.snapshot()
    assert snap.round_number == 1 and len(snap.players) == 3
    assert all(len(p.dice) == 5 for p in snap.players)
    totals = snap.totals()
    assert sum(snap.naturals().values()) == 15 and totals[2] >= snap.naturals()[2]
    turns = 0
    while not session.snapshot().finished and turns < 5000:
        session.next()
        turns += 1
        time.sleep(0.002)
    assert wait(lambda: session.snapshot().finished)
    final = session.snapshot()
    assert final.error is None and len(final.standings) == 3
    assert any("wins" in line for line in final.log)


def test_session_auto_mode_finishes_on_its_own():
    reg = BotRegistry()
    reg.scan_bundled()
    picks = [e for e in reg.entries if e.class_name in ("RandomBot", "MinRaiseBot")]
    session = GameSession(reg.make(picks), seed=3, step=False, delay=0.0)
    session.start()
    assert wait(lambda: session.snapshot().finished, timeout=30)
    assert session.result is not None


def test_human_seat_waits_for_submit_and_hides_nothing_from_the_model():
    reg = BotRegistry()
    reg.scan_bundled()
    picks = [e for e in reg.entries if e.class_name in ("RandomBot",)]
    bots = reg.make(picks) + [HumanSeat("Me")]
    session = GameSession(bots, seed=11, step=False, delay=0.0)
    session.start()
    assert wait(lambda: session.snapshot().pending is not None or session.snapshot().finished, timeout=30)
    snap = session.snapshot()
    assert snap.pending is not None, "the human got no turn"
    state = snap.pending
    assert state.players[state.my_id].name == "Me"
    # answer legally until the game ends
    for _ in range(500):
        snap = session.snapshot()
        if snap.finished:
            break
        if snap.pending is None:
            time.sleep(0.005)
            continue
        state = snap.pending
        bid = lowest_legal(state, 2) or lowest_legal(state, state.current_bid.face if state.current_bid else 3)
        if state.current_bid is not None and state.current_bid.quantity >= 6 and can(state, Dudo):
            session.submit(Dudo())
        elif bid is not None:
            session.submit(bid)
        else:
            session.submit(Dudo() if can(state, Dudo) else Calza())
        time.sleep(0.005)
    assert wait(lambda: session.snapshot().finished, timeout=30)
    assert session.snapshot().error is None


def test_session_stop_ends_the_thread():
    reg = BotRegistry()
    reg.scan_bundled()
    picks = [e for e in reg.entries if e.class_name in ("RandomBot", "MinRaiseBot")]
    session = GameSession(reg.make(picks), seed=5, step=True)
    session.start()
    assert wait(lambda: session.snapshot().to_act is not None)
    session.stop()
    assert wait(lambda: not session.running)
    assert session.snapshot().finished


def test_simulation_updates_standings_live_and_can_stop():
    reg = BotRegistry()
    reg.scan_bundled()
    picks = [e for e in reg.entries if e.class_name in ("RandomBot", "MinRaiseBot", "ExampleBot")]
    sim = Simulation(reg.make(picks), games=400, seed=1)
    sim.start()
    assert wait(lambda: sim.played >= 5, timeout=30)
    rows = sim.standings()
    assert sum(r.wins for r in rows) == rows[0].games
    assert abs(sum(r.win_pct for r in rows) - 100) < 1e-6
    sim.stop()
    assert wait(lambda: not sim.running, timeout=30)
    assert sim.error is None and 5 <= sim.played <= 400


def test_settings_persist(tmp_path):
    s = Settings(str(tmp_path / "settings.json"))
    assert s["tts"] is False
    s["tts"] = True
    s["sim_games"] = 50
    again = Settings(str(tmp_path / "settings.json"))
    assert again["tts"] is True and again["sim_games"] == 50
