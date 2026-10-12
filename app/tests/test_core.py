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


def test_feed_tells_the_story_and_a_reveal_stays_until_the_next_bid():
    reg = BotRegistry()
    reg.scan_bundled()
    picks = [e for e in reg.entries if e.class_name in ("RandomBot", "MinRaiseBot", "ExampleBot")]
    for e in picks:
        e.copies = 1
    session = GameSession(reg.make(picks), seed=11, step=True)
    session.start()
    assert wait(lambda: session.snapshot().to_act is not None)
    first = session.snapshot().feed
    assert first[0].kind == "round" and first[0].text == "Round 1"
    # step until a round has been decided and the next round waits for its opener
    held = None
    for _ in range(400):
        session.next()
        time.sleep(0.005)
        snap = session.snapshot()
        if snap.reveal is not None and snap.round_number == 2:
            held = snap
            break
    assert held is not None, "the reveal of round 1 is gone before round 2's first bid"
    assert held.current_bid is None and held.to_act is not None
    assert held.reveal["loser"] in {p.name for p in held.players} or held.reveal["good"]
    kinds = [f.kind for f in held.feed]
    assert kinds.index("bid") < kinds.index("result") and kinds.count("round") == 2
    call = next(f for f in held.feed if f.kind in ("dudo", "calza"))
    assert call.bid == held.reveal["bid"] and call.other == held.reveal["bidder"]
    # the opener of round 2 bids: the reveal is done, and the bidder is known
    for _ in range(50):
        session.next()
        time.sleep(0.005)
        snap = session.snapshot()
        if snap.current_bid is not None:
            break
    assert snap.reveal is None and snap.bidder is not None
    session.stop()


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


BOT_SOURCE = (
    "from perudo.models import Bid, Dudo\n"
    "class {cls}:\n"
    "    name = '{name}'\n"
    "    def play(self, state):\n"
    "        bids = [a for a in state.legal_actions if isinstance(a, Bid)]\n"
    "        return bids[0] if bids else Dudo()\n"
)


def test_repo_bots_are_seated_by_default_per_their_count():
    reg = BotRegistry()
    reg.scan_bundled()
    seated = {e.class_name for e in reg.entries if e.enabled}
    assert {"CarlosDosBot", "Trihard", "PokerfaceBot"} <= seated
    assert not {"RandomBot", "MinRaiseBot"} & seated          # count = 0 in the repo


def test_imported_bots_are_copied_and_come_back_after_a_restart(tmp_path):
    downloads, store = tmp_path / "downloads", tmp_path / "store"
    downloads.mkdir()
    (downloads / "mine.py").write_text(BOT_SOURCE.format(cls="MineBot", name="Mine"))
    reg = BotRegistry(str(store))
    reg.scan_bundled()
    reg.scan_user()
    added = reg.import_file(str(downloads / "mine.py"))
    assert [e.display_name for e in added] == ["Mine"] and added[0].enabled
    assert (store / "mine.py").exists() and added[0].source == str(store / "mine.py")
    (downloads / "mine.py").unlink()                           # the original may go away
    added[0].copies = 3
    saved = reg.selection()

    again = BotRegistry(str(store))                            # the next start of the app
    again.scan_bundled()
    again.scan_user()
    again.apply_selection(saved)
    mine = next(e for e in again.entries if e.class_name == "MineBot")
    assert mine.enabled and mine.copies == 3
    assert [b.name for b in again.make([mine])] == ["Mine", "Mine 2", "Mine 3"]


def test_reimport_replaces_the_file_and_keeps_the_seats(tmp_path):
    store = tmp_path / "store"
    src = tmp_path / "mine.py"
    src.write_text(BOT_SOURCE.format(cls="MineBot", name="Mine v1"))
    reg = BotRegistry(str(store))
    first = reg.import_file(str(src))[0]
    first.copies = 2
    src.write_text(BOT_SOURCE.format(cls="MineBot", name="Mine v2"))
    second = reg.import_file(str(src))
    mine = [e for e in reg.entries if e.class_name == "MineBot"]
    assert len(mine) == 1 and mine[0] is second[0]
    assert mine[0].display_name == "Mine v2" and mine[0].copies == 2


def test_a_broken_import_is_not_kept_and_remove_deletes_the_file(tmp_path):
    store = tmp_path / "store"
    bad = tmp_path / "bad.py"
    bad.write_text("this is not python(\n")
    reg = BotRegistry(str(store))
    assert reg.import_file(str(bad)) == []
    assert not (store / "bad.py").exists() and reg.errors
    good = tmp_path / "good.py"
    good.write_text(BOT_SOURCE.format(cls="GoodBot", name="Good"))
    entry = reg.import_file(str(good))[0]
    reg.remove(entry)
    assert not (store / "good.py").exists()
    assert all(e.class_name != "GoodBot" for e in reg.entries)


def test_files_dropped_into_the_folder_are_picked_up_and_deleted_ones_dropped(tmp_path):
    store = tmp_path / "store"
    reg = BotRegistry(str(store))
    reg.scan_user()
    (store / "dropped.py").write_text(BOT_SOURCE.format(cls="DroppedBot", name="Dropped"))
    reg.scan_user()
    assert any(e.class_name == "DroppedBot" for e in reg.entries)
    (store / "dropped.py").unlink()
    reg.scan_user()
    assert all(e.class_name != "DroppedBot" for e in reg.entries)


def test_the_table_selection_is_saved_in_settings(tmp_path):
    reg = BotRegistry()
    reg.scan_bundled()
    for e in reg.entries:
        e.enabled = e.class_name == "RandomBot"
    settings = Settings(str(tmp_path / "settings.json"))
    settings["table"] = reg.selection()
    fresh = BotRegistry()
    fresh.scan_bundled()
    fresh.apply_selection(Settings(str(tmp_path / "settings.json"))["table"])
    assert [e.class_name for e in fresh.selected()] == ["RandomBot"]


def test_bundled_bots_are_found_when_only_compiled_files_ship(tmp_path):
    """python-for-android puts only *.pyc files in the APK (compileall -b
    -OO, then the sources are left out); the registry must still find the
    repository's bots there."""
    import compileall
    import os
    import shutil
    import subprocess

    shutil.copytree(ROOT / "perudo", tmp_path / "perudo", ignore=shutil.ignore_patterns("__pycache__"))
    assert compileall.compile_dir(str(tmp_path / "perudo"), legacy=True, optimize=2, quiet=1)
    for source in (tmp_path / "perudo").rglob("*.py"):
        source.unlink()
    assert not list((tmp_path / "perudo" / "bots").glob("*.py"))
    script = (
        "from perudo_app.core import BotRegistry\n"
        "reg = BotRegistry(); reg.scan_bundled()\n"
        "print(sorted(e.class_name for e in reg.entries)); print(reg.errors)\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(ROOT / "app")]))
    out = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    names, errors = out.stdout.strip().splitlines()
    assert "CarlosDosBot" in names and "Trihard" in names and "PokerfaceBot" in names
    assert errors == "[]"


def test_bar_padding_pads_only_what_the_bars_cover():
    from perudo_app.core import bar_padding
    # Android 15: the app fills the screen, under a 120 px status bar and a 60 px gesture bar
    assert bar_padding(2400, 2400, 0, 120, 0, 60) == [0, 120, 0, 60]
    # older Android: the app is laid out between the bars already
    assert bar_padding(2220, 2400, 0, 120, 0, 60) == [0, 0, 0, 0]
    # drawn under the status bar only
    assert bar_padding(2340, 2400, 0, 120, 0, 60) == [0, 120, 0, 0]
