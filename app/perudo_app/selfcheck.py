"""
The app checks itself: started with PERUDO_SELFCHECK set, it plays through
its own screens, checks their layout, takes screenshots, writes what it
found and quits (with exit status 1 when something failed).

    PERUDO_SELFCHECK=all python main.py
    PERUDO_SELFCHECK=home,table PERUDO_SELFCHECK_OUT=/tmp/check python main.py

It runs the same on the desktop (tools/layout_check.py runs it at the
screen sizes of several phones), and in the iPhone simulator
(ios/simulator_check.sh), where it also tries the phone's voices and
playing over the internet.

A layout check looks at every widget on the screen (and in an open
sheet): it must lie inside the safe area (clear of the notch and the home
bar, or the status and navigation bars), and a button's or a label's text
must fit in it. What scrolls is checked as the scrolling view, not its
contents.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, List, Optional

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.layout import Layout
from kivy.uix.modalview import ModalView
from kivy.uix.screenmanager import Screen
from kivy.uix.scrollview import ScrollView

from perudo.models import Dudo

SCENARIOS = ["home", "settings", "players", "table", "profile", "watch", "together", "speech", "online"]


class Until:
    """Yielded by a scenario: wait until `cond()` is true (or fail after `timeout` s)."""

    def __init__(self, cond: Callable[[], bool], timeout: float, what: str):
        self.cond, self.timeout, self.what = cond, timeout, what


class CheckFailed(AssertionError):
    pass


def check(ok, message: str) -> None:
    if not ok:
        raise CheckFailed(message)


# ---------------------------------------------------------------------------
# the layout check
# ---------------------------------------------------------------------------

def _describe(w) -> str:
    text = getattr(w, "text", "") or ""
    text = " ".join(str(text).split())
    if len(text) > 30:
        text = text[:29] + "…"
    return f"{type(w).__name__}" + (f" '{text}'" if text else "")


def layout_problems(app) -> List[str]:
    """What sticks out of the safe area, or whose text does not fit."""
    width, height = Window.size
    left, top, right, bottom = (float(v) for v in app.frame.padding)
    safe = (left, bottom, width - right, height - top)
    problems: List[str] = []
    roots = [app.sm.current_screen] + [w for w in Window.children if isinstance(w, ModalView)]
    for root in roots:
        _walk(root, safe, problems)
    seen, unique = set(), []
    for p in problems:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _rect(w):
    x, y = w.to_window(*w.pos)
    return x, y, x + w.width, y + w.height


def _over(inner, outer) -> float:
    """How far a rectangle sticks out of another (0 or less: it is inside)."""
    return max(outer[0] - inner[0], outer[1] - inner[1], inner[2] - outer[2], inner[3] - outer[3])


def _walk(w, safe, out: List[str]) -> None:
    if w.opacity <= 0.01:
        return
    if not isinstance(w, (Screen, ModalView)) and w.width > 1 and w.height > 1:
        rect = _rect(w)
        slack = 1.5
        over = _over(rect, safe)
        if over > slack:
            out.append(f"{_describe(w)} sticks out of the safe area by {over:.0f}px")
        parent = w.parent
        if isinstance(parent, Layout) and not isinstance(parent, (Screen, ModalView)) \
                and parent.width > 1 and parent.height > 1:
            over = _over(rect, _rect(parent))       # a full layout pushes its children over others
            if over > slack:
                where = lambda r: "(" + ",".join(f"{v:.0f}" for v in r) + ")"   # noqa: E731
                out.append(f"{_describe(w)} {where(rect)} sticks out of its {type(parent).__name__} "
                           f"{where(_rect(parent))} by {over:.0f}px")
        if isinstance(w, Label) and w.text.strip():
            w.texture_update()
            tw, th = w.texture_size
            if tw > w.width + 2:
                out.append(f"{_describe(w)}: text {tw:.0f}px wide in {w.width:.0f}px")
            if th > w.height + 2:
                out.append(f"{_describe(w)}: text {th:.0f}px high in {w.height:.0f}px")
    if isinstance(w, ScrollView):
        return                                # what scrolls may be out of view
    for child in w.children:
        _walk(child, safe, out)


# ---------------------------------------------------------------------------
# running the scenarios
# ---------------------------------------------------------------------------

class SelfCheck:
    def __init__(self, app, names: List[str], out_dir: Path):
        self.app = app
        self.names = names
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.results: List[tuple] = []          # (scenario, "ok" | "skipped" | "FAIL", detail)
        self.queue = [(n, getattr(self, f"scenario_{n}")) for n in names]
        self.current = None
        self.gen = None
        self._settings = dict(app.settings.values)
        self._ticks: List = []
        self._closers: List[tuple] = []         # (what, close) for what a scenario opened
        self._log_file = open(self.out / "selfcheck.log", "w", encoding="utf-8")

    # -- reporting ---------------------------------------------------------

    def log(self, line: str) -> None:
        line = f"SELFCHECK {line}"
        for stream in (sys.__stdout__, self._log_file):
            try:
                stream.write(line + "\n")
                stream.flush()
            except Exception:
                pass

    def shot(self, name: str) -> None:
        """A screenshot of the window. Not Window.screenshot: it reads RGB
        rows, which come out skewed when a row is not a multiple of four
        bytes (a 393-point phone at 2x is 786 pixels wide)."""
        try:
            from kivy.core.image import Image as CoreImage
            from kivy.graphics.opengl import GL_RGBA, GL_UNSIGNED_BYTE, glReadPixels
            from kivy.graphics.texture import Texture
            width, height = (int(v) for v in Window.size)
            data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
            texture = Texture.create(size=(width, height), colorfmt="rgba")
            texture.blit_buffer(data, colorfmt="rgba", bufferfmt="ubyte")
            CoreImage(texture).save(str(self.out / f"{self.current}-{name}.png"), flipped=True)
        except Exception as exc:
            self.log(f"  (no screenshot: {type(exc).__name__}: {exc})")

    def look(self, name: str):
        """Wait for the screen to settle, take a screenshot and check its layout."""
        yield 0.7
        self.shot(name)
        self.log(f"  {name}: safe area padding {[round(float(v)) for v in self.app.frame.padding]}")
        problems = layout_problems(self.app)
        for p in problems:
            self.log(f"  layout {name}: {p}")
        check(not problems, f"{len(problems)} layout problem(s) on {name}")

    # -- the runner --------------------------------------------------------

    def start(self) -> None:
        self.log(f"start: {', '.join(self.names)} on a {Window.width}x{Window.height} window, "
                 f"safe area padding {list(self.app.frame.padding)}")
        Clock.schedule_once(self._next, 2.0)

    def _next(self, *_):
        self._cleanup()
        if not self.queue:
            Clock.schedule_once(self._finish, 0.5)
            return
        self.current, fn = self.queue.pop(0)
        self.log(f"run {self.current}")
        self._started = time.monotonic()
        self.gen = fn()
        Clock.schedule_once(lambda *_: self._advance(), 0.3)

    def _advance(self, error: Optional[BaseException] = None) -> None:
        try:
            item = self.gen.throw(error) if error is not None else next(self.gen)
        except StopIteration:
            self._done("ok", f"{time.monotonic() - self._started:.1f}s")
            return
        except Skip as skip:
            self._done("skipped", str(skip))
            return
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, (CheckFailed, TimeoutError)):
                for line in traceback.format_exc().rstrip().splitlines():
                    self.log(f"  {line}")
            self.shot("failure")
            self._done("FAIL", detail)
            return
        if isinstance(item, Until):
            self._poll(item)
        else:
            Clock.schedule_once(lambda *_: self._advance(), float(item or 0))

    def _poll(self, until: Until) -> None:
        start = time.monotonic()

        def tick(_dt):
            try:
                ok = until.cond()
            except Exception:
                ok = False
            if ok:
                event.cancel()
                Clock.schedule_once(lambda *_: self._advance(), 0)
            elif time.monotonic() - start > until.timeout:
                event.cancel()
                self._advance(TimeoutError(f"waited {until.timeout:.0f}s for {until.what}"))

        event = Clock.schedule_interval(tick, 0.2)

    def _done(self, status: str, detail: str) -> None:
        self.results.append((self.current, status, detail))
        self.log(f"{status} {self.current}: {detail}")
        Clock.schedule_once(self._next, 0.5)

    def _cleanup(self) -> None:
        """Back to the home screen, whatever a scenario left behind."""
        app = self.app
        for tick in self._ticks:
            tick.cancel()
        self._ticks = []
        for w in list(Window.children):
            if isinstance(w, ModalView):
                w.dismiss(animation=False)
        for name, closer in getattr(self, "_closers", []):
            try:
                closer()
            except Exception:
                pass
        self._closers = []
        if app.lobby is not None:
            app.leave_lobby(go_back=False)
        game = app.sm.get_screen("game")
        if game.session is not None and not game.multiplayer:
            game.session.stop()
            game.session = None
            if game._poll is not None:
                game._poll.cancel()
                game._poll = None
        app.speaker.clear()
        app.sm.transition.duration = 0
        app.sm.current = "home"
        app.sm.transition.duration = 0.22

    def _finish(self, *_):
        app = self.app
        app.settings.values.clear()
        app.settings.values.update(self._settings)
        app.settings.save()
        app.speaker.enabled = bool(app.settings["tts"])
        failed = [r for r in self.results if r[1] == "FAIL"]
        with open(self.out / "results.txt", "w", encoding="utf-8") as f:
            for name, status, detail in self.results:
                f.write(f"{status:8} {name}: {detail}\n")
            f.write("FAILED\n" if failed else "PASSED\n")
        self.log(f"{'FAILED' if failed else 'PASSED'} "
                 f"({sum(1 for r in self.results if r[1] == 'ok')} ok, "
                 f"{sum(1 for r in self.results if r[1] == 'skipped')} skipped, {len(failed)} failed)")
        self._log_file.close()
        app.selfcheck_exit = 1 if failed else 0
        app.stop()
        # On a phone the app may not end by itself when stopped.
        threading.Timer(3.0, lambda: os._exit(app.selfcheck_exit)).start()

    # -- helpers for the scenarios -------------------------------------------

    def go(self, screen: str) -> None:
        self.app.sm.transition.duration = 0
        self.app.go(screen)
        self.app.sm.transition.duration = 0.22

    def seat(self, *class_names: str) -> None:
        for e in self.app.registry.entries:
            e.enabled = e.class_name in class_names
            e.copies = 1

    def every(self, seconds: float, fn: Callable[[], None]) -> None:
        self._ticks.append(Clock.schedule_interval(lambda _dt: fn(), seconds))

    def modals(self) -> List[ModalView]:
        return [w for w in Window.children if isinstance(w, ModalView)]

    def press(self, text: str) -> None:
        from perudo_app.widgets import Btn
        for m in self.modals():
            for w in m.walk():
                if isinstance(w, Btn) and w.text == text:
                    w.dispatch("on_release")
                    return
        raise CheckFailed(f"no button '{text}' in the open sheet")

    def key(self, registry, class_name: str) -> str:
        return next(registry.key(e) for e in registry.entries if e.class_name == class_name)

    @staticmethod
    def play_turn(session, dudo: bool = True) -> None:
        """Play the human seat's turn, if it is up: Dudo when it can (loses dice fast)."""
        from perudo_app.multiplayer import fallback_action
        if session is None:
            return
        snap = session.snapshot()
        if snap.awaiting_next_round and getattr(session, "can_continue", False):
            session.next_round()
        state = snap.pending
        if state is not None:
            action = next((a for a in state.legal_actions if isinstance(a, Dudo)), None) if dudo else None
            session.submit(action or fallback_action(state))

    # -- the scenarios ---------------------------------------------------------

    def scenario_home(self):
        self.seat("RandomBot", "MinRaiseBot", "ExampleBot")
        self.go("home")
        self.app.sm.get_screen("home").on_pre_enter()
        yield from self.look("home")
        from kivy.utils import platform
        if platform == "ios":
            from perudo_app import ui
            self.log(f"  the safe area comes from: {ui.SAFE_AREA_SOURCE}")
        if platform == "ios" and Window.height > 2 * Window.width:
            # A phone with a notch or Dynamic Island: without the safe area
            # the layout check would measure against the whole screen.
            left, top, right, bottom = (float(v) for v in self.app.frame.padding)
            check(top > 0 and bottom > 0, f"no safe area on a phone with a notch: padding {self.app.frame.padding}")

    def scenario_settings(self):
        self.go("settings")
        yield from self.look("settings")

    def scenario_players(self):
        self.go("bots")
        yield from self.look("players")

    def scenario_table(self):
        app = self.app
        self.seat("RandomBot", "MinRaiseBot", "ExampleBot")
        app.set_setting("step_delay", 0.05)
        app.start_game(human=True)
        game = app.sm.get_screen("game")

        hold = {"reveal": False}

        def next_round():                        # the bots may decide a round before your first turn
            if (game.session is not None and game.session.snapshot().awaiting_next_round
                    and not hold["reveal"]):
                game.session.next_round()

        def scored_rows():
            return [row.name for row in game._rows.values() if "hand" in row.sub]

        self.every(0.2, next_round)
        yield Until(lambda: game.session.snapshot().pending is not None, 60, "the first turn")
        yield from self.look("table")
        # During a round nobody's hand is scored: that would tell more than the dice do.
        check(not game.hand_caption and not scored_rows(), "a hand score is shown during the round")
        game.open_settings()
        yield from self.look("settings-window")
        app.set_setting("chat", False)
        yield 0.3
        check(not any(line.bubble for line in game.ids.feed.children), "talk is shown with the chat off")
        app.set_setting("chat", True)
        self.press("Done")
        hold["reveal"] = True                    # stop at the next reveal: the hands are scored there

        def play():
            if not (hold["reveal"] and game.session.snapshot().awaiting_next_round):
                self.play_turn(game.session)

        self.every(0.1, play)
        yield Until(lambda: game.session.snapshot().awaiting_next_round, 120, "a reveal")
        yield 0.5
        yield from self.look("reveal")
        revealed = [n for n, _ in game.session.snapshot().reveal["dice"] if n != game._human]   # ours: the panel
        check(sorted(scored_rows()) == sorted(revealed), f"scored at the reveal: {scored_rows()}, revealed {revealed}")
        check("HAND" in game.hand_caption, f"our hand's score at the reveal: {game.hand_caption!r}")
        hold["reveal"] = False
        yield Until(lambda: game.out or game.session.snapshot().finished, 240, "being out")
        if game.out:
            yield from self.look("out")
            check(game.out_button == "Back to the menu", f"the out panel offers '{game.out_button}'")
            game.leave()
            yield 0.5
            check(App.get_running_app().sm.current == "home", "leaving did not go back to the menu")

    def scenario_profile(self):
        """The profile, and a recorded game watched again (the table
        scenario, before it, recorded one; on its own one is played)."""
        app = self.app
        if not app.replays.games():
            self.seat("RandomBot", "MinRaiseBot")
            app.set_setting("step_delay", 0.05)
            app.start_game(human=True)
            game = app.sm.get_screen("game")
            self.every(0.1, lambda: self.play_turn(game.session))
            yield Until(lambda: game.session.snapshot().finished or game.out, 240, "a game")
            game.leave()
            yield 0.5
        self.go("profile")
        yield 0.3
        screen = app.sm.get_screen("profile")
        rows = list(screen.ids.matches.children)
        check(rows, "no recorded game on the profile")
        yield from self.look("profile")
        rows[-1].dispatch("on_release")                 # the newest: listed first
        game = app.sm.get_screen("game")
        yield Until(lambda: app.sm.current == "game" and game.replay is not None, 10, "the replay")
        game.set_auto(True)
        yield Until(lambda: len(game.session.snapshot().feed) >= 6, 60, "the replay playing")
        check(game.show_dice and all(len(p.dice) == p.dice_count for p in game.session.snapshot().players
                                     if p.alive), "the replay hides dice")
        yield from self.look("replay")
        game.leave()
        yield 0.5
        check(app.sm.current == "profile", f"leaving a replay went to {app.sm.current}")

    def scenario_watch(self):
        app = self.app
        self.seat("RandomBot", "MinRaiseBot", "ExampleBot")
        app.set_setting("step_delay", 0.05)
        app.start_game(human=False)
        game = app.sm.get_screen("game")
        game.set_auto(True)
        yield Until(lambda: len(game.session.snapshot().feed) >= 8, 60, "a few moves")
        yield from self.look("watch")
        app.set_setting("step_delay", 0.0)
        game.set_auto(True)                     # to the end: the standings sheet
        yield Until(lambda: game._sheet is not None, 180, "the end of the game")
        yield from self.look("game-over")
        game.leave()

    def scenario_together(self):
        """Host a lobby on this phone (on 127.0.0.1) with a guest in the same
        app, play, and leave mid-game: the warning, and losing at once."""
        from perudo_app.core import BotRegistry
        from perudo_app.multiplayer import LobbyClient, LobbyHost
        app = self.app
        app.set_setting("step_delay", 0.05)
        host = LobbyHost(app.settings["human_name"] or "You", app.registry, ip="127.0.0.1")
        app.enter_lobby(host, host=True)
        registry = BotRegistry()
        registry.scan_bundled()
        guest = LobbyClient("Guest", registry, host.code, local_ips=["127.0.0.1"])
        self._closers.append(("guest", guest.close))
        guest.add_bot(self.key(registry, "RandomBot"), "Random Bot")
        host.add_bot(self.key(app.registry, "MinRaiseBot"), "MinRaise Bot")
        yield Until(lambda: len(host.state()["seats"]) == 2 and len(host.state()["members"]) == 2, 15,
                    "the guest and the bots in the lobby")
        yield from self.look("lobby")
        app.start_lobby_game()
        self.every(0.1, lambda: self.play_turn(guest.session, dudo=False))
        game = app.sm.get_screen("game")
        yield Until(lambda: game.session is not None and game.session.snapshot().players, 15, "the game")
        yield from self.look("game")
        mine = list(host.seats_of(0))
        app.back()
        yield 0.5
        check(self.modals(), "no warning before leaving a game you are still in")
        yield from self.look("leave-warning")
        self.press("Leave and lose")
        yield Until(lambda: app.sm.current == "lobby", 5, "the lobby")
        session = host.session
        yield Until(lambda: set(mine) <= {f.player for f in session.snapshot().feed if f.kind == "out"},
                    15, "you and your bot to be out")

    def scenario_speech(self):
        speaker = self.app.speaker
        if speaker.system is None:
            raise Skip("no phone voices here")
        enabled = speaker.enabled
        speaker.enabled = True
        try:
            for lang, line in (("en", "Dudo! I don't believe a word of it."), ("nl", "Dudo! Ik geloof er niets van.")):
                speaker.error = None
                speaker.say(line, lang, "Sniper")
                yield Until(lambda: speaker.idle, 30, f"the {lang.upper()} line to be spoken")
                if speaker.error and lang != "en" and "voice" in speaker.error:
                    self.log(f"  note: {speaker.error}")       # a voice that is not installed here
                    continue
                check(speaker.error is None, f"speaking {lang.upper()}: {speaker.error}")
        finally:
            speaker.enabled = enabled

    def scenario_online(self):
        """Host a lobby over the internet (Ably) and join it."""
        from perudo_app import relay
        from perudo_app.core import BotRegistry
        from perudo_app.multiplayer import LobbyClient
        app = self.app
        if not relay.ably_key():
            raise Skip("this build has no online key")
        result: dict = {}

        def in_thread(name, fn):
            def run():
                try:
                    result[name] = fn()
                except Exception as exc:
                    result[name] = exc
            threading.Thread(target=run, daemon=True).start()

        in_thread("host", lambda: app.make_host(online=True))
        yield Until(lambda: "host" in result, 30, "the lobby to open online")
        host = result["host"]
        check(not isinstance(host, Exception), f"hosting online: {host!r}")
        self._closers.append(("host", host.close))
        registry = BotRegistry()
        in_thread("guest", lambda: LobbyClient("Guest", registry, host.code, relay=app.relay(), timeout=20))
        yield Until(lambda: "guest" in result, 40, "joining over the internet")
        guest = result["guest"]
        check(not isinstance(guest, Exception), f"joining online: {guest!r}")
        self._closers.append(("guest", guest.close))
        yield Until(lambda: len(host.state()["members"]) == 2, 20, "the guest in the host's lobby")


class Skip(Exception):
    pass


def start(app) -> Optional[SelfCheck]:
    """Called by the app at start: runs the self-check when PERUDO_SELFCHECK is set."""
    wanted = os.environ.get("PERUDO_SELFCHECK", "").strip()
    if not wanted:
        return None
    names = SCENARIOS if wanted in ("1", "all") else [n.strip() for n in wanted.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown self-check scenarios: {unknown} (known: {SCENARIOS})")
    out = Path(os.environ.get("PERUDO_SELFCHECK_OUT") or Path(app.user_data_dir) / "selfcheck")
    checker = SelfCheck(app, names, out)
    app._selfcheck = checker             # Clock keeps only weak references
    checker.start()
    return checker
