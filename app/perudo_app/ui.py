"""
The Kivy user interface: a bot list, one game to watch or to join, a
simulation with a live leaderboard, and settings. All game logic lives in
perudo_app.core; this module only draws snapshots and sends decisions.
"""

from __future__ import annotations

import os
from pathlib import Path

from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.widget import Widget

from perudo.models import Bid, Calza, Dudo
from perudo_app.core import (
    FACES, ROUTER, BotRegistry, GameSession, HumanSeat, Settings, Simulation, can, face_label, fmt_bid,
    legal_bids, lowest_legal,
)
from perudo_app.speech import Speaker

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cc9", "#3aa9c9", "#b4532a",
           "#5d8a1f", "#c6397f", "#8c8c8c"]

KV = """
#:import dp kivy.metrics.dp

<Bar>:
    canvas:
        Color:
            rgba: 0.85, 0.85, 0.88, 1
        Rectangle:
            pos: self.pos
            size: self.size
        Color:
            rgba: self.color
        Rectangle:
            pos: self.pos
            size: self.width * self.fraction, self.height

<DieBox>:
    size_hint: None, None
    size: dp(30), dp(30)
    font_size: dp(16)
    bold: True
    color: (0.1, 0.1, 0.12, 1) if not self.paco else (0.8, 0.2, 0.2, 1)
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(5)]
        Color:
            rgba: 0.3, 0.3, 0.35, 1
        Line:
            rounded_rectangle: self.x, self.y, self.width, self.height, dp(5)
            width: 1.1

<NavBar@BoxLayout>:
    size_hint_y: None
    height: dp(48)
    spacing: dp(4)
    padding: dp(4)

<HomeScreen>:
    BoxLayout:
        orientation: "vertical"
        padding: dp(16)
        spacing: dp(12)
        Label:
            text: "Perudo"
            font_size: dp(34)
            bold: True
            size_hint_y: None
            height: dp(70)
        Label:
            text: root.summary
            size_hint_y: None
            height: dp(40)
        Button:
            text: "Bots"
            on_release: app.go("bots")
        Button:
            text: "Watch a game"
            on_release: app.start_game(human=False)
        Button:
            text: "Join the table"
            on_release: app.start_game(human=True)
        Button:
            text: "Simulation"
            on_release: app.go("sim")
        Button:
            text: "Settings"
            on_release: app.go("settings")

<BotsScreen>:
    BoxLayout:
        orientation: "vertical"
        NavBar:
            Button:
                text: "Back"
                size_hint_x: None
                width: dp(80)
                on_release: app.go("home")
            Label:
                text: "Bots at the table"
            Button:
                text: "Load bot file"
                size_hint_x: None
                width: dp(130)
                on_release: root.pick_file()
        Label:
            id: errors
            text: ""
            color: 0.9, 0.3, 0.3, 1
            size_hint_y: None
            height: dp(24) if self.text else 0
            text_size: self.width, None
        ScrollView:
            GridLayout:
                id: rows
                cols: 1
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(2)

<GameScreen>:
    BoxLayout:
        orientation: "vertical"
        NavBar:
            Button:
                text: "Back"
                size_hint_x: None
                width: dp(80)
                on_release: root.leave()
            Label:
                id: title
                text: root.title
            ToggleButton:
                id: dice_toggle
                text: "Show dice"
                size_hint_x: None
                width: dp(110)
                state: "down" if root.show_dice else "normal"
                disabled: root.human_playing
                on_state: root.show_dice = self.state == "down"
        Label:
            id: totals
            text: root.totals_text
            size_hint_y: None
            height: dp(26) if self.text else 0
            bold: True
        GridLayout:
            id: players
            cols: 1
            size_hint_y: None
            height: self.minimum_height
            spacing: dp(2)
            padding: dp(4), 0
        Label:
            id: status
            text: root.status
            size_hint_y: None
            height: dp(30)
            bold: True
        ScrollView:
            id: log_scroll
            Label:
                id: log
                text: root.log_text
                size_hint_y: None
                height: self.texture_size[1]
                text_size: self.width - dp(8), None
                valign: "top"
                padding: dp(4), dp(4)
        BoxLayout:
            id: human_controls
            orientation: "vertical"
            size_hint_y: None
            height: dp(110) if root.pending else 0
            opacity: 1 if root.pending else 0
            disabled: not root.pending
            padding: dp(4)
            spacing: dp(4)
            BoxLayout:
                id: faces
                size_hint_y: None
                height: dp(44)
                spacing: dp(4)
            BoxLayout:
                size_hint_y: None
                height: dp(44)
                spacing: dp(4)
                Button:
                    text: "-"
                    size_hint_x: None
                    width: dp(44)
                    on_release: root.change_quantity(-1)
                Label:
                    id: bid_label
                    text: root.bid_text
                    bold: True
                Button:
                    text: "+"
                    size_hint_x: None
                    width: dp(44)
                    on_release: root.change_quantity(1)
                Button:
                    text: "Bid"
                    on_release: root.submit_bid()
                Button:
                    id: dudo_btn
                    text: "Dudo"
                    on_release: root.submit(Dudo())
                Button:
                    id: calza_btn
                    text: "Calza"
                    on_release: root.submit(Calza())
        BoxLayout:
            id: watch_controls
            size_hint_y: None
            height: dp(48) if not root.human_playing else 0
            opacity: 0 if root.human_playing else 1
            disabled: root.human_playing
            spacing: dp(4)
            padding: dp(4)
            Button:
                text: "Next turn"
                on_release: root.next_turn()
            ToggleButton:
                id: auto
                text: "Auto"
                on_state: root.set_auto(self.state == "down")
            Button:
                text: "New game"
                on_release: root.restart()

<SimScreen>:
    BoxLayout:
        orientation: "vertical"
        NavBar:
            Button:
                text: "Back"
                size_hint_x: None
                width: dp(80)
                on_release: root.leave()
            Label:
                text: "Simulation"
            TextInput:
                id: games
                text: str(app.settings["sim_games"])
                input_filter: "int"
                multiline: False
                size_hint_x: None
                width: dp(80)
            Button:
                id: run
                text: "Start"
                size_hint_x: None
                width: dp(80)
                on_release: root.toggle()
        Label:
            id: progress
            text: root.progress
            size_hint_y: None
            height: dp(28)
        ScrollView:
            GridLayout:
                id: board
                cols: 1
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(6)
                padding: dp(8)

<SettingsScreen>:
    BoxLayout:
        orientation: "vertical"
        NavBar:
            Button:
                text: "Back"
                size_hint_x: None
                width: dp(80)
                on_release: app.go("home")
            Label:
                text: "Settings"
        GridLayout:
            cols: 2
            padding: dp(12)
            spacing: dp(8)
            row_default_height: dp(44)
            row_force_default: True
            Label:
                text: "Text to speech (edge-tts, needs internet)"
                text_size: self.width, None
            Switch:
                id: tts
                active: app.settings["tts"]
                on_active: app.set_setting("tts", self.active)
            Label:
                text: "Voice language"
            Spinner:
                id: lang
                text: app.settings["tts_lang"]
                values: ["nl", "en"]
                on_text: app.set_setting("tts_lang", self.text)
            Label:
                text: "Show everyone's dice when watching"
                text_size: self.width, None
            Switch:
                active: app.settings["show_dice"]
                on_active: app.set_setting("show_dice", self.active)
            Label:
                text: "Seconds per turn in auto mode"
                text_size: self.width, None
            Slider:
                id: delay
                min: 0
                max: 3
                step: 0.25
                value: app.settings["step_delay"]
                on_value: app.set_setting("step_delay", float(self.value))
            Label:
                text: "Your name at the table"
            TextInput:
                text: app.settings["human_name"]
                multiline: False
                on_text_validate: app.set_setting("human_name", self.text.strip() or "You")
                on_focus: if not args[1]: app.set_setting("human_name", self.text.strip() or "You")
            Label:
                text: "Games per simulation"
            TextInput:
                text: str(app.settings["sim_games"])
                input_filter: "int"
                multiline: False
                on_focus: if not args[1]: app.set_setting("sim_games", int(self.text or 100))
        Label:
            id: speech_status
            text: root.speech_status
            size_hint_y: None
            height: dp(40)
            color: 0.9, 0.5, 0.3, 1
            text_size: self.width - dp(16), None
        Widget:
"""


class Bar(Widget):
    fraction = NumericProperty(0.0)
    color = ListProperty([0.2, 0.5, 0.9, 1])


class DieBox(Label):
    paco = BooleanProperty(False)


def rgba(hex_color: str):
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1]


# ---------------------------------------------------------------------------

class HomeScreen(Screen):
    summary = StringProperty("")

    def on_pre_enter(self):
        app = App.get_running_app()
        names = [e.display_name + (f" x{e.copies}" if e.copies > 1 else "") for e in app.registry.selected()]
        self.summary = ", ".join(names) if names else "No bots selected"


class BotsScreen(Screen):
    def on_pre_enter(self):
        self.refresh()

    def refresh(self):
        app = App.get_running_app()
        rows = self.ids.rows
        rows.clear_widgets()
        for entry in app.registry.entries:
            row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(4), padding=(dp(4), 0))
            box = CheckBox(active=entry.enabled, size_hint_x=None, width=dp(44))
            box.bind(active=lambda cb, v, e=entry: setattr(e, "enabled", v))
            row.add_widget(box)
            source = "" if entry.source == "bundled" else f"  ({Path(entry.source).name})"
            row.add_widget(Label(text=f"{entry.display_name}{source}", halign="left",
                                 text_size=(None, None)))
            minus = Button(text="-", size_hint_x=None, width=dp(40))
            count = Label(text=str(entry.copies), size_hint_x=None, width=dp(30))
            plus = Button(text="+", size_hint_x=None, width=dp(40))

            def change(_, e=entry, lbl=count, d=0):
                e.copies = max(1, min(6, e.copies + d))
                lbl.text = str(e.copies)

            minus.bind(on_release=lambda b, e=entry, lbl=count: change(b, e, lbl, -1))
            plus.bind(on_release=lambda b, e=entry, lbl=count: change(b, e, lbl, 1))
            row.add_widget(minus)
            row.add_widget(count)
            row.add_widget(plus)
            rows.add_widget(row)
        self.ids.errors.text = "\n".join(app.registry.errors[-3:])

    def pick_file(self):
        app = App.get_running_app()
        chooser = FileChooserListView(path=app.bots_dir, filters=["*.py"])
        box = BoxLayout(orientation="vertical")
        box.add_widget(chooser)
        buttons = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(4))
        popup = Popup(title="Pick a bot file (.py)", content=box, size_hint=(0.95, 0.9))

        def load(_):
            if chooser.selection:
                app.registry.add_file(chooser.selection[0])
                self.refresh()
            popup.dismiss()

        ok = Button(text="Load")
        ok.bind(on_release=load)
        cancel = Button(text="Cancel")
        cancel.bind(on_release=lambda _: popup.dismiss())
        buttons.add_widget(ok)
        buttons.add_widget(cancel)
        box.add_widget(buttons)
        popup.open()


class GameScreen(Screen):
    title = StringProperty("Game")
    status = StringProperty("")
    totals_text = StringProperty("")
    log_text = StringProperty("")
    bid_text = StringProperty("")
    show_dice = BooleanProperty(True)
    human_playing = BooleanProperty(False)
    pending = ObjectProperty(None, allownone=True)
    session = ObjectProperty(None, allownone=True)

    def __init__(self, **kw):
        super().__init__(**kw)
        self._face = 2
        self._quantity = 1
        self._poll = None
        self._last_log_len = -1

    def begin(self, session: GameSession, human: bool):
        self.session = session
        self.human_playing = human
        self.title = "Your game" if human else "Watching"
        self.show_dice = (not human) and App.get_running_app().settings["show_dice"]
        self.pending = None
        self._last_log_len = -1
        self.ids.auto.state = "down" if human else "normal"
        session.set_auto(human or self.ids.auto.state == "down", App.get_running_app().settings["step_delay"])
        session.start()
        if self._poll is None:
            self._poll = Clock.schedule_interval(self.refresh, 0.15)

    def leave(self):
        if self.session is not None:
            self.session.stop()
        if self._poll is not None:
            self._poll.cancel()
            self._poll = None
        App.get_running_app().speaker.clear()
        App.get_running_app().go("home")

    def restart(self):
        if self.session is not None:
            self.session.stop()
        App.get_running_app().start_game(human=self.human_playing)

    def next_turn(self):
        if self.session is not None:
            self.session.next()

    def set_auto(self, auto: bool):
        if self.session is not None:
            self.session.set_auto(auto, App.get_running_app().settings["step_delay"])

    # -- drawing a snapshot --------------------------------------------

    def refresh(self, _dt=0):
        if self.session is None:
            return
        snap = self.session.snapshot()
        reveal = snap.reveal
        pal = " (Palifico)" if snap.palifico else ""
        if snap.error:
            self.status = f"Error: {snap.error}"
        elif snap.finished:
            self.status = f"{snap.standings[0]} wins" if snap.standings else "Stopped"
        elif snap.pending is not None:
            self.status = f"Round {snap.round_number}{pal}: your turn, facing {fmt_bid(snap.current_bid)}"
        else:
            self.status = f"Round {snap.round_number}{pal}: {snap.to_act or '...'} to act, bid {fmt_bid(snap.current_bid)}"
        show = self.show_dice and not self.human_playing
        if show and snap.players:
            tot, nat = snap.totals(), snap.naturals()
            self.totals_text = "   ".join(
                f"{face_label(f)}: {tot[f]}" + (f" ({nat[f]})" if tot[f] != nat[f] else "") for f in FACES)
        else:
            self.totals_text = ""
        grid = self.ids.players
        grid.clear_widgets()
        revealed = dict(reveal["dice"]) if reveal else {}
        for i, p in enumerate(snap.players):
            row = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(4))
            marker = ">" if p.name == snap.to_act and not snap.finished else " "
            name = Label(text=f"{marker} {p.name}  [{p.dice_count}]", halign="left", size_hint_x=0.5,
                         color=rgba(PALETTE[i % len(PALETTE)]) if p.alive else (0.5, 0.5, 0.5, 1))
            name.bind(size=lambda lbl, s: setattr(lbl, "text_size", s))
            row.add_widget(name)
            dice = p.dice if (show or p.is_human) else revealed.get(p.name, ())
            for d in dice:
                row.add_widget(DieBox(text=face_label(d), paco=(d == 1)))
            row.add_widget(Widget())
            grid.add_widget(row)
        if len(snap.log) != self._last_log_len:
            self._last_log_len = len(snap.log)
            lines = snap.log[-60:]
            if snap.commentary:
                lines = lines + [""] + [f'"{c}"' for c in snap.commentary[-6:]]
            self.log_text = "\n".join(lines)
            Clock.schedule_once(lambda dt: setattr(self.ids.log_scroll, "scroll_y", 0), 0)
        if snap.pending is not None and self.pending is None:
            self._setup_choice(snap.pending)
        self.pending = snap.pending

    # -- the human's decision ------------------------------------------

    def _setup_choice(self, state):
        faces_box = self.ids.faces
        faces_box.clear_widgets()
        bids = legal_bids(state)
        legal_faces = sorted({b.face for b in bids})
        for f in FACES:
            btn = Button(text=face_label(f), disabled=f not in legal_faces)
            btn.bind(on_release=lambda b, face=f: self._pick_face(face))
            faces_box.add_widget(btn)
        first = state.current_bid.face if state.current_bid and state.current_bid.face in legal_faces else (
            legal_faces[0] if legal_faces else 2)
        self._pick_face(first)
        self.ids.dudo_btn.disabled = not can(state, Dudo)
        self.ids.calza_btn.disabled = not can(state, Calza)

    def _pick_face(self, face: int):
        state = self.session.snapshot().pending if self.session else None
        if state is None:
            return
        self._face = face
        low = lowest_legal(state, face)
        self._quantity = low.quantity if low else 1
        self._show_bid()

    def change_quantity(self, delta: int):
        state = self.session.snapshot().pending if self.session else None
        if state is None:
            return
        legal = {b.quantity for b in legal_bids(state) if b.face == self._face}
        q = self._quantity + delta
        while q >= 1 and q <= max(legal, default=0) and q not in legal:
            q += delta
        if q in legal:
            self._quantity = q
        self._show_bid()

    def _show_bid(self):
        self.bid_text = fmt_bid(Bid(self._quantity, self._face))

    def submit_bid(self):
        self.submit(Bid(self._quantity, self._face))

    def submit(self, action):
        state = self.session.snapshot().pending if self.session else None
        if state is None or action not in state.legal_actions:
            return
        self.pending = None
        self.session.submit(action)


class SimScreen(Screen):
    progress = StringProperty("")
    sim = ObjectProperty(None, allownone=True)

    def __init__(self, **kw):
        super().__init__(**kw)
        self._poll = None
        self._rows = {}

    def toggle(self):
        if self.sim is not None and self.sim.running:
            self.sim.stop()
            return
        app = App.get_running_app()
        bots = app.registry.make(fresh=True)
        if len(bots) < 2:
            self.progress = "Select at least two bots"
            return
        try:
            games = max(1, int(self.ids.games.text or "1"))
        except ValueError:
            games = app.settings["sim_games"]
        app.set_setting("sim_games", games)
        self.sim = Simulation(bots, games)
        self.sim.start()
        self.ids.run.text = "Stop"
        self._build_board(bots)
        if self._poll is None:
            self._poll = Clock.schedule_interval(self.refresh, 0.25)

    def leave(self):
        if self.sim is not None:
            self.sim.stop()
        if self._poll is not None:
            self._poll.cancel()
            self._poll = None
        App.get_running_app().go("home")

    def _build_board(self, bots):
        board = self.ids.board
        board.clear_widgets()
        self._rows = {}
        for i, bot in enumerate(bots):
            row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
            name = Label(text=bot.name, size_hint_x=0.38, halign="left")
            name.bind(size=lambda lbl, s: setattr(lbl, "text_size", s))
            bar = Bar(color=rgba(PALETTE[i % len(PALETTE)]))
            pct = Label(text="0.0%", size_hint_x=None, width=dp(64), halign="right")
            row.add_widget(name)
            row.add_widget(bar)
            row.add_widget(pct)
            board.add_widget(row)
            self._rows[bot.name] = (row, bar, pct, name)

    def refresh(self, _dt=0):
        if self.sim is None:
            return
        rows = self.sim.standings()
        board = self.ids.board
        for rank, st in enumerate(rows):
            row, bar, pct, name = self._rows[st.name]
            bar.fraction = st.win_pct / 100
            pct.text = f"{st.win_pct:.1f}%"
            name.text = f"{rank + 1}. {st.name}   {st.wins} wins, avg place {st.avg_position:.2f}"
            if board.children[len(rows) - 1 - rank] is not row:   # keep rows in rank order
                board.remove_widget(row)
                board.add_widget(row, index=len(rows) - 1 - rank)
        self.progress = f"{self.sim.played} / {self.sim.games} games" + (f"   error: {self.sim.error}" if self.sim.error else "")
        if not self.sim.running:
            self.ids.run.text = "Start"


class SettingsScreen(Screen):
    speech_status = StringProperty("")

    def on_pre_enter(self):
        app = App.get_running_app()
        self.speech_status = f"Speech: {app.speaker.error}" if app.speaker.error else ""


class PerudoApp(App):
    def build(self):
        self.title = "Perudo"
        data_dir = Path(self.user_data_dir)
        self.settings = Settings(str(data_dir / "settings.json"))
        self.registry = BotRegistry()
        self.registry.scan_bundled()
        self.bots_dir = self._bots_dir()
        self.speaker = Speaker(str(data_dir / "edge-tts"), play=self._play_mp3, enabled=self.settings["tts"])
        ROUTER.install()
        ROUTER.speak = lambda text, lang, seed: self.speaker.say(text, self.settings["tts_lang"] or lang, seed)
        Builder.load_string(KV)
        sm = ScreenManager()
        for cls, name in ((HomeScreen, "home"), (BotsScreen, "bots"), (GameScreen, "game"),
                          (SimScreen, "sim"), (SettingsScreen, "settings")):
            sm.add_widget(cls(name=name))
        self.sm = sm
        return sm

    def _bots_dir(self) -> str:
        try:
            from android.storage import primary_external_storage_path   # type: ignore
            from android.permissions import Permission, request_permissions   # type: ignore
            request_permissions([Permission.READ_EXTERNAL_STORAGE])
            return primary_external_storage_path()
        except Exception:
            return str(Path.home())

    def go(self, name: str):
        self.sm.current = name

    def set_setting(self, key, value):
        self.settings[key] = value
        if key == "tts":
            self.speaker.enabled = bool(value)

    def start_game(self, human: bool):
        bots = self.registry.make(fresh=True)
        if human:
            bots.append(HumanSeat(self.settings["human_name"] or "You"))
        if len(bots) < 2:
            self.sm.get_screen("home").summary = "Select at least one bot first"
            self.go("bots")
            return
        session = GameSession(bots, step=not human, delay=self.settings["step_delay"])
        self.speaker.clear()
        self.sm.get_screen("game").begin(session, human)
        self.go("game")

    def _play_mp3(self, path: str):
        """Called from the speech thread; play on the UI thread and wait."""
        import threading
        done = threading.Event()

        def start(_dt):
            sound = SoundLoader.load(path)
            if sound is None:
                done.set()
                return
            sound.bind(on_stop=lambda *_: done.set())
            sound.play()

        Clock.schedule_once(start, 0)
        if not done.wait(20):
            return
        if SoundLoader.load(path) is None:        # no mp3 provider (desktop): fall back to ffplay
            try:
                from perudo.tts import _play_mp3
                _play_mp3(Path(path).read_bytes())
            except Exception as exc:
                self.speaker.error = f"{type(exc).__name__}: {exc}"


def run():
    PerudoApp().run()
