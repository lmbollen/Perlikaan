"""
The screens for playing together: host a lobby or join one with its code,
the name prompt, and the lobby (players, bots, the host's rules, start).
The game itself is shown by the ordinary game screen.
"""

from __future__ import annotations

import threading

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.screenmanager import Screen
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget

from perudo_app import theme as T
from perudo_app.widgets import Avatar, Body, Btn, Caption, IconBtn, Surface, TextField, Toggle

KV = """
#:import T perudo_app.theme

<CodeField@TextField>:
    font_size: sp(30)
    bold: True
    halign: "center"
    height: dp(64)
    hint_text: "CODE"
    write_tab: False

<PlayTogetherScreen>:
    BoxLayout:
        orientation: "vertical"
        TopBar:
            title: "Play together"
            on_back: app.back()
        ScrollView:
            do_scroll_x: False
            BoxLayout:
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16), dp(2), dp(16), dp(24)
                spacing: dp(14)
                Sub:
                    text: ("Play with friends anywhere, or on the same Wi-Fi. " if root.online_ok else "Play with friends on the same Wi-Fi, or on the hotspot of one of your phones. ") + "One phone hosts the lobby; the others join with its code."
                    font_size: sp(14)
                Surface:
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
                    padding: dp(16), dp(14), dp(16), dp(16)
                    spacing: dp(12)
                    Caption:
                        text: "Host"
                    Sub:
                        text: "Start a lobby on this phone and send its code to the others. You decide which bots play."
                    Segmented:
                        size_hint_y: None
                        height: dp(40) if root.online_ok else 0
                        opacity: 1 if root.online_ok else 0
                        disabled: not root.online_ok or root.busy
                        options: [("online", "Over the internet"), ("local", "Same Wi-Fi")]
                        value: "online" if app.settings["host_online"] else "local"
                        on_value: app.set_setting("host_online", self.value == "online")
                    Btn:
                        kind: "primary"
                        text: "Host a lobby"
                        size_hint_y: None
                        height: dp(52)
                        font_size: sp(16)
                        disabled: root.busy
                        on_release: root.host()
                Surface:
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
                    padding: dp(16), dp(14), dp(16), dp(16)
                    spacing: dp(12)
                    Caption:
                        text: "Join"
                    CodeField:
                        id: code
                        on_text: root.code_typed(self)
                        on_text_validate: root.join()
                    Btn:
                        kind: "primary"
                        text: "Join the lobby"
                        size_hint_y: None
                        height: dp(52)
                        font_size: sp(16)
                        disabled: root.busy or len(code.text) != 4
                        on_release: root.join()
                Sub:
                    text: root.status
                    color: root.status_color
                    font_size: sp(14)
                    opacity: 1 if root.status else 0
                BoxLayout:
                    size_hint_y: None
                    height: dp(40)
                    Label:
                        markup: True
                        text: "Playing as  [b]" + root.player_name + "[/b]"
                        color: T.MUTED
                        font_size: sp(14)
                        halign: "left"
                        valign: "middle"
                        text_size: self.size
                    Btn:
                        kind: "ghost"
                        text: "Change"
                        color: T.GOLD
                        font_size: sp(14)
                        size_hint_x: None
                        width: dp(90)
                        on_release: app.ask_name(root.refresh_name, force=True)

<LobbyScreen>:
    BoxLayout:
        orientation: "vertical"
        TopBar:
            title: "Lobby"
            on_back: root.leave()
            Label:
                text: root.code
                font_size: sp(18)
                bold: True
                color: T.GOLD
                size_hint_x: None
                width: dp(90)
        ScrollView:
            do_scroll_x: False
            BoxLayout:
                id: body
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16), dp(2), dp(16), dp(20)
                spacing: dp(12)
        Surface:
            size_hint_y: None
            height: dp(76)
            radius: 0
            padding: dp(20), dp(14), dp(16), dp(14)
            spacing: dp(12)
            BoxLayout:
                orientation: "vertical"
                Label:
                    text: root.foot_title
                    bold: True
                    font_size: sp(16)
                    color: T.TEXT
                    halign: "left"
                    valign: "bottom"
                    text_size: self.size
                    shorten: True
                Label:
                    text: root.foot_sub
                    font_size: sp(12.5)
                    color: T.MUTED
                    halign: "left"
                    valign: "top"
                    text_size: self.size
                    shorten: True
            Btn:
                id: action
                kind: "primary"
                text: root.action_text
                size_hint_x: None
                width: dp(150) if root.action_text else 0
                opacity: 1 if root.action_text else 0
                disabled: not root.action_enabled
                font_size: sp(15)
                on_release: root.action()
"""


def _row(height=dp(56)):
    return BoxLayout(size_hint_y=None, height=height, spacing=dp(12))


def _label(text, size=15, color=None, bold=False, markup=True):
    lbl = Label(text=text, markup=markup, font_size=sp(size), color=color or T.TEXT, bold=bold,
                halign="left", valign="middle", shorten=True)
    lbl.bind(size=lambda w, v: setattr(w, "text_size", v))
    return lbl


def _two_lines(top, bottom, color=None):
    box = BoxLayout(orientation="vertical")
    a = _label(top, 15, color, bold=True)
    a.valign = "bottom"
    b = _label(bottom, 12.5, T.MUTED)
    b.valign = "top"
    box.add_widget(a)
    box.add_widget(b)
    return box


def _card(title):
    card = Surface(orientation="vertical", size_hint_y=None, padding=(dp(16), dp(12), dp(16), dp(14)),
                   spacing=dp(8))
    card.bind(minimum_height=card.setter("height"))
    card.add_widget(Caption(text=title))
    return card


# ---------------------------------------------------------------------------

class PlayTogetherScreen(Screen):
    status = StringProperty("")
    status_color = T.MUTED
    busy = BooleanProperty(False)
    player_name = StringProperty("")
    online_ok = BooleanProperty(False)       # this build can play over the internet

    def on_pre_enter(self, *_):
        self.refresh_name()
        self.online_ok = App.get_running_app().relay() is not None

    def refresh_name(self, *_):
        self.player_name = T.esc(App.get_running_app().settings["human_name"] or "You")

    def code_typed(self, field):
        cleaned = "".join(ch for ch in field.text.upper() if ch.isalnum())[:4]
        if cleaned != field.text:
            field.text = cleaned

    def _say(self, text, error=False):
        self.status_color = T.RED if error else T.MUTED
        self.status = text

    def host(self):
        app = App.get_running_app()
        app.ask_name(self._host)

    def _host(self):
        app = App.get_running_app()
        self.refresh_name()
        if self.busy:
            return
        online = self.online_ok and bool(app.settings["host_online"])
        self.busy = True
        self._say("Opening a lobby…")

        def work():
            try:
                lobby = app.make_host(online)
                Clock.schedule_once(lambda _dt: self._hosted(lobby), 0)
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                Clock.schedule_once(lambda _dt: self._failed(f"Could not start a lobby: {message}"), 0)

        threading.Thread(target=work, daemon=True).start()

    def _hosted(self, lobby):
        self.busy = False
        self.status = ""
        App.get_running_app().enter_lobby(lobby, host=True)

    def join(self):
        if len(self.ids.code.text) != 4 or self.busy:
            return
        App.get_running_app().ask_name(self._join)

    def _join(self):
        app = App.get_running_app()
        self.refresh_name()
        code = self.ids.code.text
        self.busy = True
        self._say(f"Looking for lobby {code}…")

        def work():
            try:
                client = app.make_client(code)
                Clock.schedule_once(lambda _dt: self._joined(client), 0)
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                Clock.schedule_once(lambda _dt: self._failed(message), 0)

        threading.Thread(target=work, daemon=True).start()

    def _joined(self, client):
        self.busy = False
        self.status = ""
        App.get_running_app().enter_lobby(client, host=False)

    def _failed(self, message):
        self.busy = False
        hint = ""
        if "no lobby" in message.lower():
            hint = (" Check the code; for a Same Wi-Fi lobby, join from the host's Wi-Fi." if self.online_ok
                    else " Check that you are on the same Wi-Fi as the host.")
        self._say(message + "." + hint if not message.endswith(".") else message + hint, error=True)


class NameDialog(ModalView):
    """Asks for the player's name once; it is remembered."""

    def __init__(self, current, on_done, **kw):
        super().__init__(size_hint=(0.88, None), height=dp(250), background="", background_color=T.CLEAR,
                         overlay_color=T.BG[:3] + [0.75], auto_dismiss=False, **kw)
        self.on_done = on_done
        card = Surface(orientation="vertical", padding=dp(20), spacing=dp(10), radius=dp(22))
        card.add_widget(_label("[b]What's your name?[/b]", 19))
        card.add_widget(Body(text="The others see it at the table. You can change it in the lobby."))
        self.field = TextField(text=current, hint_text="Your name")
        self.field.bind(on_text_validate=lambda *_: self.done())
        card.add_widget(self.field)
        buttons = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(10))
        cancel = Btn(text="Cancel", kind="outline")
        cancel.bind(on_release=lambda *_: self.dismiss())
        ok = Btn(text="OK", kind="primary")
        ok.bind(on_release=lambda *_: self.done())
        buttons.add_widget(cancel)
        buttons.add_widget(ok)
        card.add_widget(buttons)
        self.add_widget(card)
        Clock.schedule_once(lambda _dt: setattr(self.field, "focus", True), 0.2)

    def done(self):
        name = " ".join(self.field.text.split())[:20]
        if not name:
            return
        app = App.get_running_app()
        app.set_setting("human_name", name)
        app.set_setting("name_confirmed", True)
        self.dismiss()
        self.on_done()


# ---------------------------------------------------------------------------

class LobbyScreen(Screen):
    code = StringProperty("")
    foot_title = StringProperty("")
    foot_sub = StringProperty("")
    action_text = StringProperty("")
    action_enabled = BooleanProperty(False)

    def __init__(self, **kw):
        super().__init__(**kw)
        self._poll = None
        self._shown = None
        self._session = None
        self._name_field = None
        self._name_card = None

    # -- life ------------------------------------------------------------------

    def on_pre_enter(self, *_):
        self._shown = None
        self.refresh()
        if self._poll is None:
            self._poll = Clock.schedule_interval(self.refresh, 0.25)

    def on_leave(self, *_):
        if self._poll is not None:
            self._poll.cancel()
            self._poll = None

    @property
    def lobby(self):
        return App.get_running_app().lobby

    @property
    def is_host(self):
        return App.get_running_app().lobby_host

    def leave(self):
        app = App.get_running_app()
        if self.is_host:
            self._confirm("Close the lobby?", "Everyone in it is sent back to the start.", "Close",
                          app.leave_lobby)
        else:
            app.leave_lobby()

    def action(self):
        app = App.get_running_app()
        if self.is_host:
            try:
                app.start_lobby_game()
            except ValueError as exc:
                app.toast(str(exc))
        else:
            app.open_lobby_game()

    # -- reading the lobby -------------------------------------------------------

    def _state(self):
        lobby = self.lobby
        if lobby is None:
            return None, None
        if self.is_host:
            return lobby.state(), 0
        return lobby.lobby, lobby.my_id

    def refresh(self, *_dt):
        app = App.get_running_app()
        lobby = self.lobby
        if lobby is None:
            return
        if not self.is_host:
            if lobby.closed:
                app.leave_lobby(message=lobby.error or "The lobby is closed")
                return
            if lobby.notice:
                app.toast(T.esc(lobby.notice))
                lobby.notice = None
            if lobby.session is not None and lobby.session is not self._session:
                self._session = lobby.session                 # the host started a game
                app.open_lobby_game()
                return
        state, me = self._state()
        if not state:
            return
        self.code = state.get("code", "")
        key = (repr(state), me)
        if key == self._shown:
            self._footer(state, me)
            return
        self._shown = key
        self._build(state, me)
        self._footer(state, me)

    def _footer(self, state, me):
        players = len(state.get("members", [])) + sum(1 for s in state.get("seats", []) if s["plays"])
        game_on = state.get("phase") == "game"
        if self.is_host:
            self.foot_title = f"{players} at the table"
            self.foot_sub = "A game is going on" if game_on else (
                "Invite someone or add a bot" if players < 2 else "Everyone in the lobby plays")
            self.action_text = "Back to the game" if game_on else "Start game"
            self.action_enabled = game_on or players >= 2
        else:
            host = state.get("host", "the host")
            session = self.lobby.session
            if game_on and session is not None:
                self.foot_title, self.foot_sub = "A game is going on", f"Hosted by {host}"
                self.action_text, self.action_enabled = "Back to the game", True
            else:
                self.foot_title = f"{players} at the table"
                self.foot_sub = f"Waiting for {host} to start"
                self.action_text, self.action_enabled = "", False

    # -- drawing it ----------------------------------------------------------------

    def _build(self, state, me):
        app = App.get_running_app()
        body = self.ids.body
        focused = self._name_field is not None and self._name_field.focus
        if not focused:
            body.clear_widgets()
        else:                                  # keep the name being typed; redraw the rest
            for w in list(body.children):
                if w is not self._name_card:
                    body.remove_widget(w)
        host = self.is_host
        members = state.get("members", [])
        seats = state.get("seats", [])
        colors = {m["id"]: T.seat_color(i) for i, m in enumerate(members)}

        # the code
        if not focused:
            code_card = Surface(orientation="vertical", size_hint_y=None, height=dp(126),
                                padding=(dp(16), dp(12)), spacing=dp(2), border=T.GOLD, border_width=dp(1.4))
            code_card.add_widget(Caption(text="Lobby code"))
            code_card.add_widget(Label(text="  ".join(state.get("code", "")), font_size=sp(40), bold=True,
                                       color=T.GOLD, size_hint_y=None, height=dp(58)))
            code_card.add_widget(Body(
                text=(("The others join from Play together with this code, from anywhere."
                       if getattr(self.lobby, "online", False) else
                       "The others join from Play together with this code, on the same Wi-Fi or your hotspot.")
                      if host else f"You are in {T.esc(state.get('host', ''))}'s lobby."),
                font_size=sp(13), halign="center"))
            body.add_widget(code_card)

            # your name
            name_card = _card("Your name")
            field = TextField(text=self._my_name(state, me))
            field.bind(on_text_validate=lambda f: self._rename(f.text),
                       focus=lambda f, on: (not on) and self._rename(f.text))
            name_card.add_widget(field)
            self._name_field, self._name_card = field, name_card
            body.add_widget(name_card)

        # the players
        players = _card(f"Players  ·  {len(members)}")
        for m in members:
            row = _row(dp(48))
            row.add_widget(Avatar(name=m["name"], bg=colors[m["id"]], size=(dp(34), dp(34)),
                                  pos_hint={"center_y": 0.5}))
            tags = []
            if m["host"]:
                tags.append("host")
            if m["id"] == me:
                tags.append("you")
            row.add_widget(_two_lines(T.esc(m["name"]), " · ".join(tags) or "player"))
            players.add_widget(row)
        body.add_widget(players)

        # the bots
        bots = _card(f"Bots  ·  {sum(1 for s in seats if s['plays'])} playing")
        if not seats:
            bots.add_widget(Body(text="No bots yet." + (" Add some, or let the guests bring theirs." if host else ""),
                                 font_size=sp(13)))
        for s in seats:
            row = _row(dp(56))
            row.add_widget(Avatar(name=s["name"], bg=colors.get(s["owner"], T.LINE) if s["plays"] else T.LINE,
                                  size=(dp(34), dp(34)), pos_hint={"center_y": 0.5}))
            owner = "your bot" if s["owner"] == me else f"{T.esc(s['owner_name'])}'s bot"
            status = "" if s["plays"] else "  ·  sits out"
            row.add_widget(_two_lines(T.esc(s["name"]), owner + status,
                                      T.TEXT if s["plays"] else T.MUTED))
            if host:
                toggle = Toggle(active=s["plays"], pos_hint={"center_y": 0.5})
                toggle.bind(active=lambda t, v, sid=s["id"]: self.lobby.set_plays(sid, v))
                row.add_widget(toggle)
            if host or s["owner"] == me:
                remove = IconBtn(text=T.icon("✕"), font_size=sp(16), color=T.MUTED,
                                 pos_hint={"center_y": 0.5})
                remove.bind(on_release=lambda _b, sid=s["id"]: self._remove(sid))
                row.add_widget(remove)
            bots.add_widget(row)
        mine = sum(1 for s in seats if s["owner"] == me)
        allowed = state.get("allow_guest_bots", True)
        limit = state.get("guest_bot_limit", 2)
        if host or (allowed and mine < limit):
            add = Btn(text="+  Add a bot", kind="outline", size_hint_y=None, height=dp(44), font_size=sp(14))
            add.bind(on_release=lambda *_: self._pick_bot())
            bots.add_widget(add)
        if not host:
            if not allowed:
                bots.add_widget(Body(text="The host does not let guests bring bots.", font_size=sp(13)))
            else:
                bots.add_widget(Body(text=f"You brought {mine} of {limit} bot{'s' if limit != 1 else ''}. "
                                          "They play on this phone.", font_size=sp(13)))
        else:
            bots.add_widget(Body(text="Switch a bot off to let it sit out. Guests' bots play on their phones.",
                                 font_size=sp(13)))
        body.add_widget(bots)

        # the host's rules
        if host:
            rules = _card("Guests' bots")
            row = _row(dp(52))
            row.add_widget(_two_lines("Guests may bring bots", "They choose from the bots on their phone"))
            allow = Toggle(active=allowed, pos_hint={"center_y": 0.5})
            allow.bind(active=lambda t, v: self.lobby.set_guest_bots(v, self.lobby.guest_bot_limit))
            row.add_widget(allow)
            rules.add_widget(row)
            row = _row(dp(52))
            row.add_widget(_two_lines("Bots per guest", "At most this many each"))
            minus = Btn(text="−", size_hint=(None, None), size=(dp(38), dp(38)), radius=dp(19),
                        pos_hint={"center_y": 0.5}, disabled=not allowed or limit <= 1)
            minus.bind(on_release=lambda *_: self._limit(-1))
            count = Label(text=str(limit), bold=True, font_size=sp(17), color=T.TEXT, size_hint_x=None,
                          width=dp(30))
            plus = Btn(text="+", size_hint=(None, None), size=(dp(38), dp(38)), radius=dp(19),
                       pos_hint={"center_y": 0.5}, disabled=not allowed or limit >= 6)
            plus.bind(on_release=lambda *_: self._limit(1))
            for w in (minus, count, plus):
                row.add_widget(w)
            rules.add_widget(row)
            body.add_widget(rules)
            app.set_setting("guest_bots", allowed)
            app.set_setting("guest_bot_limit", limit)

    def _my_name(self, state, me):
        for m in state.get("members", []):
            if m["id"] == me:
                return m["name"]
        return App.get_running_app().settings["human_name"]

    # -- what the buttons do -----------------------------------------------------------

    def _rename(self, text):
        name = " ".join(text.split())[:20]
        app = App.get_running_app()
        state, me = self._state()
        if not name or name == self._my_name(state or {}, me):
            return
        app.set_setting("human_name", name)
        app.set_setting("name_confirmed", True)
        self.lobby.rename(name)

    def _remove(self, seat_id):
        if self.is_host:
            self.lobby.remove_seat(seat_id)
        else:
            self.lobby.remove_bot(seat_id)

    def _limit(self, delta):
        lobby = self.lobby
        lobby.set_guest_bots(lobby.allow_guest_bots, max(1, min(6, lobby.guest_bot_limit + delta)))

    def _pick_bot(self):
        app = App.get_running_app()
        sheet = ModalView(size_hint=(0.92, 0.8), background="", background_color=T.CLEAR,
                          overlay_color=T.BG[:3] + [0.75])
        card = Surface(orientation="vertical", padding=dp(16), spacing=dp(8), radius=dp(22))
        card.add_widget(_label("[b]Add a bot[/b]", 19))
        card.add_widget(Body(text="Your bots: the ones from the repository and the ones you imported.",
                             font_size=sp(13)))
        scroll = ScrollView(do_scroll_x=False)
        grid = GridLayout(cols=1, size_hint_y=None, spacing=dp(6))
        grid.bind(minimum_height=grid.setter("height"))
        for i, entry in enumerate(app.registry.entries):
            b = Btn(text=T.esc(entry.display_name), kind="secondary", size_hint_y=None, height=dp(46),
                    halign="left", font_size=sp(15), padding=(dp(14), 0))

            def add(_b, e=entry):
                self.lobby.add_bot(app.registry.key(e), e.display_name)
                sheet.dismiss()

            b.bind(on_release=add)
            grid.add_widget(b)
        scroll.add_widget(grid)
        card.add_widget(scroll)
        close = Btn(text="Close", kind="outline", size_hint_y=None, height=dp(46))
        close.bind(on_release=lambda *_: sheet.dismiss())
        card.add_widget(close)
        sheet.add_widget(card)
        sheet.open()

    def _confirm(self, title, text, yes, then):
        sheet = ModalView(size_hint=(0.88, None), height=dp(200), background="", background_color=T.CLEAR,
                          overlay_color=T.BG[:3] + [0.75])
        card = Surface(orientation="vertical", padding=dp(20), spacing=dp(10), radius=dp(22))
        card.add_widget(_label(f"[b]{title}[/b]", 19))
        card.add_widget(Body(text=text))
        card.add_widget(Widget())
        buttons = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(10))
        no = Btn(text="Cancel", kind="outline")
        no.bind(on_release=lambda *_: sheet.dismiss())
        ok = Btn(text=yes, kind="danger")
        ok.bind(on_release=lambda *_: (sheet.dismiss(), then()))
        buttons.add_widget(no)
        buttons.add_widget(ok)
        card.add_widget(buttons)
        sheet.add_widget(card)
        sheet.open()


def confirm(title, text, yes, then):
    """A small yes/no sheet (used by the game screen too)."""
    LobbyScreen._confirm(None, title, text, yes, then)


def choose(title, text, options, cancel="Cancel"):
    """A sheet with a few choices, one under the other: options is a list of
    (label, button kind, then), then a button that only closes it."""
    sheet = ModalView(size_hint=(0.88, None), height=dp(150 + 58 * (len(options) + 1) + 20 * text.count("\n")),
                      background="",
                      background_color=T.CLEAR, overlay_color=T.BG[:3] + [0.75])
    card = Surface(orientation="vertical", padding=dp(20), spacing=dp(10), radius=dp(22))
    card.add_widget(_label(f"[b]{title}[/b]", 19))
    card.add_widget(Body(text=text))
    card.add_widget(Widget())
    for label, kind, then in options:
        b = Btn(text=label, kind=kind, size_hint_y=None, height=dp(48))
        b.bind(on_release=lambda *_, then=then: (sheet.dismiss(), then()))
        card.add_widget(b)
    no = Btn(text=cancel, kind="outline", size_hint_y=None, height=dp(48))
    no.bind(on_release=lambda *_: sheet.dismiss())
    card.add_widget(no)
    sheet.add_widget(card)
    sheet.open()
    return sheet


Builder.load_string(KV)
