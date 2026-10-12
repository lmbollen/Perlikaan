"""
The Kivy user interface: home, the players at the table, one game to watch
or to join, a simulation with a live leaderboard, and settings. All game
logic lives in perudo_app.core; this module draws snapshots and sends
decisions. The widgets and colours are in perudo_app.widgets and .theme.
"""

from __future__ import annotations

import random
from pathlib import Path

from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.screenmanager import Screen, ScreenManager, SlideTransition
from kivy.uix.widget import Widget

from perudo.models import Bid, Calza, Dudo
from perudo_app import theme as T
from perudo_app.core import (
    FACES, ROUTER, BotRegistry, GameSession, HumanSeat, Settings, Simulation, bar_padding, can, legal_bids,
    lowest_legal,
)
from perudo_app.speech import Speaker
from perudo_app.widgets import (  # noqa: F401  (registers the widget classes for the KV rules)
    Avatar, Body, Btn, Caption, Chip, Collapsible, Die, FaceButton, IconBtn, Meter, Segmented, Surface, TextField, Toggle,
    TopBar, TouchBox,
)

KV = """
#:import T perudo_app.theme

# ---------------------------------------------------------------- shared

<Title@Label>:
    size_hint_y: None
    height: self.texture_size[1]
    text_size: self.width, None
    halign: "left"
    bold: True
    font_size: sp(16)
    color: T.TEXT
    shorten: True

<Sub@Label>:
    markup: True
    size_hint_y: None
    height: self.texture_size[1]
    text_size: self.width, None
    halign: "left"
    font_size: sp(13)
    color: T.MUTED

<Divider@Widget>:
    size_hint_y: None
    height: dp(1)
    canvas:
        Color:
            rgba: T.LINE
        Rectangle:
            pos: self.pos
            size: self.size

<SettingRow>:
    size_hint_y: None
    height: max(dp(68), labels.height + dp(24))
    spacing: dp(14)
    BoxLayout:
        id: labels
        orientation: "vertical"
        size_hint_y: None
        height: self.minimum_height
        pos_hint: {"center_y": 0.5}
        spacing: dp(3)
        Title:
            text: root.title
            font_size: sp(15.5)
            shorten: False
        Sub:
            text: root.sub
            opacity: 1 if root.sub else 0

# ---------------------------------------------------------------- home

<ActionCard>:
    size_hint_y: None
    height: dp(86)
    padding: dp(16), dp(14)
    spacing: dp(16)
    radius: dp(20)
    base_bg: T.GOLD if self.primary else T.SURFACE
    Label:
        markup: True
        text: T.icon(root.glyph)
        font_size: sp(24)
        size_hint_x: None
        width: dp(50)
        color: T.GOLD_INK if root.primary else T.GOLD
        canvas.before:
            Color:
                rgba: (0, 0, 0, 0.12) if root.primary else T.RAISED
            RoundedRectangle:
                pos: self.x, self.center_y - dp(25)
                size: dp(50), dp(50)
                radius: [dp(15)]
    BoxLayout:
        orientation: "vertical"
        spacing: dp(2)
        Label:
            text: root.title
            bold: True
            font_size: sp(17)
            color: T.GOLD_INK if root.primary else T.TEXT
            halign: "left"
            valign: "bottom"
            text_size: self.size
        Label:
            text: root.subtitle
            font_size: sp(13.5)
            color: (0.13, 0.1, 0.04, 0.78) if root.primary else T.MUTED
            halign: "left"
            valign: "top"
            text_size: self.size
            shorten: True
    Label:
        text: "\\u203a"
        font_size: sp(30)
        size_hint_x: None
        width: dp(16)
        color: T.GOLD_INK if root.primary else T.FAINT

<HomeScreen>:
    ScrollView:
        do_scroll_x: False
        bar_width: 0
        BoxLayout:
            orientation: "vertical"
            size_hint_y: None
            height: max(self.minimum_height, root.height)
            padding: dp(20), dp(8), dp(20), dp(24)
            spacing: dp(12)
            BoxLayout:
                size_hint_y: None
                height: dp(44)
                Widget:
                IconBtn:
                    text: T.icon("\\u2699")
                    color: T.MUTED
                    on_release: app.go("settings")
            BoxLayout:
                id: hero
                size_hint_y: None
                height: dp(64)
                spacing: dp(12)
            Label:
                text: "Perudo"
                font_size: sp(46)
                bold: True
                color: T.TEXT
                size_hint_y: None
                height: self.texture_size[1]
            Label:
                text: "Liar's dice against cunning bots"
                font_size: sp(15)
                color: T.MUTED
                size_hint_y: None
                height: self.texture_size[1] + dp(14)
                valign: "top"
            ActionCard:
                primary: True
                glyph: "\\u265f"
                title: "Join the table"
                subtitle: "Play against the bots you seat"
                on_release: app.start_game(human=True)
            ActionCard:
                glyph: "\\u25c9"
                title: "Watch a game"
                subtitle: "See every cup, turn by turn"
                on_release: app.start_game(human=False)
            ActionCard:
                glyph: "\\u2694"
                title: "Simulation"
                subtitle: "Many games, live leaderboard"
                on_release: app.go("sim")
            Surface:
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16), dp(10), dp(16), dp(16)
                spacing: dp(10)
                BoxLayout:
                    size_hint_y: None
                    height: dp(40)
                    Caption:
                        text: "At the table"
                        pos_hint: {"center_y": 0.5}
                    Btn:
                        kind: "ghost"
                        text: "Edit  \\u203a"
                        color: T.GOLD
                        size_hint_x: None
                        width: dp(80)
                        font_size: sp(14)
                        on_release: app.go("bots")
                StackLayout:
                    id: chips
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)
                Sub:
                    id: table_hint
                    text: root.hint
                    opacity: 1 if root.hint else 0
                    height: self.texture_size[1] if root.hint else 0
            Widget:

# ---------------------------------------------------------------- players

<BotRow>:
    size_hint_y: None
    height: dp(96) if root.removable else dp(74)
    padding: dp(14), dp(12)
    spacing: dp(12)
    radius: dp(18)
    bg: T.SURFACE
    Avatar:
        name: root.display
        bg: root.seat if root.enabled else T.LINE
        pos_hint: {"center_y": 0.5}
    BoxLayout:
        orientation: "vertical"
        Label:
            text: root.display
            bold: True
            font_size: sp(16)
            color: T.TEXT if root.enabled else T.MUTED
            halign: "left"
            valign: "bottom"
            text_size: self.size
            shorten: True
        Label:
            text: root.origin
            font_size: sp(12.5)
            color: T.FAINT
            halign: "left"
            valign: "top" if not root.removable else "middle"
            text_size: self.size
            shorten: True
            size_hint_y: 1 if not root.removable else None
            height: dp(20)
        Btn:
            kind: "danger_ghost"
            text: "Remove"
            font_size: sp(13)
            halign: "left"
            radius: dp(8)
            size_hint: None, None
            size: (dp(64), dp(26)) if root.removable else (0, 0)
            opacity: 1 if root.removable else 0
            disabled: not root.removable
            on_release: root.ask_remove()
    BoxLayout:
        size_hint_x: None
        width: dp(108)
        spacing: dp(2)
        opacity: 1 if root.enabled else 0
        disabled: not root.enabled
        Btn:
            text: "\\u2212"
            size_hint: None, None
            size: dp(34), dp(34)
            radius: dp(17)
            font_size: sp(18)
            pos_hint: {"center_y": 0.5}
            disabled: root.copies <= 1 or not root.enabled
            on_release: root.step(-1)
        Label:
            text: "\\u00d7" + str(root.copies)
            bold: True
            font_size: sp(15)
            color: T.TEXT
        Btn:
            text: "+"
            size_hint: None, None
            size: dp(34), dp(34)
            radius: dp(17)
            font_size: sp(18)
            pos_hint: {"center_y": 0.5}
            disabled: root.copies >= 6 or not root.enabled
            on_release: root.step(1)
    Toggle:
        active: root.enabled
        pos_hint: {"center_y": 0.5}
        on_active: root.enabled = self.active

<BotsScreen>:
    BoxLayout:
        orientation: "vertical"
        TopBar:
            title: "Players"
            on_back: app.back()
            Btn:
                kind: "outline"
                text: "+  Load file"
                size_hint: None, None
                size: dp(122), dp(40)
                font_size: sp(14)
                pos_hint: {"center_y": 0.5}
                on_release: root.pick_file()
        ScrollView:
            do_scroll_x: False
            BoxLayout:
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16), dp(2), dp(16), dp(16)
                spacing: dp(10)
                Sub:
                    text: "Choose who sits at the table. Copies play as separate seats; a bot that learns keeps learning across a simulation."
                    font_size: sp(14)
                Surface:
                    id: errors_box
                    size_hint_y: None
                    height: errors.height + dp(24) if root.errors else 0
                    opacity: 1 if root.errors else 0
                    padding: dp(14), dp(12)
                    bg: T.RED[:3] + [0.14]
                    border: T.RED[:3] + [0.5]
                    Sub:
                        id: errors
                        text: root.errors
                        color: T.TEXT
                GridLayout:
                    id: rows
                    cols: 1
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(10)
        Surface:
            size_hint_y: None
            height: dp(76)
            radius: 0
            padding: dp(20), dp(14), dp(16), dp(14)
            spacing: dp(12)
            BoxLayout:
                orientation: "vertical"
                Label:
                    text: root.seats_title
                    bold: True
                    font_size: sp(16)
                    color: T.TEXT
                    halign: "left"
                    valign: "bottom"
                    text_size: self.size
                Label:
                    text: root.seats_sub
                    font_size: sp(12.5)
                    color: T.RED if root.too_few else T.MUTED
                    halign: "left"
                    valign: "top"
                    text_size: self.size
            Btn:
                kind: "primary"
                text: "Done"
                size_hint_x: None
                width: dp(116)
                on_release: app.back()

# ---------------------------------------------------------------- the game

<PlayerRow>:
    size_hint_y: None
    height: dp(54)
    padding: dp(12), dp(8), dp(14), dp(8)
    spacing: dp(12)
    radius: dp(16)
    bg: T.RAISED if root.active else T.SURFACE
    border: T.GOLD if root.active else T.CLEAR
    border_width: dp(1.5)
    opacity: 1 if root.alive else 0.45
    Avatar:
        name: root.name
        bg: root.seat
        size: dp(34), dp(34)
        ring: root.active
        pos_hint: {"center_y": 0.5}
    BoxLayout:
        orientation: "vertical"
        Label:
            text: root.name
            bold: True
            font_size: sp(15)
            color: T.TEXT
            halign: "left"
            valign: "bottom"
            text_size: self.size
            shorten: True
        Label:
            markup: True
            text: root.sub
            font_size: sp(12.5)
            color: T.MUTED
            halign: "left"
            valign: "top"
            text_size: self.size
            shorten: True
    BoxLayout:
        id: dice
        size_hint_x: None
        width: self.minimum_width
        spacing: dp(5)

<FeedLine>:
    markup: True
    size_hint_y: None
    height: self.texture_size[1]
    text_size: self.width, None
    font_size: sp(14)
    color: T.TEXT
    halign: "left"
    padding: (dp(12), dp(7)) if self.bubble else (dp(2), dp(1))
    canvas.before:
        Color:
            rgba: T.RAISED if self.bubble else T.CLEAR
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(4), dp(14), dp(14), dp(14)]

<GameScreen>:
    BoxLayout:
        orientation: "vertical"
        TopBar:
            title: root.title
            on_back: root.leave()
            Btn:
                kind: "seg_on" if root.show_dice else "outline"
                text: T.icon("\\u25c9") + "  Dice"
                size_hint: None, None
                size: (dp(92), dp(38)) if not root.human_playing else (0, 0)
                font_size: sp(14)
                pos_hint: {"center_y": 0.5}
                opacity: 0 if root.human_playing else 1
                disabled: root.human_playing
                on_release: root.show_dice = not root.show_dice
        BoxLayout:
            orientation: "vertical"
            padding: dp(14), dp(0), dp(14), dp(10)
            spacing: dp(10)
            Surface:
                size_hint_y: None
                height: dp(112)
                padding: dp(18), dp(12), dp(16), dp(12)
                spacing: dp(10)
                radius: dp(20)
                border: root.card_border
                border_width: dp(1.6)
                BoxLayout:
                    orientation: "vertical"
                    size_hint_x: 0.55
                    Caption:
                        text: root.bid_caption
                        color: root.caption_color
                    BoxLayout:
                        spacing: dp(10)
                        Label:
                            text: root.bid_qty
                            font_size: sp(44)
                            bold: True
                            color: T.TEXT
                            size_hint_x: None
                            width: self.texture_size[0]
                        Label:
                            text: "\\u00d7" if root.bid_face else ""
                            font_size: sp(24)
                            color: T.FAINT
                            size_hint_x: None
                            width: self.texture_size[0]
                        Die:
                            value: root.bid_face
                            size: dp(46), dp(46)
                            pos_hint: {"center_y": 0.5}
                            opacity: 1 if root.bid_face else 0
                        Widget:
                BoxLayout:
                    orientation: "vertical"
                    spacing: dp(2)
                    Label:
                        markup: True
                        text: root.round_text
                        font_size: sp(12.5)
                        color: T.MUTED
                        halign: "right"
                        valign: "top"
                        text_size: self.size
                    Label:
                        markup: True
                        text: root.bid_by
                        font_size: sp(15)
                        color: T.TEXT
                        halign: "right"
                        valign: "middle"
                        text_size: self.size
                        shorten: True
                    Label:
                        markup: True
                        text: root.bid_next
                        font_size: sp(13.5)
                        color: T.MUTED
                        halign: "right"
                        valign: "bottom"
                        text_size: self.size
                        shorten: True
            Surface:
                id: totals
                size_hint_y: None
                height: dp(74) if root.totals_visible else 0
                opacity: 1 if root.totals_visible else 0
                padding: dp(10), dp(8)
                spacing: dp(2)
                radius: dp(18)
            ScrollView:
                size_hint_y: None
                height: min(players.minimum_height, root.height * (0.3 if root.human_playing else 0.4))
                do_scroll_x: False
                bar_width: dp(3)
                GridLayout:
                    id: players
                    cols: 1
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(6)
            Surface:
                orientation: "vertical"
                padding: dp(14), dp(10), dp(10), dp(10)
                radius: dp(18)
                ScrollView:
                    id: feed_scroll
                    do_scroll_x: False
                    bar_width: dp(3)
                    BoxLayout:
                        id: feed
                        orientation: "vertical"
                        size_hint_y: None
                        height: self.minimum_height
                        spacing: dp(7)
                        padding: 0, 0, dp(6), 0
        Collapsible:
            shown: not root.human_playing
            full_height: dp(74)
            bg: T.CLEAR
            radius: 0
            padding: dp(14), dp(2), dp(14), dp(14)
            spacing: dp(10)
            Btn:
                kind: "outline"
                text: T.icon("\\u21bb") + "  New"
                size_hint_x: None
                width: dp(96)
                on_release: root.restart()
            Btn:
                text: T.icon("\\u25b8") + "  Step"
                disabled: root.auto or root.finished
                on_release: root.next_turn()
            Btn:
                kind: "primary"
                text: (T.icon("\\u275a\\u275a") + "  Pause") if root.auto else (T.icon("\\u25b6") + "  Play")
                disabled: root.finished
                on_release: root.toggle_auto()
        Collapsible:
            orientation: "vertical"
            shown: root.human_playing
            full_height: dp(218)
            padding: dp(16), dp(12), dp(16), dp(16)
            spacing: dp(12)
            radius: dp(24)
            bg: T.SURFACE
            border: T.GOLD if root.my_turn else T.CLEAR
            border_width: dp(1.6)
            BoxLayout:
                size_hint_y: None
                height: dp(18)
                Caption:
                    text: "Your move"
                Label:
                    markup: True
                    text: root.turn_hint
                    font_size: sp(13)
                    color: T.GOLD if root.my_turn else T.MUTED
                    halign: "right"
                    text_size: self.size
                    valign: "middle"
                    shorten: True
            BoxLayout:
                id: faces
                size_hint_y: None
                height: dp(44)
                spacing: dp(10)
            BoxLayout:
                size_hint_y: None
                height: dp(50)
                spacing: dp(8)
                Btn:
                    text: "\\u2212"
                    font_size: sp(22)
                    size_hint_x: None
                    width: dp(48)
                    disabled: not root.my_turn
                    on_release: root.change_quantity(-1)
                Label:
                    text: root.qty_text
                    bold: True
                    font_size: sp(26)
                    color: T.TEXT if root.my_turn else T.FAINT
                    size_hint_x: None
                    width: dp(44)
                Btn:
                    text: "+"
                    font_size: sp(22)
                    size_hint_x: None
                    width: dp(48)
                    disabled: not root.my_turn
                    on_release: root.change_quantity(1)
                Btn:
                    kind: "primary"
                    text: root.bid_button
                    font_size: sp(16)
                    disabled: not root.my_turn
                    on_release: root.submit_bid()
            BoxLayout:
                size_hint_y: None
                height: dp(48)
                spacing: dp(10)
                Btn:
                    kind: "danger"
                    text: "Dudo!"
                    font_size: sp(16)
                    disabled: not root.can_dudo
                    on_release: root.submit(root.DUDO)
                Btn:
                    kind: "calza"
                    text: "Calza!"
                    font_size: sp(16)
                    disabled: not root.can_calza
                    on_release: root.submit(root.CALZA)

# ---------------------------------------------------------------- simulation

<LeaderRow>:
    orientation: "vertical"
    size_hint_y: None
    height: dp(64)
    spacing: dp(7)
    BoxLayout:
        spacing: dp(10)
        Label:
            markup: True
            text: root.rank_text
            bold: True
            font_size: sp(16)
            color: T.FAINT
            size_hint_x: None
            width: dp(24)
        Avatar:
            name: root.name
            bg: root.seat
            size: dp(30), dp(30)
            pos_hint: {"center_y": 0.5}
        BoxLayout:
            orientation: "vertical"
            Label:
                text: root.name
                bold: True
                font_size: sp(15)
                color: T.TEXT
                halign: "left"
                valign: "bottom"
                text_size: self.size
                shorten: True
            Label:
                text: root.sub
                font_size: sp(12.5)
                color: T.MUTED
                halign: "left"
                valign: "top"
                text_size: self.size
                shorten: True
        Label:
            text: root.pct_text
            bold: True
            font_size: sp(18)
            color: T.TEXT
            size_hint_x: None
            width: dp(70)
            halign: "right"
            text_size: self.size
            valign: "middle"
    Meter:
        id: meter
        size_hint_y: None
        height: dp(8)
        color: root.seat

<SimScreen>:
    BoxLayout:
        orientation: "vertical"
        TopBar:
            title: "Simulation"
            on_back: root.leave()
        BoxLayout:
            orientation: "vertical"
            padding: dp(16), dp(2), dp(16), dp(16)
            spacing: dp(12)
            Surface:
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16), dp(14), dp(16), dp(16)
                spacing: dp(12)
                BoxLayout:
                    size_hint_y: None
                    height: dp(22)
                    Caption:
                        text: "Games"
                        pos_hint: {"center_y": 0.5}
                    Label:
                        text: root.seats_text
                        font_size: sp(13)
                        color: T.MUTED
                        halign: "right"
                        valign: "middle"
                        text_size: self.size
                        shorten: True
                Segmented:
                    id: games
                    options: [(100, "100"), (500, "500"), (1000, "1000"), (5000, "5000")]
                    value: root.games_value
                    on_value: root.games_value = self.value
                    disabled: root.running
                Btn:
                    kind: "outline" if root.running else "primary"
                    text: root.run_text
                    size_hint_y: None
                    height: dp(52)
                    font_size: sp(16)
                    on_release: root.toggle()
                Meter:
                    id: progress
                    size_hint_y: None
                    height: dp(6)
                    color: T.GOLD
                Label:
                    markup: True
                    text: root.progress
                    font_size: sp(13)
                    color: T.MUTED
                    size_hint_y: None
                    height: self.texture_size[1]
                    text_size: self.width, None
                    halign: "left"
            Surface:
                orientation: "vertical"
                padding: dp(16), dp(14), dp(16), dp(10)
                spacing: dp(12)
                BoxLayout:
                    size_hint_y: None
                    height: dp(18)
                    Caption:
                        text: "Leaderboard"
                    Caption:
                        text: "Win rate"
                        halign: "right"
                Sub:
                    text: "Press Start to pit the seated bots against each other. Bars show each bot's share of the games won, updated as they come in."
                    font_size: sp(14)
                    opacity: 1 if root.empty else 0
                    height: self.texture_size[1] if root.empty else 0
                ScrollView:
                    do_scroll_x: False
                    bar_width: dp(3)
                    GridLayout:
                        id: board
                        cols: 1
                        size_hint_y: None
                        height: self.minimum_height
                        spacing: dp(12)
                        padding: 0, 0, 0, dp(6)

# ---------------------------------------------------------------- settings

<SettingsScreen>:
    BoxLayout:
        orientation: "vertical"
        TopBar:
            title: "Settings"
            on_back: app.back()
        ScrollView:
            do_scroll_x: False
            BoxLayout:
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16), dp(4), dp(16), dp(28)
                spacing: dp(10)
                Caption:
                    text: "Speech"
                Surface:
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
                    padding: dp(16), dp(2)
                    SettingRow:
                        title: "Read the table talk aloud"
                        sub: "Every bot speaks with its own voice"
                        Toggle:
                            id: tts_toggle
                            active: app.settings["tts"]
                            pos_hint: {"center_y": 0.5}
                            on_active: app.set_setting("tts", self.active)
                    Divider:
                    SettingRow:
                        title: "Voices"
                        sub: "The phone's own voices work offline. Microsoft's online voices sound better, but the service refuses them at the moment; the phone's voices then take over."
                    Segmented:
                        options: [("system", "Phone"), ("edge", "Microsoft")]
                        value: app.settings["tts_engine"]
                        on_value: app.set_setting("tts_engine", self.value)
                    Widget:
                        size_hint_y: None
                        height: dp(14)
                    Divider:
                    SettingRow:
                        title: "Voice language"
                        sub: "Every bot keeps its own voice"
                        Segmented:
                            size_hint_x: None
                            width: dp(120)
                            pos_hint: {"center_y": 0.5}
                            options: [("nl", "NL"), ("en", "EN")]
                            value: app.settings["tts_lang"]
                            on_value: app.set_setting("tts_lang", self.value)
                    Divider:
                    SettingRow:
                        title: "Try it"
                        sub: "Plays a line in a bot's voice"
                        Btn:
                            kind: "outline"
                            text: T.icon("\u25b6") + "  Test"
                            font_size: sp(14)
                            size_hint: None, None
                            size: dp(96), dp(40)
                            pos_hint: {"center_y": 0.5}
                            on_release: app.test_voice()
                Sub:
                    text: root.speech_status
                    color: root.speech_color
                    opacity: 1 if root.speech_status else 0
                    height: self.texture_size[1] if root.speech_status else 0
                Widget:
                    size_hint_y: None
                    height: dp(6)
                Caption:
                    text: "Watching"
                Surface:
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
                    padding: dp(16), dp(2), dp(16), dp(16)
                    SettingRow:
                        title: "Show every player's dice"
                        sub: "Only while watching; at the table you see your own"
                        Toggle:
                            active: app.settings["show_dice"]
                            pos_hint: {"center_y": 0.5}
                            on_active: app.set_setting("show_dice", self.active)
                    Divider:
                    SettingRow:
                        title: "Auto-play speed"
                        sub: root.speed_text
                    Segmented:
                        id: speed
                        options: [(0.3, "Fast"), (1.0, "Normal"), (2.0, "Slow")]
                        value: app.settings["step_delay"]
                        on_value: root.set_speed(self.value)
                Widget:
                    size_hint_y: None
                    height: dp(6)
                Caption:
                    text: "At the table"
                Surface:
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
                    padding: dp(16), dp(14), dp(16), dp(16)
                    spacing: dp(10)
                    Title:
                        text: "Your name"
                        font_size: sp(15.5)
                    TextField:
                        text: app.settings["human_name"]
                        hint_text: "You"
                        on_text_validate: app.set_setting("human_name", self.text.strip() or "You")
                        on_focus: if not args[1]: app.set_setting("human_name", self.text.strip() or "You")
                Widget:
                    size_hint_y: None
                    height: dp(10)
                Sub:
                    text: "Perudo \\u00b7 the bots of the Perlikaan bot battle. Pacos (ones) are wild except in Palifico rounds."
                    color: T.FAINT
                    halign: "center"
"""


class SettingRow(BoxLayout):
    title = StringProperty("")
    sub = StringProperty("")


# ---------------------------------------------------------------------------
# home
# ---------------------------------------------------------------------------

class ActionCard(TouchBox):
    primary = BooleanProperty(False)
    glyph = StringProperty("")
    title = StringProperty("")
    subtitle = StringProperty("")


def seat_plan(entries):
    """(entry, first seat index) for the selected entries, in seating order."""
    plan, seat = [], 0
    for e in entries:
        plan.append((e, seat))
        seat += e.copies
    return plan, seat


class HomeScreen(Screen):
    hint = StringProperty("")

    def on_kv_post(self, *_):
        self._dice = [Die(value=v, size=(dp(44), dp(44)), pos_hint={"center_y": 0.5}) for v in (1, 5, 5, 3, 1)]
        self.ids.hero.add_widget(Label(size_hint_x=1))
        for d in self._dice:
            self.ids.hero.add_widget(d)
        self.ids.hero.add_widget(Label(size_hint_x=1))

    def on_pre_enter(self, *_):
        app = App.get_running_app()
        plan, seats = seat_plan(app.registry.selected())
        chips = self.ids.chips
        chips.clear_widgets()
        for entry, seat in plan:
            dot = T.color(T.icon("●"), T.seat_hex(seat))
            copies = f"  [color={T.HEX['muted']}]×{entry.copies}[/color]" if entry.copies > 1 else ""
            chips.add_widget(Chip(text=f"{dot}  {T.esc(entry.display_name)}{copies}"))
        if seats == 0:
            self.hint = "Nobody is seated yet. Tap Edit to choose the bots."
        elif seats == 1:
            self.hint = "One bot: enough to play against, not to watch. Seat another for a bot game."
        else:
            self.hint = ""
        self.roll()

    def roll(self):
        for i, d in enumerate(self._dice):
            d.value = random.randint(1, 6)
            final = random.uniform(-12, 12)
            d.angle = final + random.choice((-1, 1)) * random.uniform(90, 200)
            Animation.cancel_all(d)
            Animation(angle=final, d=0.55 + 0.08 * i, t="out_back").start(d)


# ---------------------------------------------------------------------------
# players
# ---------------------------------------------------------------------------

def _heading(text: str) -> Label:
    """A dialog title that wraps to the card's width."""
    label = Label(text=text, markup=True, bold=True, font_size=sp(19), color=T.TEXT, size_hint_y=None,
                  halign="left", valign="middle")
    label.bind(width=lambda w, v: setattr(w, "text_size", (v, None)),
               texture_size=lambda w, v: setattr(w, "height", max(dp(30), v[1])))
    return label


class BotRow(Surface):
    display = StringProperty("")
    origin = StringProperty("")
    enabled = BooleanProperty(True)
    copies = NumericProperty(1)
    seat = ListProperty(T.PLAYER_COLORS[0])
    removable = BooleanProperty(False)

    def __init__(self, entry, on_change, on_remove=None, **kw):
        self.entry = entry
        self._on_change = on_change
        self._on_remove = on_remove
        super().__init__(**kw)
        self.display = entry.display_name
        self.removable = entry.source != "bundled"
        self.origin = "From the repository" if entry.source == "bundled" else f"Imported \u00b7 {Path(entry.source).name}"
        self.enabled = entry.enabled
        self.copies = entry.copies

    def ask_remove(self):
        if self._on_remove is not None:
            self._on_remove(self.entry)

    def on_enabled(self, *_):
        self.entry.enabled = self.enabled
        self._on_change()

    def step(self, delta):
        self.copies = max(1, min(6, self.copies + delta))
        self.entry.copies = self.copies
        self._on_change()


class BotsScreen(Screen):
    errors = StringProperty("")
    seats_title = StringProperty("")
    seats_sub = StringProperty("")
    too_few = BooleanProperty(False)

    def on_pre_enter(self, *_):
        App.get_running_app().registry.scan_user()      # picks up files copied into the folder by hand
        self.refresh()

    def refresh(self):
        app = App.get_running_app()
        rows = self.ids.rows
        rows.clear_widgets()
        self._rows = []
        bundled = [e for e in app.registry.entries if e.source == "bundled"]
        imported = [e for e in app.registry.entries if e.source != "bundled"]
        rows.add_widget(Caption(text="From the repository"))
        for entry in bundled:
            self._add_row(entry)
        rows.add_widget(Widget(size_hint_y=None, height=dp(4)))
        rows.add_widget(Caption(text="Imported"))
        folder = T.esc(str(app.registry.user_dir)) if app.registry.user_dir else ""
        note = ("Bots you load are copied into the app and stay after a restart. "
                f"Files copied into this folder are picked up too:\n[color={T.HEX['muted']}]{folder}[/color]")
        if not imported:
            note = "None yet. Tap Load file to add a bot. " + note
        rows.add_widget(Body(text=note, color=T.FAINT, font_size=sp(13)))
        for entry in imported:
            self._add_row(entry)
        self.errors = "\n".join(app.registry.errors[-3:])
        self.update()

    def _add_row(self, entry):
        row = BotRow(entry, self.update, self.confirm_remove)
        self._rows.append(row)
        self.ids.rows.add_widget(row)

    def confirm_remove(self, entry):
        app = App.get_running_app()
        sheet = ModalView(size_hint=(0.88, None), height=dp(210), background="", background_color=T.CLEAR,
                          overlay_color=T.BG[:3] + [0.75])
        card = Surface(orientation="vertical", padding=dp(20), spacing=dp(10), radius=dp(22))
        card.add_widget(_heading(f"Remove {T.esc(entry.display_name)}?"))
        others = [e.display_name for e in app.registry.entries if e.source == entry.source and e is not entry]
        what = f"{Path(entry.source).name} is deleted from the app's bots folder"
        if others:
            what += f", with {', '.join(others)}"
        card.add_widget(Body(text=what + ". The file you loaded it from is not touched."))
        buttons = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(10))
        cancel = Btn(text="Cancel", kind="outline")
        cancel.bind(on_release=lambda _b: sheet.dismiss())
        remove = Btn(text="Remove", kind="danger")

        def do_remove(_b):
            app.registry.remove(entry)
            sheet.dismiss()
            self.refresh()
            app.toast(f"Removed {T.esc(entry.display_name)}")

        remove.bind(on_release=do_remove)
        buttons.add_widget(cancel)
        buttons.add_widget(remove)
        card.add_widget(Widget())
        card.add_widget(buttons)
        sheet.add_widget(card)
        sheet.open()

    def update(self):
        app = App.get_running_app()
        plan, seats = seat_plan(app.registry.selected())
        first = {id(e): s for e, s in plan}
        for row in getattr(self, "_rows", []):
            if id(row.entry) in first:
                row.seat = T.seat_color(first[id(row.entry)])
        app.save_table()
        self.seats_title = f"{seats} seat{'s' if seats != 1 else ''}"
        self.too_few = seats < 2
        self.seats_sub = ("Seat at least two bots to watch or simulate" if seats < 2
                          else f"{len(plan)} different bot{'s' if len(plan) != 1 else ''}")

    def pick_file(self):
        app = App.get_running_app()
        chooser = FileChooserListView(path=app.bots_dir, filters=["*.py"])
        box = BoxLayout(orientation="vertical", spacing=dp(10), padding=(0, dp(8), 0, 0))
        box.add_widget(Body(text="A single .py file with a bot class (a class with a play() method).",
                            color=T.MUTED))
        box.add_widget(chooser)
        buttons = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(10))
        popup = ModalView(size_hint=(0.94, 0.86), background="", background_color=T.CLEAR,
                          overlay_color=T.BG[:3] + [0.75])
        card = Surface(orientation="vertical", padding=dp(16), spacing=dp(6))
        card.add_widget(_heading("Load a bot"))
        card.add_widget(box)
        popup.add_widget(card)

        def load(_):
            if chooser.selection:
                added = app.registry.import_file(chooser.selection[0])
                self.refresh()
                if added:
                    app.toast(f"Added {T.esc(', '.join(e.display_name for e in added))}, kept in the app")
                else:
                    app.toast("No bot found in that file")
            popup.dismiss()

        cancel = Btn(text="Cancel", kind="outline")
        cancel.bind(on_release=lambda _: popup.dismiss())
        ok = Btn(text="Load", kind="primary")
        ok.bind(on_release=load)
        buttons.add_widget(cancel)
        buttons.add_widget(ok)
        box.add_widget(buttons)
        popup.open()


# ---------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------

class PlayerRow(Surface):
    name = StringProperty("")
    sub = StringProperty("")
    seat = ListProperty(T.PLAYER_COLORS[0])
    active = BooleanProperty(False)
    alive = BooleanProperty(True)

    def set_dice(self, values, highlight, dim):
        box = self.ids.dice
        if len(box.children) != len(values):
            box.clear_widgets()
            for _ in values:
                box.add_widget(Die(size=(dp(25), dp(25)), pos_hint={"center_y": 0.5}, shadow=False))
        for die, v, h, d in zip(reversed(box.children), values, highlight, dim):
            die.value, die.highlight, die.dim = v, h, d


class FeedLine(Label):
    bubble = BooleanProperty(False)


def _counts(face, wild):
    """Does a die count towards a bid on `face`?"""
    return lambda d: d == face or (wild and face != 1 and d == 1)


class GameScreen(Screen):
    DUDO = Dudo()
    CALZA = Calza()
    FEED_KEEP = 60

    title = StringProperty("Game")
    show_dice = BooleanProperty(True)
    human_playing = BooleanProperty(False)
    auto = BooleanProperty(True)
    finished = BooleanProperty(False)
    session = ObjectProperty(None, allownone=True)
    # the bid card
    bid_caption = StringProperty("")
    caption_color = ListProperty(T.MUTED)
    bid_qty = StringProperty("")
    bid_face = NumericProperty(0)
    bid_by = StringProperty("")
    bid_next = StringProperty("")
    round_text = StringProperty("")
    card_border = ListProperty(T.CLEAR)
    totals_visible = BooleanProperty(False)
    # the human's composer
    my_turn = BooleanProperty(False)
    can_dudo = BooleanProperty(False)
    can_calza = BooleanProperty(False)
    qty_text = StringProperty("")
    bid_button = StringProperty("Bid")
    turn_hint = StringProperty("")

    def __init__(self, **kw):
        super().__init__(**kw)
        self._face = 2
        self._quantity = 1
        self._poll = None
        self._pending = None
        self._sheet = None

    def on_kv_post(self, *_):
        self._faces = []
        for f in FACES:
            b = FaceButton(value=f, size=(dp(40), dp(40)), pos_hint={"center_y": 0.5})
            b.bind(on_release=lambda _b, face=f: self._pick_face(face))
            self._faces.append(b)
        self.ids.faces.add_widget(Label())
        for b in self._faces:
            self.ids.faces.add_widget(b)
        self.ids.faces.add_widget(Label())
        self._totals = []
        for f in FACES:
            col = BoxLayout(orientation="vertical", spacing=dp(4))
            die = Die(value=f, size=(dp(24), dp(24)), pos_hint={"center_x": 0.5}, shadow=False)
            count = Label(markup=True, bold=True, font_size=sp(16), color=T.TEXT)
            col.add_widget(die)
            col.add_widget(count)
            self.ids.totals.add_widget(col)
            self._totals.append((die, count))

    # -- lifecycle -----------------------------------------------------------

    def begin(self, session: GameSession, human: bool):
        app = App.get_running_app()
        self.session = session
        self.human_playing = human
        self.title = "At the table" if human else "Watching"
        self.show_dice = (not human) and app.settings["show_dice"]
        self.finished = False
        self._pending = None
        self._seat = {b.name: i for i, b in enumerate(session.bots)}
        self._human = session.human.name if session.human is not None else None
        self._rows = {}
        self.ids.players.clear_widgets()
        self.ids.feed.clear_widgets()
        self._feed_last = None
        self._close_sheet()
        self._sheet_shown = False
        # Watching starts paused (Play or Step starts it); at the table the bots play by themselves.
        self.auto = human
        session.set_auto(human, app.settings["step_delay"])
        session.start()
        if self._poll is None:
            self._poll = Clock.schedule_interval(self.refresh, 0.12)
        self.refresh()

    def leave(self):
        if self.session is not None:
            self.session.stop()
        if self._poll is not None:
            self._poll.cancel()
            self._poll = None
        self._close_sheet()
        app = App.get_running_app()
        app.speaker.clear()
        app.go("home", back=True)

    def restart(self):
        if self.session is not None:
            self.session.stop()
        App.get_running_app().start_game(human=self.human_playing)

    def next_turn(self):
        if self.session is not None:
            self.session.next()

    def toggle_auto(self):
        self.auto = not self.auto
        if self.session is not None:
            self.session.set_auto(self.auto, App.get_running_app().settings["step_delay"])

    def set_auto(self, auto: bool):
        if self.auto != auto:
            self.toggle_auto()

    # -- names and colours ---------------------------------------------------

    def _hex(self, name):
        return T.seat_hex(self._seat.get(name, 0))

    def _who(self, name, bold=True):
        text = T.color(T.esc(name), self._hex(name))
        return f"[b]{text}[/b]" if bold else text

    # -- drawing a snapshot --------------------------------------------------

    def refresh(self, _dt=0):
        if self.session is None:
            return
        snap = self.session.snapshot()
        self.finished = snap.finished
        self._sync_players(snap)
        self._draw_bid_card(snap)
        self._draw_feed(snap)
        self._draw_composer(snap)
        if snap.finished and not self._sheet_shown and not snap.error and snap.standings:
            self._sheet_shown = True
            Clock.schedule_once(lambda _dt: self._show_sheet(snap), 0.9)

    def _sync_players(self, snap):
        grid = self.ids.players
        if [p.name for p in snap.players] != list(self._rows):
            grid.clear_widgets()
            self._rows = {}
            for p in snap.players:
                row = PlayerRow(name=p.name, seat=T.seat_color(self._seat.get(p.name, len(self._rows))))
                self._rows[p.name] = row
                grid.add_widget(row)
        reveal = snap.reveal
        revealed = dict(reveal["dice"]) if reveal else {}
        show_all = self.show_dice and not self.human_playing
        if reveal:
            counts = _counts(reveal["bid"].face, not reveal["palifico"])
        elif snap.current_bid is not None:
            counts = _counts(snap.current_bid.face, not snap.palifico)
        else:
            counts = None
        table = []
        for p in snap.players:
            row = self._rows[p.name]
            row.alive = p.alive
            row.active = p.name == snap.to_act and not snap.finished
            if reveal and p.name in revealed:
                dice = tuple(revealed[p.name])
                table.extend(dice)
                hl = [bool(counts and counts(d)) for d in dice]
                row.set_dice(dice, hl, [not h for h in hl])
            elif p.alive and (show_all or p.is_human):
                dice = tuple(p.dice)
                table.extend(dice)
                hl = [bool(counts and counts(d)) for d in dice]
                row.set_dice(dice, hl, [False] * len(dice))
            else:
                row.set_dice((0,) * (p.dice_count if p.alive else 0), [False] * p.dice_count, [False] * p.dice_count)
            if not p.alive:
                row.sub = "out of the game"
            elif row.active and p.is_human:
                row.sub = T.color("your turn", T.HEX["gold"])
            elif row.active and reveal is None:
                row.sub = T.color("thinking…" if (self.auto or self.human_playing) else "to play",
                                  T.HEX["gold"])
            elif reveal and p.name == reveal.get("loser"):
                row.sub = T.color("lost a die", T.HEX["red"])
            else:
                row.sub = f"{p.dice_count} {'die' if p.dice_count == 1 else 'dice'}" + ("  ·  you" if p.is_human else "")
        # totals per face, for watchers with the dice shown (and during a reveal)
        self.totals_visible = bool(show_all or (reveal and not self.human_playing and self.show_dice))
        if self.totals_visible:
            wild = not (reveal["palifico"] if reveal else snap.palifico)
            face = reveal["bid"].face if reveal else (snap.current_bid.face if snap.current_bid else None)
            ones = sum(1 for d in table if d == 1)
            for f, (die, count) in zip(FACES, self._totals):
                natural = sum(1 for d in table if d == f)
                total = natural + (ones if wild and f != 1 else 0)
                extra = (f" [size={int(sp(11))}][color={T.HEX['faint']}]{natural}[/color][/size]"
                         if total != natural else "")
                count.text = f"{total}{extra}"
                count.color = T.GOLD if f == face else T.TEXT
                die.highlight = f == face

    def _draw_bid_card(self, snap):
        reveal = snap.reveal
        self.round_text = f"Round {snap.round_number}" if snap.round_number else ""
        if snap.palifico:
            self.round_text += "  ·  " + T.color("[b]PALIFICO[/b]", T.HEX["gold"])
        mine = snap.pending is not None
        if snap.error:
            self.bid_caption, self.caption_color = "A bot crashed", T.RED
            self.bid_qty, self.bid_face = "!", 0
            self.bid_by, self.bid_next = "", T.esc(snap.error)
            self.card_border = T.RED
        elif snap.finished and snap.standings:
            self.bid_caption, self.caption_color = "Game over", T.GOLD
            self.bid_qty, self.bid_face = "", 0
            self.bid_by = f"{T.color(T.icon(chr(0x2605)), T.HEX['gold'])} {self._who(snap.standings[0])} wins"
            self.bid_next = f"{len(snap.standings)} players"
            self.card_border = T.GOLD
        elif snap.finished:
            self.bid_caption, self.caption_color = "Stopped", T.MUTED
            self.bid_qty, self.bid_face, self.bid_by, self.bid_next = "", 0, "", ""
            self.card_border = T.CLEAR
        elif reveal:
            dudo = reveal["type"] == "dudo"
            tone = T.RED if dudo else T.TEAL
            self.bid_caption = f"{'Dudo' if dudo else 'Calza'}  ·  {reveal['challenger']}"
            self.caption_color = tone
            self.bid_qty, self.bid_face = str(reveal["bid"].quantity), reveal["bid"].face
            actual = reveal["actual"]
            self.bid_by = f"[b]{actual}[/b] on the table"
            if reveal.get("loser"):
                self.bid_next = f"{self._who(reveal['loser'])} loses a die"
            else:
                self.bid_next = T.color("Exact! ", T.HEX["teal"]) + self._who(reveal["challenger"]) + " is right"
            self.card_border = tone
        elif snap.current_bid is not None:
            self.bid_caption, self.caption_color = "Current bid", T.MUTED
            self.bid_qty, self.bid_face = str(snap.current_bid.quantity), snap.current_bid.face
            self.bid_by = f"by {self._who(snap.bidder)}" if snap.bidder else ""
            self.bid_next = (T.color("Your turn", T.HEX["gold"]) if mine
                             else f"{self._who(snap.to_act, bold=False)} to play" if snap.to_act else "")
            self.card_border = T.GOLD if mine else T.CLEAR
        else:
            self.bid_caption, self.caption_color = "Opening bid", T.MUTED
            self.bid_qty, self.bid_face = "—", 0
            self.bid_by = (T.color("You open", T.HEX["gold"]) if mine
                           else f"{self._who(snap.to_act)} opens" if snap.to_act else "")
            self.bid_next = "Pacos are wild" if not snap.palifico else "Pacos are not wild"
            if not self.auto and not self.human_playing and snap.round_number == 1:
                self.bid_next = T.color("Press Play or Step to start", T.HEX["gold"])
            self.card_border = T.GOLD if mine else T.CLEAR

    def _feed_markup(self, item):
        who = self._who(item.player) if item.player else ""
        if item.kind == "round":
            opener = "you open" if item.player == self._human else f"{T.esc(item.player)} opens"
            return f"[color={T.HEX['faint']}][b]{T.esc(item.text.upper())}[/b]  ·  {opener}[/color]", False, "center"
        if item.kind == "bid":
            return f"{who}  bids  [b]{T.bid_text(item.bid.quantity, item.bid.face)}[/b]", False, "left"
        if item.kind in ("dudo", "calza"):
            tone = T.HEX["red"] if item.kind == "dudo" else T.HEX["teal"]
            call = T.color(f"[b]{item.kind.capitalize()}![/b]", tone)
            return f"{who}  {call}  on {T.esc(item.other)}'s {T.bid_text(item.bid.quantity, item.bid.face)}", False, "left"
        if item.kind == "result":
            tone = T.HEX["teal"] if item.good else T.HEX["red"]
            head = T.color(f"[b]{item.other} on the table[/b]", tone)
            text = T.esc(item.text)
            for name in sorted(self._seat, key=len, reverse=True):     # "Sniper 2" before "Sniper"
                if text.startswith(T.esc(name) + " "):
                    text = self._who(name) + text[len(T.esc(name)):]
                    break
            return f"{T.icon(chr(0x2192))}  {head}  ·  {text}", False, "left"
        if item.kind == "gain":
            return f"{who}  {T.color(T.esc(item.text), T.HEX['teal'])}", False, "left"
        if item.kind == "out":
            return f"{who}  {T.color('is out', T.HEX['muted'])}", False, "left"
        if item.kind == "win":
            return f"{T.color(T.icon(chr(0x2605)), T.HEX['gold'])}  {who}  {T.color('wins the game', T.HEX['gold'])}", False, "left"
        if item.kind == "talk":
            return f"{self._who(item.player)}\n[i]{T.esc(item.text)}[/i]", True, "left"
        return T.esc(item.text), False, "left"

    def _draw_feed(self, snap):
        feed = snap.feed
        if not feed or feed[-1] is self._feed_last:
            return
        start = 0
        if self._feed_last is not None:
            for i in range(len(feed) - 1, -1, -1):
                if feed[i] is self._feed_last:
                    start = i + 1
                    break
        self._feed_last = feed[-1]
        box, scroll = self.ids.feed, self.ids.feed_scroll
        at_bottom = scroll.scroll_y <= 0.02 or box.height <= scroll.height
        for item in feed[start:][-self.FEED_KEEP:]:
            text, bubble, halign = self._feed_markup(item)
            line = FeedLine(text=text, bubble=bubble, halign=halign)
            if item.kind == "talk":
                line.font_size = sp(13.5)
                line.color = T.MUTED
            box.add_widget(line)
        while len(box.children) > self.FEED_KEEP:
            box.remove_widget(box.children[-1])
        if at_bottom:
            Clock.schedule_once(lambda _dt: setattr(scroll, "scroll_y", 0), 0)

    # -- the human's decision -----------------------------------------------

    def _draw_composer(self, snap):
        if not self.human_playing:
            return
        pending = snap.pending
        if pending is not None and self._pending is None:
            self._setup_choice(pending)
        self._pending = pending
        self.my_turn = pending is not None
        if pending is None:
            self.can_dudo = self.can_calza = False
            for b in self._faces:
                b.disabled, b.selected = True, False
            if snap.finished:
                self.turn_hint = "Game over"
            elif snap.reveal:
                self.turn_hint = "Cups are up"
            elif snap.to_act:
                self.turn_hint = f"{T.esc(snap.to_act)} is thinking…"
            else:
                self.turn_hint = ""
        else:
            self.turn_hint = "Your turn"

    def _setup_choice(self, state):
        legal_faces = sorted({b.face for b in legal_bids(state)})
        for b in self._faces:
            b.disabled = b.value not in legal_faces
        current = state.current_bid
        first = current.face if current and current.face in legal_faces else (legal_faces[0] if legal_faces else 2)
        self.can_dudo = can(state, Dudo)
        self.can_calza = can(state, Calza)
        self._pick_face(first, state)

    def _pick_face(self, face: int, state=None):
        state = state or (self.session.snapshot().pending if self.session else None)
        if state is None:
            return
        self._face = face
        for b in self._faces:
            b.selected = b.value == face
        low = lowest_legal(state, face)
        self._quantity = low.quantity if low else 1
        self._show_bid()

    def change_quantity(self, delta: int):
        state = self.session.snapshot().pending if self.session else None
        if state is None:
            return
        legal = {b.quantity for b in legal_bids(state) if b.face == self._face}
        q = self._quantity + delta
        while 1 <= q <= max(legal, default=0) and q not in legal:
            q += delta
        if q in legal:
            self._quantity = q
        self._show_bid()

    def _show_bid(self):
        self.qty_text = str(self._quantity)
        self.bid_button = f"Bid  {T.bid_text(self._quantity, self._face)}"

    def submit_bid(self):
        self.submit(Bid(self._quantity, self._face))

    def submit(self, action):
        state = self.session.snapshot().pending if self.session else None
        if state is None or action not in state.legal_actions:
            return
        self._pending = None
        self.my_turn = False
        self.session.submit(action)

    # -- the end of the game -------------------------------------------------

    def _show_sheet(self, snap):
        if self.session is None or self.manager is None or self.manager.current != self.name:
            return
        human = next((p.name for p in snap.players if p.is_human), None)
        standings = list(snap.standings)
        if human is not None:
            place = standings.index(human) + 1 if human in standings else None
            title = "You win!" if place == 1 else f"You finished {_ordinal(place)}" if place else "Game over"
        else:
            title = f"{standings[0]} wins"
        sheet = ModalView(size_hint=(0.88, None), height=dp(150 + 46 * min(len(standings), 7)),
                          background="", background_color=T.CLEAR, overlay_color=T.BG[:3] + [0.75],
                          auto_dismiss=True)
        card = Surface(orientation="vertical", padding=(dp(20), dp(18), dp(20), dp(18)), spacing=dp(8),
                       radius=dp(24), border=T.GOLD, border_width=dp(1.5))
        card.add_widget(Label(text=T.color(T.icon("★"), T.HEX["gold"]) + "  " + T.esc(title), markup=True,
                              bold=True, font_size=sp(21), color=T.TEXT, size_hint_y=None, height=dp(40)))
        for i, name in enumerate(standings[:7]):
            row = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(12))
            row.add_widget(Label(text=str(i + 1), bold=True, font_size=sp(15), size_hint_x=None, width=dp(20),
                                 color=T.GOLD if i == 0 else T.FAINT))
            row.add_widget(Avatar(name=name, bg=T.seat_color(self._seat.get(name, i)), size=(dp(28), dp(28)),
                                  pos_hint={"center_y": 0.5}))
            row.add_widget(Label(text=name + ("  (you)" if name == human else ""), font_size=sp(15),
                                 color=T.TEXT if i == 0 else T.MUTED, bold=i == 0, halign="left",
                                 valign="middle", text_size=(dp(220), None)))
            card.add_widget(row)
        buttons = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(10), padding=(0, dp(8), 0, 0))
        home = Btn(text="Home", kind="outline")
        home.bind(on_release=lambda _b: self.leave())
        again = Btn(text="Play again", kind="primary")
        again.bind(on_release=lambda _b: (self._close_sheet(), self.restart()))
        buttons.add_widget(home)
        buttons.add_widget(again)
        card.add_widget(buttons)
        sheet.add_widget(card)
        self._sheet = sheet
        sheet.open()

    def _close_sheet(self):
        if self._sheet is not None:
            self._sheet.dismiss()
            self._sheet = None


def _ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

class LeaderRow(BoxLayout):
    name = StringProperty("")
    sub = StringProperty("")
    rank_text = StringProperty("")
    pct_text = StringProperty("")
    seat = ListProperty(T.PLAYER_COLORS[0])


class SimScreen(Screen):
    progress = StringProperty("")
    run_text = StringProperty("Start")
    seats_text = StringProperty("")
    running = BooleanProperty(False)
    empty = BooleanProperty(True)
    games_value = NumericProperty(200)
    sim = ObjectProperty(None, allownone=True)

    def __init__(self, **kw):
        super().__init__(**kw)
        self._poll = None
        self._rows = {}

    def on_pre_enter(self, *_):
        app = App.get_running_app()
        if not self.running:
            self.games_value = app.settings["sim_games"]
        plan, seats = seat_plan(app.registry.selected())
        self.seats_text = f"{seats} seats" if seats >= 2 else "Seat at least two bots"

    def on_games_value(self, *_):
        App.get_running_app().set_setting("sim_games", int(self.games_value))

    def toggle(self):
        app = App.get_running_app()
        if self.sim is not None and self.sim.running:
            self.sim.stop()
            return
        bots = app.registry.make(fresh=True)
        if len(bots) < 2:
            app.toast("Seat at least two bots first")
            app.go("bots")
            return
        self.sim = Simulation(bots, int(self.games_value))
        self.sim.start()
        self.running = True
        self.run_text = "Stop"
        self.ids.progress.set(0, animate=False)
        self._build_board(bots)
        if self._poll is None:
            self._poll = Clock.schedule_interval(self.refresh, 0.25)

    def leave(self):
        if self.sim is not None:
            self.sim.stop()
        if self._poll is not None:
            self._poll.cancel()
            self._poll = None
        self.running = False
        self.run_text = "Start"
        App.get_running_app().go("home", back=True)

    def _build_board(self, bots):
        board = self.ids.board
        board.clear_widgets()
        self._rows = {}
        for i, bot in enumerate(bots):
            row = LeaderRow(name=bot.name, seat=T.seat_color(i), pct_text="0%", rank_text=str(i + 1),
                            sub="no games yet")
            board.add_widget(row)
            self._rows[bot.name] = row
        self.empty = False

    def refresh(self, _dt=0):
        if self.sim is None:
            return
        rows = self.sim.standings()
        played, games = self.sim.played, self.sim.games
        top = max((st.win_pct for st in rows), default=0) or 100
        board = self.ids.board
        for rank, st in enumerate(rows):
            row = self._rows[st.name]
            lead = rank == 0 and st.wins > 0
            row.rank_text = T.color(T.icon("★"), T.HEX["gold"]) if lead else str(rank + 1)
            row.pct_text = f"{st.win_pct:.1f}%"
            row.sub = (f"{st.wins} win{'s' if st.wins != 1 else ''}  ·  average place {st.avg_position:.2f}"
                       if played else "no games yet")
            # bars are scaled to the leader so differences stay visible
            row.ids.meter.set(st.win_pct / top if played else 0)
            index = len(rows) - 1 - rank
            if board.children[index] is not row:
                board.remove_widget(row)
                board.add_widget(row, index=index)
        self.ids.progress.set(played / games if games else 0)
        rate = self.sim.rate
        self.progress = f"[b]{played}[/b] of {games} games" + (f"  ·  {rate:.0f} games/s" if rate >= 1 else "")
        if self.sim.error:
            self.progress += "\n" + T.color(T.esc(self.sim.error), T.HEX["red"])
        if not self.sim.running:
            self.running = False
            self.run_text = "Run again" if played else "Start"
            if played < games and not self.sim.error:
                self.progress += "  ·  stopped"


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

class SettingsScreen(Screen):
    speech_status = StringProperty("")
    speech_color = ListProperty(T.RED)
    speed_text = StringProperty("")
    _poll = None

    def on_pre_enter(self, *_):
        app = App.get_running_app()
        self.ids.speed.value = float(app.settings["step_delay"])
        self._describe_speed(app.settings["step_delay"])
        self.show_speech()
        self._poll = Clock.schedule_interval(self.show_speech, 0.5)    # speech reports come in later

    def on_leave(self, *_):
        if self._poll is not None:
            self._poll.cancel()
            self._poll = None

    def show_speech(self, *_):
        speaker = App.get_running_app().speaker
        if speaker.note:
            self.speech_status, self.speech_color = f"{speaker.note}\n{speaker.error or ''}".strip(), T.MUTED
        elif speaker.error:
            self.speech_status, self.speech_color = f"Speech is not working: {speaker.error}", T.RED
        else:
            self.speech_status = ""

    def set_speed(self, value):
        if value is None:
            return
        App.get_running_app().set_setting("step_delay", float(value))
        self._describe_speed(value)

    def _describe_speed(self, value):
        value = float(value)
        self.speed_text = f"{value:g} second{'' if value == 1 else 's'} between turns when a game plays by itself"


# ---------------------------------------------------------------------------
# the app
# ---------------------------------------------------------------------------

class PerudoApp(App):
    def build(self):
        self.title = "Perudo"
        self.icon = str(Path(__file__).resolve().parents[1] / "assets" / "icon.png")
        Window.clearcolor = T.BG
        Window.softinput_mode = "below_target"
        Window.bind(on_keyboard=self._on_key)
        data_dir = Path(self.user_data_dir)
        self.settings = Settings(str(data_dir / "settings.json"))
        self.registry = BotRegistry(str(self._user_bots_dir(data_dir)))
        self.registry.scan_bundled()
        self.registry.scan_user()
        self.registry.apply_selection(self.settings["table"])
        self.bots_dir = self._bots_dir()
        self.speaker = Speaker(str(data_dir / "edge-tts"), play=self._play_mp3, enabled=self.settings["tts"],
                               engine=self.settings["tts_engine"])
        ROUTER.install()
        ROUTER.speak = lambda text, lang, seed: self.speaker.say(text, self.settings["tts_lang"] or lang, seed)
        Builder.load_string(KV)
        sm = ScreenManager(transition=SlideTransition(duration=0.22))
        for cls, name in ((HomeScreen, "home"), (BotsScreen, "bots"), (GameScreen, "game"),
                          (SimScreen, "sim"), (SettingsScreen, "settings")):
            sm.add_widget(cls(name=name))
        self.sm = sm
        # The screens sit in a frame padded clear of the phone's status and
        # navigation bars (Android 15 draws apps under them).
        self.frame = BoxLayout()
        self.frame.add_widget(sm)
        Window.bind(on_resize=lambda *_: Clock.schedule_once(self.fit_system_bars, 0.1))
        return self.frame

    def on_start(self):
        for delay in (0, 0.4, 1.5):          # the insets are known once the view is attached
            Clock.schedule_once(self.fit_system_bars, delay)

    def on_resume(self):
        Clock.schedule_once(self.fit_system_bars, 0.3)

    def fit_system_bars(self, *_):
        bars = system_bars()
        if bars is not None:
            window_height, left, top, right, bottom = bars
            self.frame.padding = bar_padding(Window.height, window_height, left, top, right, bottom)

    @staticmethod
    def _user_bots_dir(data_dir: Path) -> Path:
        """Where imported bots live. On Android the app's folder on shared
        storage (Android/data/<package>/files/bots), so files can also be
        copied in over USB; elsewhere next to the settings."""
        try:
            from android import mActivity   # type: ignore
            files = mActivity.getExternalFilesDir(None)
            if files is not None:
                return Path(files.getAbsolutePath()) / "bots"
        except Exception:
            pass
        return data_dir / "bots"

    def save_table(self):
        self.settings["table"] = self.registry.selection()

    def _bots_dir(self) -> str:
        try:
            from android.storage import primary_external_storage_path   # type: ignore
            from android.permissions import Permission, request_permissions   # type: ignore
            request_permissions([Permission.READ_EXTERNAL_STORAGE])
            return primary_external_storage_path()
        except Exception:
            return str(Path.home())

    # -- navigation ----------------------------------------------------------

    def go(self, name: str, back: bool = False):
        self.sm.transition.direction = "right" if back else "left"
        self.sm.current = name

    def back(self):
        screen = self.sm.current_screen
        if hasattr(screen, "leave"):
            screen.leave()
        else:
            self.go("home", back=True)

    def _on_key(self, _window, key, *_args):
        if key == 27 and self.sm.current != "home":     # Android back / Escape
            self.back()
            return True
        return False

    def toast(self, text: str):
        label = Label(text=text, markup=True, font_size=sp(14), color=T.TEXT, size_hint=(None, None),
                      padding=(dp(16), dp(10)), opacity=0)
        label.texture_update()
        label.size = label.texture_size
        with label.canvas.before:
            from kivy.graphics import Color, RoundedRectangle
            Color(*T.RAISED)
            bg = RoundedRectangle(radius=[dp(18)])
        label.bind(pos=lambda w, _v: setattr(bg, "pos", w.pos), size=lambda w, _v: setattr(bg, "size", w.size))
        label.pos = (Window.width / 2 - label.width / 2, dp(96) + self.frame.padding[3])
        Window.add_widget(label)
        anim = Animation(opacity=1, d=0.15) + Animation(d=2.2) + Animation(opacity=0, d=0.3)
        anim.bind(on_complete=lambda *_: Window.remove_widget(label))
        anim.start(label)

    # -- actions -------------------------------------------------------------

    def set_setting(self, key, value):
        if value is None:
            return
        self.settings[key] = value
        if key == "tts":
            self.speaker.enabled = bool(value)
        elif key == "tts_engine":
            self.speaker.engine = value
            self.speaker.error = self.speaker.note = None

    def test_voice(self):
        if not self.settings["tts"]:
            self.sm.get_screen("settings").ids.tts_toggle.active = True    # turns speech on, visibly
        lang = self.settings["tts_lang"] or "nl"
        line = "Dudo! Ik geloof er niets van." if lang == "nl" else "Dudo! I don't believe a word of it."
        self.speaker.clear()
        self.speaker.say(line, lang, random.choice(["Sniper", "Trihard", "Pokerface", "Carlos Dos"]))

    def start_game(self, human: bool):
        bots = self.registry.make(fresh=True)
        if human:
            bots.append(HumanSeat(self.settings["human_name"] or "You"))
        if len(bots) < 2:
            self.toast("Seat at least one bot to play against" if human else "Seat at least two bots to watch")
            self.go("bots")
            return
        session = GameSession(bots, step=not human, delay=self.settings["step_delay"])
        self.speaker.clear()
        self.go("game")
        self.sm.get_screen("game").begin(session, human)

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


def system_bars():
    """(window height, left, top, right, bottom) of the system bars in
    pixels on Android, or None elsewhere or when not known yet."""
    try:
        from android import mActivity    # type: ignore
        from jnius import autoclass       # type: ignore
    except Exception:
        return None
    try:
        decor = mActivity.getWindow().getDecorView()
        insets = decor.getRootWindowInsets()
        if insets is None or decor.getHeight() <= 0:
            return None
        if autoclass("android.os.Build$VERSION").SDK_INT >= 30:
            kinds = autoclass("android.view.WindowInsets$Type")
            bars = insets.getInsets(kinds.systemBars() | kinds.displayCutout())
            sides = (bars.left, bars.top, bars.right, bars.bottom)
        else:
            sides = (insets.getSystemWindowInsetLeft(), insets.getSystemWindowInsetTop(),
                     insets.getSystemWindowInsetRight(), insets.getSystemWindowInsetBottom())
        return (decor.getHeight(),) + tuple(int(v) for v in sides)
    except Exception:
        return None


def run():
    PerudoApp().run()
