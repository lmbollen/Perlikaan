"""
The app's own widgets, drawn on the canvas so the app looks the same on
every phone: dice with pips, buttons, a switch, a segmented control, cards,
avatars, chips and progress bars.
"""

from __future__ import annotations

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import Color, Ellipse, Line, PopMatrix, PushMatrix, Rotate, RoundedRectangle
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import (BooleanProperty, ListProperty, NumericProperty, ObjectProperty, OptionProperty,
                             StringProperty)
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

from perudo_app import theme as T

KV = """
#:import T perudo_app.theme

<Surface>:
    canvas.before:
        Color:
            rgba: self.bg
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [self.radius]
        Color:
            rgba: self.border
        Line:
            rounded_rectangle: self.x + dp(0.75), self.y + dp(0.75), self.width - dp(1.5), self.height - dp(1.5), self.radius
            width: self.border_width

<Btn>:
    markup: True
    bold: True
    font_size: sp(15)
    color: self.ink
    halign: "center"
    valign: "middle"
    text_size: self.size
    canvas.before:
        Color:
            rgba: self.fill
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [self.radius]
        Color:
            rgba: self.outline
        Line:
            rounded_rectangle: self.x + dp(0.6), self.y + dp(0.6), self.width - dp(1.2), self.height - dp(1.2), self.radius
            width: dp(1)

<IconBtn>:
    size_hint: None, None
    size: dp(44), dp(44)
    radius: dp(22)
    font_size: sp(21)
    bold: False

<Toggle>:
    size_hint: None, None
    size: dp(50), dp(30)
    opacity: 0.4 if self.disabled else 1
    canvas:
        Color:
            rgba: self.track
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [self.height / 2]
        Color:
            rgba: T.IVORY if self.active else T.MUTED
        Ellipse:
            pos: self.x + dp(4) + self.knob * (self.width - self.height), self.y + dp(4)
            size: self.height - dp(8), self.height - dp(8)

<Segmented>:
    size_hint_y: None
    height: dp(40)
    padding: dp(3)
    spacing: dp(3)
    canvas.before:
        Color:
            rgba: T.RAISED
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(12)]

<Avatar>:
    size_hint: None, None
    size: dp(36), dp(36)
    bold: True
    font_size: self.height * 0.44
    color: T.GOLD_INK
    canvas.before:
        Color:
            rgba: self.bg
        Ellipse:
            pos: self.pos
            size: self.size
        Color:
            rgba: T.GOLD if self.ring else T.CLEAR
        Line:
            circle: self.center_x, self.center_y, self.width / 2 + dp(3)
            width: dp(1.6)

<Chip>:
    markup: True
    size_hint: None, None
    size: self.texture_size
    padding: dp(12), dp(7)
    font_size: sp(14)
    color: T.TEXT
    canvas.before:
        Color:
            rgba: T.RAISED
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [self.height / 2]

<Caption>:
    size_hint_y: None
    height: self.texture_size[1]
    text_size: self.width, None
    halign: "left"
    font_size: sp(12)
    bold: True
    color: T.MUTED

<Body>:
    markup: True
    size_hint_y: None
    height: self.texture_size[1]
    text_size: self.width, None
    halign: "left"
    font_size: sp(14)
    color: T.MUTED

<Meter>:
    canvas:
        Color:
            rgba: self.track
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [self.height / 2]
        Color:
            rgba: self.color if self.fraction > 0 else T.CLEAR
        RoundedRectangle:
            pos: self.pos
            size: max(self.height, self.width * min(1, self.fraction)), self.height
            radius: [self.height / 2]

<TextField>:
    background_normal: ""
    background_active: ""
    background_color: T.CLEAR
    foreground_color: T.TEXT
    hint_text_color: T.FAINT
    cursor_color: T.GOLD
    selection_color: T.GOLD[:3] + [0.35]
    font_size: sp(16)
    padding: dp(14), (self.height - self.line_height) / 2
    multiline: False
    size_hint_y: None
    height: dp(48)
    canvas.before:
        Color:
            rgba: T.RAISED
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(12)]
        Color:
            rgba: T.GOLD if self.focus else T.LINE
        Line:
            rounded_rectangle: self.x + dp(0.6), self.y + dp(0.6), self.width - dp(1.2), self.height - dp(1.2), dp(12)
            width: dp(1.2) if self.focus else dp(1)
        # TextInput draws its text with the last colour set here: put the text colour back
        Color:
            rgba: self.disabled_foreground_color if self.disabled else (self.hint_text_color if not self.text else self.foreground_color)

<TopBar>:
    size_hint_y: None
    height: dp(60)
    padding: dp(6), dp(8), dp(12), dp(8)
    spacing: dp(6)
    IconBtn:
        text: T.icon("\\u2190")
        on_release: root.dispatch("on_back")
    Label:
        text: root.title
        font_size: sp(20)
        bold: True
        color: T.TEXT
        halign: "left"
        valign: "middle"
        text_size: self.size
        shorten: True
        shorten_from: "right"
"""


def _mix(c, other, t):
    return [c[i] * (1 - t) + other[i] * t for i in range(3)] + [c[3]]


class Surface(BoxLayout):
    bg = ListProperty(T.SURFACE)
    border = ListProperty(T.CLEAR)
    border_width = NumericProperty(dp(1))
    radius = NumericProperty(dp(18))


class Collapsible(Surface):
    """A panel that can be hidden. Hidden, it takes no room and passes no
    touches on: Kivy hands touches to children without checking the
    parent's bounds, so the children of a merely shrunk panel would still
    catch taps meant for whatever is drawn where they used to be."""
    shown = BooleanProperty(True)
    full_height = NumericProperty(0)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.size_hint_y = None
        self.bind(shown=self._fit, full_height=self._fit)
        self._fit()

    def _fit(self, *_):
        self.height = self.full_height if self.shown else 0
        self.opacity = 1 if self.shown else 0

    def on_touch_down(self, touch):
        return super().on_touch_down(touch) if self.shown else False

    def on_touch_move(self, touch):
        return super().on_touch_move(touch) if self.shown else False

    def on_touch_up(self, touch):
        return super().on_touch_up(touch) if self.shown else False


class Btn(ButtonBehavior, Label):
    """A rounded button. kind: primary (gold), secondary, danger (Dudo),
    calza, ghost (no fill), outline, seg_on / seg_off (segmented control)."""
    kind = OptionProperty("secondary", options=["primary", "secondary", "danger", "calza", "ghost", "outline",
                                                 "seg_on", "seg_off", "danger_ghost"])
    fill = ListProperty(T.RAISED)
    ink = ListProperty(T.TEXT)
    outline = ListProperty(T.CLEAR)
    radius = NumericProperty(dp(14))

    _STYLE = {
        "primary": (T.GOLD, T.GOLD_INK, T.CLEAR),
        "secondary": (T.RAISED, T.TEXT, T.LINE),
        "danger": (T.RED, T.rgba("#240B06"), T.CLEAR),
        "calza": (T.TEAL, T.rgba("#06211B"), T.CLEAR),
        "ghost": (T.CLEAR, T.TEXT, T.CLEAR),
        "outline": (T.CLEAR, T.TEXT, T.LINE),
        "seg_on": (T.GOLD, T.GOLD_INK, T.CLEAR),
        "seg_off": (T.CLEAR, T.MUTED, T.CLEAR),
        "danger_ghost": (T.CLEAR, T.RED, T.CLEAR),
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bind(kind=self._restyle, state=self._restyle, disabled=self._restyle)
        self._restyle()

    def _restyle(self, *_):
        fill, ink, outline = self._STYLE[self.kind]
        if self.disabled and self.kind == "seg_on":     # keep the choice visible while locked
            self.fill, self.ink, self.outline = T.GOLD[:3] + [0.35], T.GOLD_INK[:3] + [0.8], T.CLEAR
            return
        if self.disabled:
            self.fill = T.RAISED[:3] + [0.45] if fill[3] else T.CLEAR
            self.ink = T.FAINT
            self.outline = T.CLEAR if not outline[3] else T.LINE[:3] + [0.5]
            return
        if self.state == "down":
            fill = _mix(fill, [0, 0, 0, 1], 0.18) if fill[3] else T.RAISED
        self.fill, self.ink, self.outline = list(fill), list(ink), list(outline)


class IconBtn(Btn):
    def __init__(self, **kw):
        kw.setdefault("kind", "ghost")
        super().__init__(**kw)


class Toggle(ButtonBehavior, Widget):
    """An on/off switch."""
    active = BooleanProperty(False)
    knob = NumericProperty(0.0)
    track = ListProperty(T.LINE)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.knob = 1.0 if self.active else 0.0
        self._paint()
        self.bind(knob=self._paint)

    def on_release(self):
        self.active = not self.active

    def on_active(self, *_):
        Animation.cancel_all(self, "knob")
        Animation(knob=1.0 if self.active else 0.0, d=0.14, t="out_quad").start(self)

    def _paint(self, *_):
        self.track = [T.LINE[i] * (1 - self.knob) + T.GOLD[i] * self.knob for i in range(3)] + [1]


class Segmented(BoxLayout):
    """Pick one of a few options: options = [(value, label), ...]."""
    options = ListProperty()
    value = ObjectProperty(None, allownone=True)

    def on_options(self, *_):
        self.clear_widgets()
        for value, label in self.options:
            b = Btn(text=label, radius=dp(10), font_size=sp(14))
            b.bind(on_release=lambda _b, v=value: setattr(self, "value", v))
            b.option_value = value
            self.add_widget(b)
        self.on_value()

    def on_value(self, *_):
        for b in self.children:
            b.kind = "seg_on" if b.option_value == self.value else "seg_off"


class Avatar(Label):
    """A seat's colour with the player's initial."""
    bg = ListProperty(T.PLAYER_COLORS[0])
    ring = BooleanProperty(False)
    name = StringProperty("")

    def on_name(self, *_):
        self.text = (self.name.strip()[:1] or "?").upper()


class Chip(Label):
    pass


class Caption(Label):
    """A small upper-case section label."""

    def on_text(self, *_):
        if self.text != self.text.upper():
            self.text = self.text.upper()


class Body(Label):
    pass


class Meter(Widget):
    """A rounded progress bar."""
    fraction = NumericProperty(0.0)
    color = ListProperty(T.GOLD)
    track = ListProperty(T.RAISED)

    def set(self, fraction: float, animate: bool = True):
        Animation.cancel_all(self, "fraction")
        if animate:
            Animation(fraction=fraction, d=0.35, t="out_quad").start(self)
        else:
            self.fraction = fraction


class TextField(TextInput):
    pass


class TopBar(BoxLayout):
    title = StringProperty("")
    __events__ = ("on_back",)

    def on_back(self, *_):
        pass


# ---------------------------------------------------------------------------
# dice
# ---------------------------------------------------------------------------

_PIPS = {
    1: [(0.5, 0.5)],
    2: [(0.27, 0.73), (0.73, 0.27)],
    3: [(0.27, 0.73), (0.5, 0.5), (0.73, 0.27)],
    4: [(0.27, 0.27), (0.27, 0.73), (0.73, 0.27), (0.73, 0.73)],
    5: [(0.27, 0.27), (0.27, 0.73), (0.5, 0.5), (0.73, 0.27), (0.73, 0.73)],
    6: [(0.27, 0.25), (0.27, 0.5), (0.27, 0.75), (0.73, 0.25), (0.73, 0.5), (0.73, 0.75)],
}


class Die(Widget):
    """A die with pips. value 0 is a die still under the cup. The Paco (1)
    has a large red pip. highlight rings it in gold (it counts for the bid);
    dim fades it (it does not)."""
    value = NumericProperty(1)
    highlight = BooleanProperty(False)
    dim = BooleanProperty(False)
    angle = NumericProperty(0)
    shadow = BooleanProperty(True)

    def __init__(self, **kw):
        kw.setdefault("size_hint", (None, None))
        super().__init__(**kw)
        self._trigger = Clock.create_trigger(self._draw, -1)
        self.bind(pos=self._trigger, size=self._trigger, value=self._trigger, highlight=self._trigger,
                  dim=self._trigger, angle=self._trigger, disabled=self._trigger)
        self._draw()

    def _draw(self, *_):
        s = min(self.width, self.height)
        x, y = self.center_x - s / 2, self.center_y - s / 2
        r = s * 0.22
        a = 0.3 if (self.dim or self.disabled) else 1.0
        self.canvas.clear()
        with self.canvas:
            PushMatrix()
            Rotate(angle=self.angle, origin=self.center)
            if self.value <= 0:
                Color(*T.CUP[:3], a)
                RoundedRectangle(pos=(x, y), size=(s, s), radius=[r])
                Color(*T.LINE[:3], a)
                Line(rounded_rectangle=(x + 0.5, y + 0.5, s - 1, s - 1, r), width=dp(1))
                Color(*T.FAINT[:3], 0.5 * a)
                Ellipse(pos=(self.center_x - s * 0.07, self.center_y - s * 0.07), size=(s * 0.14, s * 0.14))
            else:
                if self.shadow:
                    Color(0, 0, 0, 0.28 * a)
                    RoundedRectangle(pos=(x, y - s * 0.06), size=(s, s), radius=[r])
                Color(*T.IVORY[:3], a)
                RoundedRectangle(pos=(x, y), size=(s, s), radius=[r])
                value = int(self.value)
                if value == 1:
                    d = s * 0.32
                    Color(*T.PACO[:3], a)
                    Ellipse(pos=(self.center_x - d / 2, self.center_y - d / 2), size=(d, d))
                else:
                    d = s * 0.19
                    Color(*T.PIP[:3], a)
                    for px, py in _PIPS.get(value, []):
                        Ellipse(pos=(x + px * s - d / 2, y + py * s - d / 2), size=(d, d))
            if self.highlight:
                Color(*T.GOLD)
                g = dp(2.5)
                Line(rounded_rectangle=(x - g, y - g, s + 2 * g, s + 2 * g, r + g), width=dp(1.6))
            PopMatrix()


class FaceButton(ButtonBehavior, Die):
    """A die to tap, for choosing the face of a bid."""
    selected = BooleanProperty(False)

    def on_selected(self, *_):
        self.highlight = self.selected

    def on_state(self, *_):
        self.angle = -6 if self.state == "down" else 0


class TouchBox(ButtonBehavior, Surface):
    """A card that can be tapped."""
    base_bg = ListProperty(T.SURFACE)

    def on_base_bg(self, *_):
        self.bg = self.base_bg

    def on_state(self, *_):
        self.bg = _mix(self.base_bg, [1, 1, 1, 1], 0.06) if self.state == "down" else self.base_bg


Builder.load_string(KV)
