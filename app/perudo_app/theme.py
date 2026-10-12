"""
Colours, type sizes and small helpers shared by the widgets and screens.

One deliberate look: a dark felt table with ivory dice. Gold marks bids and
whoever is to act, terracotta is Dudo, teal is Calza.
"""

from __future__ import annotations

from kivy.utils import escape_markup


def rgba(hex_color: str, alpha: float = 1.0) -> list:
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [alpha]


HEX = {
    "bg": "#0F1D18",        # the felt
    "surface": "#172A22",   # cards
    "raised": "#213A2F",    # rows, inputs, chips inside cards
    "line": "#2E4A3D",      # borders, tracks
    "text": "#F2EBDC",      # warm ivory
    "muted": "#A3B8AC",
    "faint": "#6E897C",
    "gold": "#E9B44C",
    "gold_ink": "#22190A",  # text on gold
    "red": "#E2654F",
    "teal": "#4CC3A6",
    "ivory": "#F6F0E2",     # die face
    "pip": "#1E1B17",
    "paco": "#D2402E",      # the Paco's pip
    "cup": "#28443A",       # a hidden die
}

BG = rgba(HEX["bg"])
SURFACE = rgba(HEX["surface"])
RAISED = rgba(HEX["raised"])
LINE = rgba(HEX["line"])
TEXT = rgba(HEX["text"])
MUTED = rgba(HEX["muted"])
FAINT = rgba(HEX["faint"])
GOLD = rgba(HEX["gold"])
GOLD_INK = rgba(HEX["gold_ink"])
RED = rgba(HEX["red"])
TEAL = rgba(HEX["teal"])
IVORY = rgba(HEX["ivory"])
PIP = rgba(HEX["pip"])
PACO = rgba(HEX["paco"])
CUP = rgba(HEX["cup"])
CLEAR = [0, 0, 0, 0]

# Seat colours, chosen to stay apart from each other on the felt and from
# the gold, red and teal that mean bid, Dudo and Calza.
PLAYER_HEX = ["#6FA8F5", "#F2905E", "#B993F0", "#7FD06A", "#F07FB0", "#5ED0E0",
              "#E8D36A", "#C9A27A", "#9BB0FF", "#FF9E9E", "#B8C4BC"]
PLAYER_COLORS = [rgba(h) for h in PLAYER_HEX]

# Kivy ships DejaVu Sans, which has the arrows and symbols Roboto lacks.
ICON_FONT = "data/fonts/DejaVuSans.ttf"


def icon(glyph: str) -> str:
    """Markup for a symbol from the icon font."""
    return f"[font={ICON_FONT}]{glyph}[/font]"


def color(text: str, hex_color: str) -> str:
    return f"[color={hex_color}]{text}[/color]"


def esc(text: str) -> str:
    return escape_markup(str(text))


def face_name(face: int, plural: bool = False) -> str:
    if face == 1:
        return "Pacos" if plural else "Paco"
    return str(face)


def bid_text(quantity: int, face: int) -> str:
    return f"{quantity} × {face_name(face, plural=quantity != 1)}"


def seat_color(index: int) -> list:
    return PLAYER_COLORS[index % len(PLAYER_COLORS)]


def seat_hex(index: int) -> str:
    return PLAYER_HEX[index % len(PLAYER_HEX)]
