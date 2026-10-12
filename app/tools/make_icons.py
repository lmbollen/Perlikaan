"""
Draw the app icon, the adaptive icon layers and the launch screen into
app/assets. Run after changing the look:

    cd app
    uv run --with pillow python tools/make_icons.py

Two ivory dice on the felt, the back one showing the Paco with its red
pip, in the colours of perudo_app/theme.py.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

APP = Path(__file__).resolve().parents[1]
OUT = APP / "assets"
SS = 4                                     # supersampling for smooth edges

FELT = (15, 29, 24)
FELT_LIGHT = (33, 58, 47)
IVORY = (246, 240, 226)
PIP = (30, 27, 23)
PACO = (210, 64, 46)
TEXT = (242, 235, 220)
MUTED = (163, 184, 172)

PIPS = {
    1: [(0.5, 0.5)],
    5: [(0.27, 0.27), (0.27, 0.73), (0.5, 0.5), (0.73, 0.27), (0.73, 0.73)],
    6: [(0.27, 0.25), (0.27, 0.5), (0.27, 0.75), (0.73, 0.25), (0.73, 0.5), (0.73, 0.75)],
}


def die(size: int, value: int, angle: float) -> Image.Image:
    """One die with a soft shadow, rotated, on a transparent square."""
    pad = size // 2
    canvas = Image.new("RGBA", (size + 2 * pad, size + 2 * pad), (0, 0, 0, 0))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    r = int(size * 0.22)
    ImageDraw.Draw(shadow).rounded_rectangle((pad, pad + size * 0.07, pad + size, pad + size * 1.07), r,
                                             fill=(0, 0, 0, 120))
    shadow = shadow.filter(ImageFilter.GaussianBlur(size * 0.05))
    canvas.alpha_composite(shadow)
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((pad, pad, pad + size, pad + size), r, fill=IVORY)
    for px, py in PIPS[value]:
        big = value == 1
        dia = size * (0.32 if big else 0.19)
        cx, cy = pad + px * size, pad + (1 - py) * size
        d.ellipse((cx - dia / 2, cy - dia / 2, cx + dia / 2, cy + dia / 2), fill=PACO if big else PIP)
    return canvas.rotate(angle, resample=Image.BICUBIC, expand=False)


def dice_pair(size: int) -> Image.Image:
    """The two dice, sized to fill a `size` square."""
    s = size * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    unit = int(s * 0.46)
    back = die(unit, 1, 16)
    front = die(unit, 5, -9)
    img.alpha_composite(back, (int(s * 0.36 - back.width / 2), int(s * 0.38 - back.height / 2)))
    img.alpha_composite(front, (int(s * 0.62 - front.width / 2), int(s * 0.62 - front.height / 2)))
    return img.resize((size, size), Image.LANCZOS)


def felt(size: int, rounded: bool) -> Image.Image:
    s = size * SS
    img = Image.new("RGBA", (s, s), FELT + (255,))
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((s * 0.1, s * 0.05, s * 0.9, s * 0.85), fill=FELT_LIGHT + (255,))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(s * 0.16)))
    if rounded:
        mask = Image.new("L", (s, s), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), int(s * 0.22), fill=255)
        img.putalpha(mask)
    return img.resize((size, size), Image.LANCZOS)


def font(size: int) -> ImageFont.FreeTypeFont:
    import kivy
    return ImageFont.truetype(str(Path(kivy.__file__).parent / "data" / "fonts" / "Roboto-Bold.ttf"), size)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    icon = felt(512, rounded=True)
    pair = dice_pair(400)
    icon.alpha_composite(pair, (56, 60))
    icon.save(OUT / "icon.png")

    # Adaptive icon: 108dp layers, of which the launcher shows the middle
    # 72dp and may mask it to a circle, so the dice stay within the centre.
    felt(432, rounded=False).save(OUT / "icon_bg.png")
    fg = Image.new("RGBA", (432, 432), (0, 0, 0, 0))
    fg.alpha_composite(dice_pair(264), (84, 86))
    fg.save(OUT / "icon_fg.png")

    # Launch screen, shown on the felt colour (android.presplash_color).
    splash = Image.new("RGBA", (720, 560), FELT + (255,))
    splash.alpha_composite(dice_pair(300), (210, 20))
    d = ImageDraw.Draw(splash)
    title = font(96)
    w = d.textlength("Perudo", font=title)
    d.text(((720 - w) / 2, 330), "Perudo", font=title, fill=TEXT)
    sub = font(30)
    tag = "Liar's dice against cunning bots"
    w = d.textlength(tag, font=sub)
    d.text(((720 - w) / 2, 460), tag, font=sub, fill=MUTED)
    splash.convert("RGB").save(OUT / "presplash.png")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
