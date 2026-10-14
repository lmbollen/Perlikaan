"""
Check the app's layout at the screen sizes of several phones, on the
desktop: the app is started once per phone with its window size, pixel
density and safe area (the notch and the home bar), runs its self-check
(perudo_app/selfcheck.py) and quits. Screenshots and results land in
one folder per phone.

    cd app && python tools/layout_check.py [--out DIR] [--phones iphone-se,pixel-7] [--scenarios home,table]

Needs a display: on a server, run it under Xvfb with a screen big enough
for the tallest phone (xvfb-run -s "-screen 0 1000x2000x24" ...).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parents[1]

# name: (width, height) in points (dp), safe area (left, top, right, bottom) in points.
# The iPhone app hides the status bar, so only the notch (or Dynamic Island)
# and the home bar are in the way; an iPhone SE has neither.
PHONES = {
    "iphone-se": ((375, 667), (0, 0, 0, 0)),
    "iphone-13-mini": ((375, 812), (0, 50, 0, 34)),
    "iphone-16": ((393, 852), (0, 59, 0, 34)),
    "iphone-16-pro-max": ((440, 956), (0, 62, 0, 34)),
    "android-small": ((360, 640), (0, 24, 0, 48)),
    "pixel-7": ((412, 915), (0, 32, 0, 24)),
}
DENSITY = 2          # pixels per point in the window (the layout is the same at any density)
DEFAULT_SCENARIOS = "home,settings,players,table,watch,together"


def run_phone(name: str, out: Path, scenarios: str, timeout: float) -> tuple:
    (w, h), safe = PHONES[name]
    out.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix=f"perudo-{name}-"))
    (home / ".config").mkdir()
    env = dict(os.environ)
    env.update({
        "PERUDO_SELFCHECK": scenarios,
        "PERUDO_SELFCHECK_OUT": str(out),
        "PERUDO_SAFE_AREA": ",".join(str(v) for v in safe),
        "KIVY_METRICS_DENSITY": str(DENSITY),
        "KCFG_GRAPHICS_WIDTH": str(w * DENSITY),
        "KCFG_GRAPHICS_HEIGHT": str(h * DENSITY),
        "KCFG_GRAPHICS_RESIZABLE": "0",
        "KIVY_NO_ARGS": "1",
        "KIVY_NO_CONSOLELOG": "1",
        "KIVY_HOME": str(out / ".kivy"),
        # settings and imported bots: a fresh app every time
        "HOME": str(home),
    })
    try:
        proc = subprocess.run([sys.executable, str(APP / "main.py")], env=env, timeout=timeout,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        code, output = proc.returncode, proc.stdout
    except subprocess.TimeoutExpired as exc:
        code, output = None, (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    log = out / "selfcheck.log"              # Kivy may swallow what the app prints
    text = log.read_text() if log.is_file() else output
    lines = [line for line in text.splitlines() if line.startswith("SELFCHECK")]
    if not lines:                             # it did not get far: the end of Kivy's own log
        logs = sorted((out / ".kivy" / "logs").glob("*.txt"))
        if logs:
            output = "\n".join(l for l in logs[-1].read_text(errors="replace").splitlines()
                               if "mtdev" not in l.lower() and "xclip" not in l and "xsel" not in l)
    return code, lines, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(APP / "layout-check"), help="where screenshots and results go")
    parser.add_argument("--phones", default=",".join(PHONES), help="which phones (comma separated)")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="which self-check scenarios")
    parser.add_argument("--timeout", type=float, default=600, help="seconds per phone")
    args = parser.parse_args()
    failed = []
    for name in [p.strip() for p in args.phones.split(",") if p.strip()]:
        if name not in PHONES:
            parser.error(f"unknown phone {name} (known: {', '.join(PHONES)})")
        print(f"== {name}", flush=True)
        code, lines, output = run_phone(name, Path(args.out) / name, args.scenarios, args.timeout)
        for line in lines:
            print("   " + line[len("SELFCHECK "):], flush=True)
        if code != 0:
            failed.append(name)
            if code is None:
                print(f"   timed out after {args.timeout:.0f}s", flush=True)
            elif not lines:
                print("   the app did not start:\n" + "\n".join("   | " + l for l in output.splitlines()[-30:]))
    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'All phones passed.'}  Screenshots: {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
