"""
Perudo for Android (and the desktop): entry point for Buildozer and for
`python main.py`. On the desktop the engine is imported from the repository
next to this folder; in the APK `build.sh` has copied it in.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not (HERE / "perudo").is_dir():
    sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
os.environ.setdefault("KIVY_NO_ARGS", "1")

from perudo_app.ui import run  # noqa: E402

if __name__ == "__main__":
    run()
