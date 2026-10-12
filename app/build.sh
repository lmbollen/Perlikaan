#!/usr/bin/env bash
# Build the Android APK. Run inside `nix develop` (see flake.nix); the Nix
# shell provides the Android SDK, NDK, JDK and build tools; uv provides every
# Python dependency from pyproject.toml / uv.lock (Buildozer, Cython, Kivy).
# The engine and the bots are copied in from the repository so the APK is
# self-contained.
set -euo pipefail
cd "$(dirname "$0")"
rsync -a --delete --exclude __pycache__ ../perudo/ perudo/
rm -f perudo/bots/*_old.py
if [ -z "${ANDROIDSDK:-}" ]; then
  echo "ANDROIDSDK is not set: run this inside 'nix develop'." >&2
  exit 1
fi
exec uv run buildozer android "${1:-debug}"
