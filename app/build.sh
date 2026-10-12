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
if [ -z "${ANDROIDSDK:-}" ] || [ -z "${ANDROIDNDK:-}" ]; then
  echo "ANDROIDSDK / ANDROIDNDK are not set: run this inside 'nix develop'" >&2
  echo "(or 'nix develop .#prebuilt' with your own SDK and NDK in those variables)." >&2
  exit 1
fi
# Buildozer reads any spec key from the environment as SECTION_KEY.
export APP_ANDROID_SDK_PATH="$ANDROIDSDK"
export APP_ANDROID_NDK_PATH="$ANDROIDNDK"
if [ -n "${P4A_SOURCE_DIR:-}" ]; then
  export APP_P4A_SOURCE_DIR="$P4A_SOURCE_DIR"
elif [ -d .p4a-src ]; then
  export APP_P4A_SOURCE_DIR="$(ls -d "$PWD"/.p4a-src/python*android* | head -1)"
fi
exec uv run buildozer android "${1:-debug}"
