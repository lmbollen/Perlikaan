#!/usr/bin/env bash
# Build the Android APK. Run inside `nix develop` (see flake.nix); the Nix
# shell provides the Android SDK, NDK, JDK, Ant and build tools; uv provides
# every Python dependency from pyproject.toml / uv.lock (Buildozer, Cython,
# Kivy). The engine and the bots are copied in from the repository so the APK
# is self-contained.
set -euo pipefail
cd "$(dirname "$0")"
rsync -a --delete --exclude __pycache__ ../perudo/ perudo/
rm -f perudo/bots/*_old.py
if [ -z "${ANDROIDSDK:-}" ] || [ -z "${ANDROIDNDK:-}" ]; then
  echo "ANDROIDSDK / ANDROIDNDK are not set: run this inside 'nix develop'" >&2
  echo "(or 'nix develop .#prebuilt' with your own SDK and NDK in those variables)." >&2
  exit 1
fi

# Buildozer looks for sdkmanager at <sdk>/tools/bin (the layout it makes for
# itself); SDKs from Nix, Android Studio or sdkmanager keep it under
# cmdline-tools/<version>. Give Buildozer a symlink view of the SDK with
# `tools` pointing at the newest cmdline-tools; the SDK itself stays read-only.
sdk="$ANDROIDSDK"
if [ -d "$ANDROIDSDK/cmdline-tools" ]; then
  cmdline="$ANDROIDSDK/cmdline-tools/latest"
  [ -d "$cmdline" ] || cmdline="$(ls -d "$ANDROIDSDK"/cmdline-tools/*/ | sort -V | tail -1)"
  sdk="$PWD/.buildozer/android-sdk"
  rm -rf "$sdk" && mkdir -p "$sdk"
  for entry in "$ANDROIDSDK"/* "$ANDROIDSDK"/.[!.]*; do
    [ -e "$entry" ] || continue
    [ "$(basename "$entry")" = tools ] || ln -s "$entry" "$sdk/"
  done
  ln -s "${cmdline%/}" "$sdk/tools"
fi

# Buildozer reads any spec key from the environment as SECTION_KEY.
export APP_ANDROID_SDK_PATH="$sdk"
export APP_ANDROID_NDK_PATH="$ANDROIDNDK"
if [ -n "${ANT_HOME:-}" ]; then
  export APP_ANDROID_ANT_PATH="$ANT_HOME"
fi
if [ -n "${P4A_SOURCE_DIR:-}" ]; then
  export APP_P4A_SOURCE_DIR="$P4A_SOURCE_DIR"
elif [ -d .p4a-src ]; then
  export APP_P4A_SOURCE_DIR="$(ls -d "$PWD"/.p4a-src/python*android* | head -1)"
fi
# tools/seed_sources.py leaves the requirement list with the versions it
# could fetch (python3, hostpython3, sqlite3 pinned); use it when present.
if [ -f .p4a-src/requirements ]; then
  export APP_REQUIREMENTS="$(cat .p4a-src/requirements)"
fi
# aiohttp's helpers ship C accelerators with pure-Python fallbacks.
# python-for-android installs them with the build machine's pip, which would
# otherwise pick wheels built for this machine, not the phone: take the
# sources and build them pure.
export PIP_NO_BINARY=multidict,yarl,frozenlist,propcache
export MULTIDICT_NO_EXTENSIONS=1 YARL_NO_EXTENSIONS=1 FROZENLIST_NO_EXTENSIONS=1 PROPCACHE_NO_EXTENSIONS=1
exec uv run buildozer android "${1:-debug}"
