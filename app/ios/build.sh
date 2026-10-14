#!/usr/bin/env bash
# Build the iPhone app: an unsigned Perudo.ipa, to sideload with Sideloadly or
# AltStore (they sign it with your own Apple ID). Runs on macOS with Xcode;
# .github/workflows/ios.yml runs it on GitHub's macOS machines.
#
#   cd app/ios && ./build.sh            # -> app/ios/work/Perudo.ipa
#   IOS_SIMULATOR=1 ./build.sh          # ... and work/Perudo-simulator.app
#                                         (for simulator_check.sh)
#
# It needs Python 3 with kivy-ios (pip install kivy-ios "Cython==0.29.36"),
# and from Homebrew: autoconf automake libtool pkg-config. kivy-ios compiles
# Python 3.11 and Kivy for the iPhone the first time (most of an hour); its
# build and dist folders in work/ are reused after that.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
APP="$(cd .. && pwd)"
REPO="$(cd ../.. && pwd)"
WORK="${IOS_WORK:-$HERE/work}"
BUNDLE_ID="${IOS_BUNDLE_ID:-org.perlikaan.perudo}"
VERSION="$(sed -n 's/^version = //p' "$APP/buildozer.spec")"
mkdir -p "$WORK"

# The app's files, with the engine and the bots copied in (as build.sh does
# for the APK), so the app is self-contained.
SRC="$WORK/src"
rm -rf "$SRC"
mkdir -p "$SRC"
rsync -a --exclude __pycache__ --exclude online_key.py \
  "$APP/main.py" "$APP/perudo_app" "$APP/assets" "$SRC/"
rsync -a --exclude __pycache__ "$REPO/perudo/" "$SRC/perudo/"
rm -f "$SRC"/perudo/bots/*_old.py
# Playing over the internet needs an Ably key (see perudo_app/relay.py),
# baked in from ABLY_KEY when that is set.
if [ -n "${ABLY_KEY:-}" ]; then
  printf 'KEY = %s\n' "$(printf '%s' "$ABLY_KEY" | python3 -c 'import json, sys; print(json.dumps(sys.stdin.read().strip()))')" \
    > "$SRC/perudo_app/online_key.py"
  echo "Online play: on (key from ABLY_KEY)"
else
  echo "Online play: off (set ABLY_KEY to turn it on)"
fi

cd "$WORK"
# Python, Kivy and pyobjus (for the iPhone's voices and safe area) for the
# phone (arm64), and for the simulator on an Apple Silicon Mac when asked,
# plus certifi for the TLS certificates (Ably). Both platforms in one go:
# kivy-ios packs what it built into one framework per library. Its record
# of what is built does not say for which platforms: a build folder from a
# phone-only build skips the simulator (so CI caches the two apart).
PLATFORMS=(--platform iphoneos-arm64)
if [ "${IOS_SIMULATOR:-}" = 1 ]; then
  PLATFORMS+=(--platform iphonesimulator-arm64)
fi
toolchain build hostpython3
toolchain build python3 kivy pyobjus "${PLATFORMS[@]}"
toolchain pip install certifi

# The Xcode project. It points at $SRC and copies it in on every build.
rm -rf perudo-ios
toolchain create Perudo "$SRC" --add-framework AVFoundation
python3 - "$APP/assets/icon.png" icon1024.png <<'EOF'
import sys
from PIL import Image
Image.open(sys.argv[1]).convert("RGB").resize((1024, 1024), Image.LANCZOS).save(sys.argv[2])
EOF
toolchain icon perudo-ios icon1024.png

PLIST="perudo-ios/perudo-Info.plist"
pb() { /usr/libexec/PlistBuddy -c "$1" "$PLIST"; }
pb "Set :CFBundleDisplayName Perudo" 2>/dev/null || pb "Add :CFBundleDisplayName string Perudo"
pb "Set :CFBundleShortVersionString $VERSION"
pb "Set :CFBundleVersion $VERSION"
# Portrait only, like the APK.
pb "Delete :UISupportedInterfaceOrientations"
pb "Add :UISupportedInterfaceOrientations array"
pb "Add :UISupportedInterfaceOrientations:0 string UIInterfaceOrientationPortrait"
pb "Delete :UISupportedInterfaceOrientations~ipad" 2>/dev/null || true
pb "Add :UIRequiresFullScreen bool true"      # portrait only is fine for a full-screen app
# Playing together on the same Wi-Fi: iOS asks before an app talks to the
# local network, with this reason.
pb "Add :NSLocalNetworkUsageDescription string Perudo finds and joins lobbies on your Wi-Fi to play together."
# The app's folder shows in the Files app (Perudo), so bot files can be
# dropped into its bots folder.
pb "Add :UIFileSharingEnabled bool true"
pb "Add :LSSupportsOpeningDocumentsInPlace bool true"

# Xcode's output goes to a log; when a build fails its end is shown (the
# summary alone does not say what went wrong).
xcode() {
  local log="$1"; shift
  if ! xcodebuild "$@" > "$log" 2>&1; then
    grep -E "error:|fatal|Undefined symbol|ld: " "$log" | head -40
    echo "---- the end of $log"
    tail -60 "$log"
    return 1
  fi
  grep -E "warning: .*(deprecated|ld:)" "$log" | head -5 || true
  echo "$(basename "$log" .log): build succeeded"
}

# An unsigned build for a phone; Sideloadly or AltStore sign it on install.
rm -rf build-ios Payload Perudo.ipa
xcode xcode-phone.log -project perudo-ios/perudo.xcodeproj -scheme perudo -configuration Release \
  -destination generic/platform=iOS -derivedDataPath build-ios \
  PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
  CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO CODE_SIGN_IDENTITY="" \
  build
APP_BUNDLE="$(find build-ios/Build/Products -maxdepth 2 -name 'perudo.app' -type d | head -1)"
[ -n "$APP_BUNDLE" ] || { echo "no perudo.app was built" >&2; exit 1; }
mkdir Payload
cp -R "$APP_BUNDLE" Payload/
zip -qry Perudo.ipa Payload
echo "Built $WORK/Perudo.ipa ($(du -h Perudo.ipa | cut -f1))"

if [ "${IOS_SIMULATOR:-}" = 1 ]; then
  rm -rf build-sim Perudo-simulator.app
  xcode xcode-simulator.log -project perudo-ios/perudo.xcodeproj -scheme perudo -configuration Release \
    -destination "generic/platform=iOS Simulator" -derivedDataPath build-sim ARCHS=arm64 \
    PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" CODE_SIGNING_ALLOWED=NO \
    build
  SIM_BUNDLE="$(find build-sim/Build/Products -maxdepth 2 -name 'perudo.app' -type d | head -1)"
  [ -n "$SIM_BUNDLE" ] || { echo "no perudo.app was built for the simulator" >&2; exit 1; }
  cp -R "$SIM_BUNDLE" Perudo-simulator.app
  echo "Built $WORK/Perudo-simulator.app"
fi
