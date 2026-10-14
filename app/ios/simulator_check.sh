#!/usr/bin/env bash
# Run the app's self-check (perudo_app/selfcheck.py) in the iPhone simulator:
# it plays through its screens, checks their layout, tries the iPhone's
# voices and playing over the internet, takes screenshots and quits.
#
#   cd app/ios && IOS_SIMULATOR=1 ./build.sh && ./simulator_check.sh [out dir]
#
# The results, screenshots and the app's console end up in the out dir
# (work/simulator-check). Exits 1 when the self-check failed or did not finish.
set -uo pipefail
cd "$(dirname "$0")"
WORK="${IOS_WORK:-$PWD/work}"
OUT="${1:-$WORK/simulator-check}"
APP_BUNDLE="$WORK/simulator/perudo.app"
DEVICE_NAME="${IOS_DEVICE:-iPhone 16}"
SCENARIOS="${PERUDO_SELFCHECK:-all}"
TIMEOUT="${IOS_CHECK_TIMEOUT:-900}"
rm -rf "$OUT"
mkdir -p "$OUT"
[ -d "$APP_BUNDLE" ] || { echo "build it first: IOS_SIMULATOR=1 ./build.sh" >&2; exit 1; }

# A simulator of the newest iOS this Xcode has: the named iPhone, else any iPhone.
UDID="$(xcrun simctl list devices available -j | python3 -c '
import json, sys
name = sys.argv[1]
runtimes = json.load(sys.stdin)["devices"]
ios = sorted((r for r in runtimes if ".iOS-" in r), key=lambda r: [int(p) for p in r.rsplit("iOS-", 1)[1].split("-")])
for runtime in reversed(ios):
    phones = [d for d in runtimes[runtime] if d["name"].startswith("iPhone")]
    pick = [d for d in phones if d["name"] == name] or phones
    if pick:
        print(pick[0]["udid"])
        break
' "$DEVICE_NAME")"
[ -n "$UDID" ] || { echo "no iPhone simulator available" >&2; xcrun simctl list devices available; exit 1; }
echo "Simulator: $(xcrun simctl list devices | grep "$UDID" | sed 's/^ *//')"
step() { echo "== $*"; }
fail() { echo "FAILED: $*" >&2; exit 1; }
BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$APP_BUNDLE/Info.plist")"
step "the app: $BUNDLE_ID," \
     "runs $(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$APP_BUNDLE/Info.plist")," \
     "$(du -sh "$APP_BUNDLE" | cut -f1)"
step "booting"
xcrun simctl boot "$UDID" 2>/dev/null || true
xcrun simctl bootstatus "$UDID" -b > "$OUT/boot.txt" 2>&1 || { tail -20 "$OUT/boot.txt"; fail "the simulator did not boot"; }
step "installing"
xcrun simctl install "$UDID" "$APP_BUNDLE" || fail "installing the app"
step "starting the self-check ($SCENARIOS)"

# The self-check's settings reach the app as environment variables.
export SIMCTL_CHILD_PERUDO_SELFCHECK="$SCENARIOS"
xcrun simctl launch --console-pty --terminate-running-process "$UDID" "$BUNDLE_ID" > "$OUT/console.txt" 2>&1 &
LAUNCH=$!

DATA="$(xcrun simctl get_app_container "$UDID" "$BUNDLE_ID" data)" || fail "finding the app's data folder"
echo "   data folder: $DATA"
RESULTS=""
for _ in $(seq 1 "$TIMEOUT"); do
  RESULTS="$(find "$DATA" -name results.txt -path '*selfcheck*' 2>/dev/null | head -1)"
  [ -n "$RESULTS" ] && break
  if ! kill -0 "$LAUNCH" 2>/dev/null && [ -z "$(find "$DATA" -name results.txt 2>/dev/null)" ]; then
    sleep 3                                   # the app ended: give the file a moment
    RESULTS="$(find "$DATA" -name results.txt -path '*selfcheck*' 2>/dev/null | head -1)"
    break
  fi
  sleep 1
done
xcrun simctl io "$UDID" screenshot "$OUT/simulator-last.png" >/dev/null 2>&1 || true
sleep 2
kill "$LAUNCH" 2>/dev/null || true
xcrun simctl terminate "$UDID" "$BUNDLE_ID" 2>/dev/null || true

CHECK_DIR="$(find "$DATA" -type d -name selfcheck 2>/dev/null | head -1)"
[ -n "$CHECK_DIR" ] && cp -R "$CHECK_DIR"/. "$OUT"/
LOGS="$(find "$DATA" -type d -name logs -path '*.kivy*' 2>/dev/null | head -1)"
[ -n "$LOGS" ] && cp -R "$LOGS" "$OUT/kivy-logs"

echo "---- self-check log"
cat "$OUT/selfcheck.log" 2>/dev/null || echo "(none)"
if [ -z "$RESULTS" ] || ! grep -q '^PASSED' "$OUT/results.txt" 2>/dev/null; then
  echo "---- the app's console (last 80 lines)"
  tail -80 "$OUT/console.txt"
  [ -z "$RESULTS" ] && echo "The self-check did not finish within ${TIMEOUT}s." >&2
  exit 1
fi
echo "The self-check passed in the simulator."
