[app]
title = Perudo
package.name = perudo
package.domain = org.perlikaan
source.dir = .
source.include_exts = py,kv,json,txt,md
source.exclude_dirs = tests, bin, .buildozer, __pycache__, .venv
source.exclude_patterns = flake.nix, flake.lock, build.sh, README.md
version = 0.1.0

# The engine and the bots are copied in by build.sh (app/perudo). edge-tts
# is pure Python; aiohttp and its helpers have python-for-android recipes.
requirements = python3,kivy==2.3.1,edge-tts,aiohttp,aiosignal,aiohappyeyeballs,attrs,frozenlist,multidict,yarl,propcache,idna,certifi,typing_extensions,tabulate,srt

orientation = portrait
fullscreen = 0
android.permissions = INTERNET,READ_EXTERNAL_STORAGE
android.api = 33
android.minapi = 24
android.archs = arm64-v8a
android.allow_backup = True

# The SDK and NDK come from the Nix shell (flake.nix exports ANDROIDSDK and
# ANDROIDNDK); never let Buildozer download its own.
android.skip_update = True
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 0
