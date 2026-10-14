[app]
title = Perudo
package.name = perudo
package.domain = org.perlikaan
source.dir = .
source.include_exts = py,kv,json,txt,md,png
source.exclude_dirs = tests, bin, tools, docs, manifest, ios, p4a-recipes, .buildozer, .p4a-src, __pycache__, .venv
source.exclude_patterns = flake.nix, flake.lock, build.sh, README.md
version = 0.2.0

# Drawn by tools/make_icons.py.
icon.filename = %(source.dir)s/assets/icon.png
icon.adaptive_foreground.filename = %(source.dir)s/assets/icon_fg.png
icon.adaptive_background.filename = %(source.dir)s/assets/icon_bg.png
presplash.filename = %(source.dir)s/assets/presplash.png
android.presplash_color = #0F1D18

# The engine and the bots are copied in by build.sh (app/perudo). edge-tts
# is pure Python; aiohttp and its helpers have python-for-android recipes.
# python3, hostpython3 and sqlite3 are left at the versions python-for-android
# pins (tools/seed_sources.py overrides them through APP_REQUIREMENTS on a
# restricted network).
requirements = python3,hostpython3,sqlite3,kivy==2.3.1,edge-tts,aiohttp,aiosignal,aiohappyeyeballs,attrs,frozenlist,multidict,yarl,propcache,idna,certifi,typing_extensions,tabulate,srt

orientation = portrait
fullscreen = 0
android.permissions = INTERNET,READ_EXTERNAL_STORAGE
android.api = 35
android.minapi = 24
android.archs = arm64-v8a
android.allow_backup = True
# Lets the app find the phone's text-to-speech engine (the bots' voices).
android.extra_manifest_xml = manifest/queries.xml

# The SDK, NDK and Ant come from the Nix shell (flake.nix exports ANDROIDSDK,
# ANDROIDNDK and ANT_HOME; build.sh passes them on through the environment as
# APP_ANDROID_SDK_PATH, APP_ANDROID_NDK_PATH and APP_ANDROID_ANT_PATH, which
# Buildozer only honours for keys present here); never let Buildozer download
# its own.
android.sdk_path =
android.ndk_path =
android.ant_path =
android.skip_update = True
android.accept_sdk_license = True

# Recipes overriding python-for-android's (aiohttp as pure Python, see
# p4a-recipes/aiohttp/__init__.py).
p4a.local_recipes = ./p4a-recipes

# python-for-android release 2026.05.09 (the same release tools/seed_sources.py
# takes from PyPI), so builds are reproducible.
p4a.branch = master
p4a.commit = 58d21141f17c889bf8585f5665921d72028f8831
# A python-for-android checkout to use instead of cloning it from GitHub
# (build.sh sets it through APP_P4A_SOURCE_DIR from P4A_SOURCE_DIR or
# app/.p4a-src; tools/seed_sources.py fetches the release from PyPI for
# networks where GitHub clones are blocked).
p4a.source_dir =

[buildozer]
log_level = 2
warn_on_root = 0
