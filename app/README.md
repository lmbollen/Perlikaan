# Perudo for Android

A touch interface over the engine and the bots of this repository, built with
[Kivy](https://kivy.org) and packaged with [Buildozer](https://buildozer.readthedocs.io).
Everything outside Python that the build needs comes from Nix.

What it does:

- **Bots**: every bot in `perudo/bots` is listed; tick the ones to seat and set
  copies. *Load bot file* adds any single-file bot from the phone's storage
  (it must follow the same API as the example bot).
- **Watch a game**: one game, turn by turn (*Next turn*) or automatically
  (*Auto*, seconds per turn in Settings). *Show dice* reveals every player's
  dice with the totals per face at the top (with Pacos counted, naturals in
  brackets). The bots' table talk appears under the log.
- **Join the table**: you take a seat. Only your own dice are shown; pick a
  face and quantity (the lowest legal bid is preselected), or call Dudo or
  Calza when they are legal.
- **Simulation**: many games with the same bot instances (so learning bots
  keep learning), with a live leaderboard: bars for the win percentage,
  wins and average place, re-ranked as games come in. Stop any time.
- **Settings**: text to speech with edge-tts (Microsoft's online neural
  voices, needs internet; every bot keeps its own voice), the voice language,
  whether dice are shown when watching, the auto-play speed, your name, and
  the number of games per simulation.

## Try it on the desktop

```sh
cd app
nix develop            # or: pip install kivy edge-tts
python main.py
```

The engine is imported from the repository next to this folder.

## Build the APK

```sh
cd app
nix develop            # Android SDK, NDK, JDK, build tools; creates .venv with Buildozer
./build.sh             # copies ../perudo in, runs: buildozer android debug
```

The APK lands in `app/bin/`. `./build.sh release` builds a release APK.
`flake.nix` pins the SDK platform (33), build tools (33.0.2) and NDK
(25.2.9519653) and exports them as `ANDROIDSDK` / `ANDROIDNDK`, which
`buildozer.spec` uses with `android.skip_update = True`, so Buildozer never
downloads tools of its own. The first build still fetches python-for-android
and the Python packages it compiles for the phone, and takes a while.

## Tests

```sh
uv run pytest app/tests            # from the repository root
```

The tests cover the app's model (loading bots, stepping a game with all dice
and totals, the human seat, simulations with live standings, settings) without
Kivy.
