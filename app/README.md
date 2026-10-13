# Perudo for Android

A touch interface over the engine and the bots of this repository, built with
[Kivy](https://kivy.org) and packaged with [Buildozer](https://buildozer.readthedocs.io).
Everything outside Python that the build needs comes from Nix.

What it does:

- **Home**: start a game at the table, watch one, or run a simulation; the
  bots currently seated are listed underneath.
- **Players**: every bot of this repository is listed. Nobody is seated the
  first time the app starts; switch on the bots you want and set their
  copies, and the table stays as you left it after a restart. *Clear*
  unseats everyone. *Load
  file* imports any single-file bot (same API as the example bot) by copying
  it into the app's bots folder, so it is there again after a restart; Remove
  deletes that copy. On Android the folder is
  `Android/data/org.perlikaan.perudo/files/bots`, reachable over USB: `.py`
  files copied in there by hand are picked up as well (listed, not seated).
- **Watch a game**: starts paused; Play lets it run (Pause stops it at once),
  Step plays one turn at a time. The card on top shows the current bid, who made it and who is
  thinking. The *Dice* switch shows every player's dice, with the total per
  face (Pacos counted, naturals in small print) and the dice that count for
  the bid ringed in gold. When someone calls Dudo or Calza the cups come up:
  the counted dice stay lit, the rest fade, and the card says how many were
  on the table and who loses a die. The feed below tells the game as it
  happens, with the bots' table talk in speech bubbles.
- **Warm-up games**: before a game you watch or join, the bots can first play
  a number of games among themselves (Settings: none, 10, 50 or 200; you sit
  them out), so the bots that learn get to know the table. The progress is
  shown on the game screen, with a button to skip the rest.
- **Join the table**: you take a seat. Only your own dice are shown until a
  reveal. Tap a die for the face, set the quantity (the lowest legal bid is
  preselected), and bid, or call Dudo or Calza when they are legal. When a
  round is decided the cups stay up until you tap *Next round*.
- **Play together**: friends anywhere (over the internet, when the build has
  an Ably key) or on the same Wi-Fi or one phone's hotspot play at one table,
  each on their own phone. One phone hosts a lobby and
  shows its four-character code; the others join with that code. The first
  time you host or join, the app asks your name; it remembers it, and you
  can change it in the lobby. The host decides whether guests may bring bots
  and how many each, and switches every bot on or off (the host's own bots
  and the guests'). A guest's bots run on the guest's phone: only what the
  engine tells a bot and the bot's answers travel. Every phone sees only its
  owner's dice until the cups come up; the host starts each next round and
  can deal again from the results. The host's phone runs the game; if a
  guest's phone drops out, its seats play a safe default move.
- **Simulation**: 100 to 5000 games with the same bot instances (so learning
  bots keep learning), with a live leaderboard: win rate, wins, average place
  and bars scaled to the leader, re-ranked as the games come in.
- **Talk**: a bot says one of its lines a quarter of the time (at most one
  per turn); bids are not read out, the screen shows them. Speech only
  happens in the game on screen, never in simulations or warm-up games, and
  the next turn waits until the bots have finished speaking.
- **Settings**: read the table talk aloud, every bot with its own voice. By
  default with the phone's own text-to-speech voices (offline); Microsoft's
  online neural voices through edge-tts can be chosen instead, and the
  phone's voices take over whenever that service refuses (it has refused
  edge-tts with a 403 since October 2026; the *Speech check* workflow shows
  when it works again). Also the voice language, a Test button, whether dice
  are shown when watching, the auto-play speed and your name.

The Android back button goes back a screen (and leaves a game). The look is
one dark felt table with ivory dice, drawn on the canvas
(`perudo_app/widgets.py`, colours in `perudo_app/theme.py`); the icon and
launch screen in `assets/` come from `tools/make_icons.py`.

![The six screens](docs/screens.png)

## Try it on the desktop

```sh
cd app
uv run python main.py
```

`app/` is its own [uv](https://docs.astral.sh/uv/) project: `pyproject.toml`
and `uv.lock` pin Kivy, edge-tts, Buildozer and Cython, and `uv run` creates
`app/.venv` on first use. The engine is imported from the repository next to
this folder.

## Build the APK

```sh
cd app
nix develop            # Android SDK, NDK, JDK, build tools; runs uv sync
./build.sh             # copies ../perudo in, runs: uv run buildozer android debug
```

The APK lands in `app/bin/`. `./build.sh release` builds a release APK.
`flake.nix` pins the SDK platform (35), build tools (35.0.0), NDK
(25.2.9519653) and Apache Ant and exports them as `ANDROIDSDK`, `ANDROIDNDK`
and `ANT_HOME`, which `build.sh` passes to Buildozer with
`android.skip_update = True`, so Buildozer never downloads tools of its own
(it gets a symlink view of the SDK under `app/.buildozer/android-sdk`, because
it expects sdkmanager under `tools/bin` where the SDK keeps it under
`cmdline-tools`). Nix provides nothing on the Python side: the shell
runs `uv sync`, so Buildozer and Cython come from `uv.lock`. The first build
still fetches python-for-android and the Python packages it compiles for the
phone, and takes a while.

`p4a-recipes/` holds recipes overriding python-for-android's: aiohttp is
installed as pure Python from a current release, because the bundled recipe's
3.8.3 does not compile against Python 3.14 (`build.sh` makes aiohttp's
helpers pure Python the same way).

## Build on GitHub Actions

`.github/workflows/android.yml` runs the same build on every push that
touches `app/` or `perudo/` (and on demand from the Actions tab): it installs
Nix, enters the flake's shell and runs `./build.sh`, caching the
python-for-android build between runs. The APK is attached to the run as the
artifact **perudo-apk**: open the run on the Actions tab and download it
from the Artifacts section at the bottom.

For playing over the internet, add the repository secret **ABLY_KEY** (Settings ->
Secrets and variables -> Actions) with an Ably API key; the workflow passes it
to `./build.sh`, which bakes it into the APK. Without it the APK plays on the
local network only. Locally, set `ABLY_KEY` in the environment before
`./build.sh` (or before `uv run python main.py` on the desktop).

## Restricted networks

The build fetches from four kinds of hosts: PyPI (Buildozer, Cython, the
pure-Python packages), GitHub (python-for-android itself, SDL2 and its
libraries as release assets), the upstream source hosts python-for-android
compiles from (python.org, sqlite.org, openssl.org, GitHub source archives),
and dl.google.com (the Android SDK and NDK, fetched by Nix). Where only
the last two are blocked:

```sh
cd app
uv run python tools/seed_sources.py   # ~100 MB from PyPI, GitHub releases, archive.ubuntu.com
nix develop .#prebuilt                # needs ANDROIDSDK and ANDROIDNDK set to an SDK and NDK you have
./build.sh
```

`tools/seed_sources.py` fetches the python-for-android release from PyPI
into `app/.p4a-src` (used through `p4a.source_dir`, so Buildozer never
clones GitHub) and CPython, SQLite, OpenSSL, libffi, Kivy and pyjnius from
PyPI, GitHub release assets and Ubuntu's package pool, repacked under the
file names python-for-android expects in its download cache with the
markers that make it skip the download. Ubuntu carries newer CPython and
SQLite releases than python-for-android pins, so the script writes
`app/.p4a-src/requirements`, the spec's list with `python3`, `hostpython3`
and `sqlite3` pinned to the versions it fetched, which `build.sh` passes to
Buildozer, and reshapes the one CPython file python-for-android patches so
the patch still applies.

`build.sh` passes the SDK, NDK and python-for-android checkout to Buildozer
through the environment (`APP_ANDROID_SDK_PATH`, `APP_ANDROID_NDK_PATH`,
`APP_P4A_SOURCE_DIR`; Buildozer reads any spec key as `SECTION_KEY`), so
`buildozer.spec` holds no machine-specific paths. `P4A_SOURCE_DIR` overrides
the checkout `seed_sources.py` made.

The SDK and NDK are not covered: Google serves them only from dl.google.com,
and Nix, Buildozer and the SDK manager all fetch from there. The `prebuilt`
shell is for an SDK (platform 33 and build tools installed) and an NDK
(25 to 28) from another machine or a container image such as
`saschpe/android-ndk`; `nix develop` (the default shell) is for networks
where dl.google.com is reachable.

## Tests

```sh
cd app && uv run pytest
```

The tests cover the app's model (loading bots, stepping a game with all dice
and totals, the human seat, simulations with live standings, settings) without
Kivy.

The online tests use an in-memory stand-in for Ably. To run them against Ably
itself (a whole game, joining, leaving), set `ABLY_KEY` and:

```sh
cd app && PERUDO_RELAY=ably uv run pytest tests/test_online.py
```

The key needs the Publish and Subscribe capabilities on `lobby:*`.

## How playing together works

`perudo_app/net.py` and `perudo_app/multiplayer.py`. A lobby code encodes the
last two parts of the host's IPv4 address and which of 16 ports
(47360-47375) it listens on; a joining phone puts its own first two parts in
front, so phones on one local network find each other without typing an
address, a discovery broadcast or a server. Networks that keep their devices
apart (some public and guest Wi-Fi) do not work; a phone's hotspot always
does.

Over the internet, the same messages travel through Ably, a hosted message
relay (`perudo_app/relay.py`), so nobody runs a server. An online lobby code
is random; the host listens on the channel `lobby:<code>:host` and every
guest on its own `lobby:<code>:<random token>`. Each side receives with a
server-sent event stream (`realtime.ably.io`) and sends with an HTTPS POST
(`rest.ably.io`), batching whatever is queued into one Ably message. Pings
every 4 s show a side is still there; one that stays silent for 20 s has
left. The host still runs the lobby and the game. Joining tries the internet
and the local network at once, so a guest never has to say which.

The key ships inside the APK, so give it only the *Publish* and *Subscribe*
capabilities on the channel namespace `lobby:*` (Ably dashboard -> your app ->
API Keys). Ably's free plan (6M messages a month, 200 connections at once)
covers this comfortably: a game is a few thousand small messages, sent in
far fewer batches.
