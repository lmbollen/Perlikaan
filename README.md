# Perlikaan — Perudo bot battle

A Perudo (Liar's Dice) game engine with a drop-in bot framework. Write a bot,
put it in `perudo/bots/`, and it automatically joins the table.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to manage Python (3.12) and dependencies.

1. Install uv (if you don't have it yet):

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repo and install everything:

   ```sh
   git clone https://github.com/lmbollen/Perlikaan.git
   cd Perlikaan
   uv sync
   ```

   `uv sync` fetches Python 3.12 if needed, creates `.venv/` and installs the
   dependencies pinned in `uv.lock`.

3. Check that everything works:

   ```sh
   uv run pytest
   ```

## Running games

**Simulation** — many games automatically, with aggregated statistics:

```sh
uv run python run_simulation.py --games 2000 --seed 12345
uv run python run_simulation.py --games 500 --csv results.csv   # optional per-game log
```

**Interactive** — a single game in the terminal, press ENTER per turn:

```sh
uv run python run_interactive.py
uv run python run_interactive.py --seed 938271
uv run python run_interactive.py --bot RandomBot --bot ExampleBot --bot SniperBot
```

Without `--bot`, every bot in `perudo/bots/` plays (once, or `count` times — see below).

**Loss analysis** — dump every round of a simulation to JSONL and see in which
situations one bot loses rounds (bid caught, wrong Dudo, wrong Calza; by
players left, own dice, Palifico) plus its worst calls:

```sh
uv run python analyze_losses.py run --games 300 --bot Pokerface --show 10 --csv losses.csv
uv run python analyze_losses.py dump --games 300 --out rounds.jsonl     # only simulate
uv run python analyze_losses.py analyze rounds.jsonl --bot Sniper       # analyze an existing dump
```

**Opponent profiling** — `perudo/profiler.py` learns per opponent name whether
they bid faces they hold (honest / random / bluffer), how much they risk, at
which odds they call Dudo, whether they jump or min-raise and whether they use
Calza. Any bot can use it (`OpponentProfiler(self.name)`, feed it `observe()`
events); Pokerface does. Check that it learns correctly:

```sh
uv run python verify_profiler.py --games 400 --table known   # bots with known strategies
uv run python verify_profiler.py --games 200 --table real    # the competition bots
```

**Learning and table size** — test against opponents that are not in this repo
(`zoo.py` generates them with random, known strategies), and find the best
settings per number of players (2 to 7):

```sh
uv run python benchmark_adaptive.py --tables 120 --games 60             # learning on vs off
uv run python benchmark_adaptive.py --mix strong                         # + Sniper, Trihard, Oppa Style
uv run python sweep_players.py --tables 60 --games 40                    # variants x table sizes
uv run python sweep_players.py --mix strong --variants "TARGET=0.8"      # your own variants
```

### Trash talk and text-to-speech (optional)

Bots can talk via `perudo.talk`. Enable it with environment variables:

| Variable           | Effect                                                                 |
|--------------------|------------------------------------------------------------------------|
| `TALK_TO_ME=true`  | Print the bots' chatter                                                |
| `TTS=true`         | Also speak it out loud                                                 |
| `TTS_BACKEND`      | `edge` (default, online Microsoft voices) or `piper` (local, offline)  |
| `PIPER_VOICES_DIR` | Where piper voice models are cached (default `~/.cache/perudo/piper-voices`) |
| `EDGE_CACHE_DIR`   | Where edge-tts audio is cached, so each line is fetched only once (default `~/.cache/perudo/edge-tts`) |

```sh
TALK_TO_ME=true TTS=true uv run python run_interactive.py
```

Audio playback needs a system player: `ffplay` (from ffmpeg) for edge, or
`aplay` / `paplay` / `ffplay` for piper. On Debian/Ubuntu:
`sudo apt install ffmpeg`. If speech fails you get one warning and the game
continues silently.

## Writing a bot

Drop a new `.py` file in `perudo/bots/` containing a class with a `play`
method — no registration needed. See `perudo/bots/example_bot.py` for a
documented starting point and `perudo/bot_base.py` / `perudo/models.py` for
the available state and actions.

- Set `count = 3` (class attribute or `self.count`) to seat multiple copies
  of your bot; copies are auto-numbered.
- Set `count = 0` to bench a bot without deleting its file.
- Instance state persists across games in simulation mode, so a bot can learn
  about its opponents.

## Project layout

```
perudo/
  game.py         game engine (PerudoGame)
  rules.py        rule helpers
  models.py       bids, events, observations
  bot_base.py     bot base class / interface
  bot_loader.py   bot discovery and instantiation
  talk.py         optional bot chatter
  tts.py          optional text-to-speech
  bots/           all participating bots
run_simulation.py
run_interactive.py
tests/
```
