"""
Check that perudo.profiler learns opponents' strategies correctly.

Two tables, each with Pokerface as the learner:

  known    Probe bots whose strategy we set exactly (face choice and Dudo
           threshold), plus real bots with a simple, known strategy:
           100 IQ Bot (best face, Dudo below 50%), MinRaise (smallest raise,
           never Dudo) and Random Bot (random legal action).
  real     The competition bots. No exact ground truth; we check that the
           learned labels are stable across two different seeds.

For the known table the learned labels are compared with the expected ones
after every checkpoint, so you can see how many games it takes to learn.

Usage:
    uv run python verify_profiler.py --games 200 --seed 1
"""

from __future__ import annotations

import argparse
import random

from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame
from perudo.models import Bid, Dudo
from perudo.profiler import p_bid_true
from perudo.rules import count_matches
from run_simulation import derive_game_seeds


class ProbeBot:
    """
    A bot with an exactly known strategy:
      * Dudo when P(current bid true | own dice) < dudo_at.
      * Otherwise the cheapest legal bid on a face picked by `faces`:
        "honest" = the face it holds most, "bluff" = the face it holds least,
        "random" = any face.
    """

    def __init__(self, name, dudo_at, faces):
        self.name = name
        self.dudo_at = dudo_at
        self.faces = faces
        self._rng = random.Random(name)

    def play(self, state):
        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        total = sum(p.dice_count for p in state.players if p.alive)
        cur = state.current_bid
        if cur is not None and (not bids or p_bid_true(cur, state.my_dice, total) < self.dudo_at):
            return Dudo()

        counts = {f: count_matches(state.my_dice, f, False) for f in range(2, 7)}
        if self.faces == "honest":
            best = max(counts.values())
            face = self._rng.choice([f for f, c in counts.items() if c == best])
        elif self.faces == "bluff":
            worst = min(counts.values())
            face = self._rng.choice([f for f, c in counts.items() if c == worst])
        else:
            face = self._rng.randint(2, 6)

        on_face = [b for b in bids if b.face == face]
        return min(on_face or bids, key=lambda b: (b.quantity, b.face))


# Expected labels per axis. Axes left out are not checked (no clear truth).
KNOWN = {
    "Probe Honest 0.5": {"faces": "honest", "dudo": "normal", "calza": "no-calza"},
    "Probe Honest 0.8": {"faces": "honest", "dudo": "dudo-happy", "calza": "no-calza"},
    "Probe Bluff 0.2": {"faces": "bluffer", "dudo": "dudo-shy", "calza": "no-calza"},
    "Probe Random 0.5": {"faces": "random", "dudo": "normal", "calza": "no-calza"},
    "100 IQ Bot": {"faces": "honest", "dudo": "normal", "calza": "no-calza"},
    "MinRaise Bot": {"dudo": "never-dudo", "raise": "min-raiser", "calza": "no-calza"},
    "Random Bot": {"faces": "random", "dudo": "unpredictable", "bids": "aggressive"},
}
# The exact Dudo threshold of the probes and 100 IQ Bot, to compare with
# the learned one.
TRUE_THRESHOLD = {
    "Probe Honest 0.5": 0.5,
    "Probe Honest 0.8": 0.8,
    "Probe Bluff 0.2": 0.2,
    "Probe Random 0.5": 0.5,
    "100 IQ Bot": 0.5,
}


def known_table(classes):
    probes = [
        ProbeBot("Probe Honest 0.5", 0.5, "honest"),
        ProbeBot("Probe Honest 0.8", 0.8, "honest"),
        ProbeBot("Probe Bluff 0.2", 0.2, "bluff"),
        ProbeBot("Probe Random 0.5", 0.5, "random"),
    ]
    real = [classes[n]() for n in ("HundredIQBot", "MinRaiseBot", "RandomBot")]
    real[0].name = "100 IQ Bot"
    return [classes["PokerfaceBot"](), *probes, *real]


def real_table(classes):
    names = ("SniperBot", "Trihard", "Fabulous", "OppaStyleBot", "SonnyBoy", "HundredIQBot")
    bots = [classes[n]() for n in names]
    bots[-1].name = "100 IQ Bot"
    return [classes["PokerfaceBot"](), *bots]


def play(bots, games, seed, checkpoints=(), on_checkpoint=None):
    for i, game_seed in enumerate(derive_game_seeds(seed, games), start=1):
        PerudoGame(bots=bots, seed=game_seed).play()
        if i in checkpoints and on_checkpoint:
            on_checkpoint(i)
    return bots[0].profiler


def check_known(games, seed, classes):
    print(f"\n{'=' * 78}\nTAFEL 'known': bots met bekende strategie, {games} games\n{'=' * 78}")
    bots = known_table(classes)
    checkpoints = sorted({c for c in (10, 25, 50, 100, 200, 400, games) if c <= games})
    history = []

    def snapshot(n):
        prof = bots[0].profiler
        ok = total = 0
        for name, expected in KNOWN.items():
            labels = prof.profile(name).labels()
            for axis, want in expected.items():
                total += 1
                ok += labels[axis] == want
        history.append((n, ok, total))

    profiler = play(bots, games, seed, checkpoints, snapshot)

    print("\nLeercurve (labels goed / gecontroleerd):")
    for n, ok, total in history:
        print(f"  na {n:4d} games: {ok:2d}/{total}  {'#' * round(30 * ok / total)}")

    print("\nGeleerd per bot (verwacht -> geleerd):")
    all_ok = True
    for name, expected in KNOWN.items():
        prof = profiler.profile(name)
        labels = prof.labels()
        print(prof.summary())
        for axis, want in expected.items():
            got = labels[axis]
            all_ok &= got == want
            print(f"    {'OK  ' if got == want else 'FOUT'} {axis:<6} verwacht {want:<14} geleerd {got}")
        if name in TRUE_THRESHOLD:
            t, fit = prof.dudo_threshold()
            print(f"         Dudo-drempel: echt {TRUE_THRESHOLD[name]:.2f}, geleerd {t:.2f} (fit {fit:.2f})")
    return all_ok


def check_real(games, seed, classes):
    print(f"\n{'=' * 78}\nTAFEL 'real': de echte bots, 2 x {games} games (seeds {seed} en {seed + 1})\n{'=' * 78}")
    runs = [play(real_table(classes), games, s) for s in (seed, seed + 1)]
    stable = total = 0
    for name in sorted(runs[0].profiles):
        a, b = (r.profile(name) for r in runs)
        la, lb = a.labels(), b.labels()
        print(a.summary())
        same = {k: la[k] == lb[k] for k in la}
        stable += sum(same.values())
        total += len(same)
        print("    " + ", ".join(f"{k}: {la[k]}" + ("" if same[k] else f" / {lb[k]} (!)") for k in la))
    print(f"\nStabiel tussen de twee seeds: {stable}/{total} labels")
    return stable, total


def main():
    parser = argparse.ArgumentParser(description="Verify that the opponent profiler learns correctly.")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--table", choices=("known", "real", "both"), default="both")
    args = parser.parse_args()

    classes = discover_bot_classes_by_name()
    if args.table in ("known", "both"):
        ok = check_known(args.games, args.seed, classes)
        print(f"\nAlle verwachte labels geleerd: {'JA' if ok else 'NEE'}")
    if args.table in ("real", "both"):
        check_real(args.games, args.seed, classes)


if __name__ == "__main__":
    main()
