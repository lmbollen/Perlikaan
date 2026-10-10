"""
Measure good starting values for Pokerface and Carlos.

Our bots learn per opponent during a game, but a single game leaves no time
to warm up, so the values they start from matter. This script plays many
games at tables of 2-7 players with Pokerface and Carlos against the static
bots and The Don, pools what both learned about all opponents, and prints it
next to the constants currently in the code. It writes nothing: the numbers
are for tuning the constants by hand.

Usage:
    uv run python measure_defaults.py --tables 24 --games 25
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from multiprocessing import Pool

from benchmark_adaptive import STATIC_BOTS, RoundLimit, TooManyRounds, build, fresh_seed
from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame
from perudo.profiler import OpponentProfiler, Profile
from run_simulation import derive_game_seeds

POOL = STATIC_BOTS + ("TheDonBot",)


def table(job):
    seed, t, games = job
    rng = random.Random(seed * 1009 + t)
    classes = discover_bot_classes_by_name()
    size = rng.choice([2, 3, 4, 5, 6, 7])
    ours = [classes["PokerfaceBot"](), classes["CarlosBot"]()]
    if size == 2:
        ours = [rng.choice(ours)]
    opponents = rng.sample(POOL, min(len(POOL), size - len(ours)))
    opponents += rng.choices(POOL, k=size - len(ours) - len(opponents))
    bots = [*ours, *build([("repo", n) for n in opponents], classes, taken={b.name for b in ours})]
    for s in derive_game_seeds(seed * 31 + t, games):
        try:
            PerudoGame(bots=bots, seed=s, listeners=[RoundLimit()]).play()
        except TooManyRounds:
            pass
    out = {}
    for bot in ours:
        prof = bot.profiler.profiles.values()
        data = {
            "excess": [(k, t, n) for p in prof for k, (t, n) in p.excess.items()],
            "false": [(p.false_faced, p.false_challenged) for p in prof],
            "calza": [(p.calzas, len(p.facing)) for p in prof],
            "facing": [x for p in prof for x in p.facing[-400:]],
            "risk": [(p.bid_risk_sum, p.bids) for p in prof],
        }
        if type(bot).__name__ == "CarlosBot":
            data["hist"] = [(k, r, c) for kinds in bot.hist.values() for k, rs in kinds.items() for r, c in rs.items()]
            data["other"] = [(n, k, c) for ns in bot.other.values() for n, ks in ns.items() for k, c in ks.items()]
            data["beta"] = list(bot._beta_table)
        out[type(bot).__name__] = data
    return out


def fit_logistic(points, a=2.0, b=-7.0):
    for _ in range(25):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in points:
            s = 1 / (1 + math.exp(-(a + b * x)))
            e = (1.0 if y else 0.0) - s
            w = s * (1 - s)
            ga, gb = ga + e, gb + e * x
            haa, hab, hbb = haa + w, hab + w * x, hbb + w * x * x
        det = haa * hbb - hab * hab
        if det <= 1e-9:
            break
        a += (hbb * ga - hab * gb) / det
        b += (haa * gb - hab * ga) / det
    return a, b


def main():
    parser = argparse.ArgumentParser(description="Measure starting values for Pokerface and Carlos.")
    parser.add_argument("--tables", type=int, default=24)
    parser.add_argument("--games", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    seed = fresh_seed(args.seed)
    with Pool(args.jobs) as pool:
        parts = pool.map(table, [(seed, t, args.games) for t in range(args.tables)])

    classes = discover_bot_classes_by_name()
    for name in ("PokerfaceBot", "CarlosBot"):
        ds = [p[name] for p in parts if name in p]
        excess = defaultdict(lambda: [0.0, 0])
        for d in ds:
            for k, t, n in d["excess"]:
                excess[k][0] += t
                excess[k][1] += n
        faced = sum(f for d in ds for f, _ in d["false"])
        hit = sum(h for d in ds for _, h in d["false"])
        calzas = sum(c for d in ds for c, _ in d["calza"])
        chances = calzas + sum(f for d in ds for _, f in d["calza"])
        facing = [tuple(x) for d in ds for x in d["facing"]]
        sample = random.Random(0).sample(facing, min(len(facing), 20000))
        prof = Profile("table")
        prof.facing = sample
        dudo_at, fit = prof.dudo_threshold()
        a, b = fit_logistic(sample)
        risk = sum(r for d in ds for r, _ in d["risk"]) / max(sum(n for d in ds for _, n in d["risk"]), 1)
        print(f"\n=== {name} ({len(ds)} tables)")
        print(f"  value of a bid (extra dice held), per kind  [profiler default {OpponentProfiler.PRIOR_EXCESS}]")
        for k in sorted(excess):
            t, n = excess[k]
            print(f"    {k:<12} {t / n:+.2f}  (n={n})")
        print(f"  P(challenge | false bid) {hit / max(faced, 1):.3f}  [default {OpponentProfiler.PRIOR_CHALLENGE}]")
        print(f"  P(Calza | facing a bid)  {calzas / max(chances, 1):.3f}  [default {OpponentProfiler.PRIOR_CALZA}]")
        print(f"  Dudo threshold (own dice) {dudo_at:.2f} (fit {fit:.2f}); logistic a={a:.2f} b={b:.2f}")
        print(f"  P(true) of their raises from their own dice: {risk:.2f}")
        if name == "CarlosBot":
            groups = defaultdict(lambda: defaultdict(float))
            for d in ds:
                for k, r, c in d["hist"]:
                    groups[k.rsplit("-", 1)[0]][r] += c
            print("  residual (held - implied) per kind: centre (median) and spread (mean |r - centre|)")
            for k in sorted(groups):
                hist = groups[k]
                n = sum(hist.values())
                rs = sorted(hist)
                cum, median = 0.0, 0
                for r in rs:
                    cum += hist[r]
                    if cum >= n / 2:
                        median = r
                        break
                spread = sum(abs(r - median) * c for r, c in hist.items()) / n
                print(f"    {k:<14} centre {median:+d}  spread {spread:.2f}  (n={n:.0f})")
            per_die = defaultdict(lambda: [0.0, 0.0])
            for d in ds:
                for n, k, c in d["other"]:
                    per_die[n][0] += k * c
                    per_die[n][1] += n * c
            print("  passed-over faces, matches per die (binomial default 0.333):")
            print("    " + "  ".join(f"{n} dice: {m / t:.3f}" for n, (m, t) in sorted(per_die.items())))
            betas = [d["beta"] for d in ds]
            mean = [sum(b[i] for b in betas) / len(betas) for i in range(len(betas[0]))]
            print(f"  raise-choice weights {[round(x, 2) for x in mean]}  [default {classes['CarlosBot'].RAISE_PRIOR}]")


if __name__ == "__main__":
    main()
