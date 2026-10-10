"""
Final ranking of all bots, with statistics.

Every game gets a random table size (2-7 players), a random draw of
different bots from the pool and fresh bot instances without learned
memory (as in a single game on the platform); the engine shuffles the
seats. Per bot:

  * win rate, and wins relative to a fair share (1 / players);
  * score: the share of opponents it finished ahead of (1 = won, 0 = first
    out), comparable across table sizes; with its standard error;
  * average finishing position;
  * win rate per table size;
  * head-to-head: when two bots sat at the same table, how often the row
    bot finished ahead of the column bot.

Usage:
    uv run python ranking.py --games 1200
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from multiprocessing import Pool

from benchmark_adaptive import MAX_ROUNDS, RoundLimit, TooManyRounds, fresh_seed
from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame

POOL = ("PokerfaceBot", "CarlosBot", "TheDonBot", "Trihard", "SniperBot", "Fabulous",
        "OppaStyleBot", "SonnyBoy", "HundredIQBot")


def play(job):
    seed, games = job
    rng = random.Random(seed)
    classes = discover_bot_classes_by_name()
    results = []
    aborted = 0
    for _ in range(games):
        size = rng.randint(2, 7)
        names = rng.sample(POOL, size)
        for cls in classes.values():
            if hasattr(cls, "reset_memory"):
                cls.reset_memory()
        bots = [classes[n]() for n in names]
        try:
            result = PerudoGame(bots=bots, seed=rng.randrange(2**63), listeners=[RoundLimit()]).play()
        except TooManyRounds:
            aborted += 1
            continue
        results.append(result.standings)
    return results, aborted


def mean_se(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def main():
    parser = argparse.ArgumentParser(description="Rank all bots over random tables of 2-7 players.")
    parser.add_argument("--games", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    seed = fresh_seed(args.seed)

    chunks = 4 * args.jobs
    per = [args.games // chunks + (1 if i < args.games % chunks else 0) for i in range(chunks)]
    with Pool(args.jobs) as pool:
        parts = pool.map(play, [(seed * 1000 + i, n) for i, n in enumerate(per)])
    games = [s for res, _ in parts for s in res]
    aborted = sum(a for _, a in parts)

    score = defaultdict(list)            # bot -> share of opponents beaten, per game
    pos = defaultdict(list)
    wins = defaultdict(int)
    fair = defaultdict(float)
    by_size = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    ahead = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for standings in games:
        n = len(standings)
        for i, bot in enumerate(standings):
            score[bot].append((n - 1 - i) / (n - 1))
            pos[bot].append(i + 1)
            fair[bot] += 1 / n
            by_size[bot][n][1] += 1
            if i == 0:
                wins[bot] += 1
                by_size[bot][n][0] += 1
            for j, other in enumerate(standings):
                if other != bot:
                    ahead[bot][other][0] += i < j
                    ahead[bot][other][1] += 1

    bots = sorted(score, key=lambda b: -mean_se(score[b])[0])
    print(f"{len(games)} games (tables of 2-7 players, fresh bots each game), aborted (> {MAX_ROUNDS} rounds): {aborted}\n")
    print(f"{'#':<3}{'bot':<12}{'games':>6}{'wins':>6}{'win %':>8}{'vs fair':>9}{'score':>15}{'avg pos':>9}")
    print("-" * 68)
    for rank, bot in enumerate(bots, 1):
        g = len(score[bot])
        m, se = mean_se(score[bot])
        print(f"{rank:<3}{bot:<12}{g:>6}{wins[bot]:>6}{100 * wins[bot] / g:>7.1f}%{wins[bot] / fair[bot]:>8.2f}x"
              f"{m:>9.3f} ±{se:.3f}{sum(pos[bot]) / g:>9.2f}")
    print("\nscore = share of opponents finished ahead of (0.5 = average); vs fair = wins / expected wins at 1/players")

    print("\nWin % per table size (fair share in brackets):")
    sizes = range(2, 8)
    print(f"{'bot':<12}" + "".join(f"{f'{s} ({100 / s:.0f}%)':>11}" for s in sizes))
    for bot in bots:
        cells = []
        for s in sizes:
            w, n = by_size[bot][s]
            cells.append(f"{(f'{100 * w / n:.0f}% /{n}') if n else '-':>11}")
        print(f"{bot:<12}" + "".join(cells))

    print("\nHead-to-head: % of shared games where the row bot finished ahead of the column bot")
    short = {b: b[:7] for b in bots}
    print(f"{'':<12}" + "".join(f"{short[b]:>8}" for b in bots))
    for bot in bots:
        row = []
        for other in bots:
            if other == bot:
                row.append(f"{'-':>8}")
            else:
                a, n = ahead[bot][other]
                row.append(f"{100 * a / n:>7.0f}%" if n else f"{'':>8}")
        print(f"{bot:<12}" + "".join(row))


if __name__ == "__main__":
    main()
