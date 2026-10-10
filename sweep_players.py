"""
How should Pokerface play at tables of different sizes?

For every table size (2 up to --max-players, Pokerface included) the same
random tables and game seeds are played by several Pokerface variants. Each
variant is compared with the baseline (the first one) on the same tables, so
the differences are paired and the noise of the tables cancels out.

Opponents are Sonny Boy, Oppa Style, Fabulous and Trihard (--mix core, the
default; numbered copies at tables bigger than 5). Other mixes: see
benchmark_adaptive.py.

Usage:
    uv run python sweep_players.py --tables 60 --games 40
    uv run python sweep_players.py --mix strong --variants "TARGET=0.8" "TARGET=0.9"
    uv run python sweep_players.py --base BOT=CarlosBot --variants "OTHER_WEIGHT=2"
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

from benchmark_adaptive import fresh_seed, MAX_ROUNDS, mean_se, parse_settings, run_table

DEFAULT_VARIANTS = (
    "",                     # baseline: Pokerface as it is
    "TARGET=0.75",          # bolder bids
    "TARGET=0.92",          # more careful bids
    "MIN_ACCEPT=0.5",       # Dudo sooner instead of a weak raise
    "BLUFF_RATE=0",         # never bluff
    "ADAPTIVE=False",       # no learning
)


def main():
    parser = argparse.ArgumentParser(description="Sweep Pokerface variants over table sizes.")
    parser.add_argument("--min-players", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=7)
    parser.add_argument("--tables", type=int, default=60, help="Tables per size.")
    parser.add_argument("--games", type=int, default=40, help="Games per table.")
    parser.add_argument("--seed", type=int, default=None, help="Default: a new random seed per run.")
    parser.add_argument("--mix", choices=("core", "static", "zoo", "repo", "strong"), default="core")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Settings per variant; the baseline (no changes) is always first.")
    parser.add_argument("--base", default="",
                        help="Settings shared by the baseline and every variant, e.g. BOT=CarlosBot.")
    args = parser.parse_args()

    args.seed = fresh_seed(args.seed)
    variants = ["", *(args.variants if args.variants is not None else DEFAULT_VARIANTS[1:])]
    base = parse_settings(args.base)
    settings = [{**base, **parse_settings(v)} for v in variants]
    sizes = range(args.min_players, args.max_players + 1)

    jobs = [(args.seed + 1000 * size, t, size - 1, args.mix, args.games, 0, settings[v])
            for size in sizes for t in range(args.tables) for v in range(len(variants))]
    with Pool(args.jobs) as pool:
        results = pool.map(run_table, jobs)

    # results come back in job order: size -> table -> variant
    aborted = sum(r[-1] for r in results)
    it = iter(results)
    win = {}   # (size, variant) -> per-table win %
    pos = {}   # (size, variant) -> per-table mean position
    for size in sizes:
        for t in range(args.tables):
            for v in range(len(variants)):
                _, _, _, positions, wins, _, _ = next(it)
                win.setdefault((size, v), []).append(100 * sum(wins) / max(len(wins), 1))
                pos.setdefault((size, v), []).append(sum(positions) / max(len(positions), 1))

    names = ["baseline", *variants[1:]]
    print(f"{args.tables} tafels x {args.games} games per grootte, tegenstanders: {args.mix}"
          + (f", basis: {args.base}" if args.base else ""))
    print(f"Afgebroken games (> {MAX_ROUNDS} rondes, niet meegeteld): {aborted}\n")
    print("Winst % per tafelgrootte (verschil met baseline ± standaardfout, gepaard):\n")
    width = max(len(n) for n in names) + 2
    header = f"{'spelers':<9}{'fair':>6}" + "".join(f"{n:>{width + 8}}" for n in names)
    print(header)
    print("-" * len(header))
    for size in sizes:
        base = win[size, 0]
        cells = [f"{mean_se(base)[0]:>{width + 7}.1f}%"]
        for v in range(1, len(variants)):
            d, se = mean_se([a - b for a, b in zip(win[size, v], base)])
            mark = " *" if abs(d) >= 2 * se and se > 0 else "  "
            cells.append(f"{d:>+{width + 2}.1f}±{se:3.1f}{mark}")
        print(f"{size:<9}{100 / size:>5.1f}%" + "".join(cells))
    print("\n* = verschil minstens 2 standaardfouten")

    print("\nGemiddelde eindpositie (lager = beter), verschil met baseline:\n")
    print(header.replace("fair", "    "))
    print("-" * len(header))
    for size in sizes:
        base = pos[size, 0]
        cells = [f"{mean_se(base)[0]:>{width + 8}.2f}"]
        for v in range(1, len(variants)):
            d, se = mean_se([a - b for a, b in zip(pos[size, v], base)])
            mark = " *" if abs(d) >= 2 * se and se > 0 else "  "
            cells.append(f"{d:>+{width + 1}.2f}±{se:3.2f}{mark}")
        print(f"{size:<9}{'':>6}" + "".join(cells))


if __name__ == "__main__":
    main()
