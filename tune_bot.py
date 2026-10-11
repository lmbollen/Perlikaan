"""
Paired benchmark of settings for any bot.

Every variant plays the same games: the same table (random size 2-7, the
bot plus random opponents from the pool, random seat), the same dice and
the same opponents' randomness, so the difference between variants is
paired and the noise of the tables cancels out. Bots start every game
without learned memory, as in a single game on the platform.

A variant is a list of class attributes to set on the bot, separated by
";" (values are Python literals). Module-level constants of the bot's file
can be set the same way: they are looked up in the bot's module when the
class has no such attribute.

Usage:
    uv run python tune_bot.py --bot Fabulous --games 480 "DUDO_THRESHOLD=0.45" "CHALLENGE_FACTOR=0.7"
    uv run python tune_bot.py --bot SonnyBoy "DUDO_AT=0.4; CALZA_MIN=0.6"
"""

from __future__ import annotations

import argparse
import ast
import math
import random
from multiprocessing import Pool

from benchmark_adaptive import RoundLimit, TooManyRounds, fresh_seed
from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame
from perudo.models import DieLostEvent, RevealEvent

POOL = ("PokerfaceBot", "CarlosBot", "TheDonBot", "BluffoBoyyo", "Trihard", "SniperBot",
        "Fabulous", "OppaStyleBot", "SonnyBoy")


def parse_settings(text):
    out = {}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        key, value = part.split("=", 1)
        out[key.strip()] = ast.literal_eval(value.strip())
    return out


def apply(cls, settings):
    for key, value in settings.items():
        if hasattr(cls, key):
            setattr(cls, key, value)
        else:
            cls.play.__globals__[key] = value       # a module-level constant


class Loss:
    def __init__(self, name):
        self.name, self.lost, self.fair = name, 0, 0.0

    def __call__(self, e):
        if isinstance(e, RevealEvent):
            self.fair += 1 / len(e.dice)
        elif isinstance(e, DieLostEvent) and e.player == self.name:
            self.lost += 1


def job(a):
    seed, games, bot_name, settings, pool = a
    rng = random.Random(seed)
    classes = discover_bot_classes_by_name()
    cls = classes[bot_name]
    apply(cls, settings)
    me_name = cls().name
    out = []
    for _ in range(games):
        size = rng.randint(2, 7)
        opp = rng.sample([p for p in pool if p != bot_name], size - 1)
        gseed = rng.randrange(2**63)
        seat = rng.randrange(size)
        for c in classes.values():
            if hasattr(c, "reset_memory"):
                c.reset_memory()
        bots = [classes[n]() for n in opp]
        me = cls()
        bots.insert(seat, me)
        for i, b in enumerate(bots):                 # the bots' own randomness, paired too
            for attr in ("_rng", "rng"):
                if hasattr(b, attr):
                    setattr(b, attr, random.Random(gseed * 10 + i))
        loss = Loss(me_name)
        try:
            res = PerudoGame(bots=bots, seed=gseed, listeners=[RoundLimit(), loss]).play()
        except TooManyRounds:
            out.append(None)
            continue
        pos = res.standings.index(me_name)
        out.append((size, pos == 0, (size - 1 - pos) / (size - 1), loss.lost, loss.fair))
    return out


def mean_se(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1) / len(xs))


def main():
    parser = argparse.ArgumentParser(description="Paired benchmark of settings for one bot.")
    parser.add_argument("--bot", required=True, help="Class name, e.g. Fabulous.")
    parser.add_argument("--games", type=int, default=480)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--pool", nargs="*", default=None, help="Opponent class names (default: all).")
    parser.add_argument("variants", nargs="*", help='Settings per variant, e.g. "A=1; B=0.5".')
    args = parser.parse_args()
    seed = fresh_seed(args.seed)
    pool = tuple(args.pool) if args.pool else POOL
    variants = [""] + args.variants
    chunks = 16
    jobs = [(seed * 100 + c, args.games // chunks, args.bot, parse_settings(v), pool)
            for v in variants for c in range(chunks)]
    with Pool(args.jobs) as p:
        res = p.map(job, jobs)
    per = [sum(res[i * chunks:(i + 1) * chunks], []) for i in range(len(variants))]
    ok = [i for i in range(len(per[0])) if all(v[i] is not None for v in per)]
    base = per[0]
    print(f"{args.bot}: {len(ok)} paired games, tables of 2-7")
    print(f"{'variant':<44}{'win %':>8}{'score':>8}{'lose/fair':>10}{'d win':>14}{'d score':>16}")
    for v, p in zip(variants, per):
        w = [100 * p[i][1] for i in ok]
        sc = [p[i][2] for i in ok]
        lf = sum(p[i][3] for i in ok) / max(sum(p[i][4] for i in ok), 1e-9)
        dw = mean_se([100 * (p[i][1] - base[i][1]) for i in ok])
        ds = mean_se([p[i][2] - base[i][2] for i in ok])
        print(f"{v or 'baseline':<44}{mean_se(w)[0]:>7.1f}%{mean_se(sc)[0]:>8.3f}{lf:>10.3f}"
              f"{dw[0]:>+9.1f}±{dw[1]:.1f}{ds[0]:>+10.3f}±{ds[1]:.3f}", flush=True)
    print("score = share of opponents finished ahead of; lose/fair = dice lost per round / (1/players)")


if __name__ == "__main__":
    main()
