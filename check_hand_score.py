"""
Check perudo/hand_score.py against play: does the formula rank hands the
way they turn out?

Every hand type (how many Pacos, how the other dice group; faces picked at
random) is dealt on purpose to one seat in the first round of a game, the
same games for every type, so rare hands (five Pacos: 1 in 7776) are
measured as precisely as common ones. Per type: how often that seat loses
a die in the round, next to the formula's value; then how well the
formula's order agrees with what happened (Spearman, 1 = the same order).

    PYTHONHASHSEED=0 uv run python check_hand_score.py --games 800
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from multiprocessing import Pool

from benchmark_adaptive import fresh_seed
from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame
from perudo.hand_score import value
from perudo.models import DieLostEvent, RoundEndedEvent

SETTINGS = ((2, 5), (4, 5), (6, 5), (4, 3), (4, 2))      # (players, dice of the seat)


def partitions(n, most=None):
    most = n if most is None else most
    if n == 0:
        yield ()
        return
    for k in range(min(n, most), 0, -1):
        for rest in partitions(n - k, k):
            yield (k,) + rest

def types(n):
    return [(p, part) for p in range(n, -1, -1) for part in partitions(n - p) if len(part) <= 5]

def deal(t, rng):
    pacos, part = t
    faces = rng.sample(range(2, 7), len(part))
    return [1] * pacos + [f for f, k in zip(faces, part) for _ in range(k)]

class Done(Exception):
    pass

def job(args):
    seed, players, n_dice, games = args
    classes = discover_bot_classes_by_name()
    names = sorted(classes)
    rng = random.Random(seed)
    T = types(n_dice)
    out = defaultdict(lambda: [0, 0])
    real_roll = PerudoGame._roll_round
    for _ in range(games):
        picks = rng.choices(names, k=players)
        game_seed, seat, face_seed = rng.randrange(2**63), rng.randrange(players), rng.randrange(2**32)
        for t in [None] + T:
            bots, seen = [], set()
            for name in picks:
                b = classes[name]()
                base, c = b.name, 2
                while b.name in seen:
                    b.name = f"{base} {c}"
                    c += 1
                seen.add(b.name)
                bots.append(b)
            for b in bots:
                for attr in ("_rng", "rng"):
                    if isinstance(getattr(b, attr, None), random.Random):
                        setattr(b, attr, random.Random(game_seed ^ (hash(b.name) & 0xFFFF)))
            target = bots[seat].name
            hand = deal(t, random.Random(face_seed)) if t else None
            lost = [False]
            def roll(game, hand=hand, target=target):
                real_roll(game)
                me = next(p for p in game.players if p.name == target)
                if n_dice < 5:                      # this seat has fewer dice than the rest
                    me.dice_count = n_dice
                    me.dice = me.dice[:n_dice]
                if hand:
                    me.dice = list(hand)
            def listen(e, target=target):
                if isinstance(e, DieLostEvent) and e.player == target:
                    lost[0] = True
                if isinstance(e, RoundEndedEvent):
                    raise Done
            PerudoGame._roll_round = roll
            try:
                PerudoGame(bots=bots, seed=game_seed, listeners=[listen]).play()
            except Done:
                pass
            finally:
                PerudoGame._roll_round = real_roll
            out[t][0] += 1
            out[t][1] += lost[0]
    return dict(out)

def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    for r, i in enumerate(order):
        out[i] = float(r)
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    var = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return cov / var if var else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=800, help="games per setting (every type plays each)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    seed = fresh_seed(args.seed)
    for players, n_dice in SETTINGS:
        jobs = [(seed * 31 + players * 1000 + n_dice * 7 + i, players, n_dice, args.games // args.workers)
                for i in range(args.workers)]
        with Pool(args.workers) as pool:
            parts = pool.map(job, jobs)
        tot = defaultdict(lambda: [0, 0])
        for part in parts:
            for t, (n, lost) in part.items():
                tot[t][0] += n
                tot[t][1] += lost
        normal = tot.pop(None)
        rows = sorted(((value(deal(t, random.Random(0))), lost / n, t) for t, (n, lost) in tot.items()),
                      reverse=True)
        print(f"\n{players} players, {n_dice} dice: a normal hand loses a die {normal[1] / normal[0]:.1%} "
              f"({normal[0]} games per type)")
        for v, lose, t in rows:
            pacos, part = t
            shape = "P" * pacos + "".join("abcde"[i] * k for i, k in enumerate(part))
            print(f"  {shape:6} value {v:+.2f}   loses a die {lose:6.1%}")
        agreement = spearman([-v for v, _, _ in rows], [lose for _, lose, _ in rows])
        print(f"  order agreement (Spearman): {agreement:.2f}")


if __name__ == "__main__":
    main()
