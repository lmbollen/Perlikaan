"""
Does learning help against opponents we have never seen?

Every table seats Pokerface with randomly generated opponents from zoo.py
(random face choice, aggressiveness, Dudo threshold, table reading, Calza).
The same table and the same game seeds are played twice: once with the
learning switched on (ADAPTIVE) and once with it off (static). The paired
difference per table is what learning is worth.

It also checks what was learned against the known knobs of each zoo bot:
  * learned value of a face choice (bid_excess "nat-choice") per face style:
    honest should come out high, bluff below zero;
  * learned challenge rate on false bids versus the bot's Dudo threshold:
    a higher threshold should mean more challenges.

Usage:
    uv run python benchmark_adaptive.py --tables 40 --games 60
    uv run python benchmark_adaptive.py --mix repo      # also seat repo bots
    uv run python benchmark_adaptive.py --mix strong    # Sniper, Trihard, Oppa at every table
    uv run python benchmark_adaptive.py --a "BLUFF_RATE=0" --b "BLUFF_RATE=0.5"

--a / --b compare any two settings of Pokerface's class attributes (default:
ADAPTIVE=False against ADAPTIVE=True). BOT=<class> plays another bot in our
seat, e.g. --a "BOT=PokerfaceBot" --b "BOT=CarlosBot".
"""

from __future__ import annotations

import argparse
import ast
import math
import random
from collections import defaultdict
from multiprocessing import Pool

from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame
from perudo.models import RevealEvent
from run_simulation import derive_game_seeds
from zoo import ZooBot, random_zoo_bot

REPO_BOTS = ("SniperBot", "Trihard", "Fabulous", "OppaStyleBot", "SonnyBoy", "HundredIQBot")
# The opponents that matter: --mix core seats only these (the default).
CORE_BOTS = ("SonnyBoy", "OppaStyleBot", "Fabulous", "Trihard")
# The strongest bots in the repo; --mix strong seats all three at every table
# (a random subset at smaller tables).
STRONG_BOTS = ("SniperBot", "Trihard", "OppaStyleBot")


def table_spec(seed, table, n_opponents, mix):
    """The opponents of one table, as rebuildable specs."""
    rng = random.Random(seed * 100_003 + table)
    if mix == "core":
        # Every core bot once, in random order; at bigger tables extra
        # copies of random core bots (they get a numbered name).
        picks = rng.sample(CORE_BOTS, min(len(CORE_BOTS), n_opponents))
        picks += rng.choices(CORE_BOTS, k=n_opponents - len(picks))
        return [("repo", name) for name in picks]
    specs = []
    # With fewer seats than strong bots, a random subset of them.
    strong = rng.sample(STRONG_BOTS, min(len(STRONG_BOTS), n_opponents)) if mix == "strong" else []
    for i in range(n_opponents):
        if i < len(strong):
            specs.append(("repo", strong[i]))
        elif mix == "repo" and i < 2:
            specs.append(("repo", rng.choice(REPO_BOTS)))
        else:
            bot = random_zoo_bot(rng, f"Zoo {i + 1}")
            specs.append(("zoo", dict(name=bot.name, dudo_at=bot.dudo_at, faces=bot.faces,
                                      bid_at=bot.bid_at, reads=bot.reads, calza_at=bot.calza_at,
                                      seed=rng.randrange(2**32))))
    return specs


def build(specs, classes):
    bots, seen = [], set()
    for kind, spec in specs:
        if kind == "zoo":
            bots.append(ZooBot(**spec))
        else:
            bot = classes[spec]()
            base, copy = bot.name, 2
            while bot.name in seen:
                # Same bot twice: names must be unique.
                bot.name = f"{base} {copy}"
                copy += 1
            bots.append(bot)
        seen.add(bots[-1].name)
    return bots


# A game can in principle go on forever: once someone is out, a correct
# Calza costs nobody a die, so bots that keep calling correct Calzas never
# finish. Such games are stopped and left out of the results.
MAX_ROUNDS = 1000


class TooManyRounds(Exception):
    pass


class RoundLimit:
    def __init__(self):
        self.rounds = 0

    def __call__(self, event):
        if isinstance(event, RevealEvent):
            self.rounds += 1
            if self.rounds > MAX_ROUNDS:
                raise TooManyRounds


def run_table(job):
    seed, table, n_opponents, mix, games, variant, settings = job
    classes = discover_bot_classes_by_name()
    specs = table_spec(seed, table, n_opponents, mix)
    settings = dict(settings)
    me = classes[settings.pop("BOT", "PokerfaceBot")]()
    for key, value in settings.items():
        setattr(me, key, value)
    if hasattr(me, "_rng"):
        me._rng = random.Random(table)
    bots = [me, *build(specs, classes)]

    positions, wins, aborted = [], [], 0
    for game_seed in derive_game_seeds(seed * 7 + table, games):
        try:
            result = PerudoGame(bots=bots, seed=game_seed, listeners=[RoundLimit()]).play()
        except TooManyRounds:
            aborted += 1
            continue
        positions.append(result.standings.index(me.name) + 1)
        wins.append(result.winner == me.name)

    learned = []
    if variant and hasattr(me, "profiler"):
        prof = me.profiler
        for kind, spec in specs:
            if kind != "zoo" or spec["name"] not in prof.profiles:
                continue
            learned.append(dict(
                faces=spec["faces"],
                dudo_at=spec["dudo_at"],
                choice=prof.bid_excess(spec["name"], "nat-choice"),
                n_choice=prof.profiles[spec["name"]].excess.get("nat-choice", (0, 0))[1],
                challenge=prof.challenge_rate(spec["name"]),
            ))
    return table, variant, len(bots), positions, wins, learned, aborted


def parse_settings(text):
    """"A=1,B=False" -> {"A": 1, "B": False}"""
    out = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        key, value = part.split("=", 1)
        try:
            out[key.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            out[key.strip()] = value.strip()   # a bare word, e.g. BOT=CarlosBot
    return out


def mean_se(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var / len(xs))


def main():
    parser = argparse.ArgumentParser(description="Benchmark Pokerface's learning against unseen opponents.")
    parser.add_argument("--tables", type=int, default=40)
    parser.add_argument("--games", type=int, default=60, help="Games per table (learning carries over).")
    parser.add_argument("--opponents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mix", choices=("core", "zoo", "repo", "strong"), default="core",
                        help="core: only Sonny Boy, Oppa Style, Fabulous and Trihard (copies at "
                             "big tables); zoo: only generated opponents; repo: 2 seats go to "
                             "random repo bots; strong: Sniper, Trihard and Oppa Style, rest zoo.")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--a", default="ADAPTIVE=False", help="Settings of variant A.")
    parser.add_argument("--b", default="ADAPTIVE=True", help="Settings of variant B.")
    args = parser.parse_args()

    variants = (parse_settings(args.a), parse_settings(args.b))
    jobs = [(args.seed, t, args.opponents, args.mix, args.games, v, variants[v])
            for t in range(args.tables) for v in (0, 1)]
    with Pool(args.jobs) as pool:
        results = pool.map(run_table, jobs)

    by = {(t, a): (n, pos, win, learned) for t, a, n, pos, win, learned, _ in results}
    aborted = sum(r[-1] for r in results)
    third = max(args.games // 3, 1)

    def stats(v, sl=slice(None)):
        win = [100 * sum(by[t, v][2][sl]) / max(len(by[t, v][2][sl]), 1) for t in range(args.tables)]
        pos = [sum(by[t, v][1][sl]) / max(len(by[t, v][1][sl]), 1) for t in range(args.tables)]
        return win, pos

    seats = sum(by[t, 1][0] for t in range(args.tables)) / args.tables
    print(f"{args.tables} tafels x {args.games} games, gem. {seats:.1f} spelers "
          f"(fair winst-% {100 / seats:.1f}%), tegenstanders: {args.mix}")
    print(f"A = {args.a}\nB = {args.b}")
    print(f"Afgebroken games (> {MAX_ROUNDS} rondes, niet meegeteld): {aborted}\n")
    print(f"{'':<22}{'A':>14}{'B':>14}{'B - A':>20}")
    for label, sl in (("alle games", slice(None)),
                      (f"eerste {third}", slice(0, third)),
                      (f"laatste {third}", slice(-third, None))):
        ws, ps = stats(0, sl)
        wa, pa = stats(1, sl)
        dw = mean_se([a - s for a, s in zip(wa, ws)])
        dp = mean_se([a - s for a, s in zip(pa, ps)])
        print(f"{'winst % ' + label:<22}{mean_se(ws)[0]:>13.1f}%{mean_se(wa)[0]:>13.1f}%"
              f"{dw[0]:>+12.1f} ± {dw[1]:.1f}")
        print(f"{'positie ' + label:<22}{mean_se(ps)[0]:>14.2f}{mean_se(pa)[0]:>14.2f}"
              f"{dp[0]:>+12.2f} ± {dp[1]:.2f}  (lager = beter)")

    learned = [row for t in range(args.tables) for row in by[t, 1][3]]
    if not learned:
        return
    print("\nGeleerd door B vs werkelijk (zoo-tegenstanders):")
    print(f"  {'face-stijl':<10}{'n':>5}{'geleerde extra stenen bij gekozen gezicht':>44}")
    groups = defaultdict(list)
    for row in learned:
        if row["n_choice"] >= 10:
            groups[row["faces"]].append(row["choice"])
    for style in ("honest", "mixed", "random", "bluff"):
        if groups[style]:
            m, se = mean_se(groups[style])
            print(f"  {style:<10}{len(groups[style]):>5}{m:>+38.2f} ± {se:.2f}")

    print(f"\n  {'Dudo-drempel':<14}{'n':>5}{'geleerde kans op Dudo bij vals bod':>38}")
    buckets = defaultdict(list)
    for row in learned:
        buckets[min(int(row["dudo_at"] * 10) / 10, 0.6)].append(row["challenge"])
    for b in sorted(buckets):
        m, se = mean_se(buckets[b])
        print(f"  {f'{b:.1f}-{b + 0.1:.1f}':<14}{len(buckets[b]):>5}{m:>32.2f} ± {se:.2f}")


if __name__ == "__main__":
    main()
