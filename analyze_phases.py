"""
How does each bot play in each phase of the game?

Plays random tables (2-7 seats, bots drawn from the pool, copies of the
same bot allowed with --copies) and records every decision with its phase:

  * players alive (2 / 3 / 4+), the bot's own dice (1 / 2 / 3+), the dice
    of the next player in playing order (the one who can challenge the
    bot's bid: 1 / 2 / 3+), Palifico, and the bot's position in the round
    (opening, early = first bid cycle, late);
  * what the bot did and whether it was right, judged with the dice really
    on the table: a bid true / caught, a Dudo right, a Calza exact.

Per bot it prints, per phase: decisions, dice lost against a fair share
(1 / players per round), the share of its bids that were true, how often a
bid was challenged and caught, Dudo rate and accuracy, Calza rate and
accuracy, and for openings the bid quantity relative to the expected count
on the table. Use it to see where a bot's play contradicts its own
strategy, then fix the bot and run again.

Usage:
    uv run python analyze_phases.py --games 300
    uv run python analyze_phases.py --games 200 --bots SniperBot Trihard --copies 3
    uv run python analyze_phases.py --games 300 --by next     # split by the next player's dice
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from multiprocessing import Pool

from benchmark_adaptive import RoundLimit, TooManyRounds, build, fresh_seed
from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame
from perudo.models import Bid, Calza, DieLostEvent, Dudo, RevealEvent, RoundEndedEvent
from perudo.rules import count_matches

POOL = ("PokerfaceBot", "CarlosBot", "CarlosDosBot", "TheDonBot", "BluffoBoyyo", "Trihard", "SniperBot",
        "Fabulous", "OppaStyleBot", "SonnyBoy")


def bucket(n):
    return "1" if n <= 1 else "2" if n == 2 else "3+"


def alive_bucket(n):
    return "2" if n == 2 else "3" if n == 3 else "4+"


class Recorder:
    """Wraps every bot's play() and watches the events, per game."""

    def __init__(self, bots):
        self.rows = []            # per decision
        self.rounds = []          # per round: (alive, lost player)
        self.pending = []
        self.palifico = False
        self.bots = bots
        for bot in bots:
            bot.play = self._wrap(bot, bot.play)

    def _wrap(self, bot, play):
        def wrapped(state):
            action = play(state)
            alive = [p for p in state.players if p.alive]
            n = len(state.players)
            i = (state.my_id + 1) % n
            while not state.players[i].alive:
                i = (i + 1) % n
            nxt = state.players[i]
            bids_so_far = sum(1 for r in state.history if isinstance(r.action, Bid))
            position = "open" if state.current_bid is None else ("early" if bids_so_far < len(alive) else "late")
            total = sum(p.dice_count for p in alive)
            self.pending.append(dict(
                bot=bot.name, base=type(bot).__name__, alive=len(alive), mine=state.my_dice_count,
                nxt=nxt.dice_count, pal=state.palifico, position=position, action=action,
                faced=state.current_bid, total=total, my_dice=state.my_dice,
            ))
            return action
        return wrapped

    def __call__(self, event):
        if isinstance(event, RevealEvent):
            dice = [d for _, hand in event.dice for d in hand]
            for row in self.pending:
                a = row["action"]
                if isinstance(a, Bid):
                    row["kind"] = "bid"
                    row["right"] = count_matches(dice, a.face, row["pal"]) >= a.quantity
                    row["caught"] = (event.bidder == row["bot"] and event.challenge_type == "dudo"
                                     and event.current_bid == a and event.result == "bidder_wrong")
                    row["challenged"] = event.bidder == row["bot"] and event.current_bid == a
                    if row["position"] == "open":
                        p = 1 / 6 if (row["pal"] or a.face == 1) else 1 / 3
                        expected = count_matches(row["my_dice"], a.face, row["pal"]) + (row["total"] - row["mine"]) * p
                        row["open_ratio"] = a.quantity / max(expected, 0.5)
                else:
                    f = row["faced"]
                    actual = count_matches(dice, f.face, row["pal"])
                    row["kind"] = "dudo" if isinstance(a, Dudo) else "calza"
                    row["right"] = actual < f.quantity if isinstance(a, Dudo) else actual == f.quantity
                self.rows.append(row)
            self.pending = []
            self.cur_alive = len(event.dice)
            self.lost = None
        elif isinstance(event, DieLostEvent):
            self.lost = event.player
        elif isinstance(event, RoundEndedEvent):
            self.rounds.append((self.cur_alive, [b.name for b in self.bots if b.name in self._alive_names()], self.lost))
            self.palifico = event.palifico_next

    def _alive_names(self):
        return self._names

    cur_alive = 0
    lost = None
    _names = ()


def play(job):
    seed, games, pool, copies = job
    rng = random.Random(seed)
    classes = discover_bot_classes_by_name()
    rows, rounds, aborted = [], [], 0
    for _ in range(games):
        for cls in classes.values():
            if hasattr(cls, "reset_memory"):
                cls.reset_memory()
        size = rng.randint(2, 7)
        if copies > 1 and rng.random() < 0.5:
            base = rng.choice(pool)
            k = min(size, copies)
            names = [base] * k + rng.sample([p for p in pool if p != base], min(size - k, len(pool) - 1))
            names = names[:size]
        else:
            names = rng.sample(pool, min(size, len(pool)))
        bots = build([("repo", n) for n in names], classes)
        rng.shuffle(bots)
        rec = Recorder(bots)
        alive = {b.name for b in bots}
        rec._names = alive

        def track(event, rec=rec, alive=alive):
            if isinstance(event, RevealEvent):
                alive.intersection_update({n for n, _ in event.dice})
            rec(event)

        try:
            PerudoGame(bots=bots, seed=rng.randrange(2**63), listeners=[RoundLimit(), track]).play()
        except TooManyRounds:
            aborted += 1
            continue
        rows.extend(rec.rows)
        rounds.extend(rec.rounds)
    return rows, rounds, aborted


def main():
    parser = argparse.ArgumentParser(description="Per-phase play analysis of every bot.")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bots", nargs="*", default=None, help="Class names (default: the pool).")
    parser.add_argument("--copies", type=int, default=1, help="Max copies of one bot at a table (half the tables).")
    parser.add_argument("--by", choices=("alive", "mine", "next", "position", "pal", "all"), default="all",
                        help="Which phase dimension to split the table on (all: one table each).")
    parser.add_argument("--min", type=int, default=30, help="Hide phases with fewer decisions.")
    args = parser.parse_args()
    seed = fresh_seed(args.seed)
    pool = tuple(args.bots) if args.bots else POOL
    chunks = 4 * args.jobs
    with Pool(args.jobs) as p:
        parts = p.map(play, [(seed * 100 + i, args.games // chunks + (i < args.games % chunks), pool, args.copies)
                             for i in range(chunks)])
    rows = [r for part in parts for r in part[0]]
    rounds = [r for part in parts for r in part[1]]
    aborted = sum(part[2] for part in parts)
    print(f"{len(rows)} decisions in {len(rounds)} rounds; aborted games (> 1000 rounds): {aborted}")

    for by in (("alive", "mine", "next", "position", "pal") if args.by == "all" else (args.by,)):
        report(rows, rounds, by, args.min)
    print("\nlose/fair: dice lost per round / (1/players), lower is better (only split by players alive);"
          "\nbid true: share of the bot's bids that were true; caught: share of its bids caught by a Dudo;"
          "\nopen q/E: opening quantity over the expected count of that face (own dice + chance).")


def report(rows, rounds, by, min_n):
    class A:
        pass
    args = A()
    args.by, args.min = by, min_n

    def phase(row):
        return {"alive": alive_bucket(row["alive"]), "mine": bucket(row["mine"]), "next": bucket(row["nxt"]),
                "position": row["position"], "pal": "palifico" if row["pal"] else "normal"}[args.by]

    # dice lost vs fair share, per bot base and phase (phase of the round = players alive)
    lost = defaultdict(lambda: [0, 0.0])
    if args.by == "alive":
        for alive, names, loser in rounds:
            for name in names:
                base = name.rstrip(" 23456789").strip()
                key = (base, alive_bucket(alive))
                lost[key][0] += loser == name
                lost[key][1] += 1 / alive

    stats = defaultdict(lambda: defaultdict(float))
    for r in rows:
        s = stats[(r["bot"].rstrip(" 23456789").strip(), phase(r))]
        s["n"] += 1
        if r["kind"] == "bid":
            s["bids"] += 1
            s["bid_true"] += r["right"]
            s["challenged"] += r["challenged"]
            s["caught"] += r["caught"]
            if "open_ratio" in r:
                s["opens"] += 1
                s["open_ratio"] += r["open_ratio"]
        elif r["kind"] == "dudo":
            s["dudos"] += 1
            s["dudo_right"] += r["right"]
        else:
            s["calzas"] += 1
            s["calza_right"] += r["right"]

    print()
    header = (f"{'bot':<13}{args.by:<10}{'n':>6}{'lose/fair':>10}{'bid true':>9}{'challngd':>9}{'caught':>8}"
              f"{'dudo%':>7}{'dudoOK':>7}{'calza%':>7}{'calzaOK':>8}{'open q/E':>9}")
    print(header)
    print("-" * len(header))
    for (bot, ph) in sorted(stats, key=lambda k: (k[0], k[1])):
        s = stats[(bot, ph)]
        if s["n"] < args.min:
            continue
        lf = lost.get((bot, ph))
        lf_txt = f"{lf[0] / lf[1]:>10.2f}" if lf and lf[1] else f"{'':>10}"
        print(f"{bot:<13}{ph:<10}{int(s['n']):>6}{lf_txt}"
              f"{s['bid_true'] / max(s['bids'], 1):>9.2f}{s['challenged'] / max(s['bids'], 1):>9.2f}"
              f"{s['caught'] / max(s['bids'], 1):>8.3f}{s['dudos'] / s['n']:>7.2f}"
              f"{s['dudo_right'] / max(s['dudos'], 1):>7.2f}{s['calzas'] / s['n']:>7.3f}"
              f"{s['calza_right'] / max(s['calzas'], 1):>8.2f}"
              f"{(s['open_ratio'] / s['opens']) if s['opens'] else float('nan'):>9.2f}")


if __name__ == "__main__":
    main()
