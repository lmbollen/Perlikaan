"""
Does Pokerface's bluffing actually mislead the opponents?

Plays the same tables (Pokerface + Sonny Boy, Oppa Style, Fabulous, Trihard)
with several BLUFF_RATE settings and measures, per setting:

  * how often Pokerface really bluffs: bids on a normal face it holds no
    natural die of, split into opening, following (same face as the bid
    before) and switching to a face;
  * whether the bait is taken: after a bluff of ours on face f, does an
    opponent bid on f later in the round, and who loses the die then;
  * what the opponents learned about us: their own per-opponent trust
    numbers for "Pokerface" (Fabulous: chance a bidder holds the face, Oppa
    Style: surplus of the face, Trihard: honesty exponent, Sonny Boy: share
    of challenged bids that held), next to what they believe of the others;
  * the bottom line: Pokerface's win rate and dice lost per round played.

Usage:
    uv run python diagnose_bluff.py --tables 20 --games 30 --rates 0 0.5 1
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from multiprocessing import Pool

from benchmark_adaptive import RoundLimit, TooManyRounds, build, mean_se, table_spec
from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame
from perudo.models import PACO, BidEvent, CalzaEvent, DieLostEvent, DudoEvent, RevealEvent, RoundEndedEvent
from run_simulation import derive_game_seeds

ME = "Pokerface"


class RoundWatcher:
    """Collects our bids, who followed them and who lost, per round."""

    def __init__(self):
        self.bids = Counter()        # kind -> count, kind in honest/bluff-open/bluff-follow/bluff-switch
        self.bait = defaultdict(Counter)   # kind -> outcome counter
        self.rounds = 0
        self.lost = 0
        self._reset()

    def _reset(self):
        self.actions = []
        self.reveal = None
        self.loser = None

    def __call__(self, event):
        if isinstance(event, BidEvent):
            self.actions.append((event.player, event.bid))
        elif isinstance(event, (DudoEvent, CalzaEvent)):
            self.actions.append((event.player, None))
        elif isinstance(event, RevealEvent):
            self.reveal = event
        elif isinstance(event, DieLostEvent):
            self.loser = event.player
        elif isinstance(event, RoundEndedEvent):
            self._flush()
            self._reset()

    def finish(self):
        if self.reveal is not None:
            self._flush()
        self._reset()

    def _flush(self):
        dice = self.reveal.dice_by_player()
        if ME not in dice:
            return
        self.rounds += 1
        self.lost += self.loser == ME
        mine = dice[ME]
        prev = None
        for i, (player, bid) in enumerate(self.actions):
            if bid is None:
                break
            if player == ME:
                if bid.face != PACO and bid.face not in mine:
                    kind = "bluff-open" if prev is None else (
                        "bluff-follow" if prev.face == bid.face else "bluff-switch")
                else:
                    kind = "honest-open" if prev is None else "honest"
                self.bids[kind] += 1
                # Did an opponent bid on this face after us?
                later = [p for p, b in self.actions[i + 1:] if b is not None and p != ME and b.face == bid.face]
                if not later:
                    outcome = "not followed"
                elif self.loser == ME:
                    outcome = "followed, we lost"
                elif self.loser in later:
                    outcome = "followed, a follower lost"
                else:
                    outcome = "followed, someone else lost"
                self.bait[kind][outcome] += 1
            prev = bid


def trust_in(bots, name):
    """Each core bot's own learned number about `name` (None when unknown)."""
    out = {}
    for bot in bots:
        if bot.name == name:
            continue
        cls = type(bot).__name__
        if cls == "Fabulous":
            out[bot.name] = ("P(holds face)", bot._trust(name))
        elif cls == "OppaStyleBot":
            out[bot.name] = ("surplus", bot._trust(name))
        elif cls == "Trihard":
            out[bot.name] = ("honesty beta", bot._honesty(name))
        elif cls == "SonnyBoy":
            stats = bot.bluff_stats.get(name)
            out[bot.name] = ("held when challenged", stats[0] / stats[1] if stats and stats[1] else None)
    return out


def run(job):
    seed, table, n_opp, games, rate = job
    classes = discover_bot_classes_by_name()
    me = classes["PokerfaceBot"]()
    me.BLUFF_RATE = rate
    me._rng = random.Random(table)
    bots = [me, *build(table_spec(seed, table, n_opp, "core"), classes)]
    watch = RoundWatcher()
    wins = 0
    played = 0
    for game_seed in derive_game_seeds(seed * 7 + table, games):
        try:
            result = PerudoGame(bots=bots, seed=game_seed, listeners=[watch, RoundLimit()]).play()
        except TooManyRounds:
            watch._reset()
            continue
        watch.finish()
        played += 1
        wins += result.winner == ME
    # What they believe of us, and on average of the other core bots.
    of_us = trust_in(bots, ME)
    of_others = defaultdict(list)
    for other in bots[1:]:
        for who, (label, value) in trust_in(bots, other.name).items():
            if value is not None:
                of_others[who].append(value)
    return dict(rate=rate, wins=wins, played=played, rounds=watch.rounds, lost=watch.lost,
                bids=watch.bids, bait=watch.bait, of_us=of_us,
                of_others={k: sum(v) / len(v) for k, v in of_others.items()})


def main():
    parser = argparse.ArgumentParser(description="Check whether Pokerface's bluffs mislead the core bots.")
    parser.add_argument("--tables", type=int, default=20)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    jobs = [(args.seed, t, args.players - 1, args.games, r) for r in args.rates for t in range(args.tables)]
    with Pool(args.jobs) as pool:
        results = pool.map(run, jobs)

    by_rate = defaultdict(list)
    for r in results:
        by_rate[r["rate"]].append(r)

    print(f"{args.tables} tafels x {args.games} games, {args.players} spelers, tegenstanders: core\n")
    base = by_rate[args.rates[0]]
    for rate in args.rates:
        rs = by_rate[rate]
        bids = sum((r["bids"] for r in rs), Counter())
        total_bids = sum(bids.values())
        rounds = sum(r["rounds"] for r in rs)
        lost = sum(r["lost"] for r in rs)
        win = [100 * r["wins"] / max(r["played"], 1) for r in rs]
        diff = mean_se([a - b for a, b in zip(win, [100 * r["wins"] / max(r["played"], 1) for r in base])])
        print(f"=== BLUFF_RATE={rate}")
        print(f"  winst {mean_se(win)[0]:.1f}%  (verschil met BLUFF_RATE={args.rates[0]}: "
              f"{diff[0]:+.1f} ± {diff[1]:.1f} pp)   steen verloren in {100 * lost / rounds:.1f}% van de rondes")
        print(f"  onze biedingen: {total_bids}")
        for kind in ("honest-open", "honest", "bluff-open", "bluff-switch", "bluff-follow"):
            n = bids[kind]
            bait = sum((r["bait"][kind] for r in rs), Counter())
            followed = n - bait["not followed"]
            line = f"    {kind:<13}{n:6d} ({100 * n / max(total_bids, 1):4.1f}%)"
            if n:
                line += (f"   gevolgd {100 * followed / n:4.1f}%"
                         f" -> wij verloren {bait['followed, we lost']}, volger verloor "
                         f"{bait['followed, a follower lost']}, ander {bait['followed, someone else lost']}")
            print(line)
        print("  wat de tegenstanders van ons geleerd hebben (gemiddeld over tafels):")
        labels, ours, others = {}, defaultdict(list), defaultdict(list)
        for r in rs:
            for who, (label, value) in r["of_us"].items():
                key = who.rstrip(" 0123456789")
                labels[key] = label
                if value is not None:
                    ours[key].append(value)
                if who in r["of_others"]:
                    others[key].append(r["of_others"][who])
        for key in sorted(labels):
            o = sum(ours[key]) / len(ours[key]) if ours[key] else float("nan")
            t = sum(others[key]) / len(others[key]) if others[key] else float("nan")
            print(f"    {key:<11} {labels[key]:<21} over ons {o:5.2f}   over de andere bots {t:5.2f}")
        print()


if __name__ == "__main__":
    main()
