"""
What does each factor in Carlos' Markov chain contribute?

Carlos first plays --warmup games against the static bots so his models have
learned something. Then, in --games analysed games, every decision of his is
re-evaluated from the same position and the same learned knowledge with one
factor switched off at a time. After the round all dice are known, so each
choice can be judged:

  Dudo   right if the bid was false
  Calza  right if the bid was exact
  bid    safe if the bid is true (a false bid only costs a die when it gets
         challenged, so this judges the move, not the final outcome)

Per factor: how often switching it off changes the decision, and in those
cases how often the full model vs the model without the factor was right.

Usage:
    uv run python analyze_factors.py --games 10 --warmup 30 --players 5
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

from benchmark_adaptive import RoundLimit, TooManyRounds, build, fresh_seed, table_spec
from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame
from perudo.models import PACO, Bid, Calza, Dudo, RevealEvent
from perudo.rules import count_matches
from run_simulation import derive_game_seeds

FACTORS = [
    ("hand inference from bids", {"USE_BID_INFO": False}),
    ("passed-over faces", {"USE_PASSED_OVER": False}),
    ("follow-bid prior", {"FOLLOW_PRIOR_SHIFT": 0.0}),
    ("separate Paco kinds", {"PACO_KINDS": False}),
    ("learned Dudo model", {"USE_DUDO_MODEL": False}),
    ("several raises per turn", {"RAISE_BRANCHES": 1}),
    ("first-response calibration", {"CALIBRATE_FIRST_RESPONSE": False}),
    ("later-response calibration", {"CALIBRATE_ALL_RESPONSES": False}),
    ("facing-bid calibration", {"CALIBRATE_FACING": False}),
    ("Calza free pass", {"CALZA_FREE_PASS": 0.0}),
    ("2 laps head-to-head", {"LAPS_HEADS_UP": 1}),
]


def kind(action):
    return "Dudo" if isinstance(action, Dudo) else "Calza" if isinstance(action, Calza) else "bid"


def fmt(action):
    if isinstance(action, Bid):
        return f"{action.quantity}x{'P' if action.face == PACO else action.face}"
    return kind(action).upper()


def right(action, current, dice, palifico):
    """Was this move right, given every die on the table?"""
    if isinstance(action, Dudo):
        return count_matches(dice, current.face, palifico) < current.quantity
    if isinstance(action, Calza):
        return count_matches(dice, current.face, palifico) == current.quantity
    return count_matches(dice, action.face, palifico) >= action.quantity


def main():
    parser = argparse.ArgumentParser(description="Measure each factor in Carlos' Markov chain.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--mix", default="static")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    seed = fresh_seed(args.seed)

    classes = discover_bot_classes_by_name()
    carlos = classes["CarlosBot"]()
    carlos._rng = random.Random(seed)
    bots = [carlos, *build(table_spec(seed, 0, args.players - 1, args.mix), classes, taken={carlos.name})]
    print(f"Table: {', '.join(b.name for b in bots)}")

    seeds = derive_game_seeds(seed, args.warmup + args.games)
    for s in seeds[:args.warmup]:
        try:
            PerudoGame(bots=bots, seed=s, listeners=[RoundLimit()]).play()
        except TooManyRounds:
            pass

    pending, decisions = [], []
    original_play = carlos.play

    def play(state):
        full = carlos.decide(state)
        variants = {}
        for name, settings in FACTORS:
            for key, value in settings.items():
                setattr(carlos, key, value)
            variants[name] = carlos.decide(state)
            for key in settings:
                delattr(carlos, key)
        pending.append(dict(current=state.current_bid, palifico=state.palifico,
                            alive=sum(p.alive for p in state.players), full=full, variants=variants))
        return original_play(state)

    def listen(event):
        if isinstance(event, RevealEvent):
            dice = [d for _, hand in event.dice for d in hand]
            for d in pending:
                d["right"] = {name: right(a, d["current"], dice, d["palifico"]) for name, a in d["variants"].items()}
                d["full_right"] = right(d["full"], d["current"], dice, d["palifico"])
                decisions.append(d)
            pending.clear()

    carlos.play = play
    wins = Counter()
    for s in seeds[args.warmup:]:
        try:
            result = PerudoGame(bots=bots, seed=s, listeners=[listen, RoundLimit()]).play()
            wins[result.winner] += 1
        except TooManyRounds:
            pass
        pending.clear()

    n = len(decisions)
    full_ok = sum(d["full_right"] for d in decisions)
    print(f"\n{args.games} analysed games after {args.warmup} warm-up games; winners: {dict(wins)}")
    print(f"{n} Carlos decisions; the full model was right in {100 * full_ok / n:.1f}%\n")
    print(f"{'factor switched off':<30}{'changes':>9}{'full right':>12}{'without right':>15}{'net for factor':>16}")
    print("-" * 82)
    for name, _ in FACTORS:
        changed = [d for d in decisions if d["variants"][name] != d["full"]]
        a = sum(d["full_right"] for d in changed)
        b = sum(d["right"][name] for d in changed)
        share = f"{len(changed)} ({100 * len(changed) / n:.0f}%)"
        print(f"{name:<30}{share:>9}{a:>12}{b:>15}{a - b:>+16}")

    print("\nHow the decision changes when the factor is off (full -> without), most common:")
    for name, _ in FACTORS:
        flips = Counter((kind(d["full"]), kind(d["variants"][name]))
                        for d in decisions if d["variants"][name] != d["full"])
        if flips:
            text = ", ".join(f"{a}->{b} {c}" for (a, b), c in flips.most_common(4))
            print(f"  {name:<30}{text}")


if __name__ == "__main__":
    main()
