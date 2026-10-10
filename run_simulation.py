"""
Simulation mode: run many Perudo games fully automatically, with as little
console output as possible during the games themselves.

Usage:
    python run_simulation.py --games 2000 --seed 12345

Bots are instantiated exactly once, before the loop, and the same bot
objects are reused for every game -- this is what lets a bot "learn"
about opponents across games via its own instance variables. Nothing is
ever written to disk for bots; only an optional --csv results log.

This file contains NO game rules -- it only drives PerudoGame in a loop
and aggregates the public events it emits into statistics.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from typing import List

from perudo.bot_loader import load_bots
from perudo.game import PerudoGame
from perudo.models import BidEvent, CalzaEvent, DudoEvent, GameResult, RevealEvent


class Stats:
    def __init__(self, bot_names: List[str]):
        self.bot_names = bot_names
        self.games = 0
        self.wins = defaultdict(int)
        self.position_sum = defaultdict(int)
        self.bids = 0
        self.dudos = 0
        self.dudos_successful = 0
        self.calzas = 0
        self.calzas_successful = 0
        self.rounds_total = 0

    def record_event(self, event) -> None:
        if isinstance(event, BidEvent):
            self.bids += 1
        elif isinstance(event, DudoEvent):
            self.dudos += 1
        elif isinstance(event, CalzaEvent):
            self.calzas += 1
        elif isinstance(event, RevealEvent):
            if event.challenge_type == "dudo" and event.result == "bidder_wrong":
                self.dudos_successful += 1
            elif event.challenge_type == "calza" and event.result == "correct":
                self.calzas_successful += 1

    def record_game(self, result: GameResult) -> None:
        self.games += 1
        self.wins[result.winner] += 1
        for position, name in enumerate(result.standings, start=1):
            self.position_sum[name] += position
        self.rounds_total += result.rounds_played

    def print_report(self) -> None:
        print(f"\n{'=' * 64}")
        print(f"SIMULATIE RESULTATEN -- {self.games} games")
        print(f"{'=' * 64}")
        header = f"{'Bot':<22}{'Wins':>8}{'Win %':>10}{'Gem. positie':>16}"
        print(header)
        print("-" * len(header))
        for name in self.bot_names:
            wins = self.wins[name]
            win_pct = 100 * wins / self.games if self.games else 0.0
            avg_pos = self.position_sum[name] / self.games if self.games else 0.0
            print(f"{name:<22}{wins:>8}{win_pct:>9.1f}%{avg_pos:>16.2f}")

        print()
        print(f"Totaal bids            : {self.bids}")
        print(f"Totaal Dudo's          : {self.dudos}  (succesvol voor de roeper: {self.dudos_successful})")
        print(f"Totaal Calza's         : {self.calzas}  (succesvol: {self.calzas_successful})")
        avg_rounds = self.rounds_total / self.games if self.games else 0.0
        print(f"Gem. rondes per game   : {avg_rounds:.2f}")
        print(f"{'=' * 64}")


def derive_game_seeds(base_seed: int, count: int) -> List[int]:
    """Deterministically turn one base seed into `count` per-game seeds."""
    seed_rng = random.Random(base_seed)
    return [seed_rng.randrange(2**63) for _ in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Perudo simulation mode (many automatic games).")
    parser.add_argument("--games", type=int, default=200, help="Aantal potjes.")
    parser.add_argument("--seed", type=int, default=12345, help="Basis-seed voor reproduceerbaarheid.")
    parser.add_argument("--csv", type=str, default=None, help="Optioneel: pad naar een CSV-resultatenlog.")
    args = parser.parse_args()

    # Bots worden precies EEN keer geinitialiseerd, hier, buiten de loop.
    bots = load_bots()
    bot_names = [bot.name for bot in bots]
    stats = Stats(bot_names)
    game_seeds = derive_game_seeds(args.seed, args.games)

    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["game_number", "seed", "winner", "rounds_played", "standings"])

    print(f"Simuleren van {args.games} games met bots: {', '.join(bot_names)} (seed={args.seed})...")

    for game_number, game_seed in enumerate(game_seeds, start=1):
        # Elke game krijgt een volledig nieuwe engine/state, maar dezelfde
        # bot-objecten worden hergebruikt (zij mogen dus onthouden leren).
        game = PerudoGame(bots=bots, seed=game_seed, listeners=[stats.record_event])
        result = game.play()
        stats.record_game(result)
        if csv_writer is not None:
            csv_writer.writerow(
                [game_number, game_seed, result.winner, result.rounds_played, "|".join(result.standings)]
            )

    if csv_file is not None:
        csv_file.close()

    stats.print_report()


if __name__ == "__main__":
    main()
