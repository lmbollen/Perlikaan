"""
How good is a hand? A score for one player's dice at the start of a round,
from what they know at that moment (their dice, everyone's dice counts,
who opens, Palifico):

  keep     the chance they get through the round without losing a die
  average  the same for an average hand in their spot (the table alone)
  score    0..100, how the hand ranks among hands in that spot (50 average)

A logistic model fitted on simulated rounds of the bundled bots: see
fit_hand_score.py, which fits the weights below and judges them on games
it did not see.

    >>> s = score((1, 4, 4, 4, 6), players=4, total_dice=20, turn=1)
    >>> s.score, s.label, round(s.keep, 2), round(s.average, 2)
    (94, 'Strong', 0.82, 0.76)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

from perudo.models import PACO

FEATURES = (
    # the table: how many play, how many dice we hold of them, where we bid
    "bias", "players", "log_total", "share", "opens", "turn", "last", "duel",
    "palifico", "one_die", "two_dice", "three_dice",
    # the hand
    "best", "surplus", "pacos", "spread",
)
# The features of the table alone (the same model for an average hand).
TABLE = FEATURES[:12]

# Fitted by fit_hand_score.py: log-odds of losing a die this round.
WEIGHTS: Dict[str, float] = {
    "bias": -0.5322,
    "players": -0.4526,
    "log_total": +0.9095,
    "share": -0.5691,
    "opens": +0.3356,
    "turn": +0.2235,
    "last": -0.0311,
    "duel": +0.5191,
    "palifico": -0.4069,
    "one_die": -0.2040,
    "two_dice": +0.0498,
    "three_dice": +0.0919,
    "best": -0.8519,
    "surplus": +0.7280,
    "pacos": -0.1397,
    "spread": +0.0754,
}

# The same model without the dice (only the table): the chance an average
# hand keeps its dice here. score() compares the hand with it.
BASE_WEIGHTS: Dict[str, float] = {
    "bias": -0.9390,
    "players": -0.4056,
    "log_total": +0.5450,
    "share": -1.1417,
    "opens": +0.3324,
    "turn": +0.1691,
    "last": -0.0067,
    "duel": +0.4579,
    "palifico": -0.0507,
    "one_die": +0.4640,
    "two_dice": +0.6132,
    "three_dice": +0.4399,
    "best": +0.0000,
    "surplus": +0.0000,
    "pacos": +0.0000,
    "spread": +0.0000,
}

# How far hands differ from the average one (the spread of the log-odds
# difference over the fitted rounds): one spread better scores 84.
EDGE_SD = 0.2142


def features(dice: Sequence[int], players: int, total_dice: int, turn: int,
             palifico: bool = False) -> Dict[str, float]:
    """What the model looks at. `turn` is the player's place in the bidding
    order of the round: 0 opens, players - 1 bids last."""
    n = len(dice)
    pacos = sum(1 for d in dice if d == PACO)
    if palifico:
        # No wild Pacos: the best face is just the most common one.
        best = max(sum(1 for d in dice if d == f) for f in range(1, 7)) if n else 0
        expected = n / 6
    else:
        best = max(sum(1 for d in dice if d == f) for f in range(2, 7)) + pacos if n else 0
        expected = n / 3
    return {
        "bias": 1.0,
        "players": float(players),
        "log_total": math.log(max(total_dice, 1)),
        "share": n / total_dice if total_dice else 0.0,
        "opens": 1.0 if turn == 0 else 0.0,
        "turn": turn / max(players - 1, 1),
        "last": 1.0 if turn == players - 1 else 0.0,
        "duel": 1.0 if players == 2 else 0.0,
        "palifico": 1.0 if palifico else 0.0,
        "one_die": 1.0 if n == 1 else 0.0,
        "two_dice": 1.0 if n == 2 else 0.0,
        "three_dice": 1.0 if n == 3 else 0.0,
        "best": float(best),                         # matches on our best face
        "surplus": best - expected,                  # ... beyond what chance gives
        "pacos": float(pacos),
        "spread": len(set(dice)) / n if n else 0.0,  # many different faces: little to bid on
    }


def _p_lose(weights: Dict[str, float], x: Dict[str, float]) -> float:
    z = sum(weights.get(name, 0.0) * value for name, value in x.items())
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


@dataclass(frozen=True)
class HandScore:
    keep: float          # chance of getting through the round without losing a die
    average: float       # the same for an average hand in this spot
    score: int           # 0..100: how this hand ranks among hands in its spot (50 = average)
    label: str           # a word for it


LABELS = ((80, "Strong"), (60, "Good"), (40, "Average"), (20, "Weak"), (0, "Poor"))


def score(dice: Sequence[int], players: int, total_dice: int, turn: int,
          palifico: bool = False) -> HandScore:
    x = features(dice, players, total_dice, turn, palifico)
    keep = 1.0 - _p_lose(WEIGHTS, x)
    average = 1.0 - _p_lose(BASE_WEIGHTS, x)
    # Compare in log-odds, so the score means the same at a big table and
    # in a duel: +1 log-odds (e.g. 50% -> 73%) is 25 points above average.
    points = int(round(100 * rank(_logit(keep) - _logit(average))))
    label = next(word for floor, word in LABELS if points >= floor)
    return HandScore(keep, average, points, label)


def rank(edge: float) -> float:
    """Where a hand this much better than average (in log-odds of keeping
    its dice) ranks among hands: 0.5 is average, 0.84 one spread better."""
    return 0.5 * (1.0 + math.erf(edge / (EDGE_SD * math.sqrt(2.0))))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))
