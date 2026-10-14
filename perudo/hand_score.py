"""
How good is a hand? A score for one player's dice at the start of a round:
the chance they get through the round without losing a die, from what
they know at that moment (their dice, everyone's dice counts, who opens,
Palifico).

It is a logistic model fitted on simulated rounds of the bundled bots
(fit_hand_score.py writes the WEIGHTS below and says how well it predicts).

    >>> s = score((1, 4, 4, 4, 6), players=4, total_dice=20, turn=0)
    >>> round(s.keep, 2), s.label
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

from perudo.models import PACO

FEATURES = (
    "bias", "players", "share", "surplus", "surplus_x_share", "pacos_share",
    "spread", "opens", "turn", "palifico", "one_die",
)

# Fitted by fit_hand_score.py: log-odds of losing a die this round.
WEIGHTS: Dict[str, float] = {name: 0.0 for name in FEATURES}

# The same model without the dice (only the table): the chance an average
# hand keeps its dice here. score() compares the hand with it.
BASE_WEIGHTS: Dict[str, float] = {name: 0.0 for name in FEATURES}


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
    faces = len(set(dice))
    share = n / total_dice if total_dice else 0.0
    surplus = best - expected            # matches on our best face beyond what chance gives
    return {
        "bias": 1.0,
        "players": float(players),
        "share": share,
        "surplus": surplus,
        "surplus_x_share": surplus * share,
        "pacos_share": pacos / n if n else 0.0,
        "spread": faces / n if n else 0.0,          # many different faces: little to bid on
        "opens": 1.0 if turn == 0 else 0.0,
        "turn": turn / max(players - 1, 1),
        "palifico": 1.0 if palifico else 0.0,
        "one_die": 1.0 if n == 1 else 0.0,
    }


def _p_lose(weights: Dict[str, float], x: Dict[str, float]) -> float:
    z = sum(weights.get(name, 0.0) * value for name, value in x.items())
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


@dataclass(frozen=True)
class HandScore:
    keep: float          # chance of getting through the round without losing a die
    average: float       # the same for an average hand in this spot
    score: int           # 0..100: how this hand compares (50 = an average hand)
    label: str           # a word for it


LABELS = ((80, "Sterk"), (60, "Goed"), (40, "Gemiddeld"), (20, "Zwak"), (0, "Slecht"))


def score(dice: Sequence[int], players: int, total_dice: int, turn: int,
          palifico: bool = False) -> HandScore:
    x = features(dice, players, total_dice, turn, palifico)
    keep = 1.0 - _p_lose(WEIGHTS, x)
    average = 1.0 - _p_lose(BASE_WEIGHTS, x)
    # Compare in log-odds, so the score means the same at a big table and
    # in a duel: +1 log-odds (e.g. 50% -> 73%) is 25 points above average.
    edge = _logit(keep) - _logit(average)
    points = int(round(max(0.0, min(100.0, 50.0 + 25.0 * edge))))
    label = next(word for floor, word in LABELS if points >= floor)
    return HandScore(keep, average, points, label)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))
