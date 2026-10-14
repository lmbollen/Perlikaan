"""
How good is a hand? Reasoned, not measured: a hand is worth the matches it
holds beyond what the table assumes it holds.

Nobody sees our dice, so the others count on the average: n/3 matches for
any face (a third of the dice are that face or a Paco). What we really hold
for a face is its dice plus our Pacos; the difference is an advantage only
we know about. Which face counts? Half the time we pick it (we open, or
switch to our best face on our turn), half the time someone else does, and
then it is any face. So the value is half the advantage on our best face
plus half the average advantage over the faces:

    value = 1/2 (P + M - n/3) + 1/2 (P + (n - P)/5 - n/3)
          = 0.9 P + 0.5 M - 7n/30

P our Pacos, M our most common other face, n our dice. A Paco is worth
almost a whole match (it counts whatever is bid), a die of the best face
half of one (it counts when that face is bid).

In a Palifico round Pacos are not wild and the opener fixes the face: the
opener's hand is worth its best face, everyone else's the average over the
faces (nothing: they do not know which face it will be).

The score ranks the value among all 6^n hands of n dice, counted exactly:
50 is an average hand, five Pacos 100. check_hand_score.py deals every
hand type in simulated games to check the ranking against what happens.

    >>> s = score((1, 1, 1, 1, 1))
    >>> round(s.value, 2), s.score, s.label
    (3.33, 100, 'Strong')
    >>> score((2, 3, 4, 5, 6)).score
    1
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement
from typing import Sequence, Tuple

from perudo.models import PACO

FACES = (2, 3, 4, 5, 6)          # the faces besides Pacos
BEST_FACE_SHARE = 0.5            # how often we pick the face (else: any face)


def value(dice: Sequence[int], palifico: bool = False, opens: bool = False) -> float:
    """Matches beyond what the table assumes we hold (see the module)."""
    n = len(dice)
    if not n:
        return 0.0
    counts = Counter(dice)
    if palifico:
        expected = n / 6                          # one face, no wild Pacos
        if not opens:
            return 0.0                            # the average over the faces: no advantage
        return max(counts.values()) - expected
    pacos = counts.get(PACO, 0)
    expected = n / 3
    best = pacos + max((counts.get(f, 0) for f in FACES), default=0)
    average = pacos + (n - pacos) / len(FACES)
    return BEST_FACE_SHARE * (best - expected) + (1 - BEST_FACE_SHARE) * (average - expected)


@lru_cache(maxsize=None)
def _values(n: int, palifico: bool, opens: bool) -> Tuple[Tuple[float, float], ...]:
    """(value, chance) of every hand of n dice, from the lowest value up."""
    out = Counter()
    total = 6 ** n
    for hand in combinations_with_replacement(range(1, 7), n):
        ways = math.factorial(n)
        for c in Counter(hand).values():
            ways //= math.factorial(c)
        out[round(value(hand, palifico, opens), 9)] += ways / total
    return tuple(sorted(out.items()))


def rank(dice: Sequence[int], palifico: bool = False, opens: bool = False) -> float:
    """How many hands of the same size are worse (ties count half): 0..1."""
    v = round(value(dice, palifico, opens), 9)
    below = sum(p for w, p in _values(len(dice), palifico, opens) if w < v)
    same = sum(p for w, p in _values(len(dice), palifico, opens) if w == v)
    return below + same / 2


@dataclass(frozen=True)
class HandScore:
    value: float         # matches beyond what the table assumes we hold
    score: int           # 0..100: how the hand ranks among all hands of its size (50 = average)
    label: str


LABELS = ((80, "Strong"), (60, "Good"), (40, "Average"), (20, "Weak"), (0, "Poor"))


def score(dice: Sequence[int], palifico: bool = False, opens: bool = False) -> HandScore:
    v = value(dice, palifico, opens)
    points = int(round(100 * rank(dice, palifico, opens))) if dice else 50
    label = next(word for floor, word in LABELS if points >= floor)
    return HandScore(v, points, label)
