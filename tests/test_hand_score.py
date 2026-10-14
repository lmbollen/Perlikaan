import doctest
from itertools import product

from perudo import hand_score
from perudo.hand_score import rank, score, value


def test_the_docstring_examples():
    assert doctest.testmod(hand_score).failed == 0


def test_the_formula_in_short():
    # 0.9 per Paco + 0.5 per die of the most common other face - 7n/30
    for dice in [(1, 1, 1, 1, 1), (1, 6, 6, 6, 6), (6, 6, 6, 6, 6), (2, 3, 4, 5, 6), (1, 4, 4, 2), (3,)]:
        pacos = dice.count(1)
        most = max((dice.count(f) for f in range(2, 7)), default=0)
        assert abs(value(dice) - (0.9 * pacos + 0.5 * most - 7 * len(dice) / 30)) < 1e-9


def test_pacos_beat_a_single_face():
    order = [(1, 1, 1, 1, 1), (1, 6, 6, 6, 6), (6, 6, 6, 6, 6), (1, 4, 4, 4, 6), (2, 3, 4, 5, 6)]
    scores = [score(d).score for d in order]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 100 and scores[-1] <= 2


def test_ranks_are_exact_over_all_hands():
    hands = list(product(range(1, 7), repeat=4))
    mean = sum(rank(h) for h in hands) / len(hands)
    assert abs(mean - 0.5) < 1e-9                 # mid-ranks average a half, exactly
    assert all(0 <= score(h).score <= 100 for h in hands)


def test_palifico_only_the_opener_picks_the_face():
    assert value((5, 5, 5, 2, 3), palifico=True) == 0
    assert score((5, 5, 5, 2, 3), palifico=True).score == 50
    assert value((5, 5, 5, 2, 3), palifico=True, opens=True) > value((5, 2, 3, 4, 6), palifico=True, opens=True)
    assert value((1, 1, 1), palifico=True, opens=True) == value((4, 4, 4), palifico=True, opens=True)
