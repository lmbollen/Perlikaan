import doctest

from perudo import hand_score
from perudo.hand_score import FEATURES, TABLE, features, score


def test_the_docstring_example():
    assert doctest.testmod(hand_score).failed == 0


def test_a_hand_rich_in_one_face_beats_a_spread_out_hand():
    rich = score((1, 4, 4, 4, 6), players=4, total_dice=20, turn=1)
    spread = score((2, 3, 4, 5, 6), players=4, total_dice=20, turn=1)
    assert rich.score > 50 > spread.score
    assert rich.keep > rich.average > spread.keep
    assert rich.average == spread.average          # the same spot: the same average hand


def test_scores_stay_in_range_everywhere():
    hands = [(1,), (6,), (1, 1), (2, 5), (1, 1, 1, 1, 1), (2, 3, 4, 5, 6), (6, 6, 6, 6, 6)]
    for dice in hands:
        for players in (2, 3, 6, 8):
            total = len(dice) + 5 * (players - 1)
            for turn in range(players):
                for palifico in (False, True):
                    s = score(dice, players, total, turn, palifico)
                    assert 0 <= s.score <= 100
                    assert 0 < s.keep < 1 and 0 < s.average < 1
                    assert s.label


def test_pacos_are_wild_except_in_palifico():
    assert features((1, 1, 3), 3, 11, 0)["best"] == 3          # two Pacos and a three
    assert features((1, 1, 3), 3, 11, 0, palifico=True)["best"] == 2


def test_the_average_hand_ignores_the_dice():
    assert set(TABLE) < set(FEATURES)
    assert all(hand_score.BASE_WEIGHTS[f] == 0 for f in FEATURES if f not in TABLE)
