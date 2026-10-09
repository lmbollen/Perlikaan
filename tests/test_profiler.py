"""The profiler replays revealed rounds from each opponent's point of view."""

from perudo.models import PACO, Bid, Calza, Dudo
from perudo.profiler import OpponentProfiler, Profile, p_bid_true


def test_p_bid_true_counts_own_dice_and_wilds():
    # Own 2 + Paco cover 3x2 without any unknown die.
    assert p_bid_true(Bid(3, 2), (2, 2, 1), 10) == 1.0
    assert 0 < p_bid_true(Bid(4, 2), (2, 2, 1), 10) < 1


def test_record_round_scores_faces_and_dudo_decisions():
    prof = OpponentProfiler("Me")
    dice = {"A": (2, 2, 2, 5, 6), "B": (3, 4, 5, 6, 6), "Me": (1,)}
    prof.record_round(dice, [("A", Bid(2, 2)), ("B", Bid(2, 6)), ("Me", Bid(4, 6)), ("A", Dudo())], False)

    a, b = prof.profile("A"), prof.profile("B")
    assert a.face_scores == [1.0]          # opened on its best face
    assert b.face_scores == [1.0]          # switched to its best face
    assert b.min_raises == 1               # 2x2 -> 2x6 is no jump
    assert [d for _, d in a.facing] == [True]
    assert "Me" not in prof.profiles       # we don't profile ourselves


def test_palifico_rounds_are_skipped():
    prof = OpponentProfiler("Me")
    prof.record_round({"A": (2,), "B": (3,)}, [("A", Bid(1, 2)), ("B", Dudo())], True)
    assert prof.profiles == {}


def test_forced_dudo_is_not_a_choice():
    prof = OpponentProfiler("Me")
    # 2 dice on the table: after 2xPaco no raise is possible, so B's Dudo
    # is forced and only its earlier raise counts as a decision.
    actions = [("A", Bid(1, 6)), ("B", Bid(1, PACO)), ("A", Bid(2, PACO)), ("B", Dudo())]
    prof.record_round({"A": (6,), "B": (3,)}, actions, False)
    assert [d for _, d in prof.profile("B").facing] == [False]


def test_threshold_fit_recovers_a_clean_threshold():
    p = Profile("X")
    p.facing = [(x / 20, x / 20 < 0.4) for x in range(21)] * 2
    t, fit = p.dudo_threshold()
    assert 0.35 < t < 0.45 and fit == 1.0
    assert p.labels()["dudo"] == "normal"


def test_calza_counts():
    prof = OpponentProfiler()
    prof.record_round({"A": (2,), "B": (2, 3)}, [("A", Bid(1, 2)), ("B", Calza())], False)
    assert prof.profile("B").calzas == 1
