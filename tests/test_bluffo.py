"""Bluffo Boyyo: coin-flip bluffs, min/max raises, Calza and Dudo, quick learning."""

import random

from perudo.bots.bluffo_boyyo import BluffoBoyyo
from perudo.models import PACO, ActionRecord, Bid, BidEvent, Calza, Dudo, Observation, PublicPlayer, RevealEvent
from perudo.rules import legal_actions


def _state(my_dice, history=(), current_bid=None, others=(("A", 5), ("B", 5)), palifico=False, round_number=1):
    players = (PublicPlayer(0, "Bluffo Boyyo", len(my_dice), True),
               *(PublicPlayer(i + 1, n, c, True) for i, (n, c) in enumerate(others)))
    total = sum(p.dice_count for p in players)
    return Observation(
        my_id=0, my_dice=tuple(my_dice), my_dice_count=len(my_dice), players=players,
        current_bid=current_bid, palifico=palifico, round_number=round_number, history=tuple(history),
        legal_actions=tuple(legal_actions(current_bid, palifico, None, total)),
    )


def test_plays_a_legal_action():
    bot = BluffoBoyyo()
    for state in (_state([2, 3, 4, 5, 6]),
                  _state([6, 6, 1, 2, 3], [ActionRecord("B", Bid(3, 6))], Bid(3, 6))):
        assert bot.play(state) in state.legal_actions


def test_dudo_on_a_bid_it_does_not_believe():
    bot = BluffoBoyyo()
    state = _state([2, 3], [ActionRecord("B", Bid(5, 6))], Bid(5, 6), others=(("A", 1), ("B", 1)))
    assert bot.play(state) == Dudo()


def test_calza_when_exact_is_likely_and_gains_a_die():
    bot = BluffoBoyyo()
    bot._hints = lambda state, weight=None: {f: 0.0 for f in range(1, 7)}   # read no bids
    # We hold 4 sixes of a 4x6 bid; one unknown die: exact with 2/3.
    state = _state([6, 6, 6, 6], [ActionRecord("B", Bid(4, 6))], Bid(4, 6), others=(("B", 1),))
    assert bot.play(state) == Calza()


def test_bluffs_about_half_the_rounds():
    bot = BluffoBoyyo()
    bot._rng = random.Random(1)
    bluffs = 0
    for r in range(1, 401):
        bot.play(_state([2, 3, 4, 5, 6], round_number=r))
        bluffs += bot._bluffing
    assert 150 < bluffs < 250


def test_alternates_min_and_max_raises():
    bot = BluffoBoyyo()
    bot.BLUFF_RATE = 0.0
    state = _state([6, 6, 6, 6, 6], [ActionRecord("B", Bid(2, 2))], Bid(2, 2))
    bids = [bot.play(state) for _ in range(4)]
    strengths = [b.quantity for b in bids]
    assert strengths[0] != strengths[1] and strengths[0] == strengths[2] and strengths[1] == strengths[3]


def test_learns_that_a_player_bluffs():
    bot = BluffoBoyyo()
    bot._ensure_memory()
    for _ in range(5):
        bot.observe(BidEvent("A", Bid(4, 6)))
        bot.observe(RevealEvent(dice=(("Bluffo Boyyo", (2, 3)), ("A", (2, 3, 4, 5, 2)), ("B", (2, 3, 4, 5, 2))),
                                current_bid=Bid(4, 6), actual_count=0, challenge_type="dudo",
                                challenger="B", bidder="A", result="bidder_wrong"))
    assert bot.learned["A"]["choice"] < 0
    assert bot.learned["A"]["bluff"] > 0.8
    assert bot.learned["B"]["challenge"] > 0.8


def test_learning_survives_a_new_instance():
    bot = BluffoBoyyo()
    bot._ensure_memory()
    bot.learned["A"] = dict(bot._player("A"), bluff=0.9)
    other = BluffoBoyyo()
    other._ensure_memory()
    assert other.learned["A"]["bluff"] == 0.9
