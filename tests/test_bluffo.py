"""Bluffo Boyyo: expected-value play with deception, Dudo/Calza by value, quick learning."""

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
                  _state([6, 6, 1, 2, 3], [ActionRecord("B", Bid(3, 6))], Bid(3, 6)),
                  _state([4], [ActionRecord("B", Bid(2, 4))], Bid(2, 4), others=(("B", 1),), palifico=True)):
        assert bot.play(state) in state.legal_actions


def test_dudo_on_a_bid_it_does_not_believe():
    bot = BluffoBoyyo()
    state = _state([2, 3], [ActionRecord("B", Bid(5, 6))], Bid(5, 6), others=(("A", 1), ("B", 1)))
    assert bot.play(state) == Dudo()


def test_calza_when_exact_is_likely_and_gains_a_die():
    bot = BluffoBoyyo()
    bot._hints = lambda state, weight: {f: 0.0 for f in range(1, 7)}   # read no bids
    # We hold 4 sixes of a 4x6 bid; one unknown die: exact with 2/3.
    state = _state([6, 6, 6, 6], [ActionRecord("B", Bid(4, 6))], Bid(4, 6), others=(("B", 1),))
    assert bot.play(state) == Calza()


def test_deception_adds_value_but_never_rescues_a_hopeless_bid():
    """A bid that claims dice of a face we do not hold is worth more for the
    deception (when the table reads bids), as long as it stays plausible."""
    bot = BluffoBoyyo()
    # Others bid on fours; we hold sixes. 8x4 rides their fours and claims
    # some of our own: misleading, yet plausible.
    history = [ActionRecord("A", Bid(6, 4)), ActionRecord("B", Bid(7, 4))]
    state = _state([6, 6, 6, 6, 6], history, Bid(7, 4), others=(("A", 5), ("B", 5), ("C", 5), ("D", 5)))
    t = bot._table(state)
    hints = bot._hints(state, bot.HINT_PER_BID)
    with_deception = bot._bid_value(state, t, Bid(8, 4), hints, 1.0)
    bot.DECEIVE = 0.0
    without = bot._bid_value(state, t, Bid(8, 4), hints, 1.0)
    assert with_deception > without
    bot.DECEIVE = 0.4
    honest = bot._bid_value(state, t, Bid(8, 6), hints, 1.0)
    assert bot._bid_value(state, t, Bid(14, 4), hints, 1.0) < honest


def test_trap_against_an_eager_challenger():
    """A true bid that looks unlikely to a player who challenges a lot is a trap."""
    bot = BluffoBoyyo()
    bot._ensure_memory()
    bot._player("A")["dudo_a"] = 6.0          # A calls Dudo on nearly everything
    state = _state([6, 6, 6, 6, 6], [ActionRecord("B", Bid(2, 2))], Bid(2, 2), others=(("A", 5), ("B", 5)))
    t = bot._table(state)
    hints = bot._hints(state, bot.HINT_PER_BID)
    assert bot._bid_value(state, t, Bid(6, 6), hints, 0.5) > bot._bid_value(state, t, Bid(2, 3), hints, 0.5)


def test_learns_that_a_player_bluffs_and_challenges():
    bot = BluffoBoyyo()
    bot._ensure_memory()
    for _ in range(5):
        bot.observe(BidEvent("A", Bid(4, 6)))
        bot.observe(RevealEvent(dice=(("Bluffo Boyyo", (2, 3)), ("A", (2, 3, 4, 5, 2)), ("B", (2, 3, 4, 5, 2))),
                                current_bid=Bid(4, 6), actual_count=0, challenge_type="dudo",
                                challenger="B", bidder="A", result="bidder_wrong"))
    assert bot.learned["A"]["choice"] < 0
    b = bot.learned["B"]
    assert b["dudo_a"] > bot.DUDO_A           # B challenged every time


def test_learning_survives_a_new_instance():
    bot = BluffoBoyyo()
    bot._ensure_memory()
    bot.learned["A"] = dict(bot._player("A"), calza=0.9)
    other = BluffoBoyyo()
    other._ensure_memory()
    assert other.learned["A"]["calza"] == 0.9
