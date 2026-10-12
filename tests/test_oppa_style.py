"""Oppa Style: fractional Calza need, Palifico tracked from the round end."""

from perudo.bots.oppa_style import OppaStyleBot
from perudo.models import (
    ActionRecord, Bid, BidEvent, Calza, GameEndedEvent, Observation, PublicPlayer, RevealEvent, RoundEndedEvent,
)
from perudo.rules import legal_actions

ME = OppaStyleBot.name


def _state(my_dice, history=(), current_bid=None, others=(("A", 5), ("B", 5)), palifico=False):
    players = (PublicPlayer(0, ME, len(my_dice), True),
               *(PublicPlayer(i + 1, n, c, True) for i, (n, c) in enumerate(others)))
    total = sum(p.dice_count for p in players if p.alive)
    return Observation(
        my_id=0, my_dice=tuple(my_dice), my_dice_count=len(my_dice), players=players,
        current_bid=current_bid, palifico=palifico, round_number=1, history=tuple(history),
        legal_actions=tuple(legal_actions(current_bid, palifico, None, total)),
    )


def _reveal(bot, dice, bid, actual):
    bot.observe(RevealEvent(dice=dice, current_bid=bid, actual_count=actual, challenge_type="dudo",
                            challenger=ME, bidder=dice[-1][0], result="bidder_wrong"))


def test_plays_a_legal_action():
    bot = OppaStyleBot()
    for state in (_state([2, 3, 4, 5, 6]),
                  _state([6, 6, 1, 2, 3], [ActionRecord("B", Bid(3, 6))], Bid(3, 6)),
                  _state([4], [ActionRecord("B", Bid(2, 4))], Bid(2, 4), others=(("B", 1),), palifico=True)):
        assert bot.play(state) in state.legal_actions


def test_calza_prob_uses_the_fractional_hint():
    """FIX_CALZA_ROUNDING: a hint of 0.6 used to round to a whole die, so a
    bid of exactly our count had need -1 and P(exact) 0.0 although an
    exact count was the likeliest outcome."""
    bot = OppaStyleBot()
    bot._hints = lambda state, face: 0.6
    state = _state([6, 6], [ActionRecord("B", Bid(2, 6))], Bid(2, 6), others=(("B", 2),))
    # need = 2 - 2 - 0.6: 0.4 of the way from P(X = -1) = 0 to P(X = 0) = 4/9.
    assert abs(bot._calza_prob(state, Bid(2, 6), 2) - 0.4 * (2 / 3) ** 2) < 1e-9
    assert bot._calza_prob(state, Bid(3, 6), 2) > 0.0
    bot.FIX_CALZA_ROUNDING = False
    assert bot._calza_prob(state, Bid(2, 6), 2) == 0.0


def test_calza_when_exact_is_likely_and_gains_a_die():
    bot = OppaStyleBot()
    bot._hints = lambda state, face: 0.0
    state = _state([6, 6, 6, 6], [ActionRecord("B", Bid(4, 6))], Bid(4, 6), others=(("B", 1),))
    assert bot.play(state) == Calza()


def test_palifico_is_tracked_from_the_round_end():
    """FIX_PALIFICO_TRACK: in a Palifico round with no Pacos on the table the
    reveal looks like a normal round; the followers' forced bids must not
    be graded and the opener's surplus is against 1/6, not 1/3."""
    bot = OppaStyleBot()
    bot.observe(RoundEndedEvent(next_starting_player="A", palifico_next=True))
    bot.observe(BidEvent("A", Bid(2, 4)))
    bot.observe(BidEvent("B", Bid(3, 4)))
    _reveal(bot, ((ME, (2, 3)), ("A", (4, 4)), ("B", (2, 3, 5))), Bid(3, 4), 2)
    assert "B" not in bot.hint_n
    assert abs(bot.hint_sum["A"] - (2 - 2 / 6)) < 1e-9
    # A Paco-face Palifico round: the same.
    bot.observe(RoundEndedEvent(next_starting_player="A", palifico_next=True))
    bot.observe(BidEvent("A", Bid(1, 1)))
    bot.observe(BidEvent("B", Bid(2, 1)))
    _reveal(bot, ((ME, (2, 3)), ("A", (1, 3)), ("B", (2, 2, 2))), Bid(2, 1), 1)
    assert "B" not in bot.hint_n and bot.hint_n["A"] == 2
    # A normal round afterwards grades everyone again.
    bot.observe(RoundEndedEvent(next_starting_player="A", palifico_next=False))
    bot.observe(BidEvent("A", Bid(2, 4)))
    bot.observe(BidEvent("B", Bid(3, 4)))
    _reveal(bot, ((ME, (2, 3)), ("A", (4, 4)), ("B", (2, 3, 5))), Bid(3, 4), 2)
    assert bot.hint_n["B"] == 1
    bot.observe(RoundEndedEvent(next_starting_player="A", palifico_next=True))
    bot.observe(GameEndedEvent(winner="A", standings=("A", "B", ME)))
    assert bot._palifico_round is False


def test_old_inference_graded_palifico_followers():
    bot = OppaStyleBot()
    bot.FIX_PALIFICO_TRACK = False
    bot.observe(RoundEndedEvent(next_starting_player="A", palifico_next=True))
    bot.observe(BidEvent("A", Bid(2, 4)))
    bot.observe(BidEvent("B", Bid(3, 4)))
    _reveal(bot, ((ME, (2, 3)), ("A", (4, 4)), ("B", (2, 3, 5))), Bid(3, 4), 2)
    assert bot.hint_n["B"] == 1
    assert abs(bot.hint_sum["A"] - (2 - 2 / 3)) < 1e-9
