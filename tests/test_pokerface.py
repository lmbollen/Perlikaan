"""Pokerface bluffs about faces it doesn't hold, but only with plausible bids."""

from perudo.bots.pokerface import PokerfaceBot
from perudo.models import ActionRecord, Bid, Observation, PublicPlayer
from perudo.rules import legal_actions


def _state(my_dice, history=(), current_bid=None, palifico=False):
    players = (
        PublicPlayer(0, "Pokerface", len(my_dice), True),
        PublicPlayer(1, "A", 5, True),
        PublicPlayer(2, "B", 5, True),
    )
    total = sum(p.dice_count for p in players)
    return Observation(
        my_id=0,
        my_dice=tuple(my_dice),
        my_dice_count=len(my_dice),
        players=players,
        current_bid=current_bid,
        palifico=palifico,
        round_number=1,
        history=tuple(history),
        legal_actions=tuple(legal_actions(current_bid, palifico, None, total)),
    )


def _always_bluff():
    bot = PokerfaceBot()
    bot.BLUFF_RATE = 1.0
    return bot


def test_opening_bluff_uses_a_face_we_do_not_hold():
    bot = _always_bluff()
    state = _state([2, 2, 3, 3, 4])
    for _ in range(20):
        bid = bot.play(state)
        assert isinstance(bid, Bid)
        assert bid.face in (5, 6)


def test_follows_precedence_on_a_face_we_do_not_hold():
    bot = _always_bluff()
    history = [
        ActionRecord("A", Bid(2, 6)),
        ActionRecord("B", Bid(3, 6)),
    ]
    state = _state([2, 2, 3, 3, 4], history, current_bid=Bid(3, 6))
    action = bot.play(state)
    assert isinstance(action, Bid)
    assert action.face == 6


def test_no_bluff_in_palifico():
    bot = _always_bluff()
    state = _state([2], palifico=True)
    assert bot._bluff(state, [], 10, {f: 0.0 for f in range(1, 7)}) == (None, None)


def test_bluff_bids_are_still_plausible():
    bot = _always_bluff()
    state = _state([2, 2, 3, 3, 4])
    hints = {f: 0.0 for f in range(1, 7)}
    bid = bot.play(state)
    assert bot._prob_true(state, bid, 10, hints) >= bot.BLUFF_TARGET
