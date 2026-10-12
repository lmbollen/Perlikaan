"""Sniper: HINT_ONCE counts a player's hint on a face once per round, never the judged bid."""

from perudo.bots.sniper import SniperBot
from perudo.models import ActionRecord, Bid, Observation, PublicPlayer
from perudo.rules import legal_actions

ME = SniperBot.name


def _state(my_dice, history=(), current_bid=None, others=(("A", 5), ("B", 5)), palifico=False):
    players = (PublicPlayer(0, ME, len(my_dice), True),
               *(PublicPlayer(i + 1, n, c, True) for i, (n, c) in enumerate(others)))
    total = sum(p.dice_count for p in players if p.alive)
    return Observation(
        my_id=0, my_dice=tuple(my_dice), my_dice_count=len(my_dice), players=players,
        current_bid=current_bid, palifico=palifico, round_number=1, history=tuple(history),
        legal_actions=tuple(legal_actions(current_bid, palifico, None, total)),
    )


def _hist(*bids):
    return [ActionRecord(who, bid) for who, bid in bids]


def test_default_is_unchanged():
    assert SniperBot.HINT_ONCE is False


def test_hint_once_counts_a_face_once_per_player_per_round():
    history = _hist(("A", Bid(3, 4)), ("B", Bid(4, 4)), ("A", Bid(5, 4)), ("B", Bid(6, 4)))
    state = _state([2, 3, 5, 6, 6], history, Bid(6, 4))
    bot = SniperBot()
    assert abs(bot._hint(state, 4) - 4 * bot.HINT_PER_BID) < 1e-9       # every raise counts
    bot.HINT_ONCE = True
    assert abs(bot._hint(state, 4) - 2 * bot.HINT_PER_BID) < 1e-9       # once per player
    # A player who bid on two faces counts once on each of them.
    history = _hist(("A", Bid(3, 4)), ("B", Bid(3, 5)), ("A", Bid(4, 6)))
    state = _state([2, 3, 5, 6, 6], history, Bid(4, 6))
    assert abs(bot._hint(state, 4) - bot.HINT_PER_BID) < 1e-9
    assert abs(bot._hint(state, 6) - bot.HINT_PER_BID) < 1e-9


def test_judged_bid_is_not_its_own_evidence():
    bot = SniperBot()
    bot.HINT_ONCE = True
    state = _state([4], _hist(("B", Bid(2, 4))), Bid(2, 4), others=(("B", 1),))
    assert bot._hint(state, 4) > 0.0
    assert bot._hint(state, 4, exclude_current=True) == 0.0
    faced = {f: bot._hint(state, f, exclude_current=True) for f in range(1, 7)}
    assert abs(bot._prob_true(state, Bid(2, 4), 1, faced) - 1 / 3) < 1e-9
    # B's earlier bid on the face stays evidence when B's raise is judged.
    state = _state([4], _hist(("B", Bid(2, 4)), ("A", Bid(3, 4)), ("B", Bid(4, 4))), Bid(4, 4), others=(("A", 2), ("B", 2)))
    assert abs(bot._hint(state, 4, exclude_current=True) - bot._hint(state, 4)) < 1e-9
    # Palifico: judging the opener's bid leaves no evidence; a follower's
    # raise is judged with the opener's hint.
    state = _state([4], _hist(("A", Bid(2, 4))), Bid(2, 4), others=(("A", 2), ("B", 2)), palifico=True)
    assert bot._hint(state, 4, exclude_current=True) == 0.0
    state = _state([4], _hist(("A", Bid(2, 4)), ("B", Bid(3, 4))), Bid(3, 4), others=(("A", 2), ("B", 2)), palifico=True)
    assert bot._hint(state, 4, exclude_current=True) == bot._hint(state, 4) > 0.0


def test_plays_a_legal_action_either_way():
    for once in (False, True):
        bot = SniperBot()
        bot.HINT_ONCE = once
        for state in (_state([2, 3, 4, 5, 6]),
                      _state([6, 6, 1, 2, 3], _hist(("B", Bid(3, 6))), Bid(3, 6)),
                      _state([4], _hist(("B", Bid(2, 4))), Bid(2, 4), others=(("B", 1),), palifico=True)):
            assert bot.play(state) in state.legal_actions
