"""Sonny Boy: Calza only when it can gain a die, raises scored like the faced bid."""

from perudo.bots.sonny_boy import SonnyBoy
from perudo.models import ActionRecord, Bid, Calza, Dudo, Observation, PublicPlayer
from perudo.rules import legal_actions


def _state(my_dice, history=(), current_bid=None, others=(("A", 5), ("B", 5)), palifico=False, out=()):
    players = (PublicPlayer(0, "Sonny Boy", len(my_dice), True),
               *(PublicPlayer(i + 1, n, c, True) for i, (n, c) in enumerate(others)),
               *(PublicPlayer(len(others) + 1 + i, n, 0, False) for i, n in enumerate(out)))
    total = sum(p.dice_count for p in players if p.alive)
    return Observation(
        my_id=0, my_dice=tuple(my_dice), my_dice_count=len(my_dice), players=players,
        current_bid=current_bid, palifico=palifico, round_number=1, history=tuple(history),
        legal_actions=tuple(legal_actions(current_bid, palifico, None, total)),
    )


def test_plays_a_legal_action():
    bot = SonnyBoy()
    for state in (_state([2, 3, 4, 5, 6]),
                  _state([6, 6, 1, 2, 3], [ActionRecord("B", Bid(3, 6))], Bid(3, 6)),
                  _state([4], [ActionRecord("B", Bid(2, 4))], Bid(2, 4), others=(("B", 1),), palifico=True)):
        assert bot.play(state) in state.legal_actions


def test_calza_when_exact_is_likely_and_gains_a_die():
    bot = SonnyBoy()
    # Four sixes of a 5x6 bid; B (credited one six) has one other die: exact with 2/3.
    state = _state([6, 6, 6, 6], [ActionRecord("B", Bid(5, 6))], Bid(5, 6), others=(("B", 2),))
    assert bot._p_exact(state, Bid(5, 6), False, credit=True) > bot.CALZA_MIN
    assert bot.play(state) == Calza()


def test_no_calza_without_a_die_to_gain():
    """FIX_CALZA_GAIN: with five dice, or once someone is out, a correct Calza
    gains nothing and a wrong one still costs a die."""
    bot = SonnyBoy()
    full = _state([6, 6, 6, 6, 6], [ActionRecord("B", Bid(6, 6))], Bid(6, 6), others=(("B", 2),))
    someone_out = _state([6, 6, 6, 6], [ActionRecord("B", Bid(5, 6))], Bid(5, 6), others=(("B", 2),), out=("C",))
    for state in (full, someone_out):
        pe = bot._p_exact(state, state.current_bid, False, credit=True)
        assert bot.CALZA_MIN <= pe < bot.CALZA_FREE_PASS_MIN
        assert bot.play(state) != Calza()
    bot.FIX_CALZA_GAIN = False
    for state in (full, someone_out):
        assert bot.play(state) == Calza()


def test_free_pass_calza_when_exact_is_near_certain():
    bot = SonnyBoy()
    # Five sixes, B's single die credited as a six: 6x6 is exact for sure.
    state = _state([6, 6, 6, 6, 6], [ActionRecord("B", Bid(6, 6))], Bid(6, 6), others=(("B", 1),))
    assert bot._p_exact(state, Bid(6, 6), False, credit=True) >= bot.CALZA_FREE_PASS_MIN
    assert bot.play(state) == Calza()


def test_raises_are_scored_with_the_same_credit_as_the_faced_bid():
    """FIX_CREDIT_CONSISTENT: the Dudo-vs-raise comparison uses one estimate."""
    bot = SonnyBoy()
    state = _state([2, 3, 4, 5, 6], [ActionRecord("A", Bid(3, 6))], Bid(3, 6))
    bids = [a for a in state.legal_actions if isinstance(a, Bid)]
    assert bot._p_true(state, Bid(4, 6), False, credit=True) != bot._p_true(state, Bid(4, 6), False)
    best, p = bot._best_bid(state, bids, False)
    assert p == bot._p_true(state, best, False, credit=True)
    bot.FIX_CREDIT_CONSISTENT = False
    best, p = bot._best_bid(state, bids, False)
    assert p == bot._p_true(state, best, False)


def test_dudo_on_a_bid_it_does_not_believe():
    bot = SonnyBoy()
    state = _state([2, 3], [ActionRecord("B", Bid(5, 6))], Bid(5, 6), others=(("A", 1), ("B", 1)))
    assert bot.play(state) == Dudo()
