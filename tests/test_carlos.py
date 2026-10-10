"""Carlos: Markov-chain move choice on learned opponent models."""

from perudo.bots.carlos import CarlosBot, convolve, min_quantity, tail
from perudo.models import PACO, ActionRecord, Bid, Calza, Dudo, Observation, PublicPlayer
from perudo.rules import legal_actions


def _state(my_dice, history=(), current_bid=None, others=(("A", 5), ("B", 5)), palifico=False):
    players = (PublicPlayer(0, "Carlos", len(my_dice), True),
               *(PublicPlayer(i + 1, n, c, True) for i, (n, c) in enumerate(others)))
    total = sum(p.dice_count for p in players)
    return Observation(
        my_id=0, my_dice=tuple(my_dice), my_dice_count=len(my_dice), players=players,
        current_bid=current_bid, palifico=palifico, round_number=1, history=tuple(history),
        legal_actions=tuple(legal_actions(current_bid, palifico, None, total)),
    )


def test_helpers():
    assert convolve([0.5, 0.5], [0.5, 0.5]) == [0.25, 0.5, 0.25]
    assert tail([0.25, 0.5, 0.25], 1) == 0.75
    assert min_quantity(Bid(3, 4), 5, False) == 3
    assert min_quantity(Bid(3, 4), 2, False) == 4
    assert min_quantity(Bid(3, 4), PACO, False) == 2
    assert min_quantity(Bid(2, PACO), 3, False) == 5


def test_plays_a_legal_action():
    bot = CarlosBot()
    for state in (_state([2, 3, 4, 5, 6]),
                  _state([6, 6, 1, 2, 3], [ActionRecord("B", Bid(3, 6))], Bid(3, 6))):
        assert bot.play(state) in state.legal_actions


def test_dudo_on_an_impossible_bid():
    bot = CarlosBot()
    state = _state([2, 3], [ActionRecord("B", Bid(5, 6))], Bid(5, 6), others=(("A", 1), ("B", 1)))
    assert bot.play(state) == Dudo()


def test_learns_that_a_player_bids_what_it_holds():
    bot = CarlosBot()
    from perudo.models import BidEvent, DudoEvent, RevealEvent
    for _ in range(30):
        bot.observe(BidEvent("A", Bid(3, 6)))
        bot.observe(DudoEvent("B"))
        bot.observe(RevealEvent(dice=(("Carlos", (2,)), ("A", (6, 6, 6, 2, 3)), ("B", (2, 3, 4, 5, 2))),
                                current_bid=Bid(3, 6), actual_count=3, challenge_type="dudo",
                                challenger="B", bidder="A", result="bidder_correct"))
    state = _state([2, 3], [ActionRecord("A", Bid(3, 6))], Bid(3, 6))
    post = bot._posteriors(state)["A"][6]
    assert sum(k * w for k, w in enumerate(post)) > 2.5
