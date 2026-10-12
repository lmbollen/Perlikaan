"""Carlos Dos: base model of the dice plus learned neighbour reactions."""

from perudo.bots.carlos_dos import CarlosDosBot, _conv, _min_quantity, _tail
from perudo.models import (
    PACO, ActionRecord, Bid, BidEvent, Calza, Dudo, DudoEvent, GameEndedEvent, Observation,
    PublicPlayer, RevealEvent,
)
from perudo.rules import legal_actions

ME = "Carlos Dos"


def _state(my_dice, history=(), current_bid=None, others=(("A", 5), ("B", 5)), palifico=False):
    players = (PublicPlayer(0, ME, len(my_dice), True),
               *(PublicPlayer(i + 1, n, c, True) for i, (n, c) in enumerate(others)))
    total = sum(p.dice_count for p in players)
    face = current_bid.face if current_bid else None
    return Observation(
        my_id=0, my_dice=tuple(my_dice), my_dice_count=len(my_dice), players=players,
        current_bid=current_bid, palifico=palifico, round_number=1, history=tuple(history),
        legal_actions=tuple(legal_actions(current_bid, palifico, face, total)),
    )


def _reveal(bid, dice, challenger="B", bidder="A", kind="dudo"):
    table = [d for _, h in dice for d in h]
    actual = sum(1 for d in table if d == bid.face or d == PACO)
    result = ("bidder_correct" if actual >= bid.quantity else "bidder_wrong") if kind == "dudo" \
        else ("correct" if actual == bid.quantity else "wrong")
    return RevealEvent(dice=tuple(dice), current_bid=bid, actual_count=actual, challenge_type=kind,
                       challenger=challenger, bidder=bidder, result=result)


def test_helpers():
    assert _conv([0.5, 0.5], [0.5, 0.5]) == [0.25, 0.5, 0.25]
    assert _tail([0.25, 0.5, 0.25], 1) == 0.75
    assert _min_quantity(Bid(3, 4), 5, False) == 3
    assert _min_quantity(Bid(3, 4), 2, False) == 4
    assert _min_quantity(Bid(3, 4), PACO, False) == 2
    assert _min_quantity(Bid(2, PACO), 3, False) == 5


def test_plays_a_legal_action():
    bot = CarlosDosBot()
    for state in (_state([2, 3, 4, 5, 6]),
                  _state([6, 6, 1, 2, 3], [ActionRecord("B", Bid(3, 6))], Bid(3, 6)),
                  _state([2], [ActionRecord("B", Bid(1, 2))], Bid(1, 2), others=(("A", 3), ("B", 2)), palifico=True),
                  _state([3, 3], [ActionRecord("A", Bid(2, PACO))], Bid(2, PACO))):
        for _ in range(20):
            assert bot.play(state) in state.legal_actions


def test_dudo_on_an_impossible_bid():
    bot = CarlosDosBot()
    state = _state([2, 3], [ActionRecord("B", Bid(5, 6))], Bid(5, 6), others=(("A", 1), ("B", 1)))
    assert bot.play(state) == Dudo()


def test_opens_on_a_face_it_holds_or_a_bluff():
    bot = CarlosDosBot()
    state = _state([6, 6, 6, 2, 3])
    action = bot.play(state)
    assert isinstance(action, Bid) and action.quantity >= 3


def test_a_real_bid_raises_the_count():
    bot = CarlosDosBot()
    state = _state([2, 3], [ActionRecord("A", Bid(4, 6))], Bid(4, 6))
    alive = list(state.players)
    full, _, _ = bot._face_model(state, 6, alive, 12, alive[2])
    prior = bot._face_model(_state([2, 3]), 6, alive, 12, alive[2])[0]
    assert _tail(full, 4) > _tail(prior, 4)


def test_learns_that_a_player_bluffs():
    bot = CarlosDosBot()
    state = _state([2, 3], [ActionRecord("A", Bid(5, 6))], Bid(5, 6))
    alive = list(state.players)
    trusted = _tail(bot._face_model(state, 6, alive, 12, alive[2])[0], 5)
    for _ in range(20):
        bot.observe(BidEvent("A", Bid(5, 6)))
        bot.observe(DudoEvent("B"))
        bot.observe(_reveal(Bid(5, 6), [(ME, (2, 3)), ("A", (2, 3, 4, 5, 2)), ("B", (2, 3, 4, 5, 2))]))
    assert bot.honesty("A", "choice") < 0.3
    assert _tail(bot._face_model(state, 6, alive, 12, alive[2])[0], 5) < trusted
    assert bot.play(state) == Dudo()


def test_learns_that_a_player_calls_dudo_on_us():
    bot = CarlosDosBot()
    before = bot._player("B").dudo_w[0]
    for _ in range(10):
        bot.observe(BidEvent(ME, Bid(3, 6)))
        bot.observe(DudoEvent("B"))
        bot.observe(_reveal(Bid(3, 6), [(ME, (6, 6, 2)), ("A", (2, 3, 4, 5, 2)), ("B", (6, 3, 4, 5, 2))],
                            bidder=ME))
    assert bot._player("B").dudo_w[0] > before
    assert bot._player("B").trust > bot.TRUST_START        # our bids were true


def test_trust_falls_after_a_caught_bluff():
    bot = CarlosDosBot()
    bot.observe(BidEvent(ME, Bid(6, 6)))
    bot.observe(DudoEvent("B"))
    bot.observe(_reveal(Bid(6, 6), [(ME, (2, 3)), ("A", (2, 3, 4, 5, 2)), ("B", (6, 3, 4, 5, 2))], bidder=ME))
    assert bot._player("B").trust < bot.TRUST_START


def test_learning_survives_a_new_instance_and_decays():
    CarlosDosBot.reset_memory()
    first = CarlosDosBot()
    first.observe(BidEvent("A", Bid(4, 6)))
    first.observe(DudoEvent("B"))
    first.observe(_reveal(Bid(4, 6), [(ME, (2, 3)), ("A", (2, 3, 4, 5, 2)), ("B", (2, 3, 4, 5, 2))]))
    second = CarlosDosBot()
    second._ensure_memory()
    assert second.players is first.players and "A" in second.players
    bids = second.players["A"].source.honesty["choice"][1]
    second.observe(GameEndedEvent(winner="A", standings=("A", "B", ME)))
    assert second.players["A"].source.honesty["choice"][1] == bids * CarlosDosBot.GAME_DECAY
    CarlosDosBot.reset_memory()
    assert "A" not in CarlosDosBot().players


def test_calza_when_the_bid_is_exactly_right():
    bot = CarlosDosBot()
    # A opens 5x6 with one die: the model reads that as A holding a six (or Paco).
    # We hold four sixes: 5x6 is then exact, and 6x6 is impossible.
    state = _state([6, 6, 6, 6, 2], [ActionRecord("A", Bid(5, 6))], Bid(5, 6), others=(("A", 1),))
    assert bot.play(state) == Calza()
