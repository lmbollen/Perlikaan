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
    full = bot._face_model(state, 6, alive, 12, alive[2])[0]
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


def _round(bot, dice):
    """A bids 3x6, B raises to 4x6, C calls Dudo on B."""
    bot.observe(BidEvent("A", Bid(3, 6)))
    bot.observe(BidEvent("B", Bid(4, 6)))
    bot.observe(DudoEvent("C"))
    bot.observe(_reveal(Bid(4, 6), dice, challenger="C", bidder="B"))


DICE = [(ME, (2, 3)), ("A", (6, 6, 1, 2, 3)), ("B", (2, 3, 4, 5, 2)), ("C", (2, 3, 4, 5, 6))]


def test_a_new_name_starts_from_the_prior_not_from_other_bots():
    bot = CarlosDosBot()
    for _ in range(30):
        _round(bot, DICE)
    fresh = bot._player("Z")
    assert fresh.dudo_w == list(CarlosDosBot.DUDO_PRIOR)
    assert fresh.calza_w == list(CarlosDosBot.CALZA_PRIOR)
    lam = bot._likelihood("Z", "choice")
    generic = CarlosDosBot.GENERIC_LAMBDA["3+"]["choice"]
    assert all(abs(a - b) < 1e-9 for a, b in zip(lam, generic))   # the generic model, nothing from A, B, C


def test_every_reveal_updates_every_name_in_it():
    bot = CarlosDosBot()
    before = {n: list(bot._player(n).dudo_w) for n in "ABC"}
    _round(bot, DICE)
    a, b, c = (bot._player(n) for n in "ABC")
    assert a.source.honesty["choice"][1] == 1                 # A's bid: source model
    assert b.dudo_w != before["B"] and b.n == 1               # B raised on A's bid: reaction model
    assert c.dudo_w != before["C"] and c.n == 1               # C called Dudo on B: reaction model
    assert a.n == 0                                           # A did not react to anything


def test_models_decay_towards_the_prior():
    bot = CarlosDosBot()
    for _ in range(20):
        _round(bot, DICE)
    w = list(bot._player("C").dudo_w)
    bot.observe(GameEndedEvent(winner="A", standings=("A", "B", "C", ME)))
    prior = CarlosDosBot.DUDO_PRIOR
    after = bot._player("C").dudo_w
    for i in range(len(w)):
        assert abs(after[i] - (prior[i] + CarlosDosBot.GAME_DECAY * (w[i] - prior[i]))) < 1e-9
