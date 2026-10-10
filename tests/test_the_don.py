"""The Don: attacks strong neighbours, protects weak ones."""

from perudo.bots.the_don import TheDonBot
from perudo.models import ActionRecord, Bid, Observation, PublicPlayer
from perudo.rules import legal_actions


def _state(my_dice, counts, history=(), current_bid=None, my_id=1):
    """counts: dice per seat; we sit at my_id."""
    players = tuple(PublicPlayer(i, "The Don" if i == my_id else f"P{i}", c, c > 0)
                    for i, c in enumerate(counts))
    total = sum(counts)
    return Observation(
        my_id=my_id, my_dice=tuple(my_dice), my_dice_count=len(my_dice), players=players,
        current_bid=current_bid, palifico=False, round_number=1, history=tuple(history),
        legal_actions=tuple(legal_actions(current_bid, False, None, total)),
    )


def test_neighbours_and_protection():
    state = _state([2, 3, 4, 5, 6], [2, 5, 3, 5])
    bot = TheDonBot()
    left, right = bot._neighbour(state, -1), bot._neighbour(state, 1)
    assert (left.name, right.name) == ("P0", "P2")
    assert bot._protected(state, left)          # 2 dice vs our 5
    assert bot._protected(state, right)         # 3 dice: exactly 2 fewer
    assert not bot._protected(state, PublicPlayer(9, "x", 4, True))


def test_low_on_dice_one_fewer_is_enough():
    state = _state([2, 3], [1, 2, 2])
    bot = TheDonBot()
    assert bot._protected(state, state.players[0])
    assert not bot._protected(state, state.players[2])


def test_gives_a_protected_neighbour_an_easy_bid():
    bot = TheDonBot()
    # Right neighbour (P2) has 1 die vs our 5: protected.
    state = _state([6, 6, 6, 2, 3], [5, 5, 1, 5])
    action = bot.play(state)
    assert isinstance(action, Bid)
    others = [p for p in state.players if p.alive and p.id != state.my_id]
    p = bot._prob_at_least(state, others, bot._collect_hints(state), action.face, action.quantity)
    assert p >= bot.PROTECT_MIN_TRUE
