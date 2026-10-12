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


# -- Trihard's engine inside The Don: the same two fixes ---------------------

from perudo.bots.the_don import HONESTY_DEFAULT, HONESTY_GRID
from perudo.models import BidEvent, GameEndedEvent, RevealEvent, RoundEndedEvent


def _reveal(bot, bids, dice, challenger="B"):
    for player, bid in bids:
        bot.observe(BidEvent(player, bid))
    bot.observe(RevealEvent(
        dice=tuple((name, tuple(hand)) for name, hand in dice.items()),
        current_bid=bids[-1][1], actual_count=0, challenge_type="dudo",
        challenger=challenger, bidder=bids[-1][0], result="bidder_wrong",
    ))
    bot.observe(RoundEndedEvent(next_starting_player="A", palifico_next=False))


UNINFORMATIVE = (
    [("A", Bid(2, 3))],
    {"A": (2, 4, 5, 6, 6), "B": (1, 2, 3, 4, 5), "C": (3, 3, 3, 3, 3), "D": (2, 2, 2, 2, 2),
     "The Don": (3, 3, 3, 3, 3)},
)


def test_uninformative_reveal_keeps_the_default_honesty():
    bot = TheDonBot()
    _reveal(bot, *UNINFORMATIVE)
    assert bot._honesty("A") == HONESTY_DEFAULT
    bot.FIX_HONESTY_PRIOR = False
    assert bot._honesty("A") in (HONESTY_GRID[0], HONESTY_GRID[-1])


def test_game_decay_fades_evidence_back_towards_the_default():
    bot = TheDonBot()
    for _ in range(12):           # enough bluffs to overrule the prior weight on the default
        _reveal(bot, [("A", Bid(3, 6))], {"A": (2, 2, 3, 4, 5), "B": (1, 2, 3, 4, 5), "The Don": (3, 3)})
    assert bot._honesty("A") < HONESTY_DEFAULT
    bot.GAME_DECAY = 0.5
    for _ in range(4):
        bot.observe(GameEndedEvent(winner="A", standings=("A", "B", "The Don")))
    assert bot._honesty("A") == HONESTY_DEFAULT


def test_threshold_learning_counts_the_faced_bid_as_a_hint():
    bot = TheDonBot()
    seen = []
    original = bot._player_pmf_for

    def spy(player, face, hints, *args, **kwargs):
        if player.name == "A":
            seen.append({k: dict(v) for k, v in hints.items()})
        return original(player, face, hints, *args, **kwargs)

    bot._player_pmf_for = spy
    _reveal(bot, [("A", Bid(5, 6)), ("B", Bid(6, 6))],
            {"A": (6, 6, 6, 6, 2), "B": (6, 6, 3, 4, 5), "The Don": (2, 3, 4, 5, 6)}, challenger="The Don")
    assert any(h.get("A") == {6: 5} for h in seen)

