"""Trihard: learned honesty keeps its prior, Dudo thresholds learn on the applied scale."""

from perudo.bots.trihard import HONESTY_DEFAULT, HONESTY_GRID, Trihard
from perudo.models import Bid, BidEvent, RevealEvent, RoundEndedEvent


def _reveal(bot, bids, dice, challenger="B"):
    """Play the bids of one round in front of `bot`, then reveal `dice`."""
    for player, bid in bids:
        bot.observe(BidEvent(player, bid))
    bot.observe(RevealEvent(
        dice=tuple((name, tuple(hand)) for name, hand in dice.items()),
        current_bid=bids[-1][1], actual_count=0, challenge_type="dudo",
        challenger=challenger, bidder=bids[-1][0], result="bidder_wrong",
    ))
    bot.observe(RoundEndedEvent(next_starting_player="A", palifico_next=False))


# A opens 2x3 at a 25-dice table: that says next to nothing about A's dice.
UNINFORMATIVE = (
    [("A", Bid(2, 3))],
    {"A": (2, 4, 5, 6, 6), "B": (1, 2, 3, 4, 5), "C": (3, 3, 3, 3, 3), "D": (2, 2, 2, 2, 2),
     "Trihard": (3, 3, 3, 3, 3)},
)


def test_uninformative_reveal_keeps_the_default_honesty():
    bot = Trihard()
    _reveal(bot, *UNINFORMATIVE)
    assert bot._honesty("A") == HONESTY_DEFAULT


def test_without_the_fix_one_reveal_jumped_to_the_edge_of_the_grid():
    bot = Trihard()
    bot.FIX_HONESTY_PRIOR = False
    _reveal(bot, *UNINFORMATIVE)
    assert bot._honesty("A") in (HONESTY_GRID[0], HONESTY_GRID[-1])


def test_repeated_clear_evidence_overrules_the_prior():
    liar, honest = Trihard(), Trihard()
    for _ in range(10):
        # A opens 3x6 holding no 6 at all ...
        _reveal(liar, [("A", Bid(3, 6))], {"A": (2, 2, 3, 4, 5), "B": (1, 2, 3, 4, 5), "Trihard": (3, 3)})
        # ... or holding three of them.
        _reveal(honest, [("A", Bid(3, 6))], {"A": (6, 6, 6, 2, 3), "B": (1, 2, 3, 4, 5), "Trihard": (3, 3)})
    assert liar._honesty("A") < HONESTY_DEFAULT
    assert honest._honesty("A") > HONESTY_DEFAULT


def _hints_seen_for(bot, bids, dice):
    """Hints handed to _player_pmf_for per stand-in while replaying the round."""
    seen = {}
    original = bot._player_pmf_for

    def spy(player, face, hints, *args, **kwargs):
        seen.setdefault(player.name, []).append({k: dict(v) for k, v in hints.items()})
        return original(player, face, hints, *args, **kwargs)

    bot._player_pmf_for = spy
    _reveal(bot, bids, dice, challenger="Trihard")
    return seen


ROUND = (
    [("A", Bid(5, 6)), ("B", Bid(6, 6))],
    {"A": (6, 6, 6, 6, 2), "B": (6, 6, 3, 4, 5), "Trihard": (2, 3, 4, 5, 6)},
)


def test_threshold_learning_counts_the_faced_bid_as_a_hint():
    # B faced A's 5x6: B's estimate of it must include A's bid as a hint,
    # exactly as our own bid is a hint in _best_bid's their_view.
    seen = _hints_seen_for(Trihard(), *ROUND)
    assert any(h.get("A") == {6: 5} for h in seen["A"])


def test_without_the_fix_the_faced_bid_was_left_out():
    bot = Trihard()
    bot.FIX_THRESHOLD_SCALE = False
    seen = _hints_seen_for(bot, *ROUND)
    assert all("A" not in h for h in seen["A"])


def test_faced_bid_as_hint_makes_a_raise_less_surprising():
    # B raised on 5x6 while A's bid supports the sixes: with the hint, B's
    # raise was predictable and its threshold moves less than without.
    with_fix, without = Trihard(), Trihard()
    without.FIX_THRESHOLD_SCALE = False
    _reveal(with_fix, *ROUND, challenger="Trihard")
    _reveal(without, *ROUND, challenger="Trihard")
    start = Trihard()._call_theta["B"]
    assert without._call_theta["B"] < with_fix._call_theta["B"] <= start
