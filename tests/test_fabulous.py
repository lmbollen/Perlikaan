"""Fabulous: a softer trust prior, learned once per face per round, no Palifico followers."""

import perudo.bots.fabulous as fabulous_module
from perudo.bots.fabulous import Fabulous
from perudo.models import ActionRecord, Bid, BidEvent, Observation, PublicPlayer, RevealEvent, RoundEndedEvent
from perudo.rules import legal_actions

ME = Fabulous.name


def _state(my_dice, history=(), current_bid=None, others=(("A", 5), ("B", 5)), palifico=False):
    players = (PublicPlayer(0, ME, len(my_dice), True),
               *(PublicPlayer(i + 1, n, c, True) for i, (n, c) in enumerate(others)))
    total = sum(p.dice_count for p in players if p.alive)
    return Observation(
        my_id=0, my_dice=tuple(my_dice), my_dice_count=len(my_dice), players=players,
        current_bid=current_bid, palifico=palifico, round_number=1, history=tuple(history),
        legal_actions=tuple(legal_actions(current_bid, palifico, None, total)),
    )


def _reveal(bot, dice, bid):
    bot.observe(RevealEvent(dice=dice, current_bid=bid, actual_count=0, challenge_type="dudo",
                            challenger=ME, bidder=dice[-1][0], result="bidder_wrong"))


def _p_true(bot, state, face, quantity):
    others = [p for p in state.players if p.alive and p.id != state.my_id]
    return bot._prob_at_least(state, others, bot._collect_hints(state), face, quantity)


def test_plays_a_legal_action():
    bot = Fabulous()
    for state in (_state([2, 3, 4, 5, 6]),
                  _state([6, 6, 1, 2, 3], [ActionRecord("B", Bid(3, 6))], Bid(3, 6)),
                  _state([4], [ActionRecord("B", Bid(2, 4))], Bid(2, 4), others=(("B", 1),), palifico=True)):
        assert bot.play(state) in state.legal_actions


def test_trust_starts_below_certainty(monkeypatch):
    """FIX_TRUST: a bid is a likely match, not a sure one."""
    prior = fabulous_module.TRUST_PRIOR_HITS / fabulous_module.TRUST_PRIOR_TRIES
    assert abs(Fabulous()._trust("A") - prior) < 1e-9 and prior < 1.0
    monkeypatch.setattr(fabulous_module, "FIX_TRUST", False)
    assert Fabulous()._trust("A") == 1.0


def test_a_one_die_opener_can_be_doubted(monkeypatch):
    """With trust 1.0 the single die of a 1-die bidder was pinned as a sure
    match, so 2x4 against our one four was true with certainty."""
    state = _state([4], [ActionRecord("B", Bid(2, 4))], Bid(2, 4), others=(("B", 1),))
    assert _p_true(Fabulous(), state, 4, 2) < 1.0
    monkeypatch.setattr(fabulous_module, "FIX_TRUST", False)
    assert _p_true(Fabulous(), state, 4, 2) == 1.0


def test_learns_once_per_face_per_round(monkeypatch):
    """Three raises on sixes by the same player are one sample, not three."""
    dice = ((ME, (2, 3)), ("B", (2, 3, 4)), ("A", (6, 3, 3, 4, 5)))
    bids = [("A", Bid(3, 6)), ("B", Bid(4, 6)), ("A", Bid(5, 6)), ("B", Bid(6, 6)), ("A", Bid(7, 6)), ("A", Bid(3, 2))]
    bot = Fabulous()
    for player, bid in bids:
        bot.observe(BidEvent(player, bid))
    _reveal(bot, dice, Bid(3, 2))
    hits, tries = bot._hint_stats["A"]
    assert (hits, tries) == (fabulous_module.TRUST_PRIOR_HITS + 1, fabulous_module.TRUST_PRIOR_TRIES + 2)  # 6s hit, 2s miss
    assert bot._hint_stats["B"][1] == fabulous_module.TRUST_PRIOR_TRIES + 1
    monkeypatch.setattr(fabulous_module, "FIX_TRUST", False)
    old = Fabulous()
    for player, bid in bids:
        old.observe(BidEvent(player, bid))
    _reveal(old, dice, Bid(3, 2))
    assert old._hint_stats["A"][1] == fabulous_module.HINT_PRIOR_TRIES + 4


def test_palifico_followers_are_not_graded(monkeypatch):
    """In Palifico only the opener chose the face; forced raises say nothing."""
    bot = Fabulous()
    bot.observe(RoundEndedEvent(next_starting_player="A", palifico_next=True))
    bot.observe(BidEvent("A", Bid(2, 4)))
    bot.observe(BidEvent("B", Bid(3, 4)))
    _reveal(bot, ((ME, (2, 3)), ("A", (4, 4)), ("B", (2, 3, 5))), Bid(3, 4))
    assert bot._hint_stats["A"][1] == fabulous_module.TRUST_PRIOR_TRIES + 1
    assert "B" not in bot._hint_stats
    monkeypatch.setattr(fabulous_module, "FIX_TRUST", False)
    old = Fabulous()
    old.observe(RoundEndedEvent(next_starting_player="A", palifico_next=True))
    old.observe(BidEvent("A", Bid(2, 4)))
    old.observe(BidEvent("B", Bid(3, 4)))
    _reveal(old, ((ME, (2, 3)), ("A", (4, 4)), ("B", (2, 3, 5))), Bid(3, 4))
    assert old._hint_stats["B"][1] == fabulous_module.HINT_PRIOR_TRIES + 1


def test_pinned_die_that_is_no_match_leaves_the_pool():
    """What the docstring of _player_pmf says: a pinned non-match is not a coin."""
    pmf = Fabulous()._player_pmf(1, 1 / 3, [0.5])
    assert [round(x, 9) for x in pmf] == [0.5, 0.5]
    pmf = Fabulous()._player_pmf(2, 1 / 3, [0.5])
    assert abs(pmf[0] - 0.5 * 2 / 3) < 1e-9          # non-match, other die no match
