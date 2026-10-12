"""Pokerface bluffs about faces it doesn't hold, but only with plausible bids."""

from perudo.bots.pokerface import PokerfaceBot
from perudo.bots.sniper import SniperBot
from perudo.models import PACO, ActionRecord, Bid, Dudo, Observation, PublicPlayer
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
        ActionRecord("A", Bid(1, 6)),
        ActionRecord("B", Bid(2, 6)),
    ]
    state = _state([2, 2, 3, 3, 4], history, current_bid=Bid(2, 6))
    action = bot.play(state)
    assert isinstance(action, Bid)
    assert action.face == 6


def test_no_bluff_in_palifico():
    bot = _always_bluff()
    state = _state([2], palifico=True)
    assert bot._bluff(state, [], 10, {f: 0.0 for f in range(1, 7)}, 0.75) == (None, None)


def test_bluff_bids_are_still_plausible():
    bot = _always_bluff()
    state = _state([2, 2, 3, 3, 4])
    hints = {f: 0.0 for f in range(1, 7)}
    bid = bot.play(state)
    assert bot._prob_true(state, bid, 10, hints) >= bot.BLUFF_TARGET


def test_chain_of_raises_counts_less_than_for_sniper():
    chain = [ActionRecord("A", Bid(2, PACO)), ActionRecord("B", Bid(3, PACO))]
    state = _state([2, 3], chain, current_bid=Bid(3, PACO))
    assert PokerfaceBot()._hint(state, PACO) < SniperBot()._hint(state, PACO)


def test_learned_hint_trusts_honest_and_ignores_bluffers():
    bot = PokerfaceBot()
    # Ten revealed rounds: A opens on 6 holding three of them, B opens on 6
    # holding none.
    dice = {"A": (6, 6, 6, 2, 3), "B": (2, 3, 4, 5, 5), "Pokerface": (2,)}
    for _ in range(10):
        bot.profiler.record_round(dice, [("A", Bid(1, 6)), ("Pokerface", Dudo())], False)
        bot.profiler.record_round(dice, [("B", Bid(1, 6)), ("Pokerface", Dudo())], False)
    honest = _state([2, 3], [ActionRecord("A", Bid(2, 6))], current_bid=Bid(2, 6))
    liar = _state([2, 3], [ActionRecord("B", Bid(2, 6))], current_bid=Bid(2, 6))
    assert bot._hint(honest, 6) > 1.0
    assert bot._hint(liar, 6) < 0


def test_target_is_bolder_against_a_player_who_never_challenges():
    bot = PokerfaceBot()
    bot.ADAPT_TARGET = True
    for _ in range(30):
        # A faces a false bid of 5x6 and raises; B faces one and calls Dudo.
        bot.profiler.record_round(
            {"A": (2,), "B": (3,), "Pokerface": (4,)},
            [("Pokerface", Bid(2, 2)), ("A", Bid(3, 2)), ("B", Dudo())],
            False,
        )
    state = _state([2, 3, 4])   # seats: Pokerface, A, B -> A is next
    assert bot._next_player(state) == "A"
    assert bot._target(state, bot.TARGET) < bot.TARGET


def test_calza_only_when_it_gains_a_die():
    bot = PokerfaceBot()
    assert bot._calza_gains(_state([2, 3, 4]))
    assert not bot._calza_gains(_state([2, 3, 4, 5, 6]))


def test_trap_bid_scores_higher_against_a_dudo_happy_opponent():
    # We hold four 6s: 4x6 is certain for us. To the next player, holding no
    # 6s, it looks unlikely, so a Dudo-happy player will challenge and lose.
    bot = PokerfaceBot()
    state = _state([6, 6, 6, 6, 2])
    total = 15
    hints = {f: 0.0 for f in range(1, 7)}
    args = (state, Bid(4, 6), total - 5, hints, "A", 5, 0.0)
    happy = bot._bid_ev(*args, 0.8, 0.0, 1.0)
    shy = bot._bid_ev(*args, 0.05, 0.0, 1.0)
    assert happy > shy >= 0


def test_lookahead_prefers_true_bids():
    bot = PokerfaceBot()
    bot.LOOKAHEAD = True
    action = bot.play(_state([6, 6, 6, 6, 2]))
    assert isinstance(action, Bid)
    assert bot._prob_true(_state([6, 6, 6, 6, 2]), action, 10, {f: 0.0 for f in range(1, 7)}) >= 0.5


# -- fixes -------------------------------------------------------------------

from perudo.bot_loader import instantiate_bots
from perudo.bots.pokerface import Profile
from perudo.models import BidEvent, Calza, DudoEvent, GameEndedEvent, RevealEvent


def _two_player_state(my_dice, their_n, bid):
    players = (PublicPlayer(0, "Pokerface", len(my_dice), True), PublicPlayer(1, "A", their_n, True))
    total = len(my_dice) + their_n
    return Observation(
        my_id=0, my_dice=tuple(sorted(my_dice)), my_dice_count=len(my_dice), players=players,
        current_bid=bid, palifico=False, round_number=1, history=(ActionRecord("A", bid),),
        legal_actions=tuple(legal_actions(bid, False, None, total)),
    )


def test_certain_raise_beats_a_long_shot_calza():
    # We hold 4, 4 and a Paco: 3x4 is certain. A has one die and bid 2x2,
    # which is exact only if that die is a 2 or a Paco (1 in 3). The old
    # comparison gave the raise no upside and went for Calza anyway.
    state = _two_player_state([4, 6, 1, 4], 1, Bid(2, 2))
    bot = PokerfaceBot()
    action = bot.play(state)
    assert action == Bid(3, 4)
    PokerfaceBot.reset_memory()
    old = PokerfaceBot()
    old.FIX_RAISE_UPSIDE = False
    assert old.play(state) == Calza()


def test_rng_is_seeded():
    assert PokerfaceBot()._rng.random() == PokerfaceBot()._rng.random()


def test_profile_summary_survives_decay():
    prof = Profile("A")
    prof.rounds = 3
    prof.decay(0.95)
    assert "rondes" in prof.summary()


def test_report_after_a_game_ended():
    bot = PokerfaceBot()
    bot.observe(BidEvent("A", Bid(2, 6)))
    bot.observe(DudoEvent("B"))
    bot.observe(RevealEvent(dice=(("Pokerface", (2,)), ("A", (6, 6)), ("B", (3, 4))), current_bid=Bid(2, 6),
                            actual_count=2, challenge_type="dudo", challenger="B", bidder="A",
                            result="bidder_correct"))
    bot.observe(GameEndedEvent(winner="A", standings=("A", "B", "Pokerface")))
    assert "A" in bot.profiler.report()


def test_second_copy_profiles_the_others_and_not_itself():
    first, second = instantiate_bots(["PokerfaceBot", "PokerfaceBot"])
    assert (first.name, second.name) == ("Pokerface", "Pokerface 2")
    assert first.profiler.me == "Pokerface"
    assert second.profiler.me == "Pokerface 2"

    dice = (("Pokerface", (6, 6, 2)), ("Pokerface 2", (3, 4, 5)), ("A", (2, 2)))
    for bot in (first, second):
        bot.observe(BidEvent("Pokerface", Bid(2, 6)))
        bot.observe(BidEvent("Pokerface 2", Bid(3, 6)))
        bot.observe(DudoEvent("A"))
        bot.observe(RevealEvent(dice=dice, current_bid=Bid(3, 6), actual_count=2, challenge_type="dudo",
                                challenger="A", bidder="Pokerface 2", result="bidder_wrong"))
    assert set(first.profiler.profiles) == {"Pokerface 2", "A"}
    assert set(second.profiler.profiles) == {"Pokerface", "A"}
