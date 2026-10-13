"""
Seats leaving a game while it is going on (PerudoGame.forfeit and the
Forfeit exception): they lose at once, the round they left is abandoned,
and the others play on.
"""

import random as random_module

from perudo.game import Forfeit, PerudoGame
from perudo.models import DieLostEvent, PlayerEliminatedEvent, RoundEndedEvent


class FirstBot:
    def __init__(self, name):
        self.name = name
        self.events = []

    def play(self, state):
        return state.legal_actions[0]

    def observe(self, event):
        self.events.append(event)


class LeavingBot(FirstBot):
    """Walks away from the table on its turn in round `leave_in`."""

    def __init__(self, name, leave_in):
        super().__init__(name)
        self.leave_in = leave_in

    def play(self, state):
        if state.round_number >= self.leave_in:
            raise Forfeit()
        return super().play(state)


def test_a_seat_that_raises_forfeit_loses_at_once_and_the_rest_play_on():
    bots = [FirstBot("Anna"), LeavingBot("Ben", leave_in=2), FirstBot("Carla")]
    game = PerudoGame(bots=bots, seed=80417)
    events = []
    game.listeners.append(events.append)
    result = game.play()
    assert result.standings[-1] == "Ben"                 # out first, so last
    assert sorted(result.standings) == ["Anna", "Ben", "Carla"]
    out = [e for e in events if isinstance(e, PlayerEliminatedEvent)]
    assert out[0].player == "Ben"
    # Ben's leaving cost nobody a die: the round was abandoned before a reveal.
    ben_out = events.index(out[0])
    lost_before = [e for e in events[:ben_out] if isinstance(e, DieLostEvent)]
    rounds_before = [e for e in events[:ben_out] if isinstance(e, RoundEndedEvent)]
    assert len(lost_before) == len(rounds_before)       # one die per finished round, none for the abandoned one


def test_forfeit_from_another_thread_takes_effect_before_the_next_turn():
    seen_after = []

    class Watcher(FirstBot):
        def play(self, state):
            if state.round_number == 1 and not seen_after:
                game.forfeit("Dirk")
                seen_after.append(True)
            elif seen_after and len(seen_after) == 1:
                seen_after.append([p.name for p in state.players if p.alive])
            return super().play(state)

    bots = [Watcher("Anna"), FirstBot("Ben"), FirstBot("Carla"), FirstBot("Dirk")]
    game = PerudoGame(bots=bots, seed=5521093)
    result = game.play()
    assert "Dirk" not in seen_after[1]
    assert result.standings[-1] == "Dirk"


def test_everyone_but_one_leaving_ends_the_game_with_the_last_seat_winning():
    bots = [FirstBot("Anna"), FirstBot("Ben"), FirstBot("Carla")]
    game = PerudoGame(bots=bots, seed=3170665)
    for name in ("Anna", "Ben", "Carla"):
        game.forfeit(name)
    seats = [p.name for p in game.players]                # seating is shuffled
    result = game.play()
    assert result.rounds_played == 0
    assert result.winner == seats[-1]                     # never put out: the last one standing
    assert result.standings == (seats[2], seats[1], seats[0])


def test_forfeiting_a_seat_that_is_already_out_changes_nothing():
    def play(forfeit_late):
        random_module.seed(0)
        bots = [FirstBot("Anna"), FirstBot("Ben"), FirstBot("Carla")]
        game = PerudoGame(bots=bots, seed=6402981)
        out = []

        def listener(event):
            if isinstance(event, PlayerEliminatedEvent):
                out.append(event.player)
                if forfeit_late:
                    game.forfeit(event.player)

        game.listeners.append(listener)
        return game.play(), out

    assert play(False) == play(True)


def test_without_forfeits_a_game_is_unchanged():
    def run():
        bots = [FirstBot(n) for n in ("Anna", "Ben", "Carla", "Dirk")]
        game = PerudoGame(bots=bots, seed=9035527)
        return game.play(), [type(e).__name__ for e in bots[0].events]

    assert run() == run()


def test_a_turn_cut_short_by_someone_leaving_abandons_the_round():
    from perudo.game import Interrupt

    class Thinker(FirstBot):
        """Is thinking when Carla walks away, and gives up its turn."""

        def play(self, state):
            if not left:
                left.append(True)
                game.forfeit("Carla")
                raise Interrupt()
            return super().play(state)

    left = []
    bots = [Thinker("Anna"), FirstBot("Ben"), FirstBot("Carla")]
    game = PerudoGame(bots=bots, seed=2774613)
    events = []
    game.listeners.append(events.append)
    result = game.play()
    assert result.standings[-1] == "Carla"
    first_out = next(i for i, e in enumerate(events) if isinstance(e, PlayerEliminatedEvent))
    assert not any(isinstance(e, DieLostEvent) for e in events[:first_out])    # nobody paid for it
