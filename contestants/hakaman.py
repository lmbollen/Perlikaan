"""
HakamanBot: plays exactly like MinRaiseBot, but cheats.

Before committing to its (minimum) raise it peeks at everyone's dice. If
the bid would not hold up to a Dudo, it quietly rewrites other players'
dice until it does (and its own, if the opponents' dice aren't enough).

Bots normally only ever see an immutable Observation, so this bot reaches
the engine by walking up the call stack to the PerudoGame instance that
called play(). The engine counts dice straight from those lists at reveal
time, so the edits stick for the rest of the round.
"""

import inspect

from perudo.models import PACO, Bid


def _count_matches(dice_values, face, palifico):
    """How many dice count towards a bid on `face`."""
    if palifico:
        return sum(1 for d in dice_values if d == face)
    if face == PACO:
        return sum(1 for d in dice_values if d == PACO)
    return sum(1 for d in dice_values if d == face or d == PACO)


class HakamanBot:
    name = "Hakaman"
    count = 1

    def on_game_start(self, info):
        pass

    def play(self, state):
        action = self._choose(state)
        if isinstance(action, Bid):
            self._rig(state, action)
        return action

    def observe(self, event):
        pass

    def on_game_end(self, result):
        pass

    # -- MinRaiseBot's strategy -------------------------------------------

    def _choose(self, state):
        possible_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        if possible_bids:
            return self._minraise(possible_bids)

        # Extremely rare edge case: in a Palifico round the quantity can
        # reach the total number of dice in play, at which point no
        # further bid is legal at all (the face can never change) --
        # only Dudo/Calza remain. With no raise left to make, this is the
        # one situation where MinRaiseBot is forced to pick one anyway.
        return state.legal_actions[0]

    @staticmethod
    def _claim_strength(bid):
        return bid.quantity * 2 if bid.face == PACO else bid.quantity

    @classmethod
    def _minraise(cls, possible_bids):
        return min(
            possible_bids, key=lambda b: (cls._claim_strength(b), b.face == PACO, b.face)
        )

    # -- the cheat --------------------------------------------------------

    @staticmethod
    def _find_game():
        frame = inspect.currentframe()
        try:
            while frame is not None:
                candidate = frame.f_locals.get("self")
                if type(candidate).__name__ == "PerudoGame":
                    return candidate
                frame = frame.f_back
        finally:
            del frame
        return None

    def _rig(self, state, bid):
        """Rewrite dice until `bid` holds. Returns whether anything was changed."""
        game = self._find_game()
        if game is None:
            return False

        alive = [p for p in game.players if p.alive]
        all_dice = [d for p in alive for d in p.dice]
        shortfall = bid.quantity - _count_matches(all_dice, bid.face, state.palifico)
        if shortfall <= 0:
            return False

        # A die "matches" if it counts towards the bid; Pacos are wild for
        # normal faces outside Palifico, so leave those alone.
        def matches(d):
            return _count_matches([d], bid.face, state.palifico) == 1

        # Opponents first; our own dice only as a last resort, for when the
        # bid has climbed past what the opponents' dice alone can cover.
        alive.sort(key=lambda p: p.seat_id == state.my_id)
        for player in alive:
            for i, d in enumerate(player.dice):
                if shortfall == 0:
                    return True
                if not matches(d):
                    player.dice[i] = bid.face
                    shortfall -= 1
        return True
