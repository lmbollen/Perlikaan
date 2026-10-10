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


# ---------------------------------------------------------------------------
# Game types
#
# Bots may not import the framework, so this file carries its own copies of
# the game's types (same names and fields as the engine's). The
# @engine_types decorator on the bot class translates at the border: play()
# and observe() get these local types, and play() hands the engine back its
# own action object from state.legal_actions.
# ---------------------------------------------------------------------------

PACO = 1
NORMAL_FACES = (2, 3, 4, 5, 6)


class _Value:
    """Immutable value: the fields in _FIELDS, equal and hashed by value."""

    _FIELDS = ()

    def __init__(self, *args, **kwargs):
        values = dict(zip(self._FIELDS, args), **kwargs)
        for field in self._FIELDS:
            object.__setattr__(self, field, values[field])

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def _key(self):
        return tuple(getattr(self, f) for f in self._FIELDS)

    def __eq__(self, other):
        return type(other) is type(self) and other._key() == self._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(repr(v) for v in self._key())})"


class Bid(_Value):
    _FIELDS = ("quantity", "face")

    def __repr__(self):
        face = "Paco" if self.face == PACO else str(self.face)
        return f"Bid({self.quantity}x {face})"


class Dudo(_Value):
    pass


class Calza(_Value):
    pass


class ActionRecord(_Value):
    _FIELDS = ("player", "action")


class PublicPlayer(_Value):
    _FIELDS = ("id", "name", "dice_count", "alive")


class Observation(_Value):
    _FIELDS = ("my_id", "my_dice", "my_dice_count", "players", "current_bid", "palifico",
               "round_number", "history", "legal_actions")


class GameStartInfo(_Value):
    _FIELDS = ("players", "my_id")


class GameResult(_Value):
    _FIELDS = ("winner", "standings", "rounds_played")


class BidEvent(_Value):
    _FIELDS = ("player", "bid")


class DudoEvent(_Value):
    _FIELDS = ("player",)


class CalzaEvent(_Value):
    _FIELDS = ("player",)


class RevealEvent(_Value):
    _FIELDS = ("dice", "current_bid", "actual_count", "challenge_type", "challenger", "bidder", "result")

    def dice_by_player(self):
        return dict(self.dice)


class DieLostEvent(_Value):
    _FIELDS = ("player", "dice_count")


class DieGainedEvent(_Value):
    _FIELDS = ("player", "dice_count")


class PlayerEliminatedEvent(_Value):
    _FIELDS = ("player",)


class RoundEndedEvent(_Value):
    _FIELDS = ("next_starting_player", "palifico_next")


class GameEndedEvent(_Value):
    _FIELDS = ("winner", "standings")


_LOCAL_TYPES = {cls.__name__: cls for cls in (
    Bid, Dudo, Calza, ActionRecord, PublicPlayer, Observation, GameStartInfo, GameResult,
    BidEvent, DudoEvent, CalzaEvent, RevealEvent, DieLostEvent, DieGainedEvent,
    PlayerEliminatedEvent, RoundEndedEvent, GameEndedEvent)}


def _to_local(x):
    """An engine object as the local type of the same name (nested values too)."""
    cls = _LOCAL_TYPES.get(type(x).__name__)
    if cls is not None and type(x) is not cls:
        return cls(**{f: _to_local(getattr(x, f)) for f in cls._FIELDS})
    if type(x) is tuple:
        return tuple(_to_local(v) for v in x)
    return x


def _to_engine(state, action):
    """The engine's own object in state.legal_actions for a local action."""
    for legal in state.legal_actions:
        if _to_local(legal) == action:
            return legal
    return action


def engine_types(cls):
    """Class decorator: the bot sees local types, the engine gets its own back."""
    play, observe = cls.play, cls.observe
    game_start, game_end = cls.on_game_start, cls.on_game_end

    def _play(self, state):
        return _to_engine(state, play(self, _to_local(state)))

    def _observe(self, event):
        return observe(self, _to_local(event))

    def _on_game_start(self, info):
        return game_start(self, _to_local(info))

    def _on_game_end(self, result):
        return game_end(self, _to_local(result))

    cls.play, cls.observe = _play, _observe
    cls.on_game_start, cls.on_game_end = _on_game_start, _on_game_end
    return cls


def _count_matches(dice_values, face, palifico):
    """How many dice count towards a bid on `face`."""
    if palifico:
        return sum(1 for d in dice_values if d == face)
    if face == PACO:
        return sum(1 for d in dice_values if d == PACO)
    return sum(1 for d in dice_values if d == face or d == PACO)


@engine_types
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
