"""
100 IQ Bot: a simple probability-based bot.

Splits all dice on the table into two buckets: "mine" (known exactly)
and "the rest of the table" (unknown, each die uniformly 1-6). It uses
that to estimate the probability that the current bid is actually true,
then either raises (using the face it personally holds the most of) or
calls Dudo.

Uses _count_matches, a copy of the same pure counting rule the engine
itself uses to decide bids/Dudo/Calza (wild Paco's, Palifico).
"""

import math


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
class HundredIQBot:
    def __init__(self):
            self.name = "100 IQ Bot"
            self.count = 3

    def on_game_start(self, info):
        pass

    def play(self, state):
        return self._choose(state)

    def observe(self, event):
        pass

    def on_game_end(self, result):
        pass

    def _choose(self, state):
        possible_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        best_face = self._face_i_hold_most(state.my_dice)

        if state.current_bid is None:
            # Geen vorig bod om op te rekenen -- gewoon openen op het
            # gezicht waar ik zelf de meeste van heb.
            chosen = self._cheapest_bid_for_face(possible_bids, best_face)
            return chosen if chosen is not None else state.legal_actions[0]

        if self._probability_bid_is_true(state) > 0.5:
            chosen = self._cheapest_bid_for_face(possible_bids, best_face)
            if chosen is not None:
                return chosen
            # Geen enkel bod meer legaal (uitgebolde Palifico-ronde) --
            # dan blijft alleen Dudo over.

        return Dudo()

    # -- helpers ----------------------------------------------------

    @staticmethod
    def _face_i_hold_most(my_dice):
        counts = {face: my_dice.count(face) for face in range(1, 7)}
        return max(counts, key=lambda face: counts[face])

    @staticmethod
    def _cheapest_bid_for_face(possible_bids, face):
        # "Het eerst volgende getal" op het gekozen gezicht: de goedkoopste
        # (laagste aantal) legale bid die de engine voor dat gezicht toestaat
        # -- dit dekt normale verhogingen EN de halveer/verdubbel-regels
        # van/naar Paco automatisch, zonder dat de bot die zelf hoeft te
        # kennen.
        same_face = [b for b in possible_bids if b.face == face]
        if same_face:
            return min(same_face, key=lambda b: b.quantity)
        if possible_bids:
            return min(possible_bids, key=lambda b: (b.quantity, b.face))
        return None  # only possible in a maxed-out Palifico round

    @staticmethod
    def _probability_bid_is_true(state):
        bid = state.current_bid
        total_dice_in_play = sum(p.dice_count for p in state.players)
        opponent_dice = total_dice_in_play - state.my_dice_count

        my_matches = _count_matches(state.my_dice, bid.face, state.palifico)
        still_needed = max(0, bid.quantity - my_matches)

        if state.palifico or bid.face == PACO:
            p_match = 1 / 6  # no wild: only the exact face counts
        else:
            p_match = 2 / 6  # face itself, or a wild Paco

        return HundredIQBot._probability_at_least(opponent_dice, still_needed, p_match)

    @staticmethod
    def _probability_at_least(dice_count, needed, p):
        if needed <= 0:
            return 1.0
        if needed > dice_count:
            return 0.0
        return sum(
            math.comb(dice_count, k) * (p ** k) * ((1 - p) ** (dice_count - k))
            for k in range(needed, dice_count + 1)
        )
