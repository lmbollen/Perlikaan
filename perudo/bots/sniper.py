"""
Sniper: places the highest bid that is still probably true.

The idea: a bid that sits just below the edge of what the table can
plausibly hold leaves the next player in a bad spot. Raising pushes them
into unlikely territory, and calling Dudo means challenging a bid that is
probably true. Either way the die should not come back to us.

How a bid is scored:
  * Our own dice are counted exactly (with wild Pacos outside Palifico).
  * Every unknown die matches with probability 1/3 for a normal face
    (face or Paco) or 1/6 for Paco bids and Palifico rounds.
  * Earlier bids are a hint: an opponent whose latest bid was on a face
    probably holds a bit more of it than chance predicts. That shifts the
    count we expect from the unknown dice.

Each turn the bot:
  1. Computes P(true) for the current bid and for every legal raise.
  2. Takes the strongest raise whose P(true) is at least TARGET.
  3. Calls Dudo instead when the current bid is less likely to be true
     than that raise is to be false, or when no raise is safe enough.
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
class SniperBot:
    name = "Sniper"

    # Tuned by simulation against the other bots in this folder: targets
    # between 0.75 and 0.9 do about equally well, 0.55 and lower do badly.
    #
    # Minimum P(true) for a bid we place. Lower is more aggressive.
    TARGET = 0.85
    # Below this no raise is acceptable and we call Dudo instead.
    MIN_ACCEPT = 0.3
    # Extra dice of the bid face we credit an opponent for, per bid they made
    # on that face, capped by how many dice they actually hold.
    HINT_PER_BID = 0.7
    # Weight of a Paco bid as a hint for normal faces (Pacos are wild).
    PACO_HINT_WEIGHT = 0.4

    def __init__(self):
        self.name = "Sniper"
        self._tail_cache = {}

    def on_game_start(self, info):
        pass

    # -- probability ------------------------------------------------------

    @staticmethod
    def _p_face(face, palifico):
        return 1 / 6 if (palifico or face == PACO) else 1 / 3

    def _tail(self, n, p, need):
        """P(Binomial(n, p) >= need), interpolated for fractional `need`."""
        key = (n, p, need)
        if key in self._tail_cache:
            return self._tail_cache[key]

        def tail_int(k):
            if k <= 0:
                return 1.0
            if k > n:
                return 0.0
            return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

        lo = math.floor(need)
        frac = need - lo
        result = (1 - frac) * tail_int(lo) + frac * tail_int(lo + 1)
        self._tail_cache[key] = result
        return result

    def _hint(self, state, face):
        """Extra dice of `face` we expect among opponents, based on their bids."""
        # In Palifico only the opener chose the face; later bids are forced.
        history = state.history[:1] if state.palifico else state.history
        dice_of = {p.name: p.dice_count for p in state.players}

        bids_by_player = {}
        for rec in history:
            if rec.player == self.name or not isinstance(rec.action, Bid):
                continue
            bids_by_player.setdefault(rec.player, []).append(rec.action)

        total = 0.0
        for who, bids in bids_by_player.items():
            weight = 0.0
            for bid in bids:
                if bid.face == face:
                    weight += 1.0
                elif not state.palifico and bid.face == PACO and face != PACO:
                    weight += self.PACO_HINT_WEIGHT
            if weight == 0.0:
                continue
            # Cap: they can't hold more than their dice beyond expectation.
            n = dice_of.get(who, 0)
            cap = n * (1 - self._p_face(face, state.palifico))
            total += min(weight * self.HINT_PER_BID, cap)
        return total

    def _prob_true(self, state, bid, n_unknown, hints):
        mine = _count_matches(state.my_dice, bid.face, state.palifico)
        p = self._p_face(bid.face, state.palifico)
        need = bid.quantity - mine - hints[bid.face]
        return self._tail(n_unknown, p, need)

    # -- bid ordering -----------------------------------------------------

    @staticmethod
    def _strength(bid, palifico):
        """How strong a claim a bid is, so Paco and normal bids compare."""
        if bid.face == PACO and not palifico:
            # A Paco is worth two normal dice; rank it just below 2q of a face.
            return (2 * bid.quantity - 0.5, 0)
        return (bid.quantity, bid.face)

    # -- decision ---------------------------------------------------------

    def play(self, state):
        total = sum(p.dice_count for p in state.players if p.alive)
        n_unknown = total - state.my_dice_count
        hints = {face: self._hint(state, face) for face in range(1, 7)}

        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        scored = [(self._prob_true(state, b, n_unknown, hints), b) for b in bids]

        safe = [(pt, b) for pt, b in scored if pt >= self.TARGET]
        if safe:
            # The strongest claim that still clears TARGET. Ties go to the
            # likelier bid.
            best_p, best = max(safe, key=lambda s: (self._strength(s[1], state.palifico), s[0]))
        elif scored:
            best_p, best = max(scored, key=lambda s: s[0])
        else:
            best_p, best = 0.0, None

        cur = state.current_bid
        if cur is None:
            # Opening: there is always a legal bid.
            return best

        p_cur = self._prob_true(state, cur, n_unknown, hints)
        if best is None or best_p < self.MIN_ACCEPT or p_cur < 1 - best_p:
            return Dudo()
        return best

    def observe(self, event):
        pass

    def on_game_end(self, result):
        pass
