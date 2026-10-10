"""
Sonny Boy: een kansrekenende Perudo-bot.

Strategie in het kort:
  * Voor elk bod schatten we met de binomiaalverdeling hoe groot de kans is dat
    het klopt, gegeven onze eigen dobbelstenen + de onbekende dobbelstenen van
    de rest (1/3 kans per die voor gewone getallen, 1/6 voor Paco/Palifico).
  * Biedt een tegenstander zichtbaar hoog of is zijn bod onwaarschijnlijk -> Dudo.
  * Zit het bod er precies op (hoogste kans op "exact") -> Calza.
  * Anders: het bod met de grootste kans op waarheid, met af en toe een bluf.
  * We onthouden per tegenstander (op naam) hoe vaak ze bij een Dudo bluften.
"""

import random
from math import comb


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


PACO = 1


def _p_at_least(n: int, p: float, k: int) -> float:
    """P(X >= k) voor X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _p_exact(n: int, p: float, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return comb(n, k) * p**k * (1 - p) ** (n - k)


@engine_types
class SonnyBoy:
    CREDIT = 1
    CALZA_MIN = 0.5
    DUDO_AT = 0.45

    def __init__(self):
        self.name = "Sonny Boy"
        self.count = 1
        # Per tegenstander: [aantal bods dat bij Dudo-reveal klopte, totaal bods onthuld]
        self.bluff_stats = {}
        self._rng = random.Random()
        self._round_bids = []  # (player, Bid) deze ronde -- via observe()

    def on_game_start(self, info):
        pass

    # ------------------------------------------------------------------
    # kansberekening
    # ------------------------------------------------------------------
    def _matches_mine(self, state, face, palifico):
        if palifico or face == PACO:
            return sum(1 for d in state.my_dice if d == face)
        return sum(1 for d in state.my_dice if d == face or d == PACO)

    def _unknown_and_p(self, state, face, palifico):
        total = sum(p.dice_count for p in state.players if p.alive)
        unknown = total - state.my_dice_count
        p = 1 / 6 if (palifico or face == PACO) else 1 / 3
        return unknown, p

    def _opponent_prior_shift(self, state, bid):
        """
        Extra verwacht aantal dobbelstenen van de laatste bieder op dit getal:
        wie net dit getal bood, heeft er waarschijnlijk minstens één.
        """
        return 0.0

    def _p_true(self, state, bid, palifico, credit=0):
        """credit: aantal dobbelstenen dat we de laatste bieder op dit getal toekennen."""
        mine = self._matches_mine(state, bid.face, palifico) + credit
        unknown, p = self._unknown_and_p(state, bid.face, palifico)
        return _p_at_least(max(unknown - credit, 0), p, bid.quantity - mine)

    def _p_exact(self, state, bid, palifico, credit=0):
        mine = self._matches_mine(state, bid.face, palifico) + credit
        unknown, p = self._unknown_and_p(state, bid.face, palifico)
        return _p_exact(max(unknown - credit, 0), p, bid.quantity - mine)

    def _last_bidder(self, state):
        for rec in reversed(state.history):
            if isinstance(rec.action, Bid):
                return rec.player
        return None

    def _bluff_rate(self, name):
        stats = self.bluff_stats.get(name)
        if not stats or stats[1] < 5:
            return 0.5
        return 1 - stats[0] / stats[1]

    # ------------------------------------------------------------------
    # beslissing
    # ------------------------------------------------------------------
    def play(self, state):
        legal = state.legal_actions
        bids = [a for a in legal if isinstance(a, Bid)]
        can_dudo = any(isinstance(a, Dudo) for a in legal)
        can_calza = any(isinstance(a, Calza) for a in legal)
        palifico = state.palifico
        cur = state.current_bid
        total = sum(p.dice_count for p in state.players if p.alive)

        # Beoordeel het huidige bod.
        if cur is not None and can_dudo:
            p_true = self._p_true(state, cur, palifico, credit=self.CREDIT)
            # Tegenstanders die vaak bluffen: iets sneller Dudo.
            bidder = self._last_bidder(state)
            if bidder:
                p_true -= 0.08 * (self._bluff_rate(bidder) - 0.5)

            # Beste eigen vervolgbod.
            best_bid, best_p = self._best_bid(state, bids, palifico)

            # Calza: alleen als precies-kans echt hoog is.
            if can_calza:
                pe = self._p_exact(state, cur, palifico, credit=self.CREDIT)
                threshold = self.CALZA_MIN
                if pe >= threshold:
                    return Calza()

            # Dudo als het bod waarschijnlijker fout dan goed is, of als
            # doorbieden nauwelijks veiliger is.
            if best_bid is None:
                return Dudo()
            dudo_threshold = self.DUDO_AT
            # Met weinig dobbelstenen is verliezen duurder: iets voorzichtiger bluffen.
            if p_true < dudo_threshold or p_true < best_p - 0.25:
                return Dudo()
            return best_bid

        # Openen.
        if not bids:
            return Calza() if can_calza else Dudo()
        best_bid, best_p = self._best_bid(state, bids, palifico, opening=True)
        return best_bid

    def _best_bid(self, state, bids, palifico, opening=False):
        if not bids:
            return None, 0.0
        scored = []
        for b in bids:
            p = self._p_true(state, b, palifico)
            scored.append((p, b))
        if opening:
            # Open op ons sterkste getal met een laag, redelijk veilig bod;
            # kies hoogste kans, bij gelijke kans een groter aantal niet nodig.
            # Kleine ruis zodat we niet voorspelbaar zijn.
            best_p = max(p for p, _ in scored)
            cands = [(p, b) for p, b in scored if p >= best_p - 0.05]
            # Voorkeur voor getallen waar we zelf veel van hebben (eigen dobbelstenen).
            cands.sort(key=lambda x: (-self._matches_mine(state, x[1].face, palifico), x[1].quantity))
            p, b = cands[0]
            if len(cands) > 1 and self._rng.random() < 0.3:
                p, b = self._rng.choice(cands)
            return b, p
        # Hoogste kans; bij (bijna) gelijke kans het laagste bod (zo houden we ruimte).
        best_p = max(p for p, _ in scored)
        cands = [(p, b) for p, b in scored if p >= best_p - 0.02]
        cands.sort(key=lambda x: (x[1].quantity, x[1].face))
        p, b = cands[0]
        return b, p

    # ------------------------------------------------------------------
    # leren / reageren
    # ------------------------------------------------------------------
    def observe(self, event):
        if isinstance(event, RevealEvent):
            # Zegt iets over de bluf-frequentie van de bieder.
            correct = event.actual_count >= event.current_bid.quantity
            st = self.bluff_stats.setdefault(event.bidder, [0, 0])
            st[1] += 1
            if correct:
                st[0] += 1

    def on_game_end(self, result):
        pass
