"""
Carlos Dos: Carlos rebuilt around a base model of the dice.

Every face on the table is a probability distribution: how many matching
dice each player holds. Our own dice are known, every opponent starts as a
binomial over their dice, and every bid an opponent made this round
replaces that by what such a bid says about the hand (the source model,
learned at the reveals per player and per kind of bid: a chosen face, a
followed face, Paco, Palifico):

  * when the quantity says something about the hand (the bidder needs at
    least one matching die themselves, given chance elsewhere), the
    histogram of implied - held: its mass at zero and below are the real
    bids, its mass above zero the bluffs;
  * when it does not (a low bid), the histogram of how many of the face
    the player held when bidding on it.

The sum of those distributions is the count on the table, which gives
P(bid true) and P(bid exact) for every bid, ours included.

On top of the base model sit the neighbours:

  * the player before us is read with the source model: real bid or
    bluff, and what does that say about the count? That decides our Dudo
    and Calza, valued as expected dice.
  * the player after us has a reaction model: a per-player logistic for
    Dudo and one for Calza, on how likely the bid looks from their own
    dice (we integrate over what they may hold), their share of the dice,
    and their trust in us (our own record: how many of our revealed bids
    were true). What is left is a raise.

The players beyond the neighbours are collapsed to keep it fast: after the
next player raises, every further player may end the round (free for us:
an opponent loses a die) or raise again; what reaches us again is a higher
bid, a threat, valued by our best answer to it.

Candidate bids are the faces we hold (a few quantities from the minimum
legal one, or around the expected count at the opening) plus one bluff
face: at the opening a random face we do not hold, bid as if we held it;
later in the round a face from the earlier bids, drawn in proportion to
the dice of the player who bid it. Every candidate, Dudo and Calza get an
expected value (our die -1, an opponent's +1 / opponents, a correct Calza
+1 while it gains a die) and the best one is played.

Every opponent's model is tied to their name: a new name starts from the
probability priors, every reveal updates the models of everyone in it
(the source model of every bidder, the reaction model of every player who
acted on a bid), and the players beyond the next one are valued with their
own models too. Everything learned lives in class-level memory per bot
name, survives a new instance per game, decays towards the priors by
GAME_DECAY after every game, and never touches the disk.
"""


from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from perudo.models import (
    PACO,
    Bid,
    BidEvent,
    Calza,
    Dudo,
    GameEndedEvent,
    RevealEvent,
    RoundEndedEvent,
)

_NORMAL_FACES = (2, 3, 4, 5, 6)


def _is(event, *names):
    return type(event).__name__ in names


def _count_matches(dice: Sequence[int], face: int, palifico: bool) -> int:
    if palifico:
        return sum(1 for d in dice if d == face)
    if face == PACO:
        return sum(1 for d in dice if d == PACO)
    return sum(1 for d in dice if d == face or d == PACO)


def _face_p(face: int, palifico: bool) -> float:
    return 1 / 6 if (palifico or face == PACO) else 1 / 3


_BINOM: Dict[Tuple[int, float], List[float]] = {}


def _binomial(n: int, p: float) -> List[float]:
    key = (n, p)
    pmf = _BINOM.get(key)
    if pmf is None:
        q = 1 - p
        pmf = [math.comb(n, k) * p ** k * q ** (n - k) for k in range(n + 1)]
        _BINOM[key] = pmf
    return pmf


_MOST: Dict[Tuple[int, bool], List[float]] = {}


def _most_held(n: int, pal: bool) -> List[float]:
    """Distribution of the count of the face a player holds most of (Pacos
    wild unless Palifico), over all 6^n hands."""
    key = (n, pal)
    out = _MOST.get(key)
    if out is None:
        out = [0.0] * (n + 1)
        for code in range(6 ** n):
            hand, c = [], code
            for _ in range(n):
                hand.append(c % 6 + 1)
                c //= 6
            best = max(_count_matches(hand, f, pal) for f in (range(1, 7) if pal else _NORMAL_FACES))
            out[best] += 1
        out = [v / 6 ** n for v in out]
        _MOST[key] = out
    return out


def _conv(a: Sequence[float], b: Sequence[float]) -> List[float]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def _tail(pmf: Sequence[float], q: int) -> float:
    """P(count >= q)."""
    if q <= 0:
        return 1.0
    if q >= len(pmf):
        return 0.0
    return sum(pmf[q:])


def _at(pmf: Sequence[float], q: int) -> float:
    return pmf[q] if 0 <= q < len(pmf) else 0.0


def _sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1 / (1 + math.exp(-x))


def _min_quantity(bid: Bid, face: int, palifico: bool) -> int:
    """Lowest legal quantity to raise `bid` on `face`."""
    if palifico:
        return bid.quantity + 1
    if bid.face == PACO:
        return bid.quantity + 1 if face == PACO else 2 * bid.quantity + 1
    if face == PACO:
        return (bid.quantity + 1) // 2
    return bid.quantity if face > bid.face else bid.quantity + 1


KINDS = ("choice", "follow", "paco-choice", "paco-follow", "pal-choice", "pal-follow")
R_MAX = 4


def _kind(bid: Bid, prev: Optional[Bid], pal: bool) -> str:
    """Kind of a bid: chosen face (opening or switch) or following the face just
    bid; Paco and Palifico bids are their own kinds."""
    choice = prev is None or prev.face != bid.face
    if pal:
        return "pal-choice" if choice else "pal-follow"
    if bid.face == PACO:
        return "paco-choice" if choice else "paco-follow"
    return "choice" if choice else "follow"


class _Source:
    """How a player's bids relate to their hand (the source model), learned
    from the reveals, per kind of bid.

    When the quantity says something about the hand, a bid's likelihood
    depends on the residual r = implied - held: lambda(r) is how much more
    often a bid comes with residual r than chance alone gives it (observed
    residuals over the chance mass of each residual). Real bids are r <= 0,
    bluffs r > 0. When the quantity says nothing, the held count on the face
    is kept as a histogram per dice count."""

    def __init__(self):
        size = 2 * R_MAX + 1
        self.obs = {k: [0.0] * size for k in KINDS}      # kind -> residual -> bids seen
        self.chance = {k: [0.0] * size for k in KINDS}   # kind -> residual -> chance mass
        self.chosen = {}                              # (kind, dice) -> [weight per held]
        self.honesty = {k: [0.0, 0.0] for k in KINDS}  # kind -> [real, bids]

    def decay(self, f):
        for table in (self.obs, self.chance):
            for w in table.values():
                for i in range(len(w)):
                    w[i] *= f
        for w in self.chosen.values():
            for i in range(len(w)):
                w[i] *= f
        for h in self.honesty.values():
            h[0] *= f
            h[1] *= f


class _Player:
    """What we learned about one opponent, by name."""

    def __init__(self, dudo_w, calza_w):
        self.dudo_w = list(dudo_w)        # logit = w . _dudo_x(...)
        self.calza_w = list(calza_w)      # logit = w0 + w1 * p_exact
        self.source = _Source()
        self.trust = None                 # EMA of the share of our bids that were true in the rounds this player saw revealed
        self.n = 0                        # reactions seen


class CarlosDosBot:
    name = "Carlos Dos"
    count = 1

    # -- values ---------------------------------------------------------------
    # An opponent's lost die, divided by the number of opponents.
    OTHER_WEIGHT = 1.0
    # A correct Calza of ours while it gains a die, and once it does not.
    CALZA_GAIN = 1.0
    CALZA_FREE_PASS = 0.1
    # A bid that is not on the table costs this per unit of P(false), on top
    # of what the model expects from challenges.
    FALSE_PENALTY = 0.2
    # When a bid comes back to us we weigh Dudo, Calza and our next raise,
    # the raise against the next player's reaction model. Beyond that one
    # level, raising is valued as -(1 - P(raise true)) * this.
    RETURN_RISK = 0.5
    # The continuation updates every player's hand for the raise it assumes
    # they make (a raise on a face is evidence of holding it), so the bid
    # that comes back to us is valued with that evidence. False: the hands
    # stay as they were before the raises, which made Dudo on the returned
    # bid look too good (heads-up: +0.83 predicted, +0.24 realised).
    # Paired, 640 games at tables of 2-7: +4.5 +/- 2.2 pp win rate.
    FIX_CONT_HANDS = True
    # A later player's correct Calza gains THEM a die; False counted every
    # round-ending action by a later player as an opponent losing a die.
    # Paired, 640 games: +0.3 +/- 0.7 pp (within noise).
    FIX_LATER_CALZA = True
    # Weight of the continuation (the rest of the round after the next
    # player raises) in a bid's value. It overrates the rest of the round,
    # most for weak bids. With three or more players that pressure does not
    # pay: the next player simply calls. Paired, 3 players (1280 games): 0.5
    # +4.1, 0.3 +4.9 pp; 4-7 players (960): 0.5 +2.5 pp, score +0.031.
    # With the hands updated (FIX_CONT_HANDS) the weight was re-tested: 0.75
    # -2.0 +/- 2.0, 1.0 -4.5 +/- 2.1 pp, so 0.5 stays.
    CONT_SCALE = 0.5
    # The same per number of players alive, where it differs. Heads-up the
    # pressure pays (640 paired games: 0.6 -9.2, 0.3 -13.9, 0 -25.6 pp).
    CONT_SCALE_BY_ALIVE = {2: 1.0}
    # Moves within this of the best count as equally good (picked at random).
    TIE_EPS = 0.01

    # -- candidates -----------------------------------------------------------
    # Quantities per held face: the minimum legal one and this many above.
    SPAN = 3
    # Opening: quantities around the expected count, this far down and up.
    OPEN_BELOW, OPEN_ABOVE = 1, 1
    # The opening bluff bids as if we held this many of the face.
    BLUFF_HELD = 1
    # Quantities for the bluff face: the minimum and this many above.
    BLUFF_SPAN = 2

    # -- the generic starting model --------------------------------------------
    # Every opponent starts from one generic model of "a player", and the
    # per-player learning at every reveal does the rest. The generic model
    # is fitted by fit_generic_prior.py against a generic population of its
    # own (rational players with random calling thresholds, bid reading,
    # bluff and jump rates), never against the bots of this repository.
    #
    # Dudo, per player: logit = w . x with x = (1, P(true) from their own
    # dice, their dice share, P(true) from their dice AND what the bids this
    # round say (how a reader of the bidding sees it), how far the bid jumps
    # above the minimum raise, the bid is ours (tables of three or more),
    # our caught-bluff record when it is ours, the value of an opponent's
    # die 1 / (players - 1)).
    DUDO_PRIOR = (-0.31, -6.54, 1.91, -1.1, 0.28, 0.55, 0.0, 2.89)
    # Calza: w0 + w1 * P(exact).
    CALZA_PRIOR = (-5.72, 7.98)
    TRUST_START = 0.7
    TRUST_LR = 0.15
    # Source model before we know a player. When the quantity says
    # something, a bid comes REAL_LR times as often from a hand that covers
    # what it implies as from one that does not (Bayes then weighs that
    # against the chance of such a hand: a bid that needs five of five dice
    # is mostly a bluff). Weight of that prior, and of the table's model
    # behind the player's own, in chance mass of a residual (about a bid).
    REAL_LR = 2.0
    # The same per number of players alive, where it differs (only used for
    # a kind missing from GENERIC_LAMBDA below).
    REAL_LR_BY_ALIVE = {}
    # Multiplier of that ratio for Palifico bids (face locked, nothing wild).
    # Heads-up Palifico bids are true 92-98% where the model says ~80%, but
    # 3x or 8x changed nothing in 960 paired games: off.
    PAL_LR_FACTOR = 1.0
    # Multiplier of P(exact) when we weigh a Calza of our own. Heads-up at
    # 2-4 dice his Calzas are exact 44% where the model says 70%, yet they
    # are mostly escapes from a bid that is true 64% of the time: 0.8 -0.7,
    # 0.65 -1.2 pp. Off.
    EXACT_SHRINK = 1.0
    LR_PRIOR = 1.0
    # The generic starting source model, fitted the same way: lambda per
    # residual (implied - held, -R_MAX..R_MAX) per kind of bid, heads-up
    # ("2") and with more players ("3+"). It replaces the REAL_LR formula
    # above (still used for a kind missing here, or with None).
    GENERIC_LAMBDA = {
        "2": {
            "choice": (0.613, 0.466, 0.637, 1.723, 0.86, 0.558, 0.565, 0.737, 0.861),
            "follow": (1.025, 2.149, 2.872, 2.804, 1.605, 0.961, 0.343, 0.058, 0.082),
            "paco-choice": (0.99, 0.842, 0.664, 1.799, 0.955, 0.744, 0.916, 0.929, 0.992),
            "paco-follow": (1.0, 0.997, 1.136, 2.07, 2.597, 1.593, 0.45, 0.215, 0.445),
            "pal-choice": (1.0, 1.0, 1.0, 1.0, 1.015, 0.958, 1.0, 1.0, 1.0),
        },
        "3+": {
            "choice": (0.756, 0.825, 0.818, 1.344, 0.977, 0.897, 0.758, 0.789, 0.967),
            "follow": (2.244, 3.644, 3.136, 2.347, 1.505, 0.887, 0.603, 0.419, 0.243),
            "paco-choice": (0.97, 2.071, 1.898, 1.511, 0.824, 0.926, 1.246, 1.314, 1.6),
            "paco-follow": (0.999, 1.449, 3.548, 3.091, 2.323, 1.266, 0.863, 0.802, 0.751),
            "pal-choice": (1.0, 1.0, 1.0, 1.0, 0.988, 1.034, 1.0, 1.0, 1.0),
            "pal-follow": (0.992, 0.923, 0.958, 1.166, 1.044, 1.047, 0.997, 0.888, 0.816),
        },
    }
    # A real chosen face (opening or switching) is the face the player holds
    # most of (a Paco they hold, for Paco); REAL_PRIOR of such bids are real.
    REAL_PRIOR = 0.7
    # Use that also under the quantity's likelihood (False: chance there).
    CHOSEN_BASE = True
    # A face a player picked themselves earlier in the round stays their
    # chosen face when they raise it again later (False: a raise on the face
    # just bid is read as following it). Heads-up +0.5 +/- 0.9 pp: off.
    CHOSEN_STAYS = False
    # Derive the faces a player did not bid on from their bid on another face
    # (Pacos shared). Off: it overrates the Pacos of a bidder (a player picks
    # the face with most naturals; Pacos are the same for every face), which
    # made the next player look safer than he was (paired, 480 games: +5.0
    # +/- 2.3 pp win rate with it off).
    USE_PASSED_OVER = False
    SOURCE_PRIOR = 4.0
    SOURCE_SHRINK = 4.0
    # Every opponent's model belongs to their name alone: a name we have not
    # seen starts from the probability priors above (not from what other
    # bots did), every reveal updates the models of the names in it, and
    # between games a model decays towards those priors. False: blend in the
    # model pooled over all opponents (the old behaviour, for comparison).
    NAME_ONLY = True
    # Honesty: pseudo-bids behind REAL_PRIOR.
    HONESTY_N = 4.0
    # Online learning rate of the logistic models, and the pull of a player's
    # model towards the table's per game (0.95: a game 14 games ago counts half).
    LEARN_RATE = 0.15
    GAME_DECAY = 0.95
    # Bounds of the learned weights.
    W_MIN, W_MAX = -30.0, 8.0
    # Raise step (quantity per raise) learned over the table; start.
    STEP_START = 1.0
    STEP_LR = 0.05

    _LEARNED = ("players", "table")

    # -- shared memory: learned state per bot name, kept at class level --------
    SHARED_MEMORY = True

    @classmethod
    def reset_memory(cls) -> None:
        cls._MEMORY = {}

    def _ensure_memory(self) -> None:
        if getattr(self, "_memory_bound", False):
            return
        self._memory_bound = True
        if not self.SHARED_MEMORY:
            return
        cls = type(self)
        if "_MEMORY" not in cls.__dict__:
            cls._MEMORY = {}
        store = cls._MEMORY.setdefault(self.name, {})
        for attr in self._LEARNED:
            if attr in store:
                setattr(self, attr, store[attr])
            else:
                store[attr] = getattr(self, attr)

    def __init__(self):
        self.name = "Carlos Dos"
        self._rng = random.Random()
        self.players: Dict[str, _Player] = {}
        self.table = {"dudo_w": list(self.DUDO_PRIOR), "calza_w": list(self.CALZA_PRIOR),
                      "source": _Source(), "step": self.STEP_START, "n": 0}
        self._round: List[Tuple[str, object]] = []
        self._palifico = False

    # ------------------------------------------------------------------
    # observing and learning
    # ------------------------------------------------------------------

    def on_game_start(self, info):
        self._ensure_memory()
        self._round = []
        self._palifico = False

    def on_game_end(self, result):
        pass

    def observe(self, event):
        self._ensure_memory()
        if _is(event, "BidEvent"):
            self._round.append((event.player, event.bid))
        elif _is(event, "DudoEvent"):
            self._round.append((event.player, Dudo()))
        elif _is(event, "CalzaEvent"):
            self._round.append((event.player, Calza()))
        elif _is(event, "RevealEvent"):
            self._learn(event)
            self._round = []
        elif _is(event, "RoundEndedEvent"):
            self._palifico = event.palifico_next
        elif _is(event, "GameEndedEvent"):
            self._palifico = False
            self._round = []
            self._decay_memory()

    def _player(self, name) -> _Player:
        p = self.players.get(name)
        if p is None:
            if self.NAME_ONLY:
                p = _Player(self.DUDO_PRIOR, self.CALZA_PRIOR)
            else:
                p = _Player(self.table["dudo_w"], self.table["calza_w"])
            self.players[name] = p
        return p

    def _learn(self, event):
        """Replay the round with everyone's dice known."""
        dice = dict(event.dice)
        names = [n for n, _ in event.dice]
        self._n_alive = len(names)
        counts = {n: len(d) for n, d in event.dice}
        total = sum(counts.values())
        pal = self._palifico
        if self._round and not isinstance(self._round[-1][1], Bid):
            actions = self._round
        else:                     # the challenge came through the reveal only
            actions = self._round + [(event.challenger, Dudo() if event.challenge_type == "dudo" else Calza())]
        our_truths = []
        current, bidder, before = None, None, None
        bids_so_far = []
        for who, action in actions:
            if current is not None and who != self.name and who in dice:
                self._learn_reaction(who, action, current, bidder, before, bids_so_far, dice, counts, total, pal)
            if isinstance(action, Bid):
                before = current
                bids_so_far.append((who, action))
                if who in dice:
                    truth = _count_matches([d for h in dice.values() for d in h], action.face, pal) >= action.quantity
                    if who == self.name:
                        our_truths.append(truth)
                    else:
                        self._learn_source(who, action, _kind(action, current, pal),
                                           dice[who], counts[who], total, pal)
                if current is not None:
                    step = action.quantity - current.quantity
                    if 0 <= step <= 3:
                        self.table["step"] += self.STEP_LR * (step - self.table["step"])
                current, bidder = action, who
            else:
                break
        if our_truths:
            truth = sum(our_truths) / len(our_truths)
            for n in names:
                if n != self.name:
                    p = self._player(n)
                    t = self.TRUST_START if p.trust is None else p.trust
                    p.trust = t + self.TRUST_LR * (truth - t)

    def _learn_source(self, who, bid, kind, hand, n, total, pal):
        implied = self._implied(bid, n, total, pal)
        k = _count_matches(hand, bid.face, pal)
        base = self._base_pmf(who, kind, n, pal)        # before this bid is learned
        choice = kind.endswith("choice")
        if choice:
            if bid.face == PACO and not pal:
                real = k >= 1
            else:
                faces = range(1, 7) if pal else _NORMAL_FACES
                real = all(k >= _count_matches(hand, f, pal) for f in faces)
        for src in (self._player(who).source, self.table["source"]):
            if 1 <= implied <= n:
                src.obs[kind][max(-R_MAX, min(R_MAX, implied - k)) + R_MAX] += 1
                chance = src.chance[kind]
                for kk, b in enumerate(base):
                    chance[max(-R_MAX, min(R_MAX, implied - kk)) + R_MAX] += b
            if choice:
                w = src.chosen.setdefault((kind, n), [0.0] * (n + 1))
                w[k] += 1
                h = src.honesty[kind]
                h[0] += real
                h[1] += 1

    @staticmethod
    def _implied(bid, n, total, pal):
        """Matches the bidder needs themselves for the bid, given chance elsewhere
        (negative when the bid is low)."""
        return round(bid.quantity - (total - n) * _face_p(bid.face, pal))

    def honesty(self, name, kind):
        """Share of a player's bids of this kind that were real (the hand
        covered what the bid implied); the rest were bluffs. Their own
        record, shrunk to the table's and to REAL_PRIOR."""
        own = self._player(name).source.honesty[kind]
        table = self.table["source"].honesty[kind]
        f = min(1.0, self.SOURCE_SHRINK / table[1]) if table[1] and not self.NAME_ONLY else 0.0
        real = own[0] + f * table[0] + self.REAL_PRIOR * self.HONESTY_N
        bids = own[1] + f * table[1] + self.HONESTY_N
        return real / bids

    def _blend(self, prior, name, kind, pmf_of):
        """Blend the prior distribution with what the table and the player
        showed, each weighted by its bids."""
        own_w, own = pmf_of(self._player(name).source)
        table_w, table = pmf_of(self.table["source"])
        table_w = 0.0 if self.NAME_ONLY else min(table_w, self.SOURCE_SHRINK)
        out = [self.SOURCE_PRIOR * a + table_w * b + own_w * c for a, b, c in zip(prior, table, own)]
        z = sum(out)
        return [v / z for v in out]

    def _likelihood(self, name, kind):
        """See _likelihood_for; Palifico kinds use PAL_LR_FACTOR times the ratio."""
        return self._likelihood_for(name, kind)

    def _likelihood_for(self, name, kind):
        """lambda per residual (implied - held), index r + R_MAX: the prior
        (real bids REAL_LR times as likely as bluffs), then the table's ratio
        of observed to chance residuals, then the player's own."""
        table = self.GENERIC_LAMBDA.get("2" if getattr(self, "_n_alive", 0) == 2 else "3+") if self.GENERIC_LAMBDA else None
        if table and kind in table:
            lam = list(table[kind])
        else:
            lr = self.REAL_LR_BY_ALIVE.get(getattr(self, "_n_alive", 0), self.REAL_LR)
            if kind.startswith("pal"):
                lr *= self.PAL_LR_FACTOR
            lam = [2 * lr / (lr + 1) if r <= 0 else 2 / (lr + 1) for r in range(-R_MAX, R_MAX + 1)]
        w = self.LR_PRIOR
        sources = (self._player(name).source,) if self.NAME_ONLY else (self.table["source"], self._player(name).source)
        for src in sources:
            obs, chance = src.obs[kind], src.chance[kind]
            lam = [(o + w * l) / (c + w) for o, c, l in zip(obs, chance, lam)]
        return lam

    def _base_pmf(self, name, kind, n, pal):
        """held on the bid's face before the quantity is taken into account:
        a chosen face is (mostly) the face the player holds most of, a
        followed face is chance."""
        if kind.endswith("choice") and self.CHOSEN_BASE:
            return self._chosen_pmf(name, kind, n, pal)
        return _binomial(n, 1 / 6 if (pal or kind.startswith("paco")) else 1 / 3)

    def _bid_pmf(self, name, bid, kind, n, total, pal):
        """held -> probability on the bid's face, given the bid: Bayes with the
        base distribution of the hand and the likelihood of such a quantity
        from it."""
        base = self._base_pmf(name, kind, n, pal)
        implied = self._implied(bid, n, total, pal)
        if not 1 <= implied <= n:          # a low bid, or one beyond their dice: only the face tells
            return self._chosen_pmf(name, kind, n, pal) if kind.endswith("choice") else list(base)
        lam = self._likelihood(name, kind)
        pmf = [b * lam[max(-R_MAX, min(R_MAX, implied - k)) + R_MAX] for k, b in enumerate(base)]
        z = sum(pmf)
        return [v / z for v in pmf]

    @staticmethod
    def _derive(pmf_x, x, y, n, pal):
        """held on face y, given the distribution of held on face x (the face
        the player bid on). Of the k dice that count for a normal x, each is a
        Paco with probability 1/2 (both are 1/6); the other n - k dice are
        spread over the four other normal faces. For x = Paco, the k Pacos
        count for every normal face, the rest spread over five. In Palifico
        nothing is wild."""
        out = [0.0] * (n + 1)
        for k, wk in enumerate(pmf_x):
            if wk < 1e-9:
                continue
            if pal:
                part = _binomial(n - k, 1 / 5)
            elif x == PACO:
                part = [0.0] * k + list(_binomial(n - k, 1 / 5))
            elif y == PACO:
                part = _binomial(k, 1 / 2)
            else:
                part = _conv(_binomial(k, 1 / 2), _binomial(n - k, 1 / 4))
            for j, v in enumerate(part):
                out[j] += wk * v
        return out

    def _chosen_pmf(self, name, kind, n, pal):
        """held -> probability on a face a player bid on without the quantity
        saying anything: real bids come from the face they hold most of,
        bluffs from any hand; blended with what they showed before."""
        h = self.honesty(name, kind)
        if kind.startswith("paco"):
            binom = _binomial(n, 1 / 6)
            real = [0.0] + list(binom[1:])          # a real Paco bid holds a Paco
            z = sum(real)
            real = [v / z for v in real]
        else:
            binom = _binomial(n, 1 / 6 if pal else 1 / 3)
            real = _most_held(n, pal)
        prior = [h * r + (1 - h) * b for r, b in zip(real, binom)]

        def pmf_of(src):
            w = src.chosen.get((kind, n))
            if not w or not sum(w):
                return 0.0, prior
            z = sum(w)
            return z, [v / z for v in w]
        return self._blend(prior, name, kind, pmf_of)

    def _learn_reaction(self, who, action, bid, bidder, before, bids_so_far, dice, counts, total, pal):
        """Fit the player's Dudo / Calza logistics on what they did facing `bid`."""
        p = self._player(who)
        n = counts[who]
        k = _count_matches(dice[who], bid.face, pal)
        rest = _binomial(total - n, _face_p(bid.face, pal))
        p_true = _tail(rest, bid.quantity - k)
        p_exact = _at(rest, bid.quantity - k)
        public = self._public_rest(bids_so_far, bid.face, {m: c for m, c in counts.items() if m != who}, total, pal)
        p_pub = _tail(public, bid.quantity - k)
        jump = bid.quantity - _min_quantity(before, bid.face, pal) if before is not None else 0
        ours = bidder == self.name
        x = self._dudo_x(p_true, n / total, p_pub, jump, ours, p.trust, len(counts))
        y = 1.0 if isinstance(action, Dudo) else 0.0
        self._sgd(p.dudo_w, x, y, 0.0)
        self._sgd(self.table["dudo_w"], x, y, 0.0)
        if not y:
            xc = (1.0, p_exact)
            yc = 1.0 if isinstance(action, Calza) else 0.0
            self._sgd(p.calza_w, xc, yc, 0.0)
            self._sgd(self.table["calza_w"], xc, yc, 0.0)
        p.n += 1
        self.table["n"] += 1

    def _dudo_x(self, p_true, share, p_pub, jump, ours, trust, n_alive):
        t = self.TRUST_START if trust is None else trust
        return (1.0, p_true, share, p_pub, min(max(jump, 0), 3), 1.0 if ours and n_alive >= 3 else 0.0,
                (1 - t) if ours else 0.0, 1 / max(n_alive - 1, 1))

    def _sgd(self, w, x, y, offset):
        pred = _sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + offset)
        g = self.LEARN_RATE * (y - pred)
        for i, xi in enumerate(x):
            w[i] = min(self.W_MAX, max(self.W_MIN, w[i] + g * xi))

    def _decay_memory(self):
        f = self.GAME_DECAY
        if f >= 1:
            return
        t = self.table
        for p in self.players.values():
            if self.NAME_ONLY:
                pulls = ((p.dudo_w, self.DUDO_PRIOR), (p.calza_w, self.CALZA_PRIOR))
            else:
                pulls = ((p.dudo_w, t["dudo_w"]), (p.calza_w, t["calza_w"]))
            for w, tw in pulls:
                for i in range(len(w)):
                    w[i] = tw[i] + f * (w[i] - tw[i])
            p.source.decay(f)
            p.n *= f
        t["source"].decay(f)
        t["n"] *= f

    # ------------------------------------------------------------------
    # the base model at decision time
    # ------------------------------------------------------------------

    PUBLIC = "\x00anyone"      # the model of a bidder nobody knows yet: how others read our bids

    def _pmfs(self, bids, face, players, total, pal):
        """name -> distribution of their matching dice on `face`, for the
        players in `players` (name -> dice), from their latest bid on the face,
        else derived from their latest bid on another face, else chance. Our
        own bids are read as a generic reader would read them."""
        on_face, last = {}, {}                        # name -> (bid, kind)
        chose = set()                                 # (name, face) the player picked themselves
        prev = None
        for who, a in bids:
            kind = _kind(a, prev, pal)
            if kind.endswith("choice"):
                chose.add((who, a.face))
            elif self.CHOSEN_STAYS and (who, a.face) in chose:
                kind = kind[:-len("follow")] + "choice"
            last[who] = (a, kind)
            if a.face == face:
                on_face[who] = (a, kind)
            prev = a
        out = {}
        for name, n in players.items():
            model = self.PUBLIC if name == self.name else name
            if name in on_face:
                bid, kind = on_face[name]
                pmf = self._bid_pmf(model, bid, kind, n, total, pal)
            elif name in last and self.USE_PASSED_OVER:
                bid, kind = last[name]
                pmf = self._derive(self._bid_pmf(model, bid, kind, n, total, pal), bid.face, face, n, pal)
            else:
                pmf = list(_binomial(n, _face_p(face, pal)))
            out[name] = pmf
        return out

    def _public_rest(self, bids, face, players, total, pal):
        """Distribution of the matching dice of `players` together, as the
        bids so far tell it."""
        rest = [1.0]
        for pmf in self._pmfs(bids, face, players, total, pal).values():
            rest = _conv(rest, pmf)
        return rest

    @staticmethod
    def _bids(state):
        return [(rec.player, rec.action) for rec in state.history if isinstance(rec.action, Bid)]

    def _hands(self, state, face, alive, total):
        """Per alive opponent: distribution of their matching dice on `face`."""
        players = {p.name: p.dice_count for p in alive if p.id != state.my_id}
        return self._pmfs(self._bids(state), face, players, total, state.palifico)

    def _face_model(self, state, face, alive, total, nxt):
        """(pmf of the count on the table, pmf excluding the next player, next player's pmf)."""
        hands = self._hands(state, face, alive, total)
        others = [1.0]                      # opponents except the next player
        for name, pmf in hands.items():
            if nxt is None or name != nxt.name:
                others = _conv(others, pmf)
        mine = [0.0] * (state.my_dice_count + 1)
        mine[_count_matches(state.my_dice, face, state.palifico)] = 1.0
        excl = _conv(mine, others)
        nxt_pmf = hands.get(nxt.name) if nxt is not None else None
        full = _conv(excl, nxt_pmf) if nxt_pmf is not None else excl
        return full, excl, nxt_pmf, others, hands, mine

    def _condition(self, pmf, name, bid, kind, n, total, pal):
        """The hand on the bid's face after the player makes `bid`: the
        likelihood of such a quantity from each hand, when it says something."""
        implied = self._implied(bid, n, total, pal)
        if not 1 <= implied <= n:
            return pmf
        lam = self._likelihood(name, kind)
        out = [b * lam[max(-R_MAX, min(R_MAX, implied - k)) + R_MAX] for k, b in enumerate(pmf)]
        z = sum(out)
        return [v / z for v in out] if z > 0 else pmf

    # ------------------------------------------------------------------
    # deciding
    # ------------------------------------------------------------------

    def play(self, state):
        options = self._options(state)
        if not options:
            return state.legal_actions[0]
        best = max(v for _, v in options)
        ties = [a for a, v in options if v >= best - self.TIE_EPS]
        return self._rng.choice(ties)

    def _options(self, state):
        """Every move we consider with its expected value, in dice."""
        self._ensure_memory()
        alive = [p for p in state.players if p.alive]
        total = sum(p.dice_count for p in alive)
        n_alive = len(alive)
        self._n_alive = n_alive
        self._palifico = state.palifico       # also tracked from RoundEndedEvent; the state is authoritative
        order = sorted(alive, key=lambda p: p.id)
        me_i = next(i for i, p in enumerate(order) if p.id == state.my_id)
        nxt = order[(me_i + 1) % n_alive] if n_alive > 1 else None
        w_other = self.OTHER_WEIGHT / max(n_alive - 1, 1)
        # the players after the next one, in playing order, up to us
        self._later = [order[(me_i + i) % n_alive] for i in range(2, n_alive)]
        recovery = all(p.alive for p in state.players)
        legal = set(state.legal_actions)
        pal = state.palifico
        bid = state.current_bid
        models = {}

        def model(face):
            m = models.get(face)
            if m is None:
                m = models[face] = self._face_model(state, face, alive, total, nxt)
            return m

        options = []
        if bid is not None:
            full = model(bid.face)[0]
            p_true = _tail(full, bid.quantity)
            p_exact = _at(full, bid.quantity) * self.EXACT_SHRINK
            if Dudo() in legal:
                options.append((Dudo(), (1 - p_true) * w_other - p_true))
            if Calza() in legal:
                gain = self.CALZA_GAIN if (recovery and state.my_dice_count < 5) else self.CALZA_FREE_PASS
                options.append((Calza(), p_exact * gain - (1 - p_exact)))

        bids = self._bids(state)

        def public(c, others):
            """Matching dice of everyone but the next player, as the bids
            (with our bid c) tell it to a reader."""
            ours = self._pmfs(bids + [(self.name, c)], c.face, {self.name: state.my_dice_count}, total, pal)
            return _conv(others, ours[self.name])

        for cand in self._candidates(state, alive, total, legal):
            full, excl, nxt_pmf, others, hands, mine = model(cand.face)
            options.append((cand, self._bid_value(cand, full, excl, nxt_pmf, nxt, n_alive, total,
                                                  w_other, recovery, pal, state.my_dice_count, 0,
                                                  lambda c, o=others: public(c, o), bid, hands, mine)))
        return options

    def _candidates(self, state, alive, total, legal):
        pal = state.palifico
        bid = state.current_bid
        mine = state.my_dice
        held = {f: _count_matches(mine, f, pal) for f in range(1, 7)}
        cands = []
        if bid is None:
            faces = list(range(1, 7)) if pal else list(_NORMAL_FACES)
            for f in faces:
                if held[f] == 0:
                    continue
                e = held[f] + (total - len(mine)) * _face_p(f, pal)
                for q in range(max(1, math.floor(e) - self.OPEN_BELOW), math.ceil(e) + self.OPEN_ABOVE + 1):
                    cands.append(Bid(q, f))
            bluff = self._bluff_face(state, held, faces)
            if bluff is not None:
                e = self.BLUFF_HELD + (total - len(mine)) * _face_p(bluff, pal)
                for q in range(max(1, math.floor(e)), math.floor(e) + self.BLUFF_SPAN):
                    cands.append(Bid(q, bluff))
        elif pal:
            for q in range(bid.quantity + 1, bid.quantity + 1 + self.SPAN):
                cands.append(Bid(q, bid.face))
        else:
            faces = [f for f in range(1, 7) if held[f] > 0]
            bluff = self._bluff_face(state, held, list(range(1, 7)))
            for f in faces + ([bluff] if bluff is not None else []):
                q0 = _min_quantity(bid, f, pal)
                span = self.SPAN if f in faces else self.BLUFF_SPAN
                for q in range(q0, q0 + span):
                    cands.append(Bid(q, f))
        return [c for c in dict.fromkeys(cands) if c in legal]

    def _bluff_face(self, state, held, faces):
        """A face we do not hold: at the opening a random one, later one from
        the earlier bids in proportion to the bidder's dice."""
        unheld = [f for f in faces if held[f] == 0]
        if not unheld:
            return None
        weights = {}
        counts = {p.name: p.dice_count for p in state.players}
        for rec in state.history:
            a = rec.action
            if isinstance(a, Bid) and rec.player != self.name and a.face in unheld:
                weights[a.face] = weights.get(a.face, 0) + counts.get(rec.player, 1)
        if weights:
            faces_w = list(weights.items())
            r = self._rng.random() * sum(w for _, w in faces_w)
            for f, w in faces_w:
                r -= w
                if r <= 0:
                    return f
            return faces_w[-1][0]
        return self._rng.choice(unheld)

    def _bid_value(self, cand, full, excl, nxt_pmf, nxt, n_alive, total, w_other, recovery, pal,
                   my_dice, depth, public, before, hands=None, mine=None):
        q = cand.quantity
        p_true_all = _tail(full, q)
        value = -self.FALSE_PENALTY * (1 - p_true_all)
        if nxt is None:
            return value
        pr = self._player(nxt.name)
        n = nxt.dice_count
        share = n / total
        perceived = _binomial(total - n, _face_p(cand.face, pal))
        pub = public(cand)
        jump = q - _min_quantity(before, cand.face, pal) if before is not None else 0
        w = pr.dudo_w
        gain_other = (self.CALZA_GAIN if (recovery and n < 5) else self.CALZA_FREE_PASS) * w_other
        cont = None
        for k, wk in enumerate(nxt_pmf):
            if wk < 1e-6:
                continue
            pt = _tail(perceived, q - k)
            pe = _at(perceived, q - k)
            x = self._dudo_x(pt, share, _tail(pub, q - k), jump, True, pr.trust, n_alive)
            p_dudo = _sigmoid(sum(a * b for a, b in zip(w, x)))
            p_calza = (1 - p_dudo) * _sigmoid(pr.calza_w[0] + pr.calza_w[1] * pe)
            p_raise = 1 - p_dudo - p_calza
            true_real = _tail(excl, q - k)
            exact_real = _at(excl, q - k)
            v = p_dudo * (true_real * w_other - (1 - true_real))
            v += p_calza * (exact_real * -gain_other + (1 - exact_real) * w_other)
            if p_raise > 1e-9:
                if cont is None:
                    cont = self._continuation(cand, full, excl, nxt_pmf, nxt, n_alive, total, w_other,
                                              recovery, pal, my_dice, depth, public, hands, mine)
                v += p_raise * cont * self.CONT_SCALE_BY_ALIVE.get(n_alive, self.CONT_SCALE)
            value += wk * v
        return value

    def _continuation(self, cand, full, excl, nxt_pmf, nxt, n_alive, total, w_other, recovery, pal,
                      my_dice, depth, public, hands=None, mine=None):
        """After the next player raises: the players beyond may end the round
        (an opponent loses a die, or gains one with a correct Calza) or raise;
        what comes back to us is valued by our best answer to it: Dudo, Calza,
        or a raise that the next player may call. Every raise assumed on the
        way is evidence that the raiser holds the face, and the hands are
        updated for it."""
        step = max(1, round(self.table["step"]))
        face = cand.face
        condition = self.FIX_CONT_HANDS and hands is not None
        cur = dict(hands) if condition else None
        q = cand.quantity + step
        if condition:              # the next player's assumed raise
            kind = _kind(Bid(q, face), cand, pal)
            cur[nxt.name] = self._condition(cur[nxt.name], nxt.name, Bid(q, face), kind, nxt.dice_count, total, pal)

        def belief():
            if not condition:
                return full
            out = mine
            for pmf in cur.values():
                out = _conv(out, pmf)
            return out

        alive_prob = 1.0
        value = 0.0
        for pl in getattr(self, "_later", ())[:n_alive - 2]:
            m = self._player(pl.name) if self.NAME_ONLY else None
            tw = m.dudo_w if m else self.table["dudo_w"]
            cw = m.calza_w if m else self.table["calza_w"]
            now = belief()
            pt = _tail(now, q)
            pe = _at(now, q)
            x = self._dudo_x(pt, pl.dice_count / total, pt, step - 1, False, None, n_alive)
            p_dudo = _sigmoid(sum(a * b for a, b in zip(tw, x)))
            p_calza = (1 - p_dudo) * _sigmoid(cw[0] + cw[1] * pe)
            if self.FIX_LATER_CALZA:
                gain_pl = (self.CALZA_GAIN if (recovery and pl.dice_count < 5) else self.CALZA_FREE_PASS) * w_other
                value += alive_prob * (p_dudo * w_other + p_calza * (pe * -gain_pl + (1 - pe) * w_other))
            else:
                value += alive_prob * (p_dudo + p_calza) * w_other
            alive_prob *= 1 - p_dudo - p_calza
            q += step
            if condition:          # this player's assumed raise
                cur[pl.name] = self._condition(cur[pl.name], pl.name, Bid(q, face), _kind(Bid(q, face), Bid(q - step, face), pal),
                                               pl.dice_count, total, pal)
        # back to us, facing quantity q on the same face
        now = belief()
        pt = _tail(now, q)
        pe = _at(now, q)
        dudo = (1 - pt) * w_other - pt
        gain = self.CALZA_GAIN if (recovery and my_dice < 5) else self.CALZA_FREE_PASS
        calza = pe * gain - (1 - pe)
        if depth == 0 and q + step <= total:
            if condition:
                nxt_now = cur[nxt.name]
                others_now = [1.0]
                for name, pmf in cur.items():
                    if name != nxt.name:
                        others_now = _conv(others_now, pmf)
                excl_now = _conv(mine, others_now)
            else:
                nxt_now, excl_now = nxt_pmf, excl
            raise_ = self._bid_value(Bid(q + step, face), now, excl_now, nxt_now, nxt, n_alive, total,
                                     w_other, recovery, pal, my_dice, depth + 1, public, Bid(q, face))
        else:
            raise_ = -self.RETURN_RISK * (1 - _tail(now, q + step))
        return value + alive_prob * max(dudo, calza, raise_)
