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

New players start as the table's average. Everything learned lives in
class-level memory per bot name, survives a new instance per game, counts
GAME_DECAY times as much after every game, and never touches the disk.
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
    from the reveals: per kind of bid the residual (implied - held) when the
    bid's quantity says something about the hand, and the held count on a
    face they chose when it does not. Bluffs are the residuals above zero."""

    def __init__(self):
        self.resid = {k: {} for k in KINDS}          # kind -> residual -> weight
        self.chosen = {}                              # (kind, dice) -> [weight per held]
        self.honesty = {k: [0.0, 0.0] for k in KINDS}  # kind -> [real, bids]

    def decay(self, f):
        for d in self.resid.values():
            for r in d:
                d[r] *= f
        for w in self.chosen.values():
            for i in range(len(w)):
                w[i] *= f
        for h in self.honesty.values():
            h[0] *= f
            h[1] *= f


class _Player:
    """What we learned about one opponent, by name."""

    def __init__(self, dudo_w, calza_w):
        self.dudo_w = list(dudo_w)        # logit = w0 + w1 * p_true + w2 * dice share
        self.calza_w = list(calza_w)      # logit = w0 + w1 * p_exact
        self.source = _Source()
        self.trust = None                 # share of our revealed bids that were true (EMA)
        self.n = 0                        # reactions seen


class CarlosDosBot:
    name = "Carlos Dos"
    count = 1

    # -- values ---------------------------------------------------------------
    # An opponent's lost die, divided by the number of opponents.
    OTHER_WEIGHT = 1.0
    # A correct Calza of ours while it gains a die, and once it does not.
    CALZA_GAIN = 1.0
    CALZA_FREE_PASS = 0.3
    # A bid that is not on the table costs this per unit of P(false), on top
    # of what the model expects from challenges.
    FALSE_PENALTY = 0.2
    # When a bid comes back to us we weigh Dudo, Calza and our next raise,
    # the raise against the next player's reaction model. Beyond that one
    # level, raising is valued as -(1 - P(raise true)) * this.
    RETURN_RISK = 0.5
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

    # -- starting model: probability only (the field is unknown) ---------------
    # A player calls Dudo when the bid looks unlikely from their dice: the
    # logit w0 + w1 * P(true) + w2 * dice share is 50% at P(true) = 0.3,
    # about where calling breaks even at a table of 3-4.
    DUDO_PRIOR = (3.0, -10.0, 0.0)
    # Calza: w0 + w1 * P(exact), 50% at P(exact) = 0.5 (break-even: a die
    # won against a die lost).
    CALZA_PRIOR = (-6.0, 12.0)
    # Added to the Dudo logit on OUR bids: TRUST_WEIGHT * (1 - trust).
    TRUST_WEIGHT = 2.0
    TRUST_START = 0.7
    TRUST_LR = 0.15
    # Source model before we know a player: a bid is real with probability
    # REAL_PRIOR (per kind of bid, learned per player from then on) and a
    # bluff otherwise. A real bid on a chosen face means the face is the one
    # the player holds most of (the distribution of that count follows from
    # the dice); a real bid whose quantity says something means the player
    # holds at least what it implies. A bluff says nothing about the hand.
    # Weight of that prior and of the table's pooled model, in bids.
    REAL_PRIOR = 0.7
    SOURCE_PRIOR = 4.0
    SOURCE_SHRINK = 4.0
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
            t = self.table
            p = _Player(t["dudo_w"], t["calza_w"])
            self.players[name] = p
        return p

    def _learn(self, event):
        """Replay the round with everyone's dice known."""
        dice = dict(event.dice)
        names = [n for n, _ in event.dice]
        counts = {n: len(d) for n, d in event.dice}
        total = sum(counts.values())
        pal = self._palifico
        if self._round and not isinstance(self._round[-1][1], Bid):
            actions = self._round
        else:                     # the challenge came through the reveal only
            actions = self._round + [(event.challenger, Dudo() if event.challenge_type == "dudo" else Calza())]
        our_truths = []
        current, bidder = None, None
        for who, action in actions:
            if current is not None and who != self.name and who in dice:
                self._learn_reaction(who, action, current, bidder, dice, counts, total, pal)
            if isinstance(action, Bid):
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
        for src in (self._player(who).source, self.table["source"]):
            if 1 <= implied <= n:
                r = max(-R_MAX, min(R_MAX, implied - k))
                src.resid[kind][r] = src.resid[kind].get(r, 0.0) + 1
            else:
                w = src.chosen.setdefault((kind, n), [0.0] * (n + 1))
                w[k] += 1
            h = src.honesty[kind]
            h[0] += k >= max(implied, 1 if kind.endswith("choice") else 0)
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
        f = min(1.0, self.SOURCE_SHRINK / table[1]) if table[1] else 0.0
        real = own[0] + f * table[0] + self.REAL_PRIOR * self.HONESTY_N
        bids = own[1] + f * table[1] + self.HONESTY_N
        return real / bids

    def _blend(self, prior, name, kind, pmf_of):
        """Blend the prior distribution with what the table and the player
        showed, each weighted by its bids."""
        own_w, own = pmf_of(self._player(name).source)
        table_w, table = pmf_of(self.table["source"])
        table_w = min(table_w, self.SOURCE_SHRINK)
        out = [self.SOURCE_PRIOR * a + table_w * b + own_w * c for a, b, c in zip(prior, table, own)]
        z = sum(out)
        return [v / z for v in out]

    def _resid_pmf(self, name, kind, n, implied, p):
        """held -> probability given a bid that implies `implied` of the face:
        prior from the dice (real: at least implied, bluff: anything), blended
        with the learned residual histograms."""
        h = self.honesty(name, kind)
        binom = _binomial(n, p)
        pal = kind.startswith("pal")
        # a real bid holds at least what it implies; on a chosen face it is
        # the face the player holds most of
        base = binom if kind.endswith("follow") or kind.startswith("paco") else _most_held(n, pal)
        real = [b if k >= implied else 0.0 for k, b in enumerate(base)]
        z = sum(real)
        prior = [h * r / z + (1 - h) * b for r, b in zip(real, binom)]

        def pmf_of(src):
            d = src.resid[kind]
            w = sum(d.values())
            if not w:
                return 0.0, prior
            pmf = [d.get(max(-R_MAX, min(R_MAX, implied - k)), 0.0) for k in range(n + 1)]
            z = sum(pmf)
            return (w, [v / z for v in pmf]) if z else (0.0, prior)
        return self._blend(prior, name, kind, pmf_of)

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

    def _learn_reaction(self, who, action, bid, bidder, dice, counts, total, pal):
        """Fit the player's Dudo / Calza logistics on what they did facing `bid`."""
        p = self._player(who)
        n = counts[who]
        k = _count_matches(dice[who], bid.face, pal)
        rest = _binomial(total - n, _face_p(bid.face, pal))
        p_true = _tail(rest, bid.quantity - k)
        p_exact = _at(rest, bid.quantity - k)
        share = n / total
        ours = bidder == self.name
        trust = (self.TRUST_START if p.trust is None else p.trust) if ours else None
        x = (1.0, p_true, share)
        y = 1.0 if isinstance(action, Dudo) else 0.0
        self._sgd(p.dudo_w, x, y, self.TRUST_WEIGHT * (1 - trust) if ours else 0.0)
        self._sgd(self.table["dudo_w"], x, y, self.TRUST_WEIGHT * (1 - trust) if ours else 0.0)
        if not y:
            xc = (1.0, p_exact)
            yc = 1.0 if isinstance(action, Calza) else 0.0
            self._sgd(p.calza_w, xc, yc, 0.0)
            self._sgd(self.table["calza_w"], xc, yc, 0.0)
        p.n += 1
        self.table["n"] += 1

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
            for w, tw in ((p.dudo_w, t["dudo_w"]), (p.calza_w, t["calza_w"])):
                for i in range(len(w)):
                    w[i] = tw[i] + f * (w[i] - tw[i])
            p.source.decay(f)
            p.n *= f
        t["source"].decay(f)
        t["n"] *= f

    # ------------------------------------------------------------------
    # the base model at decision time
    # ------------------------------------------------------------------

    def _hands(self, state, face, alive, total):
        """Per alive opponent: distribution of their matching dice on `face`."""
        pal = state.palifico
        p = _face_p(face, pal)
        latest = {}                                   # name -> (strongest bid on the face, kind)
        prev = None
        for rec in state.history:
            a = rec.action
            if isinstance(a, Bid):
                if a.face == face and rec.player != self.name:
                    latest[rec.player] = (a, _kind(a, prev, pal))
                prev = a
        hands = {}
        for pl in alive:
            if pl.id == state.my_id:
                continue
            n = pl.dice_count
            pmf = list(_binomial(n, p))
            if pl.name in latest:
                bid, kind = latest[pl.name]
                implied = self._implied(bid, n, total, pal)
                if 1 <= implied <= n:
                    pmf = self._resid_pmf(pl.name, kind, n, implied, p)
                else:       # a low bid, or one beyond their dice: only the face tells
                    pmf = self._chosen_pmf(pl.name, kind, n, pal)
            hands[pl.name] = pmf
        return hands

    def _face_model(self, state, face, alive, total, nxt):
        """(pmf of the count on the table, pmf excluding the next player, next player's pmf)."""
        hands = self._hands(state, face, alive, total)
        mine = [0.0] * (state.my_dice_count + 1)
        mine[_count_matches(state.my_dice, face, state.palifico)] = 1.0
        excl = mine
        for name, pmf in hands.items():
            if nxt is None or name != nxt.name:
                excl = _conv(excl, pmf)
        nxt_pmf = hands.get(nxt.name) if nxt is not None else None
        full = _conv(excl, nxt_pmf) if nxt_pmf is not None else excl
        return full, excl, nxt_pmf

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
        order = sorted(alive, key=lambda p: p.id)
        me_i = next(i for i, p in enumerate(order) if p.id == state.my_id)
        nxt = order[(me_i + 1) % n_alive] if n_alive > 1 else None
        w_other = self.OTHER_WEIGHT / max(n_alive - 1, 1)
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
            p_exact = _at(full, bid.quantity)
            if Dudo() in legal:
                options.append((Dudo(), (1 - p_true) * w_other - p_true))
            if Calza() in legal:
                gain = self.CALZA_GAIN if (recovery and state.my_dice_count < 5) else self.CALZA_FREE_PASS
                options.append((Calza(), p_exact * gain - (1 - p_exact)))

        for cand in self._candidates(state, alive, total, legal):
            full, excl, nxt_pmf = model(cand.face)
            options.append((cand, self._bid_value(cand, full, excl, nxt_pmf, nxt, n_alive, total,
                                                  w_other, recovery, pal, state.my_dice_count, 0)))
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
                   my_dice, depth):
        q = cand.quantity
        p_true_all = _tail(full, q)
        value = -self.FALSE_PENALTY * (1 - p_true_all)
        if nxt is None:
            return value
        pr = self._player(nxt.name)
        n = nxt.dice_count
        share = n / total
        trust = self.TRUST_START if pr.trust is None else pr.trust
        offset = self.TRUST_WEIGHT * (1 - trust)
        perceived = _binomial(total - n, _face_p(cand.face, pal))
        gain_other = (self.CALZA_GAIN if (recovery and n < 5) else self.CALZA_FREE_PASS) * w_other
        cont = None
        for k, wk in enumerate(nxt_pmf):
            if wk < 1e-6:
                continue
            pt = _tail(perceived, q - k)
            pe = _at(perceived, q - k)
            p_dudo = _sigmoid(pr.dudo_w[0] + pr.dudo_w[1] * pt + pr.dudo_w[2] * share + offset)
            p_calza = (1 - p_dudo) * _sigmoid(pr.calza_w[0] + pr.calza_w[1] * pe)
            p_raise = 1 - p_dudo - p_calza
            true_real = _tail(excl, q - k)
            exact_real = _at(excl, q - k)
            v = p_dudo * (true_real * w_other - (1 - true_real))
            v += p_calza * (exact_real * -gain_other + (1 - exact_real) * w_other)
            if p_raise > 1e-9:
                if cont is None:
                    cont = self._continuation(cand, full, excl, nxt_pmf, nxt, n_alive, total, w_other,
                                              recovery, pal, my_dice, depth)
                v += p_raise * cont
            value += wk * v
        return value

    def _continuation(self, cand, full, excl, nxt_pmf, nxt, n_alive, total, w_other, recovery, pal,
                      my_dice, depth):
        """After the next player raises: the players beyond may end the round
        (an opponent loses a die) or raise; what comes back to us is valued
        by our best answer to it: Dudo, Calza, or a raise that the next
        player may call."""
        step = max(1, round(self.table["step"]))
        tw = self.table["dudo_w"]
        cw = self.table["calza_w"]
        alive_prob = 1.0
        value = 0.0
        q = cand.quantity + step
        for _ in range(n_alive - 2):
            pt = _tail(full, q)
            pe = _at(full, q)
            p_end = _sigmoid(tw[0] + tw[1] * pt + tw[2] / n_alive)
            p_end += (1 - p_end) * _sigmoid(cw[0] + cw[1] * pe)
            value += alive_prob * p_end * w_other
            alive_prob *= 1 - p_end
            q += step
        # back to us, facing quantity q on the same face
        pt = _tail(full, q)
        pe = _at(full, q)
        dudo = (1 - pt) * w_other - pt
        gain = self.CALZA_GAIN if (recovery and my_dice < 5) else self.CALZA_FREE_PASS
        calza = pe * gain - (1 - pe)
        if depth == 0 and q + step <= total:
            raise_ = self._bid_value(Bid(q + step, cand.face), full, excl, nxt_pmf, nxt, n_alive, total,
                                     w_other, recovery, pal, my_dice, depth + 1)
        else:
            raise_ = -self.RETURN_RISK * (1 - _tail(full, q + step))
        return value + alive_prob * max(dudo, calza, raise_)
