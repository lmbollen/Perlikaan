"""
Carlos: kiest zijn zet met een Markov-keten over de rest van de ronde.

The rest of a round is an absorbing Markov chain. A state is (current bid,
who is to act, who made the bid). From a state the player to act either
calls Dudo or Calza (absorbing: someone loses or gains a die) or raises to a
new bid, which is the next state. Bids only go up, so the chain has no
cycles and the absorption probabilities follow from one backward pass.

Carlos fills in the transition probabilities from what it learns about every
player, by name, from the reveals at the end of each round:

  * how a player comes to a bid: for every bid, how many matching dice the
    player really held compared to what the bid implied (quantity minus the
    expected share of the other dice). That histogram, per kind of bid
    (opening/switching vs following, normal vs Palifico), is learned from the
    reveals and is directly the distribution of the player's hand given such
    a bid (Bayes with the observed frequencies). A player who bluffs often
    gets a wide histogram that reaches far below zero;
  * what a player's choice says about the other faces: someone who opened
    on or switched to a face probably holds fewer of the faces they passed
    over; that distribution is learned per player too;
  * how a player reacts to a bid: P(Dudo) as a logistic function of how
    likely the bid looks from their own dice (fitted per player), how often
    they call Calza, and how bold their own raises are (their usual P(true)).

For every candidate move (Dudo, Calza, and a short list of bids per face)
Carlos walks the chain from that move over the players after it, up to one
round around the table, and takes the move with the best expected value:
losing a die ourselves counts -1, an opponent losing one +1/(opponents), a
correct Calza of ours +1 (only while it gains a die).

New players start out as the average of everyone seen so far.

Zet TALK_TO_ME=true om Carlos te horen.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from perudo.models import PACO, Bid, Calza, Dudo, RevealEvent, RoundEndedEvent, GameEndedEvent
from perudo.models import BidEvent, CalzaEvent, DieLostEvent, DudoEvent
from perudo.profiler import OpponentProfiler
from perudo.rules import count_matches
from perudo.talk import Mouth, fmt_bid, last_bidder

QUOTES = {
    "open": [
        "{bid}. De keten is berekend, amigo.",
        "{bid}. Ik heb alle toestanden al gezien.",
    ],
    "raise": [
        "{bid}. Overgangskans in mijn voordeel.",
        "{bid}. Jouw volgende toestand is een slechte.",
        "{bid}.",
    ],
    "dudo": [
        "Dudo. {bidder}, de keten zegt nee.",
        "Dudo. Absorberende toestand bereikt, {bidder}.",
    ],
    "calza": [
        "Calza. Precies wat de verdeling zei.",
    ],
    "dudo_right": ["Zoals berekend. Gracias, {bidder}."],
    "dudo_wrong": ["{actual}. Een uitschieter in de staart. Gebeurt."],
    "bid_held": ["{challenger}, je reageerde precies zoals je model zei."],
    "won": ["Carlos wint. De keten wist het al."],
}

# Residual = matches held - matches the bid implied, clipped to this range.
R_MIN, R_MAX = -8, 8


def binom_pmf(n, p):
    return [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]


def convolve(a, b):
    out = [0.0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def tail(pmf, need):
    """P(X >= need) for a pmf over 0..len-1."""
    if need <= 0:
        return 1.0
    if need >= len(pmf):
        return 0.0
    return sum(pmf[need:])


def at(pmf, k):
    return pmf[k] if 0 <= k < len(pmf) else 0.0


def binom_tail(n, p, need):
    """P(Binomial(n, p) >= need), linearly interpolated for fractional need."""
    lo = math.floor(need)
    frac = need - lo

    def t(k):
        if k <= 0:
            return 1.0
        if k > n:
            return 0.0
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    return (1 - frac) * t(lo) + frac * t(lo + 1)


def p_face(face, palifico):
    return 1 / 6 if (palifico or face == PACO) else 1 / 3


def min_quantity(bid, face, palifico):
    """Smallest legal quantity on `face` after `bid`, or None if not allowed."""
    q0, f0 = bid.quantity, bid.face
    if palifico:
        return q0 + 1 if face == f0 else None
    if f0 == PACO:
        return q0 + 1 if face == PACO else 2 * q0 + 1
    if face == PACO:
        return math.ceil(q0 / 2)
    return q0 if face > f0 else q0 + 1


class CarlosBot:
    name = "Carlos"

    # Value of an opponent losing a die, divided by the number of opponents.
    OTHER_WEIGHT = 1.0
    # Value of gaining a die with a correct Calza.
    CALZA_GAIN = 1.0
    # Pseudo-observations for the hand-likelihood histogram (shrinks a player
    # towards the table average, which shrinks towards the base prior).
    SHRINK = 20
    # Logistic Dudo model for a player we know little about:
    # P(Dudo) = sigmoid(DUDO_A + DUDO_B * P(bid true from their dice)).
    DUDO_A, DUDO_B = 2.0, -7.0
    # Weight of that prior in the per-player fit, in observations.
    DUDO_PRIOR_N = 30
    # How bold a player raises (P(true) of their own raises from their own
    # dice) before we know them.
    DEFAULT_RAISE_P = 0.7
    # Candidate bids: per face this many quantities from the minimum up,
    # and only those at least CAND_MIN likely to us.
    CAND_PER_FACE = 5
    CAND_MIN = 0.2
    # Correct our P(true) / P(exact) of the bid we face by how often bids we
    # rated alike were really true (per table size and Palifico); weight of
    # the raw estimate in observations.
    CALIBRATION_PRIOR = 20
    # Correct the predicted Dudo / Calza chance of the player right after our
    # bid by how that player really responded to our bids before (per table
    # size and prediction bucket); weight of the prediction in observations.
    RESPONSE_PRIOR = 20
    # The chain looks only one round of turns ahead, so it underrates the risk
    # of a bid (later in the round we face higher bids). Learn how often we
    # really lost a die in rounds where we made a bid with a given predicted
    # risk (per table size, opening vs raising) and decide on that. Off: it
    # did not pay off against the core bots (+0.2 +/- 0.9 pp at 5 players,
    # -1.4 +/- 0.8 pp head-to-head).
    RISK_CALIBRATION = False
    RISK_PRIOR = 20
    # Moves whose expected value is within TIE_EPS of the best one count as
    # equally good; Carlos picks one of them at random, so he is harder to
    # read.
    TIE_EPS = 0.01
    # How many times the chain runs up to our turn (with 3+ players). 1: stop when it is our
    # turn again and value that with a rule of thumb. 2: at our next turn
    # weigh our real options (Dudo, Calza, raises) and follow each raise
    # around the table once more.
    LAPS = 1
    # Head-to-head the chain is short and stays accurate, so look further.
    # Benchmarked against the core bots: 2 laps +2.0 +/- 1.1 pp head-to-head,
    # but -1.5 (2 laps) / -2.5 (3 laps) pp at 5 players.
    LAPS_HEADS_UP = 2
    # Raises per face we consider at a later turn of ours (keeps it fast).
    CAND_INNER = 2
    # When the chain comes back to us, raising is valued as
    # -(1 - P(our best raise true)) * this.
    OUR_RAISE_RISK = 0.5

    def __init__(self):
        self.name = "Carlos"
        self.mouth = Mouth(self, QUOTES)
        self.profiler = OpponentProfiler(self.name)
        self._rng = random.Random()
        # name -> kind -> residual histogram (dict r -> count)
        self.hist = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        # name -> dice count -> k -> count: matches of a face a player did NOT
        # bid on while choosing another face this round.
        self.other = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self._round = []           # (player, action) this round
        self._palifico = False
        self._fit_cache = {}
        self._calibration = {}
        self._pending = []
        self._own_calibration = {}
        self._own_pending = []
        self._response = {}     # (kind, size, bucket) -> [hits, n]
        self._risk = {}         # (opening, size, bucket) -> [lost, n]
        self._risk_pending = []
        self._lost_this_round = False
        self._awaiting = None   # (size, pd, pc) of our last bid, until the next player acts

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------

    def observe(self, event):
        self.profiler.observe(event)
        if self._awaiting is not None and isinstance(event, (BidEvent, DudoEvent, CalzaEvent)) \
                and event.player != self.name:
            size, pd, pc = self._awaiting
            for kind, p, hit in (("dudo", pd, isinstance(event, DudoEvent)),
                                 ("calza", pc, isinstance(event, CalzaEvent))):
                stat = self._response.setdefault((kind, size, self._resp_bucket(p)), [0, 0])
                stat[0] += hit
                stat[1] += 1
            self._awaiting = None
        elif isinstance(event, (RevealEvent, GameEndedEvent)):
            self._awaiting = None
        if isinstance(event, DieLostEvent) and event.player == self.name:
            self._lost_this_round = True
        if isinstance(event, (RoundEndedEvent, GameEndedEvent)):
            for key in self._risk_pending:
                stat = self._risk.setdefault(key, [0, 0])
                stat[0] += self._lost_this_round
                stat[1] += 1
            self._risk_pending = []
            self._lost_this_round = False
        if isinstance(event, BidEvent):
            self._round.append((event.player, event.bid))
        elif isinstance(event, (DudoEvent, CalzaEvent)):
            self._round.append((event.player, None))
        elif isinstance(event, RevealEvent):
            self._learn_hands(event, self._palifico)
            self._learn_calibration(event)
            self._learn_own_calibration(event)
            self._round = []
        elif isinstance(event, RoundEndedEvent):
            self._palifico = event.palifico_next
        elif isinstance(event, GameEndedEvent):
            self._palifico = False
            self._round = []
        self.mouth.react(event)

    @staticmethod
    def _last_claims(actions, me):
        """(player, face) -> (strongest bid on it, chosen rather than followed)."""
        last = {}
        prev = None
        for player, bid in actions:
            if bid is None:
                break
            if player != me:
                chosen = prev is None or prev.face != bid.face
                old = last.get((player, bid.face))
                last[(player, bid.face)] = (bid, chosen or (old is not None and old[1]))
            prev = bid
        return last

    @staticmethod
    def _kind(chosen, palifico, n):
        """Kind of bid: chosen or followed, Palifico or not, bidder's dice 1/2/3+."""
        return ("pal-" if palifico else "") + ("choice" if chosen else "follow") + f"-{min(n, 3)}"

    def _learn_hands(self, event, palifico):
        dice = event.dice_by_player()
        total = sum(len(d) for d in dice.values())
        claims = self._last_claims(self._round, self.name)
        if not palifico:
            choosers = {pl for (pl, _), (_, chosen) in claims.items() if chosen}
            for player in choosers & set(dice):
                bid_on = {f for (pl, f) in claims if pl == player}
                own = dice[player]
                for f in range(2, 7):
                    if f not in bid_on:
                        self.other[player][len(own)][count_matches(own, f, False)] += 1
        for (player, face), (bid, chosen) in claims.items():
            if player not in dice:
                continue
            own = dice[player]
            k = count_matches(own, face, palifico)
            r = k - self._implied(bid, len(own), total, palifico)
            r = max(R_MIN, min(R_MAX, r))
            self.hist[player][self._kind(chosen, palifico, len(own))][r] += 1

    @staticmethod
    def _implied(bid, n, total, palifico=False):
        """Matches a bidder 'needs' themselves for the bid, given chance elsewhere."""
        p = p_face(bid.face, palifico)
        return max(0, min(n, round(bid.quantity - (total - n) * p)))

    def _likelihood(self, player, kind):
        """r -> P(residual r) for this player and kind, shrunk to the table."""
        base = {r: math.exp(-abs(r)) for r in range(R_MIN, R_MAX + 1)}
        z = sum(base.values())
        base = {r: v / z for r, v in base.items()}
        pooled = defaultdict(float)
        n_pooled = 0.0
        for who, kinds in self.hist.items():
            for r, c in kinds[kind].items():
                pooled[r] += c
                n_pooled += c
        table = {r: (pooled[r] + self.SHRINK * base[r]) / (n_pooled + self.SHRINK) for r in base}
        own = self.hist[player][kind] if player in self.hist else {}
        n_own = sum(own.values())
        return {r: (own.get(r, 0.0) + self.SHRINK * table[r]) / (n_own + self.SHRINK) for r in base}

    def _calibrated(self, state, bid, p, exact):
        alive = sum(pl.alive for pl in state.players)
        key = (exact, min(alive, 4), state.palifico, min(int(p * 10), 9))
        self._pending.append((key, bid, state.palifico))
        hits, n = self._calibration.get(key, (0, 0))
        k = self.CALIBRATION_PRIOR
        return (hits + k * p) / (n + k)

    def _learn_calibration(self, event):
        dice = [d for _, hand in event.dice for d in hand]
        for key, bid, palifico in self._pending:
            stat = self._calibration.setdefault(key, [0, 0])
            actual = count_matches(dice, bid.face, palifico)
            stat[0] += (actual == bid.quantity) if key[0] else (actual >= bid.quantity)
            stat[1] += 1
        self._pending = []

    def _other_face_pmf(self, player, n):
        """Matches of a face `player` passed over for another face (n dice)."""
        base = binom_pmf(n, 1 / 3)
        pooled = [sum(self.other[who][n].get(k, 0.0) for who in self.other) for k in range(n + 1)]
        n_pool = sum(pooled)
        table = [(pooled[k] + self.SHRINK * base[k]) / (n_pool + self.SHRINK) for k in range(n + 1)]
        own = [self.other[player][n].get(k, 0.0) for k in range(n + 1)] if player in self.other else [0.0] * (n + 1)
        n_own = sum(own)
        return [(own[k] + self.SHRINK * table[k]) / (n_own + self.SHRINK) for k in range(n + 1)]

    # -- calibration of our own bids ----------------------------------------

    @staticmethod
    def _own_key(ctx, p):
        return (min(len(ctx["dice"]), 4), ctx["pal"], min(int(p * 10), 9))

    def _own_shift(self, ctx, bid):
        """
        Log-odds correction for our own bid: how often our bids with a similar
        raw P(true) were really true. Applied to every conditional P(true) of
        that bid in the chain.
        """
        if bid in ctx["own_shift"]:
            return ctx["own_shift"][bid]
        p = min(max(self._p_true(ctx, bid), 1e-4), 1 - 1e-4)
        hits, n = self._own_calibration.get(self._own_key(ctx, p), (0, 0))
        k = self.CALIBRATION_PRIOR
        pc = min(max((hits + k * p) / (n + k), 1e-4), 1 - 1e-4)
        shift = math.log(pc / (1 - pc)) - math.log(p / (1 - p))
        ctx["own_shift"][bid] = shift
        return shift

    def _learn_own_calibration(self, event):
        dice = [d for _, hand in event.dice for d in hand]
        for key, bid, palifico in self._own_pending:
            stat = self._own_calibration.setdefault(key, [0, 0])
            stat[0] += count_matches(dice, bid.face, palifico) >= bid.quantity
            stat[1] += 1
        self._own_pending = []

    def _dudo_model(self, name):
        """(a, b) of the logistic Dudo model, fitted on the player's history."""
        prof = self.profiler.profiles.get(name)
        n = len(prof.facing) if prof else 0
        cached = self._fit_cache.get(name)
        if cached and n - cached[0] < 30:
            return cached[1]
        a, b = self.DUDO_A, self.DUDO_B
        if n >= 10:
            data = prof.facing[-400:]
            lam = 1.0 / self.DUDO_PRIOR_N
            for _ in range(8):
                # Newton step on log-likelihood with a Gaussian prior around
                # the default parameters.
                ga = gb = haa = hab = hbb = 0.0
                for x, y in data:
                    s = 1 / (1 + math.exp(-(a + b * x)))
                    e = (1.0 if y else 0.0) - s
                    w = s * (1 - s)
                    ga += e
                    gb += e * x
                    haa += w
                    hab += w * x
                    hbb += w * x * x
                m = len(data)
                ga -= lam * m * (a - self.DUDO_A) / 10
                gb -= lam * m * (b - self.DUDO_B) / 10
                haa += lam * m / 10
                hbb += lam * m / 10
                det = haa * hbb - hab * hab
                if det <= 1e-9:
                    break
                a += (hbb * ga - hab * gb) / det
                b += (haa * gb - hab * ga) / det
        self._fit_cache[name] = (n, (a, b))
        return a, b

    def _raise_p(self, name):
        prof = self.profiler.profiles.get(name)
        if prof is None or prof.bids < 10:
            return self.DEFAULT_RAISE_P
        return (prof.bid_risk_sum + 10 * self.DEFAULT_RAISE_P) / (prof.bids + 10)

    # ------------------------------------------------------------------
    # this round's beliefs
    # ------------------------------------------------------------------

    def _posteriors(self, state):
        """name -> face -> pmf over that player's matches for the face."""
        total = sum(p.dice_count for p in state.players if p.alive)
        pal = state.palifico
        last = self._last_claims([(r.player, r.action if isinstance(r.action, Bid) else None)
                                  for r in state.history], self.name)
        choosers = {pl for (pl, _), (_, chosen) in last.items() if chosen}
        post = {}
        for p in state.players:
            if not p.alive or p.name == self.name:
                continue
            post[p.name] = {}
            for face in range(1, 7):
                seen = last.get((p.name, face))
                if seen is None and not pal and face != PACO and p.name in choosers:
                    # They chose another face: probably fewer of this one.
                    pmf = self._other_face_pmf(p.name, p.dice_count)
                elif seen is None:
                    pmf = binom_pmf(p.dice_count, p_face(face, pal))
                else:
                    # The learned residual histogram is already the
                    # distribution of their matches given such a bid.
                    bid, chosen = seen
                    lik = self._likelihood(p.name, self._kind(chosen, pal, p.dice_count))
                    implied = self._implied(bid, p.dice_count, total, pal)
                    pmf = [lik[max(R_MIN, min(R_MAX, k - implied))] for k in range(p.dice_count + 1)]
                    z = sum(pmf)
                    pmf = [w / z for w in pmf]
                post[p.name][face] = pmf
        return post

    def _sums(self, post, face, exclude=None):
        pmf = [1.0]
        for name, faces in post.items():
            if name != exclude:
                pmf = convolve(pmf, faces[face])
        return pmf

    # ------------------------------------------------------------------
    # the chain
    # ------------------------------------------------------------------

    def _setup(self, state):
        alive = [p for p in state.players if p.alive]
        seats = list(state.players)
        me = next(i for i, p in enumerate(seats) if p.id == state.my_id)
        order = []
        for step in range(1, len(seats) + 1):
            p = seats[(me + step) % len(seats)]
            if p.alive:
                order.append(p.name)       # ends with ourselves
        ctx = dict(
            state=state,
            total=sum(p.dice_count for p in alive),
            dice={p.name: p.dice_count for p in alive},
            order=order,
            w=self.OTHER_WEIGHT / max(len(alive) - 1, 1),
            gain_ok=all(p.alive for p in state.players),
            post=self._posteriors(state),
            pal=state.palifico,
            pal_face=None,
        )
        if state.palifico and state.history:
            first = next((r.action for r in state.history if isinstance(r.action, Bid)), None)
            ctx["pal_face"] = first.face if first else None
        ctx["all"] = {f: self._sums(ctx["post"], f) for f in range(1, 7)}
        ctx["loo"] = {(n, f): self._sums(ctx["post"], f, exclude=n)
                      for n in ctx["post"] for f in range(1, 7)}
        ctx["mine"] = {f: count_matches(state.my_dice, f, state.palifico) for f in range(1, 7)}
        ctx["memo"] = {}
        ctx["own_shift"] = {}
        return ctx

    def _lose_value(self, ctx, who):
        """Value for us when `who` loses a die."""
        return -1.0 if who == self.name else ctx["w"]

    def _p_true(self, ctx, bid):
        return tail(ctx["all"][bid.face], bid.quantity - ctx["mine"][bid.face])

    def _p_exact(self, ctx, bid):
        return at(ctx["all"][bid.face], bid.quantity - ctx["mine"][bid.face])

    def _their_raise(self, ctx, who, bid):
        """The raise `who` most likely makes over `bid`, or None."""
        n = ctx["dice"][who]
        total, pal = ctx["total"], ctx["pal"]
        target = self._raise_p(who)
        faces = [ctx["pal_face"] or bid.face] if pal else [2, 3, 4, 5, 6, PACO]
        best = None
        for f in faces:
            q = min_quantity(bid, f, pal)
            if q is None or q > total:
                continue
            post = ctx["post"][who][f]
            mean_k = sum(k * w for k, w in enumerate(post))
            p = p_face(f, pal)

            def view(qq):
                return binom_tail(total - n, p, qq - mean_k)

            v = view(q)
            if best is None or v > best[0]:
                while q + 1 <= total and view(q + 1) >= target:
                    q += 1
                best = (v, Bid(q, f))
        return None if best is None else best[1]

    def _our_turn_value(self, ctx, bid, bidder):
        """Approximate value when the chain returns to us facing `bid`."""
        dudo = (1 - self._p_true(ctx, bid)) * self._lose_value(ctx, bidder) - self._p_true(ctx, bid)
        best_raise = 0.0
        for f in ([ctx["pal_face"] or bid.face] if ctx["pal"] else [2, 3, 4, 5, 6, PACO]):
            q = min_quantity(bid, f, ctx["pal"])
            if q is not None and q <= ctx["total"]:
                best_raise = max(best_raise, self._p_true(ctx, Bid(q, f)))
        return max(dudo, -(1 - best_raise) * self.OUR_RAISE_RISK)

    def _risk_key(self, ctx, opening, p_lose):
        return (opening, min(len(ctx["dice"]), 4), min(int(max(p_lose, 0.0) * 20), 19))

    def _calibrated_bid_value(self, ctx, opening, value):
        """Chain value of a bid with its risk replaced by the learned one."""
        w = ctx["w"]
        p_lose = (w - value) / (1 + w)
        key = self._risk_key(ctx, opening, p_lose)
        if not self.RISK_CALIBRATION:
            return value, key
        lost, n = self._risk.get(key, (0, 0))
        k = self.RISK_PRIOR
        p_cal = (lost + k * max(p_lose, 0.0)) / (n + k)
        return w - (1 + w) * p_cal, key

    @staticmethod
    def _resp_bucket(p):
        return min(int(p * 20), 19)

    def _response_ratio(self, ctx, kind, p):
        """Observed / predicted response rate of the next player to our bids."""
        hits, n = self._response.get((kind, min(len(ctx["dice"]), 4), self._resp_bucket(p)), (0, 0))
        k = self.RESPONSE_PRIOR
        p = max(p, 1e-3)
        return ((hits + k * p) / (n + k)) / p

    def _next_response(self, ctx, bid):
        """(P(Dudo), P(Calza)) of the next player on our `bid`, before correction."""
        actor = ctx["order"][0]
        n = ctx["dice"][actor]
        p = p_face(bid.face, ctx["pal"])
        a, b = self._dudo_model(actor)
        cr = self.profiler.calza_rate(actor)
        pd_tot = pc_tot = 0.0
        for k, pk in enumerate(ctx["post"][actor][bid.face]):
            view = binom_tail(ctx["total"] - n, p, bid.quantity - k)
            pd = 1 / (1 + math.exp(-(a + b * view)))
            pd_tot += pk * pd
            pc_tot += pk * (1 - pd) * cr
        return pd_tot, pc_tot

    def _our_options(self, ctx, bid, bidder, pos, depth, laps):
        """Our best move at a later turn of ours, each raise followed by the chain."""
        p_true = self._p_true(ctx, bid)
        best = -p_true + (1 - p_true) * self._lose_value(ctx, bidder)           # Dudo
        if ctx["gain_ok"] and ctx["state"].my_dice_count < 5:
            p_ex = self._p_exact(ctx, bid)
            best = max(best, p_ex * self.CALZA_GAIN - (1 - p_ex))               # Calza
        faces = [ctx["pal_face"] or bid.face] if ctx["pal"] else [2, 3, 4, 5, 6, PACO]
        for f in faces:
            q = min_quantity(bid, f, ctx["pal"])
            if q is None:
                continue
            for qq in range(q, min(q + self.CAND_INNER, ctx["total"] + 1)):
                raise_ = Bid(qq, f)
                if self._p_true(ctx, raise_) >= self.CAND_MIN:
                    best = max(best, self._chain(ctx, raise_, self.name, pos + 1, depth + 1, laps - 1))
        return best

    def _chain(self, ctx, bid, bidder, pos, depth, laps=None):
        """Expected value for us of the state (bid, order[pos] to act)."""
        if laps is None:
            laps = self.LAPS_HEADS_UP if len(ctx["dice"]) == 2 else self.LAPS
        key = (bid, bidder, pos, depth, laps)
        if key in ctx["memo"]:
            return ctx["memo"][key]
        order = ctx["order"]
        actor = order[pos % len(order)]
        if actor == self.name:
            if laps > 1 and depth > 0:
                value = self._our_options(ctx, bid, bidder, pos, depth, laps)
            else:
                value = self._our_turn_value(ctx, bid, bidder)
            ctx["memo"][key] = value
            return value
        if depth >= len(order) * max(self.LAPS, self.LAPS_HEADS_UP) + 1:
            ctx["memo"][key] = 0.0
            return 0.0

        n = ctx["dice"][actor]
        face, q = bid.face, bid.quantity
        p = p_face(face, ctx["pal"])
        post = ctx["post"][actor][face]
        loo = ctx["loo"][(actor, face)]
        mine = ctx["mine"][face]
        a, b = self._dudo_model(actor)
        cr = self.profiler.calza_rate(actor)
        nxt = self._their_raise(ctx, actor, bid)
        w_actor = ctx["w"]
        gain = 1.0 if (ctx["gain_ok"] and n < 5) else 0.0

        shift = self._own_shift(ctx, bid) if bidder == self.name else 0.0
        rd = rc = 1.0
        if bidder == self.name and actor == order[0]:
            pd0, pc0 = self._next_response(ctx, bid)
            rd = self._response_ratio(ctx, "dudo", pd0)
            rc = self._response_ratio(ctx, "calza", pc0)
        value = 0.0
        p_raise = 0.0
        for k, pk in enumerate(post):
            if pk == 0:
                continue
            need = q - mine - k
            p_true = tail(loo, need)
            if shift and 0 < p_true < 1:
                p_true = 1 / (1 + math.exp(-(math.log(p_true / (1 - p_true)) + shift)))
            p_exact = at(loo, need)
            their_view = binom_tail(ctx["total"] - n, p, q - k)
            pd = 1 / (1 + math.exp(-(a + b * their_view)))
            if nxt is None:
                pd = 1.0 - cr
            pc = (1 - pd) * cr if nxt is not None else cr
            if rd != 1.0 or rc != 1.0:
                pd = min(pd * rd, 1.0)
                pc = min(pc * rc, 1.0 - pd)
            pr = max(0.0, 1 - pd - pc)
            dudo = p_true * w_actor + (1 - p_true) * self._lose_value(ctx, bidder)
            calza = p_exact * (-w_actor * gain) + (1 - p_exact) * w_actor
            value += pk * (pd * dudo + pc * calza)
            p_raise += pk * pr
        if nxt is not None and p_raise > 0:
            value += p_raise * self._chain(ctx, nxt, actor, pos + 1, depth + 1, laps)
        ctx["memo"][key] = value
        return value

    # ------------------------------------------------------------------
    # decision
    # ------------------------------------------------------------------

    def _candidates(self, ctx, state):
        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        by_face = defaultdict(list)
        for b in bids:
            by_face[b.face].append(b)
        out = []
        for face, bs in by_face.items():
            bs.sort(key=lambda b: b.quantity)
            likely = [b for b in bs if self._p_true(ctx, b) >= self.CAND_MIN]
            out.extend(likely[-self.CAND_PER_FACE:] if state.current_bid is None
                       else likely[:self.CAND_PER_FACE])
        if not out and bids:
            out = [max(bids, key=lambda b: self._p_true(ctx, b))]
        return out

    def play(self, state):
        ctx = self._setup(state)
        cur = state.current_bid
        options = []
        if cur is not None:
            bidder = last_bidder(state, default=None)
            p_true = self._calibrated(state, cur, self._p_true(ctx, cur), exact=False)
            if Dudo() in state.legal_actions:
                options.append((p_true * -1.0 + (1 - p_true) * self._lose_value(ctx, bidder), Dudo()))
            if Calza() in state.legal_actions:
                gain = self.CALZA_GAIN if (ctx["gain_ok"] and state.my_dice_count < 5) else 0.0
                p_ex = self._calibrated(state, cur, self._p_exact(ctx, cur), exact=True)
                options.append((p_ex * gain - (1 - p_ex), Calza()))
        risk_keys = {}
        for b in self._candidates(ctx, state):
            value, key = self._calibrated_bid_value(ctx, cur is None, self._chain(ctx, b, self.name, 0, 0))
            risk_keys[b] = key
            options.append((value, b))

        best = max(v for v, _ in options)
        value, action = self._rng.choice([o for o in options if o[0] >= best - self.TIE_EPS])
        if isinstance(action, Bid):
            self._risk_pending.append(risk_keys[action])
        if isinstance(action, Bid):
            p = min(max(self._p_true(ctx, action), 1e-4), 1 - 1e-4)
            self._own_pending.append((self._own_key(ctx, p), action, state.palifico))
            pd, pc = self._next_response(ctx, action)
            self._awaiting = (min(len(ctx["dice"]), 4), pd, pc)
        if isinstance(action, Dudo):
            self.mouth.say("dudo", announce="Dudo!", bidder=last_bidder(state))
        elif isinstance(action, Calza):
            self.mouth.say("calza", announce="Calza!")
        else:
            self.mouth.say("open" if cur is None else "raise", announce=fmt_bid(action), bid=fmt_bid(action))
        return action
