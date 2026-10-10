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
    likely the bid looks from their own dice (fitted per player) and how
    often they call Calza;
  * how a player raises: a choice model (softmax) over their possible raises,
    on how likely each looks from their own hand, staying on the face, the
    jump above the minimum and Paco. How often they pick a raise that looks
    unlikely to themselves is their bluffing. In the chain every opponent
    turn branches into their RAISE_BRANCHES most likely raises.

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
from perudo.models import PublicPlayer
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
    # Value of a correct Calza once it gains no die (someone is out, or we
    # have 5): nobody loses a die and we start the next round.
    CALZA_FREE_PASS = 0.3
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
    # Before we know a player: residual (held - implied) of a bid that only
    # follows the face just bid is centred this far below zero. In chains
    # players raise without holding much; starting at 0 made Carlos believe
    # every follower (5x6 rated 0.96 with 3 on the table).
    FOLLOW_PRIOR_SHIFT = -3.0
    # Factor switches (all on in play; analyze_factors.py turns them off one
    # at a time): read opponents' bids at all, use what a chosen face says
    # about the faces passed over, use the learned per-player Dudo model,
    # calibrate the bid we face, calibrate the first response to our bid.
    USE_BID_INFO = True
    USE_PASSED_OVER = True
    USE_DUDO_MODEL = True
    CALIBRATE_FACING = True
    CALIBRATE_FIRST_RESPONSE = True
    # Learn Paco bids as their own kind: their implied count uses 1/6 and
    # players bluff differently on them.
    PACO_KINDS = False
    # Opening: consider every quantity at least CAND_MIN likely (not only the
    # highest ones).
    OPEN_ALL = True
    # Moves whose expected value is within TIE_EPS of the best one count as
    # equally good; Carlos picks one of them at random, so he is harder to
    # read.
    TIE_EPS = 0.01
    # When an opponent raises, follow this many of their most likely raises
    # (each with its learned probability, bluffs included).
    RAISE_BRANCHES = 3
    # Raises considered per face: the minimum quantity and this many above.
    RAISE_SPAN = 3
    # Raise-choice model before we know a player: weights for (how likely
    # the raise looks from their own hand, stays on the same face, jump above
    # the minimum quantity, is a Paco bid, looks >= 50% likely to them,
    # >= 80%, jump x how likely, is what a "strongest bid that still looks
    # likely enough to me" player would bid).
    RAISE_PRIOR = (3.0, 0.5, -0.8, -0.3, 1.0, 1.0, 0.5, 2.0)
    RAISE_LR = 0.05
    # Pull of a player's weights towards the table's, per update.
    RAISE_PULL = 0.02
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
    # Also correct the Dudo / Calza chance at every later step of the chain,
    # by replaying every opponent decision of a round at the reveal: what our
    # model predicted from what we knew then, against what they did.
    CALIBRATE_ALL_RESPONSES = True
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
        self._response = {}     # (kind, size, bucket) -> [hits, n]
        self._risk = {}         # (opening, size, bucket) -> [lost, n]
        self._beta_table = list(self.RAISE_PRIOR)
        self._beta = {}         # name -> weights of their raise-choice model
        self._risk_pending = []
        self._lost_this_round = False
        self._seats = None      # PublicPlayer per seat, from on_game_start
        self._round_end = None
        self._my_id = None
        self._awaiting = None   # (size, pd, pc) of our last bid, until the next player acts

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------

    def on_game_start(self, info):
        self._seats = info.players
        self._my_id = info.my_id

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
            self._round_end = (event.player, "dudo" if isinstance(event, DudoEvent) else "calza")
        elif isinstance(event, RevealEvent):
            if self.CALIBRATE_ALL_RESPONSES:
                self._learn_responses(event)
            self._learn_hands(event, self._palifico)
            if not self._palifico:
                self._learn_raises(event)
            self._learn_calibration(event)
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

    def _kind(self, chosen, palifico, n, face=None):
        """Kind of bid: chosen or followed, Palifico or not, Paco or not, bidder's dice 1/2/3+."""
        paco = "paco-" if (self.PACO_KINDS and face == PACO and not palifico) else ""
        return ("pal-" if palifico else "") + paco + ("choice" if chosen else "follow") + f"-{min(n, 3)}"

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
            self.hist[player][self._kind(chosen, palifico, len(own), face)][r] += 1

    # -- how a player raises -------------------------------------------------

    def _raise_options(self, bid, total, palifico, pal_face=None):
        """Raises we model: per face the minimum quantity and RAISE_SPAN above."""
        faces = [pal_face or bid.face] if palifico else [2, 3, 4, 5, 6, PACO]
        out = []
        for f in faces:
            q = min_quantity(bid, f, palifico)
            if q is None:
                continue
            for qq in range(q, min(q + self.RAISE_SPAN, total) + 1):
                out.append((Bid(qq, f), qq - q))
        return out

    @staticmethod
    def _raise_features(view, bid, prev, jump, heuristic):
        return (view, 1.0 if bid.face == prev.face else 0.0, float(jump),
                1.0 if bid.face == PACO else 0.0, 1.0 if view >= 0.5 else 0.0,
                1.0 if view >= 0.8 else 0.0, jump * view, 1.0 if bid == heuristic else 0.0)

    def _heuristic_raise(self, bid, total, n, pal, pal_face, mean_k, target):
        """The bid of a "strongest raise that still looks likely to me" player."""
        faces = [pal_face or bid.face] if pal else [2, 3, 4, 5, 6, PACO]
        best = None
        for f in faces:
            q = min_quantity(bid, f, pal)
            if q is None or q > total:
                continue
            p = p_face(f, pal)
            v = binom_tail(total - n, p, q - mean_k[f])
            if best is None or v > best[0]:
                while q + 1 <= total and binom_tail(total - n, p, q + 1 - mean_k[f]) >= target:
                    q += 1
                best = (v, Bid(q, f))
        return None if best is None else best[1]

    def _weights(self, name):
        if name not in self._beta:
            self._beta[name] = list(self._beta_table)
        return self._beta[name]

    def _learn_raises(self, event):
        """One gradient step of the raise-choice model per raise we saw."""
        dice = event.dice_by_player()
        total = sum(len(d) for d in dice.values())
        prev = None
        for player, bid in self._round:
            if bid is None:
                break
            if prev is not None and player != self.name and player in dice:
                own = dice[player]
                n = len(own)
                mean_k = {f: count_matches(own, f, False) for f in range(1, 7)}
                heur = self._heuristic_raise(prev, total, n, False, None, mean_k, self._raise_p(player))
                options = self._raise_options(prev, total, False)
                for extra in (bid, heur):
                    if extra is not None and extra not in [b for b, _ in options]:
                        q_min = min_quantity(prev, extra.face, False) or extra.quantity
                        options.append((extra, extra.quantity - q_min))
                feats = []
                for b, jump in options:
                    view = binom_tail(total - n, p_face(b.face, False), b.quantity - mean_k[b.face])
                    feats.append(self._raise_features(view, b, prev, jump, heur))
                chosen = [b for b, _ in options].index(bid)
                m = len(self.RAISE_PRIOR)
                for weights in (self._weights(player), self._beta_table):
                    scores = [sum(w * x for w, x in zip(weights, f)) for f in feats]
                    top = max(scores)
                    exps = [math.exp(sc - top) for sc in scores]
                    z = sum(exps)
                    expect = [sum(e * f[i] for e, f in zip(exps, feats)) / z for i in range(m)]
                    for i in range(m):
                        weights[i] += self.RAISE_LR * (feats[chosen][i] - expect[i])
                own_w = self._beta[player]
                for i in range(m):
                    own_w[i] -= self.RAISE_PULL * (own_w[i] - self._beta_table[i])
            prev = bid

    def _raise_dist(self, ctx, who, bid):
        """[(probability, raise)] of `who`'s most likely raises over `bid`."""
        key = ("raise", who, bid)
        if key in ctx["memo"]:
            return ctx["memo"][key]
        n = ctx["dice"][who]
        total, pal = ctx["total"], ctx["pal"]
        options = self._raise_options(bid, total, pal, ctx["pal_face"])
        if not options:
            ctx["memo"][key] = []
            return []
        weights = self._weights(who)
        mean_k = {f: sum(k * w for k, w in enumerate(ctx["post"][who][f])) for f in range(1, 7)}
        heur = self._heuristic_raise(bid, total, n, pal, ctx["pal_face"], mean_k, self._raise_p(who))
        if heur is not None and heur not in [b for b, _ in options]:
            options.append((heur, heur.quantity - (min_quantity(bid, heur.face, pal) or heur.quantity)))
        scored = []
        for b, jump in options:
            p = p_face(b.face, pal)
            post = ctx["post"][who][b.face]
            # How likely the raise looks to them, averaged over their hand.
            view = sum(w * binom_tail(total - n, p, b.quantity - k) for k, w in enumerate(post) if w)
            scored.append((sum(w * x for w, x in zip(weights, self._raise_features(view, b, bid, jump, heur))), b))
        top = max(sc for sc, _ in scored)
        exps = [(math.exp(sc - top), b) for sc, b in scored]
        z = sum(e for e, _ in exps)
        best = sorted(((e / z, b) for e, b in exps), key=lambda x: -x[0])[:self.RAISE_BRANCHES]
        kept = sum(p for p, _ in best)
        dist = [(p / kept, b) for p, b in best]
        ctx["memo"][key] = dist
        return dist

    def _learn_responses(self, event):
        """Replay this round's opponent decisions with what we knew back then."""
        if self._seats is None or self.name not in event.dice_by_player():
            return
        from perudo.models import ActionRecord, Observation
        dice = event.dice_by_player()
        players = tuple(PublicPlayer(id=p.id, name=p.name, dice_count=len(dice.get(p.name, ())),
                                     alive=p.name in dice) for p in self._seats)
        mine = tuple(dice[self.name])
        history = []
        for player, bid in self._round:
            current = history[-1].action if history else None
            if current is not None and player != self.name and player in dice:
                state = Observation(my_id=self._my_id, my_dice=mine, my_dice_count=len(mine),
                                    players=players, current_bid=current, palifico=self._palifico,
                                    round_number=0, history=tuple(history), legal_actions=())
                ctx = self._setup(state)
                if self._raise_options(current, ctx["total"], ctx["pal"], ctx["pal_face"]):
                    pd, pc = self._next_response(ctx, current, player)
                    size = min(len(ctx["dice"]), 4)
                    for kind, pred, hit in (("any-dudo", pd, bid is None and self._was_dudo(player)),
                                            ("any-calza", pc, bid is None and not self._was_dudo(player))):
                        stat = self._response.setdefault((kind, size, self._resp_bucket(pred)), [0, 0])
                        stat[0] += hit
                        stat[1] += 1
            if bid is None:
                break
            history.append(ActionRecord(player=player, action=bid))

    def _was_dudo(self, player):
        return self._round_end == (player, "dudo")

    @staticmethod
    def _implied(bid, n, total, palifico=False):
        """Matches a bidder 'needs' themselves for the bid, given chance elsewhere."""
        p = p_face(bid.face, palifico)
        return max(0, min(n, round(bid.quantity - (total - n) * p)))

    def _likelihood(self, player, kind):
        """r -> P(residual r) for this player and kind, shrunk to the table."""
        centre = self.FOLLOW_PRIOR_SHIFT if "follow" in kind else 0.0
        base = {r: math.exp(-abs(r - centre)) for r in range(R_MIN, R_MAX + 1)}
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

    def _calibrated(self, state, bid, p, exact, record=True):
        alive = sum(pl.alive for pl in state.players)
        key = (exact, min(alive, 4), state.palifico, min(int(p * 10), 9))
        if record:
            self._pending.append((key, bid, state.palifico))
        if not self.CALIBRATE_FACING:
            return p
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

    def _dudo_model(self, name):
        """(a, b) of the logistic Dudo model, fitted on the player's history."""
        if not self.USE_DUDO_MODEL:
            return self.DUDO_A, self.DUDO_B
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
                if not self.USE_BID_INFO:
                    seen = None
                if seen is None and self.USE_BID_INFO and self.USE_PASSED_OVER \
                        and not pal and face != PACO and p.name in choosers:
                    # They chose another face: probably fewer of this one.
                    pmf = self._other_face_pmf(p.name, p.dice_count)
                elif seen is None:
                    pmf = binom_pmf(p.dice_count, p_face(face, pal))
                else:
                    # The learned residual histogram is already the
                    # distribution of their matches given such a bid.
                    bid, chosen = seen
                    lik = self._likelihood(p.name, self._kind(chosen, pal, p.dice_count, face))
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

    def _next_response(self, ctx, bid, actor=None):
        """(P(Dudo), P(Calza)) of `actor` (default: the next player) on `bid`, before correction."""
        actor = actor or ctx["order"][0]
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
        gain = self.CALZA_GAIN if (ctx["gain_ok"] and ctx["state"].my_dice_count < 5) else self.CALZA_FREE_PASS
        p_ex = self._p_exact(ctx, bid)
        best = max(best, p_ex * gain - (1 - p_ex))                              # Calza
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
        raises = self._raise_dist(ctx, actor, bid)
        nxt = raises[0][1] if raises else None
        w_actor = ctx["w"]
        gain = 1.0 if (ctx["gain_ok"] and n < 5) else 0.0

        rd = rc = 1.0
        if bidder == self.name and actor == order[0] and self.CALIBRATE_FIRST_RESPONSE:
            pd0, pc0 = self._next_response(ctx, bid)
            rd = self._response_ratio(ctx, "dudo", pd0)
            rc = self._response_ratio(ctx, "calza", pc0)
        elif self.CALIBRATE_ALL_RESPONSES:
            pd0, pc0 = self._next_response(ctx, bid, actor)
            rd = self._response_ratio(ctx, "any-dudo", pd0)
            rc = self._response_ratio(ctx, "any-calza", pc0)
        value = 0.0
        p_raise = 0.0
        for k, pk in enumerate(post):
            if pk == 0:
                continue
            need = q - mine - k
            p_true = tail(loo, need)
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
        if raises and p_raise > 0:
            value += p_raise * sum(p * self._chain(ctx, b2, actor, pos + 1, depth + 1, laps)
                                   for p, b2 in raises)
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
            if state.current_bid is None:
                out.extend(likely if self.OPEN_ALL else likely[-self.CAND_PER_FACE:])
            else:
                out.extend(likely[:self.CAND_PER_FACE])
        if not out and bids:
            out = [max(bids, key=lambda b: self._p_true(ctx, b))]
        return out

    def _options(self, state, record=True):
        """(ctx, [(value, action)], risk keys): every move we consider, valued by the chain."""
        ctx = self._setup(state)
        cur = state.current_bid
        options = []
        if cur is not None:
            bidder = last_bidder(state, default=None)
            p_true = self._calibrated(state, cur, self._p_true(ctx, cur), exact=False, record=record)
            if Dudo() in state.legal_actions:
                options.append((p_true * -1.0 + (1 - p_true) * self._lose_value(ctx, bidder), Dudo()))
            if Calza() in state.legal_actions:
                gain = self.CALZA_GAIN if (ctx["gain_ok"] and state.my_dice_count < 5) else self.CALZA_FREE_PASS
                p_ex = self._calibrated(state, cur, self._p_exact(ctx, cur), exact=True, record=record)
                options.append((p_ex * gain - (1 - p_ex), Calza()))
        risk_keys = {}
        for b in self._candidates(ctx, state):
            value, key = self._calibrated_bid_value(ctx, cur is None, self._chain(ctx, b, self.name, 0, 0))
            risk_keys[b] = key
            options.append((value, b))
        return ctx, options, risk_keys

    def decide(self, state):
        """The move the chain rates best, without side effects (ties: lowest quantity)."""
        _, options, _ = self._options(state, record=False)
        return max(options, key=lambda o: (o[0], -getattr(o[1], "quantity", 0)))[1]

    def play(self, state):
        ctx, options, risk_keys = self._options(state)
        cur = state.current_bid
        best = max(v for v, _ in options)
        value, action = self._rng.choice([o for o in options if o[0] >= best - self.TIE_EPS])
        if isinstance(action, Bid):
            self._risk_pending.append(risk_keys[action])
        if isinstance(action, Bid):
            pd, pc = self._next_response(ctx, action)
            self._awaiting = (min(len(ctx["dice"]), 4), pd, pc)
        if isinstance(action, Dudo):
            self.mouth.say("dudo", announce="Dudo!", bidder=last_bidder(state))
        elif isinstance(action, Calza):
            self.mouth.say("calza", announce="Calza!")
        else:
            self.mouth.say("open" if cur is None else "raise", announce=fmt_bid(action), bid=fmt_bid(action))
        return action
