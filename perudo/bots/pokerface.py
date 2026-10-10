"""
Pokerface: plays like Sniper, but bluffs about which faces it holds.

Most bots at this table read the bids of their opponents: a player who bids
on a face probably holds some of it. Pokerface uses that against them. When
it is its turn to bid it sometimes picks a face it does not hold:

  * Opening bluff: open the round on a face we have none of. The others
    credit us with dice we don't have and overestimate that face, so they
    raise into it or skip a Dudo they should have called.
  * Precedence bluff: when opponents have already piled onto a face we
    don't hold, join in and bid on it too. That looks like confirmation, so
    the table overestimates the face even further.

A bluff is never reckless: the bid itself must still be likely enough to be
true (BLUFF_TARGET), using the same probability model as Sniper. We only lie
about *where* the dice are, not about how many there are. Bluffing is off in
Palifico rounds, where the face is fixed after the opener anyway.

Reading the others: Sniper credits an opponent with extra dice for every bid
they make on a face. In a chain (5xP, 6xP, 7xP ...) every player is just
raising the previous bid, so that adds up to far more dice than are there
and Sniper keeps raising where it should call Dudo. Pokerface counts a bid
that only raises the quantity of the face just bid at FOLLOW_WEIGHT; a bid
that opens or switches to a face is a real choice and counts in full.

Pokerface learns from every opponent by name (perudo.profiler), without
assuming anything about who they are. After each reveal it replays their
choices and learns:

  * what their bids are worth: how many extra dice of a face they really
    held when they opened on it, switched to it or followed it. Their bids
    are read with those numbers, so an honest bidder is trusted and a
    bluffer is ignored or even read in reverse;
  * how often they call Dudo on a false bid. With ADAPT_TARGET our next bid
    gets bolder against a player who rarely does and more careful against a
    sharp one (off by default: it did not pay off in benchmarks).

A new opponent starts as the average of everyone seen so far, and its own
data takes over as it comes in. `self.profiler.report()` shows what it
learned. ADAPTIVE = False switches the learning off (fixed Sniper reading
with FOLLOW_WEIGHT).

Set TALK_TO_ME=true to hear it over the radio.
"""

import math
import random

from perudo.bots.sniper import QUOTES as SNIPER_QUOTES
from perudo.bots.sniper import SniperBot
from perudo.models import PACO, Bid, Calza, Dudo, RevealEvent
from perudo.rules import count_matches
from perudo.profiler import OpponentProfiler, bid_kind
from perudo.talk import Mouth, fmt_bid, last_bidder

QUOTES = {
    **SNIPER_QUOTES,
    "open": [
        "{bid}. Zeg maar niks.",
        "{bid}. Kijk me aan. Zie je iets? Nee.",
        "{bid}.",
    ],
    "raise": [
        "{bid}. Geen spier vertrokken.",
        "{bid}. Lees me maar. Succes ermee.",
        "{bid}.",
    ],
    "bluff_open": [
        "{bid}. Mijn favoriete getal. Echt waar.",
        "{bid}. Ik heb er een stel, geloof me.",
        "{bid}. Rustig openen.",
    ],
    "bluff_follow": [
        "{bid}. Ik zit er ook vol mee, toevallig.",
        "{bid}. Daar zijn er veel van. Ik kan het weten.",
        "{bid}. Iedereen heeft ze vandaag, hè?",
    ],
    "calza": [
        "Calza. Precies goed. Ik heb het je gezicht zien zeggen.",
        "Calza. Niet meer, niet minder.",
    ],
    "bid_held": [
        "{challenger}, je had moeten kijken naar mijn ogen. Daar stond niks in.",
        "{actual}. Pokerface, {challenger}. Pokerface.",
    ],
    "bid_caught": [
        "{actual}. Oké, die zag je. Volgende keer niet.",
        "Betrapt. Ik blijf er even heel neutraal bij kijken.",
    ],
    "won": [
        "Niemand heeft me kunnen lezen. Niemand.",
    ],
}


class PokerfaceBot(SniperBot):
    name = "Pokerface"

    # Chance to bluff when a bluff is available. Below 1 keeps us
    # unpredictable: an opponent can't assume every bid of ours is a lie.
    BLUFF_RATE = 0.5
    # Minimum P(true) for a bluff bid. A bit below Sniper's TARGET: the
    # misdirection is worth some of the risk.
    BLUFF_TARGET = 0.75
    # Number of different opponents that bid a face this round before we
    # call it "strong precedence" and join in.
    PRECEDENCE = 2
    # Weight of a bid that stays on the face of the bid before it, relative
    # to a bid that opens or switches to a face. Only used while not ADAPTIVE.
    FOLLOW_WEIGHT = 0.3
    # Learn from the opponents at the table instead of fixed assumptions:
    #  * how much each opponent's bids say about their hand (bid_excess),
    #  * how likely the next player is to call Dudo on a false bid, which
    #    moves our TARGET up or down.
    ADAPTIVE = True
    # Moving the bid target with the next player's challenge rate. Off: in
    # benchmark_adaptive.py it was worth nothing (-0.2 +/- 0.7 pp win rate,
    # 120 tables), while learned bid reading alone gave +2.4 +/- 0.9 pp.
    ADAPT_TARGET = False
    # Bounds for the adapted TARGET.
    TARGET_RANGE = (0.6, 0.97)
    # Learn how often the bids we face are really true, per estimate bucket
    # and table size, and decide Dudo on that instead of the raw estimate.
    CALIBRATE = True
    # Weight of the raw estimate as a prior, in observations.
    CALIBRATION_PRIOR = 20
    # Call Calza ourselves when a correct one gains a die, P(exact) is at
    # least CALZA_AT and it beats Dudo in expected dice.
    CALZA = True
    CALZA_AT = 0.5
    # Avoid bids that the next player is likely to Calza correctly: skip a
    # bid when P(exact) * their learned Calza rate exceeds CALZA_RISK.
    AVOID_CALZA = True
    CALZA_RISK = 0.1
    # Pick bids by a one-step lookahead instead of "strongest bid that is
    # TARGET likely": for every hand the next player may hold, how likely is
    # our bid (to us) and how likely does it look to them? With their learned
    # Dudo threshold and Calza rate that gives the expected dice of each bid.
    # Bids that are true but look false to them (traps) score best; bids
    # that hand them an easy Dudo or Calza score worst.
    LOOKAHEAD = False
    # Only bids at least this likely (to us) are considered.
    LOOKAHEAD_MIN = 0.5
    # How sharp an opponent's Dudo threshold is (logistic width).
    DUDO_SOFTNESS = 0.07
    # Dudo threshold assumed for an opponent we know nothing about yet.
    DEFAULT_DUDO_AT = 0.3
    # Set to a list to record every decision with what we believed at the
    # time (see analyze_endgame.py). None = no logging.
    decision_log = None

    def __init__(self):
        super().__init__()
        self.name = "Pokerface"
        self.mouth = Mouth(self, QUOTES)
        self._rng = random.Random()
        # Learns per opponent name; lives as long as this instance, so
        # across all games of a simulation.
        self.profiler = OpponentProfiler(self.name)
        # (table size class, estimate bucket) -> [times true, times seen]
        self._calibration = {}
        self._pending = []   # (key, bid, palifico) of bids faced this round
        self._thr_cache = {}

    # -- reading the others ------------------------------------------------

    def _hint(self, state, face):
        """Extra dice of `face` we expect among opponents, based on their bids."""
        if self.ADAPTIVE and not state.palifico:
            return self._learned_hint(state, face)
        return self._static_hint(state, face)

    def _learned_hint(self, state, face):
        """
        Like _static_hint, but every opponent's bid is worth what their bids
        were worth in earlier rounds (OpponentProfiler.bid_excess): a lot for
        an honest bidder, nothing or less than nothing for a bluffer.
        """
        return sum(self._hint_by_player(state, face).values())

    def _hint_by_player(self, state, face):
        """Per opponent: extra dice of `face` we credit them for their bids."""
        bid_faces, chosen = {}, {}
        prev = None
        for rec in state.history:
            if not isinstance(rec.action, Bid):
                continue
            bid = rec.action
            if rec.player != self.name:
                bid_faces.setdefault(rec.player, set()).add(bid.face)
                if prev is None or prev.face != bid.face:
                    chosen.setdefault(rec.player, set()).add(bid.face)
            prev = bid

        dice_of = {p.name: p.dice_count for p in state.players}
        p = self._p_face(face, False)
        out = {}
        for who, faces in bid_faces.items():
            extra = 0.0
            # Natural dice of the face itself, and Pacos (wild for any face).
            for f in {face, PACO} & faces:
                extra += self.profiler.bid_excess(who, bid_kind(f, f in chosen.get(who, ())))
            n = dice_of.get(who, 0)
            # They hold between 0 and n matches: expectation is n * p.
            out[who] = min(max(extra, -n * p), n * (1 - p))
        return out

    def _static_hint(self, state, face):
        """Extra dice of `face` we expect among opponents, based on their bids."""
        # In Palifico only the opener chose the face; later bids are forced.
        history = state.history[:1] if state.palifico else state.history
        dice_of = {p.name: p.dice_count for p in state.players}

        weight_of = {}
        prev = None
        for rec in history:
            if not isinstance(rec.action, Bid):
                continue
            bid = rec.action
            if rec.player != self.name:
                weight = 0.0
                if bid.face == face:
                    weight = 1.0
                elif not state.palifico and bid.face == PACO and face != PACO:
                    weight = self.PACO_HINT_WEIGHT
                if prev is not None and prev.face == bid.face:
                    weight *= self.FOLLOW_WEIGHT
                weight_of[rec.player] = weight_of.get(rec.player, 0.0) + weight
            prev = bid

        total = 0.0
        for who, weight in weight_of.items():
            # Cap: they can't hold more than their dice beyond expectation.
            cap = dice_of.get(who, 0) * (1 - self._p_face(face, state.palifico))
            total += min(weight * self.HINT_PER_BID, cap)
        return total

    # -- bluffing ---------------------------------------------------------

    @staticmethod
    def _missing_faces(state):
        """Normal faces of which we hold no natural die."""
        return [f for f in range(2, 7) if f not in state.my_dice]

    def _best_on_face(self, state, bids, face, n_unknown, hints, target):
        """Strongest legal bid on `face` that clears `target`, or None."""
        cands = [
            (self._prob_true(state, b, n_unknown, hints), b)
            for b in bids
            if b.face == face
        ]
        cands = [(pt, b) for pt, b in cands if pt >= target]
        if not cands:
            return None
        return max(cands, key=lambda s: (self._strength(s[1], state.palifico), s[0]))

    def _bluff(self, state, bids, n_unknown, hints, target):
        """A bluff bid and its quote key, or (None, None)."""
        if state.palifico or self._rng.random() >= self.BLUFF_RATE:
            return None, None
        missing = self._missing_faces(state)
        if not missing:
            return None, None

        if state.current_bid is None:
            faces, key = missing, "bluff_open"
        else:
            backers = {
                f: {r.player for r in state.history
                    if r.player != self.name and isinstance(r.action, Bid) and r.action.face == f}
                for f in missing
            }
            faces = [f for f in missing if len(backers[f]) >= self.PRECEDENCE]
            key = "bluff_follow"
        self._rng.shuffle(faces)
        for face in faces:
            found = self._best_on_face(state, bids, face, n_unknown, hints, target)
            if found is not None:
                return found, key
        return None, None

    # -- decision ---------------------------------------------------------

    @staticmethod
    def _next_player(state):
        """Name of the player who will face our bid."""
        seats = list(state.players)
        me = next(i for i, p in enumerate(seats) if p.id == state.my_id)
        for step in range(1, len(seats)):
            p = seats[(me + step) % len(seats)]
            if p.alive:
                return p.name
        return None

    def _target(self, state, base):
        """
        Minimum P(true) for our bid. A false bid only costs a die when it is
        challenged, so in front of a player who rarely challenges false bids
        we may bid more boldly, and in front of a sharp one more carefully:
        we keep P(false) * P(challenged) where it is for an average player.
        """
        if not (self.ADAPTIVE and self.ADAPT_TARGET):
            return base
        rate = self.profiler.challenge_rate(self._next_player(state))
        avg = self.profiler.challenge_rate()
        lo, hi = self.TARGET_RANGE
        return min(max(1 - (1 - base) * avg / max(rate, 0.05), lo), hi)

    def play(self, state):
        total = sum(p.dice_count for p in state.players if p.alive)
        n_unknown = total - state.my_dice_count
        hints = {face: self._hint(state, face) for face in range(1, 7)}
        target = self._target(state, self.TARGET)

        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        scored = [(self._prob_true(state, b, n_unknown, hints), b) for b in bids]

        safe = [(pt, b) for pt, b in scored if pt >= target]
        if safe and self.AVOID_CALZA:
            rate = self.profiler.calza_rate(self._next_player(state))
            low_risk = [(pt, b) for pt, b in safe
                        if rate * self._prob_exact(state, b, n_unknown, hints) <= self.CALZA_RISK]
            safe = low_risk or safe
        if self.LOOKAHEAD and not state.palifico and scored:
            picked = self._lookahead_pick(state, scored, n_unknown, hints)
            if picked is not None:
                safe = [picked]
        if safe:
            best_p, best = max(safe, key=lambda s: (self._strength(s[1], state.palifico), s[0]))
        elif scored:
            best_p, best = max(scored, key=lambda s: s[0])
        else:
            best_p, best = 0.0, None

        cur = state.current_bid
        p_cur = None
        if cur is not None:
            p_cur = self._calibrated(state, cur, self._prob_true(state, cur, n_unknown, hints))
            if self.CALZA and Calza() in state.legal_actions and self._calza_gains(state):
                p_ex = self._calibrated(state, cur, self._prob_exact(state, cur, n_unknown, hints), exact=True)
                # Expected dice: Calza +1 / -1, Dudo +1 (they lose) / -1.
                if p_ex >= self.CALZA_AT and 2 * p_ex - 1 > 1 - 2 * p_cur:
                    self.mouth.say("calza", announce="Calza!", bidder=last_bidder(state))
                    self._log(state, Calza(), p_cur, best, best_p, hints, False)
                    return Calza()
            if best is None or best_p < self.MIN_ACCEPT or p_cur < 1 - best_p:
                key = "dudo_confident" if p_cur < 0.3 else "dudo_cornered"
                self.mouth.say(key, announce="Dudo!", bidder=last_bidder(state))
                self._log(state, Dudo(), p_cur, best, best_p, hints, False)
                return Dudo()

        # We are going to bid. Decide whether to lie about the face.
        bluff, key = self._bluff(state, bids, n_unknown, hints, self._target(state, self.BLUFF_TARGET))
        if bluff is not None:
            _, best = bluff
        elif cur is None:
            key = "open"
        else:
            key = "raise" if best_p >= target else "raise_risky"
        self.mouth.say(key, announce=fmt_bid(best), bid=fmt_bid(best))
        self._log(state, best, p_cur, best, self._prob_true(state, best, n_unknown, hints), hints,
                  bluff is not None)
        return best

    # -- lookahead ----------------------------------------------------------

    def _dudo_at(self, name):
        """Learned Dudo threshold of `name` on the own-dice scale, cached."""
        prof = self.profiler.profiles.get(name)
        if prof is None or len(prof.facing) < 20:
            return self.DEFAULT_DUDO_AT
        cache = self._thr_cache.get(name)
        if cache is None or len(prof.facing) - cache[0] >= 25:
            t, _ = prof.dudo_threshold()
            cache = (len(prof.facing), t)
            self._thr_cache[name] = cache
        return cache[1]

    def _bid_ev(self, state, bid, n_unknown, hints, nxt, dice_next, h_next, t_next, calza_next, gain_next):
        """Expected dice for us of putting `bid` in front of the next player."""
        p = self._p_face(bid.face, False)
        total = n_unknown + state.my_dice_count
        mine = count_matches(state.my_dice, bid.face, False)
        h_rest = hints[bid.face] - h_next
        m = n_unknown - dice_next          # unknown dice outside the next player
        # Their matches: binomial, nudged by what their own bids told us.
        q_next = min(max(p + (h_next / dice_next if dice_next else 0), 0.01), 0.99)
        ev = 0.0
        for k in range(dice_next + 1):
            pk = math.comb(dice_next, k) * q_next**k * (1 - q_next) ** (dice_next - k)
            need = bid.quantity - mine - k - h_rest
            p_true = self._tail(m, p, need)
            p_exact = p_true - self._tail(m, p, need + 1)
            # How likely the bid looks to them: their k plus chance elsewhere.
            their_view = self._tail(total - dice_next, p, bid.quantity - k)
            p_dudo = 1 / (1 + math.exp((their_view - t_next) / self.DUDO_SOFTNESS))
            p_calza = (1 - p_dudo) * calza_next
            ev += pk * (p_dudo * (2 * p_true - 1)
                        + p_calza * ((1 - p_exact) - p_exact * gain_next))
        return ev

    def _lookahead_pick(self, state, scored, n_unknown, hints):
        """(P(true), bid) with the best expected dice, or None."""
        nxt = self._next_player(state)
        if nxt is None:
            return None
        dice_next = next(p.dice_count for p in state.players if p.name == nxt)
        t_next = self._dudo_at(nxt)
        calza_next = self.profiler.calza_rate(nxt)
        gain_next = 1.0 if (all(p.alive for p in state.players) and dice_next < 5) else 0.0
        by_player = {f: self._hint_by_player(state, f).get(nxt, 0.0) for f in range(1, 7)}
        best = None
        for pt, b in scored:
            if pt < self.LOOKAHEAD_MIN:
                continue
            ev = self._bid_ev(state, b, n_unknown, hints, nxt, dice_next, by_player[b.face],
                              t_next, calza_next, gain_next)
            key = (ev, self._strength(b, False))
            if best is None or key > best[0]:
                best = (key, pt, b)
        return None if best is None else (best[1], best[2])

    def _prob_exact(self, state, bid, n_unknown, hints):
        """P(exactly bid.quantity matching dice on the table)."""
        more = Bid(bid.quantity + 1, bid.face)
        return self._prob_true(state, bid, n_unknown, hints) - self._prob_true(state, more, n_unknown, hints)

    @staticmethod
    def _calza_gains(state):
        """A correct Calza only gives a die while nobody is out and we have < 5."""
        return all(p.alive for p in state.players) and state.my_dice_count < 5

    # -- calibration ------------------------------------------------------

    @staticmethod
    def _size_class(state):
        alive = sum(p.alive for p in state.players)
        return min(alive, 4)

    def _calibrated(self, state, bid, p, exact=False):
        """
        Our estimate for a bid we face, corrected by how often bids we gave
        a similar estimate at a similar table size turned out true. Raw
        estimates in head-to-head play are far too low (opponents' bids say
        more about their hand than the hint model credits), and this learns
        that from the reveals, for any opponent.
        """
        key = (exact, self._size_class(state), min(int(p * 10), 9))
        self._pending.append((key, bid, state.palifico))
        if not self.CALIBRATE:
            return p
        hits, n = self._calibration.get(key, (0, 0))
        k = self.CALIBRATION_PRIOR
        return (hits + k * p) / (n + k)

    def _learn_calibration(self, event):
        dice = [d for _, hand in event.dice for d in hand]
        for key, bid, palifico in self._pending:
            stat = self._calibration.setdefault(key, [0, 0])
            actual = count_matches(dice, bid.face, palifico)
            exact = key[0]
            stat[0] += actual == bid.quantity if exact else actual >= bid.quantity
            stat[1] += 1
        self._pending = []

    def _log(self, state, action, p_cur, best, best_p, hints, bluffed):
        if self.decision_log is None:
            return
        self.decision_log.append(dict(
            round=state.round_number,
            players={p.name: p.dice_count for p in state.players if p.alive},
            my_dice=state.my_dice,
            palifico=state.palifico,
            history=[(r.player, r.action) for r in state.history],
            current=state.current_bid,
            p_current=p_cur,
            action=action,
            best_raise=best,
            p_best_raise=best_p,
            hints=dict(hints),
            bluffed=bluffed,
        ))

    def observe(self, event):
        self.profiler.observe(event)
        if isinstance(event, RevealEvent):
            self._learn_calibration(event)
        super().observe(event)
