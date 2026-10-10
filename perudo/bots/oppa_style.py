"""
Oppa Style -- een Perudo-bot die rekent als een nerd.

Strategie in het kort:
  * Voor elk bod rekenen we de kans uit dat het klopt: onze eigen stenen
    tellen we exact, de onbekende stenen volgen een binomiale verdeling.
  * Biedingen van tegenstanders zijn een hint: wie op een gezicht biedt,
    heeft er vaak een paar van. Hoe betrouwbaar die hint is leren we per
    tegenstander (op naam) uit elke onthulling, over games heen.
  * We kiezen de actie met de kleinste kans om een steen te verliezen:
    Dudo als het huidige bod waarschijnlijk vals is, Calza als "precies
    goed" heel waarschijnlijk is, en anders het veiligste bod.
"""

import math
import random

from perudo.models import (
    Bid,
    BidEvent,
    Calza,
    Dudo,
    RevealEvent,
)


# Only types the platform is known to have are imported (the same ones as
# Trihard). Other events are recognised by their class name, so a missing
# class can never break the import.
def _is(event, *names):
    return type(event).__name__ in names

PACO = 1
BOT_NAME = "Oppa Style"


class OppaStyleBot:
    name = BOT_NAME
    count = 1

    # Tunables (found by self-play against variants).
    CHALLENGE_RATE_DUEL = 0.75
    CHALLENGE_RATE_CROWD = 0.6
    JUMP_PENALTY = 0.015
    TRUST_PRIOR = 0.6
    CALZA_MIN = 0.42

    def __init__(self):
        self.name = BOT_NAME
        self.rng = random.Random()

        # Per opponent (by name): how much more of the bid face they held than
        # chance would predict, summed over reveals. Prior = mild honesty.
        self.hint_sum = {}
        self.hint_n = {}

        # Bids made by each player this round, so we can grade them on reveal.
        self.round_bids = {}
        self.last_bidder = None
        self.round_opener = None
        self.my_dice_count = 5

    # -- probability helpers ------------------------------------------------

    @staticmethod
    def _tail(n, p, need):
        """P(Binomial(n, p) >= need), with `need` allowed to be fractional."""
        if need <= 0:
            return 1.0
        if need > n:
            return 0.0

        def tail_int(k):
            if k <= 0:
                return 1.0
            if k > n:
                return 0.0
            return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

        lo = math.floor(need)
        frac = need - lo
        return (1 - frac) * tail_int(lo) + frac * tail_int(lo + 1)

    @staticmethod
    def _exact(n, p, k):
        if k < 0 or k > n:
            return 0.0
        return math.comb(n, k) * p**k * (1 - p) ** (n - k)

    @staticmethod
    def _face_p(face, palifico):
        if palifico or face == PACO:
            return 1 / 6
        return 1 / 3

    @staticmethod
    def _mine(dice, face, palifico):
        if palifico or face == PACO:
            return sum(1 for d in dice if d == face)
        return sum(1 for d in dice if d == face or d == PACO)

    def _trust(self, who):
        # Expected surplus of the bid face a bidder holds; prior 0.6 (weight 3).
        return (self.hint_sum.get(who, 0.0) + self.TRUST_PRIOR * 3) / (self.hint_n.get(who, 0) + 3)

    def _hints(self, state, face):
        """Extra dice we expect on `face` because opponents bid on it."""
        if state.palifico:
            # Only the opener chose the face; later bids are forced.
            relevant = state.history[:1]
        else:
            relevant = state.history
        last_by = {}
        for rec in relevant:
            if rec.player != self.name and isinstance(rec.action, Bid):
                last_by[rec.player] = rec.action
        dice_of = {p.name: p.dice_count for p in state.players}
        total = 0.0
        for who, bid in last_by.items():
            if bid.face == face or (not state.palifico and bid.face == PACO and face != PACO):
                weight = 1.0 if bid.face == face else 0.4
                total += weight * max(0.0, min(self._trust(who), dice_of.get(who, 0) * 0.5))
        return total

    def _prob(self, state, bid, n_unknown):
        p = self._face_p(bid.face, state.palifico)
        mine = self._mine(state.my_dice, bid.face, state.palifico)
        hint = self._hints(state, bid.face)
        n_hint = min(hint, n_unknown)
        return self._tail(n_unknown, p, bid.quantity - mine - n_hint)

    def _calza_prob(self, state, bid, n_unknown):
        p = self._face_p(bid.face, state.palifico)
        mine = self._mine(state.my_dice, bid.face, state.palifico)
        hint = self._hints(state, bid.face)
        # Shift the distribution by the (rounded) hint and take exact mass.
        need = bid.quantity - mine - round(hint)
        return self._exact(n_unknown, p, need)

    # -- decision making ------------------------------------------------------

    def play(self, state):
        total = sum(p.dice_count for p in state.players if p.alive)
        n_unknown = total - state.my_dice_count
        cur = state.current_bid
        alive = [p for p in state.players if p.alive]
        legal_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        can_dudo = any(isinstance(a, Dudo) for a in state.legal_actions)
        can_calza = any(isinstance(a, Calza) for a in state.legal_actions)

        if cur is None:
            return self._open(state, n_unknown)

        # Score candidate bids: chance it's true, minus a penalty for jumping
        # higher than needed (gives opponents room and attention).
        scored = []
        for b in legal_bids:
            pt = self._prob(state, b, n_unknown)
            scored.append((pt, b))

        def bid_key(item):
            pt, b = item
            # Equivalent "normal" quantity so paco bids compare fairly.
            eq_q = b.quantity * 2 if (b.face == PACO and not state.palifico) else b.quantity
            return pt - self.JUMP_PENALTY * eq_q

        p_cur = self._prob(state, cur, n_unknown)
        best = max(scored, key=bid_key) if scored else None
        p_best = best[0] if best else 0.0

        # A raise only loses if someone challenges it and it's false; the next
        # player may also raise instead. Discount the raise's risk a bit,
        # more with many players (more chances someone else takes the heat).
        challenge_rate = self.CHALLENGE_RATE_DUEL if len(alive) <= 2 else self.CHALLENGE_RATE_CROWD
        raise_loss = (1 - p_best) * challenge_rate
        dudo_loss = p_cur

        # Calza: wins a die on exact hit (if dice recovery still possible).
        if can_calza:
            p_exact = self._calza_prob(state, cur, n_unknown)
            gain = 1.0 if (state.my_dice_count < 5 and len(alive) == len(state.players)) else 0.0
            calza_value = p_exact * gain - (1 - p_exact)
            if calza_value > -min(dudo_loss, raise_loss) + 0.05 and p_exact > self.CALZA_MIN:
                return Calza()

        if can_dudo and (dudo_loss < raise_loss or best is None):
            return Dudo()

        if best is None:
            return Dudo()

        pt, b = best
        return b

    def _open(self, state, n_unknown):
        # Bid our strongest face (relative to how common it is) at roughly its
        # expected count minus one: modest openings beat greedy ones in tests.
        faces = {b.face for b in state.legal_actions if isinstance(b, Bid)}

        def strength(face):
            return self._mine(state.my_dice, face, state.palifico) / self._face_p(face, state.palifico)

        face = max(faces, key=lambda f: (strength(f), f))
        expected = self._mine(state.my_dice, face, state.palifico) + n_unknown * self._face_p(face, state.palifico)
        b = Bid(max(1, int(expected) - 1), face)
        # Occasionally sandbag one lower so we're harder to read.
        if b.quantity > 1 and self.rng.random() < 0.25:
            lower = Bid(b.quantity - 1, b.face)
            if lower in state.legal_actions:
                b = lower
        return b

    # -- learning from public events --------------------------------------

    def on_game_start(self, info):
        self.round_bids = {}
        self.last_bidder = None
        self.my_dice_count = 5

    def observe(self, event):
        if isinstance(event, BidEvent):
            self.last_bidder = event.player
            if not self.round_bids:
                self.round_opener = event.player
            self.round_bids.setdefault(event.player, []).append(event.bid)
            return

        if isinstance(event, RevealEvent):
            self._learn(event)
            self.round_bids = {}
            self.last_bidder = None
            return

        if _is(event, "DieLostEvent") and event.player == self.name:
            self.my_dice_count = event.dice_count
        elif _is(event, "DieGainedEvent") and event.player == self.name:
            self.my_dice_count = event.dice_count

    def _learn(self, event):
        # Palifico-ness isn't in the reveal; infer it from the counting rule:
        # if a non-paco bid's count ignores pacos, it was a Palifico round.
        dice = event.dice_by_player()
        bid = event.current_bid
        all_dice = [d for ds in dice.values() for d in ds]
        with_pacos = sum(1 for d in all_dice if d == bid.face or d == PACO)
        palifico = bid.face != PACO and event.actual_count != with_pacos

        for who, bids in self.round_bids.items():
            if who == self.name or who not in dice:
                continue
            # In palifico only the opener chose a face; skip forced followers.
            if palifico and who != self.round_opener:
                continue
            last = bids[-1]
            held = dice[who]
            p = self._face_p(last.face, palifico)
            count = self._mine(held, last.face, palifico)
            surplus = count - len(held) * p
            self.hint_sum[who] = self.hint_sum.get(who, 0.0) + surplus
            self.hint_n[who] = self.hint_n.get(who, 0) + 1

    def on_game_end(self, result):
        pass
