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

Set TALK_TO_ME=true to hear it over the radio.
"""

import random

from perudo.bots.sniper import QUOTES as SNIPER_QUOTES
from perudo.bots.sniper import SniperBot
from perudo.models import PACO, Bid, Dudo
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
    # to a bid that opens or switches to a face.
    FOLLOW_WEIGHT = 0.3

    def __init__(self):
        super().__init__()
        self.name = "Pokerface"
        self.mouth = Mouth(self, QUOTES)
        self._rng = random.Random()

    # -- reading the others ------------------------------------------------

    def _hint(self, state, face):
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

    def _best_on_face(self, state, bids, face, n_unknown, hints):
        """Strongest legal bid on `face` that clears BLUFF_TARGET, or None."""
        cands = [
            (self._prob_true(state, b, n_unknown, hints), b)
            for b in bids
            if b.face == face
        ]
        cands = [(pt, b) for pt, b in cands if pt >= self.BLUFF_TARGET]
        if not cands:
            return None
        return max(cands, key=lambda s: (self._strength(s[1], state.palifico), s[0]))

    def _bluff(self, state, bids, n_unknown, hints):
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
            found = self._best_on_face(state, bids, face, n_unknown, hints)
            if found is not None:
                return found, key
        return None, None

    # -- decision ---------------------------------------------------------

    def play(self, state):
        total = sum(p.dice_count for p in state.players if p.alive)
        n_unknown = total - state.my_dice_count
        hints = {face: self._hint(state, face) for face in range(1, 7)}

        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        scored = [(self._prob_true(state, b, n_unknown, hints), b) for b in bids]

        safe = [(pt, b) for pt, b in scored if pt >= self.TARGET]
        if safe:
            best_p, best = max(safe, key=lambda s: (self._strength(s[1], state.palifico), s[0]))
        elif scored:
            best_p, best = max(scored, key=lambda s: s[0])
        else:
            best_p, best = 0.0, None

        cur = state.current_bid
        if cur is not None:
            p_cur = self._prob_true(state, cur, n_unknown, hints)
            if best is None or best_p < self.MIN_ACCEPT or p_cur < 1 - best_p:
                key = "dudo_confident" if p_cur < 0.3 else "dudo_cornered"
                self.mouth.say(key, announce="Dudo!", bidder=last_bidder(state))
                return Dudo()

        # We are going to bid. Decide whether to lie about the face.
        bluff, key = self._bluff(state, bids, n_unknown, hints)
        if bluff is not None:
            _, best = bluff
        elif cur is None:
            key = "open"
        else:
            key = "raise" if best_p >= self.TARGET else "raise_risky"
        self.mouth.say(key, announce=fmt_bid(best), bid=fmt_bid(best))
        return best
