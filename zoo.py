"""
A zoo of parametrised opponents with a known strategy, for testing.

None of these live in perudo/bots/, so they never join a normal simulation.
They stand in for the unknown bots we will meet in a competition: sample
random ones with `random_zoo()` and see whether a learning bot adapts.

A ZooBot plays by four knobs:
  faces     how it picks the face of its bid: "honest" (the face it holds
            most), "bluff" (the face it holds least), "random", or
            "mixed" (honest half of the time, random otherwise).
  dudo_at   calls Dudo when P(current bid true) is below this.
  bid_at    bids the highest quantity on its face that is still at least
            this likely; None = the cheapest legal bid on that face.
  reads     if True it reads the table like Sniper: every opponent bid on a
            face counts as 0.7 extra dice of it. Such bots are the ones a
            bluffer can mislead.
  calza_at  calls Calza when P(bid exactly right) is at least this (None =
            never).
"""

from __future__ import annotations

import math
import random

from perudo.models import PACO, Bid, Calza, Dudo
from perudo.rules import count_matches

FACE_STYLES = ("honest", "bluff", "random", "mixed")


def _binom_tail(n, p, need):
    if need <= 0:
        return 1.0
    lo = math.floor(need)
    frac = need - lo

    def tail(k):
        if k <= 0:
            return 1.0
        if k > n:
            return 0.0
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    return (1 - frac) * tail(lo) + frac * tail(lo + 1)


class ZooBot:
    def __init__(self, name, dudo_at=0.5, faces="honest", bid_at=None, reads=False, calza_at=None, seed=None):
        self.name = name
        self.dudo_at = dudo_at
        self.faces = faces
        self.bid_at = bid_at
        self.reads = reads
        self.calza_at = calza_at
        self._rng = random.Random(seed if seed is not None else name)

    def describe(self):
        bid = "cheapest" if self.bid_at is None else f"bid>={self.bid_at:.2f}"
        calza = "" if self.calza_at is None else f" calza>={self.calza_at:.2f}"
        return f"{self.faces:<6} dudo<{self.dudo_at:.2f} {bid:<12} {'reads' if self.reads else 'own-dice'}{calza}"

    # -- probability ------------------------------------------------------

    def _p(self, state, bid, exact=False):
        total = sum(p.dice_count for p in state.players if p.alive)
        n = total - state.my_dice_count
        p = 1 / 6 if (state.palifico or bid.face == PACO) else 1 / 3
        need = bid.quantity - count_matches(state.my_dice, bid.face, state.palifico)
        if self.reads:
            need -= self._hint(state, bid.face)
        if exact:
            k = round(need)
            return math.comb(n, k) * p**k * (1 - p) ** (n - k) if 0 <= k <= n else 0.0
        return _binom_tail(n, p, need)

    def _hint(self, state, face):
        dice_of = {p.name: p.dice_count for p in state.players}
        weight = {}
        for rec in state.history:
            if rec.player != self.name and isinstance(rec.action, Bid) and rec.action.face == face:
                weight[rec.player] = weight.get(rec.player, 0) + 0.7
        return sum(min(w, dice_of.get(who, 0) * 2 / 3) for who, w in weight.items())

    # -- decision ---------------------------------------------------------

    def _pick_face(self, state):
        counts = {f: count_matches(state.my_dice, f, False) for f in range(2, 7)}
        style = self.faces
        if style == "mixed":
            style = "honest" if self._rng.random() < 0.5 else "random"
        if style == "honest":
            best = max(counts.values())
            return self._rng.choice([f for f, c in counts.items() if c == best])
        if style == "bluff":
            worst = min(counts.values())
            return self._rng.choice([f for f, c in counts.items() if c == worst])
        return self._rng.randint(2, 6)

    def play(self, state):
        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        cur = state.current_bid
        if cur is not None:
            if self.calza_at is not None and Calza() in state.legal_actions:
                if self._p(state, cur, exact=True) >= self.calza_at:
                    return Calza()
            if not bids or self._p(state, cur) < self.dudo_at:
                return Dudo()

        face = self._pick_face(state)
        on_face = [b for b in bids if b.face == face] or bids
        if self.bid_at is not None:
            ok = [b for b in on_face if self._p(state, b) >= self.bid_at]
            if ok:
                return max(ok, key=lambda b: b.quantity)
        return min(on_face, key=lambda b: (b.quantity, b.face))


def random_zoo_bot(rng, name):
    """One opponent with random knobs."""
    return ZooBot(
        name,
        dudo_at=rng.uniform(0.1, 0.7),
        faces=rng.choice(FACE_STYLES),
        bid_at=rng.choice([None, rng.uniform(0.3, 0.9)]),
        reads=rng.random() < 0.5,
        calza_at=rng.choice([None, None, rng.uniform(0.25, 0.4)]),
        seed=rng.randrange(2**32),
    )


def random_zoo(rng, count, prefix="Zoo"):
    return [random_zoo_bot(rng, f"{prefix} {i + 1}") for i in range(count)]
