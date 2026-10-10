"""
Fabulous -- a probability-driven Perudo bot.

Strategy in one breath: count my own dice, model every unknown die as a
coin with the right odds (1/3 for a normal face thanks to wild Pacos, 1/6
for Pacos or during Palifico), nudge that model with what opponents have
been bidding on (and how honest they have historically been about it),
and then pick Dudo / Calza / a bid by comparing probabilities.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from perudo.models import (
    PACO,
    Action,
    Bid,
    BidEvent,
    Calza,
    Dudo,
    Event,
    GameEndedEvent,
    GameResult,
    GameStartInfo,
    Observation,
    PlayerEliminatedEvent,
    RevealEvent,
    RoundEndedEvent,
)

MAX_DICE = 5

# ---------------------------------------------------------------------------
# Tunables (found by running run_simulation.py a lot)
# ---------------------------------------------------------------------------

DUDO_THRESHOLD = 0.55        # call Dudo when P(current bid is true) drops below this
CALZA_THRESHOLD = 0.32       # call Calza when P(exact) is at least this (and a die can be won)
OPEN_SAFETY = 0.90           # opening bid: highest quantity with P(true) >= this
BID_SLACK = 0.0              # consider bids within this much of the best P(true)
CHALLENGE_FACTOR = 1.1       # how likely a shaky bid is to be challenged (for dudo-vs-bid trade-off)
HINT_PRIOR_HITS = 2.0        # prior for "opponent holds the face they bid on"
HINT_PRIOR_TRIES = 3.0


def _count_matches(dice_values: Sequence[int], face: int, palifico: bool) -> int:
    """How many dice count towards a bid on `face`."""
    if palifico:
        return sum(1 for d in dice_values if d == face)
    if face == PACO:
        return sum(1 for d in dice_values if d == PACO)
    return sum(1 for d in dice_values if d == face or d == PACO)


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------


class Fabulous:
    name = "Fabulous"
    count = 1

    def __init__(self):
        # RNG for decisions (fixed seed -> reproducible simulations).
        self._rng = random.Random(0xFAB)

        # Learned across games: how often each opponent actually held the
        # face they bid on.  name -> [hits, tries]
        self._hint_stats: Dict[str, List[float]] = defaultdict(
            lambda: [HINT_PRIOR_HITS, HINT_PRIOR_TRIES]
        )

        # Per-game / per-round state.
        self._recovery_enabled = True
        self._palifico_now = False
        self._round_bids: List[Tuple[str, Bid]] = []

    # -- lifecycle ---------------------------------------------------------

    def on_game_start(self, info):
        self._recovery_enabled = True
        self._palifico_now = False
        self._round_bids = []

    def on_game_end(self, result):
        pass

    # -- the decision -------------------------------------------------------

    def play(self, state):
        self._palifico_now = state.palifico
        if any(not p.alive for p in state.players):
            self._recovery_enabled = False

        legal_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        others = [p for p in state.players if p.alive and p.id != state.my_id]
        hints = self._collect_hints(state)

        if state.current_bid is None:
            bid = self._opening_bid(state, legal_bids, others, hints)
            return bid

        current = state.current_bid
        p_true = self._prob_at_least(state, others, hints, current.face, current.quantity)
        p_exact = self._prob_exact(state, others, hints, current.face, current.quantity)

        # Calza: only worth it when there is actually a die to win.
        can_gain = self._recovery_enabled and state.my_dice_count < MAX_DICE
        if can_gain and p_exact >= CALZA_THRESHOLD and self._has_calza(state):
            return Calza()

        scored = [
            (self._prob_at_least(state, others, hints, b.face, b.quantity), b)
            for b in legal_bids
        ]
        best_p = max((p for p, _ in scored), default=0.0)

        # Dudo when the bid is unlikely, or when raising would be even riskier
        # than the bid we are doubting.
        bid_risk = (1.0 - best_p) * CHALLENGE_FACTOR
        if p_true < DUDO_THRESHOLD or (not legal_bids) or (p_true < bid_risk):
            return Dudo()

        bid, _ = self._pick_bid(scored, best_p)
        return bid

    # -- bid selection -------------------------------------------------------

    def _opening_bid(self, state, legal_bids, others, hints) -> Bid:
        faces = sorted({b.face for b in legal_bids})
        # Prefer the face we hold most of (Pacos count as wild outside Palifico).
        best_face = max(
            faces,
            key=lambda f: (
                _count_matches(state.my_dice, f, state.palifico),
                0 if f == PACO else 1,  # tie-break: avoid opening on Paco
                self._rng.random(),
            ),
        )
        if not state.palifico and best_face == PACO:
            # Only Pacos in hand: open on a normal face instead, they count anyway.
            best_face = self._rng.choice([f for f in faces if f != PACO])

        candidates = [b for b in legal_bids if b.face == best_face]
        safe = [
            b
            for b in candidates
            if self._prob_at_least(state, others, hints, b.face, b.quantity) >= OPEN_SAFETY
        ]
        if safe:
            return max(safe, key=lambda b: b.quantity)
        return min(candidates, key=lambda b: b.quantity)

    def _pick_bid(self, scored: List[Tuple[float, Bid]], best_p: float) -> Tuple[Bid, float]:
        pool = [(p, b) for p, b in scored if p >= best_p - BID_SLACK]
        # Slightly prefer bids that raise the quantity (they age better) and
        # that keep the face we are strong on; weights stay probability-led.
        weights = [max(p, 1e-6) ** 4 for p, _ in pool]
        choice = self._rng.choices(pool, weights=weights, k=1)[0]
        return choice[1], choice[0]

    def _has_calza(self, state: Observation) -> bool:
        return any(isinstance(a, Calza) for a in state.legal_actions)

    # -- probability model ---------------------------------------------------

    def _collect_hints(self, state: Observation) -> Dict[str, Dict[int, int]]:
        """name -> {face: number of bids this round on that face} for opponents."""
        me = state.players[state.my_id].name
        hints: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for record in state.history:
            if record.player == me or not isinstance(record.action, Bid):
                continue
            hints[record.player][record.action.face] += 1
        return hints

    def _unknown_pmf(self, state, others, hints, face: int) -> List[float]:
        """Probability mass function of how many unknown dice count for `face`."""
        palifico = state.palifico
        p_match = 1.0 / 6.0 if (palifico or face == PACO) else 1.0 / 3.0

        pmf = [1.0]
        for player in others:
            n = player.dice_count
            # How many of this player's dice do we suspect are "pinned" to the face?
            player_hints = hints.get(player.name, {})
            mentions = player_hints.get(face, 0)
            if not palifico and face != PACO:
                mentions += player_hints.get(PACO, 0)  # their Pacos count for us too
            trust = self._trust(player.name)
            pinned_probs = []
            if mentions >= 1 and n >= 1:
                pinned_probs.append(trust)
            if mentions >= 2 and n >= 2:
                pinned_probs.append(trust * 0.5)

            player_pmf = self._player_pmf(n, p_match, pinned_probs)
            pmf = self._convolve(pmf, player_pmf)
        return pmf

    @staticmethod
    def _binomial_pmf(n: int, p: float) -> List[float]:
        pmf = [0.0] * (n + 1)
        q = 1.0 - p
        coef = 1.0
        for k in range(n + 1):
            if k > 0:
                coef = coef * (n - k + 1) / k
            pmf[k] = coef * (p ** k) * (q ** (n - k))
        return pmf

    def _player_pmf(self, n: int, p: float, pinned_probs: List[float]) -> List[float]:
        """Mixture: each 'pinned' die is a match with prob `trust`, else a coin."""
        result = [0.0] * (n + 1)
        # Enumerate which pinned dice actually are matches.
        combos = [([], 1.0)]
        for prob in pinned_probs:
            combos = [
                (fixed + [True], w * prob) for fixed, w in combos
            ] + [(fixed + [False], w * (1 - prob)) for fixed, w in combos]
        for fixed, weight in combos:
            k_fixed = sum(1 for f in fixed if f)
            free = n - len(fixed)
            base = self._binomial_pmf(free, p)
            for k, mass in enumerate(base):
                result[k + k_fixed] += weight * mass
        return result

    @staticmethod
    def _convolve(a: List[float], b: List[float]) -> List[float]:
        out = [0.0] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            if x == 0.0:
                continue
            for j, y in enumerate(b):
                out[i + j] += x * y
        return out

    def _prob_at_least(self, state, others, hints, face: int, quantity: int) -> float:
        mine = _count_matches(state.my_dice, face, state.palifico)
        need = quantity - mine
        if need <= 0:
            return 1.0
        pmf = self._unknown_pmf(state, others, hints, face)
        return sum(pmf[need:]) if need < len(pmf) else 0.0

    def _prob_exact(self, state, others, hints, face: int, quantity: int) -> float:
        mine = _count_matches(state.my_dice, face, state.palifico)
        need = quantity - mine
        if need < 0:
            return 0.0
        pmf = self._unknown_pmf(state, others, hints, face)
        return pmf[need] if need < len(pmf) else 0.0

    def _trust(self, player_name: str) -> float:
        hits, tries = self._hint_stats[player_name]
        return hits / tries

    # -- watching the table ---------------------------------------------------

    def observe(self, event):
        if isinstance(event, BidEvent):
            self._round_bids.append((event.player, event.bid))

        elif isinstance(event, RevealEvent):
            self._learn_from_reveal(event)

        elif isinstance(event, PlayerEliminatedEvent):
            self._recovery_enabled = False

        elif isinstance(event, RoundEndedEvent):
            self._round_bids = []
            self._palifico_now = event.palifico_next

        elif isinstance(event, GameEndedEvent):
            self._round_bids = []

    def _learn_from_reveal(self, event: RevealEvent) -> None:
        dice = event.dice_by_player()
        for player, bid in self._round_bids:
            if player == self.name or player not in dice:
                continue
            held = _count_matches(dice[player], bid.face, self._palifico_now) > 0
            stats = self._hint_stats[player]
            stats[0] += 1.0 if held else 0.0
            stats[1] += 1.0
