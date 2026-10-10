"""
Trihard -- Fabulous' sweaty cousin. Same probability engine, but every
decision is an expected-value calculation that looks one step ahead.

What it fixes compared to Fabulous (found by replaying thousands of games
and categorising every die Fabulous lost):

* Never calls Dudo on a bid that is almost certainly true just because
  every raise looks ugly -- a raise can only cost a die if it gets
  challenged AND is false, a Dudo on a true bid costs one for sure.
* Rates its own bids by "probability of being true GIVEN that the next
  player challenges".  Opponents challenge when their own dice do not
  support the face, so Trihard simulates the next player's dice, guesses
  whether each outcome makes them call, and weighs the outcomes.
* Treats Calza as an expected-value choice, including the "free pass"
  value of a correct Calza once dice recovery is switched off.
* Learns, per opponent, how trigger-happy they are: after every reveal it
  replays the round from that opponent's point of view (their dice are
  public by then) and nudges a per-opponent "calls Dudo below this
  probability" threshold towards what they actually did.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

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
    PublicPlayer,
    RevealEvent,
    RoundEndedEvent,
)

MAX_DICE = 5

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

OPP_CHALLENGE_THRESHOLD = 0.5   # starting guess: opponents call Dudo when (from their view) P(true) is below this
OPP_CHALLENGE_SOFTNESS = 0.10   # how fuzzy that threshold is (logistic scale)
CALL_LEARN_RATE = 0.08          # how fast the per-opponent threshold adapts to observed calls/raises
CONTINUE_VALUE = 0.3            # value (in dice) of a bid that simply passes the turn along
GAIN_VALUE = 2.5                # value of winning a die with Calza (recovery on)
FREE_PASS_VALUE = 0.3           # value of a correct Calza when recovery is off (nothing happens, I start)
BID_EV_TOLERANCE = 0.0          # bids within this EV of the best are tied; prefer the higher quantity
HONESTY_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)  # candidate honesty exponents beta, see _player_pmf_for
HONESTY_DEFAULT = 1.0           # beta for a player we have not learned about yet (also for me, in their eyes)
HONESTY_FLOOR = 0.05            # even a hopeless bid gets made sometimes (bluffs, forced raises)
PACO_MENTION_WEIGHT = 0.5       # a Paco bid is weaker evidence about a normal face (Pacos still count for it)


def _count_matches(dice_values: Sequence[int], face: int, palifico: bool) -> int:
    """How many dice count towards a bid on `face`."""
    if palifico:
        return sum(1 for d in dice_values if d == face)
    if face == PACO:
        return sum(1 for d in dice_values if d == PACO)
    return sum(1 for d in dice_values if d == face or d == PACO)


def _sigmoid(x: float) -> float:
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------


class Trihard:
    name = "Trihard"

    def __init__(self) -> None:
        self._rng = random.Random(0x7A1)

        # name -> accumulated log-likelihood of each honesty exponent in HONESTY_GRID.
        # Model: P(player bids (q, f) | holds k of f) is proportional to
        # (floor + P(bid true from their view | k)) ** beta.  beta=0 means their
        # bids carry no information, big beta means they only bid what they believe.
        self._honesty_ll: Dict[str, List[float]] = defaultdict(lambda: [0.0] * len(HONESTY_GRID))

        # name -> estimated P(true) below which that opponent calls Dudo.
        self._call_theta: Dict[str, float] = defaultdict(lambda: OPP_CHALLENGE_THRESHOLD)

        self._recovery_enabled = True
        self._palifico_now = False
        self._round_bids: List[Tuple[str, Bid]] = []

    # -- lifecycle ---------------------------------------------------------

    def on_game_start(self, info: GameStartInfo) -> None:
        self._recovery_enabled = True
        self._palifico_now = False
        self._round_bids = []

    def on_game_end(self, result: GameResult) -> None:
        pass

    # -- the decision -------------------------------------------------------

    def play(self, state: Observation) -> Action:
        self._palifico_now = state.palifico
        if any(not p.alive for p in state.players):
            self._recovery_enabled = False

        legal_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        others = [p for p in state.players if p.alive and p.id != state.my_id]
        hints = self._collect_hints(state)
        next_player = self._next_player(state)

        best_bid, best_bid_ev, _ = self._best_bid(state, others, hints, next_player, legal_bids)

        if state.current_bid is None:
            return best_bid

        current = state.current_bid
        p_true = self._prob_at_least(state, others, hints, current.face, current.quantity)
        p_exact = self._prob_exact(state, others, hints, current.face, current.quantity)

        ev_dudo = (1.0 - p_true) - p_true
        if self._recovery_enabled and state.my_dice_count < MAX_DICE:
            gain = GAIN_VALUE
        else:
            gain = FREE_PASS_VALUE
        ev_calza = p_exact * gain - (1.0 - p_exact)
        if not any(isinstance(a, Calza) for a in state.legal_actions):
            ev_calza = -math.inf

        options = [("bid", best_bid_ev), ("dudo", ev_dudo), ("calza", ev_calza)]
        choice = max(options, key=lambda o: o[1])[0]

        if choice == "dudo":
            return Dudo()
        if choice == "calza":
            return Calza()
        return best_bid

    # -- bid evaluation ----------------------------------------------------------

    def _best_bid(
        self,
        state: Observation,
        others: List[PublicPlayer],
        hints: Dict[str, Dict[int, int]],
        next_player: PublicPlayer,
        legal_bids: List[Bid],
    ) -> Tuple[Optional[Bid], float, float]:
        """Return (bid, expected value in dice, P(bid true)) for the best legal bid.

        With no legal bid at all (quantity already at the table maximum) the
        bid is None and its value minus infinity."""
        palifico = state.palifico
        rest = [p for p in others if p.id != next_player.id]
        total_dice = sum(p.dice_count for p in state.players if p.alive)

        # Per-face cache: the dice distributions do not depend on the quantity.
        cache: Dict[int, Tuple[List[float], List[float], List[float], int, float]] = {}
        me_as_seen = PublicPlayer(id=state.my_id, name="__me__", dice_count=state.my_dice_count, alive=True)

        def face_model(face: int):
            if face in cache:
                return cache[face]
            p_match = self._p_match(face, palifico)
            pmf_next = self._player_pmf_for(next_player, face, hints, palifico, p_match, total_dice)
            pmf_rest = [1.0]
            for player in rest:
                pmf_rest = self._convolve(
                    pmf_rest, self._player_pmf_for(player, face, hints, palifico, p_match, total_dice)
                )
            mine = _count_matches(state.my_dice, face, palifico)
            cache[face] = (pmf_next, self._suffix(pmf_rest), pmf_rest, mine, p_match)
            return cache[face]

        def their_view(face: int, quantity: int, pmf_rest: List[float], p_match: float) -> List[float]:
            """P(unknown >= k) as the next player sees it after I bid (quantity, face):
            everyone but them is unknown, and my bid is a hint about my dice."""
            my_pmf_as_seen = self._player_pmf_for(
                me_as_seen, face, {"__me__": {face: quantity}}, palifico, p_match, total_dice,
                honesty=HONESTY_DEFAULT,
            )
            return self._suffix(self._convolve(pmf_rest, my_pmf_as_seen))

        theta = self._call_theta[next_player.name]
        scored: List[Tuple[float, float, Bid]] = []
        for bid in legal_bids:
            pmf_next, rest_at_least, pmf_rest, mine, p_match = face_model(bid.face)
            q = bid.quantity
            their_at_least = their_view(bid.face, q, pmf_rest, p_match)
            ev = 0.0
            p_true_total = 0.0
            for m, pm in enumerate(pmf_next):
                if pm == 0.0:
                    continue
                p_true_m = self._at_least(rest_at_least, q - mine - m)
                p_true_total += pm * p_true_m
                their_estimate = self._at_least(their_at_least, q - m)
                c = _sigmoid((theta - their_estimate) / OPP_CHALLENGE_SOFTNESS)
                ev += pm * (c * (2.0 * p_true_m - 1.0) + (1.0 - c) * CONTINUE_VALUE)
            scored.append((ev, p_true_total, bid))

        if not scored:
            return None, -math.inf, 0.0
        best_ev = max(ev for ev, _, _ in scored)
        tied = [s for s in scored if s[0] >= best_ev - BID_EV_TOLERANCE]
        # Among near-equal bids prefer the higher quantity (uses the room),
        # then the one more likely to be true.
        ev, p, bid = max(tied, key=lambda s: (s[2].quantity, s[1], self._rng.random()))
        return bid, ev, p

    # -- probability model ---------------------------------------------------

    @staticmethod
    def _p_match(face: int, palifico: bool) -> float:
        return 1.0 / 6.0 if (palifico or face == PACO) else 1.0 / 3.0

    def _collect_hints(self, state: Observation) -> Dict[str, Dict[int, int]]:
        """name -> {face: highest quantity that player bid on it this round}."""
        me = state.players[state.my_id].name
        hints: Dict[str, Dict[int, int]] = defaultdict(dict)
        for record in state.history:
            if record.player == me or not isinstance(record.action, Bid):
                continue
            bid = record.action
            hints[record.player][bid.face] = max(hints[record.player].get(bid.face, 0), bid.quantity)
        return hints

    def _player_pmf_for(
        self,
        player: PublicPlayer,
        face: int,
        hints: Dict[str, Dict[int, int]],
        palifico: bool,
        p_match: float,
        total_dice: int,
        honesty: Optional[float] = None,
    ) -> List[float]:
        """Distribution of how many of `player`'s dice count for `face`.

        Bayesian update of the binomial prior with the player's own bids:
        the chance they bid (q, f) holding k of f is taken proportional to
        (floor + P(bid true from their view | k)) ** beta, where beta is the
        player's learned honesty exponent."""
        n = player.dice_count
        base = self._binomial_pmf(n, p_match)
        player_hints = hints.get(player.name)
        if not player_hints:
            return base
        beta = self._honesty(player.name) if honesty is None else honesty
        if beta == 0.0:
            return base
        weights = [1.0] * (n + 1)
        q_face = player_hints.get(face)
        if q_face:
            lik = self._bid_likelihood(n, q_face, p_match, total_dice)
            weights = [w * (l ** beta) for w, l in zip(weights, lik)]
        if not palifico and face != PACO:
            q_paco = player_hints.get(PACO)
            if q_paco:
                lik = self._bid_likelihood(n, q_paco, 1.0 / 6.0, total_dice)
                weights = [w * (l ** (beta * PACO_MENTION_WEIGHT)) for w, l in zip(weights, lik)]
        tilted = [mass * w for mass, w in zip(base, weights)]
        total = sum(tilted)
        if total <= 0.0:
            return base
        return [x / total for x in tilted]

    def _bid_likelihood(self, n: int, quantity: int, p_match: float, total_dice: int) -> List[float]:
        """For k = 0..n: floor + P(bid of `quantity` is true | bidder holds k), from the bidder's view."""
        unknown = max(total_dice - n, 0)
        at_least = self._suffix(self._binomial_pmf(unknown, p_match))
        return [HONESTY_FLOOR + self._at_least(at_least, quantity - k) for k in range(n + 1)]

    def _honesty(self, player_name: str) -> float:
        ll = self._honesty_ll.get(player_name)
        if ll is None:
            return HONESTY_DEFAULT
        best = max(range(len(ll)), key=lambda i: ll[i])
        return HONESTY_GRID[best]

    def _unknown_pmf(self, state, others, hints, face: int) -> List[float]:
        p_match = self._p_match(face, state.palifico)
        total_dice = sum(p.dice_count for p in state.players if p.alive)
        pmf = [1.0]
        for player in others:
            pmf = self._convolve(
                pmf, self._player_pmf_for(player, face, hints, state.palifico, p_match, total_dice)
            )
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

    @staticmethod
    def _convolve(a: List[float], b: List[float]) -> List[float]:
        out = [0.0] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            if x == 0.0:
                continue
            for j, y in enumerate(b):
                out[i + j] += x * y
        return out

    @staticmethod
    def _suffix(pmf: List[float]) -> List[float]:
        """suffix[k] = P(X >= k), with suffix[len] = 0."""
        out = [0.0] * (len(pmf) + 1)
        acc = 0.0
        for k in range(len(pmf) - 1, -1, -1):
            acc += pmf[k]
            out[k] = acc
        return out

    @staticmethod
    def _at_least(suffix: List[float], need: int) -> float:
        if need <= 0:
            return 1.0
        if need >= len(suffix):
            return 0.0
        return suffix[need]

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

    # -- watching the table ---------------------------------------------------

    def observe(self, event: Event) -> None:
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
        palifico = self._palifico_now
        total_dice = sum(len(d) for d in dice.values())
        highest: Dict[Tuple[str, int], int] = {}
        for player, bid in self._round_bids:
            if player == self.name or player not in dice:
                continue
            key = (player, bid.face)
            highest[key] = max(highest.get(key, 0), bid.quantity)
        for (player, face), quantity in highest.items():
            n = len(dice[player])
            k = _count_matches(dice[player], face, palifico)
            p_match = self._p_match(face, palifico)
            base = self._binomial_pmf(n, p_match)
            lik = self._bid_likelihood(n, quantity, p_match, total_dice)
            ll = self._honesty_ll[player]
            for i, beta in enumerate(HONESTY_GRID):
                tilted = [mass * (l ** beta) for mass, l in zip(base, lik)]
                ll[i] += math.log(max(tilted[k] / sum(tilted), 1e-12))
        self._learn_call_thresholds(event, dice)

    def _learn_call_thresholds(self, event: RevealEvent, dice: Dict[str, Tuple[int, ...]]) -> None:
        """Replay every 'faced a bid' moment of the round from that player's view."""
        palifico = self._palifico_now
        decisions: List[Tuple[str, Bid, int, bool]] = []  # (player, bid faced, bids before, called?)
        for i in range(1, len(self._round_bids)):
            player = self._round_bids[i][0]
            decisions.append((player, self._round_bids[i - 1][1], i - 1, False))
        if self._round_bids:
            decisions.append(
                (event.challenger, event.current_bid, len(self._round_bids), event.challenge_type == "dudo")
            )
        for player, faced, n_before, called in decisions:
            if player == self.name or player not in dice:
                continue
            p_match = self._p_match(faced.face, palifico)
            own = _count_matches(dice[player], faced.face, palifico)
            # Hints visible to them at that moment: everyone else's earlier bids.
            hints: Dict[str, Dict[int, int]] = defaultdict(dict)
            for bidder, bid in self._round_bids[:n_before]:
                if bidder != player:
                    hints[bidder][bid.face] = max(hints[bidder].get(bid.face, 0), bid.quantity)
            total_dice = sum(len(d) for d in dice.values())
            pmf = [1.0]
            for other, other_dice in dice.items():
                if other == player:
                    continue
                stand_in = PublicPlayer(id=-1, name=other, dice_count=len(other_dice), alive=True)
                pmf = self._convolve(
                    pmf, self._player_pmf_for(stand_in, faced.face, hints, palifico, p_match, total_dice)
                )
            p_est = self._at_least(self._suffix(pmf), faced.quantity - own)
            theta = self._call_theta[player]
            predicted = _sigmoid((theta - p_est) / OPP_CHALLENGE_SOFTNESS)
            theta += CALL_LEARN_RATE * ((1.0 if called else 0.0) - predicted)
            self._call_theta[player] = min(0.95, max(0.05, theta))

    # -- helpers ------------------------------------------------------------------

    @staticmethod
    def _next_player(state: Observation) -> PublicPlayer:
        n = len(state.players)
        i = (state.my_id + 1) % n
        while not state.players[i].alive:
            i = (i + 1) % n
        return state.players[i]

