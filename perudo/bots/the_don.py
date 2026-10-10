"""
The Don: Trihard's engine with the manners of a mob boss.

The Don only cares about the two players next to him: the one on his left
(whose bids he faces) and the one on his right (who faces his bids).

  * A neighbour who is still strong gets hit. Toward the right neighbour
    The Don favours bids that are probably true but that the neighbour is
    likely to challenge, so they lose a die. Toward the left neighbour he is
    quicker to call Dudo on their bid.
  * A neighbour who is down to at least 2 dice fewer than The Don (1 fewer
    once The Don has 2 dice or less) is under his protection: a bodyguard.
    The right one gets an easy bid (very likely true, unlikely to be
    challenged, low enough to raise on comfortably), so they have no reason
    to Dudo him; the left one is only challenged when it clearly pays.

Everything else - the probability model, what it learns about every
opponent's honesty and Dudo threshold - is Trihard's.

Set TALK_TO_ME=true to hear him.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from perudo.bots.trihard import (
    BID_EV_TOLERANCE,
    CONTINUE_VALUE,
    FREE_PASS_VALUE,
    GAIN_VALUE,
    HONESTY_DEFAULT,
    MAX_DICE,
    OPP_CHALLENGE_SOFTNESS,
    Trihard,
    _fmt_bid,
    _sigmoid,
)
from perudo.models import Action, Bid, Calza, Dudo, Observation, PublicPlayer
from perudo.rules import count_matches
from perudo.talk import Mouth

QUOTES = {
    "open": [
        "{bid}. Ik doe je een aanbod dat je niet kunt weigeren.",
        "{bid}. Rustig. Dit is zaken, niets persoonlijks.",
    ],
    "attack": [
        "{bid}. Jij zit naast me. Dat is jammer voor je.",
        "{bid}. De familie vergeet niets.",
        "{bid}. Ga je gang, roep maar Dudo.",
    ],
    "protect": [
        "{bid}. Voor jou maak ik het makkelijk, vriend.",
        "{bid}. Jij hoort nu bij de familie.",
    ],
    "bid": ["{bid}."],
    "dudo_attack": [
        "Dudo. {bidder}, je had beter moeten weten.",
        "Dudo. Een boodschap van de familie, {bidder}.",
    ],
    "dudo": ["Dudo. Het spijt me, {bidder}. Echt."],
    "calza": ["Calza. Precies wat ik verwachtte."],
    "dudo_right": ["{bidder} betaalt. Zoals iedereen betaalt."],
    "dudo_wrong": ["{actual}. Iemand heeft gepraat."],
    "eliminated_other": ["{who} slaapt bij de vissen."],
    "won": ["De familie wint. De familie wint altijd."],
}


class TheDonBot(Trihard):
    name = "The Don"

    # Switch the two halves of the policy on or off.
    ATTACK = True
    PROTECT = True
    # Attack through Dudo on a strong neighbour's bid (besides trap bids).
    ATTACK_DUDO = True
    # Value of a strong neighbour losing a die, on top of the normal +1.
    ATTACK_BONUS = 0.6
    # Toward a protected right neighbour: penalty per unit of chance that
    # they challenge our bid (we want them comfortable, not provoked).
    PROTECT_WEIGHT = 1.0
    # A protected bid must be at least this likely to be true.
    PROTECT_MIN_TRUE = 0.75
    # Penalty on a Dudo against a protected left neighbour.
    SPARE_PENALTY = 0.5

    def __init__(self) -> None:
        super().__init__()
        self.name = "The Don"
        self.mouth = Mouth(self, QUOTES)

    # -- who is who --------------------------------------------------------

    @staticmethod
    def _neighbour(state: Observation, step: int) -> PublicPlayer:
        n = len(state.players)
        i = (state.my_id + step) % n
        while not state.players[i].alive:
            i = (i + step) % n
        return state.players[i]

    @staticmethod
    def _protected(state: Observation, player: PublicPlayer) -> bool:
        """At least 2 dice fewer than us (1 fewer once we have 2 or less)."""
        mine = state.my_dice_count
        gap = 2 if mine >= 3 else 1
        return player.dice_count <= mine - gap

    # -- the decision -------------------------------------------------------

    def play(self, state: Observation) -> Action:
        self._palifico_now = state.palifico
        if any(not p.alive for p in state.players):
            self._recovery_enabled = False

        legal_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        others = [p for p in state.players if p.alive and p.id != state.my_id]
        hints = self._collect_hints(state)
        right = self._next_player(state)
        left = self._neighbour(state, -1)
        protect_right = self.PROTECT and self._protected(state, right)

        best_bid, best_bid_ev, best_bid_p = self._best_bid_don(
            state, others, hints, right, legal_bids, protect_right)
        mood = "protect" if protect_right else "attack"

        if state.current_bid is None:
            self.mouth.say("open", announce=_fmt_bid(best_bid), bid=_fmt_bid(best_bid))
            return best_bid

        current = state.current_bid
        bidder = self._last_bidder(state)
        p_true = self._prob_at_least(state, others, hints, current.face, current.quantity)
        p_exact = self._prob_exact(state, others, hints, current.face, current.quantity)

        ev_dudo = (1.0 - p_true) - p_true
        if bidder in (left.name, right.name):
            neighbour = left if bidder == left.name else right
            if self._protected(state, neighbour):
                if self.PROTECT:
                    ev_dudo -= self.SPARE_PENALTY
            elif self.ATTACK and self.ATTACK_DUDO:
                ev_dudo += self.ATTACK_BONUS * (1.0 - p_true)
        gain = GAIN_VALUE if (self._recovery_enabled and state.my_dice_count < MAX_DICE) else FREE_PASS_VALUE
        ev_calza = p_exact * gain - (1.0 - p_exact)
        if not any(isinstance(a, Calza) for a in state.legal_actions):
            ev_calza = -math.inf

        choice = max([("bid", best_bid_ev), ("dudo", ev_dudo), ("calza", ev_calza)], key=lambda o: o[1])[0]
        if choice == "dudo":
            attack = bidder in (left.name, right.name)
            self.mouth.say("dudo_attack" if attack else "dudo", announce="Dudo!", bidder=bidder)
            return Dudo()
        if choice == "calza":
            self.mouth.say("calza", announce="Calza!")
            return Calza()
        self.mouth.say(mood, announce=_fmt_bid(best_bid), bid=_fmt_bid(best_bid))
        return best_bid

    # -- bid evaluation ----------------------------------------------------

    def _best_bid_don(
        self,
        state: Observation,
        others: List[PublicPlayer],
        hints: Dict[str, Dict[int, int]],
        next_player: PublicPlayer,
        legal_bids: List[Bid],
        protect: bool,
    ) -> Tuple[Optional[Bid], float, float]:
        """Trihard's bid evaluation, with the neighbour policy on top."""
        palifico = state.palifico
        rest = [p for p in others if p.id != next_player.id]
        total_dice = sum(p.dice_count for p in state.players if p.alive)
        cache: Dict[int, tuple] = {}
        me_as_seen = PublicPlayer(id=state.my_id, name="__me__", dice_count=state.my_dice_count, alive=True)

        def face_model(face: int):
            if face not in cache:
                p_match = self._p_match(face, palifico)
                pmf_next = self._player_pmf_for(next_player, face, hints, palifico, p_match, total_dice)
                pmf_rest = [1.0]
                for player in rest:
                    pmf_rest = self._convolve(
                        pmf_rest, self._player_pmf_for(player, face, hints, palifico, p_match, total_dice))
                mine = count_matches(state.my_dice, face, palifico)
                cache[face] = (pmf_next, self._suffix(pmf_rest), pmf_rest, mine, p_match)
            return cache[face]

        theta = self._call_theta[next_player.name]
        scored = []
        for bid in legal_bids:
            pmf_next, rest_at_least, pmf_rest, mine, p_match = face_model(bid.face)
            q = bid.quantity
            my_pmf_as_seen = self._player_pmf_for(
                me_as_seen, bid.face, {"__me__": {bid.face: q}}, palifico, p_match, total_dice,
                honesty=HONESTY_DEFAULT)
            their_at_least = self._suffix(self._convolve(pmf_rest, my_pmf_as_seen))
            ev = p_true_total = challenge = 0.0
            for m, pm in enumerate(pmf_next):
                if pm == 0.0:
                    continue
                p_true_m = self._at_least(rest_at_least, q - mine - m)
                p_true_total += pm * p_true_m
                c = _sigmoid((theta - self._at_least(their_at_least, q - m)) / OPP_CHALLENGE_SOFTNESS)
                challenge += pm * c
                value = c * (2.0 * p_true_m - 1.0) + (1.0 - c) * CONTINUE_VALUE
                if not protect and self.ATTACK:
                    # They challenge a true bid: our neighbour loses a die.
                    value += self.ATTACK_BONUS * c * p_true_m
                ev += pm * value
            if protect:
                ev -= self.PROTECT_WEIGHT * challenge
            scored.append((ev, p_true_total, bid))

        if not scored:
            return None, -math.inf, 0.0
        if protect:
            easy = [s for s in scored if s[1] >= self.PROTECT_MIN_TRUE]
            if easy:
                # The easiest bid to live with: likely, unprovoking, low.
                best_ev = max(ev for ev, _, _ in easy)
                tied = [s for s in easy if s[0] >= best_ev - 0.05]
                ev, p, bid = min(tied, key=lambda s: (s[2].quantity, -s[1]))
                return bid, ev, p
        best_ev = max(ev for ev, _, _ in scored)
        tied = [s for s in scored if s[0] >= best_ev - BID_EV_TOLERANCE]
        ev, p, bid = max(tied, key=lambda s: (s[2].quantity, s[1], self._rng.random()))
        return bid, ev, p
