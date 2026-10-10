"""
100 IQ Bot: a simple probability-based bot.

Splits all dice on the table into two buckets: "mine" (known exactly)
and "the rest of the table" (unknown, each die uniformly 1-6). It uses
that to estimate the probability that the current bid is actually true,
then either raises (using the face it personally holds the most of) or
calls Dudo.

Reuses perudo.rules.count_matches -- the same pure counting rule the
engine itself uses to decide bids/Dudo/Calza -- so the wild-Paco and
Palifico counting logic is never duplicated here.
"""

import math

from perudo.models import PACO, Bid, Dudo
from perudo.rules import count_matches


class HundredIQBot:
    def __init__(self):
            self.name = "100 IQ Bot"
            self.count = 3
    def play(self, state):
        possible_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        best_face = self._face_i_hold_most(state.my_dice)

        if state.current_bid is None:
            # Geen vorig bod om op te rekenen -- gewoon openen op het
            # gezicht waar ik zelf de meeste van heb.
            chosen = self._cheapest_bid_for_face(possible_bids, best_face)
            return chosen if chosen is not None else state.legal_actions[0]

        if self._probability_bid_is_true(state) > 0.5:
            chosen = self._cheapest_bid_for_face(possible_bids, best_face)
            if chosen is not None:
                return chosen
            # Geen enkel bod meer legaal (uitgebolde Palifico-ronde) --
            # dan blijft alleen Dudo over.

        return Dudo()

    # -- helpers ----------------------------------------------------

    @staticmethod
    def _face_i_hold_most(my_dice):
        counts = {face: my_dice.count(face) for face in range(1, 7)}
        return max(counts, key=lambda face: counts[face])

    @staticmethod
    def _cheapest_bid_for_face(possible_bids, face):
        # "Het eerst volgende getal" op het gekozen gezicht: de goedkoopste
        # (laagste aantal) legale bid die de engine voor dat gezicht toestaat
        # -- dit dekt normale verhogingen EN de halveer/verdubbel-regels
        # van/naar Paco automatisch, zonder dat de bot die zelf hoeft te
        # kennen.
        same_face = [b for b in possible_bids if b.face == face]
        if same_face:
            return min(same_face, key=lambda b: b.quantity)
        if possible_bids:
            return min(possible_bids, key=lambda b: (b.quantity, b.face))
        return None  # only possible in a maxed-out Palifico round

    @staticmethod
    def _probability_bid_is_true(state):
        bid = state.current_bid
        total_dice_in_play = sum(p.dice_count for p in state.players)
        opponent_dice = total_dice_in_play - state.my_dice_count

        my_matches = count_matches(state.my_dice, bid.face, state.palifico)
        still_needed = max(0, bid.quantity - my_matches)

        if state.palifico or bid.face == PACO:
            p_match = 1 / 6  # no wild: only the exact face counts
        else:
            p_match = 2 / 6  # face itself, or a wild Paco

        return HundredIQBot._probability_at_least(opponent_dice, still_needed, p_match)

    @staticmethod
    def _probability_at_least(dice_count, needed, p):
        if needed <= 0:
            return 1.0
        if needed > dice_count:
            return 0.0
        return sum(
            math.comb(dice_count, k) * (p ** k) * ((1 - p) ** (dice_count - k))
            for k in range(needed, dice_count + 1)
        )
