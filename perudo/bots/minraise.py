"""
MinRaiseBot: always makes the smallest legal raise (or opening bid), and
never calls Dudo or Calza -- it just keeps nudging the bid up by the
minimum amount possible and leaves challenging to everyone else.

"Smallest" is measured by claim strength, not raw quantity: a Paco bid
only needs ceil(q/2) to be as strong a claim as a normal-face bid of
quantity q (that's exactly why the halving rule exists), so for this
comparison a Paco bid's quantity counts double. Ties (e.g. a normal raise
and a Paco switch that are equally strong) go to the normal bid.

Example progression from 4 fives: 4 fives -> 4 sixes -> 2 Paco -> 5 two.
"""

from perudo.models import PACO, Bid


class MinRaiseBot:
    name = "MinRaise Bot"
    count = 0

    def play(self, state):
        possible_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        if possible_bids:
            return self._minraise(possible_bids)

        # Extremely rare edge case: in a Palifico round the quantity can
        # reach the total number of dice in play, at which point no
        # further bid is legal at all (the face can never change) --
        # only Dudo/Calza remain. With no raise left to make, this is the
        # one situation where MinRaiseBot is forced to pick one anyway.
        return state.legal_actions[0]

    @staticmethod
    def _claim_strength(bid):
        return bid.quantity * 2 if bid.face == PACO else bid.quantity

    @classmethod
    def _minraise(cls, possible_bids):
        return min(
            possible_bids, key=lambda b: (cls._claim_strength(b), b.face == PACO, b.face)
        )
