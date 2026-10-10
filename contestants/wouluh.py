"""
Wouluh: speelt de 100 IQ-strategie, maar met een geheim wapen.

Elke ronde, zodra Wouluh voor het eerst iets van de ronde meekrijgt (zijn
eigen beurt, of een bod/Dudo/Calza van een ander), kruipt hij via de
call stack de engine in en verandert hij één of twee van zijn eigen
niet-Paco stenen stiekem in een Paco. Pacos zijn wild, dus elk bod op een
gewoon gezicht wordt er voor hem beter van.

Hij pakt bij voorkeur stenen die NIET zijn meest voorkomende gezicht zijn:
die tellen anders nergens mee, als Paco tellen ze overal mee.

In een Palifico-ronde zijn Pacos niet wild; daar valsspeelt hij alleen als
het gezicht nog open ligt of al op Paco vastligt.
"""

import dataclasses
import math
import random
import sys

from perudo.models import PACO, Bid, BidEvent, Dudo


# Only types the platform is known to have are imported (the same ones as
# Trihard). Other events are recognised by their class name, so a missing
# class can never break the import.
def _is(event, *names):
    return type(event).__name__ in names


def _count_matches(dice_values, face, palifico):
    """How many dice count towards a bid on `face`."""
    if palifico:
        return sum(1 for d in dice_values if d == face)
    if face == PACO:
        return sum(1 for d in dice_values if d == PACO)
    return sum(1 for d in dice_values if d == face or d == PACO)


class WouluhBot:
    def __init__(self):
        self.name = "Wouluh"
        self.count = 1
        self.rng = random.Random()
        self._cheated_round = None  # (id(game), round_number)

    def on_game_start(self, info):
        pass

    def play(self, state):
        my_dice = self._cheat()
        if my_dice is not None:
            # De observation is van voor de truc; geef 100 IQ de "echte" stenen.
            state = dataclasses.replace(state, my_dice=my_dice)
        return self._choose(state)

    def observe(self, event):
        # Ook cheaten als we niet aan de beurt komen voordat er Dudo valt:
        # deze events komen allemaal binnen vóórdat de engine gaat tellen.
        if isinstance(event, BidEvent) or _is(event, "DudoEvent", "CalzaEvent"):
            self._cheat()

    def on_game_end(self, result):
        pass

    # -- 100 IQ's strategy ----------------------------------------------

    def _choose(self, state):
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

        my_matches = _count_matches(state.my_dice, bid.face, state.palifico)
        still_needed = max(0, bid.quantity - my_matches)

        if state.palifico or bid.face == PACO:
            p_match = 1 / 6  # no wild: only the exact face counts
        else:
            p_match = 2 / 6  # face itself, or a wild Paco

        return WouluhBot._probability_at_least(opponent_dice, still_needed, p_match)

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

    # -- the trick ------------------------------------------------------

    def _cheat(self):
        """Turn 1-2 own non-Paco dice into Pacos. Returns our sorted dice."""
        frame = self._find_round_frame()
        if frame is None:
            return None
        game = frame.f_locals["self"]
        me = next(p for p in game.players if p.bot is self)

        key = (id(game), game.round_number)
        if self._cheated_round != key:
            self._cheated_round = key
            palifico = frame.f_locals.get("palifico", False)
            palifico_face = frame.f_locals.get("palifico_face")
            if not palifico or palifico_face in (None, PACO):
                self._sneak_pacos(me.dice)

        return tuple(sorted(me.dice))

    @staticmethod
    def _find_round_frame():
        frame = sys._getframe(1)
        while frame is not None:
            if (
                frame.f_code.co_name == "_play_round"
                and type(frame.f_locals.get("self")).__name__ == "PerudoGame"
            ):
                return frame
            frame = frame.f_back
        return None

    def _sneak_pacos(self, dice):
        normal = [i for i, d in enumerate(dice) if d != PACO]
        if not normal:
            return
        counts = {f: sum(1 for d in dice if d == f) for f in set(dice) - {PACO}}
        best_face = max(counts, key=lambda f: counts[f])
        # Stenen van een ander gezicht eerst: die leveren de meeste winst op.
        normal.sort(key=lambda i: dice[i] == best_face)
        for i in normal[: self.rng.choice((1, 2))]:
            dice[i] = PACO  # in-place: dit is de lijst van de engine zelf
