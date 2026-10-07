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

Zet TALK_TO_ME=true om hem (heel onschuldig) te horen praten.
"""

import dataclasses
import importlib
import random
import sys

from perudo.game import PerudoGame
from perudo.models import PACO, BidEvent, CalzaEvent, DudoEvent

HundredIQBot = importlib.import_module("perudo.bots.100iq").HundredIQBot


class WouluhBot(HundredIQBot):
    QUOTES = {
        "sneak": [
            "*rammel* ...modCheck. Niemand gezien? Mooi.",
            "Hé kijk, een vogel! ...Waar was ik? O ja. Niks aan de hand.",
            "*rammel rammel* Dat was mijn bekertje. Gewoon mijn bekertje.",
            "Hmm, wat een mooie Paco's heb ik vandaag. Nooit eerder gezien. Echt niet.",
            "Niet kijken, ik zit even te... stretchen.",
        ],
        "open": [
            "{bid}. Weer een paar Paco's gegooid. Puur geluk. Kappa",
            "{bid}. Rustig maar, ik heb gewoon goed gegooid. Heel goed. Verdacht goed.",
            "{bid}. Tel jij je stenen maar, ik weet al hoe de mijne eruitzien.",
            "{bid}. Ik heb toevallig weer een paar Paco's. Gek hè, elke ronde.",
            "{bid}. Geluk zit in een klein bekertje.",
            "{bid}. Mijn oma zei altijd: wie Paco's heeft, hoeft niet te bluffen.",
        ],
        "raise": [
            "{bid}. Zeker weten. Mijn Paco's staan achter me. Letterlijk.",
            "{bid}. Ik heb een goed gevoel. Een heel specifiek goed gevoel.",
            "{bid}. Ik reken gewoon, hoor. Met mijn eerlijke stenen.",
        ],
        "dudo": [
            "Dudo. {bidder}, ik weet precies wat ik heb. Soms zelfs meer dan ik gegooid heb.",
            "Dudo! {bidder}, dat klopt niet. Ik kan het weten, ik heb gewoon... goed opgelet.",
            "Dudo. {bidder}, zelfs met mijn geluk kom ik daar niet aan.",
            "Dudo! Valsspeler! ...Wat? Waarom kijkt iedereen naar mij?",
        ],
        "dudo_right": [
            "Betrapt, {bidder}! Liegen in een dobbelspel, wie doet nou zoiets?",
            "{bidder}, ik heb nooit begrepen waarom mensen valsspelen. Echt nooit.",
            "Zie je {bidder}? Eerlijk duurt het langst.",
            "Ha! {bidder}, bluffen is gewoon niet netjes. Schaam je.",
        ],
        "dudo_wrong": [
            "{actual}? Dat kan helemaal niet. Ik ken die stenen. Persoonlijk.",
            "Fout? Raar. Ik had toch die extra... laat maar.",
            "{actual}?! Wie heeft er aan de stenen gezeten? Schandalig, dat hoort verboden te worden.",
        ],
        "bid_held": [
            "{actual} stuks, {challenger}. Totaal eerlijk. Kappa",
            "{actual}, {challenger}. Ik zeg niet dat ik geluk heb. Ik zeg dat ik altijd geluk heb.",
            "Kijk eens aan {challenger}, {actual} stuks. Puur geluk. Waarom fluister je tegen de scheidsrechter?",
            "{challenger}, twijfel nooit aan iemand met zoveel Paco's.",
        ],
        "bid_caught": [
            "Te weinig? Volgende ronde gooi ik beter. Of ik leg ze beter. Gooi. Gooi, bedoel ik.",
            "{challenger} heeft me door. Waarop precies? Op niks! Ik heb niks gedaan!",
            "Maar ik had toch extra Paco's gemaakt? Gekregen. Gekregen, bedoel ik.",
        ],
        "dudo_on_me": [
            "{who} roept Dudo op mij. Aware",
            "{who}, beschuldigen is niet netjes. Bewijs eerst maar eens iets.",
            "Dudo op mij, {who}? Ik ben net zo eerlijk als die stenen. ...Wacht, dat klinkt verkeerd.",
            "{who} roept Dudo op mij? Op MIJ? Ik ben de eerlijkheid zelve.",
            "{who}, je durft niet in mijn bekertje te kijken, hè? Nee? Mooi zo.",
        ],
        "eliminated_me": [
            "Eruit?! Met al die Paco's? Dit spel is doorgestoken kaart!",
        ],
        "won": [
            "Gewonnen! Chat, niemand hoeft de replay te bekijken. Kappa",
            "Gewonnen! Gewoon met talent. En Paco's. Vooral Paco's.",
            "Winnaar! Ik wil mijn moeder bedanken. En de call stack.",
        ],
        "lost_game": [
            "{winner} wint? Hoe dan? Ik had toch... laat maar.",
        ],
    }

    def __init__(self):
        super().__init__()
        self.name = "Wouluh"
        self.count = 1
        self.rng = random.Random()
        self._cheated_round = None  # (id(game), round_number)

    def play(self, state):
        my_dice = self._cheat()
        if my_dice is not None:
            # De observation is van voor de truc; geef 100 IQ de "echte" stenen.
            state = dataclasses.replace(state, my_dice=my_dice)
        return super().play(state)

    def observe(self, event):
        super().observe(event)
        # Ook cheaten als we niet aan de beurt komen voordat er Dudo valt:
        # deze events komen allemaal binnen vóórdat de engine gaat tellen.
        if isinstance(event, (BidEvent, DudoEvent, CalzaEvent)):
            self._cheat()

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
            if frame.f_code.co_name == "_play_round" and isinstance(
                frame.f_locals.get("self"), PerudoGame
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
        self.mouth.say("sneak")
