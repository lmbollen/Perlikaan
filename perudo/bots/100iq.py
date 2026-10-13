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

Set TALK_TO_ME=true to hear exactly average trash talk.
"""

import math

from perudo.models import PACO, Bid, Dudo
from perudo.rules import count_matches
from perudo.talk import Mouth, fmt_bid, last_bidder
from perudo.tts import face_name


class HundredIQBot:
    LANG = "nl"
    QUOTES = {
        "open": [
            "{bid}. 5Head opening. Jullie zijn er nog niet klaar voor.",
            "{bid}. De meeste {face} in mijn beker, dus {face}. Schakers haten dit ene trucje.",
            "{bid}. Ik heb de meeste {face}, dus ik bied op {face}. Ze noemen me een strateeg.",
            "{bid}. Ik heb mijn eigen stenen geteld. Allemaal. Twee keer.",
            "Ik open met {bid}. Perfect gemiddeld, net als ik. Perfect in balans.",
        ],
        "raise": [
            "{bid}. Big brain play. Of medium brain. Precies medium.",
            "{bid}. Nog steeds {face}. Consistentie is ook een vorm van genialiteit.",
            "{bid}. Boven de vijftig procent, dus ik verhoog. Dat is het hele algoritme. Prachtig.",
            "{bid}. Mijn IQ is precies honderd. Dat is letterlijk het midden. Van alles.",
            "{bid}. De klokkromme staat aan mijn kant. Ik BEN de klokkromme.",
        ],
        "dudo": [
            "Dudo. {bidder}, 5Head move incoming.",
            "Dudo. {bidder}, ik zeg niet dat jij dom bent. Dat zegt de binomiaalverdeling.",
            "Dudo! Kans onder de helft. Een half is een breuk. Ik ken breuken.",
            "Dudo. Onder de vijftig procent. Ik maak de regels niet, ik heb ze op school geleerd.",
            "Dudo! {bidder}, ik heb de binomiaal gedaan. Ja, ik weet wat binomiaal betekent. Ongeveer.",
            "Dudo. {bidder}, de cijfers zeggen nee, en ik ben een cijfermens.",
        ],
        "dudo_right": [
            "{bidder}, je bent net afgetroefd door het meest gemiddelde brein aan tafel.",
            "Klopt! Ik zou de wiskunde uitleggen, {bidder}, maar daar heb je minstens honderd IQ voor nodig.",
            "Precies zoals berekend. {bidder}, zo zien honderd IQ-punten eruit.",
            "{bidder}, probeer volgende keer eens na te denken. Het is gratis. Ik doe het continu.",
        ],
        "dudo_wrong": [
            "{actual}?! Clueless. Nee wacht, Aware. Nee, toch Clueless.",
            "{actual}. Oké, maar heeft iemand anders hier überhaupt een kans uitgerekend? Dacht ik al.",
            "{actual}? Dat was onder de vijftig procent! De stenen hebben de theorie niet gelezen.",
            "Hm, {actual}. Tja, met honderd IQ heb ik ongeveer de helft van de tijd gelijk. Dit was de andere helft.",
        ],
        "bid_held": [
            "{challenger}, dat was een monkaS-Dudo met nul hersencellen. 5Head ik.",
            "{challenger}, ik leg je kansrekening een keer uit. Langzaam.",
            "{challenger}, je daagde een bod uit met meer dan vijftig procent kans. Kun je eigenlijk wel rekenen?",
            "Klopt. Hou het misschien bij dammen, {challenger}.",
        ],
        "bid_caught": [
            "{challenger} heeft me. Vast de enige keer vandaag dat {challenger} heeft nagedacht.",
            "Maar {actual}... ik ging uit van mijn eigen stenen. Mijn eigen stenen hebben me verraden.",
            "Betrapt. Kijk, de gemiddelde mens had precies hetzelfde gedaan. Ik kan het weten, ik bén de gemiddelde mens.",
        ],
        "dudo_on_me": [
            "{who}, aan mij twijfelen is het beste bod dat ik vandaag van je heb gezien.",
            "Dudo, {who}? Gedurfd, van iemand met een ondergemiddelde... alles.",
            "{who} roept Dudo op mij? Gedurfd, van iemand die binomiaal waarschijnlijk niet eens kan spellen.",
        ],
        "eliminated_me": [
            "Eruit. Verslagen door mensen die statistisch dommer zijn dan ik. De helft dan.",
        ],
        "others_bluff_caught": [
            "{bidder} bood {bid} met {actual}. Onder de vijftig procent. Had ik kunnen zeggen.",
            "{challenger} pakt {bidder}. Gemiddeld goed gedaan. Mijn hoogste compliment.",
            "{bidder} betrapt. 5Head ik, Pepega {bidder}.",
        ],
        "others_dudo_wrong": [
            "{challenger} twijfelde aan een kloppend bod. Dat is een kans van... laat maar, te moeilijk voor jou.",
            "Het bod van {bidder} klopte. {challenger}, de binomiaal had het je kunnen vertellen.",
        ],
        "others_calza_right": [
            "Calza raak van {challenger}. Statistisch onwaarschijnlijk. Ik ben onder de indruk. Gemiddeld.",
        ],
        "others_calza_wrong": [
            "{challenger} gaat voor exact en mist. Exact is een kleine kans. Ik weet dat soort dingen.",
        ],
        "last_die": [
            "Nog één steen. Mijn steekproef is nu precies gemiddeld klein.",
            "Eén steen. Ik heb het uitgerekend: dit is minder dan vijf.",
        ],
        "eliminated_other": [
            "{who} ligt eruit. Het gemiddelde IQ aan tafel gaat omhoog. Door mij.",
            "Dag {who}. Er moest iemand onder het gemiddelde zitten.",
            "{who} ligt eruit. Skill issue. Gemiddeld skill issue.",
        ],
        "won": [
            "Gewonnen! Blijkt dat gemiddeld alles is wat je nodig hebt.",
            "Gewonnen. Slimmer dan de halve tafel, precies zoals de klokkromme beloofde.",
        ],
        "lost_game": [
            "{winner} wint. Vast zo iemand met honderdeen IQ.",
        ],
    }

    def __init__(self):
            self.name = "100 IQ Bot"
            self.count = 3
            self.mouth = Mouth(self, self.QUOTES, lang=self.LANG)

    def play(self, state):
        action = self._choose(state)
        if isinstance(action, Bid):
            face = face_name(action.face, quantity=2)  # "vieren", "Paco's"
            key = "open" if state.current_bid is None else "raise"
            self.mouth.say(key, announce=fmt_bid(action), bid=fmt_bid(action), face=face)
        else:
            self.mouth.say("dudo", announce="Dudo!", bidder=last_bidder(state))
        return action

    def observe(self, event):
        self.mouth.react(event)

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
