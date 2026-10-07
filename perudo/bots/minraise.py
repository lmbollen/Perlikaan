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

Set TALK_TO_ME=true to hear it plead for peace.
"""

from perudo.models import PACO, Bid, Dudo
from perudo.talk import Mouth, fmt_bid


class MinRaiseBot:
    name = "MinRaise Bot"
    count = 0

    QUOTES = {
        "open": [
            "{bid}. Laag ingezet, net als mijn zelfvertrouwen.",
            "{bid}. Klein. Bescheiden. Onaantastbaar. Alsjeblieft.",
            "{bid}. Ik zou lager gaan, maar dat mag niet van de regels.",
        ],
        "raise": [
            "{bid}. monkaS. Doe er één bij en hou je adem in.",
            "{bid}. Ik ben hier niet om vijanden te maken. Ik ben hier om er één bij te doen.",
            "{bid}. Het kleinst mogelijke stapje. Ik ben een bambi op het ijs.",
            "{bid}. Gewoon een duwtje. Niemand in paniek. Vooral ik niet.",
            "{bid}. Ik verhoog niet, ik... stel bij.",
            "{bid}. Je kúnt Dudo roepen. Of, luister even, je kunt ook gewoon aardig zijn.",
            "{bid}. Nog eentje dan. Zoals nog één aflevering. Onschuldig.",
        ],
        "raise_paco": [
            "{bid}. Paco's, want dat is technisch gezien kleiner. Ik heb het nagerekend.",
        ],
        "forced": [
            "Er valt niks meer te bieden? Dus ik moet... iemand aanvallen? Sorry, sorry. Dit ben ik niet.",
        ],
        "dudo_on_me": [
            "{who}... waarom? PepeHands",
            "{who}! Ik was nog wel zo beleefd!",
            "Jij ook, {who}? Na al die lieve kleine verhoginkjes die ik je gaf?",
            "{who}, waarom doe je dat nou? Ik heb nog nooit van mijn LEVEN iemand uitgedaagd.",
            "Dudo? Op mij? Ik deed niks, ik deed er gewoon één bij!",
        ],
        "bid_held": [
            "{challenger}, ik vergeef het je. Net aan.",
            "{actual}. Zie je, {challenger}? Je had er ook gewoon één bij kunnen doen. Zoals een normaal mens.",
            "Zie je wel, {challenger}? Geweld is nooit de oplossing.",
            "Het klopte. Natuurlijk klopte het. Ik ben geen leugenaar, ik ben gewoon verlegen.",
        ],
        "bid_caught": [
            "Maar {actual}. BibleThump. Ik wilde alleen maar vrienden zijn.",
            "Maar {actual}. Prima. Volgende keer verhoog ik nog kleiner. Bestaat er kleiner dan één?",
            "Maar {actual}. Oké. Ik zit wel in de hoek, waar niemand Dudo roept.",
            "{challenger} pakt mijn steen af. Ik hoop dat het het waard was, pestkop.",
        ],
        "dudo_right": [
            "Ik daagde iemand uit en ik had GELIJK? Ik word er misselijk van. Voelt macht zo?",
        ],
        "eliminated_me": [
            "Ik lig eruit. Sadge. F in de chat voor de enige vriendelijke bot aan tafel.",
            "Ik lig eruit. Ik heb nooit iemand kwaad gedaan. Vertel dat maar op mijn begrafenis.",
        ],
        "won": [
            "Gewonnen zonder één Dudo. widepeepoHappy. Liefde wint altijd.",
            "Gewonnen? Zonder één Dudo? Pacifisme werkt, mensen!",
        ],
        "lost_game": [
            "Gefeliciteerd, {winner}. Ik hoop dat al die agressie je gelukkig heeft gemaakt.",
        ],
    }

    def __init__(self):
        self.mouth = Mouth(self, self.QUOTES)

    def play(self, state):
        action = self._choose(state)
        self._comment(state, action)
        return action

    def observe(self, event):
        self.mouth.react(event)

    def _choose(self, state):
        possible_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        if possible_bids:
            return self._minraise(possible_bids)

        # Extremely rare edge case: in a Palifico round the quantity can
        # reach the total number of dice in play, at which point no
        # further bid is legal at all (the face can never change) --
        # only Dudo/Calza remain. With no raise left to make, this is the
        # one situation where MinRaiseBot is forced to pick one anyway.
        return state.legal_actions[0]

    def _comment(self, state, action):
        if not isinstance(action, Bid):
            self.mouth.say("forced", announce="Dudo!" if isinstance(action, Dudo) else "Calza!")
        elif state.current_bid is None:
            self.mouth.say("open", announce=fmt_bid(action), bid=fmt_bid(action))
        elif action.face == PACO and state.current_bid.face != PACO:
            self.mouth.say("raise_paco", announce=fmt_bid(action), bid=fmt_bid(action))
        else:
            self.mouth.say("raise", announce=fmt_bid(action), bid=fmt_bid(action))

    @staticmethod
    def _claim_strength(bid):
        return bid.quantity * 2 if bid.face == PACO else bid.quantity

    @classmethod
    def _minraise(cls, possible_bids):
        return min(
            possible_bids, key=lambda b: (cls._claim_strength(b), b.face == PACO, b.face)
        )
