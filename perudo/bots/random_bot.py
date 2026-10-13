"""
The simplest possible bot: picks a uniformly random legal action.

This demonstrates the absolute minimum needed to write a Perudo bot -- it
never has to know a single rule of the game. (The talking is optional
garnish; set TALK_TO_ME=true to hear it.)
"""

import random

from perudo.models import Bid, Calza, Dudo
from perudo.talk import Mouth, fmt_bid, last_bidder

QUOTES = {
    "open": [
        "{bid}. Ik heb met een dobbelsteen bepaald wat ik met mijn dobbelstenen ga bieden.",
        "{bid}. Plan? Ik heb niet eens een plan voor de lunch.",
        "{bid}. Ogen dicht en gewoon ergens op gewezen. Wetenschap.",
        "Ik open met {bid}. Vraag me niet waarom. Ik weet het oprecht niet.",
        "{bid}! De stemmen in mijn hoofd kozen deze.",
    ],
    "bid": [
        "{bid}. Chat, is dit een goed bod? Chat? CHAT?",
        "{bid}. Clueless",
        "{bid}. Ik heb niet naar mijn stenen gekeken. Ik kan niet lezen.",
        "{bid}! Is dat goed? Dat klinkt goed.",
        "{bid}. Mijn strategie is zo geavanceerd dat zelfs ik hem niet snap.",
        "{bid}. Je kunt me niet lezen als er niks te lezen valt.",
        "{bid}. Chaos is een ladder, en ik val er net vanaf.",
    ],
    "dudo": [
        "Dudo! Jebaited. Ik weet alleen niet wie.",
        "Dudo! Daar wacht ik al de hele avond op. Of op bieden. Vijftig-vijftig eigenlijk.",
        "DUDO! ...wacht, wat doet die ook alweer?",
        "Dudo! {bidder}, ik heb geen enkele reden om aan je te twijfelen. Dát maakt het eng.",
        "Dudo. Niet vanwege de wiskunde. Vanwege de vibes.",
    ],
    "calza": [
        "CALZA! Geen idee wat het betekent, maar het klinkt pittig.",
        "Calza! Het geluk is met de onwetenden.",
    ],
    "dudo_right": [
        "{bidder} verliest van random.choice. KEKW",
        "Berekend. En met berekend bedoel ik: muntje opgegooid.",
        "Pure skill. Kijk alsjeblieft niet in mijn broncode.",
        "{bidder}, je bent net verslagen door een muntje. Laat dat even bezinken.",
    ],
    "dudo_wrong": [
        "Fout, hè? Nou ja, ook een kapotte klok staat twee keer per dag goed. Dit was niet een van die keren.",
        "Hé, {actual} stuks. Wie had dat kunnen voorspellen? Ik niet. Ik voorspel niks.",
    ],
    "bid_held": [
        "{challenger} roept Dudo en het klopte gewoon. LUL. Ik snap het zelf ook niet.",
        "{challenger}, je hebt verloren van een muntje. Laat dat maar even marineren.",
        "{challenger} twijfelde aan mij? Ik wist zelf niet eens dat ik gelijk had!",
        "Klopt! Precies zoals ik het níét gepland had.",
    ],
    "bid_caught": [
        "Betrapt op bluffen? Dan had ik eerst moeten weten dát ik bluf.",
        "{challenger} heeft me. Kan ik niet boos om worden, ik had mezelf ook Dudo gegeven.",
        "Maar {actual}? Ter verdediging: ik had ook geen idee.",
    ],
    "calza_right": [
        "PRECIES {actual}! Ook een blinde kip vindt weleens een graankorrel. Buig voor de kip.",
    ],
    "calza_wrong": [
        "{actual}, geen {bid}. Ik zei het wel met overtuiging, en dát telt.",
    ],
    "dudo_on_me": [
        "{who}, je probeert een bot zonder gedachten te slim af te zijn. Succes ermee.",
        "Dudo op mij, {who}? Zelfs ik weet niet of dat klopte.",
        "{who} roept Dudo op mij. Gedurfd, want ik weet zelf niet eens wat ik bood.",
    ],
    "eliminated_me": [
        "Ik lig eruit! Volgend potje zelfde tijd, zelfde nul plan.",
    ],
    "others_bluff_caught": [
        "{bidder} bluft en wordt gepakt. Ik bluf nooit. Ik weet gewoon niks.",
        "{challenger} wint. Ik had ook op {challenger} gegokt. Of op {bidder}. Vijftig-vijftig.",
        "{bidder} betrapt. KEKW. Ik snap niet wat er gebeurde, maar KEKW.",
    ],
    "others_dudo_wrong": [
        "{bidder} had gelijk! Wie had dat kunnen weten? Niet ik. Ik weet nooit iets.",
        "{challenger} gokte verkeerd. Welkom bij de club, {challenger}.",
    ],
    "others_calza_right": [
        "Calza raak, {challenger}! Ik ben jaloers. Of hongerig. Iets in die richting.",
    ],
    "others_calza_wrong": [
        "{challenger} mist de Calza. Had je een muntje opgegooid, net als ik.",
    ],
    "last_die": [
        "Nog één steen! Eindelijk overzicht. Eén ding om niet naar te kijken.",
        "Eén steen. Ik ga er willekeurig mee om. Zoals altijd.",
        "Eén steen over. Pepega tijd. Ik bedoel: strategie tijd.",
    ],
    "eliminated_other": [
        "{who} ligt eruit. Ik hoop dat ik dat niet per ongeluk deed.",
        "Dag {who}! Of hallo. Ik haal die twee altijd door elkaar.",
    ],
    "won": [
        "Ik heb gewonnen. Chat, is dit echt? Chat, clip dat. Clip dat NU.",
        "IK HEB GEWONNEN?! Laat iemand de engine checken. Nee, toch maar niet.",
    ],
    "lost_game": [
        "{winner} wint. Gefeliciteerd met het verslaan van een bot die dobbelt om te bepalen hoe hij dobbelt.",
    ],
}


class RandomBot:
    name = "Random Bot"
    count = 0

    def __init__(self):
        self.mouth = Mouth(self, QUOTES)

    def play(self, state):
        action = random.choice(state.legal_actions)
        if isinstance(action, Bid):
            bid = fmt_bid(action)
            self.mouth.say("open" if state.current_bid is None else "bid", announce=bid, bid=bid)
        elif isinstance(action, Dudo):
            self.mouth.say("dudo", announce="Dudo!", bidder=last_bidder(state))
        elif isinstance(action, Calza):
            self.mouth.say("calza", announce="Calza!")
        return action

    def observe(self, event):
        self.mouth.react(event)
