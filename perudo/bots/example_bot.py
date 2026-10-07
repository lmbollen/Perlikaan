"""
DIT IS IN PRINCIPE DE ENIGE FILE DIE JE ALS DEELNEMER HOEFT AAN TE PASSEN.

Kopieer dit bestand, geef je bot een nieuwe class-naam en naam, en pas
play() (en optioneel observe()) aan met je eigen strategie.

Je hoeft GEEN Perudo-regels te implementeren: de engine berekent voor je
welke acties legaal zijn (state.legal_actions) en regelt zelf de
beurtvolgorde, Palifico, Calza, eliminaties en geheime dobbelstenen.

Beschikbaar op `state` tijdens jouw beurt:
    state.my_dice          - jouw eigen dobbelstenen, bv. (1, 3, 3, 5, 6)
    state.my_dice_count    - aantal dobbelstenen dat je nog hebt
    state.players          - publieke info over ALLE spelers (dus ook hoeveel
                              spelers er zijn, wie er nog meedoet en met hoeveel
                              dobbelstenen -- ook spelers die al uitgeschakeld
                              zijn staan erin, met alive=False)
    state.current_bid      - het huidige bod (Bid) of None als je opent
    state.palifico         - True/False: is dit een Palifico-ronde
    state.round_number     - het huidige ronde-nummer
    state.history           - alle acties die deze ronde al gedaan zijn, elk
                              met .player (de NAAM van de bot die het deed)
                              en .action (Bid/Dudo/Calza)
    state.legal_actions     - alle acties die je nu mag teruggeven

Je bot heeft een `name`. Je mag die net als hieronder in __init__ zetten
(instance variable), of gewoon als class-attribute (`name = "..."`) zoals
RandomBot dat doet -- beide werken. De naam is je stabiele identiteit: hij
duikt overal op (state.players, state.history, observe()-events), ook in
simulation mode over honderden games heen, dus je kunt tegenstanders
herkennen en van ze leren ook al wisselt elke game de zitpositie.

Wil je bot praten (en trash talken)? Zie QUOTES hieronder en perudo/talk.py.
Met TALK_TO_ME=true worden de praatjes geprint, met TTS=true ook uitgesproken.
"""

from perudo.models import Bid, BidEvent, Calza, Dudo
from perudo.talk import Mouth, fmt_bid, last_bidder

# Optioneel: per situatie een paar zinnen, er wordt er willekeurig één
# gekozen. {bid}, {bidder} enz. worden ingevuld; zie perudo/talk.py voor de
# situaties waar Mouth.react() zelf op reageert (onthullingen, eliminaties...).
QUOTES = {
    "open": [
        "{bid}. Eerste bod uit het boekje. Letterlijk, ik bén het boekje.",
        "{bid}. Ik ben maar een voorbeeld, verwacht geen wonderen.",
        "{bid}. Kopieer mij gerust, maar dan wel met een betere strategie graag.",
    ],
    "raise": [
        "{bid}. ResidentSleeper. Ja, ik ben saai, ik ben de tutorial.",
        "{bid}. Het goedkoopste bod. Ik ben een voorbeeld, geen genie.",
        "{bid}. Zie je, zo bied je. Stap één uit de handleiding.",
        "{bid}. Precies zoals in de tutorial.",
    ],
    "dudo": [
        "Dudo! PauseChamp... {bidder}, ga je dit echt verliezen van de voorbeeldbot?",
        "Dudo! {bidder}, dat is meer dan de helft van alle stenen. Regel één van de handleiding.",
        "Dudo! {bidder}, meer dan de helft van alle stenen? Zelfs ík zie dat dat niet klopt.",
        "Dudo. Dit staat letterlijk in mijn commentaar, {bidder}. Lees het eens.",
    ],
    "calza": [
        "Calza. Geen idee eigenlijk, er was gewoon niks anders meer over.",
    ],
    "dudo_right": [
        "{bidder} verliest van de tutorial. KEKW",
        "Ha, {bidder}, je bent net verslagen door de standaardinstellingen.",
        "Kijk {bidder}, zelfs de voorbeeldbot weet dit. Pijnlijk.",
    ],
    "dudo_wrong": [
        "Oké, misschien moet je mij toch iets aanpassen voordat je me inlevert.",
        "{actual}?! Oké, ik ben nooit getest. Dat staat ook in de docs. Ergens.",
    ],
    "bid_held": [
        "Wie twijfelt aan de handleiding, {challenger}, verliest een steen. Dat staat er niet in, maar het klopt wel.",
        "{challenger}, je verliest van de voorbeeldbot. Vertel dat thuis maar niet.",
    ],
    "bid_caught": [
        "Ja, ja, {challenger}. Ik ben een voorbeeld, geen meesterwerk.",
        "Betrapt. Kopieer mij en maak het beter, daar ben ik voor.",
        "Betrapt. Ik ben dan ook maar een paar regels code.",
    ],
    "dudo_on_me": [
        "{who}, Dudo op de voorbeeldbot? Dat is als winnen van het oefenpotje.",
        "{who} valt de tutorial aan. Mooi, je leest in elk geval mee.",
        "{who} roept Dudo op de voorbeeldbot. Wie pest er nou de tutorial?",
    ],
    "eliminated_me": [
        "Eruit. Geeft niks, ik ben gemaakt om verslagen te worden. Dat was het hele punt.",
    ],
    "won": [
        "De voorbeeldbot wint. Chat, is dit echt? PogChamp",
        "Gewonnen?! Door de VOORBEELDBOT? Iedereen terug naar de tekentafel.",
    ],
    "lost_game": [
        "Gefeliciteerd {winner}, je hebt de tutorial uitgespeeld.",
    ],
}


class ExampleBot:
    def __init__(self):
        self.name = "Example Bot"
        self.count = 0

        # Gewone instance variables. In simulation mode blijft dit bot-object
        # bestaan over meerdere games heen, dus je kunt hier dingen over
        # tegenstanders onthouden (bijv. hoe vaak iemand bluft) -- gekoppeld
        # aan hun naam, niet aan hun (elke game willekeurige) zitpositie.
        self.opponent_bid_counts = {}

        # Optioneel: een "mond" om mee te praten (doet niks zonder TALK_TO_ME/TTS).
        self.mouth = Mouth(self, QUOTES, lang="nl")

    def on_game_start(self, info):
        # Optioneel: wordt één keer aangeroepen aan het begin van elk potje.
        pass

    def play(self, state):
        # 1) Kijk naar je eigen dobbelstenen.
        my_dice = state.my_dice  # bv. (1, 2, 2, 4, 6)

        # 2) Hoeveel spelers zitten er in totaal aan tafel, en hoeveel
        #    dobbelstenen zijn er in totaal nog in het spel? Dat heb je
        #    nodig om te bepalen of een bod "hoog" is -- 8 vieren is
        #    doodnormaal bij 6 spelers, maar bijna zeker gebluft bij 2.
        num_players = len(state.players)
        total_dice_in_play = sum(p.dice_count for p in state.players)

        # 3) Kijk wat er deze ronde al gezegd is, en door wie.
        for record in state.history:
            pass  # record.player (naam) en record.action (Bid/Dudo/Calza)

        # 4) Kijk naar het huidige bod en of dit een Palifico-ronde is.
        current_bid = state.current_bid  # None als jij de ronde opent
        is_palifico = state.palifico

        # 5) Splits de legale acties op in categorieën.
        possible_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        can_say_dudo = any(isinstance(a, Dudo) for a in state.legal_actions)
        can_say_calza = any(isinstance(a, Calza) for a in state.legal_actions)

        # Simpele voorbeeldstrategie: roep Dudo zodra het bod méér dan de
        # helft van alle dobbelstenen op tafel claimt (relatief aan het
        # aantal spelers/dobbelstenen, niet een vast getal).
        if current_bid is not None and can_say_dudo:
            if current_bid.quantity > total_dice_in_play / 2:
                self.mouth.say("dudo", announce="Dudo!", bidder=last_bidder(state))
                return Dudo()

        # Anders: als je nog mag bieden, doe het goedkoopste legale bod.
        if possible_bids:
            bid = min(possible_bids, key=lambda b: (b.quantity, b.face))
            self.mouth.say("open" if current_bid is None else "raise", announce=fmt_bid(bid), bid=fmt_bid(bid))
            return bid

        # Geen bids meer mogelijk (zeldzaam) -> val terug op Calza of Dudo.
        if can_say_calza:
            self.mouth.say("calza", announce="Calza!")
            return Calza()
        self.mouth.say("dudo", announce="Dudo!", bidder=last_bidder(state))
        return Dudo()

    def observe(self, event):
        # Optioneel: wordt aangeroepen voor elke publieke gebeurtenis, ook
        # als jij niet aan zet bent. Zo kun je tegenstanders leren kennen --
        # event.player is steeds hun stabiele naam.
        if isinstance(event, BidEvent):
            self.opponent_bid_counts[event.player] = (
                self.opponent_bid_counts.get(event.player, 0) + 1
            )

        # Laat de mond reageren op onthullingen, Dudo's op jou, eliminaties...
        self.mouth.react(event)

    def on_game_end(self, result):
        # Optioneel: wordt één keer aangeroepen als het potje afgelopen is.
        pass
