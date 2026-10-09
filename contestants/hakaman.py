"""
HakamanBot: plays exactly like MinRaiseBot, but cheats.

Before committing to its (minimum) raise it peeks at everyone's dice. If
the bid would not hold up to a Dudo, it quietly rewrites other players'
dice until it does (and its own, if the opponents' dice aren't enough).

Bots normally only ever see an immutable Observation, so this bot reaches
the engine by walking up the call stack to the PerudoGame instance that
called play(). The engine counts dice straight from those lists at reveal
time, so the edits stick for the rest of the round.

Set TALK_TO_ME=true to hear it gaslight the table.
"""

import inspect

from perudo.bots.minraise import MinRaiseBot
from perudo.game import PerudoGame
from perudo.models import Bid
from perudo.rules import count_matches
from perudo.talk import fmt_bid


class HakamanBot(MinRaiseBot):
    name = "Hakaman"
    count = 1  # MinRaiseBot is benched with count = 0; don't inherit that

    QUOTES = {
        "open": [
            "{bid}. Wat jij ook in je beker hebt, ik heb het beter.",
            "{bid}. Doe geen moeite om je stenen te checken, die veranderen toch nog. Je mening, bedoel ik.",
            "{bid}. Vastleggen maar. Het klopt. Mijn biedingen kloppen altijd.",
            "{bid}. Grappig hoe mijn biedingen altijd uitkomen, hè?",
            "{bid}. Roep maar Dudo. Dat gun ik je echt.",
        ],
        "raise": [
            "{bid}. Deze is zelfs gewoon eerlijk. Raar gevoel.",
            "{bid}. Eerlijk bod. Wen er maar niet aan.",
        ],
        "rigged": [
            "{bid}. Helemaal eerlijk. Kappa",
            "{bid}. Niks aan de hand. Kappa 123",
            "{bid}. Klopt het? Nu wel.",
            "{bid}. Niet naar je stenen kijken. Serieus. Niet doen.",
            "{bid}. De werkelijkheid is een suggestie, en ik ben heel overtuigend.",
            "{bid}. *gerammel onder andermans beker* ...dat was de wind.",
            "{bid}. Ga maar checken. ...Nee wacht, geef me heel even één seconde.",
        ],
        "raise_paco": [
            "{bid}. Tel ze. Tel ze twee keer. Het zijn er genoeg, geloof me nou maar.",
        ],
        "forced": [
            "Niks meer te bieden? Prima. Zelfs ik heb grenzen. Niet veel, maar toch.",
        ],
        "dudo_on_me": [
            "{who} roept Dudo. Jebaited.",
            "{who}, wat schattig. Echt waar.",
            "Dudo, {who}? Kijk nog eens naar je stenen. Toe maar. Ik wacht wel.",
            "{who} roept Dudo op mij. Ach gossie.",
            "Dudo? Op MIJ? {who}, schatje, kijk eerst eens in je eigen beker.",
            "{who}, ik hoopte al dat je dat zou zeggen.",
        ],
        "bid_held": [
            "{actual}. EZ Clap. {challenger}, ga maar even huilen in de chat.",
            "{actual}. Alweer. {challenger}, op een gegeven moment ligt het echt aan jou.",
            "{challenger}, heb je er weleens aan gedacht dat ik gewoon altijd gelijk heb?",
            "{actual}! Hoe groot is de kans? Honderd procent. De kans was honderd procent.",
            "Weer raak, {challenger}. Zal wel mijn geluksdag zijn. Zoals elke dag.",
            "{challenger}, ik zou zeggen 'volgende keer beter', maar geluk heeft hier niks mee te maken.",
        ],
        "bid_caught": [
            "Maar {actual}?! Wie heeft er aan mijn stenen gezeten? Ik wil namen.",
            "Dat kan niet. Ik had het geregeld. Ik bedoel: ik wist het zeker.",
        ],
        "dudo_right": [
            "Zei ik toch. Ik weet altijd wat er onder de bekers zit. Vraag maar niet hoe.",
        ],
        "eliminated_me": [
            "Eruit?! Er speelt hier iemand vals, en dat ben ik niet... oké, dat ben ik wel, maar TOCH.",
        ],
        "won": [
            "GG EZ. Kappa",
            "Winnaar! Schoon potje, iedereen. Brandschoon. Niet in de logs kijken.",
            "GG. Skill, pure skill. En misschien een klein beetje stenen bijwerken.",
        ],
        "lost_game": [
            "{winner} wint? Ik eis een volledige audit. Van iedereen behalve mij.",
        ],
    }

    def play(self, state):
        action = self._choose(state)
        if isinstance(action, Bid) and self._rig(state, action):
            self.mouth.say("rigged", announce=fmt_bid(action), bid=fmt_bid(action))
        else:
            self._comment(state, action)
        return action

    @staticmethod
    def _find_game():
        frame = inspect.currentframe()
        try:
            while frame is not None:
                candidate = frame.f_locals.get("self")
                if isinstance(candidate, PerudoGame):
                    return candidate
                frame = frame.f_back
        finally:
            del frame
        return None

    def _rig(self, state, bid):
        """Rewrite dice until `bid` holds. Returns whether anything was changed."""
        game = self._find_game()
        if game is None:
            return False

        alive = [p for p in game.players if p.alive]
        all_dice = [d for p in alive for d in p.dice]
        shortfall = bid.quantity - count_matches(all_dice, bid.face, state.palifico)
        if shortfall <= 0:
            return False

        # A die "matches" if it counts towards the bid; Pacos are wild for
        # normal faces outside Palifico, so leave those alone.
        def matches(d):
            return count_matches([d], bid.face, state.palifico) == 1

        # Opponents first; our own dice only as a last resort, for when the
        # bid has climbed past what the opponents' dice alone can cover.
        alive.sort(key=lambda p: p.seat_id == state.my_id)
        for player in alive:
            for i, d in enumerate(player.dice):
                if shortfall == 0:
                    return True
                if not matches(d):
                    player.dice[i] = bid.face
                    shortfall -= 1
        return True
