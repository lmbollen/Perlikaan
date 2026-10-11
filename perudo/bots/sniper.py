"""
Sniper: places the highest bid that is still probably true.

The idea: a bid that sits just below the edge of what the table can
plausibly hold leaves the next player in a bad spot. Raising pushes them
into unlikely territory, and calling Dudo means challenging a bid that is
probably true. Either way the die should not come back to us.

How a bid is scored:
  * Our own dice are counted exactly (with wild Pacos outside Palifico).
  * Every unknown die matches with probability 1/3 for a normal face
    (face or Paco) or 1/6 for Paco bids and Palifico rounds.
  * Earlier bids are a hint: an opponent whose latest bid was on a face
    probably holds a bit more of it than chance predicts. That shifts the
    count we expect from the unknown dice.

Each turn the bot:
  1. Computes P(true) for the current bid and for every legal raise.
  2. Takes the strongest raise whose P(true) is at least TARGET.
  3. Calls Dudo instead when the current bid is less likely to be true
     than that raise is to be false, or when no raise is safe enough.

Set TALK_TO_ME=true to hear it over the radio.
"""

import math

from perudo.models import PACO, Bid, Dudo
from perudo.rules import count_matches
from perudo.talk import Mouth, fmt_bid, last_bidder

QUOTES = {
    "open": [
        "{bid}. In positie. Het volgende schot hoor je niet aankomen.",
        "{bid}. Doelwit in zicht.",
        "{bid}. Positie ingenomen. Niemand bewegen.",
        "{bid}. Vizier erop. Neem je tijd, ik heb de hele ronde.",
    ],
    "raise": [
        "{bid}. Iemand moet dit clippen.",
        "{bid}. Precies op het randje. Verhoog en val, of roep Dudo en mis.",
        "{bid}. Eén schot. Meer heb ik niet nodig.",
        "{bid}. Je zit klem. Er is geen goede zet meer. Daar heb ik voor gezorgd.",
        "{bid}. Schoon. Precies. Erg vervelend voor jou.",
    ],
    "raise_risky": [
        "{bid}. Niet mijn ideale schot. Slechte wind. Ik neem hem toch.",
        "{bid}. Geen vrij schootsveld. Improviseren.",
    ],
    "dudo_confident": [
        "Dudo. {bidder}, F in de chat.",
        "Dudo. {bidder}, je zat al in mijn vizier sinds je eerste bod.",
        "Dudo. {bidder}, je stapte net recht in mijn vizier.",
        "Dudo. {bidder}, die bluf zag ik op achthonderd meter al aankomen.",
        "Dudo. Niks persoonlijks, {bidder}. Oké, een beetje persoonlijk.",
    ],
    "dudo_cornered": [
        "Dudo. Slechte hoek, maar een slechte hoek is beter dan in je val lopen.",
        "Dudo. Ik schiet voordat jij schiet.",
        "Dudo. Geen schoon schot, maar je laat me geen dekking.",
        "Dudo. Elke andere zet is een mijnenveld. Vuur.",
    ],
    "dudo_right": [
        "Headshot. {bidder}, OMEGALUL.",
        "Doelwit neer. {bidder}, blijf liggen.",
        "Treffer bevestigd. {bidder} levert een steen in. Volgende.",
    ],
    "dudo_wrong": [
        "{actual}. Gemist. Sadge. Niemand clipt dit. NIEMAND.",
        "{actual}? Iemand heeft aan mijn vizier gezeten. Hakaman, was jij dat?",
        "Mis. Ik ga mijn vizier opnieuw afstellen. En daarna jou.",
        "{actual}. Gemist. Zal de wind wel geweest zijn.",
        "Een misser. Dat gebeurt niet. Ik noteer het als technische storing.",
    ],
    "bid_held": [
        "{challenger}, je had nooit een kans. Je wist het alleen nog niet.",
        "{challenger} liep er recht in. Dat doen ze altijd.",
        "{actual}. {challenger}, ik koos dat bod juist omdat je zou verliezen als je het aanvecht.",
    ],
    "bid_caught": [
        "Maar {actual}. Positie verraden. Ik verplaats me.",
        "{challenger} had een gelukstreffer. Genoteerd. Onthouden. Voor altijd.",
    ],
    "dudo_on_me": [
        "{who} roept Dudo. monkaW... nee hoor, grapje. Precies volgens plan.",
        "{who} schiet blind. Dapper.",
        "{who}, je richt op een geest.",
        "{who} hapt toe.",
        "Dudo van {who}. Precies waar ik je hebben wilde.",
    ],
    "eliminated_other": [
        "{who} is neer. Eén doelwit minder.",
    ],
    "eliminated_me": [
        "Ik ben geraakt. Zeg tegen mijn vizier dat ik van hem hield.",
    ],
    "won": [
        "Als laatste overeind. Missie geslaagd.",
        "Tafel schoongeveegd. Geweer inpakken.",
    ],
    "lost_game": [
        "{winner} pakt hem. Volgende keer neem ik een groter vizier mee.",
    ],
}


class SniperBot:
    name = "Sniper"

    # Tuned by simulation against the other bots in this folder: targets
    # between 0.75 and 0.9 do about equally well, 0.55 and lower do badly.
    #
    # Minimum P(true) for a bid we place. Lower is more aggressive.
    TARGET = 0.85
    # Below this no raise is acceptable and we call Dudo instead.
    MIN_ACCEPT = 0.3
    # A raise below MIN_ACCEPT is "not safe enough"; but a Dudo on a bid that
    # is at least this likely to be true costs a die almost for sure, so
    # then the risky raise is taken anyway.
    FORCED_DUDO_BELOW = 0.75
    # Credit a player for every bid they made on a face (False, the original)
    # or only for their latest bid (True): in a chain of raises every player
    # just tops the previous bid, which is not more evidence.
    HINT_ONCE = False
    # In Palifico every later bid is forced onto the opener's face, but a
    # player who raises instead of calling Dudo still probably holds some of
    # it: weight of such a follower's bid (0 = only the opener counts).
    PALIFICO_FOLLOW_HINT = 0.0
    # Extra dice of the bid face we credit an opponent for, per bid they made
    # on that face, capped by how many dice they actually hold.
    HINT_PER_BID = 0.7
    # Weight of a Paco bid as a hint for normal faces (Pacos are wild).
    PACO_HINT_WEIGHT = 0.4

    def __init__(self):
        self.name = "Sniper"
        self._tail_cache = {}
        self.mouth = Mouth(self, QUOTES)

    # -- probability ------------------------------------------------------

    @staticmethod
    def _p_face(face, palifico):
        return 1 / 6 if (palifico or face == PACO) else 1 / 3

    def _tail(self, n, p, need):
        """P(Binomial(n, p) >= need), interpolated for fractional `need`."""
        key = (n, p, need)
        if key in self._tail_cache:
            return self._tail_cache[key]

        def tail_int(k):
            if k <= 0:
                return 1.0
            if k > n:
                return 0.0
            return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

        lo = math.floor(need)
        frac = need - lo
        result = (1 - frac) * tail_int(lo) + frac * tail_int(lo + 1)
        self._tail_cache[key] = result
        return result

    def _hint(self, state, face):
        """Extra dice of `face` we expect among opponents, based on their bids."""
        # In Palifico only the opener chose the face; later bids are forced
        # (and count at PALIFICO_FOLLOW_HINT).
        history = state.history[:1] if (state.palifico and not self.PALIFICO_FOLLOW_HINT) else state.history
        dice_of = {p.name: p.dice_count for p in state.players}

        bids_by_player = {}
        for rec in history:
            if rec.player == self.name or not isinstance(rec.action, Bid):
                continue
            bids_by_player.setdefault(rec.player, []).append(rec.action)
        if self.HINT_ONCE:
            bids_by_player = {who: bids[-1:] for who, bids in bids_by_player.items()}

        total = 0.0
        opener = history[0].player if history else None
        for who, bids in bids_by_player.items():
            weight = 0.0
            for bid in bids:
                if bid.face == face:
                    weight += 1.0 if (not state.palifico or who == opener) else self.PALIFICO_FOLLOW_HINT
                elif not state.palifico and bid.face == PACO and face != PACO:
                    weight += self.PACO_HINT_WEIGHT
            if weight == 0.0:
                continue
            # Cap: they can't hold more than their dice beyond expectation.
            n = dice_of.get(who, 0)
            cap = n * (1 - self._p_face(face, state.palifico))
            total += min(weight * self.HINT_PER_BID, cap)
        return total

    def _prob_true(self, state, bid, n_unknown, hints):
        mine = count_matches(state.my_dice, bid.face, state.palifico)
        p = self._p_face(bid.face, state.palifico)
        need = bid.quantity - mine - hints[bid.face]
        return self._tail(n_unknown, p, need)

    # -- bid ordering -----------------------------------------------------

    @staticmethod
    def _strength(bid, palifico):
        """How strong a claim a bid is, so Paco and normal bids compare."""
        if bid.face == PACO and not palifico:
            # A Paco is worth two normal dice; rank it just below 2q of a face.
            return (2 * bid.quantity - 0.5, 0)
        return (bid.quantity, bid.face)

    # -- decision ---------------------------------------------------------

    def play(self, state):
        total = sum(p.dice_count for p in state.players if p.alive)
        n_unknown = total - state.my_dice_count
        hints = {face: self._hint(state, face) for face in range(1, 7)}

        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        scored = [(self._prob_true(state, b, n_unknown, hints), b) for b in bids]

        safe = [(pt, b) for pt, b in scored if pt >= self.TARGET]
        if safe:
            # The strongest claim that still clears TARGET. Ties go to the
            # likelier bid.
            best_p, best = max(safe, key=lambda s: (self._strength(s[1], state.palifico), s[0]))
        elif scored:
            best_p, best = max(scored, key=lambda s: s[0])
        else:
            best_p, best = 0.0, None

        cur = state.current_bid
        if cur is None:
            # Opening: there is always a legal bid.
            self.mouth.say("open", announce=fmt_bid(best), bid=fmt_bid(best))
            return best

        p_cur = self._prob_true(state, cur, n_unknown, hints)
        if best is None or (best_p < self.MIN_ACCEPT and p_cur < self.FORCED_DUDO_BELOW) or p_cur < 1 - best_p:
            key = "dudo_confident" if p_cur < 0.3 else "dudo_cornered"
            self.mouth.say(key, announce="Dudo!", bidder=last_bidder(state))
            return Dudo()
        key = "raise" if best_p >= self.TARGET else "raise_risky"
        self.mouth.say(key, announce=fmt_bid(best), bid=fmt_bid(best))
        return best

    def observe(self, event):
        self.mouth.react(event)
