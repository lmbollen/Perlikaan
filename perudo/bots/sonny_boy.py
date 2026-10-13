"""
Sonny Boy: een kansrekenende Perudo-bot met een grote mond.

Strategie in het kort:
  * Voor elk bod schatten we met de binomiaalverdeling hoe groot de kans is dat
    het klopt, gegeven onze eigen dobbelstenen + de onbekende dobbelstenen van
    de rest (1/3 kans per die voor gewone getallen, 1/6 voor Paco/Palifico).
  * Biedt een tegenstander zichtbaar hoog of is zijn bod onwaarschijnlijk -> Dudo.
  * Zit het bod er precies op (hoogste kans op "exact") -> Calza.
  * Anders: het bod met de grootste kans op waarheid, met af en toe een bluf.
  * We onthouden per tegenstander (op naam) hoe vaak ze bij een Dudo bluften.

Zet de omgevingsvariabele TALK_TO_ME=true om Sonny Boy te laten praten, en
TTS=true om hem dat ook echt hardop te laten zeggen (Piper).
"""

import random
from math import comb

from perudo.models import (
    Bid,
    BidEvent,
    Calza,
    CalzaEvent,
    DieLostEvent,
    Dudo,
    DudoEvent,
    GameEndedEvent,
    PlayerEliminatedEvent,
    RevealEvent,
)
from perudo.talk import Mouth

PACO = 1


def _p_at_least(n: int, p: float, k: int) -> float:
    """P(X >= k) voor X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _p_exact(n: int, p: float, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def _fmt_bid(bid) -> str:
    return f"{bid.quantity}x {'Paco' if bid.face == PACO else bid.face}"


QUOTES = {
    "open": [
        "{bid}. Ik open zacht. Als een kat die net uit bed komt.",
        "Ik open met {bid}. Mijn dobbelstenen zeggen: 'vertrouw ons'. Dat is verdacht.",
        "{bid}. Een bescheiden openingsbod. Ik ben nederigheid in een bekertje.",
    ],
    "safe_bid": [
        "{bid}. Ik bied laag. Niet bang hoor, gewoon... voorzichtig. Heel voorzichtig.",
        "{bid}. Een veilig bodje. Mijn moeder noemde mij altijd 'verstandig'. Of was het 'saai'?",
        "{bid}. Klein bod, grote vibes.",
    ],
    "raise": [
        "Ik verhoog naar {bid}! De dobbelstenen en ik hebben een afspraak.",
        "{bid}! Hoger! Zwaartekracht is maar een suggestie.",
        "{bid}. Ik ruik zwakte. Of het is mijn sok. We gaan door.",
    ],
    "bluff": [
        "{bid}. Helemaal waar. Kappa",
        "{bid}. Dat is honderd procent waar. Ik kruis mijn vingers niet. Echt niet.",
        "{bid}. Mijn bekertje zit er vol mee. Eerlijk. Ik zweer het op mijn bekertje.",
        "{bid}. Bluffen? IK? Ik kan niet eens liegen over mijn leeftijd.",
    ],
    "dudo": [
        "DUDO! Jebaited, jij, niet ik.",
        "DUDO! Dat bod is zo dun dat je er doorheen kunt lezen.",
        "Nee. Nee nee nee. DUDO.",
        "Ik geloof je net zoveel als een verzekeringsagent met een gratis cruise.",
    ],
    "dudo_unsure": [
        "Dudo, op gevoel. Mijn gevoel is vaak fout, maar het is wel zelfverzekerd.",
        "Ik zeg Dudo en hoop dat het universum meewerkt.",
    ],
    "calza": [
        "CALZA! Precies zoveel. Ik voel het in mijn dobbelstenen.",
        "Calza! Als het klopt ben ik een genie, anders... gewoon een pechvogel.",
    ],
    "won_dudo": [
        "KEKW. Ik wist het! (Ik wist het niet.)",
        "Haha! Ik wist het. Ik wist het altijd al. (Dat is gelogen, maar het voelt goed.)",
        "Gebluft en betrapt. Op naar de volgende!",
    ],
    "lost_dudo": [
        "Au. PepeHands. Mijn dobbelsteen-ziel.",
        "Au. Dat deed pijn in mijn dobbelsteen-ziel.",
        "Oké, die zag ik niet aankomen. Mijn dobbelstenen zijn verraders.",
    ],
    "calza_win": [
        "CALZA GERAAKT! Buigen graag. Dieper. Nog dieper.",
    ],
    "calza_lose": [
        "Calza mislukt. We doen alsof dit nooit gebeurd is.",
    ],
    "scared_opponent": [
        "{who} doet er het kleinste stapje bij. Bang voor het eigen bekertje?",
        "{who} fluistert een bod. Bang voor de dobbelstenen?",
        "{who} schuift een millimetertje op. Spannend, hoor.",
    ],
    "bold_opponent": [
        "{who} springt omhoog. PogChamp of KEKW? We gaan het zien.",
        "{who} gaat hoog in. Dapper of dwaas? Ik ga voor dwaas.",
        "{who} biedt alsof er geen morgen is. Er is wel een morgen, namelijk met minder dobbelstenen.",
    ],
    "opp_lost": [
        "{who} verliest een dobbelsteen. Een moment stilte... en nu lachen.",
        "Een minuutje stilte voor {who}'s dobbelsteen. Klaar. Verder.",
    ],
    "opp_out": [
        "{who} is eruit. F in de chat.",
        "{who} is eruit. Dag {who}, schrijf nog eens!",
    ],
    "dudo_on_me": [
        "{who} roept Dudo op mij. Brutaal!",
        "{who} twijfelt aan Sonny Boy? Aan MIJ? Ik ben gekwetst.",
        "Dudo, {who}? Oké, oké. Even mijn bekertje een knuffel geven.",
        "{who} roept Dudo op mij? monkaS. Oké, ademen, Sonny.",
    ],
    "bid_held": [
        "Het klopte! {who}, je moet Sonny Boy gewoon geloven. Altijd.",
        "Ha, {who}! Eerlijk als een padvinder. Een padvinder met dobbelstenen.",
        "Ha {who}! EZ Clap! Sonny Boy liegt nooit. Bijna nooit.",
    ],
    "bid_caught": [
        "Betrapt door {who}. Ik had mijn vingers wel gekruist, hoor.",
        "{who} heeft me. Mijn bekertje en ik gaan even praten.",
        "Betrapt door {who}. Sadge. Copium: het was een test.",
    ],
    "last_die": [
        "Nog één dobbelsteen. Mijn kleine kampioen. Jij en ik tegen de wereld.",
        "Eén steen over. Niet in paniek raken. Ik raak in paniek.",
    ],
    "eliminated_me": [
        "Ik lig eruit. Zeg tegen mijn bekertje dat ik van hem hield.",
        "Sonny Boy is uitgeschakeld. Applaus graag. Ook voor de dobbelstenen.",
    ],
    "won": [
        "IK HEB GEWONNEN! Mam, kijk! Ik ben op de dobbeltafel!",
        "Gewonnen! Een grote mond en een klein bekertje, meer heb je niet nodig.",
        "GG EZ! Clip it, chat! Sonny Boy, kampioen van de bekertjes!",
    ],
    "lost_game": [
        "{winner} wint. Goed gedaan. Ik ben niet jaloers. Een beetje.",
        "{winner} wint. Volgende keer neem ik een groter bekertje mee.",
        "{winner} wint. Skill issue? Nee. Bekertje issue.",
    ],
    "palifico": [
        "Palifico! Alle regels zijn anders. Ik ook. Ik ben ook anders.",
    ],
}


class SonnyBoy:
    CREDIT = 1
    CALZA_MIN = 0.5
    DUDO_AT = 0.45
    # A shaky raise only costs a die when it is challenged and false: how
    # likely the challenge is, for the Dudo-vs-raise comparison.
    CHALLENGE_FACTOR = 0.6
    # Now and then, on a face we hold, bid one step higher than the safest
    # raise (the occasional bluff), when that raise is still likely enough.
    BLUFF_RATE = 0.12
    BLUFF_MIN_P = 0.6
    # Most dice we credit the bidders on a face with, all together.
    CREDIT_CAP = 1.0

    # -- fixes (False restores the old behaviour, for benchmarks) ----------
    # A correct Calza gains a die only while nobody is out and we hold under
    # five; a wrong one always costs one. Calza was called at CALZA_MIN even
    # when it could gain nothing. Now, when it cannot gain, only a near
    # certain exact count (CALZA_FREE_PASS_MIN) is worth the free pass.
    # Paired, 480 games at tables of 2-6: +1.7 +/- 1.2 pp win rate.
    FIX_CALZA_GAIN = True
    CALZA_FREE_PASS_MIN = 0.85
    # The bid we face was scored with the learned credit for the bidders on
    # its face, the candidate raises without, so the Dudo-vs-raise comparison
    # mixed two estimates. Now both sides use the credit.
    # Paired, 480 games: +0.2 +/- 2.0 pp (within noise).
    FIX_CREDIT_CONSISTENT = True

    def __init__(self):
        self.name = "Sonny Boy"
        self.count = 1
        # Per tegenstander: [aantal bods dat bij Dudo-reveal klopte, totaal bods onthuld]
        self.bluff_stats = {}
        self.mouth = Mouth(self, QUOTES)
        self._rng = random.Random()
        self._round_bids = []  # (player, Bid) deze ronde -- via observe()
        self._last_event_quote_turn = 0
        self._talk_prev = None  # het laatste bod deze ronde, om op te reageren

    # ------------------------------------------------------------------
    # praatjes
    # ------------------------------------------------------------------
    def _say(self, key, announce=None, **fmt):
        self.mouth.say(key, announce, **fmt)

    @staticmethod
    def _strength(bid):
        """Hoe sterk een bod is: een Paco telt als twee gewone stenen."""
        return bid.quantity * 2 if bid.face == PACO else bid.quantity

    # ------------------------------------------------------------------
    # kansberekening
    # ------------------------------------------------------------------
    def _matches_mine(self, state, face, palifico):
        if palifico or face == PACO:
            return sum(1 for d in state.my_dice if d == face)
        return sum(1 for d in state.my_dice if d == face or d == PACO)

    def _unknown_and_p(self, state, face, palifico):
        total = sum(p.dice_count for p in state.players if p.alive)
        unknown = total - state.my_dice_count
        p = 1 / 6 if (palifico or face == PACO) else 1 / 3
        return unknown, p

    def _opponent_prior_shift(self, state, bid):
        """
        Extra verwacht aantal dobbelstenen van de laatste bieder op dit getal:
        wie net dit getal bood, heeft er waarschijnlijk minstens één.
        """
        return 0.0

    def _credit(self, state, face):
        """Dobbelstenen die we de tegenstanders die op dit getal boden toekennen:
        CREDIT per bieder (hun laatste bod), gewogen met hoe eerlijk ze bleken."""
        if state.current_bid is None:
            return 0.0
        latest = {}
        for rec in state.history:
            if isinstance(rec.action, Bid) and rec.player != self.name:
                latest[rec.player] = rec.action.face
        credit = 0.0
        for player, f in latest.items():
            if f == face or (f == PACO and face != PACO and not state.palifico):
                honesty = min(1.5, max(0.5, 2 * (1 - self._bluff_rate(player))))
                credit += self.CREDIT * honesty
        return min(credit, self.CREDIT_CAP)

    def _p_true(self, state, bid, palifico, credit=0):
        """credit: dobbelstenen die we de bieders op dit getal toekennen (True = bereken)."""
        if credit is True:
            credit = self._credit(state, bid.face)
        mine = self._matches_mine(state, bid.face, palifico) + credit
        unknown, p = self._unknown_and_p(state, bid.face, palifico)
        return _p_at_least(max(int(round(unknown - credit)), 0), p, int(round(bid.quantity - mine)))

    def _p_exact(self, state, bid, palifico, credit=0):
        if credit is True:
            credit = self._credit(state, bid.face)
        mine = self._matches_mine(state, bid.face, palifico) + credit
        unknown, p = self._unknown_and_p(state, bid.face, palifico)
        return _p_exact(max(int(round(unknown - credit)), 0), p, int(round(bid.quantity - mine)))

    def _last_bidder(self, state):
        for rec in reversed(state.history):
            if isinstance(rec.action, Bid):
                return rec.player
        return None

    def _bluff_rate(self, name):
        stats = self.bluff_stats.get(name)
        if not stats or stats[1] < 5:
            return 0.5
        return 1 - stats[0] / stats[1]

    # ------------------------------------------------------------------
    # beslissing
    # ------------------------------------------------------------------
    def play(self, state):
        legal = state.legal_actions
        bids = [a for a in legal if isinstance(a, Bid)]
        can_dudo = any(isinstance(a, Dudo) for a in legal)
        can_calza = any(isinstance(a, Calza) for a in legal)
        palifico = state.palifico
        cur = state.current_bid
        total = sum(p.dice_count for p in state.players if p.alive)

        if palifico and cur is None and state.round_number is not None:
            self._say("palifico")

        # Beoordeel het huidige bod.
        if cur is not None and can_dudo:
            p_true = self._p_true(state, cur, palifico, credit=True)
            # Tegenstanders die vaak bluffen: iets sneller Dudo.
            bidder = self._last_bidder(state)
            if bidder:
                p_true -= 0.08 * (self._bluff_rate(bidder) - 0.5)

            # Beste eigen vervolgbod.
            best_bid, best_p = self._best_bid(state, bids, palifico)

            # Calza: alleen als precies-kans echt hoog is.
            if can_calza:
                pe = self._p_exact(state, cur, palifico, credit=True)
                threshold = self.CALZA_MIN
                if self.FIX_CALZA_GAIN:
                    can_gain = state.my_dice_count < 5 and all(p.alive for p in state.players)
                    if not can_gain:
                        threshold = self.CALZA_FREE_PASS_MIN
                if pe >= threshold:
                    self._say("calza", "Calza!")
                    return Calza()

            # Dudo als het bod waarschijnlijker fout dan goed is, of als
            # doorbieden nauwelijks veiliger is.
            if best_bid is None:
                self._say("dudo", "Dudo!")
                return Dudo()
            # Dudo als het bod waarschijnlijk vals is, of als een Dudo minder
            # vaak een steen kost dan doorbieden (dat kost er pas een als het
            # bod wordt aangevochten EN vals is).
            if p_true < self.DUDO_AT or p_true < (1 - best_p) * self.CHALLENGE_FACTOR:
                self._say("dudo" if p_true < 0.3 else "dudo_unsure", "Dudo!")
                return Dudo()
            return self._bid(state, best_bid, best_p, cur)

        # Openen.
        if not bids:
            return Calza() if can_calza else Dudo()
        best_bid, best_p = self._best_bid(state, bids, palifico, opening=True)
        self._say("open", _fmt_bid(best_bid), bid=_fmt_bid(best_bid))
        return best_bid

    def _best_bid(self, state, bids, palifico, opening=False):
        if not bids:
            return None, 0.0
        scored = []
        for b in bids:
            p = self._p_true(state, b, palifico, credit=True if self.FIX_CREDIT_CONSISTENT else 0)
            scored.append((p, b))
        if opening:
            # Open op ons sterkste getal met een laag, redelijk veilig bod;
            # kies hoogste kans, bij gelijke kans een groter aantal niet nodig.
            # Kleine ruis zodat we niet voorspelbaar zijn.
            best_p = max(p for p, _ in scored)
            cands = [(p, b) for p, b in scored if p >= best_p - 0.05]
            # Voorkeur voor getallen waar we zelf veel van hebben (eigen dobbelstenen).
            cands.sort(key=lambda x: (-self._matches_mine(state, x[1].face, palifico), x[1].quantity))
            p, b = cands[0]
            if len(cands) > 1 and self._rng.random() < 0.3:
                p, b = self._rng.choice(cands)
            return b, p
        # Hoogste kans; bij (bijna) gelijke kans het laagste bod (zo houden we ruimte).
        best_p = max(p for p, _ in scored)
        cands = [(p, b) for p, b in scored if p >= best_p - 0.02]
        cands.sort(key=lambda x: (x[1].quantity, x[1].face))
        p, b = cands[0]
        return b, p

    def _bid(self, state, best_bid, best_p, cur):
        # Bluf af en toe: ga bewust één stap hoger op een getal dat we zelf hebben.
        if best_p >= self.BLUFF_MIN_P and self._rng.random() < self.BLUFF_RATE:
            higher = Bid(best_bid.quantity + 1, best_bid.face)
            if higher in state.legal_actions and self._matches_mine(state, best_bid.face, state.palifico) > 0:
                best_bid, best_p = higher, self._p_true(state, higher, state.palifico)
        bid = _fmt_bid(best_bid)
        if best_p < 0.45:
            self._say("bluff", bid, bid=bid)
        elif best_p > 0.75:
            self._say("safe_bid", bid, bid=bid)
        else:
            self._say("raise", bid, bid=bid)
        return best_bid

    # ------------------------------------------------------------------
    # leren / reageren
    # ------------------------------------------------------------------
    def observe(self, event):
        self.mouth.notice(event)          # Dudo op mijn bod, mijn laatste steen
        if isinstance(event, RevealEvent):
            self._talk_prev = None
            # Zegt iets over de bluf-frequentie van de bieder.
            correct = event.actual_count >= event.current_bid.quantity
            st = self.bluff_stats.setdefault(event.bidder, [0, 0])
            st[1] += 1
            if correct:
                st[0] += 1
            if event.challenger == self.name:
                self._say("won_dudo" if not correct and event.challenge_type == "dudo" else
                          "calza_win" if event.challenge_type == "calza" and event.result == "correct"
                          else "calza_lose" if event.challenge_type == "calza" else "lost_dudo")
            elif event.bidder == self.name and event.challenge_type == "dudo":
                self._say("bid_held" if correct else "bid_caught", who=event.challenger)
        elif isinstance(event, BidEvent):
            prev, self._talk_prev = self._talk_prev, event.bid
            if event.player != self.name and prev is not None:
                # Alleen commentaar als het bod opvalt: een sprong, of het kleinst mogelijke stapje.
                jump = self._strength(event.bid) - self._strength(prev)
                if jump >= 2:
                    self._say("bold_opponent", who=event.player)
                elif jump == 0:
                    self._say("scared_opponent", who=event.player)
        elif isinstance(event, DieLostEvent) and event.player != self.name:
            self._say("opp_lost", who=event.player)
        elif isinstance(event, PlayerEliminatedEvent):
            if event.player == self.name:
                self._say("eliminated_me")
            else:
                self._say("opp_out", who=event.player)
        elif isinstance(event, GameEndedEvent):
            self._talk_prev = None
            if event.winner == self.name:
                self._say("won")
            else:
                self._say("lost_game", winner=event.winner)
        elif isinstance(event, CalzaEvent) or isinstance(event, DudoEvent):
            pass
