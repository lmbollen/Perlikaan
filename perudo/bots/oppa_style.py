"""
Oppa Style -- een Perudo-bot die rekent als een nerd en praat als een popster.

Strategie in het kort:
  * Voor elk bod rekenen we de kans uit dat het klopt: onze eigen stenen
    tellen we exact, de onbekende stenen volgen een binomiale verdeling.
  * Biedingen van tegenstanders zijn een hint: wie op een gezicht biedt,
    heeft er vaak een paar van. Hoe betrouwbaar die hint is leren we per
    tegenstander (op naam) uit elke onthulling, over games heen.
  * We kiezen de actie met de kleinste kans om een steen te verliezen:
    Dudo als het huidige bod waarschijnlijk vals is, Calza als "precies
    goed" heel waarschijnlijk is, en anders het veiligste bod.

Zet TALK_TO_ME=true in je environment om Oppa Style te horen praten, en
TTS=true om het ook hardop te horen (Piper).
"""

import math
import random
from pathlib import Path

from perudo.models import (
    Bid,
    BidEvent,
    Calza,
    DieGainedEvent,
    DieLostEvent,
    Dudo,
    PlayerEliminatedEvent,
    RevealEvent,
)
from perudo.talk import Mouth

PACO = 1
BOT_NAME = Path(__file__).stem.replace("_", " ").title()  # "Oppa Style"


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

QUOTES = {
    "open": [
        "{bid}. catJAM. Oppa voelt de beat.",
        "Oppa opent met {bid}. Hey, sexy dice!",
        "{bid}. Ik begin rustig... zoals het intro voordat de beat dropt.",
        "Ik open met {bid}. Eeeeh, sexy lady!",
        "{bid} om te beginnen. Mijn stenen dragen een zonnebril, je kunt ze niet lezen.",
    ],
    "open_bluff": [
        "{bid}. Of ik ze heb? Vraag het maar aan mijn paardendansleraar.",
        "{bid}! Helemaal echt. Geloof me, ik ben een beroemdheid.",
    ],
    "raise_safe": [
        "{bid}. Kreygasm. Wat een bod, wat een dans.",
        "{bid}. Mijn stenen hebben het me ingefluisterd. Ze liegen nooit. Meestal.",
        "{bid}, makkie. Ik heb de goederen onder dit bekertje.",
        "{bid}. Veiliger dan een K-pop-dansrepetitie.",
        "{bid}. Soepel als het verkeer in Gangnam om drie uur 's nachts.",
    ],
    "raise_risky": [
        "{bid}. monkaS. De paardendans begint te wiebelen.",
        "{bid}... *zweet in een smoking*",
        "Eh... {bid}? Kijk alsjeblieft niemand naar mij.",
        "{bid}. Als dit misgaat, ben ik hier nooit geweest.",
        "{bid}. Ik ben niet bang, mijn benen trillen altijd als ik dans.",
    ],
    "raise_paco": [
        "{bid}! Paco-tijd, het paard is uit de stal.",
        "Over op Paco's: {bid}. Galopperen!",
    ],
    "dudo_sure": [
        "DUDO! {bid}? Jij bluft harder dan mijn dansmoves.",
        "{bid}?? DUDO! Niet onder mijn toezicht, {who}.",
        "Dudo! {who}, die stenen bestaan alleen in je dromen.",
        "{bid}... wat schattig. DUDO!",
    ],
    "dudo_forced": [
        "Dudo... ik geloof het niet, maar ik geloof ook niet in mezelf.",
        "Dudo, denk ik? Nog hoger gaan zou nog gekker zijn.",
    ],
    "calza": [
        "CALZA! Precies {bid}. Ik voel het in mijn paardenbotten.",
        "Calza! Precies {bid}, als een perfect synchrone dansmove.",
    ],
    "palifico": [
        "Palifico-ronde! Geen wilde Paco's, dus ook geen wilde Oppa. Focus.",
        "Palifico... de muziek stopt, iedereen bevriezen.",
    ],
    "big_bid_seen": [
        "Wow {who}, {bid}? Iemand heeft aan de soju gezeten.",
        "{bid}, {who}? Gedurfd. Ik hou van gedurfd. Ik hou ook van Dudo.",
    ],
    "small_bid_seen": [
        "{who} gaat voor {bid}. Bang voor Oppa, hè?",
        "{bid}, {who}? Op je tenen lopen als een kat. Miauw.",
    ],
    "won_challenge": [
        "POGGERS! {who} verliest een steen. Oppa Gangnam style!",
        "Hah! {who} verliest een steen. Oppa Gangnam style!",
        "Zei ik toch. {who}, geef die steen maar aan de dansvloer.",
        "Ik las je als een songtekst, {who}.",
    ],
    "lost_challenge": [
        "Au. Die steen was mijn favoriete achtergronddanser.",
        "Ik ben een steen kwijt... ik huil niet, het is zweet van het dansen.",
        "Oké, oké, die is voor mij. Videoclip terugspoelen.",
    ],
    "gained_die": [
        "Er komt een steen terug! De comebacktour begint NU.",
        "Calza gelukt! Encore, encore!",
    ],
    "eliminated_other": [
        "{who} ligt eruit. F in de chat. Oké, genoeg, catJAM verder.",
        "Dag dag {who}! Jij bent van het podium afgedanst.",
        "{who} ligt eruit. Moment van stilte... oké, genoeg geweest.",
    ],
    "eliminated_me": [
        "Oppa ligt eruit... onthoud mijn moves, onthoud mij.",
    ],
    "won_game": [
        "WINNAAR! PogChamp! Oppa Gangnam style, helemaal naar de top!",
        "WINNAAR! Oppa Gangnam style, helemaal naar de top!",
    ],
}


class OppaStyleBot:
    name = BOT_NAME

    # Tunables (found by self-play against variants).
    CHALLENGE_RATE_DUEL = 0.75
    CHALLENGE_RATE_CROWD = 0.6
    JUMP_PENALTY = 0.015
    TRUST_PRIOR = 0.6
    CALZA_MIN = 0.42

    def __init__(self):
        self.name = BOT_NAME
        self.rng = random.Random()
        self.mouth = Mouth(self, QUOTES)

        # Per opponent (by name): how much more of the bid face they held than
        # chance would predict, summed over reveals. Prior = mild honesty.
        self.hint_sum = {}
        self.hint_n = {}

        # Bids made by each player this round, so we can grade them on reveal.
        self.round_bids = {}
        self.last_bidder = None
        self.round_opener = None
        self.my_dice_count = 5

    # -- talking --------------------------------------------------------

    def _say(self, key, announce=None, **kw):
        self.mouth.say(key, announce, **kw)

    @staticmethod
    def _fmt(bid):
        if bid is None:
            return "niks"
        return f"{bid.quantity}x {'paco' if bid.face == PACO else bid.face}"

    # -- probability helpers ------------------------------------------------

    @staticmethod
    def _tail(n, p, need):
        """P(Binomial(n, p) >= need), with `need` allowed to be fractional."""
        if need <= 0:
            return 1.0
        if need > n:
            return 0.0

        def tail_int(k):
            if k <= 0:
                return 1.0
            if k > n:
                return 0.0
            return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

        lo = math.floor(need)
        frac = need - lo
        return (1 - frac) * tail_int(lo) + frac * tail_int(lo + 1)

    @staticmethod
    def _exact(n, p, k):
        if k < 0 or k > n:
            return 0.0
        return math.comb(n, k) * p**k * (1 - p) ** (n - k)

    @staticmethod
    def _face_p(face, palifico):
        if palifico or face == PACO:
            return 1 / 6
        return 1 / 3

    @staticmethod
    def _mine(dice, face, palifico):
        if palifico or face == PACO:
            return sum(1 for d in dice if d == face)
        return sum(1 for d in dice if d == face or d == PACO)

    def _trust(self, who):
        # Expected surplus of the bid face a bidder holds; prior 0.6 (weight 3).
        return (self.hint_sum.get(who, 0.0) + self.TRUST_PRIOR * 3) / (self.hint_n.get(who, 0) + 3)

    def _hints(self, state, face):
        """Extra dice we expect on `face` because opponents bid on it."""
        if state.palifico:
            # Only the opener chose the face; later bids are forced.
            relevant = state.history[:1]
        else:
            relevant = state.history
        last_by = {}
        for rec in relevant:
            if rec.player != self.name and isinstance(rec.action, Bid):
                last_by[rec.player] = rec.action
        dice_of = {p.name: p.dice_count for p in state.players}
        total = 0.0
        for who, bid in last_by.items():
            if bid.face == face or (not state.palifico and bid.face == PACO and face != PACO):
                weight = 1.0 if bid.face == face else 0.4
                total += weight * max(0.0, min(self._trust(who), dice_of.get(who, 0) * 0.5))
        return total

    def _prob(self, state, bid, n_unknown):
        p = self._face_p(bid.face, state.palifico)
        mine = self._mine(state.my_dice, bid.face, state.palifico)
        hint = self._hints(state, bid.face)
        n_hint = min(hint, n_unknown)
        return self._tail(n_unknown, p, bid.quantity - mine - n_hint)

    def _calza_prob(self, state, bid, n_unknown):
        p = self._face_p(bid.face, state.palifico)
        mine = self._mine(state.my_dice, bid.face, state.palifico)
        hint = self._hints(state, bid.face)
        # Shift the distribution by the (rounded) hint and take exact mass.
        need = bid.quantity - mine - round(hint)
        return self._exact(n_unknown, p, need)

    # -- decision making ------------------------------------------------------

    def play(self, state):
        total = sum(p.dice_count for p in state.players if p.alive)
        n_unknown = total - state.my_dice_count
        cur = state.current_bid
        alive = [p for p in state.players if p.alive]
        legal_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        can_dudo = any(isinstance(a, Dudo) for a in state.legal_actions)
        can_calza = any(isinstance(a, Calza) for a in state.legal_actions)

        if state.palifico and not state.history:
            self._say("palifico")

        if cur is None:
            return self._open(state, n_unknown)

        # Score candidate bids: chance it's true, minus a penalty for jumping
        # higher than needed (gives opponents room and attention).
        scored = []
        for b in legal_bids:
            pt = self._prob(state, b, n_unknown)
            scored.append((pt, b))

        def bid_key(item):
            pt, b = item
            # Equivalent "normal" quantity so paco bids compare fairly.
            eq_q = b.quantity * 2 if (b.face == PACO and not state.palifico) else b.quantity
            return pt - self.JUMP_PENALTY * eq_q

        p_cur = self._prob(state, cur, n_unknown)
        best = max(scored, key=bid_key) if scored else None
        p_best = best[0] if best else 0.0

        # A raise only loses if someone challenges it and it's false; the next
        # player may also raise instead. Discount the raise's risk a bit,
        # more with many players (more chances someone else takes the heat).
        challenge_rate = self.CHALLENGE_RATE_DUEL if len(alive) <= 2 else self.CHALLENGE_RATE_CROWD
        raise_loss = (1 - p_best) * challenge_rate
        dudo_loss = p_cur

        # Calza: wins a die on exact hit (if dice recovery still possible).
        if can_calza:
            p_exact = self._calza_prob(state, cur, n_unknown)
            gain = 1.0 if (state.my_dice_count < 5 and len(alive) == len(state.players)) else 0.0
            calza_value = p_exact * gain - (1 - p_exact)
            if calza_value > -min(dudo_loss, raise_loss) + 0.05 and p_exact > self.CALZA_MIN:
                self._say("calza", "Calza!", bid=self._fmt(cur))
                return Calza()

        if can_dudo and (dudo_loss < raise_loss or best is None):
            key = "dudo_sure" if p_cur < 0.35 else "dudo_forced"
            self._say(key, "Dudo!", bid=self._fmt(cur), who=self.last_bidder or "vriend")
            return Dudo()

        if best is None:
            self._say("dudo_forced", "Dudo!")
            return Dudo()

        pt, b = best
        self._react_to_bid(state, cur, n_unknown)
        if b.face == PACO and not state.palifico and cur.face != PACO:
            self._say("raise_paco", self._fmt(b), bid=self._fmt(b))
        elif pt >= 0.6:
            self._say("raise_safe", self._fmt(b), bid=self._fmt(b))
        else:
            self._say("raise_risky", self._fmt(b), bid=self._fmt(b))
        return b

    def _open(self, state, n_unknown):
        # Bid our strongest face (relative to how common it is) at roughly its
        # expected count minus one: modest openings beat greedy ones in tests.
        faces = {b.face for b in state.legal_actions if isinstance(b, Bid)}

        def strength(face):
            return self._mine(state.my_dice, face, state.palifico) / self._face_p(face, state.palifico)

        face = max(faces, key=lambda f: (strength(f), f))
        expected = self._mine(state.my_dice, face, state.palifico) + n_unknown * self._face_p(face, state.palifico)
        b = Bid(max(1, int(expected) - 1), face)
        # Occasionally sandbag one lower so we're harder to read.
        if b.quantity > 1 and self.rng.random() < 0.25:
            lower = Bid(b.quantity - 1, b.face)
            if lower in state.legal_actions:
                b = lower
        mine = self._mine(state.my_dice, b.face, state.palifico)
        self._say("open" if mine > 0 else "open_bluff", self._fmt(b), bid=self._fmt(b))
        return b

    def _react_to_bid(self, state, cur, n_unknown):
        # Commentary on what the previous player did, if it stood out.
        if self.last_bidder is None:
            return
        p = self._face_p(cur.face, state.palifico)
        expected = state.my_dice_count * p + n_unknown * p
        if cur.quantity >= expected + 2:
            self._say("big_bid_seen", bid=self._fmt(cur), who=self.last_bidder)
        elif cur.quantity <= expected - 2:
            self._say("small_bid_seen", bid=self._fmt(cur), who=self.last_bidder)

    # -- learning from public events --------------------------------------

    def on_game_start(self, info):
        self.round_bids = {}
        self.last_bidder = None
        self.my_dice_count = 5

    def observe(self, event):
        if isinstance(event, BidEvent):
            self.last_bidder = event.player
            if not self.round_bids:
                self.round_opener = event.player
            self.round_bids.setdefault(event.player, []).append(event.bid)
            return

        if isinstance(event, RevealEvent):
            self._learn(event)
            if event.challenger == self.name or event.bidder == self.name:
                if event.challenge_type == "dudo":
                    i_won = (event.challenger == self.name) == (event.result == "bidder_wrong")
                    if i_won:
                        loser = event.bidder if event.challenger == self.name else event.challenger
                        self._say("won_challenge", who=loser)
            self.round_bids = {}
            self.last_bidder = None
            return

        if isinstance(event, DieLostEvent) and event.player == self.name:
            self.my_dice_count = event.dice_count
            if event.dice_count > 0:
                self._say("lost_challenge")
        elif isinstance(event, DieGainedEvent) and event.player == self.name:
            self.my_dice_count = event.dice_count
            self._say("gained_die")
        elif isinstance(event, PlayerEliminatedEvent):
            if event.player == self.name:
                self._say("eliminated_me")
            else:
                self._say("eliminated_other", who=event.player)

    def _learn(self, event):
        # Palifico-ness isn't in the reveal; infer it from the counting rule:
        # if a non-paco bid's count ignores pacos, it was a Palifico round.
        dice = event.dice_by_player()
        bid = event.current_bid
        all_dice = [d for ds in dice.values() for d in ds]
        with_pacos = sum(1 for d in all_dice if d == bid.face or d == PACO)
        palifico = bid.face != PACO and event.actual_count != with_pacos

        for who, bids in self.round_bids.items():
            if who == self.name or who not in dice:
                continue
            # In palifico only the opener chose a face; skip forced followers.
            if palifico and who != self.round_opener:
                continue
            last = bids[-1]
            held = dice[who]
            p = self._face_p(last.face, palifico)
            count = self._mine(held, last.face, palifico)
            surplus = count - len(held) * p
            self.hint_sum[who] = self.hint_sum.get(who, 0.0) + surplus
            self.hint_n[who] = self.hint_n.get(who, 0) + 1

    def on_game_end(self, result):
        if result.winner == self.name:
            self._say("won_game")
