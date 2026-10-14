"""
Fabulous -- a probability-driven Perudo bot with a big mouth.

Strategy in one breath: count my own dice, model every unknown die as a
coin with the right odds (1/3 for a normal face thanks to wild Pacos, 1/6
for Pacos or during Palifico), nudge that model with what opponents have
been bidding on (and how honest they have historically been about it),
and then pick Dudo / Calza / a bid by comparing probabilities.

Set the environment variable TALK_TO_ME=true to make the bot narrate its
inner life on the terminal.
"""

from __future__ import annotations

import functools
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from perudo.bot_base import BaseBot
from perudo.models import (
    PACO,
    Action,
    Bid,
    BidEvent,
    Calza,
    DieGainedEvent,
    Dudo,
    Event,
    GameEndedEvent,
    GameResult,
    GameStartInfo,
    Observation,
    PlayerEliminatedEvent,
    RevealEvent,
    RoundEndedEvent,
)
from perudo.rules import count_matches
from perudo.talk import Mouth

MAX_DICE = 5

# ---------------------------------------------------------------------------
# Tunables (found by running run_simulation.py a lot)
# ---------------------------------------------------------------------------

DUDO_THRESHOLD = 0.50        # call Dudo when P(current bid is true) drops below this
CALZA_THRESHOLD = 0.32       # call Calza when P(exact) is at least this (and a die can be won)
OPEN_SAFETY = 0.90           # opening bid: highest quantity with P(true) >= this
BID_SLACK = 0.0              # consider bids within this much of the best P(true)
CHALLENGE_FACTOR = 0.8       # how likely a shaky bid is to be challenged (for dudo-vs-bid trade-off);
                             # above 1 Fabulous called Dudo on bids that were almost surely true
                             # whenever every raise looked ugly, and lost a die for sure
HINT_PRIOR_HITS = 3.0        # prior for "opponent holds the face they bid on"
HINT_PRIOR_TRIES = 3.0

# -- fixes (False restores the old behaviour, for benchmarks) -----------------
# Trust (P(holds the face | bid on it)) started at 3/3 = certainty, so a fresh
# 1-die bidder's die was pinned as a sure match (a 1-die opener's 2x4 could
# never be called), and it was learned from every raise, forced Palifico bids
# included. Now: prior 2/3, one sample per face per round, no Palifico followers.
# Paired, 640 games at tables of 2-7: with the prior 2/3 the fix cost 3.3 +/- 1.8
# pp against the old certainty; 4/5 recovers 2.3 +/- 1.2 of that (3/4 +1.6, 3/3
# +2.8), so the prior is 4/5: a chosen face is nearly, not fully, certain.
FIX_TRUST = True
TRUST_PRIOR_HITS = 4.0
TRUST_PRIOR_TRIES = 5.0


def _fmt_bid(bid: Bid) -> str:
    face = "Paco" if bid.face == PACO else str(bid.face)
    return f"{bid.quantity}x {face}"


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

QUOTES: Dict[str, Sequence[str]] = {
    "open_strong": [
        "Ik open met {bid}. Ik gooide als een godin, probeer maar bij te houden.",
        "{bid}. Ja, ik heb ze echt. Schokkend, ik weet het.",
        "We beginnen met {bid}. Mijn stenen zijn zo mooi dat ik ze bijna niet wil verstoppen.",
    ],
    "open_meh": [
        "{bid} om te openen. Mijn stenen zijn... werk in uitvoering.",
        "Ik open met {bid}. Laag bod, nog lagere verwachtingen.",
        "{bid}. Iemand moet beginnen, en blijkbaar koos het universum mij.",
    ],
    "bid_safe": [
        "{bid}. Een klein stapje. Ik ben niet bang, ik ben gewoon... voorzichtig. Oké, bang.",
        "{bid}. Nauwelijks een verhoging. Ik speel het veiliger dan een gordelreclame.",
        "{bid}. Niks te zien hier, loop maar door naar het volgende slachtoffer.",
        "{bid}. Ik kan hoger, maar waarom een prima vriendschap verpesten.",
    ],
    "bid_normal": [
        "{bid}. Kan kloppen. Kan ook een schreeuw om hulp zijn.",
        "{bid}. Ik heb het uitgerekend. De wiskunde was niet enthousiast, maar tekende wel.",
        "{bid}. De kansen zeggen 'meh', mijn buik zegt 'gaan'. We hebben een compromis gesloten.",
    ],
    "bid_bluff": [
        "{bid}. Ik heb ze echt. Kappa",
        "{bid}! Met een stalen gezicht gezegd en helemaal niks om het te onderbouwen.",
        "{bid}. Als iemand het vraagt: ik heb ze allemaal. Allemaal. Echt. Allemaal.",
        "{bid}. Roep maar Dudo, ik daag je uit. Nee, toch maar liever niet.",
        "{bid}. Dit noemen wij in het vak 'creatief boekhouden'.",
    ],
    "dudo_confident": [
        "Dudo! {bidder}, chat, kijk dit. Clip dit.",
        "Dudo! {bidder}, dat bod heeft meer gaten dan mijn alibi.",
        "Dudo. {bidder}, ik bewonder de ambitie, maar nee.",
        "DUDO. {bidder}, ik heb eerlijkere biedingen gezien van een tweedehandsautoverkoper.",
    ],
    "dudo_scared": [
        "Dudo... denk ik? Of jij bluft, of ik ga zo huilen.",
        "Dudo. Vijftig-vijftig, net als mijn liefdesleven. Daar gaan we.",
        "Dudo, met het zelfvertrouwen van een kat op een natte vloer.",
    ],
    "dudo_forced": [
        "Dudo, omdat elke andere optie slechter was. Verhaal van mijn leven.",
        "Dudo. Niet omdat ik het geloof, maar omdat verhogen pure comedy zou zijn.",
        "Dudo. In het nauw gedreven als een wasbeer in een keuken. Eens kijken hoe dit afloopt.",
    ],
    "calza": [
        "CALZA! Precies {bid}. Ik voel het in mijn ellebogen.",
        "Calza! {bidder} zit er precies op, en ik kom de bonus ophalen.",
        "Calza. Veel risico voor één steen, maar zo fabulous ben ik nou eenmaal.",
    ],
    "my_dudo_right": [
        "Zei ik toch! {bidder}, bied volgende keer op stenen die je echt hebt.",
        "Gezien. {bidder}, die bluf was slapper dan tankstationkoffie.",
    ],
    "my_dudo_wrong": [
        "{actual}?! Copium. De stenen waren gewoon lelijk.",
        "O. Er waren er echt {actual}. Ik wil de manager van de kansrekening spreken.",
        "Tja, {actual} stuks. Mijn ellebogen hebben tegen me gelogen.",
    ],
    "my_bluff_caught": [
        "Oké {challenger}, je hebt me. Er waren er maar {actual}. Onbeschoft.",
        "Op heterdaad betrapt. Ter verdediging: het klonk waar toen ik het zei.",
    ],
    "my_bid_held": [
        "{actual}! {challenger}, gezellig dat je meedeed. EZ Clap.",
        "{challenger}, twijfelde je aan mij? Er waren er {actual}. Ik ben gekwetst. En ik win.",
        "Zie je? {actual}. Eerlijkheid is mijn beste beleid. Soms.",
    ],
    "my_calza_right": [
        "PRECIES {actual}! PogChamp! Chat, buig voor mij.",
        "PRECIES {actual}! Geef me die steen. Geef. Dank je.",
        "Calza binnen. Ik zou dit professioneel moeten doen. Ik dóé dit professioneel.",
    ],
    "my_calza_wrong": [
        "Niet precies. {actual}, geen {bid}. Zo dichtbij dat ik de steen die ik net kwijt ben kan proeven.",
        "Nee, {actual}. Calza was een gedurfde keuze, Fabulous. Gedurfd en fout.",
    ],
    "others_bluff_caught": [
        "{challenger} ruikt de bluf! {bidder}, jij deugniet.",
        "Ha! {bidder} bood {bid} en er waren er {actual}. Ik maak aantekeningen.",
    ],
    "others_dudo_wrong": [
        "{challenger} twijfelde aan {bidder} en betaalt de prijs. Vertrouwensproblemen, zie ik.",
        "Het bod van {bidder} klopte. {challenger}, geloof mensen eens een keertje.",
    ],
    "others_calza_right": [
        "{challenger} raakt een Calza?! Oké, dat is eigenlijk best indrukwekkend. Ik ben jaloers.",
        "Een exacte Calza van {challenger}. Hekserij, vast en zeker.",
    ],
    "others_calza_wrong": [
        "{challenger} ging voor Calza en mistte. Gedurfd. Tragisch. Vermakelijk.",
        "Calza-misser van {challenger}. Die voelde ik helemaal hier.",
    ],
    "eliminated_other": [
        "Doei {player}! Pas op dat de deur je nul stenen niet raakt op weg naar buiten.",
        "{player} ligt eruit. Eén bluffer minder aan tafel. Wacht even, nu ben ik de bluffer.",
    ],
    "eliminated_me": [
        "Ik lig eruit. Sadge. Maar wel een fabulous Sadge.",
        "Uitgeschakeld?! Prima. Ik droeg deze tafel toch al.",
        "Ik lig eruit. Onthoud me zoals ik was: fabulous en een tikje overmoedig.",
    ],
    "palifico": [
        "Volgende ronde Palifico! Geen wilde Paco's. Alleen rauwe, eerlijke stenen. Ik haat het.",
        "Er komt een Palifico-ronde aan. Iedereen doet normaal.",
    ],
    "dudo_on_me": [
        "Dudo, {who}? Op MIJ? Hoe durf je. Nee echt, hoe durf je.",
        "{who} roept Dudo op Fabulous. Wat een lef. Ik ben zelfs een beetje onder de indruk.",
        "{who}, twijfelen aan mij is twijfelen aan schoonheid zelf.",
        "Dudo op Fabulous, {who}? Ratio. En blokkeren.",
    ],
    "last_die": [
        "Nog één steen. Minimalistisch. Elegant. Fabulous.",
        "Eén steen over. Een diva heeft maar één spotlight nodig.",
        "Nog één steen. Clutch, schatjes. Clip it.",
    ],
    "won": [
        "IK HEB GEWONNEN! Niemand is verrast. Ik al helemaal niet.",
        "Victorie! Stenen vrezen mij, tafels respecteren mij, bots leren van mij.",
    ],
    "lost_game": [
        "{winner} wint. Ik eis een hertelling van de stenen die ik niet meer heb.",
        "Gefeliciteerd, {winner}. Je hebt de beste verslagen. Namelijk mij. Wacht.",
    ],
}


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------


class Fabulous(BaseBot):
    name = Path(__file__).stem.capitalize()

    def __init__(self) -> None:
        # Separate RNGs: one for decisions (fixed seed -> reproducible
        # simulations), one for quips (does not affect gameplay).
        self._rng = random.Random(0xFAB)
        self.mouth = Mouth(self, QUOTES)
        self._pmf_memo = None   # (state, hints, pmfs): see _unknown_pmf

        # Learned across games: how often each opponent actually held the
        # face they bid on.  name -> [hits, tries]
        self._hint_stats: Dict[str, List[float]] = defaultdict(
            lambda: [TRUST_PRIOR_HITS, TRUST_PRIOR_TRIES] if FIX_TRUST else [HINT_PRIOR_HITS, HINT_PRIOR_TRIES]
        )

        # Per-game / per-round state.
        self._recovery_enabled = True
        self._palifico_now = False
        self._round_bids: List[Tuple[str, Bid]] = []

    # -- lifecycle ---------------------------------------------------------

    def on_game_start(self, info: GameStartInfo) -> None:
        self._recovery_enabled = True
        self._palifico_now = False
        self._round_bids = []

    def on_game_end(self, result: GameResult) -> None:
        pass  # handled via GameEndedEvent so we also react when watching

    # -- the decision -------------------------------------------------------

    def play(self, state: Observation) -> Action:
        self._palifico_now = state.palifico
        if any(not p.alive for p in state.players):
            self._recovery_enabled = False

        legal_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        others = [p for p in state.players if p.alive and p.id != state.my_id]
        hints = self._collect_hints(state)

        if state.current_bid is None:
            bid = self._opening_bid(state, legal_bids, others, hints)
            my_count = count_matches(state.my_dice, bid.face, state.palifico)
            self.mouth.say("open_strong" if my_count >= 2 else "open_meh", announce=_fmt_bid(bid), bid=_fmt_bid(bid))
            return bid

        current = state.current_bid
        bidder = self._last_bidder(state)
        p_true = self._prob_at_least(state, others, hints, current.face, current.quantity)
        p_exact = self._prob_exact(state, others, hints, current.face, current.quantity)

        # Calza: only worth it when there is actually a die to win.
        can_gain = self._recovery_enabled and state.my_dice_count < MAX_DICE
        if can_gain and p_exact >= CALZA_THRESHOLD and self._has_calza(state):
            self.mouth.say("calza", announce="Calza!", bid=_fmt_bid(current), bidder=bidder)
            return Calza()

        scored = [
            (self._prob_at_least(state, others, hints, b.face, b.quantity), b)
            for b in legal_bids
        ]
        best_p = max((p for p, _ in scored), default=0.0)

        # Compare the chance of losing a die: a Dudo loses one when the bid
        # is true (p_true); a raise only when it is challenged AND false.
        bid_risk = (1.0 - best_p) * CHALLENGE_FACTOR
        if p_true < DUDO_THRESHOLD or (not legal_bids) or (p_true < bid_risk):
            if p_true < 0.2:
                self.mouth.say("dudo_confident", announce="Dudo!", bidder=bidder)
            elif p_true >= DUDO_THRESHOLD:
                self.mouth.say("dudo_forced", announce="Dudo!", bidder=bidder)
            else:
                self.mouth.say("dudo_scared", announce="Dudo!", bidder=bidder)
            return Dudo()

        bid, p_bid = self._pick_bid(scored, best_p)
        if p_bid >= 0.7:
            self.mouth.say("bid_safe", announce=_fmt_bid(bid), bid=_fmt_bid(bid))
        elif p_bid >= 0.4:
            self.mouth.say("bid_normal", announce=_fmt_bid(bid), bid=_fmt_bid(bid))
        else:
            self.mouth.say("bid_bluff", announce=_fmt_bid(bid), bid=_fmt_bid(bid))
        return bid

    # -- bid selection -------------------------------------------------------

    def _opening_bid(self, state, legal_bids, others, hints) -> Bid:
        faces = sorted({b.face for b in legal_bids})
        # Prefer the face we hold most of (Pacos count as wild outside Palifico).
        best_face = max(
            faces,
            key=lambda f: (
                count_matches(state.my_dice, f, state.palifico),
                0 if f == PACO else 1,  # tie-break: avoid opening on Paco
                self._rng.random(),
            ),
        )
        if not state.palifico and best_face == PACO:
            # Only Pacos in hand: open on a normal face instead, they count anyway.
            best_face = self._rng.choice([f for f in faces if f != PACO])

        candidates = [b for b in legal_bids if b.face == best_face]
        safe = [
            b
            for b in candidates
            if self._prob_at_least(state, others, hints, b.face, b.quantity) >= OPEN_SAFETY
        ]
        if safe:
            return max(safe, key=lambda b: b.quantity)
        return min(candidates, key=lambda b: b.quantity)

    def _pick_bid(self, scored: List[Tuple[float, Bid]], best_p: float) -> Tuple[Bid, float]:
        pool = [(p, b) for p, b in scored if p >= best_p - BID_SLACK]
        # Slightly prefer bids that raise the quantity (they age better) and
        # that keep the face we are strong on; weights stay probability-led.
        weights = [max(p, 1e-6) ** 4 for p, _ in pool]
        choice = self._rng.choices(pool, weights=weights, k=1)[0]
        return choice[1], choice[0]

    def _has_calza(self, state: Observation) -> bool:
        return any(isinstance(a, Calza) for a in state.legal_actions)

    # -- probability model ---------------------------------------------------

    def _collect_hints(self, state: Observation) -> Dict[str, Dict[int, int]]:
        """name -> {face: number of bids this round on that face} for opponents."""
        me = state.players[state.my_id].name
        hints: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for record in state.history:
            if record.player == me or not isinstance(record.action, Bid):
                continue
            hints[record.player][record.action.face] += 1
        return hints

    def _unknown_pmf(self, state, others, hints, face: int) -> List[float]:
        """Probability mass function of how many unknown dice count for `face`.
        Remembered for the decision (the same state and hints): it is the
        same for every bid on the face."""
        memo = self._pmf_memo
        if memo is None or memo[0] is not state or memo[1] is not hints:
            memo = self._pmf_memo = (state, hints, {})
        key = (face, tuple((p.name, p.dice_count) for p in others))
        pmf = memo[2].get(key)
        if pmf is None:
            pmf = memo[2][key] = self._work_out_unknown_pmf(state, others, hints, face)
        return pmf

    def _work_out_unknown_pmf(self, state, others, hints, face: int) -> List[float]:
        palifico = state.palifico
        p_match = 1.0 / 6.0 if (palifico or face == PACO) else 1.0 / 3.0

        pmf = [1.0]
        for player in others:
            n = player.dice_count
            # How many of this player's dice do we suspect are "pinned" to the face?
            player_hints = hints.get(player.name, {})
            mentions = player_hints.get(face, 0)
            if not palifico and face != PACO:
                mentions += player_hints.get(PACO, 0)  # their Pacos count for us too
            trust = self._trust(player.name)
            pinned_probs = []
            if mentions >= 1 and n >= 1:
                pinned_probs.append(trust)
            if mentions >= 2 and n >= 2:
                pinned_probs.append(trust * 0.5)

            player_pmf = self._player_pmf(n, p_match, pinned_probs)
            pmf = self._convolve(pmf, player_pmf)
        return pmf

    @staticmethod
    @functools.lru_cache(maxsize=None)   # pure: remembered by its arguments (callers only read it)
    def _binomial_pmf(n: int, p: float) -> List[float]:
        pmf = [0.0] * (n + 1)
        q = 1.0 - p
        coef = 1.0
        for k in range(n + 1):
            if k > 0:
                coef = coef * (n - k + 1) / k
            pmf[k] = coef * (p ** k) * (q ** (n - k))
        return pmf

    def _player_pmf(self, n: int, p: float, pinned_probs: List[float]) -> List[float]:
        """Mixture over the pinned dice: each pinned die is a match with its
        probability (`trust`), otherwise it is removed from the pool, i.e. it
        counts as a non-match and does NOT get a coin flip. Only the n -
        len(pinned_probs) other dice are coins with probability `p`."""
        result = [0.0] * (n + 1)
        # Enumerate which pinned dice actually are matches.
        combos = [([], 1.0)]
        for prob in pinned_probs:
            combos = [
                (fixed + [True], w * prob) for fixed, w in combos
            ] + [(fixed + [False], w * (1 - prob)) for fixed, w in combos]
        for fixed, weight in combos:
            k_fixed = sum(1 for f in fixed if f)
            free = n - len(fixed)
            base = self._binomial_pmf(free, p)
            for k, mass in enumerate(base):
                result[k + k_fixed] += weight * mass
        return result

    @staticmethod
    def _convolve(a: List[float], b: List[float]) -> List[float]:
        out = [0.0] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            if x == 0.0:
                continue
            for j, y in enumerate(b):
                out[i + j] += x * y
        return out

    def _prob_at_least(self, state, others, hints, face: int, quantity: int) -> float:
        mine = count_matches(state.my_dice, face, state.palifico)
        need = quantity - mine
        if need <= 0:
            return 1.0
        pmf = self._unknown_pmf(state, others, hints, face)
        return sum(pmf[need:]) if need < len(pmf) else 0.0

    def _prob_exact(self, state, others, hints, face: int, quantity: int) -> float:
        mine = count_matches(state.my_dice, face, state.palifico)
        need = quantity - mine
        if need < 0:
            return 0.0
        pmf = self._unknown_pmf(state, others, hints, face)
        return pmf[need] if need < len(pmf) else 0.0

    def _trust(self, player_name: str) -> float:
        hits, tries = self._hint_stats[player_name]
        return hits / tries

    # -- watching the table ---------------------------------------------------

    def observe(self, event: Event) -> None:
        me = self.name
        self.mouth.notice(event)      # a Dudo on my bid, my last die
        if isinstance(event, BidEvent):
            self._round_bids.append((event.player, event.bid))

        elif isinstance(event, RevealEvent):
            self._learn_from_reveal(event)
            ctx = dict(
                challenger=event.challenger,
                bidder=event.bidder,
                actual=event.actual_count,
                bid=_fmt_bid(event.current_bid),
            )
            if event.challenge_type == "dudo":
                if event.challenger == me:
                    self.mouth.say("my_dudo_wrong" if event.result == "bidder_correct" else "my_dudo_right", **ctx)
                elif event.bidder == me:
                    self.mouth.say("my_bid_held" if event.result == "bidder_correct" else "my_bluff_caught", **ctx)
                else:
                    self.mouth.say("others_dudo_wrong" if event.result == "bidder_correct" else "others_bluff_caught", **ctx)
            else:
                if event.challenger == me:
                    self.mouth.say("my_calza_right" if event.result == "correct" else "my_calza_wrong", **ctx)
                else:
                    self.mouth.say("others_calza_right" if event.result == "correct" else "others_calza_wrong", **ctx)

        elif isinstance(event, PlayerEliminatedEvent):
            self._recovery_enabled = False
            self.mouth.say("eliminated_me" if event.player == me else "eliminated_other", player=event.player)

        elif isinstance(event, RoundEndedEvent):
            self._round_bids = []
            self._palifico_now = event.palifico_next
            if event.palifico_next:
                self.mouth.say("palifico")

        elif isinstance(event, GameEndedEvent):
            self._round_bids = []
            if event.winner == me:
                self.mouth.say("won")
            elif me in event.standings:
                self.mouth.say("lost_game", winner=event.winner)

        elif isinstance(event, DieGainedEvent):
            pass

    def _learn_from_reveal(self, event: RevealEvent) -> None:
        dice = event.dice_by_player()
        opener = self._round_bids[0][0] if self._round_bids else None
        seen = set()
        for player, bid in self._round_bids:
            if player == self.name or player not in dice:
                continue
            if FIX_TRUST:
                # Only a bid that chose its face is a sample: the player's
                # first bid on that face this round, and in Palifico only the
                # opener's (every later bid is forced onto the same face).
                if (player, bid.face) in seen or (self._palifico_now and player != opener):
                    continue
                seen.add((player, bid.face))
            held = count_matches(dice[player], bid.face, self._palifico_now) > 0
            stats = self._hint_stats[player]
            stats[0] += 1.0 if held else 0.0
            stats[1] += 1.0

    # -- helpers ------------------------------------------------------------------

    @staticmethod
    def _last_bidder(state: Observation) -> str:
        for record in reversed(state.history):
            if isinstance(record.action, Bid):
                return record.player
        return "iemand"

