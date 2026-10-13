"""
Bluffo Boyyo: a quick-learning deceiver.

No fixed plays. Every legal action gets an expected value, in dice, and the
best one is played (near-equal options at random, so he is hard to read).
Deception is part of that value, so he misleads whenever it is safe enough:

  * a bid is worth: the chance it is true, against the chance the next
    player challenges it (learned per player from how likely the bid looks
    from their seat: a true bid that looks unlikely to them is a trap);
  * plus its deception: how far the bid claims more of that face than his
    hand and the table's bids support, weighted by how much this table
    reads bids (learned from how often players follow the face just bid).
    Opponents who read bids then believe in dice that are not there;
  * consistency: one bluff per round. Once he has bluffed on a face, a
    misleading bid on another face is out (INCONSISTENT); switching to an
    honest face (one he holds plenty of, or that the table's bids make
    plentiful) stays open;
  * Dudo is worth its chance of being right; Calza the chance of an exact
    count, with the die it gains while nobody is out and he has fewer than
    five, or the free pass once it gains nothing.

After every reveal he learns about each opponent by name with a fast moving
average (LEARN_RATE): how many dice of a face they really hold behind a bid
that chooses or follows it, how often they follow a face, their Calza rate,
and a Dudo model (how likely a bid must look to them before they call). He
also learns how often the bids he faces were really true, per bucket of his
own estimate, and corrects that estimate. Everything learned lives in memory
at class level, so it survives a new instance per game; nothing is written
to disk. The moving averages make recent games count more than old ones.

Set TALK_TO_ME=true to hear him sell it.
"""

import math
import random

from perudo.models import (
    PACO,
    Bid,
    BidEvent,
    Calza,
    Dudo,
    GameEndedEvent,
    RevealEvent,
    RoundEndedEvent,
)
from perudo.talk import Mouth, fmt_bid, last_bidder


def _count_matches(dice_values, face, palifico):
    """How many dice count towards a bid on `face`."""
    if palifico:
        return sum(1 for d in dice_values if d == face)
    if face == PACO:
        return sum(1 for d in dice_values if d == PACO)
    return sum(1 for d in dice_values if d == face or d == PACO)


def _p_face(face, palifico):
    return 1 / 6 if (palifico or face == PACO) else 1 / 3


def _binom_pmf(n, p, k):
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def _binom_tail(n, p, k):
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(_binom_pmf(n, p, j) for j in range(k, n + 1))


def _tail_at(n, p, need):
    """P(X >= need) for a fractional need (interpolated)."""
    lo = math.floor(need)
    frac = need - lo
    return (1 - frac) * _binom_tail(n, p, lo) + frac * _binom_tail(n, p, lo + 1)


def _pmf_at(n, p, need):
    lo = math.floor(need)
    frac = need - lo
    return (1 - frac) * _binom_pmf(n, p, lo) + frac * _binom_pmf(n, p, lo + 1)


def _sigmoid(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _is(event, *names):
    return type(event).__name__ in names


QUOTES = {
    "open": [
        "{bid}. Eerlijk bod. Ja, echt. Ook ik doe dat soms.",
        "{bid}. Deze keer heb ik ze gewoon. Geloof je me niet? Mooi zo.",
        "{bid}. Rustig begin. Boyyo houdt het netjes.",
        "{bid}. Kijk maar goed naar mijn gezicht. Leer het uit je hoofd.",
    ],
    "open_bluff": [
        "{bid}. Mijn favoriete getal. Altijd al geweest. Sinds net.",
        "{bid}. Mijn beker zit er vol mee, boyyo. Propvol.",
        "{bid}. Ik open waar het pijn doet. Bij jou, niet bij mij.",
        "{bid}. Vertrouw me maar. Iedereen vertrouwt Bluffo. Eén keer.",
        "{bid}. Schrijf het op: Bluffo heeft ze. Met potlood graag.",
        "{bid}. Kappa. Nee serieus. Kappa.",
    ],
    "raise": [
        "{bid}. Gewoon de waarheid. Voelt raar, maar oké.",
        "{bid}. Deze is echt. Juist daarom geloof je me niet.",
        "{bid}. Eerlijk duurt het langst. Ik ben het aan het testen.",
        "{bid}. Geen trucjes. Nou ja. Geen grote.",
    ],
    "raise_bluff": [
        "{bid}. Smooth. Zo smooth dat ik het zelf bijna geloof.",
        "{bid}. Ja boyyo. Ik heb ze. Allemaal. Vraag niet waar.",
        "{bid}. Jij biedt op stenen, ik bied op verhalen.",
        "{bid}. Mijn verhaal klopt. De stenen moeten nog even bijtrekken.",
        "{bid}. Kijk me aan en zeg dat ik lieg. Durf je niet, hè?",
        "{bid}. Ik verkoop dit bod. Wie koopt?",
        "{bid}. Chat, niet verklappen. Kappa",
        "{bid}. Skill issue als je dit gelooft. En ook als je het niet gelooft.",
    ],
    "dudo": [
        "Dudo. {bidder}, ik ken een bluf als ik er een zie. Ik maak ze zelf.",
        "Dudo! {bidder}, dat trucje is van mij. Copyright Bluffo.",
        "Dudo. {bidder}, ik heb je de vorige keer gezien. Ik onthoud alles.",
        "Dudo. Sorry {bidder}, maar liegen doen we hier niet. Behalve ik.",
        "Dudo! {bidder}, nette poging. Ik geef je een zesje.",
        "Dudo! {bidder}, Jebaited? Nee. Jij bent de gejebaite.",
    ],
    "calza": [
        "Calza. Precies {bid}. Bluffers weten altijd precies hoeveel er liggen.",
        "Calza! Op de kop af. Vraag niet hoe, boyyo.",
    ],
    "dudo_right": [
        "Betrapt, {bidder}! Takes one to know one.",
        "{bidder}, van een bluffer kun je niet stelen. Les één.",
        "Zie je {bidder}, ik lees jou als een open boek. Een dun boek.",
        "Zei ik toch. {bidder}, die bluf was amateurwerk.",
        "{bidder}, dacht je echt dat je Bluffo kon bluffen? KEKW",
        "Get rekt, {bidder}. Clip it, chat.",
    ],
    "dudo_wrong": [
        "{actual}? {bidder} sprak de waarheid? Wie doet dat nou?",
        "Maar {actual}. Oké {bidder}, die had ik niet zien aankomen. Genoteerd.",
        "{actual}. Ik dacht dat je loog, {bidder}. Ik projecteer, zegt mijn therapeut.",
        "{actual}. Oké chat, deze knippen we eruit. Sadge",
    ],
    "bid_held": [
        "{actual}! {challenger}, dit keer was het echt. Verwarrend, hè?",
        "Klopt gewoon, {challenger}. Ik lieg alleen als het telt.",
        "{challenger} dacht dat ik bluftte. Dat is precies wat ik wilde.",
        "{actual}. {challenger}, de beste leugen is soms de waarheid.",
        "{actual}! {challenger} OMEGALUL. Mind games, boyyo.",
    ],
    "bid_caught": [
        "Maar {actual}. Oké {challenger}, die is voor jou. De volgende niet.",
        "Betrapt. Prima. Ik heb nog honderd verhalen, {challenger}.",
        "{challenger} heeft me. Respect. Ik leer snel, boyyo. Héél snel.",
        "Maar {actual}? Iemand heeft mijn stenen verwisseld. Niet ik, voor de verandering.",
        "Betrapt. Copium: het was een zet voor de lange termijn.",
    ],
    "calza_right": [
        "PRECIES {actual}! Bluffo pakt een steen terug. Smooth.",
    ],
    "calza_wrong": [
        "{actual}, geen {bid}. Zelfs mijn calza's bluffen.",
    ],
    "dudo_on_me": [
        "{who} roept Dudo. Weet je het zeker? Echt zeker?",
        "Dudo op mij, {who}? Spannend. Voor jou.",
        "{who}, je denkt dat je me doorhebt. Iedereen denkt dat.",
        "Dudo, {who}? Ik zeg niks. Ik glimlach alleen.",
        "{who} roept Dudo. monkaS... grapje. Kappa",
    ],
    "others_bluff_caught": [
        "{bidder} betrapt op {bid}. Pff, amateurs. Zo doe je dat niet.",
        "{bidder} bluft en gaat onderuit. Kom maar eens bij mij op les.",
    ],
    "others_dudo_wrong": [
        "{bidder} had ze gewoon. {challenger}, je moet beter leren lezen. Bij mij.",
        "{challenger} dacht aan een bluf. Ik bluf hier, {challenger}, niet {bidder}.",
    ],
    "others_calza_right": [
        "Calza raak van {challenger}. Netjes. Bijna net zo netjes als ik.",
    ],
    "others_calza_wrong": [
        "{challenger} gaat voor Calza en mist. Te veel geloof in de waarheid.",
    ],
    "last_die": [
        "Nog één steen. Nu moet ik écht goed liegen.",
        "Eén steen over. Een goede bluffer heeft er maar één nodig.",
    ],
    "eliminated_other": [
        "Dag {who}. Ik had je bijna door. Nu hoeft het niet meer.",
        "{who} ligt eruit. Eén publiek minder voor mijn show.",
    ],
    "eliminated_me": [
        "Eruit. Maar geef toe: het waren mooie verhalen.",
        "Bluffo ligt eruit. Niemand geloofde me. Terecht, meestal.",
    ],
    "won": [
        "Gewonnen, boyyo! Met eerlijke stenen. Grapje.",
        "Winnaar! Ik heb tegen iedereen gelogen en iedereen tegen mij. Ik deed het beter.",
        "Gewonnen. Ik heb jullie allemaal geleerd. Jullie mij niet.",
        "GG EZ boyyo. Kappa 123. Ratio.",
    ],
    "lost_game": [
        "{winner} wint. Goed gespeeld. Volgende keer weet ik alles van je.",
        "{winner} wint. Ik heb aantekeningen gemaakt. Veel aantekeningen.",
        "{winner} wint. Skill issue. Van mij. Eén keer.",
    ],
}


class BluffoBoyyo:
    name = "Bluffo Boyyo"
    count = 1

    # -- values, in dice --------------------------------------------------
    # Worth of an opponent losing a die, divided by the number of opponents.
    OTHER_WEIGHT = 1.0
    # A bid that is not challenged passes the turn; a false one may still be
    # caught later in the round: that risk, per unit of P(false).
    LATER_RISK = 0.35
    # Cost per unit of P(false) for any bid, whether or not it gets called:
    # a bid that is not on the table is a bad bid.
    FALSE_PENALTY = 0.0
    # Worth of misleading the table: per die the bid overstates our holding
    # of its face (as a share of our dice), times how much the table reads
    # bids (0..1).
    DECEIVE = 0.4
    # Worth of a correct Calza that gains no die (someone is out, or we have
    # five): nothing happens and we start the next round.
    CALZA_FREE_PASS = 0.3
    # Options within this value of the best count as equal: one is played at
    # random, so we are hard to read.
    TIE_EPS = 0.02
    # Penalty on a second bluff face in the same round (one bluff per round).
    INCONSISTENT = 2.0

    # -- learning -----------------------------------------------------------
    # Quick learner: weight of the newest observation in the moving averages.
    LEARN_RATE = 0.4
    # Starting beliefs about a player we do not know.
    PRIOR_EXCESS = {"choice": 0.5, "follow": 0.0}   # extra dice behind a bid
    PRIOR_FOLLOW = 0.5                               # share of raises that follow the face
    PRIOR_CALZA = 0.03
    # Dudo model: P(call) = sigmoid(DUDO_A + DUDO_B * how likely the bid looks
    # to them); learned per player by gradient steps of DUDO_LR.
    DUDO_A, DUDO_B = 2.0, -7.0
    DUDO_LR = 0.5
    # How much of what a bid suggests about the bidder's hand we count when
    # choosing our own bid, and when judging the bid we face.
    HINT_PER_BID = 0.5
    HINT_FACED = 0.75
    # Correct our estimate of a bid we face by how often such bids were
    # really true (learned per bucket); FACED_SHIFT is the measured start.
    FACED_SHIFT = 0.2
    CALIBRATION_PRIOR = 8

    # -- fixes (False restores the old behaviour, for benchmarks) -------------
    # The "choice"/"follow" statistic was trained on held - implied, where
    # implied clips to 0 for ordinary low bids (so the sample was simply the
    # dice held, about 2 for an honest opener), but applied only on high bids
    # as implied + stat: those dice were counted twice. Now the statistic is
    # the residual of bids where implied >= 1, trained and applied there only.
    # Paired, 480 games at tables of 2-6: -0.8 +/- 1.9 pp (within noise).
    FIX_HINT_SAMPLE = True
    # gain_ok is OUR die-gain eligibility; the Calza term of a bid we place
    # is about the NEXT player's Calza, which gains them a die only while
    # nobody is out and THEY hold under five dice.
    # Paired, 480 games: -0.2 +/- 0.6 pp (within noise).
    FIX_OPP_CALZA_GAIN = True

    # Learned state, shared by every instance of this class (per bot name).
    _MEMORY = {}

    @classmethod
    def reset_memory(cls):
        """Forget everything learned (benchmarks use this)."""
        cls._MEMORY = {}

    def __init__(self):
        self.name = "Bluffo Boyyo"
        self._rng = random.Random()
        self.learned = None            # shared: player -> stats, ("faced", bucket) -> [hits, n]
        self._actions = []             # (player, bid or None, "dudo"/"calza"/None) this round
        self._palifico = False
        self._faced_pending = []       # (bid faced, our raw estimate, palifico)
        self._round_no = None
        self._bluff_face = None        # the face we bluffed on this round, if any
        self.mouth = Mouth(self, QUOTES)

    # -- memory -------------------------------------------------------------

    def _ensure_memory(self):
        if self.learned is None:
            self.learned = type(self)._MEMORY.setdefault(self.name, {})

    def _player(self, name):
        self._ensure_memory()
        if name not in self.learned:
            self.learned[name] = {
                "choice": self.PRIOR_EXCESS["choice"], "follow": self.PRIOR_EXCESS["follow"],
                "follow_rate": self.PRIOR_FOLLOW, "calza": self.PRIOR_CALZA,
                "dudo_a": self.DUDO_A, "dudo_b": self.DUDO_B, "bids": 0,
            }
        return self.learned[name]

    def _update(self, stats, key, value):
        stats[key] += self.LEARN_RATE * (value - stats[key])

    # -- API ----------------------------------------------------------------

    def on_game_start(self, info):
        self._ensure_memory()
        self._actions = []
        self._palifico = False
        self._faced_pending = []

    def play(self, state):
        self._ensure_memory()
        if state.round_number != self._round_no:
            self._round_no = state.round_number
            self._bluff_face = None
        options = self._options(state)
        best = max(v for v, _ in options)
        value, action = self._rng.choice([o for o in options if o[0] >= best - self.TIE_EPS])
        if isinstance(action, Bid) and self._bluff_face is None:
            t = self._table(state)
            if self._overstated(state, t, action, self._hints(state, self.HINT_PER_BID)) > 0:
                self._bluff_face = action.face
        self._comment(state, action)
        return action

    def _comment(self, state, action):
        if isinstance(action, Dudo):
            self.mouth.say("dudo", announce="Dudo!", bidder=last_bidder(state))
        elif isinstance(action, Calza):
            self.mouth.say("calza", announce="Calza!", bid=fmt_bid(state.current_bid))
        else:
            t = self._table(state)
            bluff = self._overstated(state, t, action, self._hints(state, self.HINT_PER_BID)) > 0
            key = ("open" if state.current_bid is None else "raise") + ("_bluff" if bluff else "")
            self.mouth.say(key, announce=fmt_bid(action), bid=fmt_bid(action))

    def observe(self, event):
        self._ensure_memory()
        self.mouth.react(event)
        if isinstance(event, BidEvent):
            self._actions.append((event.player, event.bid, None))
        elif _is(event, "DudoEvent", "CalzaEvent"):
            self._actions.append((event.player, None, "dudo" if _is(event, "DudoEvent") else "calza"))
        elif isinstance(event, RevealEvent):
            if not self._actions or self._actions[-1][1] is not None:
                # No Dudo/Calza event seen: the reveal says who called.
                self._actions.append((event.challenger, None, event.challenge_type))
            self._learn(event)
            self._actions = []
        elif isinstance(event, RoundEndedEvent):
            self._palifico = event.palifico_next
        elif isinstance(event, GameEndedEvent):
            self._palifico = False
            self._actions = []
            self._faced_pending = []

    def on_game_end(self, result):
        pass

    # -- reading the table --------------------------------------------------

    def _table(self, state):
        """Static facts of this turn."""
        alive = [p for p in state.players if p.alive]
        total = sum(p.dice_count for p in alive)
        n = len(state.players)
        i = (state.my_id + 1) % n
        while not state.players[i].alive:
            i = (i + 1) % n
        return dict(total=total, unknown=total - state.my_dice_count, alive=len(alive),
                    nxt=state.players[i], w=self.OTHER_WEIGHT / max(len(alive) - 1, 1),
                    gain_ok=len(alive) == len(state.players) and state.my_dice_count < 5,
                    nxt_gain_ok=len(alive) == len(state.players) and state.players[i].dice_count < 5)

    def _hints(self, state, weight):
        """face -> extra matches beyond chance that the others' bids this round suggest."""
        total = sum(p.dice_count for p in state.players if p.alive)
        dice = {p.name: p.dice_count for p in state.players}
        hints = {f: 0.0 for f in range(1, 7)}
        last = {}
        prev = None
        for record in state.history:
            if not isinstance(record.action, Bid):
                break
            bid = record.action
            if record.player != self.name:
                chosen = prev is None or prev.face != bid.face
                last[(record.player, bid.face)] = (bid, chosen)
            prev = bid
        for (who, face), (bid, chosen) in last.items():
            n = dice.get(who, 0)
            if n == 0:
                continue
            p = _p_face(face, state.palifico)
            implied = max(0, min(n, round(bid.quantity - (total - n) * p)))
            if self.FIX_HINT_SAMPLE:
                if implied < 1:
                    continue                 # the quantity says nothing: not in the learned sample
            elif implied <= n * p:
                continue                     # a bid chance alone explains says nothing
            stats = self._player(who)
            expected = max(0.0, min(float(n), implied + stats["choice" if chosen else "follow"]))
            extra = (expected - n * p) * weight
            hints[face] += extra
            if face == PACO and not state.palifico:
                for f in range(2, 7):        # extra Pacos count for every face
                    hints[f] += extra
        return hints

    def _p_true(self, state, t, bid, hints):
        p = _p_face(bid.face, state.palifico)
        need = bid.quantity - _count_matches(state.my_dice, bid.face, state.palifico) - hints[bid.face]
        return _tail_at(t["unknown"], p, need)

    def _p_exact(self, state, t, bid, hints):
        p = _p_face(bid.face, state.palifico)
        need = bid.quantity - _count_matches(state.my_dice, bid.face, state.palifico) - hints[bid.face]
        return _pmf_at(t["unknown"], p, need)

    def _calibrated(self, p):
        """Our estimate of a bid we face, corrected by how often such bids were true."""
        self._ensure_memory()
        key = ("faced", min(int(p * 10), 9))
        stats = self.learned.setdefault(key, [0.0, 0.0])
        prior = min(1.0, p + self.FACED_SHIFT * (1 - p))
        k = self.CALIBRATION_PRIOR
        return (stats[0] + k * prior) / (stats[1] + k)

    def _their_view(self, state, t, bid, hints, who):
        """How likely `bid` looks to `who`: their expected matches (from their
        own bids) plus chance on every die they cannot see, ours included."""
        p = _p_face(bid.face, state.palifico)
        theirs = min(float(who.dice_count), who.dice_count * p + max(0.0, hints[bid.face]))
        return _tail_at(t["total"] - who.dice_count, p, bid.quantity - theirs)

    def _p_challenge(self, who, view):
        s = self._player(who.name)
        return _sigmoid(s["dudo_a"] + s["dudo_b"] * view)

    def _reads(self, state):
        """How much this table reads bids: the alive opponents' learned follow rate."""
        rates = [self._player(p.name)["follow_rate"] for p in state.players
                 if p.alive and p.name != self.name]
        return sum(rates) / len(rates) if rates else self.PRIOR_FOLLOW

    # -- the decision -------------------------------------------------------

    def _overstated(self, state, t, bid, hints):
        """Dice of this face the bid claims for us beyond what our hand and the
        table's bids support (0 = an honest bid)."""
        p = _p_face(bid.face, state.palifico)
        implied = bid.quantity - (t["total"] - state.my_dice_count) * p - hints[bid.face]
        held = _count_matches(state.my_dice, bid.face, state.palifico)
        return max(0.0, min(float(state.my_dice_count), implied) - held)

    def _bid_value(self, state, t, bid, hints, reads):
        pt = self._p_true(state, t, bid, hints)
        nxt = t["nxt"]
        view = self._their_view(state, t, bid, hints, nxt)
        pc = self._p_challenge(nxt, view)
        calza = (1 - pc) * self._player(nxt.name)["calza"]
        pe = self._p_exact(state, t, bid, hints)
        overstated = self._overstated(state, t, bid, hints)
        deceive = self.DECEIVE * reads * overstated / max(state.my_dice_count, 1)
        if overstated > 0 and self._bluff_face is not None and bid.face != self._bluff_face:
            deceive -= self.INCONSISTENT          # a second bluff this round
        challenged = pt * t["w"] - (1 - pt)
        their_gain = t["nxt_gain_ok"] if self.FIX_OPP_CALZA_GAIN else t["gain_ok"]
        calza_out = pe * (-t["w"] if their_gain else 0.0) + (1 - pe) * t["w"]
        passed = -self.LATER_RISK * (1 - pt) + deceive
        return pc * challenged + calza * calza_out + (1 - pc - calza) * passed - self.FALSE_PENALTY * (1 - pt)

    def _options(self, state):
        """(value, action) for every legal action."""
        t = self._table(state)
        hints = self._hints(state, self.HINT_PER_BID)
        reads = self._reads(state)
        options = []
        cur = state.current_bid
        if cur is not None:
            faced = self._hints(state, self.HINT_FACED)
            p_raw = self._p_true(state, t, cur, faced)
            p = self._calibrated(p_raw)
            self._faced_pending.append((cur, p_raw, state.palifico))
            if any(isinstance(a, Dudo) for a in state.legal_actions):
                options.append(((1 - p) * t["w"] - p, Dudo()))
            if any(isinstance(a, Calza) for a in state.legal_actions):
                pe = self._p_exact(state, t, cur, faced)
                gain = 1.0 if t["gain_ok"] else self.CALZA_FREE_PASS
                options.append((pe * gain - (1 - pe), Calza()))
        for a in state.legal_actions:
            if isinstance(a, Bid):
                options.append((self._bid_value(state, t, a, hints, reads), a))
        return options

    # -- learning -----------------------------------------------------------

    def _learn(self, event):
        dice = event.dice_by_player()
        total = sum(len(d) for d in dice.values())
        palifico = self._palifico
        all_dice = [d for hand in dice.values() for d in hand]
        # Were the bids we faced really true, per bucket of our estimate?
        for bid, p_raw, pal in self._faced_pending:
            stats = self.learned.setdefault(("faced", min(int(p_raw * 10), 9)), [0.0, 0.0])
            stats[0] += _count_matches(all_dice, bid.face, pal) >= bid.quantity
            stats[1] += 1
        self._faced_pending = []
        # What they really held behind their last bid on each face, and
        # whether they follow faces.
        last = {}
        prev = None
        for player, bid, _ in self._actions:
            if bid is None:
                break
            if player != self.name and player in dice:
                chosen = prev is None or prev.face != bid.face
                last[(player, bid.face)] = (bid, chosen)
                if prev is not None:
                    self._update(self._player(player), "follow_rate", 0.0 if chosen else 1.0)
            prev = bid
        for (player, face), (bid, chosen) in last.items():
            own = dice[player]
            n = len(own)
            p = _p_face(face, palifico)
            implied = max(0, min(n, round(bid.quantity - (total - n) * p)))
            held = _count_matches(own, face, palifico)
            stats = self._player(player)
            if not (self.FIX_HINT_SAMPLE and implied < 1):
                self._update(stats, "choice" if chosen else "follow", held - implied)
            stats["bids"] += 1
        # How they responded to the bid in front of them: Dudo model and
        # Calza rate, from how likely that bid looked to them.
        prev_bid = None
        for player, bid, call in self._actions:
            if prev_bid is not None and player != self.name and player in dice:
                own = dice[player]
                n = len(own)
                p = _p_face(prev_bid.face, palifico)
                view = _binom_tail(total - n, p, prev_bid.quantity - _count_matches(own, prev_bid.face, palifico))
                stats = self._player(player)
                called = 1.0 if call == "dudo" else 0.0
                pred = _sigmoid(stats["dudo_a"] + stats["dudo_b"] * view)
                stats["dudo_a"] += self.DUDO_LR * (called - pred)
                stats["dudo_b"] += self.DUDO_LR * (called - pred) * (view - 0.5) * 4
                self._update(stats, "calza", 1.0 if call == "calza" else 0.0)
            if bid is None:
                break
            prev_bid = bid
