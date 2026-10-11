"""
Bluffo Boyyo: a quick-learning bluffer.

At the start of every round he flips a coin (BLUFF_RATE): in a bluff round he
pretends to hold BLUFF_DICE extra dice of a face he has few of, and bids as
if that were true; in an honest round he bids on what he really holds.

When he raises he alternates between two styles, one per bid:

  * min raise: the lowest legal bid on each face, taking the one most
    likely to be true (MIN_RAISE_LIKELY; off: the smallest claim, a Paco
    bid counting double); in a bluff round the bluff face when it is
    plausible enough. Paired benchmarks: the smallest claim got caught
    far more (true 27-42% when forced), this +2.3 +/- 1.6 pp win rate
    together with HINT_FACED = 1.0;
  * max raise: the highest bid that is still very likely on the table for
    real (MAX_TARGET, raised against a next player who challenges a lot), by
    preference on a six so the next player must raise the quantity; in a
    bluff round the bluff face wins that preference.

Facing a bid he calls Calza when the exact count is likely: easily while
nobody is out yet and it gains him a die (CALZA_AT_GAIN), only when it beats
a Dudo once it gains nothing (CALZA_AT_FREE). He calls Dudo on a bid he does
not believe (DUDO_AT), judged from his real dice.

After every reveal he learns about each opponent by name, with a fast
moving average (LEARN_RATE): how many dice of the bid face they really hold
behind a bid that chooses or follows a face (their bluffing), how often they
call Dudo or Calza when facing a bid. What he learned stays in memory at
class level, so it survives a new instance per game; nothing is written to
disk. The moving average makes recent games count more than older ones.
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


def _is(event, *names):
    return type(event).__name__ in names


class BluffoBoyyo:
    name = "Bluffo Boyyo"
    count = 1

    # Chance, decided at the start of every round, to bluff this round.
    BLUFF_RATE = 0.5
    # In a bluff round: pretend to hold this many extra dice of the bluff face.
    BLUFF_DICE = 2
    # Max raise: the highest bid still at least this likely (from our view).
    MAX_TARGET = 0.75
    # How much the next player's learned Dudo rate moves that target.
    TARGET_PER_CHALLENGE = 0.4
    # Dudo when the bid we face is less likely than this.
    DUDO_AT = 0.45
    # A min raise less likely than this is not made: Dudo instead.
    MIN_RAISE_P = 0.35
    # Min raise: the most likely of the lowest bids per face (True), or the
    # smallest claim (False).
    MIN_RAISE_LIKELY = True
    # Also Dudo when the raise we would make is less likely to be true than
    # the bid we face is to be false (do not raise on a coin flip).
    DUDO_VS_RAISE = False
    # Calza when the exact count is at least this likely, while it gains a
    # die (nobody out, fewer than 5 dice) / once it gains nothing.
    CALZA_AT_GAIN = 0.30
    CALZA_AT_FREE = 0.45
    # Quick learner: weight of the newest observation in the moving averages.
    LEARN_RATE = 0.4
    # Starting beliefs about a player we do not know: extra dice of the face
    # behind a bid that chooses / follows a face, Dudo and Calza rate facing
    # a bid.
    PRIOR_EXCESS = {"choice": 0.5, "follow": 0.0}
    PRIOR_CHALLENGE = 0.4
    PRIOR_CALZA = 0.03
    # How much of what a bid suggests about the bidder's hand we count when
    # choosing our own raise (1 = all of it), and when judging the bid we
    # face for a Dudo or Calza (bids are more honest than that).
    HINT_PER_BID = 0.5
    HINT_FACED = 1.0

    # Learned state, shared by every instance of this class (per bot name).
    _MEMORY = {}

    @classmethod
    def reset_memory(cls):
        """Forget everything learned (benchmarks use this)."""
        cls._MEMORY = {}

    def __init__(self):
        self.name = "Bluffo Boyyo"
        self._rng = random.Random()
        self.learned = None          # name -> dict of moving averages (shared)
        self._round_no = None
        self._bluffing = False
        self._bluff_face = None
        self._max_next = False       # alternates: next raise is a max raise?
        self._actions = []           # (player, bid or None, "dudo"/"calza"/None) this round
        self._palifico = False

    # -- memory -------------------------------------------------------------

    def _ensure_memory(self):
        if self.learned is None:
            self.learned = type(self)._MEMORY.setdefault(self.name, {})

    def _player(self, name):
        if name not in self.learned:
            self.learned[name] = {
                "choice": self.PRIOR_EXCESS["choice"], "follow": self.PRIOR_EXCESS["follow"],
                "challenge": self.PRIOR_CHALLENGE, "calza": self.PRIOR_CALZA,
                "bluff": 0.3, "bids": 0,
            }
        return self.learned[name]

    def _update(self, stats, key, value):
        stats[key] += self.LEARN_RATE * (value - stats[key])

    # -- API ----------------------------------------------------------------

    def on_game_start(self, info):
        self._ensure_memory()
        self._round_no = None
        self._actions = []
        self._palifico = False

    def play(self, state):
        self._ensure_memory()
        if state.round_number != self._round_no:
            self._start_round(state)
        if state.current_bid is None:
            return self._open(state)
        return self._respond(state)

    def observe(self, event):
        self._ensure_memory()
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

    def on_game_end(self, result):
        pass

    # -- the round ----------------------------------------------------------

    def _start_round(self, state):
        self._round_no = state.round_number
        self._bluffing = (not state.palifico) and self._rng.random() < self.BLUFF_RATE
        self._bluff_face = None
        if self._bluffing:
            # A face we hold few of (never Paco: those are wild).
            counts = {f: _count_matches(state.my_dice, f, False) for f in range(2, 7)}
            least = min(counts.values())
            self._bluff_face = self._rng.choice([f for f, c in counts.items() if c == least])

    def _my_matches(self, state, face, pretend):
        k = _count_matches(state.my_dice, face, state.palifico)
        if pretend and self._bluffing and face == self._bluff_face:
            k += self.BLUFF_DICE
        return k

    def _hints(self, state, weight=None):
        """face -> extra matches beyond chance that the others' bids this round suggest."""
        weight = self.HINT_PER_BID if weight is None else weight
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
            if implied <= n * p:
                continue                     # a bid chance alone explains says nothing
            stats = self._player(who)
            expected = max(0.0, min(float(n), implied + stats["choice" if chosen else "follow"]))
            extra = (expected - n * p) * weight
            hints[face] += extra
            if face == PACO and not state.palifico:
                for f in range(2, 7):        # extra Pacos count for every face
                    hints[f] += extra
        return hints

    def _p_true(self, state, bid, hints, pretend=False):
        total = sum(p.dice_count for p in state.players if p.alive)
        unknown = total - state.my_dice_count
        p = _p_face(bid.face, state.palifico)
        need = bid.quantity - self._my_matches(state, bid.face, pretend) - hints[bid.face]
        lo = math.floor(need)
        frac = need - lo
        return (1 - frac) * _binom_tail(unknown, p, lo) + frac * _binom_tail(unknown, p, lo + 1)

    def _p_exact(self, state, bid, hints):
        total = sum(p.dice_count for p in state.players if p.alive)
        unknown = total - state.my_dice_count
        p = _p_face(bid.face, state.palifico)
        need = bid.quantity - self._my_matches(state, bid.face, False) - hints[bid.face]
        lo = math.floor(need)
        frac = need - lo
        return (1 - frac) * _binom_pmf(unknown, p, lo) + frac * _binom_pmf(unknown, p, lo + 1)

    def _next_player(self, state):
        n = len(state.players)
        i = (state.my_id + 1) % n
        while not state.players[i].alive:
            i = (i + 1) % n
        return state.players[i]

    def _target(self, state):
        """Max-raise target: higher against a next player who challenges a lot."""
        challenge = self._player(self._next_player(state).name)["challenge"]
        t = self.MAX_TARGET + self.TARGET_PER_CHALLENGE * (challenge - self.PRIOR_CHALLENGE)
        return max(0.55, min(0.95, t))

    @staticmethod
    def _strength(bid):
        return bid.quantity * 2 if bid.face == PACO else bid.quantity

    def _min_raise(self, state, bids, hints):
        """The smallest raise: the most likely of the lowest bids per face, or the
        smallest claim; in a bluff round the bluff face when plausible enough."""
        if not self.MIN_RAISE_LIKELY:
            least = min(self._strength(b) for b in bids)
            smallest = [b for b in bids if self._strength(b) == least]
            if self._bluffing and any(b.face == self._bluff_face for b in smallest):
                return next(b for b in smallest if b.face == self._bluff_face)
            return max(smallest, key=lambda b: (self._my_matches(state, b.face, True), -(b.face == PACO), -b.face))
        lowest = {}
        for b in bids:
            if b.face not in lowest or b.quantity < lowest[b.face].quantity:
                lowest[b.face] = b
        if self._bluffing and self._bluff_face in lowest:
            bluff = lowest[self._bluff_face]
            if self._p_true(state, bluff, hints, pretend=True) >= self.MIN_RAISE_P:
                return bluff
        return max(lowest.values(), key=lambda b: (self._p_true(state, b, hints), -self._strength(b)))

    def _max_raise(self, state, bids, hints):
        """The highest bid still likely enough for real; ties go to the bluff face
        in a bluff round, else to a six (the next player must raise the quantity)."""
        target = self._target(state)
        likely = [b for b in bids if self._p_true(state, b, hints) >= target]
        if not likely:
            return None
        preferred = self._bluff_face if self._bluffing else 6
        return max(likely, key=lambda b: (self._strength(b), b.face == preferred, b.face != PACO, b.face))

    def _raise(self, state, bids, hints):
        self._max_next = not self._max_next
        bid = self._max_raise(state, bids, hints) if self._max_next else None
        return bid if bid is not None else self._min_raise(state, bids, hints)

    def _open(self, state):
        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        hints = self._hints(state)
        return self._raise(state, bids, hints)

    def _respond(self, state):
        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        hints = self._hints(state)
        faced = self._hints(state, self.HINT_FACED)
        cur = state.current_bid
        p = self._p_true(state, cur, faced)            # judged from our real dice
        can_dudo = any(isinstance(a, Dudo) for a in state.legal_actions)
        can_calza = any(isinstance(a, Calza) for a in state.legal_actions)

        if can_calza:
            pe = self._p_exact(state, cur, faced)
            gains = all(pl.alive for pl in state.players) and state.my_dice_count < 5
            threshold = self.CALZA_AT_GAIN if gains else self.CALZA_AT_FREE
            if pe >= threshold and (gains or pe > 1 - p):
                return Calza()
        if can_dudo and p < self.DUDO_AT:
            return Dudo()
        if bids:
            bid = self._raise(state, bids, hints)
            p_bid = self._p_true(state, bid, hints)
            if p_bid < self.MIN_RAISE_P and can_dudo and p < 0.5:
                return Dudo()
            if self.DUDO_VS_RAISE and can_dudo and p_bid < 1 - p:
                return Dudo()
            return bid
        return Dudo() if can_dudo else Calza()

    # -- learning -----------------------------------------------------------

    def _learn(self, event):
        dice = event.dice_by_player()
        total = sum(len(d) for d in dice.values())
        palifico = self._palifico
        # What they really held behind their last bid on each face.
        last = {}
        prev = None
        for player, bid, _ in self._actions:
            if bid is None:
                break
            if player != self.name and player in dice:
                chosen = prev is None or prev.face != bid.face
                last[(player, bid.face)] = (bid, chosen)
            prev = bid
        for (player, face), (bid, chosen) in last.items():
            own = dice[player]
            n = len(own)
            p = _p_face(face, palifico)
            implied = max(0, min(n, round(bid.quantity - (total - n) * p)))
            held = _count_matches(own, face, palifico)
            stats = self._player(player)
            self._update(stats, "choice" if chosen else "follow", held - implied)
            self._update(stats, "bluff", 1.0 if held < implied else 0.0)
            stats["bids"] += 1
        # What they did when facing a bid.
        for i, (player, bid, call) in enumerate(self._actions):
            if i == 0 or player == self.name or player not in dice:
                continue
            stats = self._player(player)
            self._update(stats, "challenge", 1.0 if call == "dudo" else 0.0)
            self._update(stats, "calza", 1.0 if call == "calza" else 0.0)
            if bid is None:
                break
