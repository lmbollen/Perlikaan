"""
Pokerface: plays like Sniper, but bluffs about which faces it holds.

Most bots at this table read the bids of their opponents: a player who bids
on a face probably holds some of it. Pokerface uses that against them. When
it is its turn to bid it sometimes picks a face it does not hold:

  * Opening bluff: open the round on a face we have none of. The others
    credit us with dice we don't have and overestimate that face, so they
    raise into it or skip a Dudo they should have called.
  * Precedence bluff: when opponents have already piled onto a face we
    don't hold, join in and bid on it too. That looks like confirmation, so
    the table overestimates the face even further.

A bluff is never reckless: the bid itself must still be likely enough to be
true (BLUFF_TARGET), using the same probability model as Sniper. We only lie
about *where* the dice are, not about how many there are. Bluffing is off in
Palifico rounds, where the face is fixed after the opener anyway.

Reading the others: Sniper credits an opponent with extra dice for every bid
they make on a face. In a chain (5xP, 6xP, 7xP ...) every player is just
raising the previous bid, so that adds up to far more dice than are there
and Sniper keeps raising where it should call Dudo. Pokerface counts a bid
that only raises the quantity of the face just bid at FOLLOW_WEIGHT; a bid
that opens or switches to a face is a real choice and counts in full.

Pokerface learns from every opponent by name (its built-in profiler), without
assuming anything about who they are. After each reveal it replays their
choices and learns:

  * what their bids are worth: how many extra dice of a face they really
    held when they opened on it, switched to it or followed it. Their bids
    are read with those numbers, so an honest bidder is trusted and a
    bluffer is ignored or even read in reverse;
  * how often they call Dudo on a false bid. With ADAPT_TARGET our next bid
    gets bolder against a player who rarely does and more careful against a
    sharp one (off by default: it did not pay off in benchmarks).

A new opponent starts as the average of everyone seen so far, and its own
data takes over as it comes in. `self.profiler.report()` shows what it
learned. ADAPTIVE = False switches the learning off (fixed Sniper reading
with FOLLOW_WEIGHT).
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


# Only types the platform is known to have are imported (the same ones as
# Trihard). Other events are recognised by their class name, so a missing
# class can never break the import.
def _is(event, *names):
    return type(event).__name__ in names


_NORMAL_FACES = (2, 3, 4, 5, 6)


# ---------------------------------------------------------------------------
# Rule helpers (private copies, so this file depends only on perudo.models)
# ---------------------------------------------------------------------------

def _count_matches(dice_values: Sequence[int], face: int, palifico: bool) -> int:
    """How many dice count towards a bid on `face`."""
    if palifico:
        return sum(1 for d in dice_values if d == face)
    if face == PACO:
        return sum(1 for d in dice_values if d == PACO)
    return sum(1 for d in dice_values if d == face or d == PACO)


def _legal_bids(
    current_bid: Optional[Bid],
    palifico: bool,
    palifico_face: Optional[int],
    total_dice: int,
) -> List[Bid]:
    """All legal Bid actions given the current state of a round."""
    max_q = max(total_dice, 1)
    bids: List[Bid] = []

    if current_bid is None:
        # Opening bid of the round.
        faces = range(1, 7) if palifico else _NORMAL_FACES
        for face in faces:
            for q in range(1, max_q + 1):
                bids.append(Bid(q, face))
        return bids

    q0, f0 = current_bid.quantity, current_bid.face

    if palifico:
        # Face is locked for the whole round; only quantity may increase.
        face = palifico_face
        for q in range(q0 + 1, max_q + 1):
            bids.append(Bid(q, face))
        return bids

    # Normal (non-Palifico) round, a bid already exists.
    if f0 == PACO:
        # Paco -> higher Paco.
        for q in range(q0 + 1, max_q + 1):
            bids.append(Bid(q, PACO))
        # Paco -> normal face, quantity must at least double (+1).
        min_q_normal = 2 * q0 + 1
        for q in range(min_q_normal, max_q + 1):
            for face in _NORMAL_FACES:
                bids.append(Bid(q, face))
    else:
        # Same quantity, strictly higher normal face.
        for face in range(f0 + 1, 7):
            bids.append(Bid(q0, face))
        # Higher quantity, any normal face (including the same one).
        for q in range(q0 + 1, max_q + 1):
            for face in _NORMAL_FACES:
                bids.append(Bid(q, face))
        # Normal -> Paco, quantity at least ceil(q0 / 2).
        min_q_paco = math.ceil(q0 / 2)
        for q in range(min_q_paco, max_q + 1):
            bids.append(Bid(q, PACO))

    return bids


# ---------------------------------------------------------------------------
# Opponent profiling (self-contained copy: this bot depends on no other file
# than the game framework). Learns per opponent name from every reveal.
# ---------------------------------------------------------------------------

# Minimum number of observations before a label is given instead of "?".
MIN_SAMPLES = 15

_tail_cache: Dict[Tuple[int, float, int], float] = {}


def p_at_least(n: int, p: float, need: int) -> float:
    """P(Binomial(n, p) >= need)."""
    if need <= 0:
        return 1.0
    if need > n:
        return 0.0
    key = (n, p, need)
    if key not in _tail_cache:
        _tail_cache[key] = sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(need, n + 1))
    return _tail_cache[key]


def p_bid_true(bid: Bid, own_dice: Sequence[int], total_dice: int, palifico: bool = False) -> float:
    """P(bid holds | own dice), every unknown die independent and uniform."""
    p = 1 / 6 if (palifico or bid.face == PACO) else 1 / 3
    need = bid.quantity - _count_matches(own_dice, bid.face, palifico)
    return p_at_least(total_dice - len(own_dice), p, need)


def _strength(bid: Bid) -> float:
    """
    Claim size so Paco and normal bids compare (a Paco counts double). The
    face is ignored: a min-raise is a raise without a jump in quantity.
    """
    return 2 * bid.quantity if bid.face == PACO else bid.quantity


class Profile:
    """What we learned about one opponent. Plain numbers, easy to inspect.

    A plain class rather than a dataclass: the bot loader runs this file
    without registering it as a module, which dataclasses cannot handle."""

    def __init__(self, name):
        self.name = name
        self.rounds = 0
        self.bids = 0
        self.face_scores: List[float] = []
        self.follows = 0
        self.raises = 0
        self.min_raises = 0
        self.bid_risk_sum = 0.0
        # (P(bid faced is true | own dice), called Dudo?) per decision facing a bid.
        self.facing: List[Tuple[float, bool]] = []
        self.calzas = 0
        # What their bids say about their hand: per kind of bid ("nat-choice",
        # "nat-follow", "paco-choice", "paco-follow") the summed excess of that
        # face in their hand over what chance gives, and the number of samples.
        self.excess: Dict[str, List[float]] = {}
        # How often they call Dudo on a bid that is really false (a choice, not
        # forced): the risk of putting a false bid in front of them.
        self.false_faced = 0
        self.false_challenged = 0

    def decay(self, factor: float) -> None:
        """Make what was learned count `factor` times as much (once per game),
        so recent games weigh more than older ones."""
        for attr in ("rounds", "bids", "follows", "raises", "min_raises", "bid_risk_sum",
                     "calzas", "false_faced", "false_challenged"):
            setattr(self, attr, getattr(self, attr) * factor)
        for stat in self.excess.values():
            stat[0] *= factor
            stat[1] *= factor
        # Lists of single decisions: drop the same share of the oldest ones.
        for seq in (self.face_scores, self.facing):
            del seq[:int(len(seq) * (1 - factor))]

    # -- derived numbers ------------------------------------------------

    @property
    def face_score(self) -> Optional[float]:
        return sum(self.face_scores) / len(self.face_scores) if self.face_scores else None

    @property
    def follow_rate(self) -> Optional[float]:
        return self.follows / self.raises if self.raises else None

    @property
    def min_raise_rate(self) -> Optional[float]:
        return self.min_raises / self.raises if self.raises else None

    @property
    def bid_risk(self) -> Optional[float]:
        return self.bid_risk_sum / self.bids if self.bids else None

    @property
    def calza_rate(self) -> Optional[float]:
        n = len(self.facing) + self.calzas
        return self.calzas / n if n else None

    @property
    def dudo_rate(self) -> Optional[float]:
        return sum(d for _, d in self.facing) / len(self.facing) if self.facing else None

    def dudo_separation(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Mean P(true) of the bids they raised on minus that of the bids they
        called Dudo on, and the standard error of that difference. Near 0:
        their Dudos don't depend on what they hold.
        """
        dudo = [p for p, d in self.facing if d]
        rais = [p for p, d in self.facing if not d]
        if len(dudo) < 2 or len(rais) < 2:
            return None, None

        def mean_var(xs):
            m = sum(xs) / len(xs)
            return m, sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

        md, vd = mean_var(dudo)
        mr, vr = mean_var(rais)
        return mr - md, math.sqrt(vd / len(dudo) + vr / len(rais))

    def dudo_threshold(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Best single threshold t for "Dudo iff P(bid true) < t", and the share
        of decisions it explains. (None, None) without data.
        """
        if not self.facing:
            return None, None
        pts = sorted(self.facing)
        n = len(pts)
        # t below every point: everything predicted "raise", so the errors
        # are exactly the Dudos. Sweep t upwards through the points.
        errors = sum(d for _, d in pts)
        best_err, best_t = errors, 0.0
        i = 0
        while i < n:
            p = pts[i][0]
            # Move every point with this p to the "Dudo" side.
            while i < n and pts[i][0] == p:
                errors += -1 if pts[i][1] else 1
                i += 1
            nxt = pts[i][0] if i < n else 1.0
            if errors < best_err:
                best_err, best_t = errors, (p + nxt) / 2
        return best_t, 1 - best_err / n

    def _dudo_follows_dice(self) -> bool:
        """Do they call Dudo on clearly less likely bids than they raise on?"""
        sep, se = self.dudo_separation()
        return sep is not None and sep >= 0.15 and sep >= 3 * se

    # -- categories -----------------------------------------------------

    def labels(self) -> Dict[str, str]:
        """One category per axis; "?" when there is not enough data yet."""
        out = {}

        fs = self.face_score
        if fs is None or len(self.face_scores) < MIN_SAMPLES:
            out["faces"] = "?"
        elif fs >= 0.6:
            out["faces"] = "honest"      # bids what it holds
        elif fs >= 0.25:
            out["faces"] = "mixed"
        elif fs >= -0.2:
            out["faces"] = "random"      # face choice unrelated to its dice
        else:
            out["faces"] = "bluffer"     # prefers faces it does not hold

        risk = self.bid_risk
        if risk is None or self.bids < MIN_SAMPLES:
            out["bids"] = "?"
        elif risk >= 0.7:
            out["bids"] = "cautious"
        elif risk >= 0.45:
            out["bids"] = "normal"
        else:
            out["bids"] = "aggressive"

        t, fit = self.dudo_threshold()
        if t is None or len(self.facing) < MIN_SAMPLES:
            out["dudo"] = "?"
        elif self.dudo_rate == 0:
            out["dudo"] = "never-dudo"
        elif fit < 0.75 or not self._dudo_follows_dice():
            # No single threshold explains it, or Dudos and raises happen at
            # the same odds: the choice doesn't follow from their own dice.
            out["dudo"] = "unpredictable"
        elif t >= 0.55:
            out["dudo"] = "dudo-happy"
        elif t >= 0.3:
            out["dudo"] = "normal"
        else:
            out["dudo"] = "dudo-shy"

        mr = self.min_raise_rate
        if mr is None or self.raises < MIN_SAMPLES:
            out["raise"] = "?"
        elif mr >= 0.85:
            out["raise"] = "min-raiser"
        elif mr >= 0.4:
            out["raise"] = "steady"
        else:
            out["raise"] = "jumper"

        cr = self.calza_rate
        out["calza"] = "?" if cr is None else ("calza-user" if self.calzas >= 5 else "no-calza")
        return out

    def summary(self) -> str:
        def f(x, fmt="{:.2f}"):
            return "-" if x is None else fmt.format(x)

        t, fit = self.dudo_threshold()
        return (
            f"{self.name:<14} rondes {self.rounds:4.0f}  face {f(self.face_score):>5}  "
            f"follow {f(self.follow_rate):>4}  minraise {f(self.min_raise_rate):>4}  "
            f"risk {f(self.bid_risk):>4}  dudo<{f(t):>4} (fit {f(fit):>4})  "
            f"calza {f(self.calza_rate):>4}"
        )


def bid_kind(face: int, chosen: bool) -> str:
    return f"{'paco' if face == PACO else 'nat'}-{'choice' if chosen else 'follow'}"


class OpponentProfiler:
    """
    Builds a Profile per opponent. Feed it with `observe(event)` from a bot's
    observe(), or directly with `record_round(...)` from logged rounds.

    Estimates for decisions (`bid_excess`, `challenge_rate`) are shrunk
    towards the average over every opponent seen so far, which is in turn
    shrunk towards a fixed prior. A new opponent therefore starts out as "an
    average player" and becomes itself as the evidence comes in.
    """

    # Starting beliefs before any data: extra natural dice of a face held
    # per bid on it, and the chance a false bid gets challenged.
    PRIOR_EXCESS = {"nat-choice": 0.5, "nat-follow": 0.15, "paco-choice": 0.3, "paco-follow": 0.1}
    PRIOR_CHALLENGE = 0.5
    PRIOR_CALZA = 0.05
    # Weight of the prior / pooled estimate, in observations.
    SHRINK = 10

    def __init__(self, me):
        self.me = me
        self.profiles: Dict[str, Profile] = {}
        self._actions: List[Tuple[str, object]] = []
        self._dice: Optional[Dict[str, Tuple[int, ...]]] = None
        self._palifico = False

    # `me` is the owner's name: a string, or the owning bot itself, whose
    # current name is read at use. The loader renames extra copies of a bot
    # after construction, so a name captured in __init__ went stale and a
    # second copy profiled itself.
    @property
    def me(self):
        owner = self._owner
        return owner if owner is None or isinstance(owner, str) else owner.name

    @me.setter
    def me(self, owner):
        self._owner = owner

    def decay(self, factor: float) -> None:
        """Older games count less: scale every profile by `factor`."""
        for prof in self.profiles.values():
            prof.decay(factor)

    def profile(self, name: str) -> Profile:
        if name not in self.profiles:
            self.profiles[name] = Profile(name)
        return self.profiles[name]

    # -- estimates for decisions ------------------------------------------

    def _shrunk(self, own_sum, own_n, pooled_sum, pooled_n, prior):
        pooled = (pooled_sum + self.SHRINK * prior) / (pooled_n + self.SHRINK)
        return (own_sum + self.SHRINK * pooled) / (own_n + self.SHRINK)

    def bid_excess(self, name: str, kind: str) -> float:
        """Extra natural dice of a face `name` holds when making a bid of `kind`."""
        stats = [p.excess.get(kind, (0.0, 0)) for p in self.profiles.values()]
        own = self.profiles[name].excess.get(kind, (0.0, 0)) if name in self.profiles else (0.0, 0)
        return self._shrunk(own[0], own[1], sum(s for s, _ in stats), sum(n for _, n in stats),
                            self.PRIOR_EXCESS[kind])

    def challenge_rate(self, name: Optional[str] = None) -> float:
        """P(`name` calls Dudo on a false bid); the table average without a name."""
        faced = sum(p.false_faced for p in self.profiles.values())
        hit = sum(p.false_challenged for p in self.profiles.values())
        prof = self.profiles.get(name) if name else None
        if prof is None:
            return self._shrunk(0, 0, hit, faced, self.PRIOR_CHALLENGE)
        return self._shrunk(prof.false_challenged, prof.false_faced, hit, faced, self.PRIOR_CHALLENGE)

    def calza_rate(self, name: Optional[str] = None) -> float:
        """P(`name` calls Calza when facing a bid); the table average without a name."""
        calls = sum(p.calzas for p in self.profiles.values())
        chances = calls + sum(len(p.facing) for p in self.profiles.values())
        prof = self.profiles.get(name) if name else None
        if prof is None:
            return self._shrunk(0, 0, calls, chances, self.PRIOR_CALZA)
        return self._shrunk(prof.calzas, prof.calzas + len(prof.facing), calls, chances, self.PRIOR_CALZA)

    # -- event stream ---------------------------------------------------

    def observe(self, event):
        if isinstance(event, BidEvent):
            self._actions.append((event.player, event.bid))
        elif _is(event, "DudoEvent"):
            self._actions.append((event.player, Dudo()))
        elif _is(event, "CalzaEvent"):
            self._actions.append((event.player, Calza()))
        elif isinstance(event, RevealEvent):
            if not self._actions or isinstance(self._actions[-1][1], Bid):
                # No Dudo/Calza event seen: the reveal says who called.
                call = Dudo() if event.challenge_type == "dudo" else Calza()
                self._actions.append((event.challenger, call))
            self.record_round(event.dice_by_player(), self._actions, self._palifico)
            self._actions = []
        elif isinstance(event, RoundEndedEvent):
            self._palifico = event.palifico_next
        elif isinstance(event, GameEndedEvent):
            self._palifico = False
            self._actions = []

    # -- core -----------------------------------------------------------

    def record_round(
        self,
        dice: Dict[str, Sequence[int]],
        actions: Iterable[Tuple[str, object]],
        palifico: bool,
    ) -> None:
        """Replay one revealed round from every opponent's point of view."""
        if palifico:
            return
        total = sum(len(d) for d in dice.values())
        all_dice = [d for hand in dice.values() for d in hand]
        seen = set()
        # Per player: the faces they bid, and the ones they chose (opened or
        # switched to) rather than followed.
        bid_faces: Dict[str, set] = {}
        chosen_faces: Dict[str, set] = {}
        prev: Optional[Bid] = None
        for player, action in actions:
            if player == self.me or player not in dice:
                if isinstance(action, Bid):
                    prev = action
                continue
            own = dice[player]
            prof = self.profile(player)
            if player not in seen:
                prof.rounds += 1
                seen.add(player)

            if isinstance(action, Calza):
                prof.calzas += 1
                continue
            options = _legal_bids(prev, False, None, total) if prev is not None else []
            # A Dudo is only a choice when a raise was possible.
            if prev is not None and options:
                p_cur = p_bid_true(prev, own, total)
                prof.facing.append((p_cur, isinstance(action, Dudo)))
                if _count_matches(all_dice, prev.face, False) < prev.quantity:
                    prof.false_faced += 1
                    prof.false_challenged += isinstance(action, Dudo)
            if not isinstance(action, Bid):
                continue

            prof.bids += 1
            prof.bid_risk_sum += p_bid_true(action, own, total)
            bid_faces.setdefault(player, set()).add(action.face)
            if prev is None or action.face != prev.face:
                chosen_faces.setdefault(player, set()).add(action.face)
                self._score_face(prof, action, own)
            if prev is not None:
                prof.raises += 1
                prof.follows += action.face == prev.face
                if options and _strength(action) <= min(_strength(b) for b in options):
                    prof.min_raises += 1
            prev = action

        # Once per player and face: how many more of that face they really
        # held than chance. Natural dice only, so a face and Paco never
        # count the same die twice.
        for player, faces in bid_faces.items():
            own = dice[player]
            for face in faces:
                key = bid_kind(face, face in chosen_faces.get(player, ()))
                stat = self.profile(player).excess.setdefault(key, [0.0, 0])
                stat[0] += sum(1 for d in own if d == face) - len(own) / 6
                stat[1] += 1

    @staticmethod
    def _score_face(prof: Profile, bid: Bid, own: Sequence[int]) -> None:
        """How good a pick was this face, given their dice? Skips Paco bids."""
        if bid.face == PACO:
            return
        counts = [_count_matches(own, f, False) for f in range(2, 7)]
        best, mean = max(counts), sum(counts) / len(counts)
        if best == mean:
            return  # every face equally good: no information
        prof.face_scores.append((_count_matches(own, bid.face, False) - mean) / (best - mean))

    # -- reporting --------------------------------------------------------

    def report(self) -> str:
        lines = []
        for name in sorted(self.profiles):
            prof = self.profiles[name]
            labels = prof.labels()
            lines.append(prof.summary())
            lines.append(" " * 15 + "-> " + ", ".join(f"{k}: {v}" for k, v in labels.items()))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared memory: learned state lives at class level, so a new instance
# per game (same process) keeps what was learned. No disk access.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sniper's engine (self-contained copy of perudo/bots/sniper.py, without play())
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pokerface
# ---------------------------------------------------------------------------

class PokerfaceBot:
    name = "Pokerface"
    count = 1

    # -- shared memory: learned state per bot name, kept at class level --------

    SHARED_MEMORY = True

    @classmethod
    def reset_memory(cls) -> None:
        """Forget everything instances of this class learned (benchmarks use this)."""
        cls._MEMORY = {}

    def _shared_ensure_memory(self) -> None:
        if getattr(self, "_memory_bound", False):
            return
        self._memory_bound = True
        if not self.SHARED_MEMORY:
            return
        cls = type(self)
        if "_MEMORY" not in cls.__dict__:
            cls._MEMORY = {}
        store = cls._MEMORY.setdefault(self.name, {})
        for attr in self._LEARNED:
            if attr in store:
                setattr(self, attr, store[attr])
            else:
                store[attr] = getattr(self, attr)

    # -- Sniper's engine -------------------------------------------------------


    # Tuned by simulation against the other bots in this folder: targets
    # between 0.75 and 0.9 do about equally well, 0.55 and lower do badly.
    #
    # Minimum P(true) for a bid we place. Lower is more aggressive.
    TARGET = 0.85
    # Below this no raise is acceptable and we call Dudo instead.
    MIN_ACCEPT = 0.3
    # Extra dice of the bid face we credit an opponent for, per bid they made
    # on that face, capped by how many dice they actually hold.
    HINT_PER_BID = 0.7
    # Weight of a Paco bid as a hint for normal faces (Pacos are wild).
    PACO_HINT_WEIGHT = 0.4

    def _sniper_init(self):
        self.name = "Sniper"
        self._tail_cache = {}

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


    def _prob_true(self, state, bid, n_unknown, hints):
        mine = _count_matches(state.my_dice, bid.face, state.palifico)
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

    def _sniper_observe(self, event):
        pass

    # -- Pokerface -------------------------------------------------------------

    # Learned state kept in shared memory (survives a new instance per game).
    _LEARNED = ("profiler", "_calibration", "_thr_cache")

    # Chance to bluff when a bluff is available. Below 1 keeps us
    # unpredictable: an opponent can't assume every bid of ours is a lie.
    BLUFF_RATE = 0.5
    # Minimum P(true) for a bluff bid. A bit below Sniper's TARGET: the
    # misdirection is worth some of the risk.
    BLUFF_TARGET = 0.75
    # Number of different opponents that bid a face this round before we
    # call it "strong precedence" and join in.
    PRECEDENCE = 2
    # Weight of a bid that stays on the face of the bid before it, relative
    # to a bid that opens or switches to a face. Only used while not ADAPTIVE.
    FOLLOW_WEIGHT = 0.3
    # Learn from the opponents at the table instead of fixed assumptions:
    #  * how much each opponent's bids say about their hand (bid_excess),
    #  * how likely the next player is to call Dudo on a false bid, which
    #    moves our TARGET up or down.
    ADAPTIVE = True
    # Moving the bid target with the next player's challenge rate. Off: in
    # benchmark_adaptive.py it was worth nothing (-0.2 +/- 0.7 pp win rate,
    # 120 tables), while learned bid reading alone gave +2.4 +/- 0.9 pp.
    ADAPT_TARGET = False
    # Bounds for the adapted TARGET.
    TARGET_RANGE = (0.6, 0.97)
    # Learn how often the bids we face are really true, per estimate bucket
    # and table size, and decide Dudo on that instead of the raw estimate.
    CALIBRATE = True
    # Weight of the raw estimate as a prior, in observations.
    CALIBRATION_PRIOR = 20
    # Call Calza ourselves when it beats Dudo and raising in expected dice;
    # while a correct Calza gains a die, also P(exact) >= CALZA_AT.
    CALZA = True
    CALZA_AT = 0.4
    # Value of a correct Calza once it gains no die (someone is out, or we
    # have 5): nobody loses a die and we start the next round.
    CALZA_FREE_PASS = 0.3
    # Consider Calza up to this many players left (it is judged on expected
    # value, so at big tables with many dice it rarely wins anyway).
    CALZA_PLAYERS = 7
    # Avoid bids that the next player is likely to Calza correctly: skip a
    # bid when P(exact) * their learned Calza rate exceeds CALZA_RISK.
    AVOID_CALZA = True
    CALZA_RISK = 0.1
    # Pick bids by a one-step lookahead (in the endgame, see ENDGAME_PLAYERS)
    # instead of "strongest bid that is TARGET likely": for every hand the next player may hold, how likely is
    # our bid (to us) and how likely does it look to them? With their learned
    # Dudo threshold and Calza rate that gives the expected dice of each bid.
    # Bids that are true but look false to them (traps) score best; bids
    # that hand them an easy Dudo or Calza score worst.
    LOOKAHEAD = True
    # Only bids at least this likely (to us) are considered.
    LOOKAHEAD_MIN = 0.3
    # How sharp an opponent's Dudo threshold is (logistic width).
    DUDO_SOFTNESS = 0.02
    # Dudo threshold assumed for an opponent we know nothing about yet.
    DEFAULT_DUDO_AT = 0.3
    # CALIBRATE, AVOID_CALZA, LOOKAHEAD and DUDO_EV only kick in once at most
    # this many players are left (Calza itself is gated by CALZA_PLAYERS).
    # They win big head-to-head and cost a little at bigger tables
    # (benchmark_adaptive.py / sweep_players.py).
    ENDGAME_PLAYERS = 2
    # Dudo when it loses less often than raising: a Dudo loses when the bid
    # is true, a raise only when it is false AND the next player challenges
    # it (their learned challenge rate). Only in the endgame (ENDGAME_PLAYERS):
    # head-to-head it adds +3.8 pp, at 5 players it costs 5.2 pp (there an
    # unchallenged raise still tends to come back). Otherwise Sniper's rule:
    # Dudo when P(bid true) < P(best raise false).
    DUDO_EV = True
    # In the Calza comparison ev_raise counted only the raise's downside
    # (challenged and false), while ev_dudo and ev_calza both include the
    # upside of an opponent losing a die. Add the raise's upside: a true
    # raise that still gets challenged costs the challenger a die.
    # Paired, 480 games at tables of 2-7: +0.2 +/- 0.8 pp (within noise).
    FIX_RAISE_UPSIDE = True
    # In Palifico a follower could have called Dudo instead of raising, so
    # their raise says something about their hand too (Sniper ignores it).
    PALIFICO_FOLLOWERS = True
    # Starting values, measured with measure_defaults.py over many games at
    # tables of 2-7 players (static bots and The Don): a single game leaves no
    # time to warm up. Learning in the game starts from these.
    TUNED_DEFAULTS = True
    TUNED_EXCESS = {"nat-choice": 0.80, "nat-follow": 0.67, "paco-choice": 0.39, "paco-follow": 0.43}
    TUNED_CHALLENGE = 0.37
    TUNED_CALZA = 0.025
    TUNED_DUDO_AT = 0.20
    # Set to a list to record every decision with what we believed at the
    # time (see analyze_endgame.py). None = no logging.
    decision_log = None

    # Once per game, everything learned is multiplied by this, so the latest
    # games count more than older ones (0.95: a game 14 games ago counts half).
    GAME_DECAY = 0.95

    def __init__(self):
        self._sniper_init()
        self.name = "Pokerface"
        self._rng = random.Random(0x90C3)   # fixed seed: reproducible games
        # Learns per opponent name; lives as long as this instance, so
        # across all games of a simulation. Gets the bot, not its name: the
        # loader renames extra copies after construction.
        self.profiler = OpponentProfiler(self)
        # (table size class, estimate bucket) -> [times true, times seen]
        self._calibration = {}
        self._pending = []   # (key, bid, palifico) of bids faced this round
        self._thr_cache = {}

    # -- reading the others ------------------------------------------------

    def _hint(self, state, face):
        """Extra dice of `face` we expect among opponents, based on their bids."""
        if self.ADAPTIVE and not state.palifico:
            return self._learned_hint(state, face)
        return self._static_hint(state, face)

    def _learned_hint(self, state, face):
        """
        Like _static_hint, but every opponent's bid is worth what their bids
        were worth in earlier rounds (OpponentProfiler.bid_excess): a lot for
        an honest bidder, nothing or less than nothing for a bluffer.
        """
        return sum(self._hint_by_player(state, face).values())

    def _hint_by_player(self, state, face):
        """Per opponent: extra dice of `face` we credit them for their bids."""
        bid_faces, chosen = {}, {}
        prev = None
        for rec in state.history:
            if not isinstance(rec.action, Bid):
                continue
            bid = rec.action
            if rec.player != self.name:
                bid_faces.setdefault(rec.player, set()).add(bid.face)
                if prev is None or prev.face != bid.face:
                    chosen.setdefault(rec.player, set()).add(bid.face)
            prev = bid

        dice_of = {p.name: p.dice_count for p in state.players}
        p = self._p_face(face, False)
        out = {}
        for who, faces in bid_faces.items():
            extra = 0.0
            # Natural dice of the face itself, and Pacos (wild for any face).
            for f in {face, PACO} & faces:
                extra += self.profiler.bid_excess(who, bid_kind(f, f in chosen.get(who, ())))
            n = dice_of.get(who, 0)
            # They hold between 0 and n matches: expectation is n * p.
            out[who] = min(max(extra, -n * p), n * (1 - p))
        return out

    def _static_hint(self, state, face):
        """Extra dice of `face` we expect among opponents, based on their bids."""
        # Sniper: in Palifico only the opener chose the face; later bids are
        # forced. Unless PALIFICO_FOLLOWERS: raising instead of Dudo is a choice.
        history = state.history[:1] if (state.palifico and not self.PALIFICO_FOLLOWERS) else state.history
        dice_of = {p.name: p.dice_count for p in state.players}

        weight_of = {}
        prev = None
        for rec in history:
            if not isinstance(rec.action, Bid):
                continue
            bid = rec.action
            if rec.player != self.name:
                weight = 0.0
                if bid.face == face:
                    weight = 1.0
                elif not state.palifico and bid.face == PACO and face != PACO:
                    weight = self.PACO_HINT_WEIGHT
                if prev is not None and prev.face == bid.face:
                    weight *= self.FOLLOW_WEIGHT
                weight_of[rec.player] = weight_of.get(rec.player, 0.0) + weight
            prev = bid

        total = 0.0
        for who, weight in weight_of.items():
            # Cap: they can't hold more than their dice beyond expectation.
            cap = dice_of.get(who, 0) * (1 - self._p_face(face, state.palifico))
            total += min(weight * self.HINT_PER_BID, cap)
        return total

    # -- bluffing ---------------------------------------------------------

    @staticmethod
    def _missing_faces(state):
        """Normal faces of which we hold no natural die."""
        return [f for f in range(2, 7) if f not in state.my_dice]

    def _best_on_face(self, state, bids, face, n_unknown, hints, target):
        """Strongest legal bid on `face` that clears `target`, or None."""
        cands = [
            (self._prob_true(state, b, n_unknown, hints), b)
            for b in bids
            if b.face == face
        ]
        cands = [(pt, b) for pt, b in cands if pt >= target]
        if not cands:
            return None
        return max(cands, key=lambda s: (self._strength(s[1], state.palifico), s[0]))

    def _bluff(self, state, bids, n_unknown, hints, target):
        """A bluff bid and its kind ("bluff_open" / "bluff_follow"), or (None, None)."""
        if state.palifico or self._rng.random() >= self.BLUFF_RATE:
            return None, None
        missing = self._missing_faces(state)
        if not missing:
            return None, None

        if state.current_bid is None:
            faces, key = missing, "bluff_open"
        else:
            backers = {
                f: {r.player for r in state.history
                    if r.player != self.name and isinstance(r.action, Bid) and r.action.face == f}
                for f in missing
            }
            faces = [f for f in missing if len(backers[f]) >= self.PRECEDENCE]
            key = "bluff_follow"
        self._rng.shuffle(faces)
        for face in faces:
            found = self._best_on_face(state, bids, face, n_unknown, hints, target)
            if found is not None:
                return found, key
        return None, None

    # -- decision ---------------------------------------------------------

    @staticmethod
    def _next_player(state):
        """Name of the player who will face our bid."""
        seats = list(state.players)
        me = next(i for i, p in enumerate(seats) if p.id == state.my_id)
        for step in range(1, len(seats)):
            p = seats[(me + step) % len(seats)]
            if p.alive:
                return p.name
        return None

    def _target(self, state, base):
        """
        Minimum P(true) for our bid. A false bid only costs a die when it is
        challenged, so in front of a player who rarely challenges false bids
        we may bid more boldly, and in front of a sharp one more carefully:
        we keep P(false) * P(challenged) where it is for an average player.
        """
        if not (self.ADAPTIVE and self.ADAPT_TARGET):
            return base
        rate = self.profiler.challenge_rate(self._next_player(state))
        avg = self.profiler.challenge_rate()
        lo, hi = self.TARGET_RANGE
        return min(max(1 - (1 - base) * avg / max(rate, 0.05), lo), hi)

    def _ensure_memory(self):
        """Apply the starting values once, then bind to the shared memory."""
        if not getattr(self, "_defaults_applied", False):
            self._defaults_applied = True
            if self.TUNED_DEFAULTS:
                self.profiler.PRIOR_EXCESS = dict(self.TUNED_EXCESS)
                self.profiler.PRIOR_CHALLENGE = self.TUNED_CHALLENGE
                self.profiler.PRIOR_CALZA = self.TUNED_CALZA
                self.DEFAULT_DUDO_AT = self.TUNED_DUDO_AT
        self._shared_ensure_memory()
        self.profiler.me = self   # a profiler from shared memory follows this instance's name

    def play(self, state):
        self._ensure_memory()
        total = sum(p.dice_count for p in state.players if p.alive)
        n_unknown = total - state.my_dice_count
        hints = {face: self._hint(state, face) for face in range(1, 7)}
        target = self._target(state, self.TARGET)

        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        scored = [(self._prob_true(state, b, n_unknown, hints), b) for b in bids]

        safe = [(pt, b) for pt, b in scored if pt >= target]
        endgame = sum(p.alive for p in state.players) <= self.ENDGAME_PLAYERS
        if safe and self.AVOID_CALZA and endgame:
            rate = self.profiler.calza_rate(self._next_player(state))
            low_risk = [(pt, b) for pt, b in safe
                        if rate * self._prob_exact(state, b, n_unknown, hints) <= self.CALZA_RISK]
            safe = low_risk or safe
        if self.LOOKAHEAD and endgame and not state.palifico and scored:
            picked = self._lookahead_pick(state, scored, n_unknown, hints)
            if picked is not None:
                safe = [picked]
        if safe:
            best_p, best = max(safe, key=lambda s: (self._strength(s[1], state.palifico), s[0]))
        elif scored:
            best_p, best = max(scored, key=lambda s: s[0])
        else:
            best_p, best = 0.0, None

        cur = state.current_bid
        p_cur = None
        if cur is not None:
            p_cur = self._calibrated(state, cur, self._prob_true(state, cur, n_unknown, hints))
            alive = sum(p.alive for p in state.players)
            if self.CALZA and alive <= self.CALZA_PLAYERS and Calza() in state.legal_actions:
                p_ex = self._calibrated(state, cur, self._prob_exact(state, cur, n_unknown, hints), exact=True)
                gains = self._calza_gains(state)
                # Expected dice for us: a correct Calza +1 (or a free pass once
                # it no longer gains a die: nobody loses, we start), a wrong one
                # -1; Dudo: the bidder losing is worth 1/(opponents), us -1; a
                # raise loses when it is challenged and false, and wins the
                # same share when it is challenged and true. With few dice
                # P(exact) is high and Calza wins this comparison more often.
                w = 1.0 / max(alive - 1, 1)
                ev_calza = p_ex * (1.0 if gains else self.CALZA_FREE_PASS) - (1 - p_ex)
                ev_dudo = (1 - p_cur) * w - p_cur
                challenge = self.profiler.challenge_rate(self._next_player(state))
                if best is None:
                    ev_raise = -math.inf
                else:
                    ev_raise = -(1 - best_p) * challenge
                    if self.FIX_RAISE_UPSIDE:
                        # The profiler only learns the challenge rate on
                        # false bids; it stands in for the rate on true ones.
                        ev_raise += best_p * challenge * w
                if (p_ex >= self.CALZA_AT or not gains) and ev_calza > max(ev_dudo, ev_raise):
                    self._log(state, Calza(), p_cur, best, best_p, hints, False)
                    return Calza()
            if self.DUDO_EV and sum(p.alive for p in state.players) <= self.ENDGAME_PLAYERS:
                challenge = self.profiler.challenge_rate(self._next_player(state))
                call = best is None or p_cur < (1 - best_p) * challenge
            else:
                call = best is None or best_p < self.MIN_ACCEPT or p_cur < 1 - best_p
            if call:
                self._log(state, Dudo(), p_cur, best, best_p, hints, False)
                return Dudo()

        # We are going to bid. Decide whether to lie about the face.
        bluff, _ = self._bluff(state, bids, n_unknown, hints, self._target(state, self.BLUFF_TARGET))
        if bluff is not None:
            _, best = bluff
        self._log(state, best, p_cur, best, self._prob_true(state, best, n_unknown, hints), hints,
                  bluff is not None)
        return best

    # -- lookahead ----------------------------------------------------------

    def _dudo_at(self, name):
        """Learned Dudo threshold of `name` on the own-dice scale, cached."""
        prof = self.profiler.profiles.get(name)
        if prof is None or len(prof.facing) < 20:
            return self.DEFAULT_DUDO_AT
        cache = self._thr_cache.get(name)
        if cache is None or len(prof.facing) - cache[0] >= 25:
            t, _ = prof.dudo_threshold()
            cache = (len(prof.facing), t)
            self._thr_cache[name] = cache
        return cache[1]

    def _bid_ev(self, state, bid, n_unknown, hints, nxt, dice_next, h_next, t_next, calza_next, gain_next):
        """Expected dice for us of putting `bid` in front of the next player."""
        p = self._p_face(bid.face, False)
        total = n_unknown + state.my_dice_count
        mine = _count_matches(state.my_dice, bid.face, False)
        h_rest = hints[bid.face] - h_next
        m = n_unknown - dice_next          # unknown dice outside the next player
        # Their matches: binomial, nudged by what their own bids told us.
        q_next = min(max(p + (h_next / dice_next if dice_next else 0), 0.01), 0.99)
        ev = 0.0
        for k in range(dice_next + 1):
            pk = math.comb(dice_next, k) * q_next**k * (1 - q_next) ** (dice_next - k)
            need = bid.quantity - mine - k - h_rest
            p_true = self._tail(m, p, need)
            p_exact = p_true - self._tail(m, p, need + 1)
            # How likely the bid looks to them: their k plus chance elsewhere.
            their_view = self._tail(total - dice_next, p, bid.quantity - k)
            p_dudo = 1 / (1 + math.exp((their_view - t_next) / self.DUDO_SOFTNESS))
            p_calza = (1 - p_dudo) * calza_next
            ev += pk * (p_dudo * (2 * p_true - 1)
                        + p_calza * ((1 - p_exact) - p_exact * gain_next))
        return ev

    def _lookahead_pick(self, state, scored, n_unknown, hints):
        """(P(true), bid) with the best expected dice, or None."""
        nxt = self._next_player(state)
        if nxt is None:
            return None
        dice_next = next(p.dice_count for p in state.players if p.name == nxt)
        t_next = self._dudo_at(nxt)
        calza_next = self.profiler.calza_rate(nxt)
        gain_next = 1.0 if (all(p.alive for p in state.players) and dice_next < 5) else 0.0
        by_player = {f: self._hint_by_player(state, f).get(nxt, 0.0) for f in range(1, 7)}
        best = None
        for pt, b in scored:
            if pt < self.LOOKAHEAD_MIN:
                continue
            ev = self._bid_ev(state, b, n_unknown, hints, nxt, dice_next, by_player[b.face],
                              t_next, calza_next, gain_next)
            key = (ev, self._strength(b, False))
            if best is None or key > best[0]:
                best = (key, pt, b)
        return None if best is None else (best[1], best[2])

    def _prob_exact(self, state, bid, n_unknown, hints):
        """P(exactly bid.quantity matching dice on the table)."""
        more = Bid(bid.quantity + 1, bid.face)
        return self._prob_true(state, bid, n_unknown, hints) - self._prob_true(state, more, n_unknown, hints)

    @staticmethod
    def _calza_gains(state):
        """A correct Calza only gives a die while nobody is out and we have < 5."""
        return all(p.alive for p in state.players) and state.my_dice_count < 5

    # -- calibration ------------------------------------------------------

    @staticmethod
    def _size_class(state):
        alive = sum(p.alive for p in state.players)
        return min(alive, 4)

    def _calibrated(self, state, bid, p, exact=False):
        """
        Our estimate for a bid we face, corrected by how often bids we gave
        a similar estimate at a similar table size turned out true. Raw
        estimates in head-to-head play are far too low (opponents' bids say
        more about their hand than the hint model credits), and this learns
        that from the reveals, for any opponent.
        """
        key = (exact, self._size_class(state), min(int(p * 10), 9))
        self._pending.append((key, bid, state.palifico))
        if not self.CALIBRATE or sum(pl.alive for pl in state.players) > self.ENDGAME_PLAYERS:
            return p
        hits, n = self._calibration.get(key, (0, 0))
        k = self.CALIBRATION_PRIOR
        return (hits + k * p) / (n + k)

    def _learn_calibration(self, event):
        dice = [d for _, hand in event.dice for d in hand]
        for key, bid, palifico in self._pending:
            stat = self._calibration.setdefault(key, [0, 0])
            actual = _count_matches(dice, bid.face, palifico)
            exact = key[0]
            stat[0] += actual == bid.quantity if exact else actual >= bid.quantity
            stat[1] += 1
        self._pending = []

    def _log(self, state, action, p_cur, best, best_p, hints, bluffed):
        if self.decision_log is None:
            return
        self.decision_log.append(dict(
            round=state.round_number,
            players={p.name: p.dice_count for p in state.players if p.alive},
            my_dice=state.my_dice,
            palifico=state.palifico,
            history=[(r.player, r.action) for r in state.history],
            current=state.current_bid,
            p_current=p_cur,
            action=action,
            best_raise=best,
            p_best_raise=best_p,
            hints=dict(hints),
            bluffed=bluffed,
        ))

    def observe(self, event):
        self._ensure_memory()
        self.profiler.observe(event)
        if isinstance(event, RevealEvent):
            self._learn_calibration(event)
        elif isinstance(event, GameEndedEvent):
            self._decay_memory()
        self._sniper_observe(event)

    def _decay_memory(self):
        """End of a game: what was learned counts GAME_DECAY times as much."""
        f = self.GAME_DECAY
        if f >= 1:
            return
        self.profiler.decay(f)
        for stat in self._calibration.values():
            stat[0] *= f
            stat[1] *= f
        self._thr_cache.clear()   # thresholds are refitted on the decayed data

    def on_game_start(self, info):
        pass

    def on_game_end(self, result):
        pass
