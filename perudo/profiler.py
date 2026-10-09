"""
Opponent profiling: learn, per opponent name, what kind of player they are.

Every round ends with a reveal, so afterwards everyone's dice are public.
That lets us replay each opponent's decisions from *their* point of view
(their own dice, the bid they faced) and see what they actually did. The
profile is kept per name, so it carries over between games as long as the
bot instance lives (run_simulation.py reuses bot objects).

Situations an opponent can be in, and what we measure there:

  * Choosing a face (opening, or switching to another face than the bid
    before): did they pick a face they hold a lot of, or a face they hold
    little of? -> `face_score`, 1 = always their best face, 0 = no better
    than a random face, negative = deliberately their worst face (bluffing).
  * Following (raising on the face just bid): how often they stay on the
    face of the bid before -> `follow_rate`.
  * Raise size: did they make the smallest legal raise or jump ahead?
    -> `min_raise_rate`.
  * Risk of their own bid: P(bid true | their own dice) -> `bid_risk`, the
    mean of that. Low means they bid beyond what they can see.
  * Facing a bid, Dudo or raise: P(that bid true | their own dice) at the
    moment they chose. We fit the threshold `dudo_at` below which they call
    Dudo, and how well a single threshold explains their choices
    (`dudo_fit`; low means they use more than their own dice, or chance),
    and `dudo_separation`: how much likelier the bids they raised on were
    than the ones they called Dudo on (near 0 = Dudo regardless of dice).
  * Calza: how often they choose it when facing a bid -> `calza_rate`.

Two numbers are meant for decisions rather than labels, and work for any
opponent without knowing its code:

  * `bid_excess(name, kind)`: how many more dice of a face they really hold
    when they bid it, split by opening/switching to a face ("choice") versus
    raising on the face just bid ("follow"), and by Paco versus normal face.
    Honest bidders score high, bluffers zero or below.
  * `challenge_rate(name)`: how often they call Dudo on a bid that is really
    false, i.e. how risky it is to put a false bid in front of them.

Palifico rounds are skipped: faces are forced there, so they say little
about style.

P(true) only uses that player's own dice, never the bids of others. Bots
that read the table will look a bit noisier (lower `dudo_fit`), but every
bot is measured on the same scale.

From these numbers `labels()` derives a category per axis, e.g.
("honest", "cautious", "dudo-happy", "min-raiser").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from perudo.models import (
    PACO,
    Bid,
    BidEvent,
    Calza,
    CalzaEvent,
    Dudo,
    DudoEvent,
    GameEndedEvent,
    RevealEvent,
    RoundEndedEvent,
)
from perudo.rules import count_matches, legal_bids

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
    need = bid.quantity - count_matches(own_dice, bid.face, palifico)
    return p_at_least(total_dice - len(own_dice), p, need)


def _strength(bid: Bid) -> float:
    """
    Claim size so Paco and normal bids compare (a Paco counts double). The
    face is ignored: a min-raise is a raise without a jump in quantity.
    """
    return 2 * bid.quantity if bid.face == PACO else bid.quantity


@dataclass
class Profile:
    """What we learned about one opponent. Plain numbers, easy to inspect."""

    name: str
    rounds: int = 0
    bids: int = 0
    face_scores: List[float] = field(default_factory=list)
    follows: int = 0
    raises: int = 0
    min_raises: int = 0
    bid_risk_sum: float = 0.0
    # (P(bid faced is true | own dice), called Dudo?) per decision facing a bid.
    facing: List[Tuple[float, bool]] = field(default_factory=list)
    calzas: int = 0
    # What their bids say about their hand: per kind of bid ("nat-choice",
    # "nat-follow", "paco-choice", "paco-follow") the summed excess of that
    # face in their hand over what chance gives, and the number of samples.
    excess: Dict[str, List[float]] = field(default_factory=dict)
    # How often they call Dudo on a bid that is really false (a choice, not
    # forced): the risk of putting a false bid in front of them.
    false_faced: int = 0
    false_challenged: int = 0

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
            f"{self.name:<14} rondes {self.rounds:4d}  face {f(self.face_score):>5}  "
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
    # Weight of the prior / pooled estimate, in observations.
    SHRINK = 10

    def __init__(self, me: Optional[str] = None):
        self.me = me
        self.profiles: Dict[str, Profile] = {}
        self._actions: List[Tuple[str, object]] = []
        self._dice: Optional[Dict[str, Tuple[int, ...]]] = None
        self._palifico = False

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

    # -- event stream ---------------------------------------------------

    def observe(self, event) -> None:
        if isinstance(event, BidEvent):
            self._actions.append((event.player, event.bid))
        elif isinstance(event, DudoEvent):
            self._actions.append((event.player, Dudo()))
        elif isinstance(event, CalzaEvent):
            self._actions.append((event.player, Calza()))
        elif isinstance(event, RevealEvent):
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
            options = legal_bids(prev, False, None, total) if prev is not None else []
            # A Dudo is only a choice when a raise was possible.
            if prev is not None and options:
                p_cur = p_bid_true(prev, own, total)
                prof.facing.append((p_cur, isinstance(action, Dudo)))
                if count_matches(all_dice, prev.face, False) < prev.quantity:
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
        counts = [count_matches(own, f, False) for f in range(2, 7)]
        best, mean = max(counts), sum(counts) / len(counts)
        if best == mean:
            return  # every face equally good: no information
        prof.face_scores.append((count_matches(own, bid.face, False) - mean) / (best - mean))

    # -- reporting --------------------------------------------------------

    def report(self) -> str:
        lines = []
        for name in sorted(self.profiles):
            prof = self.profiles[name]
            labels = prof.labels()
            lines.append(prof.summary())
            lines.append(" " * 15 + "-> " + ", ".join(f"{k}: {v}" for k, v in labels.items()))
        return "\n".join(lines)
