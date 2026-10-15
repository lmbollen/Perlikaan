"""
The app's model, without any UI toolkit: loading bots, running one game
step by step (optionally with a human at the table), running simulations
with live standings, settings, and routing the bots' talk to the UI and to
speech. The engine (perudo.game) is synchronous, so a game runs in a worker
thread; the UI polls `snapshot()` and feeds decisions in with `submit()`.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import random
import shutil
import sys
import threading
import time
import types
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from perudo import hand_score
from perudo import memory as bot_memory
from perudo.game import Forfeit, Interrupt, PerudoGame
from perudo.models import (
    PACO,
    Bid,
    BidEvent,
    Calza,
    CalzaEvent,
    DieGainedEvent,
    DieLostEvent,
    Dudo,
    GameEndedEvent,
    Observation,
    PlayerEliminatedEvent,
    RevealEvent,
    RoundEndedEvent,
)
from perudo.rules import count_matches

FACES = (1, 2, 3, 4, 5, 6)


def face_label(face: int) -> str:
    return "P" if face == PACO else str(face)


def fmt_bid(bid: Optional[Bid]) -> str:
    if bid is None:
        return "-"
    return f"{bid.quantity} x {'Paco' if bid.face == PACO else bid.face}"


def fmt_action(action: Any) -> str:
    if isinstance(action, Bid):
        return fmt_bid(action)
    if isinstance(action, Dudo):
        return "Dudo!"
    if isinstance(action, Calza):
        return "Calza!"
    return str(action)


# ---------------------------------------------------------------------------
# bots
# ---------------------------------------------------------------------------

@dataclass
class BotEntry:
    class_name: str
    display_name: str
    cls: type
    source: str                  # "bundled" or the file it was loaded from
    enabled: bool = True
    copies: int = 1


class BotRegistry:
    """The bots the app can seat: the ones bundled from the repository
    (perudo/bots, seated by default per their `count`) plus the bots the
    user imported. Imports are copied into `user_dir`, which is loaded again
    on every start, so they persist."""

    def __init__(self, user_dir: Optional[str] = None):
        self.entries: List[BotEntry] = []
        self.errors: List[str] = []
        self.user_dir = Path(user_dir) if user_dir else None
        self._mtimes: Dict[str, float] = {}

    def scan_bundled(self) -> None:
        for cls in self._discover_bundled():
            self._add(cls, "bundled")

    def _discover_bundled(self) -> List[type]:
        """The bot classes in perudo.bots, by the rules of
        perudo.bot_loader.discover_bot_classes (a class defined in its
        module, with a play() method), but found through the import system:
        the loader looks for *.py files, and an APK built by
        python-for-android carries only compiled *.pyc files."""
        import perudo.bots as package
        classes: List[type] = []
        for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
            if info.ispkg:
                continue
            name = f"{package.__name__}.{info.name}"
            try:
                module = importlib.import_module(name)
            except Exception as exc:         # one broken bot must not hide the others
                self.errors.append(f"{info.name}: {type(exc).__name__}: {exc}")
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__ and hasattr(obj, "play"):
                    classes.append(obj)
        return classes

    # -- imported bots ---------------------------------------------------

    def scan_user(self) -> None:
        """Load the bot files in user_dir: new and changed files are
        (re)loaded, entries of deleted files are dropped."""
        if self.user_dir is None:
            return
        self.user_dir.mkdir(parents=True, exist_ok=True)
        present = {str(p): p for p in sorted(self.user_dir.glob("*.py"))}
        for gone in [src for src in self._mtimes if src not in present]:
            self._drop_source(gone)
            del self._mtimes[gone]
        for src, p in present.items():
            mtime = p.stat().st_mtime
            if self._mtimes.get(src) != mtime:
                self._reload(p, seat=False)      # found in the folder: listed, not seated
                self._mtimes[src] = mtime

    def import_file(self, path: str) -> List[BotEntry]:
        """Copy a bot file into user_dir (replacing an earlier import with
        the same file name) and load it. A file without a working bot is
        not kept."""
        if self.user_dir is None:
            return self.add_file(path)
        src = Path(path)
        self.user_dir.mkdir(parents=True, exist_ok=True)
        dest = self.user_dir / src.name
        previous = dest.read_bytes() if dest.exists() else None
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        added = self._reload(dest)
        if not added:                  # broken: put back what was there
            if previous is not None:
                dest.write_bytes(previous)
                self._reload(dest)
            elif dest.exists():
                dest.unlink()
            self._mtimes.pop(str(dest), None)
        else:
            self._mtimes[str(dest)] = dest.stat().st_mtime
        return added

    def remove(self, entry: BotEntry) -> None:
        """Forget an imported bot and delete its file (with every bot in it)."""
        if entry.source == "bundled":
            return
        self._drop_source(entry.source)
        self._mtimes.pop(entry.source, None)
        try:
            Path(entry.source).unlink()
        except OSError:
            pass

    def _reload(self, p: Path, seat: bool = True) -> List[BotEntry]:
        """(Re)load one file, keeping the seat settings of its bots; new bots
        are seated when `seat` (an import), else just listed."""
        kept = {e.class_name: (e.enabled, e.copies) for e in self.entries if e.source == str(p)}
        self._drop_source(str(p))
        added = self.add_file(str(p))
        for e in added:
            if e.class_name in kept:
                e.enabled, e.copies = kept[e.class_name]
            else:
                e.enabled = seat
        return added

    def _drop_source(self, source: str) -> None:
        self.entries = [e for e in self.entries if e.source != source]

    # -- the table, saved between runs -------------------------------------

    @staticmethod
    def key(entry: BotEntry) -> str:
        where = "bundled" if entry.source == "bundled" else Path(entry.source).name
        return f"{where}:{entry.class_name}"

    def selection(self) -> Dict[str, list]:
        return {self.key(e): [e.enabled, e.copies] for e in self.entries}

    def apply_selection(self, saved: Optional[Dict[str, Any]], seats: bool = True) -> None:
        """Restore the copies (and, with seats, who is seated) saved earlier."""
        for e in self.entries:
            value = (saved or {}).get(self.key(e))
            if isinstance(value, (list, tuple)) and len(value) == 2:
                e.copies = max(1, min(6, int(value[1])))
                if seats:
                    e.enabled = bool(value[0])

    def clear_table(self) -> None:
        """Nobody seated: the app starts with an empty table."""
        for e in self.entries:
            e.enabled = False

    def add_file(self, path: str) -> List[BotEntry]:
        """Load every bot class from a Python file. Returns the new entries."""
        p = Path(path)
        module_name = f"perudo_app_user_bots.{p.stem}"
        # Compiled straight from the source, not through the import system:
        # its bytecode cache could serve an earlier version of a re-imported
        # file (it trusts whole-second mtimes and sizes).
        module = types.ModuleType(module_name)
        module.__file__ = str(p)
        try:
            source = p.read_text(encoding="utf-8")
            sys.modules[module_name] = module
            exec(compile(source, str(p), "exec"), module.__dict__)
        except Exception as exc:       # a user's file may fail in any way
            sys.modules.pop(module_name, None)
            self.errors.append(f"{p.name}: {type(exc).__name__}: {exc}")
            return []
        added = []
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__ or not hasattr(obj, "play"):
                continue
            entry = self._add(obj, str(p))
            if entry is not None:
                added.append(entry)
        if not added:
            self.errors.append(f"{p.name}: no class with a play() method")
        return added

    def _add(self, cls: type, source: str) -> Optional[BotEntry]:
        try:
            display = getattr(cls(), "name", cls.__name__)
        except Exception as exc:
            self.errors.append(f"{cls.__name__}: {exc}")
            return None
        for e in self.entries:
            if e.class_name == cls.__name__ and e.source == source:
                e.cls = cls
                return e
        default_copies = getattr(cls, "count", 1)
        entry = BotEntry(cls.__name__, display, cls, source,
                         enabled=default_copies > 0 if source == "bundled" else True,
                         copies=max(1, default_copies))
        self.entries.append(entry)
        return entry

    def selected(self) -> List[BotEntry]:
        return [e for e in self.entries if e.enabled]

    def make(self, entries: Optional[Sequence[BotEntry]] = None, fresh: bool = True,
             memory: Optional["BotMemory"] = None) -> List[Any]:
        """Instantiate the selected bots, numbering copies like the loader does;
        with `memory`, each gets back what it learned in earlier games."""
        bots: List[Any] = []
        for e in entries if entries is not None else self.selected():
            if fresh:
                for cls in {x.cls for x in self.entries}:
                    if hasattr(cls, "reset_memory"):
                        cls.reset_memory()
            for i in range(1, e.copies + 1):
                bot = e.cls()
                if i > 1:
                    bot.name = f"{bot.name} {i}"
                bots.append(bot)
        if memory is not None:
            for bot in bots:
                memory.load(bot)
        return bots


class BotMemory:
    """What the bots learned (perudo.memory), kept in a folder: one JSON
    file per bot, by its kind and its name at the table (\"Carlos\" and
    \"Carlos 2\" learn apart). The app reads and writes the files; the bots
    only keep their learning in the attributes they name."""

    def __init__(self, folder: str):
        self.folder = Path(folder)
        self._lock = threading.Lock()

    def _file(self, bot: Any) -> Path:
        # memory_name: its own name, when the table renamed it ("Carlos (Ben)")
        name = getattr(bot, "memory_name", None) or bot.name
        kind = f"{type(bot).__module__}.{type(bot).__qualname__}:{name}"
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in kind)
        return self.folder / f"{safe}.json"

    def load(self, bot: Any) -> bool:
        if not bot_memory.learned_names(bot):
            return False
        try:
            data = json.loads(self._file(bot).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        try:
            return bot_memory.restore(bot, data)
        except Exception:                 # a newer bot that no longer fits what was saved
            return False

    def save(self, bot: Any) -> bool:
        if not bot_memory.learned_names(bot):
            return False
        try:
            text = json.dumps({**bot_memory.save(bot), "bot": bot.name})
        except Exception:                 # something it keeps is not plain data
            return False
        path = self._file(bot)
        with self._lock:
            self.folder.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)             # never half a file
        return True

    def save_all(self, bots: Sequence[Any]) -> None:
        for bot in bots:
            self.save(bot)

    def who_remembers(self, name: str) -> List[str]:
        """The bots whose memory mentions this player (by name)."""
        needle = json.dumps(name)
        out = []
        for path in sorted(self.folder.glob("*.json")):
            try:
                text = path.read_text(encoding="utf-8")
                if needle in json.dumps(json.loads(text).get("learned")):
                    out.append(json.loads(text).get("bot") or path.stem)
            except (OSError, ValueError, AttributeError):
                pass
        return out

    def forget(self) -> None:
        """Every bot starts from scratch again."""
        with self._lock:
            for path in self.folder.glob("*.json"):
                try:
                    path.unlink()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# talk and speech routing
# ---------------------------------------------------------------------------

class TalkRouter:
    """Catches what the bots say (perudo.talk.Mouth) and hands it to the
    current session's commentary and to the speaker. Installed once."""

    def __init__(self):
        self.listener: Optional[Callable[[str, str, str], None]] = None   # (speaker, text, lang)
        self.speak: Optional[Callable[[str, str, str], None]] = None      # (text, lang, voice_seed)
        # Only talk from this thread (the game on screen) goes anywhere: the
        # same bots also talk in simulations and warm-up games, silently.
        self.thread: Optional[threading.Thread] = None
        self._installed = False

    def attach(self, listener, thread: threading.Thread) -> None:
        self.listener, self.thread = listener, thread

    def detach(self, listener) -> None:
        if self.listener == listener:
            self.listener, self.thread = None, None

    def install(self) -> None:
        if self._installed:
            return
        try:
            import perudo.talk as talk
        except Exception:
            return
        router = self

        def speak(text: str, lang: str = "nl", voice_seed: str = "") -> None:
            if router.thread is None or threading.current_thread() is not router.thread:
                return                       # a simulation, a warm-up game, or no game on screen
            listener = router.listener
            if listener is not None and listener(voice_seed, text, lang) is False:
                return                       # the session dropped it
            if router.speak is not None:
                router.speak(text, lang, voice_seed)

        talk.speak = speak
        talk.tts_enabled = lambda: True      # Mouth.say only composes a line when talk or tts is on
        talk.talk_enabled = lambda: False    # never print to the console from the app

        # The app shows every move, so a bot no longer announces its bid (or
        # "Dudo!", "Calza!") when it has nothing else to say; its quotes stay.
        original_say = talk.Mouth.say

        def say(mouth, key, announce=None, **fmt):
            return original_say(mouth, key, None, **fmt)

        talk.Mouth.say = say
        # A bot picks a quote whenever it has one for the moment; the game on
        # screen then lets each line through with GameSession.TALK_CHANCE.
        talk.QUOTE_CHANCE = 1.0
        self._installed = True


ROUTER = TalkRouter()


# ---------------------------------------------------------------------------
# one game, step by step, with or without a human seat
# ---------------------------------------------------------------------------

class _RoundLimit(Exception):
    """A game that would not end (two stubborn bots): abandoned."""


class SessionStopped(Exception):
    pass


class HumanSeat:
    """A bot whose play() waits for the person at the phone."""

    def __init__(self, name: str = "You"):
        self.name = name
        self.session: Optional["GameSession"] = None

    def on_game_start(self, info):
        pass

    def observe(self, event):
        pass

    def on_game_end(self, result):
        pass

    def play(self, state: Observation):
        return self.session.ask_human(state)


@dataclass
class PlayerView:
    name: str
    dice_count: int
    alive: bool
    dice: tuple = ()
    is_human: bool = False        # the person looking at this screen
    person: bool = False          # played by a person (here or on another phone)


@dataclass(frozen=True)
class FeedItem:
    """One line of the game's story, for the UI to style by kind.

    kind is one of: round, bid, dudo, calza, result, gain, out, win, talk.
    `player` is who acted (or spoke); `other` the player acted on."""
    kind: str
    player: Optional[str] = None
    text: str = ""
    bid: Optional[Bid] = None
    other: Optional[str] = None
    good: Optional[bool] = None        # result: did the call succeed
    loser: Optional[str] = None        # result: who lost a die (None: an exact Calza)


@dataclass(frozen=True)
class HandResult:
    """A hand once the cups are up: how good it was (perudo.hand_score),
    and how the round went for it."""
    round: int
    name: str
    score: int          # 0..100, 50 an average hand of its size
    value: float        # matches beyond what the table assumed it held
    lost: bool          # it lost a die


def revealed_hands(round_number: int, reveal: Optional[dict], starter: Optional[str]) -> List[HandResult]:
    """Score the hands of a reveal: the revealed dice, and who opened (in a
    Palifico round the opener picks the face). Only from what everyone sees
    once the cups are up, so nobody learns anything early."""
    if not reveal or not starter:
        return []
    out = []
    for name, dice in reveal["dice"]:
        hs = hand_score.score(tuple(dice), reveal["palifico"], opens=name == starter)
        out.append(HandResult(round_number, name, hs.score, hs.value, name == reveal.get("loser")))
    return out


@dataclass(frozen=True)
class HandSummary:
    """A player's hands over a game."""
    name: str
    rounds: int         # rounds played (and revealed)
    average: int        # their average hand score
    value: float        # matches beyond the table's guess, added up over their hands
    lost: int           # dice they lost


def hand_summaries(hands: Sequence[HandResult]) -> Dict[str, HandSummary]:
    per: Dict[str, List[HandResult]] = defaultdict(list)
    for h in hands:
        per[h.name].append(h)
    return {name: HandSummary(name, len(hs), round(sum(h.score for h in hs) / len(hs)),
                              sum(h.value for h in hs), sum(1 for h in hs if h.lost))
            for name, hs in per.items()}


@dataclass
class Snapshot:
    round_number: int = 0
    palifico: bool = False
    players: List[PlayerView] = field(default_factory=list)
    history: List[str] = field(default_factory=list)      # this round's actions, formatted
    current_bid: Optional[Bid] = None
    to_act: Optional[str] = None
    log: List[str] = field(default_factory=list)          # everything that happened, formatted
    commentary: List[str] = field(default_factory=list)   # what the bots said
    pending: Optional[Observation] = None                 # the human's turn, waiting for submit()
    reveal: Optional[dict] = None                         # the last reveal, kept until the next bid
    finished: bool = False
    standings: tuple = ()
    error: Optional[str] = None
    feed: List[FeedItem] = field(default_factory=list)    # log and talk in order, structured
    bidder: Optional[str] = None                          # who made current_bid
    warmup: Optional[tuple] = None                        # (done, total) while warm-up games run
    awaiting_next_round: bool = False                     # a reveal is up; next_round() goes on
    hands: List["HandResult"] = field(default_factory=list)  # every revealed hand, scored (see revealed_hands)
    places: Dict[str, int] = field(default_factory=dict)  # the players who are out: where they finished

    def totals(self) -> Dict[int, int]:
        """Count per face over every alive player's dice (Pacos wild unless Palifico)."""
        dice = [d for p in self.players if p.alive for d in p.dice]
        return {f: count_matches(dice, f, self.palifico) for f in FACES}

    def naturals(self) -> Dict[int, int]:
        dice = [d for p in self.players if p.alive for d in p.dice]
        return {f: sum(1 for d in dice if d == f) for f in FACES}


class GameSession:
    """Runs one game in a thread. In step mode every turn waits for next();
    in auto mode turns follow each other `delay` seconds apart. With a
    HumanSeat among the bots, that seat's turns wait for submit()."""

    TALK_CHANCE = 0.25      # how often a bot's line is said (at most one per bot per turn)
    can_continue = True     # this screen may start the next round (a guest's may not)

    def __init__(self, bots: Sequence[Any], seed: Optional[int] = None, step: bool = True,
                 delay: float = 1.0, log_limit: int = 400, warmup: int = 0,
                 speech_idle: Optional[Callable[[], bool]] = None, speech_timeout: float = 20.0,
                 hold_reveals: bool = False, memory: Optional[BotMemory] = None):
        self.bots = list(bots)
        self.memory = memory                    # where the bots' learning is kept when the game ends
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2**63)
        self.step = step
        self.delay = delay
        self.log_limit = log_limit
        self.human = next((b for b in self.bots if isinstance(b, HumanSeat)), None)
        if self.human is not None:
            self.human.session = self
        self._lock = threading.Lock()
        self._go = threading.Event()        # step mode: let one turn happen
        self._pause = threading.Event()     # auto mode: cut the wait between turns short
        self._answer = threading.Event()
        self._stop = False
        self._action: Any = None
        self._snap = Snapshot()
        self._game: Optional[PerudoGame] = None
        self._thread: Optional[threading.Thread] = None
        self.result = None
        self.warmup = max(0, int(warmup))       # bot-only games first, so learning bots learn
        self._skip_warmup = False
        self._warming = False
        self.speech_idle = speech_idle          # the next turn waits until this says True
        self.speech_timeout = speech_timeout
        self.hold_reveals = hold_reveals        # after a reveal, wait for next_round()
        self.hold_timeout: Optional[float] = None   # ... or at most this long (nobody at the table)
        self._left: set = set()                 # seats that walked away (forfeit)
        self._next_round = threading.Event()
        self._talk_rng = random.Random(self.seed ^ 0x7A1C)
        self._spoke: set = set()                # bots that said something this turn

    # -- control from the UI ---------------------------------------------

    def start(self) -> None:
        ROUTER.install()
        self._thread = threading.Thread(target=self._run, name="perudo-game", daemon=True)
        ROUTER.attach(self._on_talk, self._thread)
        self._thread.start()

    def skip_warmup(self) -> None:
        self._skip_warmup = True

    def next_round(self) -> None:
        """Hold mode: the reveal has been seen, start the next round."""
        self._next_round.set()

    def next(self) -> None:
        """Step mode: let the next turn happen."""
        self._go.set()

    def set_auto(self, auto: bool, delay: Optional[float] = None) -> None:
        """Play by itself (auto) or one turn per next(). Pausing takes effect
        at once: a game waiting between turns stops there."""
        self.step = not auto
        if delay is not None:
            self.delay = delay
        if auto:
            self._go.set()
        else:
            self._go.clear()
            self._pause.set()

    def submit(self, action: Any) -> None:
        """The human's decision for the pending observation."""
        with self._lock:
            self._action = action
            self._snap.pending = None
        self._answer.set()

    def forfeit(self, names: Sequence[str]) -> None:
        """These seats leave the game: they lose at once and the others play
        on (see PerudoGame.forfeit). The human's pending turn, if any, ends."""
        names = [n for n in names if n not in self._left]
        if not names:
            return
        with self._lock:
            self._left.update(names)
            game = self._game
        if game is not None:
            for name in names:
                game.forfeit(name)
        if self.human is not None and self.human.name in names:
            self._answer.set()
        self._pause.set()                       # no need to sit out the pause between turns

    def has_left(self, name: str) -> bool:
        return name in self._left

    def stop(self) -> None:
        self._stop = True
        self._go.set()
        self._pause.set()
        self._next_round.set()
        self._answer.set()

    def snapshot(self) -> Snapshot:
        with self._lock:
            s = self._snap
            return replace(s, players=list(s.players), history=list(s.history), log=list(s.log),
                           commentary=list(s.commentary), feed=list(s.feed), hands=list(s.hands),
                           places=dict(s.places))

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- called from the game thread ------------------------------------

    def ask_human(self, state: Observation):
        with self._lock:
            self._snap.pending = state
            self._action = None
        if self.human is not None and self.human.name in self._left:
            raise Forfeit()
        self._answer.clear()
        interrupted = False
        while not self._answer.wait(0.2):
            if self.forfeit_waiting():          # someone else left: this turn is cut short
                interrupted = True
                break
        if self._stop:
            raise SessionStopped()
        if (self.human is not None and self.human.name in self._left) or interrupted:
            with self._lock:
                self._snap.pending = None
            raise Forfeit() if not interrupted else Interrupt()
        return self._action

    def forfeit_waiting(self) -> bool:
        """Has a seat left that the game has not put out yet?"""
        game = self._game
        return game is not None and game.forfeit_pending()

    def _wait_turn(self) -> None:
        if self._stop:
            raise SessionStopped()
        if not self.step and self.delay > 0:
            self._pause.clear()
            self._pause.wait(self.delay)       # set_auto(False) or stop() ends it early
        if self.step and not self._stop:
            self._go.wait()
            self._go.clear()
        if self._stop:
            raise SessionStopped()

    def _wait_speech(self) -> None:
        """Let the bots finish speaking before the game goes on."""
        if self.speech_idle is None:
            return
        end = time.monotonic() + self.speech_timeout
        while not self._stop and not self.speech_idle() and time.monotonic() < end:
            time.sleep(0.05)

    def _hold_after_reveal(self, obs: Observation) -> None:
        """A round was decided: keep its reveal up (and the round as it was)
        until next_round(), instead of going straight to the next opener."""
        with self._lock:
            if not (self.hold_reveals and self._snap.reveal is not None
                    and obs.round_number != self._snap.round_number):
                return
            self._next_round.clear()
            self._snap.awaiting_next_round = True
            self._snap.to_act = None
        start = time.monotonic()
        while not self._stop and not self._next_round.wait(0.1):
            if self.hold_timeout is not None and time.monotonic() - start >= self.hold_timeout:
                break
        with self._lock:
            self._snap.awaiting_next_round = False
            self._snap.reveal = None
        if self._stop:
            raise SessionStopped()

    def _before_turn(self, obs: Observation) -> None:
        time.sleep(0)                       # let the UI thread in between turns
        self._wait_speech()
        self._hold_after_reveal(obs)
        self._spoke.clear()
        game = self._game
        with self._lock:
            s = self._snap
            if obs.round_number != s.round_number:
                starter = obs.players[obs.my_id].name
                self._starter = starter
                self._feed(FeedItem("round", starter, f"Round {obs.round_number}" + (" \u00b7 Palifico" if obs.palifico else "")))
            s.round_number = obs.round_number
            s.palifico = obs.palifico
            s.current_bid = obs.current_bid
            s.bidder = next((r.player for r in reversed(obs.history) if isinstance(r.action, Bid)), None)
            s.to_act = obs.players[obs.my_id].name
            s.history = [f"{r.player}: {fmt_action(r.action)}" for r in obs.history]
            s.players = [PlayerView(p.name, p.dice_count, p.alive, tuple(p.dice), isinstance(p.bot, HumanSeat),
                                    isinstance(p.bot, HumanSeat) or getattr(p.bot, "person", False))
                         for p in game.players]
            if obs.history:              # the new round's first bid is in: the old reveal is done
                s.reveal = None
        if not (self.human is not None and obs.players[obs.my_id].name == self.human.name):
            self._wait_turn()

    def _before_reveal(self, challenge_type, challenger, bidder, current_bid) -> None:
        self._log(f"{challenger} calls {'Dudo' if challenge_type == 'dudo' else 'Calza'} on {bidder}'s {fmt_bid(current_bid)}")
        with self._lock:
            self._feed(FeedItem(challenge_type, challenger, bid=current_bid, other=bidder))
            self._snap.to_act = None
        self._wait_speech()
        self._wait_turn()

    def _on_event(self, event) -> None:
        if isinstance(event, BidEvent):
            self._log(f"{event.player} bids {fmt_bid(event.bid)}")
            with self._lock:
                self._feed(FeedItem("bid", event.player, bid=event.bid))
        elif isinstance(event, RevealEvent):
            if event.challenge_type == "dudo":
                good = event.result != "bidder_correct"
                loser = event.challenger if not good else event.bidder
                text = f"{loser} loses a die"
                self._log(f"{event.actual_count} on the table: {loser} loses a die")
            else:
                good = event.result == "correct"
                loser = None if good else event.challenger
                text = "exact!" if good else f"{event.challenger} loses a die"
                self._log(f"{event.actual_count} on the table: Calza {'exact' if good else 'wrong'}")
            with self._lock:
                self._snap.reveal = {
                    "dice": [(n, tuple(d)) for n, d in event.dice],
                    "bid": event.current_bid, "actual": event.actual_count,
                    "type": event.challenge_type, "challenger": event.challenger,
                    "bidder": event.bidder, "result": event.result,
                    "good": good, "loser": loser, "palifico": self._snap.palifico,
                    "round": self._snap.round_number,     # it stays up into the next round
                }
                self._feed(FeedItem("result", event.challenger, text, bid=event.current_bid,
                                    other=str(event.actual_count), good=good, loser=loser))
                self._snap.hands.extend(revealed_hands(self._snap.round_number, self._snap.reveal,
                                                       getattr(self, "_starter", None)))
        elif isinstance(event, DieGainedEvent):
            self._log(f"{event.player} gains a die ({event.dice_count})")
            with self._lock:
                self._feed(FeedItem("gain", event.player, f"gains a die ({event.dice_count})"))
        elif isinstance(event, PlayerEliminatedEvent):
            text = "left the game" if event.player in self._left else "is out"
            self._log(f"{event.player} {text}")
            with self._lock:
                self._feed(FeedItem("out", event.player, text))
                seats = len(self._game.players) if self._game is not None else len(self._snap.players)
                self._snap.places[event.player] = seats - len(self._snap.places)
        elif isinstance(event, RoundEndedEvent):
            if event.palifico_next:
                self._log(f"Palifico next, {event.next_starting_player} starts")
        elif isinstance(event, GameEndedEvent):
            with self._lock:
                self._snap.finished = True
                self._snap.standings = event.standings
                self._snap.to_act = None
                self._snap.pending = None
                self._feed(FeedItem("win", event.winner, "wins the game"))
            self._log(f"{event.winner} wins")

    def _on_talk(self, speaker: str, text: str, lang: str):
        """A bot wants to say a line: shown and spoken with TALK_CHANCE, at
        most one line per bot per turn, never during warm-up games, and never
        once it is out of the game."""
        if self._warming or speaker in self._spoke or self._is_out(speaker):
            return False
        if self._talk_rng.random() >= self.TALK_CHANCE:
            return False
        self._spoke.add(speaker)
        with self._lock:
            self._snap.commentary.append(f"{speaker}: {text}")
            del self._snap.commentary[:-self.log_limit]
            self._feed(FeedItem("talk", speaker, text))

    def _is_out(self, name: str) -> bool:
        """Out of the game (or left it): no more say at the table."""
        game = self._game
        if game is None:
            return False
        return name in self._left or any(p.name == name and not p.alive for p in game.players)

    def inject_talk(self, speaker: str, text: str, lang: str = "nl") -> bool:
        """A line from a bot playing on another phone: the same rules as a
        local bot's line; said here too when it gets through."""
        if self._on_talk(speaker, text, lang) is False:
            return False
        if ROUTER.speak is not None:
            ROUTER.speak(text, lang, speaker)
        return True

    def _feed(self, item: FeedItem) -> None:
        """Append to the feed; the caller holds the lock."""
        self._snap.feed.append(item)
        del self._snap.feed[:-self.log_limit]

    def _log(self, line: str) -> None:
        with self._lock:
            self._snap.log.append(line)
            del self._snap.log[:-self.log_limit]

    def _warm_up(self) -> None:
        """Bot-only games before the shown one (without a human seat), so
        bots that learn have played this table before. Quiet: their talk is
        dropped, and nothing is logged."""
        players = [b for b in self.bots if not isinstance(b, HumanSeat)]
        if self.warmup <= 0 or len(players) < 2:
            return
        rng = random.Random(self.seed ^ 0x5EED)
        self._warming = True
        try:
            for done in range(self.warmup):
                if self._stop or self._skip_warmup:
                    break
                with self._lock:
                    self._snap.warmup = (done, self.warmup)
                rounds = {"n": 0}

                def limit(event, rounds=rounds):
                    if isinstance(event, RoundEndedEvent):
                        rounds["n"] += 1
                        if rounds["n"] > 2000:
                            raise _RoundLimit()

                try:
                    PerudoGame(bots=players, seed=rng.randrange(2**63), listeners=[limit]).play()
                except _RoundLimit:
                    pass
        finally:
            self._warming = False
            with self._lock:
                self._snap.warmup = None

    def _run(self) -> None:
        try:
            self._warm_up()
            if self._stop:
                raise SessionStopped()
            game = PerudoGame(bots=self.bots, seed=self.seed, before_turn=self._before_turn,
                              before_reveal=self._before_reveal, listeners=[self._on_event])
            with self._lock:
                self._game = game
                left = set(self._left)
            for name in left:                   # left during the warm-up
                game.forfeit(name)
            self.result = game.play()
        except SessionStopped:
            with self._lock:
                self._snap.finished = True
                self._snap.pending = None
        except Exception as exc:        # a bot crashed: show it instead of hanging
            with self._lock:
                self._snap.finished = True
                self._snap.pending = None
                self._snap.error = f"{type(exc).__name__}: {exc}"
        finally:
            ROUTER.detach(self._on_talk)
            if self.memory is not None:         # what they learned, for next time
                self.memory.save_all(self.bots)


# ---------------------------------------------------------------------------
# simulations with live standings
# ---------------------------------------------------------------------------

@dataclass
class Standing:
    name: str
    wins: int
    games: int
    position_sum: int

    @property
    def win_pct(self) -> float:
        return 100.0 * self.wins / self.games if self.games else 0.0

    @property
    def avg_position(self) -> float:
        return self.position_sum / self.games if self.games else 0.0


class Simulation:
    """Many games in a thread; the same bot instances play every game, so
    bots that learn keep learning. standings() is safe to call any time."""

    def __init__(self, bots: Sequence[Any], games: int, seed: Optional[int] = None, round_limit: int = 2000):
        self.bots = list(bots)
        self.games = games
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2**63)
        self.round_limit = round_limit
        self._lock = threading.Lock()
        self._stop = False
        self._played = 0
        self._wins = defaultdict(int)
        self._positions = defaultdict(int)
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    @property
    def rate(self) -> float:
        """Games per second so far."""
        if self.started_at is None:
            return 0.0
        elapsed = (self.finished_at or time.monotonic()) - self.started_at
        return self.played / elapsed if elapsed > 0 else 0.0

    def start(self) -> None:
        ROUTER.install()                     # its thread is never the router's: the bots stay quiet
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="perudo-sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def played(self) -> int:
        with self._lock:
            return self._played

    def standings(self) -> List[Standing]:
        with self._lock:
            rows = [Standing(b.name, self._wins[b.name], self._played, self._positions[b.name]) for b in self.bots]
        return sorted(rows, key=lambda r: (-r.wins, r.position_sum, r.name))

    def _run(self) -> None:
        rng = random.Random(self.seed)
        try:
            for _ in range(self.games):
                if self._stop:
                    break
                rounds = {"n": 0}

                def limit(event, rounds=rounds):
                    if isinstance(event, RoundEndedEvent):
                        rounds["n"] += 1
                        if rounds["n"] > self.round_limit:
                            raise RuntimeError("round limit")

                try:
                    result = PerudoGame(bots=self.bots, seed=rng.randrange(2**63), listeners=[limit]).play()
                except RuntimeError as exc:
                    if "round limit" in str(exc):
                        continue
                    raise
                with self._lock:
                    self._played += 1
                    self._wins[result.winner] += 1
                    for pos, name in enumerate(result.standings, start=1):
                        self._positions[name] += pos
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.finished_at = time.monotonic()


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS = {
    "tts": False,
    "chat": True,              # show what the bots say in the game's feed
    "tts_engine": "system",   # the phone's own voices, or "edge" (Microsoft's online voices)
    "tts_lang": "nl",
    "show_dice": True,
    "hand_score": True,        # after each round and the game: how good the hands were
    "sim_games": 200,
    "step_delay": 1.0,
    "human_name": "You",
    "warmup_games": 0,         # bot-only games before a watched or joined game
    "name_confirmed": False,   # asked for the name once (playing together)
    "guest_bots": True,        # hosting: guests may bring bots
    "guest_bot_limit": 2,      # ... at most this many each
    "host_online": True,       # hosting: over the internet (when the app has an online key), else Wi-Fi
    "table": {},          # who is seated: BotRegistry.selection()
}


class Settings:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self.values: Dict[str, Any] = dict(DEFAULT_SETTINGS)
        self.load()

    def load(self) -> None:
        if self.path and self.path.is_file():
            try:
                self.values.update(json.loads(self.path.read_text()))
            except Exception:
                pass

    def save(self) -> None:
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.values, indent=1))
            except OSError:
                pass

    def __getitem__(self, key):
        return self.values.get(key, DEFAULT_SETTINGS.get(key))

    def __setitem__(self, key, value):
        self.values[key] = value
        self.save()


# ---------------------------------------------------------------------------
# helpers for the human's bid picker
# ---------------------------------------------------------------------------

def legal_bids(state: Observation) -> List[Bid]:
    return [a for a in state.legal_actions if isinstance(a, Bid)]


def lowest_legal(state: Observation, face: int) -> Optional[Bid]:
    bids = [b for b in legal_bids(state) if b.face == face]
    return min(bids, key=lambda b: b.quantity) if bids else None


def can(state: Observation, kind: type) -> bool:
    return any(isinstance(a, kind) for a in state.legal_actions)


# ---------------------------------------------------------------------------
# the phone's system bars
# ---------------------------------------------------------------------------

def bar_padding(surface_height: int, window_height: int, left: int, top: int, right: int,
                bottom: int) -> List[int]:
    """Padding [left, top, right, bottom] that keeps the app clear of the
    status and navigation bars. Since Android 15 apps are drawn edge to
    edge, under the bars; older systems lay the app out between them. Only
    the part of the bars the app's surface actually reaches under is
    padded: the surface is `surface_height` tall in a window (the screen)
    of `window_height`, with bars of `top` and `bottom` pixels."""
    under = max(0, min(top + bottom, surface_height - (window_height - top - bottom)))
    pad_top = min(top, under)
    pad_bottom = min(bottom, under - pad_top)
    sides = under > 0
    return [left if sides else 0, pad_top, right if sides else 0, pad_bottom]
