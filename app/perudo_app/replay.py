"""
Games you played, kept so you can watch them again.

A game is recorded as what it takes to play it again: the seed (it seats
the players and picks the first opener), the seating order given to the
engine, every round's dice (as revealed: what decided the round), every
move, what was said, and who left when. A replay runs the engine again with
a scripted seat for every player, each making the recorded moves, on the
recorded dice: the game screen shows it like any game it watches.

Only a game with every cup in view can be recorded: a game at your own
table, or one you host. A guest's phone never sees the others' dice.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from perudo.game import Interrupt, PerudoGame
from perudo.models import (
    Bid,
    BidEvent,
    Calza,
    CalzaEvent,
    Dudo,
    DudoEvent,
    PlayerEliminatedEvent,
    RevealEvent,
)

FORMAT = 1
KEEP = 100                       # the newest recordings kept


class Recorder:
    """Records one game as it is played (fed by its GameSession)."""

    def __init__(self, seed: int, bots, human: Optional[str]):
        self.data: Dict[str, Any] = {
            "format": FORMAT, "seed": seed, "order": [b.name for b in bots], "human": human,
            "people": [b.name for b in bots if getattr(b, "person", False) or b.name == human],
            "moves": [],
        }
        self._round_at: Optional[int] = None       # index of this round's dice in moves
        self._left: set = set()
        self._lock = threading.Lock()

    @property
    def moves(self) -> List[list]:
        return self.data["moves"]

    def round(self, game: PerudoGame) -> None:
        with self._lock:
            self._round_at = len(self.moves)
            self.moves.append(["round", {p.name: list(p.dice) for p in game.players if p.alive}])

    def left(self, names) -> None:
        with self._lock:
            for name in names:
                if name not in self._left:
                    self._left.add(name)
                    self.moves.append(["left", name])

    def talk(self, speaker: str, text: str) -> None:
        with self._lock:
            self.moves.append(["talk", speaker, text])

    def event(self, event: Any) -> None:
        if isinstance(event, BidEvent):
            with self._lock:
                self.moves.append(["bid", event.player, event.bid.quantity, event.bid.face])
        elif isinstance(event, DudoEvent):
            with self._lock:
                self.moves.append(["dudo", event.player])
        elif isinstance(event, CalzaEvent):
            with self._lock:
                self.moves.append(["calza", event.player])
        elif isinstance(event, RevealEvent):
            with self._lock:                         # the dice that decided the round
                if self._round_at is not None:
                    self.moves[self._round_at][1] = {name: list(d) for name, d in event.dice}


class ReplayStore:
    """The recordings, one JSON file each, newest first."""

    def __init__(self, folder: str, keep: int = KEEP):
        self.folder = Path(folder)
        self.keep = keep

    def save(self, recorder: Recorder, snap) -> Optional[str]:
        """Keep a recording, with what the profile shows of it; None when the
        game never got going."""
        data = dict(recorder.data)
        if not any(m[0] in ("bid", "dudo", "calza") for m in data["moves"]):
            return None
        human = data.get("human")
        places = dict(getattr(snap, "places", {}) or {})
        for place, name in enumerate(snap.standings, start=1):
            places.setdefault(name, place)
        mine = [h for h in snap.hands if h.name == human]
        stamp = time.time()
        game_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(stamp)) + f"-{int(stamp * 1000) % 1000:03d}"
        data["meta"] = {
            "id": game_id, "time": stamp, "players": len(data["order"]),
            "winner": snap.standings[0] if snap.standings else None,
            "finished": bool(snap.standings), "place": places.get(human),
            "rounds": snap.round_number,
            "hand": round(sum(h.score for h in mine) / len(mine)) if mine else None,
            "others": [n for n in data["order"] if n != human],
        }
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{game_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
        for old in sorted(self.folder.glob("*.json"))[:-self.keep]:
            try:
                old.unlink()
            except OSError:
                pass
        return game_id

    def games(self) -> List[dict]:
        """What each recording is about (its meta), newest first."""
        out = []
        for path in sorted(self.folder.glob("*.json"), reverse=True):
            try:
                meta = json.loads(path.read_text(encoding="utf-8")).get("meta")
            except (OSError, ValueError):
                continue
            if meta:
                out.append(meta)
        return out

    def load(self, game_id: str) -> Optional[dict]:
        try:
            data = json.loads((self.folder / f"{game_id}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if data.get("format") == FORMAT else None


# ---------------------------------------------------------------------------
# playing a recording again
# ---------------------------------------------------------------------------

class ReplayEnded(Exception):
    """The recording has no more moves (it stopped early), or the game went
    another way than recorded."""


class Script:
    """The recorded moves, handed out in order."""

    def __init__(self, data: dict):
        self.moves = list(data["moves"])
        self.at = 0
        self.game: Optional[PerudoGame] = None
        self.say = None                   # (speaker, text): a recorded line, said again
        self.leave = None                 # (name): a recorded seat leaving (else straight to the game)

    def _peek(self):
        return self.moves[self.at] if self.at < len(self.moves) else None

    def catch_up(self) -> None:
        """Lines said and seats left, up to the next move or round."""
        while True:
            move = self._peek()
            if move is None:
                return
            if move[0] == "talk":
                self.at += 1
                if self.say is not None:
                    self.say(move[1], move[2])
            elif move[0] == "left":
                self.at += 1
                if self.leave is not None:
                    self.leave(move[1])
                elif self.game is not None:
                    self.game.forfeit(move[1])
            else:
                return

    def dice(self, alive: List[str]) -> Optional[Dict[str, List[int]]]:
        """The next round's recorded dice (None: a round the recording did
        not have, given up when someone left)."""
        self.catch_up()
        move = self._peek()
        if move is None or move[0] != "round" or set(move[1]) != set(alive):
            return None
        self.at += 1
        return move[1]

    def action(self, name: str):
        self.catch_up()
        if self.game is not None and self.game.forfeit_pending():
            raise Interrupt()                 # someone left here: the round is given up
        move = self._peek()
        if move is None:
            raise ReplayEnded("the recording ends here")
        if move[0] == "round":
            self.at += 1                      # a round the engine skipped ahead of: its dice were rolled
            return self.action(name)
        kind, who = move[0], move[1]
        if who != name:
            raise ReplayEnded(f"the recording has {who} to move, not {name}")
        self.at += 1
        if kind == "bid":
            return Bid(move[2], move[3])
        return Dudo() if kind == "dudo" else Calza()


class ScriptedSeat:
    """A player of the recording: makes its recorded moves."""

    def __init__(self, name: str, person: bool, script: Script):
        self.name = name
        self.person = person
        self.script = script

    def play(self, state):
        return self.script.action(self.name)


class ReplayGame(PerudoGame):
    """The engine, on the recorded dice."""

    script: Optional[Script] = None

    def _roll_round(self) -> None:
        super()._roll_round()                 # the same random numbers drawn as when it was played
        if self.script is None:
            return
        dice = self.script.dice([p.name for p in self.players if p.alive])
        if dice is not None:
            for p in self.players:
                if p.alive and p.name in dice:
                    p.dice = list(dice[p.name])
