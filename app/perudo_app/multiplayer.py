"""
Playing together, on one local network or over the internet.

The host's phone runs the lobby and the game (a perudo_app.core.GameSession).
Guests join with the lobby code. On the host, every guest's own seat, and
every bot a guest brought, is a RemoteSeat: when the engine asks it to play,
it sends the observation to that guest's phone and waits for the answer (a
legal fallback when the phone is gone or answers nonsense). A guest's bots
run on the guest's phone; only observations, events and actions travel. The
host sends each guest the table as that guest may see it: only their own
dice, everyone's when the cups come up.

The messages travel over a TCP connection on the local network (see
perudo_app.net), or through a hosted relay over the internet (see
perudo_app.relay); the lobby and the game do not care which.

Messages are JSON objects with a "t" field:

  guest -> host  hello{name, code, protocol}  rename{name}  add_bot{key, name}
                 remove_bot{seat}  action{req, action}  talk{speaker, text, lang}
  host -> guest  welcome{id}  reject{reason}  lobby{lobby}  notice{text}
                 game_start{game, seats, you, order, host}  act{req, seat, obs}
                 call{seat, method, arg}  state{...}
"""

from __future__ import annotations

import dataclasses
import queue
import socket
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional

from perudo.game import Forfeit, Interrupt
from perudo.models import PACO, Calza, Dudo, Observation
from perudo_app import net, relay as online
from perudo_app.core import (
    ROUTER, BotRegistry, FeedItem, GameSession, HumanSeat, PlayerView, SessionStopped, Snapshot, legal_bids,
    revealed_hands,
)

NAME_MAX = 20
INTERRUPTED = object()          # LobbyHost.ask: the turn was cut short (someone left)
BOT_ANSWER_TIMEOUT = 60.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fallback_action(state: Observation):
    """A sensible legal move for a seat whose phone does not answer: the
    lowest bid on the face it holds most of, else a Dudo."""
    bids = legal_bids(state)
    if not bids:
        return next(a for a in state.legal_actions if isinstance(a, (Dudo, Calza)))
    held = Counter(d for d in state.my_dice if d != PACO)
    for face, _n in held.most_common():
        options = [b for b in bids if b.face == face]
        if options:
            return min(options, key=lambda b: b.quantity)
    return min(bids, key=lambda b: (b.quantity, b.face))


def clean_name(name: str, default: str = "Player") -> str:
    name = " ".join(str(name or "").split())[:NAME_MAX]
    return name or default


def encode_feed(item: FeedItem) -> dict:
    out = dataclasses.asdict(item)
    out["bid"] = net.encode(item.bid)
    return out


def decode_feed(data: dict) -> FeedItem:
    data = dict(data)
    data["bid"] = net.decode(data.get("bid"))
    return FeedItem(**{f.name: data.get(f.name) for f in dataclasses.fields(FeedItem)})


def _reset_learning(classes) -> None:
    for cls in classes:
        if hasattr(cls, "reset_memory"):
            try:
                cls.reset_memory()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# the host
# ---------------------------------------------------------------------------

@dataclass
class Member:
    id: int
    name: str
    conn: Optional[net.Connection] = None      # None for the host itself
    player_id: Optional[str] = None            # who they are to the bots, whatever their name

    @property
    def is_host(self) -> bool:
        return self.conn is None


@dataclass
class LobbySeat:
    """A bot someone brought to the lobby."""
    id: int
    owner: int           # member id
    key: str             # BotRegistry.key on the owner's phone
    name: str            # the bot's own name
    plays: bool = True   # the host decides


class RemoteSeat:
    """A seat played on another phone: a guest, or a bot running there."""

    def __init__(self, host: "LobbyHost", member_id: int, seat_id: Optional[int], name: str,
                 person: bool = False):
        self.host = host
        self.member_id = member_id
        self.seat_id = seat_id
        self.name = name
        self.person = person

    def play(self, state: Observation):
        if self.host.has_left(self.name):
            raise Forfeit()
        action = self.host.ask(self, state)
        if self.host.has_left(self.name):          # left while it was this seat's turn
            raise Forfeit()
        if action is INTERRUPTED:
            raise Interrupt()
        if action is None or action not in state.legal_actions:
            action = fallback_action(state)
        return action

    def observe(self, event) -> None:
        if not self.person:
            self.host.tell(self, "observe", event)

    def on_game_start(self, info) -> None:
        if not self.person:
            self.host.tell(self, "on_game_start", info)

    def on_game_end(self, result) -> None:
        if not self.person:
            self.host.tell(self, "on_game_end", result)


class LobbyHost:
    """The lobby on the host's phone. The host is member 0."""

    def __init__(self, name: str, registry: BotRegistry, allow_guest_bots: bool = True,
                 guest_bot_limit: int = 2, ip: Optional[str] = None, slots=range(net.SLOTS),
                 relay=None, memory=None, replays=None, player_id: Optional[str] = None):
        """`relay`: a perudo_app.relay backend to host over the internet;
        None hosts on the local network. `memory`: a core.BotMemory, so its
        bots remember earlier games. `replays`: a perudo_app.replay.ReplayStore
        to record the games in (the host sees every cup)."""
        self.registry = registry
        self.memory = memory
        self.replays = replays
        self._lock = threading.RLock()
        self.members: Dict[int, Member] = {0: Member(0, clean_name(name, "Host"), None, player_id)}
        self.seats: List[LobbySeat] = []
        self.allow_guest_bots = allow_guest_bots
        self.guest_bot_limit = guest_bot_limit
        self.phase = "lobby"
        self.session: Optional[GameSession] = None
        self.game_id = 0
        self.version = 0                         # bumped on every lobby change (the UI polls it)
        self.closed = False
        self._ids = iter(range(1, 1_000_000))
        self._seat_ids = iter(range(1, 1_000_000))
        self._reqs = iter(range(1, 1_000_000_000))
        self._waiting: Dict[int, dict] = {}
        self._seat_names: Dict[int, str] = {}    # member id -> name of their seat in the game
        self._game_seats: Dict[int, List[str]] = {}   # member id -> every seat they play in the game
        self.online = relay is not None
        self._server = self._hub = None
        if self.online:
            self._hub = online.RelayHost(relay, self._on_message, self._on_close)
            self.code = self._hub.code
        else:
            self._server, slot = net.listen(slots)
            self.ip = ip or net.local_ipv4s()[0]
            self.code = net.make_code(self.ip, slot)
            threading.Thread(target=self._accept, name="perudo-lobby", daemon=True).start()

    # -- state for the screens ---------------------------------------------

    def state(self) -> dict:
        with self._lock:
            return {
                "code": self.code, "phase": self.phase, "game": self.game_id,
                "host": self.members[0].name,
                "members": [{"id": m.id, "name": m.name, "host": m.is_host} for m in self.members.values()],
                "seats": [{"id": s.id, "owner": s.owner, "owner_name": self._owner_name(s.owner),
                           "name": s.name, "plays": s.plays} for s in self.seats],
                "allow_guest_bots": self.allow_guest_bots, "guest_bot_limit": self.guest_bot_limit,
            }

    def _owner_name(self, member_id: int) -> str:
        m = self.members.get(member_id)
        return m.name if m else "?"

    def _changed(self) -> None:
        self.version += 1
        lobby = self.state()
        for m in list(self.members.values()):
            if m.conn is not None:
                m.conn.send({"t": "lobby", "lobby": lobby})

    def _unique(self, name: str, member_id: Optional[int] = None) -> str:
        taken = {m.name for m in self.members.values() if m.id != member_id}
        base = clean_name(name)
        candidate, n = base, 2
        while candidate in taken:
            candidate, n = f"{base} {n}", n + 1
        return candidate

    # -- what the host does --------------------------------------------------

    def rename(self, name: str) -> None:
        with self._lock:
            self.members[0].name = self._unique(name, 0)
            self._changed()

    def add_bot(self, key: str, name: str) -> None:
        with self._lock:
            self.seats.append(LobbySeat(next(self._seat_ids), 0, key, clean_name(name, "Bot")))
            self._changed()

    def remove_seat(self, seat_id: int) -> None:
        with self._lock:
            self.seats = [s for s in self.seats if s.id != seat_id]
            self._changed()

    def set_plays(self, seat_id: int, plays: bool) -> None:
        with self._lock:
            for s in self.seats:
                if s.id == seat_id:
                    s.plays = bool(plays)
            self._changed()

    def set_guest_bots(self, allowed: bool, limit: int) -> None:
        with self._lock:
            self.allow_guest_bots, self.guest_bot_limit = bool(allowed), max(1, int(limit))
            self._changed()

    def players_ready(self) -> int:
        with self._lock:
            return len(self.members) + sum(1 for s in self.seats if s.plays)

    def start_game(self, delay: float = 1.0, speech_idle: Optional[Callable[[], bool]] = None) -> GameSession:
        with self._lock:
            if self.session is not None and self.session.running:
                return self.session
            playing = [s for s in self.seats if s.plays]
            if len(self.members) + len(playing) < 2:
                raise ValueError("At least two players are needed: invite someone or add a bot")
            self.game_id += 1
            used: set = set()

            def unique(name: str) -> str:
                candidate, n = name, 2
                while candidate in used:
                    candidate, n = f"{name} {n}", n + 1
                used.add(candidate)
                return candidate

            host = self.members[0]
            bots: List[Any] = [HumanSeat(unique(host.name))]
            self._seat_names = {0: bots[0].name}
            self._game_seats = defaultdict(list)
            self._game_seats[0].append(bots[0].name)
            guests = [m for m in self.members.values() if not m.is_host]
            for m in guests:
                seat = RemoteSeat(self, m.id, None, unique(m.name), person=True)
                self._seat_names[m.id] = seat.name
                self._game_seats[m.id].append(seat.name)
                bots.append(seat)
            assignments: Dict[int, list] = defaultdict(list)
            entries = {self.registry.key(e): e for e in self.registry.entries}
            local_classes = set()
            for s in playing:
                if s.owner == 0:
                    entry = entries.get(s.key)
                    if entry is None:
                        continue
                    local_classes.add(entry.cls)
                    bots.append((entry.cls, unique(s.name)))   # made after the learning reset below
                    self._game_seats[0].append(bots[-1][1])
                else:
                    owner = self.members.get(s.owner)
                    if owner is None:
                        continue
                    name = unique(f"{s.name} ({owner.name})")
                    bots.append(RemoteSeat(self, s.owner, s.id, name))
                    self._game_seats[s.owner].append(name)
                    assignments[s.owner].append({"seat": s.id, "key": s.key, "name": name})
            _reset_learning(local_classes)
            # The people at the table, by who they are (the bots learned about them by name).
            people = {m.player_id: self._seat_names[m.id] for m in [host] + guests if m.player_id}
            for i, b in enumerate(bots):
                if isinstance(b, tuple):
                    cls, name = b
                    bot = cls()
                    bot.memory_name, bot.name = bot.name, name
                    if self.memory is not None:
                        self.memory.load(bot, people)
                    bots[i] = bot
            self.phase = "game"
            session = GameSession(bots, step=False, delay=delay, hold_reveals=True, speech_idle=speech_idle,
                                  memory=self.memory, replays=self.replays, people=people)
            self.session = session
            order = [b.name for b in bots]
            for m in guests:
                m.conn.send({"t": "game_start", "game": self.game_id, "seats": assignments[m.id],
                             "you": self._seat_names[m.id], "order": order, "host": host.name,
                             "people": people})
            self._changed()
        session.start()
        threading.Thread(target=self._sync, args=(session, self.game_id), name="perudo-sync",
                         daemon=True).start()
        return session

    def seats_of(self, member_id: int) -> List[str]:
        """The seats this member plays in the game (themselves and their bots)."""
        with self._lock:
            return list(self._game_seats.get(member_id, ()))

    def has_left(self, name: str) -> bool:
        session = self.session
        return session is not None and session.has_left(name)

    def forfeit(self, member_id: int = 0) -> None:
        """A member walks away from the game: they and their bots lose at
        once, and the others play on."""
        with self._lock:
            session = self.session
            names = list(self._game_seats.get(member_id, ()))
            if session is None or not names:
                return
            for w in self._waiting.values():          # their pending turns end
                if w["member"] == member_id:
                    w["event"].set()
        session.forfeit(names)

    def end_game(self) -> None:
        """Back to the lobby (ends a game still going on)."""
        with self._lock:
            session, self.session = self.session, None
            self.phase = "lobby"
            for w in self._waiting.values():
                w["event"].set()
            self._changed()
        if session is not None:
            session.stop()

    def close(self) -> None:
        with self._lock:
            self.closed = True
            session, self.session = self.session, None
            members = [m for m in self.members.values() if m.conn is not None]
        if session is not None:
            session.stop()
        if self._hub is not None:
            self._hub.close()
        if self._server is not None:
            # Shut it down first: closing alone does not wake the accept()
            # blocked in _accept, and the port goes on taking connections.
            try:
                self._server.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._server.close()
            except OSError:
                pass
        for m in members:
            m.conn.close()

    # -- the seats played elsewhere ------------------------------------------

    def ask(self, seat: RemoteSeat, state: Observation):
        """Send a seat's turn to its phone and wait for the action (None when
        the phone left, or a bot took too long)."""
        session = self.session
        with self._lock:
            member = self.members.get(seat.member_id)
            if member is None or member.conn is None or member.conn.closed:
                return None
            req = next(self._reqs)
            waiting = {"event": threading.Event(), "action": None, "member": seat.member_id}
            self._waiting[req] = waiting
            member.conn.send({"t": "act", "req": req, "seat": seat.seat_id, "game": self.game_id,
                              "obs": net.encode(state)})
        deadline = None if seat.person else time.monotonic() + BOT_ANSWER_TIMEOUT
        try:
            while not waiting["event"].wait(0.2):
                if session is None or session._stop:
                    raise SessionStopped()
                if deadline is not None and time.monotonic() > deadline:
                    return None
                if session.forfeit_waiting() and not self.has_left(seat.name):
                    if seat.person:                   # take the question off their screen
                        member.conn.send({"t": "cancel", "req": req})
                    return INTERRUPTED
            if session is None or session._stop:
                raise SessionStopped()
            data = waiting["action"]
            return net.decode(data) if data else None
        finally:
            with self._lock:
                self._waiting.pop(req, None)

    def tell(self, seat: RemoteSeat, method: str, arg: Any) -> None:
        with self._lock:
            member = self.members.get(seat.member_id)
            if member is not None and member.conn is not None:
                member.conn.send({"t": "call", "seat": seat.seat_id, "method": method, "game": self.game_id,
                                  "arg": net.encode(arg)})

    # -- the network -----------------------------------------------------------

    def _accept(self) -> None:
        while not self.closed:
            try:
                sock, _addr = self._server.accept()
            except OSError:
                if self.closed:
                    return
                time.sleep(0.2)
                continue
            net.Connection(sock, self._on_message, self._on_close).start()

    def _on_message(self, conn: net.Connection, msg: dict) -> None:
        kind = msg.get("t")
        session = None
        with self._lock:
            if conn.member_id is None:
                if kind != "hello":
                    conn.close()
                    return
                reason = None
                if net.normalise_code(str(msg.get("code", ""))) != self.code:
                    reason = "That is not this lobby's code"
                elif msg.get("protocol") != net.PROTOCOL:
                    reason = "This lobby runs a different version of the app"
                elif self.phase != "lobby":
                    reason = "A game is going on in this lobby; try again when it is over"
                if reason:
                    conn.send({"t": "reject", "reason": reason})
                    conn.close()
                    return
                pid = msg.get("player_id")
                if not isinstance(pid, str) or not pid or any(m.player_id == pid for m in self.members.values()):
                    pid = None                          # none (an older app), or one already here
                member = Member(next(self._ids), self._unique(msg.get("name", "")), conn, pid[:64] if pid else None)
                conn.member_id = member.id
                self.members[member.id] = member
                conn.send({"t": "welcome", "id": member.id})
                self._changed()
                return
            member = self.members.get(conn.member_id)
            if member is None:
                return
            if kind == "rename":
                member.name = self._unique(msg.get("name", ""), member.id)
                self._changed()
            elif kind == "add_bot":
                mine = sum(1 for s in self.seats if s.owner == member.id)
                if not self.allow_guest_bots:
                    conn.send({"t": "notice", "text": "The host does not let guests bring bots"})
                elif mine >= self.guest_bot_limit:
                    conn.send({"t": "notice", "text": f"You can bring at most {self.guest_bot_limit} "
                                                      f"bot{'s' if self.guest_bot_limit != 1 else ''}"})
                else:
                    self.seats.append(LobbySeat(next(self._seat_ids), member.id, str(msg.get("key", "")),
                                                clean_name(msg.get("name", ""), "Bot")))
                    self._changed()
            elif kind == "remove_bot":
                before = len(self.seats)
                self.seats = [s for s in self.seats if not (s.id == msg.get("seat") and s.owner == member.id)]
                if len(self.seats) != before:
                    self._changed()
            elif kind == "action":
                waiting = self._waiting.get(msg.get("req"))
                if waiting is not None and waiting["member"] == member.id:
                    waiting["action"] = msg.get("action")
                    waiting["event"].set()
            elif kind == "talk":
                session = self.session
            elif kind == "forfeit":
                if msg.get("game") == self.game_id:
                    threading.Thread(target=self.forfeit, args=(member.id,), daemon=True).start()
                return
        if session is not None:
            session.inject_talk(str(msg.get("speaker", "")), str(msg.get("text", "")), str(msg.get("lang", "nl")))

    def _on_close(self, conn: net.Connection) -> None:
        with self._lock:
            member = self.members.pop(conn.member_id, None) if conn.member_id is not None else None
            if member is None:
                return
            if self.phase == "lobby":
                self.seats = [s for s in self.seats if s.owner != member.id]
            playing = self.phase == "game" and self.session is not None
            if not self.closed:
                self._changed()
        if playing:                                   # gone mid-game: they and their bots lose
            self.forfeit(member.id)

    # -- the table, as each guest may see it -------------------------------------

    def _sync(self, session: GameSession, game_id: int) -> None:
        sent: Dict[int, tuple] = {}
        while not self.closed and self.session is session:
            snap = session.snapshot()
            key = (snap.round_number, snap.palifico, snap.current_bid, snap.bidder, snap.to_act,
                   id(snap.feed[-1]) if snap.feed else 0, snap.reveal is not None, snap.finished,
                   tuple(snap.standings), snap.error, snap.awaiting_next_round,
                   tuple((p.dice_count, p.alive) for p in snap.players))
            with self._lock:
                guests = [m for m in self.members.values() if m.conn is not None]
            for m in guests:
                previous = sent.get(m.id)
                if previous is not None and previous[0] == key:
                    continue
                msg, last = self._view(snap, m.id, previous[1] if previous else None, game_id)
                m.conn.send(msg)
                sent[m.id] = (key, last)
            time.sleep(0.1 if not snap.finished else 0.5)

    def _view(self, snap: Snapshot, member_id: int, last_item, game_id: int):
        you = self._seat_names.get(member_id)
        feed = snap.feed
        start = 0
        if last_item is not None:
            for i in range(len(feed) - 1, -1, -1):
                if feed[i] is last_item:
                    start = i + 1
                    break
        new = feed[start:] if last_item is not None else feed[-60:]
        players = [{"name": p.name, "count": p.dice_count, "alive": p.alive,
                    "dice": list(p.dice) if p.name == you else [], "you": p.name == you, "person": p.person}
                   for p in snap.players]
        msg = {"t": "state", "game": game_id, "round": snap.round_number, "palifico": snap.palifico,
               "players": players, "bid": net.encode(snap.current_bid), "bidder": snap.bidder,
               "to_act": snap.to_act, "reveal": net.encode(snap.reveal), "finished": snap.finished,
               "standings": list(snap.standings), "error": snap.error, "awaiting": snap.awaiting_next_round,
               "places": dict(snap.places),
               "feed": [encode_feed(item) for item in new]}
        return msg, (feed[-1] if feed else last_item)


# ---------------------------------------------------------------------------
# a guest
# ---------------------------------------------------------------------------

class _Named:
    def __init__(self, name: str):
        self.name = name


class RemoteSession:
    """A guest's view of the host's game, shaped like a GameSession for the
    game screen: snapshots come from the host, the guest's own turn waits
    for submit()."""
    multiplayer = True
    can_continue = False          # the host starts each next round

    def __init__(self, client: "LobbyClient", game_id: int, order: List[str], you: str, host_name: str):
        self.client = client
        self.game_id = game_id
        self.bots = [_Named(n) for n in order]
        self.human = _Named(you)
        self.host_name = host_name
        self._lock = threading.Lock()
        self._snap = Snapshot()
        self._feed: List[FeedItem] = []
        self._hands: list = []                 # scored hands of every reveal seen (core.revealed_hands)
        self._pending = None
        self._req = None
        self.ended = False
        self.mine: List[str] = [you]           # this phone's seats: the guest and their bots
        self.left = False                      # walked away from this game

    # what the game screen calls
    def snapshot(self) -> Snapshot:
        with self._lock:
            return replace(self._snap, players=list(self._snap.players), feed=list(self._feed),
                           pending=self._pending)

    def submit(self, action) -> None:
        with self._lock:
            req, self._req, self._pending = self._req, None, None
        if req is not None:
            self.client.send({"t": "action", "req": req, "action": net.encode(action)})

    @property
    def running(self) -> bool:
        return not self.ended

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def next(self) -> None:
        pass

    def set_auto(self, auto: bool, delay: Optional[float] = None) -> None:
        pass

    def skip_warmup(self) -> None:
        pass

    def next_round(self) -> None:
        pass

    # what the host sends
    def apply(self, msg: dict) -> List[FeedItem]:
        players = [PlayerView(p["name"], p["count"], p["alive"], tuple(p["dice"]), p["you"], p["person"])
                   for p in msg.get("players", [])]
        reveal = net.decode(msg.get("reveal"))
        new = [decode_feed(item) for item in msg.get("feed", [])]
        with self._lock:
            self._feed.extend(new)
            del self._feed[:-400]
            # A new reveal: score its hands here, from what it shows (the
            # host sends nothing more).
            revealed = reveal.get("round") if reveal is not None else None
            if revealed is not None and not any(h.round == revealed for h in self._hands):
                starter = next((f.player for f in reversed(self._feed)
                                if f.kind == "round" and f.text.split(" ")[1] == str(revealed)), None)
                self._hands.extend(revealed_hands(revealed, reveal, starter))
            self._snap = Snapshot(
                round_number=msg.get("round", 0), palifico=msg.get("palifico", False), players=players,
                current_bid=net.decode(msg.get("bid")), to_act=msg.get("to_act"), reveal=reveal,
                finished=msg.get("finished", False), standings=tuple(msg.get("standings", ())),
                error=msg.get("error"), bidder=msg.get("bidder"),
                awaiting_next_round=msg.get("awaiting", False), hands=list(self._hands),
                places=dict(msg.get("places", {})))
            if self._snap.finished:
                self.ended = True
                self._pending = self._req = None
        return new

    def ask(self, req: int, obs) -> None:
        with self._lock:
            if self.left:
                return
            self._pending = net.decode(obs)
            self._req = req

    def cancel(self, req) -> None:
        """The host took back its question (someone left mid-round)."""
        with self._lock:
            if self._req == req:
                self._pending = self._req = None

    def end(self, reason: str) -> None:
        with self._lock:
            if self.ended:
                return
            self.ended = True
            self._pending = self._req = None
            self._snap = replace(self._snap, finished=True, error=reason, to_act=None)


class LobbyClient:
    """A guest's side of a lobby: joins with the code, keeps the lobby state,
    runs the bots it brought and shows the host's game as a RemoteSession."""

    def __init__(self, name: str, registry: BotRegistry, code: str, local_ips: Optional[List[str]] = None,
                 speak: Optional[Callable[[str, str, str], None]] = None, timeout: float = 5.0,
                 relay=None, memory=None, player_id: Optional[str] = None):
        """`relay`: a perudo_app.relay backend to join over the internet;
        None joins on the local network. `memory`: a core.BotMemory for the
        bots this guest brings."""
        self.registry = registry
        self.memory = memory
        self._saved = None                     # the game whose bots' learning is saved
        self.player_id = player_id
        self._people: Dict[str, str] = {}      # player id -> name of the people in the game
        self.code = net.normalise_code(code)
        self.name = clean_name(name)
        self.speak = speak
        self.lobby: dict = {}
        self.my_id: Optional[int] = None
        self.error: Optional[str] = None
        self.notice: Optional[str] = None
        self.session: Optional[RemoteSession] = None
        self.version = 0
        self.closed = False
        self._bots: Dict[int, Any] = {}
        self._jobs: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._runner = threading.Thread(target=self._run_jobs, name="perudo-guest-bots", daemon=True)
        self._runner.start()
        self._welcome = threading.Event()
        self.online = relay is not None
        if self.online:
            self.conn = online.RelayGuest(relay, self.code, self._on_message, self._on_close).link
        else:
            sock = net.dial(self.code, local_ips)
            self.conn = net.Connection(sock, self._on_message, self._on_close).start()
        self.send({"t": "hello", "name": self.name, "code": self.code, "protocol": net.PROTOCOL,
                   "player_id": self.player_id})
        if not self._welcome.wait(timeout) or self.my_id is None:
            reason = self.error or "The lobby did not answer"
            self.close()
            raise ConnectionError(reason)

    # -- what the guest does ---------------------------------------------------

    def send(self, msg: dict) -> None:
        self.conn.send(msg)

    def rename(self, name: str) -> None:
        self.name = clean_name(name)
        self.send({"t": "rename", "name": self.name})

    def add_bot(self, key: str, name: str) -> None:
        self.send({"t": "add_bot", "key": key, "name": name})

    def remove_bot(self, seat_id: int) -> None:
        self.send({"t": "remove_bot", "seat": seat_id})

    def forfeit(self) -> None:
        """Walk away from the game going on: this guest and their bots lose."""
        session = self.session
        if session is None or session.left:
            return
        with session._lock:
            session.left = True
            session._pending = session._req = None
        self.send({"t": "forfeit", "game": session.game_id})

    def my_name(self) -> str:
        for m in self.lobby.get("members", []):
            if m["id"] == self.my_id:
                return m["name"]
        return self.name

    def close(self) -> None:
        self.closed = True
        if ROUTER.listener == self._on_bot_talk:
            ROUTER.detach(self._on_bot_talk)
        try:
            self.conn.close()
        except AttributeError:
            pass

    # -- the network -------------------------------------------------------------

    def _on_message(self, _conn, msg: dict) -> None:
        kind = msg.get("t")
        if kind == "welcome":
            self.my_id = msg.get("id")
            self._welcome.set()
        elif kind == "reject":
            self.error = msg.get("reason", "The lobby refused")
            self._welcome.set()
        elif kind == "lobby":
            previous = self.lobby
            self.lobby = msg.get("lobby", {})
            if (previous.get("phase") == "game" and self.lobby.get("phase") == "lobby"
                    and self.session is not None):
                self.session.end("The host ended the game")
            self.version += 1
        elif kind == "notice":
            self.notice = msg.get("text")
            self.version += 1
        elif kind == "game_start":
            self._start_game(msg)
        elif kind == "state":
            session = self.session
            if session is not None and msg.get("game") == session.game_id:
                new = session.apply(msg)
                if msg.get("finished") and self.memory is not None and self._saved != session.game_id:
                    self._saved = session.game_id          # what our bots learned, for next time
                    self._jobs.put(lambda bots=list(self._bots.values()), people=dict(self._people):
                                   self.memory.save_all(bots, people))
                if self.speak is not None:             # the lines the host let through, said here too
                    for item in new:
                        if item.kind == "talk":
                            self.speak(item.text, "nl", item.player or "")
        elif kind == "act":
            if msg.get("seat") is None:
                if self.session is not None:
                    self.session.ask(msg.get("req"), msg.get("obs"))
            else:
                self._jobs.put(lambda m=msg: self._bot_act(m))
        elif kind == "call":
            self._jobs.put(lambda m=msg: self._bot_call(m))
        elif kind == "cancel":
            if self.session is not None:
                self.session.cancel(msg.get("req"))

    def _on_close(self, _conn) -> None:
        if not self.closed:
            self.error = self.error or "The host left; the lobby is closed"
            if self.session is not None:
                self.session.end("The host left")
        self.closed = True
        self._welcome.set()
        self.version += 1

    # -- this phone's bots ---------------------------------------------------------

    def _start_game(self, msg: dict) -> None:
        entries = {self.registry.key(e): e for e in self.registry.entries}
        classes = {entries[a["key"]].cls for a in msg.get("seats", []) if a["key"] in entries}
        bots: Dict[int, Any] = {}

        people = msg.get("people") or {}
        self._people = {str(k): str(v) for k, v in people.items()} if isinstance(people, dict) else {}

        def setup():
            _reset_learning(classes)
            for a in msg.get("seats", []):
                entry = entries.get(a["key"])
                if entry is None:
                    continue
                try:
                    bot = entry.cls()
                    bot.memory_name, bot.name = bot.name, a["name"]
                    if self.memory is not None:
                        self.memory.load(bot, self._people)
                    bots[a["seat"]] = bot
                except Exception:
                    pass
            self._bots = bots
            ROUTER.install()
            ROUTER.attach(self._on_bot_talk, self._runner)    # their talk goes to the host

        self._jobs.put(setup)
        self.session = RemoteSession(self, msg.get("game"), msg.get("order", []), msg.get("you", ""),
                                     msg.get("host", ""))
        self.session.mine += [a["name"] for a in msg.get("seats", [])]
        self.version += 1

    def _bot_act(self, msg: dict) -> None:
        bot = self._bots.get(msg.get("seat"))
        action = None
        if bot is not None:
            try:
                action = bot.play(net.decode(msg.get("obs")))
            except Exception:
                action = None
        self.send({"t": "action", "req": msg.get("req"), "action": net.encode(action)})

    def _bot_call(self, msg: dict) -> None:
        bot = self._bots.get(msg.get("seat"))
        method = msg.get("method")
        if bot is None or method not in ("observe", "on_game_start", "on_game_end"):
            return
        fn = getattr(bot, method, None)
        if fn is not None:
            try:
                fn(net.decode(msg.get("arg")))
            except Exception:
                pass

    def _on_bot_talk(self, speaker: str, text: str, lang: str):
        self.send({"t": "talk", "speaker": speaker, "text": text, "lang": lang})
        return False                      # said (or not) by the host's rules, then shown to all

    def _run_jobs(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                job()
            except Exception:
                pass


def join_lobby(name: str, registry: BotRegistry, code: str, relay=None,
               local_ips: Optional[List[str]] = None, speak=None, timeout: float = 6.0,
               memory=None, player_id: Optional[str] = None) -> LobbyClient:
    """Join the lobby with this code wherever it is: over the internet (when
    there is a relay) and on the local network at the same time; the first to
    answer wins."""
    if relay is None:
        return LobbyClient(name, registry, code, local_ips=local_ips, speak=speak, timeout=timeout,
                           memory=memory, player_id=player_id)
    done = threading.Event()
    lock = threading.Lock()
    found: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    def attempt(kind: str) -> None:
        try:
            client = LobbyClient(name, registry, code, local_ips=local_ips, speak=speak, timeout=timeout,
                                 memory=memory, player_id=player_id,
                                 relay=relay if kind == "online" else None)
        except Exception as exc:
            with lock:
                errors[kind] = str(exc) or type(exc).__name__
                if len(errors) == 2:
                    done.set()
            return
        with lock:
            if "client" in found:
                client.close()
                return
            found["client"] = client
        done.set()

    for kind in ("online", "local"):
        threading.Thread(target=attempt, args=(kind,), daemon=True).start()
    done.wait(timeout + 20)
    with lock:
        if "client" in found:
            return found["client"]
        found["client"] = None                 # a late answer is closed
    # A refusal from a lobby that was found says more than "not found".
    for kind in ("online", "local"):
        message = errors.get(kind, "")
        if message and "did not answer" not in message and "no lobby" not in message.lower():
            raise ConnectionError(message)
    raise ConnectionError("No lobby with that code")
