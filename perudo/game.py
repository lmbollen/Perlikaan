"""
The Perudo game engine.

This is the ONLY place game rules live. Bots never see this module's
internal state -- they only ever receive an `Observation` (models.py),
built fresh each turn from data the current player is allowed to know.

Usage (see run_interactive.py / run_simulation.py for real examples)::

    game = PerudoGame(bots=[bot_a, bot_b, bot_c], seed=12345)
    result = game.play()
"""

from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

from perudo.models import (
    PACO,
    ActionRecord,
    Bid,
    BidEvent,
    Calza,
    CalzaEvent,
    DieGainedEvent,
    DieLostEvent,
    Dudo,
    DudoEvent,
    Event,
    GameEndedEvent,
    GameResult,
    GameStartInfo,
    IllegalActionError,
    Observation,
    PlayerEliminatedEvent,
    PublicPlayer,
    RevealEvent,
    RoundEndedEvent,
)
from perudo.rules import count_matches, legal_actions

STARTING_DICE = 5
MAX_DICE = 5


def _call(bot: Any, method_name: str, *args: Any) -> None:
    """Invoke an optional bot callback if the bot implements it."""
    method = getattr(bot, method_name, None)
    if method is not None:
        method(*args)


class Forfeit(Exception):
    """Raised from a bot's play() when its seat leaves the game (a player
    walked away from the table). The seat loses at once; see
    PerudoGame.forfeit."""


class Interrupt(Exception):
    """Raised from a bot's play() to give up its turn because another seat
    has left (forfeit_pending()): the round is abandoned at once instead of
    after this turn."""


class _RoundAbandoned(Exception):
    """A seat left mid-round: the round ends without a reveal."""

    def __init__(self, next_idx: int, palifico: bool):
        super().__init__()
        self.next_idx = next_idx
        self.palifico = palifico


@dataclass
class _PlayerState:
    seat_id: int
    bot: Any
    name: str
    dice_count: int = STARTING_DICE
    dice: List[int] = field(default_factory=list)
    alive: bool = True
    has_had_palifico: bool = False


class PerudoGame:
    """
    One complete Perudo game (from 5 dice each to a single winner).

    Create a fresh PerudoGame for every game you play -- it holds all the
    mutable per-game state. Bot *instances* passed in, however, may be
    reused across many PerudoGame objects (that's how bots "remember"
    opponents across a simulation).
    """

    def __init__(
        self,
        bots: Sequence[Any],
        seed: int,
        before_turn: Optional[Callable[[Observation], None]] = None,
        before_reveal: Optional[Callable[..., None]] = None,
        listeners: Optional[Sequence[Callable[[Event], None]]] = None,
    ):
        if len(bots) < 2:
            raise ValueError("Perudo requires at least 2 bots.")

        names = [bot.name for bot in bots]
        if len(set(names)) != len(names):
            raise ValueError(f"Bot names must be unique, got: {names}")

        self.seed = seed
        self.rng = random.Random(seed)
        self.before_turn = before_turn
        self.before_reveal = before_reveal
        self.listeners: List[Callable[[Event], None]] = list(listeners or [])

        # Randomize seating for this game; identity travels with `name`.
        order = list(range(len(bots)))
        self.rng.shuffle(order)
        self.players: List[_PlayerState] = [
            _PlayerState(seat_id=i, bot=bots[order[i]], name=bots[order[i]].name)
            for i in range(len(bots))
        ]

        self.dice_recovery_enabled = True
        self.round_number = 0
        self._eliminated_order: List[str] = []
        self._forfeits: set = set()           # names that asked to leave, not yet out
        self._forfeit_lock = threading.Lock()

    def forfeit(self, name: str) -> None:
        """Take a seat out of the game: it loses at once, below everyone
        still playing. Safe to call from any thread; it takes effect before
        the next turn. A round going on is abandoned (nobody loses a die)
        and the next one is rolled without that seat."""
        with self._forfeit_lock:
            self._forfeits.add(name)

    # -- public API ---------------------------------------------------

    def play(self) -> GameResult:
        for player in self.players:
            info = GameStartInfo(players=self._public_snapshot(), my_id=player.seat_id)
            _call(player.bot, "on_game_start", info)

        starter_idx = self.rng.randrange(len(self.players))
        palifico_next = False

        while True:
            if self._apply_forfeits():
                if not self.players[starter_idx].alive:
                    starter_idx = self._next_alive(starter_idx)
            if self._alive_count() <= 1:
                break
            self.round_number += 1
            self._roll_round()
            try:
                next_starter_idx, palifico_next = self._play_round(
                    starter_idx, palifico_next
                )
            except _RoundAbandoned as abandoned:
                self._apply_forfeits()
                next_starter_idx, palifico_next = abandoned.next_idx, abandoned.palifico
                if self._alive_count() > 1 and not self.players[next_starter_idx].alive:
                    next_starter_idx = self._next_alive(next_starter_idx)
            starter_idx = next_starter_idx

            if self._alive_count() > 1:
                self._emit(
                    RoundEndedEvent(
                        next_starting_player=self.players[starter_idx].name,
                        palifico_next=palifico_next,
                    )
                )

        winner = next(p for p in self.players if p.alive)
        standings = (winner.name, *reversed(self._eliminated_order))
        result = GameResult(
            winner=winner.name, standings=standings, rounds_played=self.round_number
        )
        self._emit(GameEndedEvent(winner=winner.name, standings=standings))
        for player in self.players:
            _call(player.bot, "on_game_end", result)
        return result

    # -- round / turn loop ---------------------------------------------

    def _play_round(self, starter_idx: int, palifico: bool) -> Tuple[int, bool]:
        current_bid: Optional[Bid] = None
        palifico_face: Optional[int] = None
        bidder_idx: Optional[int] = None
        history: List[ActionRecord] = []
        current_idx = starter_idx
        total_dice = sum(p.dice_count for p in self.players if p.alive)

        while True:
            actions = tuple(
                legal_actions(current_bid, palifico, palifico_face, total_dice)
            )
            observation = self._build_observation(
                current_idx, current_bid, palifico, tuple(history), actions
            )

            if self.forfeit_pending():
                raise _RoundAbandoned(current_idx, palifico)
            if self.before_turn is not None:
                self.before_turn(observation)
            if self.forfeit_pending():
                raise _RoundAbandoned(current_idx, palifico)

            player = self.players[current_idx]
            try:
                action = player.bot.play(observation)
            except Forfeit:
                self.forfeit(player.name)
                raise _RoundAbandoned(current_idx, palifico)
            except Interrupt:
                if self.forfeit_pending():
                    raise _RoundAbandoned(current_idx, palifico)
                continue                              # nobody left after all: ask again

            if action not in actions:
                raise IllegalActionError(player.name, action, current_bid, actions)

            history.append(ActionRecord(player=player.name, action=action))

            if isinstance(action, Bid):
                current_bid = action
                bidder_idx = current_idx
                if palifico and palifico_face is None:
                    palifico_face = action.face
                self._emit(BidEvent(player=player.name, bid=action))
                current_idx = self._next_alive(current_idx)
                continue

            if isinstance(action, Dudo):
                self._emit(DudoEvent(player=player.name))
                assert current_bid is not None and bidder_idx is not None
                return self._resolve_dudo(current_idx, bidder_idx, current_bid, palifico)

            if isinstance(action, Calza):
                self._emit(CalzaEvent(player=player.name))
                assert current_bid is not None and bidder_idx is not None
                return self._resolve_calza(
                    current_idx, bidder_idx, current_bid, palifico
                )

            raise TypeError(f"Unknown action type returned: {action!r}")

    # -- Dudo / Calza resolution ----------------------------------------

    def _resolve_dudo(
        self, caller_idx: int, bidder_idx: int, current_bid: Bid, palifico: bool
    ) -> Tuple[int, bool]:
        actual = self._count_on_table(current_bid.face, palifico)
        bidder_correct = actual >= current_bid.quantity
        loser_idx = caller_idx if bidder_correct else bidder_idx
        result = "bidder_correct" if bidder_correct else "bidder_wrong"

        if self.before_reveal is not None:
            self.before_reveal(
                challenge_type="dudo",
                challenger=self.players[caller_idx].name,
                bidder=self.players[bidder_idx].name,
                current_bid=current_bid,
            )

        self._emit(
            RevealEvent(
                dice=self._all_dice_snapshot(),
                current_bid=current_bid,
                actual_count=actual,
                challenge_type="dudo",
                challenger=self.players[caller_idx].name,
                bidder=self.players[bidder_idx].name,
                result=result,
            )
        )

        palifico_triggered = self._lose_die(loser_idx)
        next_starter = (
            loser_idx if self.players[loser_idx].alive else self._next_alive(loser_idx)
        )
        return next_starter, palifico_triggered

    def _resolve_calza(
        self, caller_idx: int, bidder_idx: int, current_bid: Bid, palifico: bool
    ) -> Tuple[int, bool]:
        actual = self._count_on_table(current_bid.face, palifico)
        correct = actual == current_bid.quantity
        result = "correct" if correct else "wrong"

        if self.before_reveal is not None:
            self.before_reveal(
                challenge_type="calza",
                challenger=self.players[caller_idx].name,
                bidder=self.players[bidder_idx].name,
                current_bid=current_bid,
            )

        self._emit(
            RevealEvent(
                dice=self._all_dice_snapshot(),
                current_bid=current_bid,
                actual_count=actual,
                challenge_type="calza",
                challenger=self.players[caller_idx].name,
                bidder=self.players[bidder_idx].name,
                result=result,
            )
        )

        palifico_triggered = False
        if correct:
            if self.dice_recovery_enabled:
                self._gain_die(caller_idx)
            # else: recovery disabled, correct Calza does nothing.
        else:
            palifico_triggered = self._lose_die(caller_idx)

        next_starter = (
            caller_idx
            if self.players[caller_idx].alive
            else self._next_alive(caller_idx)
        )
        return next_starter, palifico_triggered

    # -- dice / elimination helpers --------------------------------------

    def _lose_die(self, idx: int) -> bool:
        """Remove one die from a player. Returns True if this triggers Palifico."""
        player = self.players[idx]
        before = player.dice_count
        player.dice_count -= 1
        self._emit(DieLostEvent(player=player.name, dice_count=player.dice_count))

        if player.dice_count == 0:
            player.alive = False
            self._eliminated_order.append(player.name)
            self._emit(PlayerEliminatedEvent(player=player.name))
            if self.dice_recovery_enabled:
                self.dice_recovery_enabled = False
            return False

        if before == 2 and not player.has_had_palifico:
            player.has_had_palifico = True
            return True

        return False

    def forfeit_pending(self) -> bool:
        """Has a seat still in the game asked to leave? (Any thread.)"""
        if not self._forfeits:
            return False
        with self._forfeit_lock:
            names = set(self._forfeits)
        return any(p.alive and p.name in names for p in self.players)

    def _apply_forfeits(self) -> bool:
        """Put the seats that left out of the game, in seat order. The last
        seat standing is never put out: it has won. True if anyone left."""
        with self._forfeit_lock:
            names, self._forfeits = self._forfeits, set()
        left = False
        for player in self.players:
            if player.name in names and player.alive and self._alive_count() > 1:
                player.alive = False
                player.dice_count = 0
                player.dice = []
                self._eliminated_order.append(player.name)
                self._emit(PlayerEliminatedEvent(player=player.name))
                self.dice_recovery_enabled = False
                left = True
        return left

    def _gain_die(self, idx: int) -> None:
        player = self.players[idx]
        if player.dice_count < MAX_DICE:
            player.dice_count += 1
            self._emit(DieGainedEvent(player=player.name, dice_count=player.dice_count))

    def _count_on_table(self, face: int, palifico: bool) -> int:
        all_dice = [d for p in self.players if p.alive for d in p.dice]
        return count_matches(all_dice, face, palifico)

    def _all_dice_snapshot(self):
        return tuple(
            (p.name, tuple(p.dice)) for p in self.players if p.alive
        )

    def _roll_round(self) -> None:
        for player in self.players:
            if player.alive:
                player.dice = [self.rng.randint(1, 6) for _ in range(player.dice_count)]
            else:
                player.dice = []

    # -- seating / turn order --------------------------------------------

    def _alive_count(self) -> int:
        return sum(1 for p in self.players if p.alive)

    def _next_alive(self, idx: int) -> int:
        n = len(self.players)
        i = (idx + 1) % n
        while not self.players[i].alive:
            i = (i + 1) % n
        return i

    # -- observation building ---------------------------------------------

    def _public_snapshot(self) -> tuple:
        return tuple(
            PublicPlayer(id=p.seat_id, name=p.name, dice_count=p.dice_count, alive=p.alive)
            for p in self.players
        )

    def _build_observation(
        self,
        idx: int,
        current_bid: Optional[Bid],
        palifico: bool,
        history: tuple,
        actions: tuple,
    ) -> Observation:
        player = self.players[idx]
        return Observation(
            my_id=idx,
            my_dice=tuple(sorted(player.dice)),
            my_dice_count=player.dice_count,
            players=self._public_snapshot(),
            current_bid=current_bid,
            palifico=palifico,
            round_number=self.round_number,
            history=history,
            legal_actions=actions,
        )

    # -- event fan-out ---------------------------------------------------

    def _emit(self, event: Event) -> None:
        for player in self.players:
            _call(player.bot, "observe", event)
        for listener in self.listeners:
            listener(event)


def run_game(game: PerudoGame) -> GameResult:
    """Trivial convenience wrapper: game.play()."""
    return game.play()
