"""
Shared trash-talk helper for bots that like to talk.

Zet TALK_TO_ME=true om de praatjes te printen, en TTS=true om ze ook hardop
te horen (zie perudo.tts).

Gebruik vanuit een bot::

    from perudo.talk import Mouth, fmt_bid

    QUOTES = {"open": ["{bid}. Kom maar op!"], "won": ["GG EZ."]}

    class MyBot:
        def __init__(self):
            self.name = "My Bot"
            self.mouth = Mouth(self, QUOTES)

        def play(self, state):
            ...
            self.mouth.say("open", announce=fmt_bid(bid), bid=fmt_bid(bid))

        def observe(self, event):
            self.mouth.react(event)  # reveals, Dudo on you, eliminations, game end
            # (or self.mouth.notice(event) when the bot comments on the rest itself)

Only QUOTE_CHANCE of the situations get a quote, so the table doesn't drown
in chatter. When a bot makes a move, pass `announce` (the bid, "Dudo!" or
"Calza!"): the move is then always said, as a bare announcement when no quote
is picked. Quotes contain the bid themselves. Situations without quotes are
silently skipped, so a bot only has to write lines for the moments it cares
about. The keys `react()` uses:

    dudo_on_me         someone calls Dudo on my bid       {who}
    dudo_right         my Dudo was right                  {bid} {actual} {bidder} {challenger}
    dudo_wrong         my Dudo was wrong                  (same)
    bid_held           my bid got Dudo'd and held         (same)
    bid_caught         my bid got Dudo'd and was false    (same)
    calza_right        my Calza was exact                 (same)
    calza_wrong        my Calza missed                    (same)
    others_bluff_caught / others_dudo_wrong               (same; between two other players)
    others_calza_right / others_calza_wrong               (same; someone else's Calza)
    last_die           I am down to my last die           (no fields)
    eliminated_me / eliminated_other                      {who}
    won / lost_game                                       {winner}
"""

from __future__ import annotations

import os
import random
from typing import Any

from perudo.models import (
    PACO,
    Bid,
    BidEvent,
    DieLostEvent,
    DudoEvent,
    Event,
    GameEndedEvent,
    Observation,
    PlayerEliminatedEvent,
    RevealEvent,
)
from perudo.tts import speak, tts_enabled

# Fraction of situations in which a bot says a quote instead of nothing (or
# just its move).
QUOTE_CHANCE = 0.25


def talk_enabled() -> bool:
    return os.environ.get("TALK_TO_ME", "").strip().lower() in ("1", "true", "yes", "on")


def fmt_bid(bid: Bid | None) -> str:
    if bid is None:
        return "niks"
    return f"{bid.quantity}x {'Paco' if bid.face == PACO else bid.face}"


def last_bidder(state: Observation, default: str = "maat") -> str:
    for record in reversed(state.history):
        if isinstance(record.action, Bid):
            return record.player
    return default


class Mouth:
    def __init__(self, bot: Any, quotes: dict[str, list[str]], lang: str = "nl"):
        self.bot = bot  # read bot.name at speak time: copies get renamed after __init__
        self.quotes = quotes
        self.lang = lang
        # Own RNG, so chatter never shifts a seeded game's random stream.
        self._rng = random.Random()
        self._recent: list[str] = []
        self._last_bidder: str | None = None

    def say(self, key: str, announce: str | None = None, **fmt: object) -> None:
        """Maybe say a quote for `key`; always say `announce` (a move) if given."""
        talk = talk_enabled()
        if not (talk or tts_enabled()):
            return
        if self.quotes.get(key) and self._rng.random() < QUOTE_CHANCE:
            options = [q for q in self.quotes[key] if q not in self._recent] or list(self.quotes[key])
            quote = self._rng.choice(options)
            self._recent = (self._recent + [quote])[-6:]
            line = quote.format(**fmt)
        elif announce:
            line = announce
        else:
            return
        if talk:
            print(f"  [{self.bot.name}] {line}", flush=True)
        speak(line, lang=self.lang, voice_seed=self.bot.name)

    def react(self, event: Event) -> None:
        """Comment on public events that involve this bot."""
        me = self.bot.name
        self.notice(event)
        if isinstance(event, RevealEvent):
            key = self._reveal_key(event, me)
            if key:
                self.say(
                    key,
                    bid=fmt_bid(event.current_bid),
                    actual=event.actual_count,
                    bidder=event.bidder,
                    challenger=event.challenger,
                )
        elif isinstance(event, PlayerEliminatedEvent):
            self.say("eliminated_me" if event.player == me else "eliminated_other", who=event.player)
        elif isinstance(event, GameEndedEvent):
            self.say("won" if event.winner == me else "lost_game", winner=event.winner)

    def notice(self, event: Event) -> None:
        """Only the moments that happen to this bot itself: a Dudo on its bid
        ("dudo_on_me") and its last die ("last_die"). For bots that comment
        on reveals, eliminations and the end of the game themselves."""
        me = self.bot.name
        if isinstance(event, BidEvent):
            self._last_bidder = event.player
        elif isinstance(event, DudoEvent):
            if self._last_bidder == me and event.player != me:
                self.say("dudo_on_me", who=event.player)
        elif isinstance(event, RevealEvent):
            self._last_bidder = None
        elif isinstance(event, DieLostEvent):
            if event.player == me and event.dice_count == 1:
                self.say("last_die")

    @staticmethod
    def _reveal_key(event: RevealEvent, me: str) -> str | None:
        if event.challenge_type == "calza":
            exact = event.result == "correct"
            if event.challenger != me:
                return "others_calza_right" if exact else "others_calza_wrong"
            return "calza_right" if exact else "calza_wrong"
        held = event.result == "bidder_correct"
        if event.challenger == me:
            return "dudo_wrong" if held else "dudo_right"
        if event.bidder == me:
            return "bid_held" if held else "bid_caught"
        return "others_dudo_wrong" if held else "others_bluff_caught"
