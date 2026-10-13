"""Every bot has something to say, in its own words, in every situation it meets."""

import importlib
import random

import pytest

import perudo.talk as talk
from perudo.bot_loader import discover_bot_classes
from perudo.game import PerudoGame
from perudo.models import Bid, BidEvent, DieLostEvent, DudoEvent, GameStartInfo, PublicPlayer

CONTESTANTS = ("contestants.hakaman", "contestants.wouluh")


def _classes():
    found = {c.__name__: c for c in discover_bot_classes()}
    for mod in CONTESTANTS:
        m = importlib.import_module(mod)
        for c in vars(m).values():
            if isinstance(c, type) and c.__module__ == m.__name__ and hasattr(c, "play"):
                found[c.__name__] = c
    return found


@pytest.fixture
def talking(monkeypatch):
    """Every bot talks every time it can; records (bot class, key, quoted)."""
    said = []
    monkeypatch.setattr(talk, "QUOTE_CHANCE", 1.0)
    monkeypatch.setattr(talk, "tts_enabled", lambda: True)
    monkeypatch.setattr(talk, "talk_enabled", lambda: False)
    monkeypatch.setattr(talk, "speak", lambda *a, **k: None)
    original = talk.Mouth.say

    def say(mouth, key, announce=None, **fmt):
        lines = mouth.quotes.get(key) or []
        for line in lines:
            line.format(**fmt)               # every line fits what this call site passes
        said.append((type(mouth.bot).__name__, key, bool(lines)))
        return original(mouth, key, announce, **fmt)

    monkeypatch.setattr(talk.Mouth, "say", say)
    return said


def test_every_bot_has_lines_for_every_situation_it_meets(talking):
    classes = _classes()
    names = sorted(classes)
    rng = random.Random(20261013)
    for _ in range(60):
        pick = rng.sample(names, rng.randint(2, 6))
        PerudoGame(bots=[classes[n]() for n in pick], seed=rng.randrange(2**31)).play()
    silent = sorted({(bot, key) for bot, key, quoted in talking if not quoted})
    assert silent == []
    heard = {}
    for bot, key, _ in talking:
        heard.setdefault(bot, set()).add(key)
    assert set(heard) == set(names)              # nobody is mute
    for bot, keys in heard.items():
        assert len(keys) >= 10, bot


@pytest.mark.parametrize("name", sorted(_classes()))
def test_every_bot_reacts_to_what_happens_to_it(talking, name):
    bot = _classes()[name]()
    me = bot.name
    players = (PublicPlayer(0, me, 2, True), PublicPlayer(1, "Someone", 5, True))
    if hasattr(bot, "on_game_start"):
        bot.on_game_start(GameStartInfo(players=players, my_id=0))
    bot.observe(BidEvent(player=me, bid=Bid(3, 4)))
    bot.observe(DudoEvent(player="Someone"))
    bot.observe(DieLostEvent(player=me, dice_count=1))
    keys = {key for _, key, quoted in talking if quoted}
    assert {"dudo_on_me", "last_die"} <= keys
