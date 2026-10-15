import json
import random
from collections import defaultdict

from perudo import memory
from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame
from perudo.models import BidEvent, CalzaEvent, DudoEvent


class Seen:
    def __init__(self, name):
        self.name = name
        self.calls = 0
        self.faces = [0] * 6


class Profiler:
    def __init__(self, owner):
        self._owner = owner                      # a reference back up: not kept
        self.people = {}                         # name -> Seen, a plain dict


class Learner:
    _LEARNED = ("counts", "pairs", "profiler", "rate")

    def __init__(self):
        self.name = "Learner"
        self.counts = defaultdict(lambda: [0, 0])
        self.pairs = {}
        self.profiler = Profiler(self)
        self.rate = 0.5
        self.scratch = "not learned"

    def play(self, state):
        return state.legal_actions[0]


def test_what_a_bot_learned_comes_back_in_a_new_one():
    old = Learner()
    old.counts["Ann"][1] += 3
    old.pairs[("Ann", 5)] = (2, 0.25)
    old.profiler.people["Ann"] = Seen("Ann")
    old.profiler.people["Ann"].calls = 4
    old.rate = 0.75
    old.scratch = "changed"
    data = json.loads(json.dumps(memory.save(old)))         # it survives JSON

    new = Learner()
    assert memory.restore(new, data) and new._memory_restored
    assert new.counts["Ann"] == [0, 3] and new.counts["Bob"] == [0, 0]   # still a defaultdict
    assert new.pairs == {("Ann", 5): (2, 0.25)}
    seen = new.profiler.people["Ann"]
    assert isinstance(seen, Seen) and seen.calls == 4 and seen.faces == [0] * 6
    assert new.profiler._owner is new                        # its own owner, not the old bot
    assert new.rate == 0.75 and new.scratch == "not learned"


def test_nothing_saved_changes_nothing():
    bot = Learner()
    assert not memory.restore(bot, {})
    assert not memory.restore(bot, {"format": 999, "learned": {"rate": 1}})
    assert bot.rate == 0.5 and not hasattr(bot, "_memory_restored")


def _play(bots, seeds):
    moves = []
    for seed in seeds:
        for b in bots:                                       # the same choices on both sides
            for attr in ("_rng", "rng"):
                if isinstance(getattr(b, attr, None), random.Random):
                    setattr(b, attr, random.Random(seed * 7919 + len(b.name)))
        PerudoGame(bots=bots, seed=seed, listeners=[
            lambda e: moves.append(repr(e)) if isinstance(e, (BidEvent, DudoEvent, CalzaEvent)) else None,
        ]).play()
    return moves


def test_learning_bots_play_on_the_same_after_a_restore():
    classes = discover_bot_classes_by_name()
    names = ["CarlosBot", "PokerfaceBot", "TheDonBot", "BluffoBoyyo", "CarlosDosBot", "SniperBot"]
    for cls in classes.values():
        if hasattr(cls, "reset_memory"):
            cls.reset_memory()
    bots = [classes[n]() for n in names]
    _play(bots, [11, 12])                                    # they learn
    saved = {b.name: json.loads(json.dumps(memory.save(b))) for b in bots}
    assert all(saved[b.name]["learned"] for b in bots if memory.learned_names(b))
    went_on = _play(bots, [21])

    for cls in classes.values():                             # the app starts again
        if hasattr(cls, "reset_memory"):
            cls.reset_memory()
    fresh = [classes[n]() for n in names]
    for b in fresh:
        memory.restore(b, saved[b.name])
    assert _play(fresh, [21]) == went_on
