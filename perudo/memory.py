"""
Learned knowledge that survives a new bot instance.

The competition platform may create a new bot object for every game, which
would throw away everything a bot learned in instance variables. Bots that
use SharedMemory keep their learned state in a class-level store instead:
the first instance creates it, every later instance (same class, same name,
same process) binds to it, and every update while learning lands in it. No
disk access is involved.

A bot lists the attributes that hold learned state in _LEARNED; they must
be mutable objects (dicts, lists, profilers) that are updated in place.
Per-round state stays per instance.
"""

from __future__ import annotations


class SharedMemory:
    SHARED_MEMORY = True
    _LEARNED: tuple = ()

    @classmethod
    def reset_memory(cls) -> None:
        """Forget everything instances of this class learned (benchmarks use this)."""
        cls._MEMORY = {}

    def _ensure_memory(self) -> None:
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
