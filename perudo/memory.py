"""
What a bot learns, kept between games and between runs of the app.

The interface for a bot is one class attribute: the names of the attributes
that hold what it learned.

    class MyBot:
        _LEARNED = ("stats",)          # kept for it between games

        def __init__(self):
            self.name = "My Bot"
            self.stats = defaultdict(lambda: [0, 0])   # player -> [calls, right]

The host (the app, a tournament) saves those attributes after a game and
puts them back into a new bot of the same kind before the next one. Bots
never read or write files themselves: they only keep their learning in
those attributes, of plain data -- numbers, strings, lists, tuples, dicts
(any keys) and objects of their own classes holding the same.

restore() marks the bot with `_memory_restored = True`: a bot that sets
its starting values late (at its first move) must leave the learned
attributes alone then.

Saving turns them into JSON (save); restoring pours the saved values back
into the new bot's own attributes (restore): into its dicts and objects
where it has them, so a defaultdict keeps its default and an object its
class and methods. A value whose shape changed since it was saved (a newer
bot) is left as the new bot made it.
"""

from __future__ import annotations

import ast
import inspect
import sys
from typing import Any, Dict, Mapping, Optional

FORMAT = 1


def learned_names(bot: Any) -> tuple:
    names = getattr(bot, "_LEARNED", ())
    return tuple(n for n in names if isinstance(n, str))


def save(bot: Any) -> Dict[str, Any]:
    """The bot's learned attributes as JSON-ready data."""
    return {"format": FORMAT,
            "learned": {name: _dump(getattr(bot, name), frozenset({id(bot)}))
                        for name in learned_names(bot) if hasattr(bot, name)}}


def restore(bot: Any, data: Mapping[str, Any]) -> bool:
    """Put saved learning back into a (new) bot. False when there was none
    for it (or of another format)."""
    if not isinstance(data, Mapping) or data.get("format") != FORMAT:
        return False
    saved = data.get("learned") or {}
    done = False
    for name in learned_names(bot):
        if name not in saved:
            continue
        current = getattr(bot, name, None)
        value = _fill(current, saved[name], _home(bot))
        if value is not current:
            setattr(bot, name, value)
        done = True
    if done:
        bot._memory_restored = True   # a bot that sets starting values late keeps these
    return done


def _home(bot: Any) -> Optional[dict]:
    """The globals of the bot's own module (a bot file the loader ran
    without registering it as a module has its classes only there)."""
    init = getattr(type(bot), "__init__", None)
    return getattr(init, "__globals__", None)


# ---------------------------------------------------------------------------

def _dump(value: Any, above: frozenset = frozenset()) -> Any:
    """`above`: the objects this one sits in (the bot among them). A
    reference back up to one of them (a profiler's owner) is not kept: the
    new bot has its own."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if id(value) in above:
        return {"back": True}
    inner = above | {id(value)}
    if isinstance(value, Mapping):
        return {"map": [[_key(k), _dump(v, inner)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_dump(v, inner) for v in value]}
    if isinstance(value, list):
        return {"list": [_dump(v, inner) for v in value]}
    if isinstance(value, (set, frozenset)):
        return {"set": [_dump(v, inner) for v in value]}
    if hasattr(value, "__dict__"):
        return {"object": {k: _dump(v, inner) for k, v in vars(value).items() if not callable(v)},
                "class": _class_name(value)}
    if hasattr(value, "__slots__"):
        return {"object": {k: _dump(getattr(value, k), inner) for k in value.__slots__ if hasattr(value, k)},
                "class": _class_name(value)}
    raise TypeError(f"cannot keep a {type(value).__name__}")


def _key(key: Any) -> str:
    """Dict keys as text: repr of plain keys (numbers, strings, tuples of them)."""
    text = repr(key)
    ast.literal_eval(text)            # only keys that come back the same
    return text


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}:{cls.__qualname__}"


def _new(name: Optional[str], home: Optional[dict]) -> Any:
    """A fresh object of a class the bot's own module (`home`, its globals)
    or a loaded module has -- nothing is imported from saved data -- made
    by its constructor so it has its own defaults (with None for arguments
    it insists on: the saved values replace them); None when impossible."""
    if not name or ":" not in name:
        return None
    module_name, qualname = name.split(":", 1)
    first, *rest = qualname.split(".")
    module = sys.modules.get(module_name)
    obj = (home or {}).get(first) if home and home.get("__name__") == module_name else None
    if obj is None and module is not None:
        obj = getattr(module, first, None)
    for part in rest:
        obj = getattr(obj, part, None)
    if not isinstance(obj, type):
        return None
    try:
        return obj()
    except TypeError:
        pass
    except Exception:
        return None
    try:
        params = [p for p in inspect.signature(obj).parameters.values()
                  if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        return obj(*[None] * len(params))
    except Exception:
        try:
            return obj.__new__(obj)
        except Exception:
            return None


def _fill(current: Any, data: Any, home: Optional[dict] = None) -> Any:
    """The saved value, poured into `current` where it has the shape for it;
    returns what the attribute should hold."""
    if not isinstance(data, dict):
        return data                                   # a number, string, None
    if data.get("back"):
        return current                                # a reference up: the new bot's own
    if "map" in data:
        target = current if isinstance(current, dict) else {}
        for text, item in data["map"]:
            try:
                key = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                continue
            existing = target.get(key) if key in target else (target[key] if hasattr(target, "default_factory")
                                                                and target.default_factory else None)
            target[key] = _fill(existing, item, home)
        return target
    if "tuple" in data:
        return tuple(_fill(None, v, home) for v in data["tuple"])
    if "list" in data:
        if isinstance(current, list) and len(current) == len(data["list"]):
            return [_fill(c, v, home) for c, v in zip(current, data["list"])]
        return [_fill(None, v, home) for v in data["list"]]
    if "set" in data:
        return set(_fill(None, v, home) for v in data["set"])
    if "object" in data:
        if current is None:
            current = _new(data.get("class"), home)   # e.g. a profile kept in a plain dict
        if current is None or not (hasattr(current, "__dict__") or hasattr(current, "__slots__")):
            return current                            # nothing of the bot's own to pour it into
        for name, item in data["object"].items():
            if hasattr(current, name):
                setattr(current, name, _fill(getattr(current, name), item, home))
            else:
                try:
                    setattr(current, name, _fill(None, item, home))
                except AttributeError:
                    pass
        return current
    return current
