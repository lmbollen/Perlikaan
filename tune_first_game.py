"""
Tune a bot's starting parameters on one game, one lost die at a time.

The game for --seed is played with fresh bots (no learned memory) and every
bot's own randomness seeded, so it replays exactly. Then, for the first die
our bot loses:

  * the decision that cost the die is the bot's last call in that round (the
    bid that got caught, or the wrong Dudo / Calza);
  * only a call the bot could have known was bad is tuned. That is judged from
    its own dice and chance alone, never from the others' bids (they may have
    bluffed or not, the bot cannot know): every legal action gets its chance
    of not losing a die, and the call counts as a mistake only when it was
    more likely wrong than right (below --likely) and another action was at
    least --margin more likely to be right. Otherwise the die was bad luck or
    a bluff the bot could not see, and it is skipped;
  * every parameter in PARAMS is moved a little at a time (smallest change
    first), and the game is replayed up to that decision;
  * a change is accepted when all of the bot's earlier calls stay exactly the
    same and that call becomes a correct one, judged with the dice that were
    really on the table: a true bid, a Dudo on a false bid or a Calza on an
    exact one.

The game is then replayed with the new values (it changes from that call on)
and the next lost die is handled the same way. Each parameter is moved up and
down in growing steps until the call is fixed or an earlier call starts to
change; the smallest fix over all parameters wins. The search stops at a lost
die that no parameter fixes before earlier calls start changing. A lost die
that no change touches at all (bad luck: no parameter makes the bot see it)
is skipped. It writes nothing: it prints the values to put in the bot.

Usage:
    uv run python tune_first_game.py --bot CarlosBot --seed 12345
    uv run python tune_first_game.py --bot PokerfaceBot --seed 7 --players 4
    uv run python tune_first_game.py --bot TheDonBot --seed 7 --opponents Trihard SniperBot SonnyBoy
"""

from __future__ import annotations

import argparse
import math
import random
from multiprocessing import Pool

from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame
from perudo.models import Bid, Calza, DieLostEvent, Dudo, RevealEvent
from perudo.rules import count_matches

# Opponents drawn from when --opponents is not given.
POOL = ("PokerfaceBot", "CarlosBot", "TheDonBot", "Trihard", "SniperBot", "Fabulous",
        "OppaStyleBot", "SonnyBoy")

# Starting parameters per bot: (name, where). where is "class" (a class
# attribute), "global" (a module-level constant of the bot's file), or a key
# / index into a dict or tuple class attribute.
PARAMS = {
    "CarlosBot": [
        ("RESPONSE_SHIFT", "class"), ("OTHER_WEIGHT", "class"), ("CALZA_FREE_PASS", "class"),
        ("CAND_MIN", "class"), ("OUR_RAISE_RISK", "class"), ("TUNED_CHALLENGE", "class"),
        ("TUNED_CALZA", "class"), ("TUNED_RAISE_P", "class"),
        ("TUNED_DUDO", 0), ("TUNED_DUDO", 1),
    ],
    "PokerfaceBot": [
        ("TARGET", "class"), ("MIN_ACCEPT", "class"), ("HINT_PER_BID", "class"),
        ("BLUFF_RATE", "class"), ("BLUFF_TARGET", "class"), ("CALZA_AT", "class"),
        ("CALZA_FREE_PASS", "class"), ("CALZA_RISK", "class"), ("LOOKAHEAD_MIN", "class"),
        ("DUDO_SOFTNESS", "class"), ("TUNED_CHALLENGE", "class"), ("TUNED_CALZA", "class"),
        ("TUNED_DUDO_AT", "class"),
        ("TUNED_EXCESS", "nat-choice"), ("TUNED_EXCESS", "nat-follow"),
        ("TUNED_EXCESS", "paco-choice"), ("TUNED_EXCESS", "paco-follow"),
    ],
    "TheDonBot": [
        ("ATTACK_BONUS", "class"), ("PROTECT_WEIGHT", "class"), ("PROTECT_MIN_TRUE", "class"),
        ("SPARE_PENALTY", "class"),
        ("OPP_CHALLENGE_THRESHOLD", "global"), ("OPP_CHALLENGE_SOFTNESS", "global"),
        ("CONTINUE_VALUE", "global"), ("GAIN_VALUE", "global"), ("FREE_PASS_VALUE", "global"),
        ("HONESTY_DEFAULT", "global"), ("HONESTY_FLOOR", "global"), ("PACO_MENTION_WEIGHT", "global"),
    ],
}

# Relative steps tried per parameter, smallest first (both directions).
STEPS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
# Absolute steps for a parameter that is 0.
ZERO_STEPS = (0.02, 0.05, 0.1, 0.2)
# A losing call is a mistake the bot could know about when, from its own dice
# and chance alone, it was less likely to be right than LIKELY (--likely) and
# MARGIN less likely than the best legal action (--margin). A call that was
# more likely right than wrong is defensible, however it turned out.
LIKELY = 0.5
MARGIN = 0.1


class Reached(Exception):
    """The replay got to the decision we want to look at."""


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

def read_param(cls, name, where):
    if where == "global":
        return cls.play.__globals__[name]
    value = getattr(cls, name)
    return value if where == "class" else value[where]


def write_param(cls, name, where, value):
    if where == "global":
        cls.play.__globals__[name] = value
    elif where == "class":
        setattr(cls, name, value)
    else:
        old = getattr(cls, name)
        if isinstance(old, dict):
            new = dict(old)
            new[where] = value
        else:
            new = list(old)
            new[where] = value
            new = tuple(new)
        setattr(cls, name, new)   # a copy: the class default stays untouched


def label(name, where):
    return name if where in ("class", "global") else f"{name}[{where!r}]"


def candidates(base):
    """(size, param, new value, direction) for every one-parameter move, smallest first.
    A move never takes a parameter to 0 or flips its sign."""
    out = []
    for key, value in base.items():
        for sign in (1, -1):
            if value == 0:
                moves = [(s, sign * s) for s in ZERO_STEPS]
            else:
                moves = [(s, value * (1 + sign * s)) for s in STEPS if sign > 0 or s < 1]
            out.extend((size, key, new, sign) for size, new in moves)
    out.sort(key=lambda c: c[0])
    return out


# ---------------------------------------------------------------------------
# playing the game
# ---------------------------------------------------------------------------

def play(seed, bot_name, names, values, stop_at=None):
    """Replay the game. Returns (our decisions, dice per round, lost-die decision indices).

    Each decision is (round index, state, action). With stop_at the replay
    ends right after our decision with that index.
    """
    classes = discover_bot_classes_by_name()
    for cls in classes.values():
        if hasattr(cls, "reset_memory"):
            cls.reset_memory()
    me_cls = classes[bot_name]
    for (name, where), value in values.items():
        write_param(me_cls, name, where, value)
    bots = [classes[n]() for n in names]
    for i, bot in enumerate(bots):          # the bots' own randomness, so the game replays
        for attr in ("_rng", "rng"):
            if hasattr(bot, attr):
                setattr(bot, attr, random.Random(seed * 10 + i))
    me = bots[names.index(bot_name)]

    decisions, dice, lost = [], [], []
    play_me = me.play

    def recorded(state):
        action = play_me(state)
        decisions.append((len(dice), state, action))
        if stop_at is not None and len(decisions) > stop_at:
            raise Reached
        return action

    me.play = recorded

    def listener(event):
        if isinstance(event, RevealEvent):
            dice.append([d for _, hand in event.dice for d in hand])
        elif isinstance(event, DieLostEvent) and event.player == me.name:
            # The die is lost for our last call in the round that just ended.
            lost.append(len(decisions) - 1)

    try:
        PerudoGame(bots=bots, seed=seed, listeners=[listener]).play()
    except Reached:
        pass
    return decisions, dice, lost


def correct(state, action, dice):
    """Would this call have been right with the dice really on the table?"""
    if isinstance(action, Bid):
        return count_matches(dice, action.face, state.palifico) >= action.quantity
    bid = state.current_bid
    count = count_matches(dice, bid.face, state.palifico)
    if isinstance(action, Dudo):
        return count < bid.quantity
    if isinstance(action, Calza):
        return count == bid.quantity
    return False


def own_view(state, action):
    """Chance that `action` does not lose a die, from our own dice and chance alone
    (the others' bids are ignored: we cannot know whether they bluffed)."""
    total = sum(p.dice_count for p in state.players if p.alive)
    unknown = total - state.my_dice_count
    bid = action if isinstance(action, Bid) else state.current_bid
    p = 1 / 6 if (state.palifico or bid.face == 1) else 1 / 3
    need = bid.quantity - count_matches(state.my_dice, bid.face, state.palifico)
    pmf = [math.comb(unknown, k) * p ** k * (1 - p) ** (unknown - k) for k in range(unknown + 1)]
    true = sum(pmf[max(need, 0):])
    if isinstance(action, Bid):
        return true
    if isinstance(action, Dudo):
        return 1 - true
    return pmf[need] if 0 <= need <= unknown else 0.0     # Calza: exactly right


def knowable(state, action, likely, margin):
    """(was it a mistake the bot could know about?, its chance, best chance, best action)"""
    chance = own_view(state, action)
    best = max(state.legal_actions, key=lambda a: own_view(state, a))
    best_chance = own_view(state, best)
    return chance < likely and chance < best_chance - margin, chance, best_chance, best


def try_candidate(job):
    """("fixed" | "wrong" | "earlier", call at k): what this change does to decision k."""
    seed, bot_name, names, values, k, before, dice = job
    try:
        decisions, _, _ = play(seed, bot_name, names, values, stop_at=k)
    except Exception:                  # a value the bot cannot work with
        return "broken", None
    if len(decisions) <= k or [a for _, _, a in decisions[:k]] != before:
        return "earlier", None        # an earlier call changed (or the game ended sooner)
    _, state, action = decisions[k]
    return ("fixed" if correct(state, action, dice) else "wrong"), action


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------

def table(seed, bot_name, players, opponents):
    rng = random.Random(seed)
    if opponents is None:
        opponents = rng.sample([n for n in POOL if n != bot_name], players - 1)
    names = [bot_name, *opponents]
    rng.shuffle(names)
    return names


def fmt(action):
    if isinstance(action, Bid):
        return f"{action.quantity}x{'P' if action.face == 1 else action.face}"
    return type(action).__name__


def main():
    parser = argparse.ArgumentParser(description="Tune a bot's starting parameters on one game.")
    parser.add_argument("--bot", required=True, choices=sorted(PARAMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--players", type=int, default=5, help="Table size (with random opponents).")
    parser.add_argument("--opponents", nargs="*", default=None, help="Opponent class names.")
    parser.add_argument("--likely", type=float, default=LIKELY,
                        help="A losing call this likely to be right (from the bot's own dice) "
                             "or more was defensible and is not tuned.")
    parser.add_argument("--margin", type=float, default=MARGIN,
                        help="How much less likely to be right (from the bot's own dice) a "
                             "losing call must be than the best action to count as a mistake.")
    parser.add_argument("--max-fixes", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    names = table(args.seed, args.bot, args.players, args.opponents)
    cls = discover_bot_classes_by_name()[args.bot]
    values = {(n, w): read_param(cls, n, w) for n, w in PARAMS[args.bot]}
    start = dict(values)
    print(f"seed {args.seed}, table: {', '.join(names)}")

    fixed_upto = -1
    with Pool(args.jobs) as pool:
        for _ in range(args.max_fixes):
            decisions, dice, lost = play(args.seed, args.bot, names, values)
            todo = [k for k in lost if k > fixed_upto]
            if not todo:
                print("\nNo more lost dice: done.")
                break
            k = todo[0]
            rnd, state, action = decisions[k]
            print(f"\nLost a die in round {rnd + 1} on {fmt(action)} "
                  f"(facing {fmt(state.current_bid) if state.current_bid else 'nothing'}, "
                  f"call {k + 1} of the bot)")
            mistake, chance, best_chance, best = knowable(state, action, args.likely, args.margin)
            if not mistake:
                print(f"  The bot could not know: from its own dice this call was "
                      f"{chance:.0%} likely to be right, the best action ({fmt(best)}) "
                      f"{best_chance:.0%}. Skipped.")
                fixed_upto = k
                continue
            print(f"  A mistake it could know: {chance:.0%} likely to be right from its own "
                  f"dice, {fmt(best)} was {best_chance:.0%}.")
            before = [a for _, _, a in decisions[:k]]
            jobs, moves = [], []
            for move in candidates(values):
                size, key, new, _ = move
                trial = dict(values)
                trial[key] = new
                jobs.append((args.seed, args.bot, names, trial, k, before, dice[rnd]))
                moves.append(move)
            results = pool.map(try_candidate, jobs)
            # Per parameter and direction, growing steps: a fix counts only if
            # it comes before earlier calls start to change in that direction.
            fixes, touched = [], False
            groups = {}
            for move, result in zip(moves, results):
                groups.setdefault((move[1], move[3]), []).append((move, result))
            for steps in groups.values():
                for (size, key, new, _), (status, new_action) in sorted(steps, key=lambda x: x[0][0]):
                    if status == "earlier":
                        touched = True
                        break
                    if status == "broken":
                        break
                    if status == "fixed":
                        fixes.append((size, key, new, new_action))
                        break
            fixes.sort(key=lambda f: f[0])
            fix = ((fixes[0][0], fixes[0][1], fixes[0][2]), fixes[0][3]) if fixes else None
            if fix is None and touched:
                print("  No parameter fixes this call before earlier calls start to change: stop.")
                break
            if fix is None:
                print("  No change makes this call correct (the bot cannot see it): skipped.")
                fixed_upto = k
                continue
            (size, key, new), new_action = fix
            print(f"  {label(*key)}: {values[key]:.4g} -> {new:.4g}  "
                  f"(call becomes {fmt(new_action)})")
            values[key] = new
            fixed_upto = k

    changed = {key: v for key, v in values.items() if v != start[key]}
    print("\nTuned starting values:" if changed else "\nNo values changed.")
    for key, v in changed.items():
        print(f"  {label(*key):<28} {start[key]:.4g} -> {v:.4g}")


if __name__ == "__main__":
    main()
