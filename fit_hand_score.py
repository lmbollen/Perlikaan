"""
Fit perudo/hand_score.py: how well does a hand at the start of a round
predict whether its player loses a die that round?

Plays games at random tables of the bundled bots (2 to 8 seats; bots keep
learning over a table's games, as in the app) and records, for every player
at the start of every round, the features of perudo.hand_score and whether
they lost a die. Fits two logistic models, one on the hand and the table,
one on the table only (the average hand), and judges both on games they did
not see (log loss, AUC, calibration). Prints the weights for hand_score.py.

    uv run python fit_hand_score.py --tables 160 --games 6
"""

from __future__ import annotations

import argparse
import random
from multiprocessing import Pool

import numpy as np

from benchmark_adaptive import fresh_seed
from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame
from perudo.hand_score import FEATURES, features
from perudo.models import DieLostEvent, RevealEvent

TABLE_ONLY = ("bias", "players", "share", "opens", "turn", "palifico", "one_die")
MAX_ROUNDS = 400


class _Recorder:
    """Rows of (features, lost a die) for every player of every round."""

    def __init__(self):
        self.rows, self.open = [], {}

    def start(self, game, starter_idx, palifico):
        order, i = [], starter_idx
        for _ in range(len(game.players)):        # bidding order from the starter
            if game.players[i].alive:
                order.append(game.players[i])
            i = (i + 1) % len(game.players)
        total = sum(p.dice_count for p in order)
        self.open = {p.name: features(tuple(p.dice), len(order), total, turn, palifico)
                     for turn, p in enumerate(order)}
        self.lost = set()

    def __call__(self, event):
        if isinstance(event, DieLostEvent):
            self.lost.add(event.player)

    def end(self):
        for name, x in self.open.items():
            self.rows.append(([x[f] for f in FEATURES], name in self.lost))
        self.open = {}


def run_table(job):
    seed, table, games = job
    rng = random.Random(seed * 1_000_003 + table)
    classes = discover_bot_classes_by_name()
    for cls in classes.values():                  # every table starts without memory
        if hasattr(cls, "reset_memory"):
            cls.reset_memory()
    names = sorted(classes)
    seats = rng.randint(2, 8)
    bots, seen = [], set()
    for name in rng.choices(names, k=seats):
        bot = classes[name]()
        base, copy = bot.name, 2
        while bot.name in seen:
            bot.name = f"{base} {copy}"
            copy += 1
        seen.add(bot.name)
        bots.append(bot)

    rec = _Recorder()
    real = PerudoGame._play_round

    def play_round(game, starter_idx, palifico):
        rec.start(game, starter_idx, palifico)
        try:
            return real(game, starter_idx, palifico)
        finally:
            rec.end()

    PerudoGame._play_round = play_round
    out = []
    try:
        for g in range(games):
            rounds = {"n": 0}

            def limit(event, rounds=rounds):
                rec(event)
                if isinstance(event, RevealEvent):
                    rounds["n"] += 1
                    if rounds["n"] > MAX_ROUNDS:
                        raise RuntimeError("round limit")

            rec.rows = []
            try:
                PerudoGame(bots=bots, seed=rng.randrange(2**63), listeners=[limit]).play()
            except RuntimeError:
                continue
            out.append(rec.rows)                   # one game's rows
    finally:
        PerudoGame._play_round = real
    return table, out


def fit(X, y, l2=1e-3, steps=60):
    """Logistic regression by Newton's method."""
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (p - y) / len(y) + l2 * w
        H = (X.T * (p * (1 - p))) @ X / len(y) + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def judge(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
    return logloss, auc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=int, default=160)
    ap.add_argument("--games", type=int, default=6, help="games per table")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    seed = fresh_seed(args.seed)

    with Pool(args.workers) as pool:
        results = pool.map(run_table, [(seed, t, args.games) for t in range(args.tables)], chunksize=1)
    games = [g for _, gs in sorted(results) for g in gs]
    rng = random.Random(seed)
    rng.shuffle(games)
    cut = len(games) // 5                          # a fifth of the games to judge on
    test = [r for g in games[:cut] for r in g]
    train = [r for g in games[cut:] for r in g]
    Xtr = np.array([r[0] for r in train]); ytr = np.array([r[1] for r in train], float)
    Xte = np.array([r[0] for r in test]); yte = np.array([r[1] for r in test], float)
    print(f"{len(games)} games: {len(train)} hands to fit on, {len(test)} to judge on; "
          f"{ytr.mean():.1%} of hands lose a die")

    cols = [FEATURES.index(f) for f in TABLE_ONLY]
    w_hand = fit(Xtr, ytr)
    w_base = np.zeros(len(FEATURES))
    w_base[cols] = fit(Xtr[:, cols], ytr)

    for name, w in (("table only", w_base), ("hand + table", w_hand)):
        ll, auc = judge(1 / (1 + np.exp(-Xte @ w)), yte)
        print(f"  {name:13} log loss {ll:.4f}   AUC {auc:.3f}")
    ll0, _ = judge(np.full(len(yte), ytr.mean()), yte)
    print(f"  {'nothing':13} log loss {ll0:.4f}   AUC 0.500")

    # Calibration of the hand model on the held-out games, by tenths.
    p = 1 / (1 + np.exp(-Xte @ w_hand))
    print("\n  predicted keep   actual keep   hands")
    edges = np.quantile(p, np.linspace(0, 1, 11))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        print(f"     {1 - p[m].mean():6.1%}        {1 - yte[m].mean():6.1%}     {m.sum():6}")

    # How much the hand moves the chance, against the average hand in its spot.
    from perudo import hand_score
    hand_score.WEIGHTS.update({f: float(v) for f, v in zip(FEATURES, w_hand)})
    hand_score.BASE_WEIGHTS.update({f: float(v) for f, v in zip(FEATURES, w_base)})
    scores = []
    for x in Xte:
        x = dict(zip(FEATURES, x))
        keep = 1 - hand_score._p_lose(hand_score.WEIGHTS, x)
        avg = 1 - hand_score._p_lose(hand_score.BASE_WEIGHTS, x)
        scores.append(int(round(max(0, min(100, 50 + 25 * (hand_score._logit(keep) - hand_score._logit(avg)))))))
    scores = np.array(scores)
    print("\n  score     hands   actual keep")
    for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 101)):
        m = (scores >= lo) & (scores < hi)
        if m.any():
            print(f"  {lo:3}-{min(hi, 100):<3}  {m.sum():7}   {1 - yte[m].mean():6.1%}")

    def fmt(w):
        return "{\n" + "".join(f'    "{f}": {v:+.4f},\n' for f, v in zip(FEATURES, w)) + "}"
    print("\nWEIGHTS: Dict[str, float] =", fmt(w_hand))
    print("\nBASE_WEIGHTS: Dict[str, float] =", fmt(w_base))


if __name__ == "__main__":
    main()
