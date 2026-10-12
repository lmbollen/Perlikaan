"""
Fit Carlos Dos's generic starting model against a generic population.

Carlos Dos starts every opponent he does not know from one generic model:
how a player reacts to a bid (the Dudo and Calza logistics) and how a
player's bid relates to their hand (the likelihood per residual implied -
held, per kind of bid). That starting model must not be fitted to the bots
in this repository: in an unknown field it has to describe "a player", and
the per-player learning at every reveal does the rest.

So this script defines a generic population of its own: rational players
whose style is drawn at random per seat and per game:

  * calling: Dudo with probability sigmoid(sharpness * (threshold - P(true)))
    where the threshold is the break-even point 1 / players, scaled by a
    random factor (timid to trigger-happy) and P(true) is what they believe;
  * reading: whether (and how much) they read the other players' bids, or
    only count their own dice;
  * bluffing: how often they raise on a face they do not hold;
  * jumping: how often they raise further than the minimum;
  * Calza: the P(exact) above which they call it.

Carlos Dos plays many games against fresh random generic players with all
learning pooled, and the pooled models he ends up with are printed as the
generic starting model (constants to paste into perudo/bots/carlos_dos.py).
The bots of the repository are only used afterwards, as a held-out test.

Usage:
    uv run python fit_generic_prior.py --games 600
"""

from __future__ import annotations

import argparse
import math
import random

from perudo.bots.carlos_dos import (
    KINDS, R_MAX, CarlosDosBot, _at, _binomial, _conv, _count_matches, _face_p, _min_quantity, _tail,
)
from perudo.game import PerudoGame
from perudo.models import PACO, Bid, Calza, Dudo

NORMAL = (2, 3, 4, 5, 6)


class GenericPlayer:
    """A rational Perudo player with a randomly drawn style."""

    def __init__(self, name, rng):
        self.name = name
        self.rng = rng
        self.threshold = rng.uniform(0.6, 1.5)      # x break-even (1 / players)
        self.sharpness = rng.uniform(6.0, 20.0)
        self.reading = rng.choice((0.0, 0.5, 1.0))   # weight of the other players' bids
        self.bluff = rng.uniform(0.0, 0.35)
        self.jump = rng.uniform(0.0, 0.3)
        self.calza_at = rng.uniform(0.4, 0.75)

    def on_game_start(self, info):
        pass

    def observe(self, event):
        pass

    def on_game_end(self, result):
        pass

    # -- beliefs ------------------------------------------------------------

    def _count_pmf(self, state, face):
        """Distribution of the matching dice of everyone else, from chance and
        (as much as this player reads) the latest bid of each player on the face."""
        pal = state.palifico
        p = _face_p(face, pal)
        latest = {}
        for rec in state.history:
            if isinstance(rec.action, Bid) and rec.action.face == face:
                latest[rec.player] = rec.action
        total = sum(pl.dice_count for pl in state.players if pl.alive)
        rest = [1.0]
        for pl in state.players:
            if not pl.alive or pl.id == state.my_id:
                continue
            n = pl.dice_count
            pmf = list(_binomial(n, p))
            bid = latest.get(pl.name)
            if bid is not None and self.reading > 0:
                implied = round(bid.quantity - (total - n) * p)
                if 1 <= implied <= n:
                    real = [v if k >= implied else 0.0 for k, v in enumerate(pmf)]
                    z = sum(real)
                    if z > 0:
                        w = 0.6 * self.reading
                        pmf = [w * r / z + (1 - w) * v for r, v in zip(real, pmf)]
            rest = _conv(rest, pmf)
        return rest

    def _p_true(self, state, bid):
        k = _count_matches(state.my_dice, bid.face, state.palifico)
        rest = self._count_pmf(state, bid.face)
        return _tail(rest, bid.quantity - k), _at(rest, bid.quantity - k)

    # -- deciding -------------------------------------------------------------

    def play(self, state):
        legal = set(state.legal_actions)
        alive = sum(1 for pl in state.players if pl.alive)
        bid = state.current_bid
        if bid is not None:
            p_true, p_exact = self._p_true(state, bid)
            if Calza() in legal and p_exact > self.calza_at:
                return Calza()
            t = self.threshold / alive
            if self.rng.random() < 1 / (1 + math.exp(-self.sharpness * (t - p_true))):
                return Dudo()
        pal = state.palifico
        faces = (bid.face,) if (pal and bid is not None) else (tuple(range(1, 7)) if pal else NORMAL + (PACO,))
        if bid is None and not pal:
            faces = NORMAL
        cands = []
        for f in faces:
            if bid is None:
                total = sum(pl.dice_count for pl in state.players if pl.alive)
                k = _count_matches(state.my_dice, f, pal)
                q = max(1, round(k + (total - state.my_dice_count) * _face_p(f, pal)) - 1)
            else:
                q = _min_quantity(bid, f, pal)
            if self.rng.random() < self.jump:
                q += 1
            b = Bid(q, f)
            if b in legal:
                cands.append(b)
        if not cands:
            return Dudo() if Dudo() in legal else state.legal_actions[0]
        if self.rng.random() < self.bluff:
            return self.rng.choice(cands)
        return max(cands, key=lambda b: self._p_true(state, b)[0])


def fit(games, sizes, seed):
    """Carlos Dos learns, pooled over all opponents, against fresh generic players."""
    CarlosDosBot.reset_memory()
    keep = ("GAME_DECAY", "GENERIC_LAMBDA", "DUDO_PRIOR", "CALZA_PRIOR", "LEARN_RATE")
    old = {k: getattr(CarlosDosBot, k) for k in keep}
    CarlosDosBot.GAME_DECAY = 1.0           # every game counts the same
    CarlosDosBot.GENERIC_LAMBDA = None      # start the source model from the REAL_LR formula
    CarlosDosBot.LEARN_RATE = 0.05          # average over more reactions
    # start the reaction model from probability alone: the break-even point,
    # split between reading own dice and reading the bids, nothing else
    CarlosDosBot.DUDO_PRIOR = (1.0, -5.0, 0.0, -5.0, 0.0, 0.0, 0.0, 4.2)
    CarlosDosBot.CALZA_PRIOR = (-6.0, 12.0)
    rng = random.Random(seed)
    try:
        for g in range(games):
            size = rng.choice(sizes)
            bots = [GenericPlayer(f"G{g}-{i}", random.Random(rng.randrange(2**63))) for i in range(size - 1)]
            bots.append(CarlosDosBot())
            rng.shuffle(bots)
            PerudoGame(bots=bots, seed=rng.randrange(2**63)).play()
        me = CarlosDosBot()
        me._ensure_memory()
        return me.table
    finally:
        for k, v in old.items():
            setattr(CarlosDosBot, k, v)


def lambdas(table, prior_w=10.0, min_bids=200):
    """Likelihood per residual: observed over chance, smoothed towards 1 with
    prior_w bids of chance mass per cell, so a cell that a few bids happened
    to land in does not become a factor of 5. Kinds with fewer than min_bids
    are left out (the bot falls back to its formula)."""
    src = table["source"]
    out = {}
    for kind in KINDS:
        obs, chance = src.obs[kind], src.chance[kind]
        if sum(obs) < min_bids:
            continue
        lam = [(o + prior_w) / (c + prior_w) for o, c in zip(obs, chance)]
        out[kind] = tuple(round(v, 3) for v in lam)
    return out


def main():
    parser = argparse.ArgumentParser(description="Fit Carlos Dos's generic starting model.")
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    multi = fit(args.games, (3, 4, 5, 6, 7), args.seed)
    heads_up = fit(args.games, (2,), args.seed + 1)
    print("# Fitted by fit_generic_prior.py against its generic population "
          f"({args.games} games at 3-7 players, {args.games} heads-up, seed {args.seed}).")
    dudo = [round(w, 2) for w in multi["dudo_w"]]
    dudo[6] = 0.0        # our caught-bluff record: the generic players do not keep one,
    #                      so whatever the fit finds for it is a spurious correlation
    print(f"DUDO_PRIOR = {tuple(dudo)}")
    print(f"CALZA_PRIOR = {tuple(round(w, 2) for w in multi['calza_w'])}")
    print("GENERIC_LAMBDA = {")
    for label, table in (("2", heads_up), ("3+", multi)):
        print(f'    "{label}": {{')
        for kind, lam in lambdas(table).items():
            print(f'        "{kind}": {lam},')
        print("    },")
    print("}")
    print(f"# heads-up Dudo weights for comparison: {tuple(round(w, 2) for w in heads_up['dudo_w'])}")


if __name__ == "__main__":
    main()
