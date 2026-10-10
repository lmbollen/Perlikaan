"""
Why does Pokerface lose head-to-head and endgame rounds?

Plays Pokerface against the core bots (Sonny Boy, Oppa Style, Fabulous,
Trihard) with Pokerface's decision log on, so every decision comes with what
Pokerface believed at that moment. After each reveal the truth is known, so
for the rounds with few players left we can check:

  * calibration: when Pokerface said "P(this bid is true) = x", how often was
    it really true? Both for bids we faced (Dudo or raise) and for our own
    raises. Next to it the naive estimate from our own dice only, to see
    whether reading the opponent's bids helps or hurts here;
  * how we lose: wrong Dudo vs our bid caught, and how sure we were;
  * how the opponents do in the very same situations (their Dudo accuracy);
  * the worst calls, in full.

Usage:
    uv run python analyze_endgame.py --tables 40 --games 40 --players 2 --show 10
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from multiprocessing import Pool

from benchmark_adaptive import RoundLimit, TooManyRounds, build, table_spec
from perudo.bot_loader import discover_bot_classes_by_name
from perudo.game import PerudoGame
from perudo.models import PACO, Bid, Calza, Dudo, DieLostEvent, RevealEvent, RoundEndedEvent
from perudo.profiler import p_bid_true
from perudo.rules import count_matches
from run_simulation import derive_game_seeds

ME = "Pokerface"


class Rounds:
    """Per round of one game: the reveal and who lost a die."""

    def __init__(self):
        self.rounds = {}
        self.n = 1
        self.loser = None
        self.reveal = None

    def __call__(self, event):
        if isinstance(event, RevealEvent):
            self.reveal = event
        elif isinstance(event, DieLostEvent):
            self.loser = event.player
        elif isinstance(event, RoundEndedEvent):
            self._close()
            self.n += 1

    def _close(self):
        if self.reveal is not None:
            self.rounds[self.n] = (self.reveal, self.loser)
        self.reveal, self.loser = None, None

    def finish(self):
        self._close()


def fmt(action):
    if isinstance(action, Bid):
        return f"{action.quantity}x{'P' if action.face == PACO else action.face}"
    return "DUDO" if isinstance(action, Dudo) else "CALZA" if isinstance(action, Calza) else str(action)


def holds(bid, dice, palifico):
    return count_matches([d for hand in dice.values() for d in hand], bid.face, palifico) >= bid.quantity


def run(job):
    seed, table, n_opp, games = job
    classes = discover_bot_classes_by_name()
    me = classes["PokerfaceBot"]()
    me._rng = random.Random(table)
    bots = [me, *build(table_spec(seed, table, n_opp, "core"), classes)]
    rows = []
    for g, game_seed in enumerate(derive_game_seeds(seed * 7 + table, games)):
        me.decision_log = []
        rounds = Rounds()
        try:
            PerudoGame(bots=bots, seed=game_seed, listeners=[rounds, RoundLimit()]).play()
        except TooManyRounds:
            continue
        rounds.finish()
        by_round = defaultdict(list)
        for d in me.decision_log:
            by_round[d["round"]].append(d)
        for n, (reveal, loser) in rounds.rounds.items():
            dice = reveal.dice_by_player()
            if ME not in dice:
                continue
            decisions = by_round.get(n, [])
            pal = decisions[0]["palifico"] if decisions else None
            rows.append(dict(
                table=table, game=g, round=n, players=len(dice), total=sum(len(v) for v in dice.values()),
                my_count=len(dice[ME]), dice={k: tuple(v) for k, v in dice.items()},
                bid=reveal.current_bid, challenger=reveal.challenger, bidder=reveal.bidder,
                kind=reveal.challenge_type, result=reveal.result, loser=loser,
                palifico=pal, decisions=decisions,
            ))
    return rows


def classify(r):
    """Why a lost round was lost, judged by what we believed when deciding."""
    if not r["decisions"]:
        return "geen beslissing van ons (?)"
    last = r["decisions"][-1]
    if r["challenger"] == ME:
        if r["kind"] == "calza":
            return "foute Calza"
        if last["p_best_raise"] is not None and last["p_best_raise"] >= 0.5:
            return "foute Dudo, terwijl een veilige verhoging bestond"
        return "foute Dudo, klem (geen verhoging >= 50%)"
    # Our bid was challenged and false.
    if last["current"] is None:
        return "ons openingsbod gepakt"
    p_dudo_wins = 1 - last["p_current"]
    if p_dudo_wins > last["p_best_raise"]:
        return "bod gepakt, Dudo was volgens ons kansrijker"
    if last["p_best_raise"] >= 0.5:
        return "bod gepakt, bod leek veilig (>= 50%)"
    return "bod gepakt, klem (alles < 50%)"


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   - "


def calibration(title, items):
    """items: (predicted p, naive p, was true)"""
    print(f"\n{title}")
    print(f"  {'voorspeld':<11}{'n':>6}{'echt waar':>11}{'naief geschat':>15}")
    buckets = defaultdict(list)
    for p, naive, true in items:
        buckets[min(int(p * 10), 9)].append((naive, true))
    for b in sorted(buckets):
        xs = buckets[b]
        print(f"  {b / 10:.1f}-{b / 10 + 0.1:.1f}  {len(xs):6d}{pct(sum(t for _, t in xs), len(xs)):>11}"
              f"{sum(n for n, _ in xs) / len(xs):>15.2f}")


def main():
    parser = argparse.ArgumentParser(description="Analyse Pokerface's endgame and head-to-head losses.")
    parser.add_argument("--tables", type=int, default=40)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--start-players", type=int, default=5, help="Players at the start of each game.")
    parser.add_argument("--players", type=int, default=2, help="Analyse rounds with at most this many players left.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--show", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    jobs = [(args.seed, t, args.start_players - 1, args.games) for t in range(args.tables)]
    with Pool(args.jobs) as pool:
        rows = [r for part in pool.map(run, jobs) for r in part]

    end = [r for r in rows if r["players"] <= args.players]
    print(f"{args.tables} tafels x {args.games} games, start met {args.start_players} spelers "
          f"(Pokerface + core bots). Rondes met <= {args.players} spelers over: {len(end)}\n")

    # Loss rate per number of players left, as context.
    print("Verlies per aantal spelers over:")
    for n in sorted({r["players"] for r in rows}):
        rs = [r for r in rows if r["players"] == n]
        lost = sum(r["loser"] == ME for r in rs)
        print(f"  {n} spelers: {len(rs):6d} rondes, wij verliezen {pct(lost, len(rs))}  (eerlijk {100 / n:.1f}%)")

    # How we lose in the endgame.
    lost = [r for r in end if r["loser"] == ME]
    how = Counter()
    for r in lost:
        r["why"] = classify(r)
        how[r["why"]] += 1
    print(f"\nEindspel: {len(lost)} van {len(end)} rondes verloren ({pct(len(lost), len(end)).strip()})")
    print("  (volgens wat Pokerface op dat moment geloofde)")
    for k, v in how.most_common():
        print(f"  {k:<48}{v:6d}  ({pct(v, len(lost)).strip()})")
    pal = [r for r in end if r["palifico"]]
    print(f"  waarvan Palifico-rondes: {len(pal)} gespeeld, {sum(r['loser'] == ME for r in pal)} verloren")

    # Dudo accuracy, ours vs the opponents', in the same endgame rounds.
    print("\nDudo in het eindspel: hoe vaak klopt de Dudo (bod was vals)?")
    calls = defaultdict(lambda: [0, 0])
    for r in end:
        if r["kind"] == "dudo":
            who = ME if r["challenger"] == ME else r["challenger"].rstrip(" 0123456789")
            calls[who][0] += r["result"] == "bidder_wrong"
            calls[who][1] += 1
    for who, (ok, n) in sorted(calls.items(), key=lambda kv: -kv[1][1]):
        print(f"  {who:<12}{n:6d} Dudo's, {pct(ok, n)} terecht")
    caught = defaultdict(lambda: [0, 0])
    for r in end:
        if r["kind"] == "dudo":
            who = ME if r["bidder"] == ME else r["bidder"].rstrip(" 0123456789")
            caught[who][0] += r["result"] == "bidder_wrong"
            caught[who][1] += 1
    print("Aangevochten biedingen in het eindspel: hoe vaak was het bod vals?")
    for who, (bad, n) in sorted(caught.items(), key=lambda kv: -kv[1][1]):
        print(f"  {who:<12}{n:6d} keer aangevochten, {pct(bad, n)} vals")

    # Calibration of what we believed.
    facing, raising = [], []
    for r in end:
        for d in r["decisions"]:
            total = sum(d["players"].values())
            if d["current"] is not None and d["p_current"] is not None:
                naive = p_bid_true(d["current"], d["my_dice"], total, d["palifico"])
                facing.append((d["p_current"], naive, holds(d["current"], r["dice"], d["palifico"]),
                               isinstance(d["action"], Dudo)))
            if isinstance(d["action"], Bid):
                naive = p_bid_true(d["action"], d["my_dice"], total, d["palifico"])
                raising.append((d["p_best_raise"], naive, holds(d["action"], r["dice"], d["palifico"])))
    calibration("Bod waar we voor stonden: onze P(waar) vs werkelijkheid",
                [(p, n, t) for p, n, t, _ in facing])
    calibration("  ... alleen waar we Dudo riepen", [(p, n, t) for p, n, t, dudo in facing if dudo])
    calibration("  ... alleen waar we verhoogden", [(p, n, t) for p, n, t, dudo in facing if not dudo])
    calibration("Ons eigen bod: onze P(waar) vs werkelijkheid", raising)

    if args.show:
        def badness(r):
            d = r["decisions"][-1] if r["decisions"] else None
            if d is None:
                return 0
            if isinstance(d["action"], Dudo):
                return -(d["p_current"] or 0)
            return -(1 - (d["p_best_raise"] or 0))

        print(f"\nVOORBEELDEN: {args.show} verloren eindspelrondes, meest zelfverzekerd fout eerst")
        for r in sorted(lost, key=badness)[:args.show]:
            print(f"\n  tafel {r['table']} game {r['game']} ronde {r['round']}: {r['players']} spelers, "
                  f"{r['total']} stenen{', PALIFICO' if r['palifico'] else ''}")
            print("  stenen: " + "  ".join(f"{k}={''.join(map(str, sorted(v)))}" for k, v in r["dice"].items()))
            for d in r["decisions"]:
                seq = " ".join(f"{p}:{fmt(a)}" for p, a in d["history"]) or "(wij openen)"
                cur = f"P(huidig bod waar)={d['p_current']:.2f}" if d["p_current"] is not None else ""
                print(f"    na [{seq}]  {cur}  beste verhoging {fmt(d['best_raise'])} "
                      f"P={d['p_best_raise']:.2f} -> wij: {fmt(d['action'])}"
                      f"{' (bluf)' if d['bluffed'] else ''}")
            print(f"    uitkomst: {r['kind']} door {r['challenger']} op {fmt(r['bid'])} van {r['bidder']}, "
                  f"werkelijk {sum(count_matches(v, r['bid'].face, bool(r['palifico'])) for v in r['dice'].values())}")


if __name__ == "__main__":
    main()
