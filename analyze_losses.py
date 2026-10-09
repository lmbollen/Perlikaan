"""
Find out when a bot loses rounds.

Two steps, usable separately or together:

  dump     Run a simulation and write every round to a JSONL file: who was
           alive, everyone's dice, every bid/Dudo/Calza in order, the reveal
           and who lost or gained a die.
  analyze  Read such a file, keep only the rounds in which BOT lost a die,
           and break those down: how it lost (bid caught, wrong Dudo, wrong
           Calza), number of players left, dice per player, the calls of the
           others and our own call. Calls that look bad from what we could
           see (our own dice only) are flagged.

Usage:
    uv run python analyze_losses.py dump --games 200 --seed 1 --out rounds.jsonl
    uv run python analyze_losses.py analyze rounds.jsonl --bot Pokerface --show 15
    uv run python analyze_losses.py run --games 200 --bot Pokerface   # both

`--csv losses.csv` writes one row per lost round for further digging.

P(true) is computed from the bot's own dice only, without reading the bids
of the others, so it is a baseline: a Dudo on a bid that was already likely
from our own dice alone, or a bid of ours that was unlikely even counting our
own dice, is a call worth looking at.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict

from perudo.bot_loader import load_bots
from perudo.game import PerudoGame
from perudo.models import (
    PACO,
    BidEvent,
    CalzaEvent,
    DieGainedEvent,
    DieLostEvent,
    DudoEvent,
    PlayerEliminatedEvent,
    RevealEvent,
    RoundEndedEvent,
)
from perudo.rules import count_matches
from run_simulation import derive_game_seeds

# A Dudo on a bid at least this likely (from our dice) is flagged as a bad
# call; so is a bid of ours that was less likely than 1 - BAD_CALL.
BAD_CALL = 0.5


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------


class RoundRecorder:
    """Listener that turns the public event stream into one dict per round."""

    def __init__(self, out, game_number, seed):
        self.out = out
        self.game_number = game_number
        self.seed = seed
        self.round_number = 0
        self.palifico = False
        self._new_round()

    def _new_round(self):
        self.round_number += 1
        self.actions = []
        self.reveal = None
        self.lost = []
        self.gained = []
        self.eliminated = []

    def __call__(self, event):
        if isinstance(event, BidEvent):
            self.actions.append({"player": event.player, "type": "bid",
                                 "quantity": event.bid.quantity, "face": event.bid.face})
        elif isinstance(event, DudoEvent):
            self.actions.append({"player": event.player, "type": "dudo"})
        elif isinstance(event, CalzaEvent):
            self.actions.append({"player": event.player, "type": "calza"})
        elif isinstance(event, RevealEvent):
            self.reveal = event
        elif isinstance(event, DieLostEvent):
            self.lost.append(event.player)
        elif isinstance(event, DieGainedEvent):
            self.gained.append(event.player)
        elif isinstance(event, PlayerEliminatedEvent):
            self.eliminated.append(event.player)
        elif isinstance(event, RoundEndedEvent):
            self._flush()
            self.palifico = event.palifico_next
            self._new_round()

    def finish(self):
        # The last round of a game ends with GameEndedEvent, not RoundEndedEvent.
        if self.reveal is not None:
            self._flush()

    def _flush(self):
        r = self.reveal
        record = {
            "game": self.game_number,
            "seed": self.seed,
            "round": self.round_number,
            "palifico": self.palifico,
            # Seat order, alive players only, dice as rolled this round.
            "dice": {name: list(d) for name, d in r.dice},
            "actions": self.actions,
            "challenge": r.challenge_type,
            "challenger": r.challenger,
            "bidder": r.bidder,
            "bid": {"quantity": r.current_bid.quantity, "face": r.current_bid.face},
            "actual": r.actual_count,
            "result": r.result,
            "lost": self.lost,
            "gained": self.gained,
            "eliminated": self.eliminated,
        }
        self.out.write(json.dumps(record) + "\n")


def dump(games, seed, out_path):
    bots = load_bots()
    print(f"Simuleren van {games} games met: {', '.join(b.name for b in bots)} (seed={seed})")
    with open(out_path, "w", encoding="utf-8") as out:
        for game_number, game_seed in enumerate(derive_game_seeds(seed, games), start=1):
            recorder = RoundRecorder(out, game_number, game_seed)
            PerudoGame(bots=bots, seed=game_seed, listeners=[recorder]).play()
            recorder.finish()
            if game_number % 50 == 0:
                print(f"  {game_number}/{games}")
    print(f"Rondes geschreven naar {out_path}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def p_true(bid_q, bid_face, my_dice, total_dice, palifico, exact=False):
    """P(bid holds | our own dice), every unknown die independent."""
    mine = count_matches(my_dice, bid_face, palifico)
    n = total_dice - len(my_dice)
    p = 1 / 6 if (palifico or bid_face == PACO) else 1 / 3
    need = bid_q - mine

    def pmf(k):
        return math.comb(n, k) * p**k * (1 - p) ** (n - k) if 0 <= k <= n else 0.0

    if exact:
        return pmf(need)
    if need <= 0:
        return 1.0
    return sum(pmf(k) for k in range(need, n + 1))


def fmt_bid(q, face):
    return f"{q}x{'P' if face == PACO else face}"


def fmt_action(a):
    if a["type"] == "bid":
        return f"{a['player']}: {fmt_bid(a['quantity'], a['face'])}"
    return f"{a['player']}: {a['type'].upper()}"


def bucket(p):
    lo = min(int(p * 10), 9) / 10
    return f"{lo:.1f}-{lo + 0.1:.1f}"


def describe_loss(rec, bot):
    """Everything worth knowing about a round `bot` lost, as a flat dict."""
    dice = rec["dice"]
    my_dice = dice[bot]
    total = sum(len(d) for d in dice.values())
    others = {n: len(d) for n, d in dice.items() if n != bot}
    bid = rec["bid"]
    pal = rec["palifico"]

    if rec["challenger"] == bot and rec["challenge"] == "dudo":
        kind = "dudo_wrong"
        p = p_true(bid["quantity"], bid["face"], my_dice, total, pal)
        bad = p >= BAD_CALL
        our_call = f"DUDO op {fmt_bid(bid['quantity'], bid['face'])} van {rec['bidder']}"
    elif rec["challenger"] == bot:
        kind = "calza_wrong"
        p = p_true(bid["quantity"], bid["face"], my_dice, total, pal, exact=True)
        bad = p < 0.25
        our_call = f"CALZA op {fmt_bid(bid['quantity'], bid['face'])} van {rec['bidder']}"
    else:
        kind = "bid_caught"
        p = p_true(bid["quantity"], bid["face"], my_dice, total, pal)
        bad = p < 1 - BAD_CALL
        our_call = f"bod {fmt_bid(bid['quantity'], bid['face'])}, DUDO door {rec['challenger']}"

    # A bluff: we bid a normal face we held none of (Pacos excluded). For a
    # caught bid we look at that bid only, otherwise at all our bids.
    our_bids = [a for a in rec["actions"] if a["player"] == bot and a["type"] == "bid"]
    if kind == "bid_caught":
        our_bids = [{"face": bid["face"]}]
    bluffs = [a for a in our_bids if a["face"] != PACO and a["face"] not in my_dice]
    opened = bool(rec["actions"]) and rec["actions"][0]["player"] == bot

    return {
        "game": rec["game"],
        "round": rec["round"],
        "kind": kind,
        "players": len(dice),
        "my_dice_count": len(my_dice),
        "opp_dice": " ".join(f"{n}={c}" for n, c in others.items()),
        "opp_avg_dice": sum(others.values()) / len(others),
        "total_dice": total,
        "palifico": pal,
        "opened": opened,
        "n_bids": sum(1 for a in rec["actions"] if a["type"] == "bid"),
        "my_dice": "".join(map(str, sorted(my_dice))),
        "our_call": our_call,
        "p_true": p,
        "actual": rec["actual"],
        "margin": rec["actual"] - bid["quantity"],
        "bluffed": bool(bluffs),
        "bad_call": bad,
        "sequence": " | ".join(fmt_action(a) for a in rec["actions"]),
        "all_dice": " ".join(f"{n}:{''.join(map(str, sorted(d)))}" for n, d in dice.items()),
    }


def print_table(title, rows, headers):
    print(f"\n{title}")
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(f"{h:>{w}}" for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(f"{str(c):>{w}}" for c, w in zip(r, widths)))


def rate_table(title, played, lost, label):
    rows = []
    for key in sorted(played):
        n, l = played[key], lost.get(key, 0)
        rows.append((key, n, l, f"{100 * l / n:.1f}%"))
    print_table(title, rows, (label, "rondes", "verloren", "verlies %"))


def analyze(path, bot, show, csv_path):
    played_by_players = Counter()
    played_by_dice = Counter()
    played_pal = Counter()
    losses = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if bot not in rec["dice"]:
                continue  # already eliminated
            played_by_players[len(rec["dice"])] += 1
            played_by_dice[len(rec["dice"][bot])] += 1
            played_pal[rec["palifico"]] += 1
            if bot in rec["lost"]:
                losses.append(describe_loss(rec, bot))

    total_played = sum(played_by_players.values())
    if not total_played:
        raise SystemExit(f"Geen rondes gevonden waarin '{bot}' meespeelde.")

    print(f"{'=' * 72}")
    print(f"VERLOREN RONDES van {bot}: {len(losses)} van {total_played} "
          f"({100 * len(losses) / total_played:.1f}%)")
    print(f"{'=' * 72}")

    kinds = Counter(l["kind"] for l in losses)
    bad = Counter(l["kind"] for l in losses if l["bad_call"])
    bluff = Counter(l["kind"] for l in losses if l["bluffed"])
    print_table(
        "Hoe verloren",
        [(k, kinds[k], f"{100 * kinds[k] / len(losses):.1f}%", bad[k], bluff[k]) for k in sorted(kinds)],
        ("soort", "aantal", "aandeel", "slechte call", "met bluf"),
    )

    rate_table("Per aantal spelers over", played_by_players,
               Counter(l["players"] for l in losses), "spelers")
    rate_table("Per aantal eigen dobbelstenen", played_by_dice,
               Counter(l["my_dice_count"] for l in losses), "dice")
    rate_table("Palifico", played_pal, Counter(l["palifico"] for l in losses), "palifico")

    # Loss kind split by players left: where does each mistake happen?
    by = defaultdict(Counter)
    for l in losses:
        by[l["players"]][l["kind"]] += 1
    all_kinds = sorted(kinds)
    print_table("Soort verlies per aantal spelers",
                [(p, *(by[p][k] for k in all_kinds)) for p in sorted(by)],
                ("spelers", *all_kinds))

    # How sure were we? Only meaningful per kind.
    for kind, label in (("dudo_wrong", "P(bod waar) bij onze foute Dudo"),
                        ("bid_caught", "P(ons bod waar) toen het werd gepakt")):
        ps = Counter(bucket(l["p_true"]) for l in losses if l["kind"] == kind)
        if ps:
            print_table(label, [(b, ps[b]) for b in sorted(ps)], ("P(true)", "aantal"))

    caught = [l for l in losses if l["kind"] == "bid_caught"]
    if caught:
        margins = Counter(l["margin"] for l in caught)
        print_table("Ons gepakte bod: hoeveel te hoog (werkelijk - bod)",
                    [(m, margins[m]) for m in sorted(margins)], ("marge", "aantal"))
    wrong = [l for l in losses if l["kind"] == "dudo_wrong"]
    if wrong:
        margins = Counter(l["margin"] for l in wrong)
        print_table("Onze foute Dudo: hoeveel meer er lag dan geboden",
                    [(m, margins[m]) for m in sorted(margins)], ("marge", "aantal"))

    if show:
        # Worst first: the most confident wrong Dudos and the least likely bids.
        def badness(l):
            return l["p_true"] if l["kind"] == "dudo_wrong" else 1 - l["p_true"]

        print(f"\n{'=' * 72}\nSLECHTSTE {show} CALLS (volgens eigen dobbelstenen)\n{'=' * 72}")
        for l in sorted(losses, key=badness, reverse=True)[:show]:
            print(f"\ngame {l['game']} ronde {l['round']}  [{l['kind']}]  "
                  f"{l['players']} spelers, {l['total_dice']} dice"
                  f"{', PALIFICO' if l['palifico'] else ''}")
            print(f"  wij ({l['my_dice_count']}): {l['my_dice']}   tegenstanders: {l['opp_dice']}")
            print(f"  calls : {l['sequence']}")
            print(f"  onze  : {l['our_call']}  P(true)={l['p_true']:.2f}  "
                  f"werkelijk {l['actual']}{'  (bluf)' if l['bluffed'] else ''}")
            print(f"  dice  : {l['all_dice']}")

    if csv_path and losses:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(losses[0]))
            writer.writeheader()
            writer.writerows(losses)
        print(f"\n{len(losses)} verloren rondes geschreven naar {csv_path}")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def sim_args(p):
        p.add_argument("--games", type=int, default=200)
        p.add_argument("--seed", type=int, default=12345)
        p.add_argument("--out", default="rounds.jsonl", help="JSONL-bestand met alle rondes.")

    def ana_args(p):
        p.add_argument("--bot", default="Pokerface", help="Naam van de bot om te analyseren.")
        p.add_argument("--show", type=int, default=10, help="Aantal slechtste calls om te tonen.")
        p.add_argument("--csv", default=None, help="Optioneel: CSV met alle verloren rondes.")

    sim_args(sub.add_parser("dump", help="Simuleer en schrijf alle rondes weg."))
    p = sub.add_parser("analyze", help="Analyseer een eerder weggeschreven bestand.")
    p.add_argument("path")
    ana_args(p)
    p = sub.add_parser("run", help="dump en daarna analyze.")
    sim_args(p)
    ana_args(p)

    args = parser.parse_args()
    if args.cmd in ("dump", "run"):
        dump(args.games, args.seed, args.out)
    if args.cmd == "analyze":
        analyze(args.path, args.bot, args.show, args.csv)
    elif args.cmd == "run":
        analyze(args.out, args.bot, args.show, args.csv)


if __name__ == "__main__":
    main()
