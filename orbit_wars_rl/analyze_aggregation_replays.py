"""Replay probe for target-level attack aggregation.

Groups each turn's attack launches by resolved target and measures whether
multi-source groups are essential: combined ships meet a projected capture floor
while no single source does.

Timing follows the repo replay convention: action at steps[t] was selected from
observation at steps[t-1].

Usage:
  python3 orbit_wars_rl/analyze_aggregation_replays.py leader-replays/rank1 --mode winners
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


PHASES = ((0, 50, "<50"), (50, 100, "50-100"), (100, 10**9, ">100"))
SHIP_SPEED = 6.0


def _phase(step: int) -> str:
    for lo, hi, name in PHASES:
        if lo <= step < hi:
            return name
    return ">100"


def _pid(p) -> int:
    return int(p[0])


def _owner(p) -> int:
    return int(p[1])


def _xy(p) -> tuple[float, float]:
    return float(p[2]), float(p[3])


def _distance(a, b) -> float:
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return math.hypot(ax - bx, ay - by)


def _eta(src, tgt, ships: int) -> int:
    # Approximate replay probe ETA. The exact engine accounts for surfaces/orbits;
    # this is enough to classify floors at corpus scale.
    speed = 1.0 + (SHIP_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5
    speed = min(speed, SHIP_SPEED)
    return max(1, int(math.ceil(_distance(src, tgt) / speed)))


def _capture_floor(tgt, owner_id: int, eta: int) -> float:
    owner = _owner(tgt)
    ships_at_arrival = min(float(tgt[5]) + float(tgt[6]) * float(eta), 500.0)
    if owner == owner_id:
        return 1.0
    if owner < 0:
        return ships_at_arrival + 1.0
    return ships_at_arrival + float(tgt[6]) * 3.0 + 1.0


def _resolve_launch_target(planets, src, angle: float):
    sx, sy = _xy(src)
    best, best_delta = None, 0.6
    for p in planets:
        if _pid(p) == _pid(src):
            continue
        px, py = _xy(p)
        pa = math.atan2(py - sy, px - sx)
        delta = abs((pa - angle + math.pi) % (2 * math.pi) - math.pi)
        if delta < best_delta:
            best, best_delta = p, delta
    return best


def _winner_seats(replay: dict) -> list[int]:
    rewards = replay.get("rewards") or []
    if not rewards:
        return []
    best = max(rewards)
    return [i for i, r in enumerate(rewards) if r == best]


def _seat_names(replay: dict) -> list[str]:
    return list(replay.get("info", {}).get("TeamNames") or [])


def _iter_paths(inputs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(Path(x) for x in glob.glob(str(p / "*.json")))
        elif any(ch in item for ch in "*?[]"):
            out.extend(Path(x) for x in glob.glob(item))
        else:
            out.append(p)
    return sorted(set(out))


def _selected_seats(replay: dict, mode: str, player: str | None) -> list[int]:
    steps = replay.get("steps") or []
    if not steps:
        return []
    n_players = len(steps[0])
    if player:
        names = _seat_names(replay)
        return [i for i, name in enumerate(names) if name == player]
    if mode == "all":
        return list(range(n_players))
    if mode == "winners":
        return _winner_seats(replay)
    raise ValueError(f"unknown mode: {mode}")


def _pct(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def _add_ratio(c: Counter, key: str, value: float) -> None:
    c[f"{key}_sum"] += float(value)
    c[f"{key}_n"] += 1


def _avg(c: Counter, key: str) -> float:
    return _pct(c[f"{key}_sum"], c[f"{key}_n"])


def analyze(paths: list[Path], mode: str, player: str | None) -> dict:
    totals = Counter()
    phase_totals = defaultdict(Counter)
    per_player = defaultdict(Counter)

    for path in paths:
        try:
            replay = json.loads(path.read_text())
        except Exception:
            totals["read_fail"] += 1
            continue
        steps = replay.get("steps") or []
        if len(steps) < 2:
            totals["short_replay"] += 1
            continue
        names = _seat_names(replay)
        seats = _selected_seats(replay, mode, player)
        if not seats:
            totals["no_selected_seat"] += 1
            continue
        totals["replays"] += 1

        for seat in seats:
            pname = names[seat] if seat < len(names) else f"seat{seat}"
            per_player[pname]["replays"] += 1
            for t in range(1, len(steps)):
                if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
                    continue
                obs = (steps[t - 1][seat] or {}).get("observation") or {}
                action = (steps[t][seat] or {}).get("action") or []
                planets = obs.get("planets") or []
                if not planets or not action:
                    continue
                pmap = {_pid(p): p for p in planets}
                own_pids = {_pid(p) for p in planets if _owner(p) == seat}
                groups: dict[int, list[dict]] = defaultdict(list)

                for move in action:
                    if not isinstance(move, (list, tuple)) or len(move) < 3:
                        continue
                    try:
                        src_pid = int(move[0])
                        angle = float(move[1])
                        ships = int(move[2])
                    except Exception:
                        continue
                    src = pmap.get(src_pid)
                    if src is None or src_pid not in own_pids:
                        continue
                    tgt = _resolve_launch_target(planets, src, angle)
                    if tgt is None:
                        continue
                    # Attack aggregation only. Reinforcement aggregation is a separate
                    # logistics behavior and is already covered by reinforce diagnostics.
                    if _owner(tgt) == seat:
                        continue
                    eta = _eta(src, tgt, ships)
                    groups[_pid(tgt)].append({
                        "src": src,
                        "target": tgt,
                        "ships": float(ships),
                        "eta": eta,
                    })

                if not groups:
                    continue

                ph = _phase(t)
                attack_moves = sum(len(g) for g in groups.values())
                attack_groups = len(groups)
                agg_groups = [g for g in groups.values() if len({ _pid(x["src"]) for x in g }) >= 2]
                agg_moves = sum(len(g) for g in agg_groups)

                base = Counter({
                    "attack_turns": 1,
                    "attack_moves": attack_moves,
                    "attack_groups": attack_groups,
                    "agg_turns": int(bool(agg_groups)),
                    "agg_groups": len(agg_groups),
                    "agg_moves": agg_moves,
                })
                max_sources_per_target = max(
                    (len({_pid(x["src"]) for x in g}) for g in groups.values()), default=0)
                for c in (totals, phase_totals[ph], per_player[pname]):
                    c.update(base)
                    c["max_sources_per_target"] = max(
                        c["max_sources_per_target"], max_sources_per_target)

                for g in agg_groups:
                    target = g[0]["target"]
                    max_eta = max(int(x["eta"]) for x in g)
                    floor = _capture_floor(target, seat, max_eta)
                    total_ships = sum(float(x["ships"]) for x in g)
                    max_single = max(float(x["ships"]) for x in g)
                    source_count = len({_pid(x["src"]) for x in g})
                    coverage = total_ships / max(floor, 1.0)
                    essential = total_ships >= floor and max_single < floor
                    solo_capable = max_single >= floor
                    under_floor = total_ships < floor
                    overkill = max(0.0, total_ships - floor) / max(floor, 1.0)
                    for c in (totals, phase_totals[ph], per_player[pname]):
                        c["agg_eval_groups"] += 1
                        c["agg_essential"] += int(essential)
                        c["agg_solo_capable"] += int(solo_capable)
                        c["agg_under_floor"] += int(under_floor)
                        c["agg_target_enemy"] += int(_owner(target) >= 0)
                        c["agg_target_neutral"] += int(_owner(target) < 0)
                        c["agg_sources_sum"] += source_count
                        _add_ratio(c, "coverage", coverage)
                        _add_ratio(c, "overkill", overkill)

    return {
        "totals": totals,
        "phase_totals": phase_totals,
        "per_player": per_player,
    }


def _print_counter(label: str, c: Counter) -> None:
    turns = c["attack_turns"]
    moves = c["attack_moves"]
    agg_n = c["agg_eval_groups"]
    print(
        f"{label:14s} turns={turns:6d} moves={moves:7d} "
        f"aggTurn={_pct(c['agg_turns'], turns):.3f} "
        f"aggMoves={_pct(c['agg_moves'], moves):.3f} "
        f"aggGroups={agg_n:5d} "
        f"essential={_pct(c['agg_essential'], agg_n):.3f} "
        f"soloCap={_pct(c['agg_solo_capable'], agg_n):.3f} "
        f"underFloor={_pct(c['agg_under_floor'], agg_n):.3f} "
        f"cov={_avg(c, 'coverage'):.2f} "
        f"overkill={_avg(c, 'overkill'):.2f} "
        f"src/group={_pct(c['agg_sources_sum'], agg_n):.2f} "
        f"maxSrcTgt={c['max_sources_per_target']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="Replay JSON files, directories, or globs.")
    ap.add_argument("--mode", choices=("winners", "all"), default="winners")
    ap.add_argument("--player", default=None, help="Analyze this exact TeamNames entry.")
    ap.add_argument("--top-players", type=int, default=8)
    args = ap.parse_args()

    paths = _iter_paths(args.paths)
    res = analyze(paths, args.mode, args.player)
    totals: Counter = res["totals"]

    print(f"paths={len(paths)} mode={args.mode} player={args.player or '-'}")
    print(f"replays_used={totals['replays']} read_fail={totals['read_fail']} no_selected={totals['no_selected_seat']}")
    _print_counter("ALL", totals)
    for _, _, ph in PHASES:
        _print_counter(ph, res["phase_totals"][ph])

    if args.top_players > 0:
        print("\nTop players by attack turns:")
        ranked = sorted(res["per_player"].items(), key=lambda kv: kv[1]["attack_turns"], reverse=True)
        for name, c in ranked[:args.top_players]:
            _print_counter(name[:14], c)


if __name__ == "__main__":
    main()
