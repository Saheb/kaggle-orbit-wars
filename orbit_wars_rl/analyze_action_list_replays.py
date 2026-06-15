"""Replay probe for action-list architecture pressure.

Measures two capabilities the official environment supports but the current
policy grammar does not fully express:

1. same-source multi-move: one source planet appears in multiple moves in the
   same turn;
2. same-turn aggregation: multiple sources aim at the same target planet in the
   same turn.

Timing follows the replay convention used elsewhere in this repo: action at
steps[t] was selected from observation at steps[t-1].

Usage:
  python3 orbit_wars_rl/analyze_action_list_replays.py leader-replays/rank1 \
      archive/replays/top_agent_replays --mode winners
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PHASES = ((0, 50, "<50"), (50, 100, "50-100"), (100, 10**9, ">100"))


def _phase(step: int) -> str:
    for lo, hi, name in PHASES:
        if lo <= step < hi:
            return name
    return ">100"


def _pid(p) -> int:
    return int(p[0])


def _owner(p) -> int:
    return int(p[1])


def _resolve_launch_target(planets, src, angle: float):
    """Direction-match target resolver, matching eval's replay conversion metric."""
    sx, sy = float(src[2]), float(src[3])
    best, best_delta = None, 0.6
    for p in planets:
        if _pid(p) == _pid(src):
            continue
        pa = math.atan2(float(p[3]) - sy, float(p[2]) - sx)
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

                source_counts = Counter()
                source_targets = defaultdict(set)
                target_sources = defaultdict(set)
                resolved_moves = 0
                owned_source_moves = 0

                for move in action:
                    if not isinstance(move, (list, tuple)) or len(move) < 3:
                        continue
                    try:
                        src_pid = int(move[0])
                        angle = float(move[1])
                    except Exception:
                        continue
                    src = pmap.get(src_pid)
                    if src is None or src_pid not in own_pids:
                        continue
                    owned_source_moves += 1
                    tgt = _resolve_launch_target(planets, src, angle)
                    if tgt is None:
                        continue
                    tgt_pid = _pid(tgt)
                    resolved_moves += 1
                    source_counts[src_pid] += 1
                    source_targets[src_pid].add(tgt_pid)
                    target_sources[tgt_pid].add(src_pid)

                if owned_source_moves == 0:
                    continue

                ph = _phase(t)
                same_source_sources = sum(1 for c in source_counts.values() if c > 1)
                same_source_moves = sum(c for c in source_counts.values() if c > 1)
                split_sources = sum(1 for src, tgts in source_targets.items()
                                    if source_counts[src] > 1 and len(tgts) > 1)
                aggregate_targets = sum(1 for sources in target_sources.values()
                                        if len(sources) > 1)
                aggregate_moves = sum(len(sources) for sources in target_sources.values()
                                      if len(sources) > 1)

                max_moves_turn = owned_source_moves
                max_source_repeats = max(source_counts.values(), default=0)
                max_sources_per_target = max((len(s) for s in target_sources.values()), default=0)
                row = Counter({
                    "turns": 1,
                    "moves": owned_source_moves,
                    "resolved_moves": resolved_moves,
                    "turns_gt16_moves": int(owned_source_moves > 16),
                    "turns_same_source": int(same_source_sources > 0),
                    "same_source_sources": same_source_sources,
                    "same_source_moves": same_source_moves,
                    "split_sources": split_sources,
                    "turns_split_source": int(split_sources > 0),
                    "aggregate_targets": aggregate_targets,
                    "aggregate_moves": aggregate_moves,
                    "turns_aggregate": int(aggregate_targets > 0),
                })

                totals.update(row)
                phase_totals[ph].update(row)
                per_player[pname].update(row)
                totals["max_moves_turn"] = max(totals["max_moves_turn"], max_moves_turn)
                totals["max_source_repeats"] = max(totals["max_source_repeats"], max_source_repeats)
                totals["max_sources_per_target"] = max(totals["max_sources_per_target"], max_sources_per_target)
                phase_totals[ph]["max_moves_turn"] = max(phase_totals[ph]["max_moves_turn"], max_moves_turn)
                phase_totals[ph]["max_source_repeats"] = max(phase_totals[ph]["max_source_repeats"], max_source_repeats)
                phase_totals[ph]["max_sources_per_target"] = max(
                    phase_totals[ph]["max_sources_per_target"], max_sources_per_target)
                per_player[pname]["max_moves_turn"] = max(per_player[pname]["max_moves_turn"], max_moves_turn)
                per_player[pname]["max_source_repeats"] = max(per_player[pname]["max_source_repeats"], max_source_repeats)
                per_player[pname]["max_sources_per_target"] = max(
                    per_player[pname]["max_sources_per_target"], max_sources_per_target)

    return {
        "totals": totals,
        "phase_totals": phase_totals,
        "per_player": per_player,
    }


def _print_counter(label: str, c: Counter) -> None:
    turns = c["turns"]
    moves = c["moves"]
    print(
        f"{label:14s} turns={turns:6d} moves={moves:7d} "
        f"sameSrcTurn={_pct(c['turns_same_source'], turns):.3f} "
        f"sameSrcMoves={_pct(c['same_source_moves'], moves):.3f} "
        f"splitSrcTurn={_pct(c['turns_split_source'], turns):.3f} "
        f"aggTurn={_pct(c['turns_aggregate'], turns):.3f} "
        f"aggMoves={_pct(c['aggregate_moves'], moves):.3f} "
        f">16turn={_pct(c['turns_gt16_moves'], turns):.3f} "
        f"maxMoves={c['max_moves_turn']} "
        f"maxSrcRepeats={c['max_source_repeats']} "
        f"maxSourcesTgt={c['max_sources_per_target']}"
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
        print("\nTop players by analyzed turns:")
        ranked = sorted(res["per_player"].items(), key=lambda kv: kv[1]["turns"], reverse=True)
        for name, c in ranked[:args.top_players]:
            _print_counter(name[:14], c)


if __name__ == "__main__":
    main()
