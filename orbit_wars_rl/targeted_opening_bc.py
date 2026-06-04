"""Analyze top-agent openings and extract targeted BC samples.

This script is intentionally narrow:
1. Summarize opening behavior for selected winner agents from replay JSONs.
2. Emit BC samples for a bounded opening window in the exact format bc.py uses.

Unlike the older replay-mining pipeline, this includes empty-action turns by
default. That matters for opening-policy work because "do nothing" is part of
the target policy, not missing data.

Usage:
  python orbit_wars_rl/targeted_opening_bc.py \
      --replay-dir archive/replays/top_agent_replays \
      --agent "Isaiah @ Tufa Labs" \
      --steps-max 50 \
      --summary-out /tmp/isaiah_openings.json \
      --samples-out /tmp/isaiah_openings.pkl

  python orbit_wars_rl/targeted_opening_bc.py \
      --replay-dir archive/replays/top_agent_replays \
      --agent "Isaiah @ Tufa Labs" \
      --steps-max 50 \
      --require-opponent-first-fire-by 12 \
      --summary-out /tmp/isaiah_under_pressure.json \
      --samples-out /tmp/isaiah_under_pressure.pkl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import statistics
from dataclasses import asdict, dataclass
from json import JSONDecodeError

from bc import trajectory_to_training_sample


@dataclass
class SlotOpeningStats:
    name: str
    first_fire_step: int | None
    fire_steps: int
    fire_rate: float
    avg_moves_per_fire_step: float
    avg_ships_per_move: float
    multi_step_rate: float
    planet_count_end: int
    prod_end: float


@dataclass
class ReplayOpeningSummary:
    replay_path: str
    replay_id: int | str | None
    n_players: int
    n_steps: int
    window_steps: int
    winner_idx: int
    winner_name: str
    loser_idx: int
    loser_name: str
    winner: SlotOpeningStats
    loser: SlotOpeningStats
    opponent_first_fire_by_cutoff: int | None
    under_early_pressure: bool
    planet_count_delta_end: int
    prod_delta_end: float
    selected_for_bc: bool


def strict_winner_index(rewards) -> int | None:
    if not rewards or any(r is None for r in rewards):
        return None
    mx = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == mx]
    if len(winners) != 1:
        return None
    return winners[0]


def slot_name(data: dict, idx: int) -> str:
    agents = data.get("info", {}).get("Agents", [])
    if idx < len(agents):
        return agents[idx].get("Name", f"player_{idx}")
    return f"player_{idx}"


def agent_name_matches(name: str, filters: list[str]) -> bool:
    if not filters:
        return True
    lower = name.lower()
    return any(f.lower() in lower for f in filters)


def obs_from_step_agent(step_agent: dict, player_idx: int, step_idx: int) -> dict | None:
    obs = step_agent.get("observation")
    if not obs or "planets" not in obs:
        return None
    out = dict(obs)
    out["player"] = player_idx
    out.setdefault("step", step_idx)
    out.setdefault("angular_velocity", 0.0)
    out.setdefault("comet_planet_ids", [])
    out.setdefault("initial_planets", out["planets"])
    return out


def move_counts(action) -> tuple[int, int]:
    if not isinstance(action, list):
        return 0, 0
    moves = [mv for mv in action if isinstance(mv, list) and len(mv) >= 3]
    return len(moves), sum(int(mv[2]) for mv in moves)


def opening_stats_for_slot(steps: list, slot_idx: int, max_steps: int) -> SlotOpeningStats:
    window = steps[1 : min(len(steps), max_steps + 1)]
    name = slot_name({"info": {"Agents": []}}, slot_idx)
    # caller overwrites the name with replay metadata; keep local shape simple
    first_fire_step = None
    fire_steps = 0
    move_counts_on_fire_steps: list[int] = []
    all_move_ships: list[int] = []
    last_obs = None

    for t, step in enumerate(window, start=1):
        if slot_idx >= len(step):
            continue
        agent_data = step[slot_idx]
        action = agent_data.get("action") or []
        n_moves, _ = move_counts(action)
        if n_moves > 0:
            if first_fire_step is None:
                first_fire_step = t
            fire_steps += 1
            move_counts_on_fire_steps.append(n_moves)
            for mv in action:
                if isinstance(mv, list) and len(mv) >= 3:
                    all_move_ships.append(int(mv[2]))
        if "observation" in agent_data:
            last_obs = agent_data["observation"]

    n_window = max(1, len(window))
    prod_end = 0.0
    planet_count_end = 0
    if last_obs and "planets" in last_obs:
        for p in last_obs["planets"]:
            if int(p[1]) == slot_idx:
                planet_count_end += 1
                prod_end += float(p[6])

    return SlotOpeningStats(
        name=name,
        first_fire_step=first_fire_step,
        fire_steps=fire_steps,
        fire_rate=fire_steps / n_window,
        avg_moves_per_fire_step=(
            statistics.mean(move_counts_on_fire_steps) if move_counts_on_fire_steps else 0.0
        ),
        avg_ships_per_move=statistics.mean(all_move_ships) if all_move_ships else 0.0,
        multi_step_rate=(
            sum(1 for c in move_counts_on_fire_steps if c > 1) / max(1, len(move_counts_on_fire_steps))
        ),
        planet_count_end=planet_count_end,
        prod_end=prod_end,
    )


def summarize_replay(
    replay_path: str,
    max_steps: int,
    require_opponent_first_fire_by: int | None,
    agent_filters: list[str],
) -> ReplayOpeningSummary | None:
    try:
        with open(replay_path) as f:
            data = json.load(f)
    except (OSError, JSONDecodeError):
        return None

    steps = data.get("steps") or []
    if not steps:
        return None
    n_players = len(steps[0])
    if n_players != 2:
        return None

    winner_idx = strict_winner_index(data.get("rewards", []))
    if winner_idx is None:
        return None
    loser_idx = 1 - winner_idx

    winner_name = slot_name(data, winner_idx)
    loser_name = slot_name(data, loser_idx)
    if not agent_name_matches(winner_name, agent_filters):
        return None

    winner_stats = opening_stats_for_slot(steps, winner_idx, max_steps)
    loser_stats = opening_stats_for_slot(steps, loser_idx, max_steps)
    winner_stats.name = winner_name
    loser_stats.name = loser_name

    under_early_pressure = True
    if require_opponent_first_fire_by is not None:
        under_early_pressure = (
            loser_stats.first_fire_step is not None
            and loser_stats.first_fire_step <= require_opponent_first_fire_by
        )

    return ReplayOpeningSummary(
        replay_path=replay_path,
        replay_id=data.get("id"),
        n_players=n_players,
        n_steps=len(steps),
        window_steps=min(max_steps, max(0, len(steps) - 1)),
        winner_idx=winner_idx,
        winner_name=winner_name,
        loser_idx=loser_idx,
        loser_name=loser_name,
        winner=winner_stats,
        loser=loser_stats,
        opponent_first_fire_by_cutoff=require_opponent_first_fire_by,
        under_early_pressure=under_early_pressure,
        planet_count_delta_end=winner_stats.planet_count_end - loser_stats.planet_count_end,
        prod_delta_end=winner_stats.prod_end - loser_stats.prod_end,
        selected_for_bc=under_early_pressure,
    )


def extract_bc_samples(
    replay_path: str,
    winner_idx: int,
    max_steps: int,
    include_empty_actions: bool,
) -> list[dict]:
    try:
        with open(replay_path) as f:
            data = json.load(f)
    except (OSError, JSONDecodeError):
        return []
    steps = data.get("steps") or []
    samples: list[dict] = []

    for step_idx in range(1, min(len(steps), max_steps + 1)):
        step = steps[step_idx]
        if winner_idx >= len(step):
            continue
        agent_data = step[winner_idx]
        action = agent_data.get("action")
        if not include_empty_actions and not action:
            continue
        obs = obs_from_step_agent(agent_data, winner_idx, step_idx)
        if obs is None:
            continue
        sample = trajectory_to_training_sample({
            "obs": obs,
            "action": action or [],
        })
        if sample is not None:
            samples.append(sample)
    return samples


def print_report(summaries: list[ReplayOpeningSummary]) -> None:
    if not summaries:
        print("No matching 2P strict-winner replays found.")
        return

    by_agent: dict[str, list[ReplayOpeningSummary]] = {}
    for s in summaries:
        by_agent.setdefault(s.winner_name, []).append(s)

    print(f"Matching replays: {len(summaries)}")
    for agent_name, rows in sorted(by_agent.items()):
        first_fire = [r.winner.first_fire_step for r in rows if r.winner.first_fire_step is not None]
        opp_first_fire = [r.loser.first_fire_step for r in rows if r.loser.first_fire_step is not None]
        print()
        print(agent_name)
        print(f"  replays={len(rows)} selected_for_bc={sum(r.selected_for_bc for r in rows)}")
        print(f"  winner first-fire avg={statistics.mean(first_fire):.2f}" if first_fire else "  winner first-fire avg=n/a")
        print(f"  opp first-fire avg={statistics.mean(opp_first_fire):.2f}" if opp_first_fire else "  opp first-fire avg=n/a")
        print(f"  winner fire_rate avg={statistics.mean(r.winner.fire_rate for r in rows):.3f}")
        print(f"  winner avg ships/move={statistics.mean(r.winner.avg_ships_per_move for r in rows):.1f}")
        print(f"  winner avg moves/fire-step={statistics.mean(r.winner.avg_moves_per_fire_step for r in rows):.2f}")
        print(f"  winner prod_delta_end avg={statistics.mean(r.prod_delta_end for r in rows):.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replay-dir", action="append", required=True,
                   help="Directory containing replay JSONs. Repeatable.")
    p.add_argument("--glob", default="*.json",
                   help="Replay filename glob inside each replay dir.")
    p.add_argument("--agent", action="append", default=[],
                   help="Winner-agent substring filter. Repeatable.")
    p.add_argument("--steps-max", type=int, default=50,
                   help="Opening window to analyze and extract.")
    p.add_argument("--require-opponent-first-fire-by", type=int, default=None,
                   help="Optional filter: keep only wins where the loser first fired by this step.")
    p.add_argument("--drop-empty-actions", action="store_true",
                   help="If set, exclude no-op turns from BC samples.")
    p.add_argument("--summary-out", default="",
                   help="Optional JSON summary output path.")
    p.add_argument("--samples-out", default="",
                   help="Optional .pkl output path for BC samples.")
    args = p.parse_args()

    replay_paths: list[str] = []
    for replay_dir in args.replay_dir:
        replay_paths.extend(sorted(glob.glob(os.path.join(replay_dir, args.glob))))
    print(f"Found {len(replay_paths)} replay files")

    summaries: list[ReplayOpeningSummary] = []
    samples: list[dict] = []

    for replay_path in replay_paths:
        summary = summarize_replay(
            replay_path=replay_path,
            max_steps=args.steps_max,
            require_opponent_first_fire_by=args.require_opponent_first_fire_by,
            agent_filters=args.agent,
        )
        if summary is None:
            continue
        summaries.append(summary)
        if summary.selected_for_bc:
            samples.extend(extract_bc_samples(
                replay_path=replay_path,
                winner_idx=summary.winner_idx,
                max_steps=args.steps_max,
                include_empty_actions=not args.drop_empty_actions,
            ))

    print_report(summaries)
    print()
    print(f"BC samples emitted: {len(samples)}")

    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        payload = {
            "config": {
                "replay_dirs": args.replay_dir,
                "glob": args.glob,
                "agents": args.agent,
                "steps_max": args.steps_max,
                "require_opponent_first_fire_by": args.require_opponent_first_fire_by,
                "drop_empty_actions": args.drop_empty_actions,
            },
            "replay_count": len(summaries),
            "sample_count": len(samples),
            "summaries": [asdict(s) for s in summaries],
        }
        with open(args.summary_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Summary saved to {args.summary_out}")

    if args.samples_out:
        os.makedirs(os.path.dirname(args.samples_out) or ".", exist_ok=True)
        with open(args.samples_out, "wb") as f:
            pickle.dump(samples, f)
        print(f"Samples saved to {args.samples_out}")


if __name__ == "__main__":
    main()
