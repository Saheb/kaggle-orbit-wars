"""Head-to-head runner for arbitrary local Orbit Wars agent files."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from statistics import mean

from kaggle_environments import make

from evaluate_ablation import final_material


ROOT = Path(__file__).resolve().parent


_LOAD_COUNTER = 0


def load_agent(path: Path):
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    spec = importlib.util.spec_from_file_location(f"agent_{path.stem}_{_LOAD_COUNTER}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "agent"):
        raise RuntimeError(f"{path} does not define agent")
    return module.agent


def run_match(candidate_path, baseline_path, seed, slot, debug=False):
    agents = [load_agent(baseline_path) for _ in range(4)]
    agents[slot] = load_agent(candidate_path)
    env = make("orbit_wars", configuration={"seed": seed}, debug=debug)
    env.run(agents)
    rewards = [state.reward for state in env.steps[-1]]
    winner = max(range(4), key=lambda idx: rewards[idx])
    return {
        "winner": winner,
        "candidate_won": winner == slot,
        "candidate_reward": rewards[slot],
        "rewards": rewards,
        "material": final_material(env, 4),
        "steps": len(env.steps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--slot", type=int, choices=range(4), default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    candidate_path = ROOT / args.candidate
    baseline_path = ROOT / args.baseline
    slots = range(4) if args.slot is None else [args.slot]
    rows = []
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        for slot in slots:
            row = run_match(candidate_path, baseline_path, seed, slot, args.debug)
            rows.append(row)
            print(
                f"seed={seed:03d} slot={slot} winner={row['winner']} "
                f"candidate_won={row['candidate_won']} "
                f"candidate_reward={row['candidate_reward']} "
                f"steps={row['steps']} material={row['material']}",
                flush=True,
            )

    print(
        f"\n{args.candidate} vs {args.baseline}: "
        f"wins={sum(row['candidate_won'] for row in rows)}/{len(rows)} "
        f"avg_reward={mean(row['candidate_reward'] for row in rows):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
