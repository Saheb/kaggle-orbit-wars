"""Run a small local Orbit Wars league across arbitrary agent files."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from evaluate_agents import load_agent
from evaluate_ablation import final_material
from kaggle_environments import make


ROOT = Path(__file__).resolve().parent

DEFAULT_AGENTS = {
    "main": "main.py",
    "zach": "candidate_zach_public.py",
    "suneet": "candidate_suneet_lb1200.py",
    "marco": "kernels/marco-dg-v3-3-top-score-1060-5/marco-dg-v3-3-top-score-1060-5.py",
    "roman": "kernels/roman-lb-1224/submission.py",
}


def run_match(candidate_path, baseline_path, seed, slot):
    agents = [load_agent(baseline_path) for _ in range(4)]
    agents[slot] = load_agent(candidate_path)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    rewards = [state.reward for state in env.steps[-1]]
    winner = max(range(4), key=lambda idx: rewards[idx])
    return {
        "winner": winner,
        "candidate_won": winner == slot,
        "candidate_reward": rewards[slot],
        "rewards": rewards,
        "material": final_material(env, 4),
        "candidate_material": final_material(env, 4)[slot],
        "steps": len(env.steps),
    }


def parse_agent_specs(specs):
    agents = dict(DEFAULT_AGENTS)
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Agent spec must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        agents[name] = path
    return agents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="append", default=[], help="Additional/override agent as name=path")
    parser.add_argument("--include", nargs="*", default=None, help="Agent names to include")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--slots", type=int, nargs="*", default=[0], choices=range(4))
    args = parser.parse_args()

    agents = parse_agent_specs(args.agent)
    names = args.include or list(agents)
    missing = [name for name in names if name not in agents]
    if missing:
        raise SystemExit(f"Unknown agents: {', '.join(missing)}")

    totals = {
        name: {"games": 0, "wins": 0, "reward": 0.0, "material": []}
        for name in names
    }
    pair_rows = []
    for candidate in names:
        for baseline in names:
            if candidate == baseline:
                continue
            rows = []
            candidate_path = ROOT / agents[candidate]
            baseline_path = ROOT / agents[baseline]
            for seed in range(args.start_seed, args.start_seed + args.seeds):
                for slot in args.slots:
                    row = run_match(candidate_path, baseline_path, seed, slot)
                    rows.append(row)
                    totals[candidate]["games"] += 1
                    totals[candidate]["wins"] += int(row["candidate_won"])
                    totals[candidate]["reward"] += row["candidate_reward"]
                    totals[candidate]["material"].append(row["material"][slot])
                    print(
                        f"{candidate:>8} vs {baseline:<8} seed={seed:03d} slot={slot} "
                        f"win={row['candidate_won']} reward={row['candidate_reward']} "
                        f"steps={row['steps']} material={row['material']}",
                        flush=True,
                    )
            pair_rows.append((candidate, baseline, rows))

    print("\nPair Summary", flush=True)
    for candidate, baseline, rows in pair_rows:
        print(
            f"{candidate:>8} vs {baseline:<8} "
            f"wins={sum(row['candidate_won'] for row in rows)}/{len(rows)} "
            f"avg_reward={mean(row['candidate_reward'] for row in rows):.3f} "
            f"avg_material={mean(row['candidate_material'] for row in rows):.1f}",
            flush=True,
        )

    print("\nAgent Summary", flush=True)
    for name, stats in sorted(
        totals.items(),
        key=lambda item: (
            item[1]["wins"] / max(1, item[1]["games"]),
            item[1]["reward"] / max(1, item[1]["games"]),
        ),
        reverse=True,
    ):
        games = stats["games"]
        print(
            f"{name:>8}: wins={stats['wins']}/{games} "
            f"win_rate={stats['wins'] / max(1, games):.3f} "
            f"avg_reward={stats['reward'] / max(1, games):.3f} "
            f"avg_material={mean(stats['material']) if stats['material'] else 0:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
