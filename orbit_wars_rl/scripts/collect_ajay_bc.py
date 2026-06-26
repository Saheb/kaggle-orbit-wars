"""Collect BC dataset from Ajay heuristic for pairwise-feature fine-tuning.

Runs Ajay as P0 in N self-play games against itself (P1 also Ajay),
records (obs, action) pairs from both sides, converts to tensors via
trajectory_to_training_sample, and saves a .pkl for bc.py --samples.

Usage:
    python orbit_wars_rl/collect_ajay_bc.py \\
        --games 1000 \\
        --out seed_checkpoints/ajay_bc_1k.pkl

Then fine-tune:
    python orbit_wars_rl/bc.py \\
        --samples seed_checkpoints/ajay_bc_1k.pkl \\
        --init-checkpoint seed_checkpoints/rev32b_6M_pairwise15.pt \\
        --steps 5000 \\
        --lr 1e-4 \\
        --save seed_checkpoints/rev32b_6M_ajay_bc.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.bc import trajectory_to_training_sample  # noqa: E402


def _load_agent(path: str):
    spec = importlib.util.spec_from_file_location("agent_module", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # required on Python 3.14: dataclasses resolves __module__
    spec.loader.exec_module(mod)
    return mod.agent


def collect(agent_path: str, num_games: int, verbose: bool = True) -> list[dict]:
    from orbit_wars_rl.env import OrbitWarsEnv  # available inside venv

    agent_fn = _load_agent(agent_path)
    raw: list[dict] = []

    for seed in range(num_games):
        env = OrbitWarsEnv(num_players=2, seed=seed)
        obs0 = env.reset(seed=seed)
        done = False
        while not done:
            obs1 = env._get_obs(1)
            action0 = agent_fn(obs0)
            action1 = agent_fn(obs1)
            if action0:
                raw.append({"obs": obs0, "action": action0})
            if action1:
                raw.append({"obs": obs1, "action": action1})
            obs0, _, done, _ = env.step(action0 or [], opponent_actions=[action1 or []])

        if verbose and (seed + 1) % 100 == 0:
            print(f"  {seed + 1}/{num_games} games  {len(raw)} raw transitions")

    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--agent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Collecting {args.games} games from {args.agent} ...")
    raw = collect(args.agent, args.games)
    print(f"Raw transitions: {len(raw)}")

    print("Converting to training samples ...")
    samples = []
    skipped = 0
    for r in raw:
        s = trajectory_to_training_sample(r)
        if s is None:
            skipped += 1
        else:
            samples.append(s)
    print(f"Samples: {len(samples)}  (skipped {skipped} empty-owned states)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(samples, f)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
