"""Diagnostic: run heuristic-vs-heuristic through the torch env conversion path.

If the conversion is bug-free, the same heuristic playing both seats should win
~50/50 in symmetric games. A strong skew is direct evidence that
`_heuristic_moves_to_action_tensor` is corrupting actions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from torch_env import VecTorchEnv, to_legacy_obs
from train_torch import _heuristic_moves_to_action_tensor


def load_agent(py_path: str):
    spec = importlib.util.spec_from_file_location("opp", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def play_heuristic_vs_heuristic(opp_path: str, n_games: int = 32, episode_steps: int = 500):
    agent_fn = load_agent(opp_path)
    device = "cpu"

    env = VecTorchEnv(num_envs=n_games, num_players=2, device=device,
                      episode_steps=episode_steps)
    env.reset(seeds=list(range(50000, 50000 + n_games)))

    p0_wins = 0
    p1_wins = 0
    draws = 0
    done_count = 0

    for step in range(episode_steps + 50):
        actions_per_player = {}
        for player in (0, 1):
            moves_per_env = []
            for e in range(n_games):
                obs = to_legacy_obs(env, env_idx=e, player=player)
                try:
                    moves = agent_fn(obs) or []
                except Exception:
                    moves = []
                moves_per_env.append(moves)
            actions_per_player[player] = _heuristic_moves_to_action_tensor(
                moves_per_env, env, player, device
            )

        _, rewards, done = env.step(actions_per_player)
        for i in torch.where(done)[0].tolist():
            r0, r1 = rewards[i, 0].item(), rewards[i, 1].item()
            if r0 > r1:   p0_wins += 1
            elif r1 > r0: p1_wins += 1
            else:         draws += 1
            done_count += 1
        if done_count >= n_games:
            break

    print(f"After {done_count} games of {opp_path} vs itself via torch-env conversion:")
    print(f"  p0 wins:  {p0_wins}  ({p0_wins/max(done_count,1):.1%})")
    print(f"  p1 wins:  {p1_wins}  ({p1_wins/max(done_count,1):.1%})")
    print(f"  draws:    {draws}")
    if done_count > 0:
        skew = abs(p0_wins - p1_wins) / done_count
        if skew > 0.4:
            print(f"  *** SKEW = {skew:.1%}  — likely seat-asymmetric bug in conversion ***")
        elif skew > 0.2:
            print(f"  ?   skew = {skew:.1%}  — borderline, repeat with more games")
        else:
            print(f"  ok  skew = {skew:.1%}  — conversion looks symmetric")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--opponent", required=True)
    p.add_argument("--games", type=int, default=32)
    args = p.parse_args()
    play_heuristic_vs_heuristic(args.opponent, n_games=args.games)
