"""Vectorized environment pool for parallel rollout collection.

Architecture: N worker processes, each running their own OrbitWarsEnv.
Workers handle env.step + feature extraction + action masking in parallel.
Main process batches model inference across all N workers for GPU efficiency.

Expected throughput on M4 (N=4, MPS batched):
  env+features parallel: ~9ms
  model forward BS=4 MPS: ~12ms
  IPC overhead:           ~3ms
  total per round:        ~12ms for 4 transitions → ~333 SPS
  vs single-env CPU:      ~21ms per transition   →  ~47 SPS
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from multiprocessing.connection import Connection
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _worker_fn(
    conn: Connection,
    worker_id: int,
    num_players: int,
    opponent_policy: str,
    env_backend: str,
):
    """Worker entry point — runs in a separate process.

    Protocol:
        recv ("reset", seed)      → send ("ok", obs_data)
        recv ("step", (actions,)) → send ("ok", step_data) [auto-resets on done]
        recv ("close", None)      → exit
    """
    # Ensure orbit_wars_rl is importable in the spawned process
    _dir = os.path.dirname(os.path.abspath(__file__))
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

    if env_backend == "fast":
        from fast_env import FastOrbitWarsEnv as EnvCls
    elif env_backend == "kaggle":
        from env import OrbitWarsEnv as EnvCls
    else:
        raise ValueError(f"Unknown env_backend: {env_backend!r}")
    from features import extract_features
    from action_mask import compute_action_masks

    env = EnvCls(num_players=num_players, opponent_policy=opponent_policy)

    def _extract(obs):
        player = obs.get("player", 0)
        feats = extract_features(obs, player, num_players=num_players)
        masks = compute_action_masks(obs, player)
        return {
            # features (as numpy for low-overhead pickling)
            "planet_features": feats["planet_features"].numpy(),
            "fleet_features":  feats["fleet_features"].numpy(),
            "global_features": feats["global_features"].numpy(),
            "planet_mask":     feats["planet_mask"].numpy(),
            "fleet_mask":      feats["fleet_mask"].numpy(),
            "owned_indices":   feats["owned_indices"].numpy(),
            "owned_count":     feats["owned_count"],
            # masks
            "fire_mask":   masks["fire_mask"][0].numpy(),    # (max_owned,)
            "angle_mask":  masks["angle_mask"][0].numpy(),   # (max_owned, 72)
            "slot_valid":  masks["slot_valid"][0].numpy(),   # (max_owned,)
            "max_ships":   masks["max_ships"][0].numpy(),    # (max_owned,)
            # raw obs for actions_from_policy
            "obs": obs,
        }

    while True:
        try:
            cmd, data = conn.recv()
        except EOFError:
            break

        if cmd == "reset":
            seed = int(data)
            obs = env.reset(seed=seed)
            conn.send(("ok", _extract(obs)))

        elif cmd == "step":
            actions = data
            obs, reward, done, _ = env.step(actions)

            step_data = _extract(obs)
            step_data["reward"] = float(reward)
            step_data["done"] = bool(done)

            if done:
                # Capture terminal reward, then auto-reset for next episode
                final_reward = env._compute_reward()
                step_data["terminal_reward"] = float(final_reward) if final_reward is not None else 0.0
                next_seed = np.random.randint(0, 2**31)
                new_obs = env.reset(seed=int(next_seed))
                step_data.update(_extract(new_obs))  # overwrite obs data with post-reset
                step_data["done"] = True             # keep done flag
            else:
                step_data["terminal_reward"] = None

            conn.send(("ok", step_data))

        elif cmd == "close":
            conn.close()
            break


# ---------------------------------------------------------------------------
# Pool class
# ---------------------------------------------------------------------------

class VecEnvPool:
    """N parallel environment workers with shared-memory-free IPC via Pipe.

    Usage::
        with VecEnvPool(num_envs=4) as pool:
            obs_data = pool.reset()
            step_data = pool.step(actions_list)
    """

    def __init__(
        self,
        num_envs: int,
        num_players: int = 2,
        base_seed: int = 0,
        opponent_policy: str = "random",
        env_backend: str = "kaggle",
    ):
        self.num_envs = num_envs
        self.num_players = num_players
        self.base_seed = base_seed
        self.opponent_policy = opponent_policy
        self.env_backend = env_backend
        self._closed = False

        ctx = mp.get_context("spawn")
        self._parent_conns: list[Connection] = []
        self._workers: list[mp.Process] = []

        for i in range(num_envs):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_worker_fn,
                args=(child_conn, i, num_players, opponent_policy, env_backend),
                daemon=True,
            )
            p.start()
            child_conn.close()   # close child end in parent process
            self._parent_conns.append(parent_conn)
            self._workers.append(p)

    # ------------------------------------------------------------------
    def reset(self, seeds: Optional[list] = None) -> list[dict]:
        if seeds is None:
            seeds = [self.base_seed + i for i in range(self.num_envs)]
        for conn, seed in zip(self._parent_conns, seeds):
            conn.send(("reset", int(seed)))
        return [conn.recv()[1] for conn in self._parent_conns]

    def step(self, actions_list: list) -> list[dict]:
        """Send actions to all workers and receive (obs, reward, done) dicts."""
        for conn, actions in zip(self._parent_conns, actions_list):
            conn.send(("step", actions))
        return [conn.recv()[1] for conn in self._parent_conns]

    # ------------------------------------------------------------------
    def close(self):
        if self._closed:
            return
        self._closed = True
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
                conn.close()
            except Exception:
                pass
        for w in self._workers:
            w.join(timeout=3)
            if w.is_alive():
                w.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Helpers for converting worker dicts → batched tensors
# ---------------------------------------------------------------------------

def _to_tensors(obs_data_list: list[dict]) -> tuple[dict, dict]:
    """Stack numpy arrays from N worker dicts into (feature_batch, mask_batch) tensors."""
    N = len(obs_data_list)
    d = obs_data_list

    features = {
        "planet_features": torch.from_numpy(np.stack([x["planet_features"] for x in d])),
        "fleet_features":  torch.from_numpy(np.stack([x["fleet_features"]  for x in d])),
        "global_features": torch.from_numpy(np.stack([x["global_features"] for x in d])),
        "planet_mask":     torch.from_numpy(np.stack([x["planet_mask"]     for x in d])),
        "fleet_mask":      torch.from_numpy(np.stack([x["fleet_mask"]      for x in d])),
    }
    masks = {
        "fire_mask":    torch.from_numpy(np.stack([x["fire_mask"]   for x in d])),  # (N, max_owned)
        "angle_mask":   torch.from_numpy(np.stack([x["angle_mask"]  for x in d])),  # (N, max_owned, 72)
        "slot_valid":   torch.from_numpy(np.stack([x["slot_valid"]  for x in d])),  # (N, max_owned)
        "owned_indices":torch.from_numpy(np.stack([x["owned_indices"]for x in d])),  # (N, max_owned)
        "max_ships":    torch.from_numpy(np.stack([x["max_ships"]   for x in d])),  # (N, max_owned)
        "owned_counts": [x["owned_count"] for x in d],
    }
    return features, masks
