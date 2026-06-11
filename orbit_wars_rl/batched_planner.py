"""GPU-resident, in-process planner-opponent adapter (PROTOTYPE).

Replaces the multiprocessing ``HeuristicWorkerPool`` dispatch for orbit_lite
planner opponents (Ajay / Producer / carbon / flowdiff / jek). Instead of
pickling one obs dict per env to a ``spawn`` pool of CPU workers, it runs the
planner's ``run_turn`` IN-PROCESS on a chosen device, with one persistent
``ProducerLiteRuntime`` per env so the movement cache stays consistent across
turns. Output is a list-of-moves per env — identical to what the worker pool
returns — so it feeds the existing ``_heuristic_moves_to_action_tensor`` unchanged.

WHY THIS SHAPE (read before "why not real batching?"):
  orbit_lite is single-env. ``PlanetMovement.P`` keys its leading dim on planets,
  not a batch; ``run_turn`` uses ``int(obs.P)`` for tensor shapes, has
  data-dependent early returns, and carries per-turn stateful movement memory.
  A true B-axis kernel would mean rewriting the 1962-line movement engine +
  planner core with ragged-padded P/S/T/F and masking. That's a project, not a
  prototype. So this is a per-env Python loop.

  The win is NOT fewer Python iterations — it's:
    (a) the ~30ms/env of torch FLOPs runs on the GPU, OFF the CPU cores that
        deadlocked rev43/44 (7 workers saturated all 8 vCPUs, GPU at 0%);
    (b) no spawn / pickle / IPC per env per step.
  Whether the GPU loop's per-call kernel-launch overhead nets out faster than
  the CPU mp-pool is the empirical question — run bench_batched.py --device cuda
  on the training box for the decisive number.
"""
from __future__ import annotations
import importlib.util, os, sys


def load_planner_module(path: str):
    """Import a candidate planner .py as a module (so @dataclass __module__ resolves)."""
    name = "planner_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    for attr in ("single_obs_to_tensor", "sparse_action_row_to_moves", "ProducerLiteRuntime"):
        if not hasattr(mod, attr):
            raise AttributeError(f"{path} does not export {attr!r} — not a ProducerLite planner?")
    return mod


class BatchedPlannerOpponent:
    """One planner agent, run in-process across ``num_envs`` envs on ``device``.

    Holds a persistent per-env runtime (movement memory). ``moves(env, player)``
    returns ``[[ [from_pid, angle, ships], ... ], ...]`` — one move-list per env.
    """

    def __init__(self, agent_path: str, num_envs: int, device: str = "cpu"):
        import torch
        self.mod = load_planner_module(agent_path)
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.runtimes = [self.mod.ProducerLiteRuntime() for _ in range(num_envs)]

    def reset(self, env_idx: int | None = None):
        idxs = range(self.num_envs) if env_idx is None else [env_idx]
        for e in idxs:
            self.runtimes[e].reset()

    def moves(self, env, player: int, env_slice: slice | None = None):
        """Move-list per env over ``env_slice`` (default all envs).

        Runtimes are keyed by GLOBAL env index, so within a rollout (fixed
        opponent assignment) the movement cache stays warm; on env reset the
        planner sees step==0 and ``ensure_planet_movement`` rebuilds, so it
        self-heals across rollouts without manual reset bookkeeping.
        """
        import torch
        from torch_env import to_legacy_obs
        mod = self.mod
        dev = self.device
        start = (env_slice.start or 0) if env_slice is not None else 0
        stop = (env_slice.stop if env_slice is not None and env_slice.stop is not None
                else self.num_envs)
        out = []
        with torch.no_grad():
            for e in range(start, stop):
                obs = to_legacy_obs(env, env_idx=e, player=player)
                obs_t = mod.single_obs_to_tensor(obs, player_id=player, device=dev)
                row = self.runtimes[e].tensor_action(obs_t)
                out.append(mod.sparse_action_row_to_moves(row, obs, player_id=player))
        return out
