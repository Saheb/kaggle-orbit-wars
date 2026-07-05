"""Vectorized checkpoint-vs-checkpoint panel eval on the GPU (MPS/CUDA/CPU).

Why this exists: eval.py runs 256 games sequentially through kaggle_environments
(one game at a time, single core) — ~45 min per panel. For MODEL-vs-MODEL
comparisons (e.g. a new checkpoint vs the rev11-2M starting point) both sides are
the EntityTransformer, which already runs batched in the training env
(torch_env.VecTorchEnv). So we can run all 256 games in parallel on the local
Metal GPU in seconds.

Scope: this ONLY does checkpoint-vs-checkpoint (both sides neural nets). Rule-based
opponents (Zach/HB/Suneet) are written against the kaggle obs format and can't be
batched here — keep using eval.py for those.

Fidelity note: this uses torch_env (the training env), not kaggle_environments.
For a RELATIVE head-to-head metric (both checkpoints in the same env) that is the
right tool. Greedy actions (sample=False) match eval.py's panel semantics.

Usage:
    python orbit_wars_rl/eval_vec.py \
        --checkpoint <new_ckpt.pt> \
        --opponent-checkpoint <initial_ckpt.pt> \
        --games 256 --device mps
"""
from __future__ import annotations

import os
# Let unsupported MPS ops fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time

import torch

from orbit_wars_rl.config import Config
from orbit_wars_rl.model import EntityTransformer
from orbit_wars_rl.torch_env import VecTorchEnv
from orbit_wars_rl.eval import load_checkpoint


def _build_model(path: str, device: torch.device) -> tuple[EntityTransformer, str, str]:
    """Load a checkpoint into a model on `device`. Returns (model, action_decode, ship_bin_mode)."""
    cfg = Config()
    state_dict, action_decode = load_checkpoint(path, cfg)
    model = EntityTransformer(cfg.model).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, action_decode, cfg.model.ship_bin_mode


@torch.no_grad()
def _greedy_actions(model: EntityTransformer, env: VecTorchEnv, player: int) -> torch.Tensor:
    """Greedy (argmax / prob>0.5) action tensor (N, MAX_OWNED, 4) for `player`.

    Matches eval.py's panel (sample=False): fire if prob>0.5 (logit>0), argmax
    ship bin, argmax target planet. angle column is zero (target-decode).
    """
    f = env.get_features(player, max_planets=env_max_planets, max_fleets=128)
    outs = model(
        f["planet_features"], f["fleet_features"], f["global_features"],
        f["planet_mask"], f["fleet_mask"],
        fire_mask=f["fire_mask"], angle_mask=f["angle_mask"],
        slot_valid=f["slot_valid"], owned_indices=f["owned_indices"],
        owned_count=f["owned_count"], pairwise_features=f.get("pairwise_features"),
    )
    fire_logits = outs["fire_logits"].masked_fill(~f["fire_mask"], -1e9)
    fire_a = (fire_logits > 0.0).long()                      # prob>0.5
    ship_a = outs["ship_logits"].argmax(dim=-1)              # argmax bin
    target_logits = outs["target_logits"]
    tm = f.get("target_mask")
    if tm is not None:
        target_logits = target_logits.masked_fill(~tm, -1e9)
    target_a = target_logits.argmax(dim=-1)
    angle_a = torch.zeros_like(fire_a)
    return torch.stack([fire_a, angle_a, ship_a, target_a], dim=-1)


def _run_batch(model_p0, model_p1, seeds, device, action_decode, ship_bin_mode,
               episode_steps) -> torch.Tensor:
    """Play one game per seed: model_p0 as player 0, model_p1 as player 1.

    Returns a (len(seeds),) tensor: +1 = player 0 won, -1 = player 1 won, 0 = draw.
    Captures each env's result at its FIRST done (env auto-resets afterwards).
    """
    n = len(seeds)
    env = VecTorchEnv(num_envs=n, num_players=2, device=device,
                      episode_steps=episode_steps, ship_bin_mode=ship_bin_mode,
                      action_decode=action_decode)
    env.reset(seeds=seeds)

    finished = torch.zeros(n, dtype=torch.bool, device=device)
    result = torch.zeros(n, device=device)   # +1 p0, -1 p1, 0 draw

    for _ in range(episode_steps + 2):
        a0 = _greedy_actions(model_p0, env, 0)
        a1 = _greedy_actions(model_p1, env, 1)
        _, rewards, done = env.step({0: a0, 1: a1})   # rewards (N,2), done (N,)
        newly = done & (~finished)
        if newly.any():
            margin = rewards[:, 0] - rewards[:, 1]
            r = torch.zeros(n, device=device)
            r[margin > 0] = 1.0
            r[margin < 0] = -1.0
            result[newly] = r[newly]
            finished |= done
        if finished.all():
            break
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Model A (the checkpoint under test)")
    ap.add_argument("--opponent-checkpoint", required=True, help="Model B (e.g. the initial/resume ckpt)")
    ap.add_argument("--games", type=int, default=256, help="Total games (split into seeds x 2 seats)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--device", default="mps", help="mps | cuda | cpu")
    ap.add_argument("--episode-steps", type=int, default=500)
    args = ap.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable — falling back to CPU"); args.device = "cpu"
    device = torch.device(args.device)

    model_a, dec_a, sbm_a = _build_model(args.checkpoint, device)
    model_b, dec_b, sbm_b = _build_model(args.opponent_checkpoint, device)
    assert dec_a == dec_b, f"action_decode mismatch: {dec_a} vs {dec_b}"
    assert sbm_a == sbm_b, f"ship_bin_mode mismatch: {sbm_a} vs {sbm_b}"

    n_seeds = args.games // 2
    seeds = list(range(args.seed_start, args.seed_start + n_seeds))

    t0 = time.time()
    # Seat-balanced panel: A as p0 over all seeds, then A as p1 over same seeds.
    res_a_p0 = _run_batch(model_a, model_b, seeds, device, dec_a, sbm_a, args.episode_steps)
    res_a_p1 = _run_batch(model_b, model_a, seeds, device, dec_a, sbm_a, args.episode_steps)

    # Count A's wins: as p0, A wins when result == +1; as p1, when result == -1.
    a_wins = int((res_a_p0 > 0).sum() + (res_a_p1 < 0).sum())
    a_draws = int((res_a_p0 == 0).sum() + (res_a_p1 == 0).sum())
    total = 2 * n_seeds
    dt = time.time() - t0

    print(f"\nVectorized head-to-head ({args.device}, {total} games, {dt:.1f}s, "
          f"{total/dt:.0f} games/s)")
    print(f"  A = {os.path.basename(args.checkpoint)}")
    print(f"  B = {os.path.basename(args.opponent_checkpoint)}")
    print(f"Overall: {a_wins}/{total}  ({a_wins/total:.1%})   draws={a_draws}")


# get_features needs a fixed max_planets; mirror training/eval default (48).
env_max_planets = 48

if __name__ == "__main__":
    main()
