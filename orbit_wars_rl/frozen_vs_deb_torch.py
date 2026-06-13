"""Closed-loop fidelity check for the DEB opponent (the load-bearing pool pressure).

Companion to frozen_vs_pins_torch.py. That harness validated OUR model's closed loop
(get_features -> forward -> decode -> step) for RL pins. This one validates the OTHER
opponent we actually train against: the deb (debatreya_1300) heuristic.

deb is NOT "faithful by construction". In training it runs through two torch_env-specific
adapters that have never been compared to kaggle for deb's behavior:
    to_legacy_obs (torch state -> kaggle-format obs)  --[real deb agent()]-->  moves
    -> _heuristic_moves_to_action_tensor (moves -> action tensor + continuous-angle override)
This harness reuses those EXACT functions from train_torch so deb sees and acts identically
to the live pool, then plays frozen p2rev5 4M vs deb inside torch_env+comets, threshold-decode.

Anchor: the kaggle-env panel of the same frozen checkpoint vs deb (run separately via eval.py).
    torch WR near the kaggle anchor  => deb-in-torch is faithful; pool pressure is real.
    torch WR far off (e.g. the ~27-30% in-training sampled number vs a much lower kaggle WR)
        => the deb adapters distort the opponent; the pool gradient is partly fictional.

Run (CPU; deb agent is CPU-only):
    orbit_wars_rl/.venv/bin/python orbit_wars_rl/frozen_vs_deb_torch.py [--games 128] [--envs 64] [--workers 8]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model import EntityTransformer
from eval import load_checkpoint
from torch_env import VecTorchEnv, to_legacy_obs
from train_torch import HeuristicWorkerPool, _heuristic_moves_to_action_tensor

FROZEN = "gpu_run_artifacts/p2rev5/checkpoints/torch_step_4194304_p2rev5_20260612_083540.pt"
DEB = "opponents/candidate_debatreya_1300.py"


def load_frozen(path: str, device: str):
    cfg = Config()
    cfg.env.num_players = 2
    sd, _ = load_checkpoint(path, cfg)
    m = EntityTransformer(cfg.model).to(device)
    m.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    m.reinforce_gate_min_planets = 3            # p2rev5 masks: gate3 / floor0 / no-forward
    m.reinforce_forward_only = False
    m.reinforce_garrison_floor = 0.0
    m.sufficient_commit_factor = 0.0
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def decode_frozen(model, feats):
    """Threshold/argmax decode (matches eval.py + frozen_vs_pins): (N, MAX_OWNED, 4)
    [fire, angle=0, ship_bin, target_idx]."""
    with torch.no_grad():
        o = model(
            feats["planet_features"], feats["fleet_features"], feats["global_features"],
            feats["planet_mask"], feats["fleet_mask"],
            fire_mask=feats["fire_mask"], angle_mask=feats["angle_mask"],
            slot_valid=feats["slot_valid"], owned_indices=feats["owned_indices"],
            owned_count=feats["owned_count"], pairwise_features=feats.get("pairwise_features"),
        )
    fire = (o["fire_logits"].masked_fill(~feats["fire_mask"], -1e9) > 0.0).long()
    ship = o["ship_logits"].argmax(dim=-1)
    tl = o["target_logits"]
    tm = feats.get("target_mask")
    if tm is not None:
        tl = tl.masked_fill(~tm, -1e9)
    target = tl.argmax(dim=-1)
    angle = torch.zeros_like(fire)
    return torch.stack([fire, angle, ship, target], dim=-1)


def deb_action(env, wp, deb_seat, device):
    """Build deb's action through the EXACT training adapters: to_legacy_obs -> real deb
    agent() (worker pool) -> _heuristic_moves_to_action_tensor + continuous-angle override.
    Returns (action (N,MAX_OWNED,4) with target=-1 sentinel, cont_angle (N,MAX_OWNED))."""
    N = env.num_envs
    obs_list = [to_legacy_obs(env, env_idx=e, player=deb_seat) for e in range(N)]
    moves = wp.map(obs_list)
    act, cont_angle = _heuristic_moves_to_action_tensor(moves, env, deb_seat, device)
    # action_decode="target": pad a -1 target sentinel so the env keeps ANGLE decoding for
    # deb's rows (the cont_angle override then bypasses 144-bin quantization), exactly as
    # train_torch.compute_pool_actions does.
    pad = torch.full(act.shape[:-1] + (1,), -1, dtype=act.dtype, device=act.device)
    return torch.cat([act, pad], dim=-1), cont_angle


def run_duel(m_frozen, wp, n_envs, target_games, frozen_seat, device, seed0=0):
    env = VecTorchEnv(num_envs=n_envs, num_players=2, device=device,
                      action_decode="target", allow_reinforce=True,
                      reinforce_gate_min_planets=3, reinforce_garrison_floor=0.0)
    env.reset(seeds=[seed0 + i for i in range(n_envs)])
    deb_seat = 1 - frozen_seat
    wins = games = steps = 0
    max_steps = target_games * 12 + 5000
    while games < target_games and steps < max_steps:
        f_fro = env.get_features(frozen_seat, max_planets=48, max_fleets=128)
        fro_act = decode_frozen(m_frozen, f_fro)
        deb_act, cont = deb_action(env, wp, deb_seat, device)
        actions = {frozen_seat: fro_act, deb_seat: deb_act}
        _state, rewards, done = env.step(actions, angle_overrides={deb_seat: cont})
        d = done.bool()
        if d.any():
            rw = rewards[d]
            games += int(d.sum().item())
            wins += int((rw[:, frozen_seat] > rw[:, deb_seat]).sum().item())
        steps += 1
    return wins, games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=128, help="games total (split over both seats)")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("=" * 78)
    print("FROZEN p2rev5 4M  vs  DEB (debatreya_1300)   (torch_env + comets, threshold-decode)")
    print(f"  envs={args.envs}  games={args.games} (both seats)  workers={args.workers}  device={args.device}")
    print("  deb driven through the live training adapters (to_legacy_obs + moves->action).")
    print("=" * 78)

    frozen = load_frozen(FROZEN, args.device)
    wp = HeuristicWorkerPool(DEB, args.workers)
    try:
        per_seat = args.games // 2
        w0, g0 = run_duel(frozen, wp, args.envs, per_seat, frozen_seat=0, device=args.device, seed0=0)
        w1, g1 = run_duel(frozen, wp, args.envs, per_seat, frozen_seat=1, device=args.device, seed0=100000)
    finally:
        wp.close()

    w, g = w0 + w1, g0 + g1
    wr = 100.0 * w / max(g, 1)
    s0 = 100.0 * w0 / max(g0, 1)
    s1 = 100.0 * w1 / max(g1, 1)
    print(f"\ndeb:  WR {wr:5.1f}%  ({w}/{g})   seat0 {s0:.1f}% ({w0}/{g0})  seat1 {s1:.1f}% ({w1}/{g1})")
    print("\n" + "=" * 78)
    print("READ: compare to the kaggle-env eval.py panel of the SAME frozen ckpt vs deb.")
    print("  near the kaggle WR  => deb-in-torch faithful (pool pressure is real).")
    print("  far off (toward the ~27-30% in-training sampled WR) => deb adapters distort the")
    print("  opponent -> the pool gradient is partly fictional; investigate to_legacy_obs /")
    print("  the moves->action converter before relying on deb-driven selection.")
    print("=" * 78)


if __name__ == "__main__":
    main()
