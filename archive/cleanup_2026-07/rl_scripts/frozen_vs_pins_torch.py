"""Step 3 — frozen-vs-pins WR duel INSIDE torch_env+comets, threshold-decode.

Closes the last evidence seam after the comet fix. We have two anchor numbers for the frozen
p2rev5 4M policy vs the RL pins (rev38 / rev53b):
    - kaggle env, frozen, threshold-decode (cross-eval) : rev38 27.1% / rev53b 37.5%
    - torch_env+comets, EVOLVING policy, SAMPLED  (in-training pin WR): ~53% / ~60%
Three things differ between them: env (kaggle vs torch), policy (frozen vs evolving), action mode
(decode vs sampled). This harness fixes policy=frozen and mode=decode and runs in torch_env, so the
ONLY remaining difference from the cross-eval is the ENGINE:
    - lands ~27-37%  => torch_env behaviorally matches kaggle for the closed loop (get_features ->
      forward -> decode -> step), confirming the comet fix end-to-end; the in-training 53-60% is
      purely the sampled+evolving confound.
    - lands ~53-60%  => a residual sim gap survives the comet fix -> investigate before from-scratch.

Byte-replay (sim_gap_probe) can't answer this: it feeds FIXED actions, so it never drives a policy
through get_features. This does.

Env config = p2rev5's TRAINING config (allow_reinforce / gate3 / floor0, comets on, no reward
shaping so reward sign is a clean win/loss) — i.e. identical to the setup the in-training pin WR was
measured under, which is exactly the number we're explaining. The pins (allow_reinforce=False, but
acting under the reinforce-allowed mask here) never choose own targets — their own-target head is
untrained — so the shared mask is a near-no-op for them. Both seats are played to balance seat bias.

Run: orbit_wars_rl/.venv/bin/python orbit_wars_rl/frozen_vs_pins_torch.py [--games 256] [--envs 128]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orbit_wars_rl.config import Config
from orbit_wars_rl.model import EntityTransformer
from orbit_wars_rl.eval import load_checkpoint
from orbit_wars_rl.torch_env import VecTorchEnv

FROZEN = "gpu_run_artifacts/p2rev5/checkpoints/torch_step_4194304_p2rev5_20260612_083540.pt"
PINS = {
    "rev38": "gpu_run_artifacts/rev38/checkpoints/torch_step_5242880_rev38_20260605_181635.pt",
    "rev53b": "gpu_run_artifacts/preseed_pool/torch_step_10485760_rev53b.pt",
}
XEVAL = {"rev38": 27.1, "rev53b": 37.5}   # kaggle-env frozen+decode anchors (48g, seat-0)


def load_model(path: str, device: str):
    cfg = Config()
    cfg.env.num_players = 2
    sd, _ = load_checkpoint(path, cfg)               # patches cfg.model dims to the ckpt
    m = EntityTransformer(cfg.model).to(device)
    m.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    m.reinforce_gate_min_planets = 3                 # p2rev5 masks: gate3 / floor0 / no-forward
    m.reinforce_forward_only = False
    m.reinforce_garrison_floor = 0.0
    m.sufficient_commit_factor = 0.0
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def decode_actions(model, feats):
    """Threshold/argmax decode (matches eval.py: fire>0.5, argmax ship/target). Returns the
    (N, MAX_OWNED, 4) [fire, angle=0, ship_bin, target_idx] action tensor env.step expects."""
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


def run_duel(m_frozen, m_pin, n_envs, target_games, frozen_seat, device, seed0=0):
    """Play frozen vs pin in torch_env; frozen controls `frozen_seat`. Returns (wins, games)."""
    env = VecTorchEnv(num_envs=n_envs, num_players=2, device=device,
                      action_decode="target", allow_reinforce=True,
                      reinforce_gate_min_planets=3, reinforce_garrison_floor=0.0)
    env.reset(seeds=[seed0 + i for i in range(n_envs)])
    other = 1 - frozen_seat
    wins = games = 0
    steps = 0
    max_steps = target_games * 12 + 5000   # generous cap; ~150-step games, N parallel
    while games < target_games and steps < max_steps:
        f0 = env.get_features(0, max_planets=48, max_fleets=128)
        f1 = env.get_features(1, max_planets=48, max_fleets=128)
        if frozen_seat == 0:
            act = {0: decode_actions(m_frozen, f0), 1: decode_actions(m_pin, f1)}
        else:
            act = {0: decode_actions(m_pin, f0), 1: decode_actions(m_frozen, f1)}
        _state, rewards, done = env.step(act)
        d = done.bool()
        if d.any():
            rw = rewards[d]
            games += int(d.sum().item())
            wins += int((rw[:, frozen_seat] > rw[:, other]).sum().item())
        steps += 1
    return wins, games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=256, help="games per pin (split over both seats)")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("=" * 78)
    print("FROZEN p2rev5 4M  vs  RL PINS   (torch_env + comets, threshold-decode)")
    print(f"  envs={args.envs}  games/pin={args.games} (both seats)  device={args.device}")
    print("=" * 78)
    frozen = load_model(FROZEN, args.device)
    per_seat = args.games // 2

    for pin_name, pin_path in PINS.items():
        pin = load_model(pin_path, args.device)
        w0, g0 = run_duel(frozen, pin, args.envs, per_seat, frozen_seat=0,
                          device=args.device, seed0=0)
        w1, g1 = run_duel(frozen, pin, args.envs, per_seat, frozen_seat=1,
                          device=args.device, seed0=100000)
        w, g = w0 + w1, g0 + g1
        wr = 100.0 * w / max(g, 1)
        s0 = 100.0 * w0 / max(g0, 1)
        s1 = 100.0 * w1 / max(g1, 1)
        print(f"\n{pin_name}:  WR {wr:5.1f}%  ({w}/{g})   seat0 {s0:.1f}% ({w0}/{g0})  "
              f"seat1 {s1:.1f}% ({w1}/{g1})")
        print(f"    kaggle cross-eval (frozen+decode) anchor: {XEVAL[pin_name]:.1f}%   "
              f"in-training (sampled+evolving): ~{'53' if pin_name=='rev38' else '60'}%")

    print("\n" + "=" * 78)
    print("READ: torch_env decode WR near the kaggle cross-eval anchor => engine behaviorally")
    print("matches kaggle (closed loop), comet fix confirmed end-to-end, and the in-training")
    print("53-60% is the sampled+evolving confound. Near 53-60% => residual sim gap remains.")
    print("=" * 78)


if __name__ == "__main__":
    main()
