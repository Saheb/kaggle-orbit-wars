"""
Compare FireP at steps 0/1/2 across checkpoints.

Strategy: run one reference game (seed=6462, seat=0 vs Zach) with the first
checkpoint, capture the raw observations at steps 0..N. Then replay those
SAME observations through every checkpoint so all models see identical boards.

Usage:
  python3 orbit_wars_rl/step_firep.py \
    --checkpoints path1.pt path2.pt ... \
    --target-decode
"""
import argparse, sys, torch
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orbit_wars_rl.config import Config
from orbit_wars_rl.model import EntityTransformer
from orbit_wars_rl.eval import load_checkpoint, build_agent_fn
from orbit_wars_rl.features import extract_features
from orbit_wars_rl.action_mask import compute_action_masks

SEED = 6462
SEAT = 0       # our seat
N_STEPS = 4    # steps 0..3


def capture_reference_obs(checkpoint, target_decode=False, seed=SEED, seat=SEAT):
    """Step through a game manually and capture obs at steps 0..N_STEPS-1."""
    import subprocess, json, tempfile, os
    from kaggle_environments import make

    cfg = Config()
    sd, _ = load_checkpoint(checkpoint, cfg)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd)
    device = torch.device("cpu")
    model = model.to(device).eval()
    agent_fn = build_agent_fn(model, device,
                               ship_bin_mode=cfg.model.ship_bin_mode,
                               target_decode=target_decode)

    opp_path = str(Path(__file__).parent.parent / "opponents" / "candidate_zach_public.py")
    # Load opponent as a python function
    opp_ns = {}
    exec(open(opp_path).read(), opp_ns)
    opp_fn = opp_ns.get("agent") or opp_ns.get("orbit_agent") or list(
        v for v in opp_ns.values() if callable(v) and not v.__name__.startswith("_"))[-1]

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    obs_list = env.reset()

    captured = {}
    for step in range(N_STEPS + 50):  # run enough to get N_STEPS obs for our seat
        if len(captured) >= N_STEPS:
            break
        # Each player's obs
        obs0 = obs_list[0].observation
        obs1 = obs_list[1].observation

        if seat == 0:
            our_obs = obs0
        else:
            our_obs = obs1

        player_idx = int(getattr(our_obs, "player", seat))
        captured[len(captured)] = (our_obs, player_idx)

        # Get actions
        try:
            act0 = agent_fn(obs0, env.configuration) if seat == 0 else opp_fn(obs0, env.configuration)
        except Exception:
            act0 = []
        try:
            act1 = opp_fn(obs1, env.configuration) if seat == 0 else agent_fn(obs1, env.configuration)
        except Exception:
            act1 = []

        obs_list = env.step([act0, act1])
        if env.done:
            break

    return captured  # {step: (obs, player_idx)}


def firep_from_obs(model, obs, player, device):
    """Given a frozen obs + player index, return max FireP across valid ships."""
    feats = extract_features(obs, player, num_players=2)
    masks = compute_action_masks(obs, player)

    planet_feats = feats["planet_features"].unsqueeze(0).to(device)
    fleet_feats  = feats["fleet_features"].unsqueeze(0).to(device)
    global_feats = feats["global_features"].unsqueeze(0).to(device)
    pw = feats.get("pairwise_features")
    pw = pw.unsqueeze(0).to(device) if pw is not None else None

    planet_mask = feats["planet_mask"].bool().unsqueeze(0).to(device)
    fleet_mask  = feats["fleet_mask"].bool().unsqueeze(0).to(device)
    slot_valid  = masks["slot_valid"].to(device)     # (1, max_owned)
    owned_idx   = masks["owned_indices"].to(device)
    owned_cnt   = masks["owned_count"]

    with torch.no_grad():
        out = model(planet_feats, fleet_feats, global_feats,
                    planet_mask, fleet_mask,
                    slot_valid=slot_valid,
                    owned_indices=owned_idx,
                    owned_count=owned_cnt,
                    pairwise_features=pw)

    # model returns dict; fire_logits shape (B, max_owned) — Bernoulli logit per slot
    fl = out["fire_logits"][0]    # (max_owned,)
    sv = slot_valid[0]            # (max_owned,) bool
    if sv.any():
        fire_prob = torch.sigmoid(fl[sv])   # P(fire) per valid slot
        return fire_prob.max().item()
    return 0.0


def firep_for_checkpoint(checkpoint, ref_obs, target_decode=False):
    cfg = Config()
    sd, _ = load_checkpoint(checkpoint, cfg)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd)
    device = torch.device("cpu")
    model = model.to(device).eval()

    return [firep_from_obs(model, obs, player, device)
            for obs, player in ref_obs.values()]


def short_label(ckpt_path):
    stem = Path(ckpt_path).stem  # e.g. torch_step_31490048_rev31_...
    parts = stem.split("_")
    # find step number
    for i, p in enumerate(parts):
        if p == "step" and i + 1 < len(parts):
            steps = int(parts[i + 1])
            return f"{steps / 1e6:.0f}M"
    return stem[:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--target-decode", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--seat", type=int, default=SEAT)
    args = ap.parse_args()

    seed = args.seed
    seat = args.seat

    print(f"Capturing reference observations (seed={seed}, seat={seat}) from {Path(args.checkpoints[0]).name}...")
    ref_obs = capture_reference_obs(args.checkpoints[0], args.target_decode, seed=seed, seat=seat)
    print(f"  Captured {len(ref_obs)} steps.")

    rows = []
    for ckpt in args.checkpoints:
        label = short_label(ckpt)
        print(f"  {label}: {Path(ckpt).name}", flush=True)
        fp = firep_for_checkpoint(ckpt, ref_obs, args.target_decode)
        rows.append((label, fp))

    # Print table
    print()
    step_headers = "".join(f"  Step {s}" for s in range(len(ref_obs)))
    print(f"{'Checkpoint':<14}{step_headers}")
    print("-" * (14 + 9 * len(ref_obs)))
    for label, fp in rows:
        vals = "".join(f"  {v:6.3f}" for v in fp)
        note = " ✓" if fp[1] > 0.5 else (" →" if fp[1] > 0.1 else "")
        print(f"{label:<14}{vals}{note}")


if __name__ == "__main__":
    main()
