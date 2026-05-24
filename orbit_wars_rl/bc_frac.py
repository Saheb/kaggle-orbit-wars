"""BC experiment: fraction-of-source ship head (instead of absolute-count head).

Hypothesis (from community discussion): predicting "send X% of source's
sendable ships" generalizes better than "send N ships" because:
  - scale-free (same fraction at any planet size)
  - no single "always-legal smallest bin" trap (bin 0 = 1 ship was the
    universal escape hatch that our policy collapsed to in the 10M Run A/B
    replays — see docs/bugs.md)
  - smaller, denser action space

Setup:
  - 10 fraction bins on (0, 1]: bin i represents fraction (i+1)/10
  - Source max_ships = source_planet.ships - 1 (env leaves 1 for defense,
    matching action_mask.py compute_action_masks)
  - bin = round(fraction * 10) - 1, clamped to [0, 9]
  - Inference (NOT in this script): ships = max(1, round(fraction * max_ships))

Runs end-to-end: collect teacher trajectories → label with fraction bins →
train fraction-head model → report gate. No re-use of the existing
absolute-count teacher_samples.pkl because that lost the raw ship counts.

Usage:
    python bc_frac.py --agent teacher.py --num-games 200 \\
        --steps 5000 --save checkpoints/bc_teacher_frac_v1.pt
"""

from __future__ import annotations
import argparse
import os
import pickle

import torch

from config import Config
from model import EntityTransformer
from features import extract_features
from action_mask import compute_action_masks
from bc import (
    collect_heuristic_trajectories,
    _find_angle_bin,
    _find_target_planet_index,
    pid_to_slot_src_idx,
    train_bc,
)


NUM_FRACTION_BINS = 10
FRACTION_BIN_VALUES = [(i + 1) / NUM_FRACTION_BINS for i in range(NUM_FRACTION_BINS)]


def fraction_to_bin(fraction: float) -> int:
    """Bin index for a fraction in (0, 1]. bin (i+1)/10 means 'send (i+1)*10%'."""
    f = max(0.0, min(1.0, float(fraction)))
    b = int(round(f * NUM_FRACTION_BINS)) - 1
    return max(0, min(NUM_FRACTION_BINS - 1, b))


def trajectory_to_fraction_sample(traj: dict, max_owned: int = 10, max_planets: int = 48) -> dict | None:
    """Like bc.trajectory_to_training_sample but emits a fraction-bin ship_target.

    Fraction = ship_count / max(source_ships - 1, 1) (matches env's "keep 1
    for defense" convention).
    """
    obs = traj["obs"]
    action = traj["action"]
    player = obs["player"]

    features = extract_features(obs, player, num_players=2)
    masks = compute_action_masks(obs, player)
    n_owned = masks["owned_count"]
    if n_owned == 0:
        return None

    planets = obs["planets"]
    owned_indices = masks["owned_indices"].numpy()
    pid_to_slot: dict[int, int] = {}
    for slot in range(n_owned):
        pidx = int(owned_indices[slot])
        if pidx < len(planets):
            pid_to_slot[int(planets[pidx][0])] = slot

    fire_target = torch.zeros(max_owned, dtype=torch.long)
    angle_target = torch.zeros(max_owned, dtype=torch.long)
    ship_target = torch.zeros(max_owned, dtype=torch.long)
    target_target = torch.full((max_owned,), -1, dtype=torch.long)

    initial_planets = obs.get("initial_planets", planets)
    angular_velocity = float(obs.get("angular_velocity", 0.0))
    current_step = int(obs.get("step", 0))

    for move in action:
        if len(move) < 3:
            continue
        from_pid, angle_rad, ship_count = int(move[0]), float(move[1]), int(move[2])
        slot = pid_to_slot.get(from_pid)
        if slot is None:
            continue
        # Look up source planet's current ship count
        src_idx = pid_to_slot_src_idx(planets, from_pid)
        if src_idx is None:
            continue
        src_ships = int(planets[src_idx][5])
        max_sendable = max(1, src_ships - 1)
        fraction = ship_count / max_sendable
        # Clamp upward at 1.0 (rare: teacher emitted more than max_sendable)
        fraction = min(1.0, max(1e-6, fraction))

        fire_target[slot] = 1
        angle_target[slot] = _find_angle_bin(angle_rad)
        ship_target[slot] = fraction_to_bin(fraction)

        src_p = planets[src_idx]
        tgt_idx = _find_target_planet_index(
            (float(src_p[2]), float(src_p[3])), angle_rad, ship_count,
            planets, initial_planets, angular_velocity, current_step,
            max_planets=max_planets,
        )
        if tgt_idx >= 0:
            target_target[slot] = tgt_idx

    pairwise_features = features.get("pairwise_features")
    return {
        "planet_features": features["planet_features"],
        "fleet_features": features["fleet_features"],
        "global_features": features["global_features"],
        "planet_mask": features["planet_mask"],
        "fleet_mask": features["fleet_mask"],
        "fire_mask": masks["fire_mask"][0],
        "angle_mask": masks["angle_mask"][0],
        "slot_valid": masks["slot_valid"][0],
        "owned_indices": masks["owned_indices"],
        "owned_count": masks["owned_count"],
        "fire_target": fire_target,
        "angle_target": angle_target,
        "ship_target": ship_target,
        "target_target": target_target,
        "pairwise_features": pairwise_features,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="teacher.py")
    p.add_argument("--num-games", type=int, default=200)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--save", required=True)
    p.add_argument("--cache-samples", default="",
                   help="Optional path to save/load fraction-labeled samples")
    args = p.parse_args()

    cfg = Config()
    cfg.bc.num_steps = args.steps
    # Tell the model to use a 10-output ship head
    cfg.model.num_ship_bins = NUM_FRACTION_BINS

    # ---- extract or load samples ----
    if args.cache_samples and os.path.exists(args.cache_samples):
        print(f"Loading cached samples from {args.cache_samples}")
        with open(args.cache_samples, "rb") as f:
            samples = pickle.load(f)
    else:
        print(f"Collecting {args.num_games} games from {args.agent}...")
        trajs = collect_heuristic_trajectories(args.agent, num_games=args.num_games,
                                                opponent="random", verbose=False)
        print(f"  {len(trajs)} raw transitions")
        samples = []
        for t in trajs:
            s = trajectory_to_fraction_sample(t)
            if s is not None:
                samples.append(s)
        print(f"  {len(samples)} usable samples (with fraction labels)")
        if args.cache_samples:
            with open(args.cache_samples, "wb") as f:
                pickle.dump(samples, f)
            print(f"  cached → {args.cache_samples}")

    print(f"\nFraction-bin distribution in labels:")
    from collections import Counter
    bins = Counter()
    for s in samples:
        sv = s["slot_valid"]
        ft = s["fire_target"]
        st = s["ship_target"]
        for slot in range(len(sv)):
            if bool(sv[slot]) and int(ft[slot]) == 1:
                bins[int(st[slot])] += 1
    total = sum(bins.values())
    for b in range(NUM_FRACTION_BINS):
        n = bins.get(b, 0)
        frac = FRACTION_BIN_VALUES[b]
        bar = "#" * int(40 * n / max(total, 1))
        print(f"  bin {b} (frac={frac:.1f}): {n:5d}  {bar}")

    # ---- train ----
    device = torch.device(cfg.device)
    model = EntityTransformer(cfg.model)
    print(f"\nModel ship_head output dim: {model.num_ship_bins} "
          f"(fraction bins, was 32 absolute counts)")

    val_metrics = train_bc(model, samples, cfg.bc, device)

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(),
                "config": {"num_ship_bins": NUM_FRACTION_BINS,
                           "fraction_bin_values": FRACTION_BIN_VALUES}},
               args.save)
    print(f"Saved → {args.save}")
    print(f"\nNote: ship_red in the gate is on FRACTION bins (uniform = 1/10), "
          f"not directly comparable to absolute-bin runs (uniform = 1/32). "
          f"Compare target_top1 and target_red to teacher-bc gate result.")


if __name__ == "__main__":
    main()
