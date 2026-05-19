"""Behavioral Cloning trainer for the Entity Transformer architecture.

Collects trajectories from the heuristic agent (main.py) and trains
the entity transformer to imitate via cross-entropy on (fire, angle, ship) actions.
Use this as a quick architecture smoke test (~5K steps), NOT as PPO initialization.

Usage:
    python bc.py --agent ../main.py --num-games 200 --steps 5000
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from config import Config, BCConfig
from model import EntityTransformer, NUM_ANGLE_BINS, NUM_SHIP_BINS, ANGLE_BIN_WIDTH
from features import extract_features
from action_mask import compute_action_masks, _ship_bin_to_count


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def _load_agent_fn(agent_path: str):
    """Load a kaggle agent function from a Python file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("heuristic_agent", agent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "agent"):
        return module.agent
    raise AttributeError(f"No 'agent' function found in {agent_path}")


def collect_heuristic_trajectories(
    agent_path: str,
    num_games: int = 100,
    opponent: str = "random",
    verbose: bool = True,
) -> list[dict]:
    """Collect (obs, action) pairs from a heuristic agent.

    Uses OrbitWarsEnv (direct Python calls, ~196 SPS) instead of kaggle_environments
    env.run() (subprocess JSON serialization, ~0.7 SPS). The agent function is loaded
    directly via importlib.

    Returns a list of dicts with keys: obs (dict), action (list of moves).
    """
    from env import OrbitWarsEnv

    agent_fn = _load_agent_fn(agent_path)
    trajectories = []

    for seed in range(num_games):
        env = OrbitWarsEnv(num_players=2, seed=seed)
        obs = env.reset(seed=seed)
        done = False

        while not done:
            # Call heuristic agent on current observation
            action = agent_fn(obs)

            if action and len(action) > 0:
                trajectories.append({"obs": obs, "action": action})

            obs, reward, done, _ = env.step(action or [])

        if verbose and (seed + 1) % 20 == 0:
            print(f"  Collected {seed + 1}/{num_games} games, "
                  f"{len(trajectories)} transitions so far")

    return trajectories


# ---------------------------------------------------------------------------
# Action target conversion
# ---------------------------------------------------------------------------

def _find_angle_bin(angle_rad: float) -> int:
    return int(angle_rad / ANGLE_BIN_WIDTH) % NUM_ANGLE_BINS


def _find_ship_bin(ships: int, max_ships: int = 10000) -> int:
    """Find the closest log-scale ship bin for a given ship count."""
    best_bin, best_diff = 0, float("inf")
    for b in range(NUM_SHIP_BINS):
        count = _ship_bin_to_count(b, max_ships)
        diff = abs(count - ships)
        if diff < best_diff:
            best_diff, best_bin = diff, b
    return best_bin


def trajectory_to_training_sample(traj: dict, max_owned: int = 10) -> dict | None:
    """Convert a (obs, action) trajectory dict to model-ready tensors.

    Returns None if the observation has no owned planets.
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
    owned_indices = masks["owned_indices"].numpy()  # (max_owned,)

    # Build a map from planet_id -> owned slot index
    pid_to_slot: dict[int, int] = {}
    for slot in range(n_owned):
        pidx = int(owned_indices[slot])
        if pidx < len(planets):
            pid_to_slot[int(planets[pidx][0])] = slot

    # Target tensors: default = no fire
    fire_target = torch.zeros(max_owned, dtype=torch.long)
    angle_target = torch.zeros(max_owned, dtype=torch.long)
    ship_target = torch.zeros(max_owned, dtype=torch.long)

    for move in action:
        if len(move) < 3:
            continue
        from_pid, angle_rad, ship_count = int(move[0]), float(move[1]), int(move[2])
        slot = pid_to_slot.get(from_pid)
        if slot is None:
            continue
        fire_target[slot] = 1
        angle_target[slot] = _find_angle_bin(angle_rad)
        ship_target[slot] = _find_ship_bin(ship_count)

    return {
        "planet_features": features["planet_features"],   # (max_planets, 18)
        "fleet_features": features["fleet_features"],     # (max_fleets, 9)
        "global_features": features["global_features"],   # (10,)
        "planet_mask": features["planet_mask"],           # (max_planets,)
        "fleet_mask": features["fleet_mask"],             # (max_fleets,)
        "fire_mask": masks["fire_mask"][0],               # (max_owned,)
        "angle_mask": masks["angle_mask"][0],             # (max_owned, 72)
        "slot_valid": masks["slot_valid"][0],             # (max_owned,)
        "owned_indices": masks["owned_indices"],          # (max_owned,)
        "owned_count": n_owned,
        "fire_target": fire_target,                       # (max_owned,)
        "angle_target": angle_target,                     # (max_owned,)
        "ship_target": ship_target,                       # (max_owned,)
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _collate(samples: list[dict], device) -> dict:
    """Stack a list of samples into a batched dict."""
    keys_to_stack = [
        "planet_features", "fleet_features", "global_features",
        "planet_mask", "fleet_mask", "fire_mask", "angle_mask",
        "slot_valid", "owned_indices",
        "fire_target", "angle_target", "ship_target",
    ]
    batch = {}
    for k in keys_to_stack:
        batch[k] = torch.stack([s[k] for s in samples]).to(device)
    return batch


def bc_loss(outputs: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    """Cross-entropy BC loss across all owned-planet slots."""
    fire_logits = outputs["fire_logits"]     # (B, max_owned)
    angle_logits = outputs["angle_logits"]   # (B, max_owned, 72)
    ship_logits = outputs["ship_logits"]     # (B, max_owned, 16)

    slot_valid = batch["slot_valid"].float()  # (B, max_owned)
    fire_target = batch["fire_target"]        # (B, max_owned)
    angle_target = batch["angle_target"]      # (B, max_owned)
    ship_target = batch["ship_target"]        # (B, max_owned)

    # Fire loss (binary cross-entropy per slot, masked)
    # Clamp logits to ±30 to avoid MPS float16 overflow from -1e9 mask values
    fire_loss = F.binary_cross_entropy_with_logits(
        fire_logits.clamp(-30, 30), fire_target.float(), reduction="none"
    ) * slot_valid
    fire_loss = fire_loss.sum() / slot_valid.sum().clamp(min=1)

    # Angle loss: only on slots where heuristic actually fired
    fired = (fire_target == 1).float() * slot_valid  # (B, max_owned)
    B, max_owned, _ = angle_logits.shape
    angle_loss = F.cross_entropy(
        angle_logits.view(B * max_owned, -1),
        angle_target.view(B * max_owned),
        reduction="none",
    ).view(B, max_owned)
    angle_loss = (angle_loss * fired).sum() / fired.sum().clamp(min=1)

    # Ship loss: only on slots where heuristic actually fired
    ship_loss = F.cross_entropy(
        ship_logits.view(B * max_owned, -1),
        ship_target.view(B * max_owned),
        reduction="none",
    ).view(B, max_owned)
    ship_loss = (ship_loss * fired).sum() / fired.sum().clamp(min=1)

    total = fire_loss + angle_loss + ship_loss

    metrics = {
        "fire_loss": fire_loss.item(),
        "angle_loss": angle_loss.item(),
        "ship_loss": ship_loss.item(),
        "loss": total.item(),
    }
    return total, metrics


def train_bc(
    model: EntityTransformer,
    samples: list[dict],
    cfg_bc: BCConfig,
    device: torch.device,
    val_frac: float = 0.1,
) -> dict:
    """Train model via BC for cfg_bc.num_steps gradient steps.

    Uses cosine LR decay (lr → lr/10) to prevent divergence when the dataset
    is small relative to num_steps.

    Returns final val metrics.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_bc.learning_rate, eps=1e-5)
    # Cosine decay: lr falls from learning_rate to learning_rate/10 over num_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg_bc.num_steps, eta_min=cfg_bc.learning_rate / 10
    )

    # Train / val split
    n_val = max(1, int(len(samples) * val_frac))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    if not train_samples:
        raise ValueError("No training samples after split")

    steps_per_epoch = max(1, len(train_samples) // cfg_bc.batch_size)
    num_epochs = max(1, cfg_bc.num_steps // steps_per_epoch)
    print(f"BC training: {len(train_samples)} train / {len(val_samples)} val samples")
    print(f"Steps: {cfg_bc.num_steps}, batch: {cfg_bc.batch_size}, "
          f"~{steps_per_epoch} steps/epoch, ~{num_epochs} epochs")

    best_val_loss = float("inf")
    best_state = None
    patience = max(5, num_epochs // 4)  # stop if no improvement for 25% of epochs
    epochs_no_improve = 0
    step = 0
    epoch = 0

    while step < cfg_bc.num_steps:
        model.train()
        np.random.shuffle(train_samples)
        epoch += 1

        batch_start = 0
        while batch_start < len(train_samples) and step < cfg_bc.num_steps:
            batch_samples = train_samples[batch_start: batch_start + cfg_bc.batch_size]
            batch = _collate(batch_samples, device)

            outputs = model(
                batch["planet_features"], batch["fleet_features"], batch["global_features"],
                batch["planet_mask"], batch["fleet_mask"],
                fire_mask=batch["fire_mask"],
                angle_mask=batch["angle_mask"],
                slot_valid=batch["slot_valid"],
                owned_indices=batch["owned_indices"],
            )

            loss, metrics = bc_loss(outputs, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            batch_start += cfg_bc.batch_size
            step += 1

            if step % 100 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  step {step:4d} | loss {metrics['loss']:.4f} | "
                      f"fire {metrics['fire_loss']:.4f} | "
                      f"angle {metrics['angle_loss']:.4f} | "
                      f"ship {metrics['ship_loss']:.4f} | lr {lr_now:.2e}")

        # End-of-epoch validation for early stopping
        model.eval()
        ep_val_loss = 0.0
        ep_val_batches = 0
        with torch.no_grad():
            for bs in range(0, len(val_samples), cfg_bc.batch_size):
                vbatch = _collate(val_samples[bs: bs + cfg_bc.batch_size], device)
                vout = model(
                    vbatch["planet_features"], vbatch["fleet_features"], vbatch["global_features"],
                    vbatch["planet_mask"], vbatch["fleet_mask"],
                    fire_mask=vbatch["fire_mask"], angle_mask=vbatch["angle_mask"],
                    slot_valid=vbatch["slot_valid"], owned_indices=vbatch["owned_indices"],
                )
                _, vm = bc_loss(vout, vbatch)
                ep_val_loss += vm["loss"]
                ep_val_batches += 1
        ep_val_loss /= max(ep_val_batches, 1)

        if ep_val_loss < best_val_loss - 0.01:
            best_val_loss = ep_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and step >= cfg_bc.num_steps // 4:
            print(f"  Early stopping at epoch {epoch} (step {step}): "
                  f"no val improvement for {patience} epochs")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best weights (val_loss={best_val_loss:.4f})")

    # Validation
    model.eval()
    val_metrics_sum = {}
    n_val_batches = 0
    with torch.no_grad():
        for batch_start in range(0, len(val_samples), cfg_bc.batch_size):
            batch = _collate(val_samples[batch_start: batch_start + cfg_bc.batch_size], device)
            outputs = model(
                batch["planet_features"], batch["fleet_features"], batch["global_features"],
                batch["planet_mask"], batch["fleet_mask"],
                fire_mask=batch["fire_mask"],
                angle_mask=batch["angle_mask"],
                slot_valid=batch["slot_valid"],
                owned_indices=batch["owned_indices"],
            )
            _, m = bc_loss(outputs, batch)
            for k, v in m.items():
                val_metrics_sum[k] = val_metrics_sum.get(k, 0.0) + v
            n_val_batches += 1

    val_metrics = {f"val_{k}": v / max(n_val_batches, 1) for k, v in val_metrics_sum.items()}
    print(f"\nBC validation: {val_metrics}")
    return val_metrics


def validate_bc(cfg: Config, agent_path: str, save_path: str = "", verbose: bool = True):
    """Full BC validation pipeline."""
    device = torch.device(cfg.device)

    print("Collecting heuristic trajectories...")
    raw_trajectories = collect_heuristic_trajectories(
        agent_path,
        num_games=cfg.bc.num_trajectories,
        opponent="random",
        verbose=verbose,
    )
    print(f"Collected {len(raw_trajectories)} raw transitions")

    print("Converting to training samples...")
    samples = []
    for traj in raw_trajectories:
        s = trajectory_to_training_sample(traj)
        if s is not None:
            samples.append(s)
    print(f"Usable samples: {len(samples)}")

    if not samples:
        print("ERROR: No usable samples collected. Check that the agent file is correct.")
        return {}

    model = EntityTransformer(cfg.model)
    val_metrics = train_bc(model, samples, cfg.bc, device)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({"model": model.state_dict()}, save_path)
        print(f"BC model saved → {save_path}")

    print("\nBC validation complete!")
    print(f"  Final val loss: {val_metrics.get('val_loss', float('nan')):.4f}")
    return val_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="../main.py",
                        help="Path to heuristic agent file")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="checkpoints/bc_warmstart.pt",
                        help="Where to save the BC-pretrained model")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    cfg.bc.num_trajectories = args.num_games
    cfg.bc.num_steps = args.steps

    validate_bc(cfg, agent_path=args.agent, save_path=args.save)
