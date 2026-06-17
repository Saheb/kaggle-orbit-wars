"""Behavioral Cloning trainer for the Entity Transformer architecture.

Collects trajectories from the heuristic agent (main.py) and trains
the entity transformer to imitate via cross-entropy on (fire, angle, ship) actions.
Use this as a quick architecture smoke test (~5K steps), NOT as PPO initialization.

Usage:
    python bc.py --agent ../main.py --num-games 200 --steps 5000
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from config import Config, BCConfig
from model import EntityTransformer, NUM_ANGLE_BINS, NUM_SHIP_BINS, ANGLE_BIN_WIDTH, SHIP_COUNTS
from features import extract_features, MAX_OWNED_PLANETS
from action_mask import compute_action_masks


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def _load_agent_fn(agent_path: str):
    """Load a kaggle agent function from a Python file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("heuristic_agent", agent_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # required for @dataclass __module__ resolution
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


FRACTION_BIN_VALUES = [(i + 1) / 10 for i in range(10)]


def _find_ship_bin(ships: int, max_ships: int = 10000, mode: str = "absolute") -> int:
    if mode == "fraction":
        max_ships = max(1, int(max_ships))
        frac = max(0.0, min(float(ships) / max_ships, 1.0))
        best_bin, best_diff = 0, float("inf")
        for b, value in enumerate(FRACTION_BIN_VALUES):
            diff = abs(value - frac)
            if diff < best_diff:
                best_diff, best_bin = diff, b
        return best_bin
    if mode != "absolute":
        raise ValueError(f"unknown ship bin mode: {mode}")
    best_bin, best_diff = 0, float("inf")
    for b in range(NUM_SHIP_BINS):
        count = SHIP_COUNTS[b]
        diff = abs(count - ships)
        if diff < best_diff:
            best_diff, best_bin = diff, b
    return best_bin


_MAX_SHIP_SPEED = 6.0
_BC_CENTER = 50.0
_ROTATION_LIMIT = 50.0


def pid_to_slot_src_idx(planets, from_pid: int) -> int | None:
    """Find array index of planet with id == from_pid; None if not present."""
    for i, p in enumerate(planets):
        if int(p[0]) == from_pid:
            return i
    return None


def _bc_fleet_speed(ships: int) -> float:
    if ships <= 0:
        return 1.0
    s = 1.0 + (_MAX_SHIP_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5
    return min(s, _MAX_SHIP_SPEED)


def _find_target_planet_index(src_xy, emitted_angle, ship_count, planets, initial_planets,
                              angular_velocity, current_step, max_planets=48):
    """Recover which planet the teacher meant by its (angle, ships) launch.

    Replicates teacher's logic: for each planet, ETA-iterate to find the predicted
    arrival position (handles orbital intercept), compute angle from src to that
    predicted position, pick the planet whose predicted-angle is circularly
    closest to the teacher's emitted angle.

    Returns int index into planets[:max_planets], or -1 if no match.
    """
    if not planets:
        return -1
    sx, sy = src_xy
    speed = _bc_fleet_speed(ship_count)

    # Build orbital table from initial_planets
    init_by_id = {int(p[0]): p for p in initial_planets}

    best_pid_idx = -1
    best_circ_err = float("inf")
    for j, tgt in enumerate(planets[:max_planets]):
        pid = int(tgt[0])
        tx, ty, tr = float(tgt[2]), float(tgt[3]), float(tgt[4])

        # Iterate ETA→position fixed-point (4 iters; converges fast)
        ax, ay = tx, ty
        ip = init_by_id.get(pid)
        if ip is not None:
            irx = float(ip[2]) - _BC_CENTER
            iry = float(ip[3]) - _BC_CENTER
            init_angle = math.atan2(iry, irx)
            orbital_r = math.hypot(irx, iry)
            is_orbiting = (orbital_r + tr) < _ROTATION_LIMIT
        else:
            is_orbiting = False
            init_angle = 0.0
            orbital_r = 0.0

        for _ in range(4):
            dist = math.hypot(ax - sx, ay - sy)
            eta = max(1, int(math.ceil(dist / speed)))
            if is_orbiting:
                ang = init_angle + angular_velocity * (current_step + eta)
                nax = _BC_CENTER + orbital_r * math.cos(ang)
                nay = _BC_CENTER + orbital_r * math.sin(ang)
            else:
                nax, nay = tx, ty
            if abs(nax - ax) < 0.5 and abs(nay - ay) < 0.5:
                ax, ay = nax, nay
                break
            ax, ay = nax, nay

        predicted_angle = math.atan2(ay - sy, ax - sx)
        # Circular angular error (in radians)
        d = abs(predicted_angle - emitted_angle)
        d = min(d, 2 * math.pi - d)
        if d < best_circ_err:
            best_circ_err = d
            best_pid_idx = j

    # Reject if even the best match is way off (teacher fired at empty space or
    # something we can't decode) — > 15° circular error is suspicious
    if best_circ_err > math.radians(15.0):
        return -1
    return best_pid_idx


def trajectory_to_training_sample(
    traj: dict,
    max_owned: int = MAX_OWNED_PLANETS,
    max_planets: int = 48,
    ship_bin_mode: str = "absolute",
) -> dict | None:
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

    # Target tensors: default = no fire / ignore-index for target prediction
    fire_target   = torch.zeros(max_owned, dtype=torch.long)
    ship_target   = torch.zeros(max_owned, dtype=torch.long)
    target_target = torch.full((max_owned,), -1, dtype=torch.long)  # -1 = ignore

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
        src_idx = pid_to_slot_src_idx(planets, from_pid)
        fire_target[slot] = 1
        src_planet_ships = int(planets[src_idx][5]) if src_idx is not None else ship_count
        ship_target[slot] = _find_ship_bin(ship_count, max_ships=src_planet_ships, mode=ship_bin_mode)

        # Target-index label: which planet did the teacher MEAN by this angle?
        # Uses ETA-iterated predicted position so orbital intercepts are decoded
        # correctly (teacher aims at where target WILL be, not where it is now).
        if src_idx is not None:
            src_p = planets[src_idx]
            tgt_idx = _find_target_planet_index(
                (float(src_p[2]), float(src_p[3])), angle_rad, ship_count,
                planets, initial_planets, angular_velocity, current_step,
                max_planets=max_planets,
            )
            if tgt_idx >= 0:
                target_target[slot] = tgt_idx

    return {
        "planet_features": features["planet_features"],   # (max_planets, 20)
        "fleet_features": features["fleet_features"],     # (max_fleets, 13)
        "global_features": features["global_features"],   # (11,)
        "planet_mask": features["planet_mask"],           # (max_planets,)
        "fleet_mask": features["fleet_mask"],             # (max_fleets,)
        "fire_mask": masks["fire_mask"][0],               # (max_owned,)
        "angle_mask": masks["angle_mask"][0],             # (max_owned, 72)
        "slot_valid": masks["slot_valid"][0],             # (max_owned,)
        "owned_indices": masks["owned_indices"],          # (max_owned,)
        "owned_count": n_owned,
        "fire_target": fire_target,                       # (max_owned,)
        "ship_target": ship_target,                       # (max_owned,)
        "target_target": target_target,                   # (max_owned,) -1 = ignore
        "pairwise_features": features["pairwise_features"],  # (max_owned, max_planets, F_pair)
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
        "fire_target", "ship_target",
        "target_target",
        "pairwise_features",
    ]
    batch = {}
    for k in keys_to_stack:
        batch[k] = torch.stack([s[k] for s in samples]).to(device)
    if any("threat_target" in s for s in samples):
        threat_targets = []
        threat_masks = []
        for s in samples:
            if "threat_target" in s:
                threat_targets.append(s["threat_target"])
                threat_masks.append(s["threat_mask"])
            else:
                threat_targets.append(torch.zeros_like(s["fire_target"], dtype=torch.float32))
                threat_masks.append(torch.zeros_like(s["slot_valid"], dtype=torch.bool))
        batch["threat_target"] = torch.stack(threat_targets).to(device)
        batch["threat_mask"] = torch.stack(threat_masks).to(device)
    return batch


def _is_training_sample(record: dict) -> bool:
    return isinstance(record, dict) and "planet_features" in record


def _records_to_training_samples(records: list[dict], ship_bin_mode: str = "absolute") -> list[dict]:
    """Accept tensor samples or compact obs/action records."""
    samples = []
    for record in records:
        if _is_training_sample(record):
            record_mode = str(record.get("ship_bin_mode", "absolute"))
            if ship_bin_mode != "absolute" and record_mode != ship_bin_mode:
                raise ValueError(
                    f"Cannot use pre-materialized {record_mode!r} ship labels "
                    f"for requested ship_bin_mode={ship_bin_mode!r}; use compact "
                    "frame shards or regenerate samples with matching metadata."
                )
            samples.append(record)
            continue
        sample = trajectory_to_training_sample(record, ship_bin_mode=ship_bin_mode)
        if sample is not None:
            samples.append(sample)
    return samples


def _collate_records(records: list[dict], device, ship_bin_mode: str = "absolute") -> dict:
    samples = _records_to_training_samples(records, ship_bin_mode=ship_bin_mode)
    if not samples:
        raise ValueError("Batch contained no usable training samples")
    return _collate(samples, device)


def bc_loss(outputs: dict, batch: dict, fire_pos_weight: float = 1.0,
            threat_loss_weight: float = 0.0,
            threat_pos_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Cross-entropy BC loss across all owned-planet slots (target-decode).

    Trains fire, ship, and target heads. Angle head is dead weight in
    target-decode mode and is excluded from the loss.
    """
    fire_logits = outputs["fire_logits"]     # (B, max_owned)
    ship_logits = outputs["ship_logits"]     # (B, max_owned, num_ship_bins)

    slot_valid = batch["slot_valid"].float()  # (B, max_owned)
    fire_target = batch["fire_target"]        # (B, max_owned)
    ship_target = batch["ship_target"]        # (B, max_owned)

    # Fire loss (binary cross-entropy per slot, masked)
    # Clamp logits to ±30 to avoid MPS float16 overflow from -1e9 mask values
    fire_loss_raw = F.binary_cross_entropy_with_logits(
        fire_logits.clamp(-30, 30), fire_target.float(), reduction="none"
    )
    fire_weight = slot_valid * (1.0 + (float(fire_pos_weight) - 1.0) * fire_target.float())
    fire_loss = (fire_loss_raw * fire_weight).sum() / fire_weight.sum().clamp(min=1)

    # Ship loss: only on slots where heuristic actually fired
    B, max_owned = fire_logits.shape
    fired = (fire_target == 1).float() * slot_valid  # (B, max_owned)
    ship_loss = F.cross_entropy(
        ship_logits.view(B * max_owned, -1),
        ship_target.view(B * max_owned),
        reduction="none",
    ).view(B, max_owned)
    ship_loss = (ship_loss * fired).sum() / fired.sum().clamp(min=1)

    # Target-index loss: which planet did the teacher choose? Cross-entropy over
    # max_planets logits, only on slots that fired AND have a valid (non-(-1)) label.
    target_logits = outputs["target_logits"]            # (B, MO, max_planets)
    target_target = batch["target_target"]              # (B, MO) with -1 = ignore
    valid_tgt = (target_target >= 0).float() * fired    # (B, MO)
    # Replace -1 with 0 to avoid OOB in cross_entropy; mask result instead
    safe_tgt = target_target.clamp(min=0)
    B, MO, MP = target_logits.shape
    target_loss_raw = F.cross_entropy(
        target_logits.view(B * MO, MP),
        safe_tgt.view(B * MO),
        reduction="none",
    ).view(B, MO)
    n_valid_tgt = valid_tgt.sum().clamp(min=1)
    target_loss = (target_loss_raw * valid_tgt).sum() / n_valid_tgt

    threat_loss = fire_logits.new_tensor(0.0)
    if threat_loss_weight > 0 and "threat_logits" in outputs and "threat_target" in batch:
        threat_mask = batch["threat_mask"].float() * slot_valid
        threat_target = batch["threat_target"].float()
        threat_raw = F.binary_cross_entropy_with_logits(
            outputs["threat_logits"].clamp(-30, 30), threat_target, reduction="none"
        )
        threat_weight = threat_mask * (1.0 + (float(threat_pos_weight) - 1.0) * threat_target)
        threat_loss = (threat_raw * threat_weight).sum() / threat_weight.sum().clamp(min=1)

    total = fire_loss + ship_loss + target_loss + float(threat_loss_weight) * threat_loss

    # Top-k accuracy on target prediction (only on valid slots)
    with torch.no_grad():
        topk = target_logits.topk(min(3, MP), dim=-1).indices  # (B, MO, k)
        match_top1 = (topk[..., 0] == safe_tgt).float() * valid_tgt
        match_top3 = (topk == safe_tgt.unsqueeze(-1)).any(dim=-1).float() * valid_tgt
        top1_acc = (match_top1.sum() / n_valid_tgt).item()
        top3_acc = (match_top3.sum() / n_valid_tgt).item()

    # Normalized losses (entropy-reduction fraction vs uniform baseline).
    import math as _m
    fire_uniform   = _m.log(2)
    ship_uniform   = _m.log(max(2, ship_logits.shape[-1]))
    target_uniform = _m.log(MP)
    fire_red   = 1.0 - fire_loss.item()   / fire_uniform
    ship_red   = 1.0 - ship_loss.item()   / ship_uniform
    target_red = 1.0 - target_loss.item() / target_uniform

    metrics = {
        "fire_loss":   fire_loss.item(),
        "ship_loss":   ship_loss.item(),
        "target_loss": target_loss.item(),
        "loss":        total.item(),
        "fire_red":    fire_red,
        "ship_red":    ship_red,
        "target_red":  target_red,
        "target_top1": top1_acc,
        "target_top3": top3_acc,
        "fire_pos_weight": float(fire_pos_weight),
    }
    if threat_loss_weight > 0 and "threat_logits" in outputs and "threat_target" in batch:
        with torch.no_grad():
            threat_mask = batch["threat_mask"].float() * slot_valid
            threat_pred = (torch.sigmoid(outputs["threat_logits"]) > 0.5).float()
            threat_target = batch["threat_target"].float()
            n_threat = threat_mask.sum().clamp(min=1)
            threat_acc = ((threat_pred == threat_target).float() * threat_mask).sum() / n_threat
            threat_pos_rate = (threat_target * threat_mask).sum() / n_threat
        metrics["threat_loss"] = threat_loss.item()
        metrics["threat_acc"] = threat_acc.item()
        metrics["threat_pos_rate"] = threat_pos_rate.item()
        metrics["threat_loss_weight"] = float(threat_loss_weight)
        metrics["threat_pos_weight"] = float(threat_pos_weight)
    return total, metrics


def _save_bc_checkpoint(model: EntityTransformer, cfg, save_path: str):
    """Save BC checkpoint with config metadata so load_checkpoint auto-detects it."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {
            "action_decode":  "target",
            "ship_bin_mode":  cfg.model.ship_bin_mode,
            "num_ship_bins":  cfg.model.num_ship_bins,
            "min_ship_bin":   cfg.model.min_ship_bin,
            "allow_reinforce": bool(getattr(cfg.model, "allow_reinforce", False)),
            "use_threat_head": bool(getattr(cfg.model, "use_threat_head", False)),
            "trainer": "supervised_bc",
            "supervised_only": True,
        },
    }, save_path)
    print(f"BC model saved → {save_path}")


def _expand_sample_paths(sample_args: list[str]) -> list[str]:
    """Expand direct pkls, shard dirs, globs, txt lists, and manifest JSON files."""
    paths: list[str] = []
    for arg in sample_args:
        if any(ch in arg for ch in "*?["):
            paths.extend(sorted(glob.glob(arg)))
            continue
        if os.path.isdir(arg):
            paths.extend(sorted(glob.glob(os.path.join(arg, "*.pkl"))))
            continue
        if arg.endswith(".txt"):
            with open(arg) as f:
                paths.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
            continue
        if arg.endswith(".json"):
            with open(arg) as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "sample_paths" in data:
                    paths.extend(str(p) for p in data["sample_paths"])
                    continue
                if "shards" in data:
                    for shard in data["shards"]:
                        paths.append(str(shard["path"] if isinstance(shard, dict) else shard))
                    continue
            if isinstance(data, list):
                paths.extend(str(p) for p in data)
                continue
        paths.append(arg)
    # Preserve first occurrence order after expansion.
    out: list[str] = []
    seen = set()
    for path in paths:
        if path and path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _checkpoint_looks_like_rl_training(ckpt) -> bool:
    """Return True for PPO learner checkpoints, False for plain model/BC saves."""
    return (
        isinstance(ckpt, dict)
        and "model" in ckpt
        and (
            "optimizer" in ckpt
            or "total_steps" in ckpt
            or "update_count" in ckpt
        )
    )


def _assert_supervised_init_checkpoint(init_checkpoint: str, allow_rl_init: bool) -> None:
    if not init_checkpoint or allow_rl_init:
        return
    ckpt = torch.load(init_checkpoint, map_location="cpu")
    if _checkpoint_looks_like_rl_training(ckpt):
        raise SystemExit(
            f"{init_checkpoint} looks like a PPO/RL training checkpoint. "
            "This replay-supervised trainer refuses RL init by default; pass "
            "--allow-rl-init only for an explicit diagnostic."
        )


def _checkpoint_state_dict(path: str) -> dict:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def _load_compatible_init(model: EntityTransformer, init_checkpoint: str) -> dict:
    src = _checkpoint_state_dict(init_checkpoint)
    dst = model.state_dict()
    loaded = []
    skipped_shape = []
    skipped_missing = []
    for name, tensor in src.items():
        if name not in dst:
            skipped_missing.append(name)
            continue
        if tuple(dst[name].shape) != tuple(tensor.shape):
            skipped_shape.append((name, tuple(tensor.shape), tuple(dst[name].shape)))
            continue
        dst[name] = tensor
        loaded.append(name)
    model.load_state_dict(dst)
    return {
        "loaded": loaded,
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
    }


def _base_metric_name(metric_name: str) -> str:
    return metric_name[4:] if metric_name.startswith("val_") else metric_name


def _metric_lower_is_better(metric_name: str) -> bool:
    return _base_metric_name(metric_name).endswith("loss")


def _metric_improved(metric_name: str, new_value: float, best_value: float) -> bool:
    if _metric_lower_is_better(metric_name):
        return new_value < best_value - 0.01
    return new_value > best_value + 0.001


def train_bc(
    model: EntityTransformer,
    samples: list[dict],
    cfg_bc: BCConfig,
    device: torch.device,
    val_frac: float = 0.1,
    trainable_param_patterns: list[str] | None = None,
    fire_pos_weight: float = 1.0,
    threat_loss_weight: float = 0.0,
    threat_pos_weight: float = 1.0,
    select_metric: str = "val_loss",
) -> dict:
    """Train model via BC for cfg_bc.num_steps gradient steps.

    Uses cosine LR decay (lr → lr/10) to prevent divergence when the dataset
    is small relative to num_steps.

    Returns final val metrics.
    """
    model = model.to(device)
    if trainable_param_patterns:
        for p in model.parameters():
            p.requires_grad = False
        trainable = []
        for name, p in model.named_parameters():
            if any(pattern in name for pattern in trainable_param_patterns):
                p.requires_grad = True
                trainable.append((name, p))
        if not trainable:
            raise ValueError(f"No trainable params matched patterns: {trainable_param_patterns}")
        optimizer_params = [p for _, p in trainable]
        print("BC trainable params:")
        for name, _ in trainable:
            print(f"  - {name}")
    else:
        optimizer_params = list(model.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=cfg_bc.learning_rate, eps=1e-5)
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

    metric_key = _base_metric_name(select_metric)
    best_metric_value = float("inf") if _metric_lower_is_better(select_metric) else -float("inf")
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
                pairwise_features=batch["pairwise_features"],
            )

            loss, metrics = bc_loss(
                outputs, batch,
                fire_pos_weight=fire_pos_weight,
                threat_loss_weight=threat_loss_weight,
                threat_pos_weight=threat_pos_weight,
            )
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
                      f"fire {metrics['fire_loss']:.4f} (red {metrics['fire_red']:+.2f}) | "
                      f"ship {metrics['ship_loss']:.4f} (red {metrics['ship_red']:+.2f}) | "
                      f"tgt {metrics['target_loss']:.3f} (red {metrics['target_red']:+.2f} top1 {metrics['target_top1']:.2f} top3 {metrics['target_top3']:.2f}) | "
                      f"lr {lr_now:.2e}")

        # End-of-epoch validation for early stopping
        model.eval()
        ep_val_metrics = {}
        ep_val_batches = 0
        with torch.no_grad():
            for bs in range(0, len(val_samples), cfg_bc.batch_size):
                vbatch = _collate(val_samples[bs: bs + cfg_bc.batch_size], device)
                vout = model(
                    vbatch["planet_features"], vbatch["fleet_features"], vbatch["global_features"],
                    vbatch["planet_mask"], vbatch["fleet_mask"],
                    fire_mask=vbatch["fire_mask"], angle_mask=vbatch["angle_mask"],
                    slot_valid=vbatch["slot_valid"], owned_indices=vbatch["owned_indices"],
                    pairwise_features=vbatch["pairwise_features"],
                )
                _, vm = bc_loss(
                    vout, vbatch,
                    fire_pos_weight=fire_pos_weight,
                    threat_loss_weight=threat_loss_weight,
                    threat_pos_weight=threat_pos_weight,
                )
                for k, v in vm.items():
                    ep_val_metrics[k] = ep_val_metrics.get(k, 0.0) + v
                ep_val_batches += 1
        ep_val_metrics = {
            k: v / max(ep_val_batches, 1)
            for k, v in ep_val_metrics.items()
        }
        if metric_key not in ep_val_metrics:
            raise ValueError(
                f"--select-metric {select_metric!r} not available. "
                f"Available validation metrics: {', '.join('val_' + k for k in sorted(ep_val_metrics))}"
            )
        ep_metric_value = ep_val_metrics[metric_key]

        if _metric_improved(select_metric, ep_metric_value, best_metric_value):
            best_metric_value = ep_metric_value
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
        print(f"  Restored best weights ({select_metric}={best_metric_value:.4f})")

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
                pairwise_features=batch["pairwise_features"],
            )
            _, m = bc_loss(
                outputs, batch,
                fire_pos_weight=fire_pos_weight,
                threat_loss_weight=threat_loss_weight,
                threat_pos_weight=threat_pos_weight,
            )
            for k, v in m.items():
                val_metrics_sum[k] = val_metrics_sum.get(k, 0.0) + v
            n_val_batches += 1

    val_metrics = {f"val_{k}": v / max(n_val_batches, 1) for k, v in val_metrics_sum.items()}
    print(f"\nBC validation: {val_metrics}")
    # Phase-A gate: target_red >= 0.40 OR target_top1 >= 0.30
    #               — predicting WHICH planet is the semantic action.
    ship_red = val_metrics.get("val_ship_red", 0.0)
    fire_red = val_metrics.get("val_fire_red", 0.0)
    tgt_red  = val_metrics.get("val_target_red", 0.0)
    tgt_top1 = val_metrics.get("val_target_top1", 0.0)
    tgt_top3 = val_metrics.get("val_target_top3", 0.0)
    tgt_pass = (tgt_red >= 0.40) or (tgt_top1 >= 0.30)
    gate = "PASS" if tgt_pass else "FAIL"
    print(f"\nPhase-A target-head gate: target_red={tgt_red:+.2f}  "
          f"top1={tgt_top1:.2f}  top3={tgt_top3:.2f}  → {gate}")
    print(f"  side metrics: ship_red={ship_red:+.2f}  fire_red={fire_red:+.2f}")
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
        s = trajectory_to_training_sample(traj, ship_bin_mode=cfg.model.ship_bin_mode)
        if s is not None:
            samples.append(s)
    print(f"Usable samples: {len(samples)}")

    if not samples:
        print("ERROR: No usable samples collected. Check that the agent file is correct.")
        return {}

    model = EntityTransformer(cfg.model)
    val_metrics = train_bc(model, samples, cfg.bc, device)

    if save_path:
        _save_bc_checkpoint(model, cfg, save_path)

    print("\nBC validation complete!")
    print(f"  Final val loss: {val_metrics.get('val_loss', float('nan')):.4f}")
    return val_metrics


def validate_bc_from_samples(cfg: Config, sample_pkls: list[str],
                             save_path: str = "",
                             init_checkpoint: str = "",
                             allow_rl_init: bool = False,
                             partial_init_compatible: bool = False,
                             trainable_param_patterns: list[str] | None = None,
                             fire_pos_weight: float = 1.0,
                             threat_loss_weight: float = 0.0,
                             threat_pos_weight: float = 1.0,
                             select_metric: str = "val_loss") -> dict:
    """BC training from one or more pre-extracted sample .pkl files.

    Used by the replay-mining pipeline (replay_bc_v2.py emits these pkls).
    Multiple pkls are concatenated — combine teacher + replay samples this way.
    """
    import pickle
    device = torch.device(cfg.device)
    samples = []
    expanded_paths = _expand_sample_paths(sample_pkls)
    for path in expanded_paths:
        with open(path, "rb") as f:
            chunk = pickle.load(f)
        converted = _records_to_training_samples(chunk, ship_bin_mode=cfg.model.ship_bin_mode)
        print(f"Loaded {len(chunk)} records / {len(converted)} samples from {path}")
        samples.extend(converted)
    print(f"Total samples: {len(samples)}")
    if not samples:
        print("ERROR: No samples loaded.")
        return {}
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(samples)

    if init_checkpoint and partial_init_compatible:
        _assert_supervised_init_checkpoint(init_checkpoint, allow_rl_init)
        model = EntityTransformer(cfg.model)
        report = _load_compatible_init(model, init_checkpoint)
        print(
            f"Partially loaded compatible init checkpoint: {init_checkpoint} "
            f"({len(report['loaded'])} tensors loaded, "
            f"{len(report['skipped_shape'])} shape-skipped, "
            f"{len(report['skipped_missing'])} missing-skipped)"
        )
        for name, src_shape, dst_shape in report["skipped_shape"][:8]:
            print(f"  shape-skip {name}: {src_shape} -> {dst_shape}")
    elif init_checkpoint:
        _assert_supervised_init_checkpoint(init_checkpoint, allow_rl_init)
        from eval import load_checkpoint
        force_allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
        force_threat_head = bool(getattr(cfg.model, "use_threat_head", False))
        requested_ship_bin_mode = str(getattr(cfg.model, "ship_bin_mode", "absolute"))
        sd, _ = load_checkpoint(init_checkpoint, cfg)
        if str(getattr(cfg.model, "ship_bin_mode", "absolute")) != requested_ship_bin_mode:
            raise SystemExit(
                f"--init-checkpoint uses ship_bin_mode={cfg.model.ship_bin_mode!r}, "
                f"but this run requested {requested_ship_bin_mode!r}. Train from scratch "
                "or initialize from a checkpoint with the same ship label space."
            )
        if force_allow_reinforce:
            cfg.model.allow_reinforce = True
        if force_threat_head:
            cfg.model.use_threat_head = True
        model = EntityTransformer(cfg.model)
        model.load_state_dict(sd)
        print(f"Loaded init checkpoint: {init_checkpoint}")
    else:
        model = EntityTransformer(cfg.model)
    val_metrics = train_bc(model, samples, cfg.bc, device,
                           trainable_param_patterns=trainable_param_patterns,
                           fire_pos_weight=fire_pos_weight,
                           threat_loss_weight=threat_loss_weight,
                           threat_pos_weight=threat_pos_weight,
                           select_metric=select_metric)

    if save_path:
        _save_bc_checkpoint(model, cfg, save_path)

    print("\nBC validation complete!")
    print(f"  Final val loss: {val_metrics.get('val_loss', float('nan')):.4f}")
    return val_metrics


def _validate_on_samples(
    model: EntityTransformer,
    val_samples: list[dict],
    cfg_bc: BCConfig,
    device: torch.device,
    fire_pos_weight: float,
    threat_loss_weight: float,
    threat_pos_weight: float,
    ship_bin_mode: str = "absolute",
) -> dict:
    model.eval()
    val_metrics_sum = {}
    n_val_batches = 0
    with torch.no_grad():
        for bs in range(0, len(val_samples), cfg_bc.batch_size):
            vbatch = _collate_records(
                val_samples[bs: bs + cfg_bc.batch_size],
                device,
                ship_bin_mode=ship_bin_mode,
            )
            vout = model(
                vbatch["planet_features"], vbatch["fleet_features"], vbatch["global_features"],
                vbatch["planet_mask"], vbatch["fleet_mask"],
                fire_mask=vbatch["fire_mask"], angle_mask=vbatch["angle_mask"],
                slot_valid=vbatch["slot_valid"], owned_indices=vbatch["owned_indices"],
                pairwise_features=vbatch["pairwise_features"],
            )
            _, vm = bc_loss(
                vout, vbatch,
                fire_pos_weight=fire_pos_weight,
                threat_loss_weight=threat_loss_weight,
                threat_pos_weight=threat_pos_weight,
            )
            for k, v in vm.items():
                val_metrics_sum[k] = val_metrics_sum.get(k, 0.0) + v
            n_val_batches += 1
    return {k: v / max(n_val_batches, 1) for k, v in val_metrics_sum.items()}


def _sample_streaming_validation_records(
    paths: list[str],
    val_frac: float,
    max_val_samples: int,
    rng: np.random.Generator,
    pickle_module=None,
) -> tuple[list[dict], dict[str, set[int]], int]:
    """Sample validation records inside each shard without concatenating all shards."""
    if pickle_module is None:
        import pickle as pickle_module

    val_indices_by_path: dict[str, set[int]] = {}
    val_samples = []
    val_path_count = 0
    for path in paths:
        if len(val_samples) >= max_val_samples:
            break
        with open(path, "rb") as f:
            chunk = pickle_module.load(f)
        if not chunk:
            continue
        want = int(round(len(chunk) * val_frac))
        if val_frac > 0.0:
            want = max(1, want)
        want = min(want, len(chunk), max_val_samples - len(val_samples))
        if want <= 0:
            continue
        idx = rng.choice(len(chunk), size=want, replace=False)
        val_idx = {int(i) for i in idx}
        val_indices_by_path[path] = val_idx
        val_samples.extend(chunk[i] for i in val_idx)
        val_path_count += 1
    rng.shuffle(val_samples)
    return val_samples, val_indices_by_path, val_path_count


def validate_bc_from_sample_shards(cfg: Config, sample_pkls: list[str],
                                   save_path: str = "",
                                   init_checkpoint: str = "",
                                   allow_rl_init: bool = False,
                                   partial_init_compatible: bool = False,
                                   trainable_param_patterns: list[str] | None = None,
                                   fire_pos_weight: float = 1.0,
                                   threat_loss_weight: float = 0.0,
                                   threat_pos_weight: float = 1.0,
                                   select_metric: str = "val_loss",
                                   val_frac: float = 0.1,
                                   max_val_samples: int = 8192,
                                   eval_every: int = 1000) -> dict:
    """BC training that streams pickle shards instead of concatenating all samples."""
    import pickle
    device = torch.device(cfg.device)
    paths = _expand_sample_paths(sample_pkls)
    if not paths:
        print("ERROR: No sample shards found.")
        return {}

    rng = np.random.default_rng(cfg.seed)
    paths = list(paths)
    rng.shuffle(paths)
    val_samples, val_indices_by_path, val_path_count = _sample_streaming_validation_records(
        paths, val_frac, max_val_samples, rng, pickle
    )
    if not val_samples:
        print("ERROR: No validation samples loaded.")
        return {}

    if init_checkpoint and partial_init_compatible:
        _assert_supervised_init_checkpoint(init_checkpoint, allow_rl_init)
        model = EntityTransformer(cfg.model)
        report = _load_compatible_init(model, init_checkpoint)
        print(
            f"Partially loaded compatible init checkpoint: {init_checkpoint} "
            f"({len(report['loaded'])} tensors loaded, "
            f"{len(report['skipped_shape'])} shape-skipped, "
            f"{len(report['skipped_missing'])} missing-skipped)"
        )
        for name, src_shape, dst_shape in report["skipped_shape"][:8]:
            print(f"  shape-skip {name}: {src_shape} -> {dst_shape}")
    elif init_checkpoint:
        _assert_supervised_init_checkpoint(init_checkpoint, allow_rl_init)
        from eval import load_checkpoint
        force_allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
        force_threat_head = bool(getattr(cfg.model, "use_threat_head", False))
        requested_ship_bin_mode = str(getattr(cfg.model, "ship_bin_mode", "absolute"))
        sd, _ = load_checkpoint(init_checkpoint, cfg)
        if str(getattr(cfg.model, "ship_bin_mode", "absolute")) != requested_ship_bin_mode:
            raise SystemExit(
                f"--init-checkpoint uses ship_bin_mode={cfg.model.ship_bin_mode!r}, "
                f"but this run requested {requested_ship_bin_mode!r}. Train from scratch "
                "or initialize from a checkpoint with the same ship label space."
            )
        if force_allow_reinforce:
            cfg.model.allow_reinforce = True
        if force_threat_head:
            cfg.model.use_threat_head = True
        model = EntityTransformer(cfg.model)
        model.load_state_dict(sd)
        print(f"Loaded init checkpoint: {init_checkpoint}")
    else:
        model = EntityTransformer(cfg.model)
    model = model.to(device)

    if trainable_param_patterns:
        for p in model.parameters():
            p.requires_grad = False
        trainable = []
        for name, p in model.named_parameters():
            if any(pattern in name for pattern in trainable_param_patterns):
                p.requires_grad = True
                trainable.append((name, p))
        if not trainable:
            raise ValueError(f"No trainable params matched patterns: {trainable_param_patterns}")
        optimizer_params = [p for _, p in trainable]
        print("BC trainable params:")
        for name, _ in trainable:
            print(f"  - {name}")
    else:
        optimizer_params = list(model.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=cfg.bc.learning_rate, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.bc.num_steps, eta_min=cfg.bc.learning_rate / 10
    )

    metric_key = _base_metric_name(select_metric)
    best_metric_value = float("inf") if _metric_lower_is_better(select_metric) else -float("inf")
    best_state = None
    step = 0
    epoch = 0
    eval_every = max(1, eval_every)
    train_paths = paths
    print(f"BC streaming training: {len(train_paths)} train shards / {val_path_count} val-sampled shards")
    print(f"Validation samples: {len(val_samples)}; steps: {cfg.bc.num_steps}; batch: {cfg.bc.batch_size}")

    while step < cfg.bc.num_steps:
        epoch += 1
        rng.shuffle(train_paths)
        for path in train_paths:
            with open(path, "rb") as f:
                chunk = pickle.load(f)
            if not chunk:
                continue
            val_idx = val_indices_by_path.get(path, set())
            train_idx = [i for i in range(len(chunk)) if i not in val_idx]
            if not train_idx:
                continue
            rng.shuffle(train_idx)
            for batch_start in range(0, len(train_idx), cfg.bc.batch_size):
                if step >= cfg.bc.num_steps:
                    break
                batch_indices = train_idx[batch_start: batch_start + cfg.bc.batch_size]
                batch_samples = [chunk[i] for i in batch_indices]
                try:
                    batch = _collate_records(batch_samples, device, ship_bin_mode=cfg.model.ship_bin_mode)
                except ValueError:
                    continue
                model.train()
                outputs = model(
                    batch["planet_features"], batch["fleet_features"], batch["global_features"],
                    batch["planet_mask"], batch["fleet_mask"],
                    fire_mask=batch["fire_mask"],
                    angle_mask=batch["angle_mask"],
                    slot_valid=batch["slot_valid"],
                    owned_indices=batch["owned_indices"],
                    pairwise_features=batch["pairwise_features"],
                )
                loss, metrics = bc_loss(
                    outputs, batch,
                    fire_pos_weight=fire_pos_weight,
                    threat_loss_weight=threat_loss_weight,
                    threat_pos_weight=threat_pos_weight,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                step += 1

                if step % 100 == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    print(f"  step {step:5d} | loss {metrics['loss']:.4f} | "
                          f"tgt top1 {metrics['target_top1']:.2f} top3 {metrics['target_top3']:.2f} | "
                          f"lr {lr_now:.2e}")
                if step % eval_every == 0 or step >= cfg.bc.num_steps:
                    ep_val_metrics = _validate_on_samples(
                        model, val_samples, cfg.bc, device,
                        fire_pos_weight=fire_pos_weight,
                        threat_loss_weight=threat_loss_weight,
                        threat_pos_weight=threat_pos_weight,
                        ship_bin_mode=cfg.model.ship_bin_mode,
                    )
                    if metric_key not in ep_val_metrics:
                        raise ValueError(
                            f"--select-metric {select_metric!r} not available. "
                            f"Available validation metrics: {', '.join('val_' + k for k in sorted(ep_val_metrics))}"
                        )
                    ep_metric_value = ep_val_metrics[metric_key]
                    improved = _metric_improved(select_metric, ep_metric_value, best_metric_value)
                    if improved:
                        best_metric_value = ep_metric_value
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        if save_path:
                            _save_bc_checkpoint(model, cfg, save_path)
                    print(f"  eval step {step:5d} | "
                          f"{select_metric}={ep_metric_value:.4f} | "
                          f"target_top1={ep_val_metrics.get('target_top1', 0.0):.2f} "
                          f"target_top3={ep_val_metrics.get('target_top3', 0.0):.2f}"
                          f"{' *' if improved else ''}")
            if step >= cfg.bc.num_steps:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best weights ({select_metric}={best_metric_value:.4f})")
    final_metrics_raw = _validate_on_samples(
        model, val_samples, cfg.bc, device,
        fire_pos_weight=fire_pos_weight,
        threat_loss_weight=threat_loss_weight,
        threat_pos_weight=threat_pos_weight,
        ship_bin_mode=cfg.model.ship_bin_mode,
    )
    val_metrics = {f"val_{k}": v for k, v in final_metrics_raw.items()}
    print(f"\nBC streaming validation: {val_metrics}")

    if save_path:
        _save_bc_checkpoint(model, cfg, save_path)
    return val_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="",
                        help="Path to heuristic agent file (collect trajectories on-the-fly)")
    parser.add_argument("--samples", action="append", default=[],
                        help="Path to pre-extracted samples .pkl, shard dir, glob, "
                             "txt list, or manifest JSON. Repeatable.")
    parser.add_argument("--stream-shards", action="store_true",
                        help="Stream sample shards from disk instead of loading all samples.")
    parser.add_argument("--max-val-samples", type=int, default=8192,
                        help="Validation sample cap for --stream-shards.")
    parser.add_argument("--eval-every", type=int, default=1000,
                        help="Validation interval in gradient steps for --stream-shards.")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="checkpoints/bc_warmstart.pt",
                        help="Where to save the BC-pretrained model")
    parser.add_argument("--init-checkpoint", type=str, default="",
                        help="Optional checkpoint to load before BC training. "
                             "By default this must be a supervised/model-only checkpoint, "
                             "not a PPO learner checkpoint.")
    parser.add_argument("--allow-rl-init", action="store_true",
                        help="Allow --init-checkpoint to load a PPO/RL learner checkpoint. "
                             "Use only for explicit diagnostics, not the standalone "
                             "replay-supervised track.")
    parser.add_argument("--partial-init-compatible", action="store_true",
                        help="Load only same-name/same-shape tensors from --init-checkpoint. "
                             "Useful for transplanting an absolute ship-head checkpoint "
                             "into a fraction ship-head model; mismatched heads are skipped.")
    parser.add_argument("--trainable-param", action="append", default=[],
                        help="Optional substring filter for trainable params. Repeatable. "
                             "If set, only parameters whose names contain one of these "
                             "substrings are updated.")
    parser.add_argument("--lr", type=float, default=0.0,
                        help="Learning rate override (default: use BCConfig.learning_rate=3e-4). "
                             "For fine-tuning from a strong checkpoint, use 1e-4.")
    parser.add_argument("--fire-pos-weight", type=float, default=1.0,
                        help="Positive fire-label weight in BCE. Use >1 when no-fire slots "
                             "dominate replay-supervised data.")
    parser.add_argument("--ship-bin-mode", choices=["absolute", "fraction"], default="absolute",
                        help="Ship label/decode space. 'absolute' uses legacy SHIP_COUNTS; "
                             "'fraction' uses 10 buckets for 10%..100% of source ships.")
    parser.add_argument("--min-ship-bin", type=int, default=0,
                        help="Mask ship bins below this index in the model forward pass.")
    parser.add_argument("--allow-reinforce", action="store_true",
                        help="Save BC checkpoint with own-planet target decode enabled.")
    parser.add_argument("--threat-loss-weight", type=float, default=0.0,
                        help="Auxiliary BCE weight for per-owned-planet threat labels.")
    parser.add_argument("--threat-pos-weight", type=float, default=1.0,
                        help="Positive-label weight for threat BCE.")
    parser.add_argument("--select-metric", type=str, default="val_loss",
                        help="Validation metric used to restore best weights. "
                             "Loss metrics are minimized; all other metrics are maximized. "
                             "Examples: val_loss, val_target_top3, val_target_red.")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    cfg.bc.num_trajectories = args.num_games
    cfg.bc.num_steps = args.steps
    if args.lr > 0:
        cfg.bc.learning_rate = args.lr
    cfg.model.allow_reinforce = bool(args.allow_reinforce)
    cfg.model.use_threat_head = args.threat_loss_weight > 0
    cfg.model.ship_bin_mode = args.ship_bin_mode
    cfg.model.num_ship_bins = len(FRACTION_BIN_VALUES) if args.ship_bin_mode == "fraction" else NUM_SHIP_BINS
    cfg.model.min_ship_bin = int(args.min_ship_bin)

    if args.samples:
        if args.stream_shards:
            validate_bc_from_sample_shards(cfg, args.samples, save_path=args.save,
                                           init_checkpoint=args.init_checkpoint,
                                           allow_rl_init=args.allow_rl_init,
                                           partial_init_compatible=args.partial_init_compatible,
                                           trainable_param_patterns=args.trainable_param or None,
                                           fire_pos_weight=args.fire_pos_weight,
                                           threat_loss_weight=args.threat_loss_weight,
                                           threat_pos_weight=args.threat_pos_weight,
                                           select_metric=args.select_metric,
                                           max_val_samples=args.max_val_samples,
                                           eval_every=args.eval_every)
        else:
            validate_bc_from_samples(cfg, args.samples, save_path=args.save,
                                     init_checkpoint=args.init_checkpoint,
                                     allow_rl_init=args.allow_rl_init,
                                     partial_init_compatible=args.partial_init_compatible,
                                     trainable_param_patterns=args.trainable_param or None,
                                     fire_pos_weight=args.fire_pos_weight,
                                     threat_loss_weight=args.threat_loss_weight,
                                     threat_pos_weight=args.threat_pos_weight,
                                     select_metric=args.select_metric)
    else:
        if not args.agent:
            raise SystemExit("--agent or --samples required")
        validate_bc(cfg, agent_path=args.agent, save_path=args.save)
