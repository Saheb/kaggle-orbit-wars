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


def _find_ship_bin(ships: int, max_ships: int = 10000) -> int:
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


def trajectory_to_training_sample(traj: dict, max_owned: int = MAX_OWNED_PLANETS, max_planets: int = 48) -> dict | None:
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
        fire_target[slot] = 1
        ship_target[slot] = _find_ship_bin(ship_count)

        # Target-index label: which planet did the teacher MEAN by this angle?
        # Uses ETA-iterated predicted position so orbital intercepts are decoded
        # correctly (teacher aims at where target WILL be, not where it is now).
        src_idx = pid_to_slot_src_idx(planets, from_pid)
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
    return batch


_FIRE_POS_WEIGHT = 1.0  # set from --fire-pos-weight in main; >1 counters fire-head class
                        # imbalance (winners fire ~9.5% of owned slots -> BCE collapses to no-fire)


def bc_loss(outputs: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    """Cross-entropy BC loss across all owned-planet slots (target-decode).

    Trains fire, ship, and target heads. Angle head is dead weight in
    target-decode mode and is excluded from the loss.
    """
    # Phase 5 per-target heads: fire/ship are scored per (slot, target), so the BC
    # loss must evaluate them at the TEACHER'S CHOSEN target column (gather), not a
    # slot-only head. The target head selects a planet column for a launch, or the
    # synthetic NO_OP column (idx == width-1) to "do nothing" — so a valid source
    # that the teacher did not launch from is an explicit NO_OP label.
    fire_logits = outputs["fire_logits"]     # (B, MO, TW)    per-(slot, target)
    ship_logits = outputs["ship_logits"]     # (B, MO, TW, bins)
    target_logits = outputs["target_logits"] # (B, MO, TW), NO_OP at index TW-1

    B, MO, TW = target_logits.shape
    NO_OP = TW - 1
    slot_valid_b = batch["slot_valid"].bool()         # (B, MO)
    fire_target = batch["fire_target"]                # (B, MO) in {0,1}
    ship_target = batch["ship_target"]                # (B, MO) bin
    target_target = batch["target_target"]            # (B, MO), -1 = unresolved/none

    fired    = (fire_target == 1) & slot_valid_b
    resolved = fired & (target_target >= 0)           # fired with a known target planet
    # The column each head is trained at: the teacher's planet for resolved launches,
    # else NO_OP for a valid no-launch slot. Fired-but-unresolved slots (teacher fired
    # at unrecoverable space) are ignored — there is no column to gather.
    train_target = resolved | (slot_valid_b & ~fired)
    chosen = torch.where(resolved, target_target.clamp(min=0),
                         torch.full_like(target_target, NO_OP))   # (B, MO)
    tt_f = train_target.float()
    n_tgt = tt_f.sum().clamp(min=1)

    # Target loss: which column (planet or NO_OP) did the teacher choose?
    target_loss_raw = F.cross_entropy(
        target_logits.reshape(B * MO, TW), chosen.reshape(B * MO), reduction="none",
    ).view(B, MO)
    target_loss = (target_loss_raw * tt_f).sum() / n_tgt

    # Fire loss: BCE at the chosen column (pos_weight counters the no-fire imbalance).
    # For NO_OP slots the chosen column's fire logit is the fixed -100, so label 0
    # contributes ~0 — the "don't act" signal is carried by the target head's NO_OP pick.
    fire_at_chosen = torch.gather(fire_logits, -1, chosen.unsqueeze(-1)).squeeze(-1)  # (B, MO)
    fire_loss = F.binary_cross_entropy_with_logits(
        fire_at_chosen.clamp(-30, 30), fired.float(), reduction="none",
        pos_weight=torch.tensor(_FIRE_POS_WEIGHT, device=fire_logits.device),
    ) * tt_f
    fire_loss = fire_loss.sum() / n_tgt

    # Ship loss: CE at the chosen column, only on resolved (actually-launched) slots.
    ship_at_chosen = torch.gather(
        ship_logits, 2,
        chosen.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, ship_logits.shape[-1]),
    ).squeeze(2)                                                                      # (B, MO, bins)
    res_f = resolved.float()
    ship_loss = F.cross_entropy(
        ship_at_chosen.reshape(B * MO, -1), ship_target.reshape(B * MO), reduction="none",
    ).view(B, MO)
    ship_loss = (ship_loss * res_f).sum() / res_f.sum().clamp(min=1)

    total = fire_loss + ship_loss + target_loss

    # Top-k accuracy on target/NO_OP prediction (over trained slots)
    with torch.no_grad():
        topk = target_logits.topk(min(3, TW), dim=-1).indices  # (B, MO, k)
        match_top1 = (topk[..., 0] == chosen).float() * tt_f
        match_top3 = (topk == chosen.unsqueeze(-1)).any(dim=-1).float() * tt_f
        top1_acc = (match_top1.sum() / n_tgt).item()
        top3_acc = (match_top3.sum() / n_tgt).item()

    # Normalized losses (entropy-reduction fraction vs uniform baseline).
    import math as _m
    fire_uniform   = _m.log(2)
    ship_uniform   = _m.log(max(2, ship_logits.shape[-1]))
    target_uniform = _m.log(TW)
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
    }
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
            # 15-global feature flag — load_checkpoint/train_torch restore the global dim from
            # this so a 15-global BC warmstart loads correctly (omitting it silently breaks resume).
            "game_phase_features": cfg.model.game_phase_features,
        },
    }, save_path)
    print(f"BC model saved → {save_path}")


def train_bc(
    model: EntityTransformer,
    samples: list[dict],
    cfg_bc: BCConfig,
    device: torch.device,
    val_frac: float = 0.1,
    trainable_param_patterns: list[str] | None = None,
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
                pairwise_features=batch["pairwise_features"],
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
                      f"fire {metrics['fire_loss']:.4f} (red {metrics['fire_red']:+.2f}) | "
                      f"ship {metrics['ship_loss']:.4f} (red {metrics['ship_red']:+.2f}) | "
                      f"tgt {metrics['target_loss']:.3f} (red {metrics['target_red']:+.2f} top1 {metrics['target_top1']:.2f} top3 {metrics['target_top3']:.2f}) | "
                      f"lr {lr_now:.2e}")

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
                    pairwise_features=vbatch["pairwise_features"],
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
                pairwise_features=batch["pairwise_features"],
            )
            _, m = bc_loss(outputs, batch)
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
        _save_bc_checkpoint(model, cfg, save_path)

    print("\nBC validation complete!")
    print(f"  Final val loss: {val_metrics.get('val_loss', float('nan')):.4f}")
    return val_metrics


def validate_bc_from_samples(cfg: Config, sample_pkls: list[str],
                             save_path: str = "",
                             init_checkpoint: str = "",
                             trainable_param_patterns: list[str] | None = None) -> dict:
    """BC training from one or more pre-extracted sample .pkl files.

    Used by the replay-mining pipeline (replay_bc_v2.py emits these pkls).
    Multiple pkls are concatenated — combine teacher + replay samples this way.
    """
    import pickle
    device = torch.device(cfg.device)
    samples = []
    for path in sample_pkls:
        with open(path, "rb") as f:
            chunk = pickle.load(f)
        print(f"Loaded {len(chunk)} samples from {path}")
        samples.extend(chunk)
    print(f"Total samples: {len(samples)}")
    if not samples:
        print("ERROR: No samples loaded.")
        return {}

    model = EntityTransformer(cfg.model)
    if init_checkpoint:
        from eval import load_checkpoint
        sd, _ = load_checkpoint(init_checkpoint, cfg)
        model.load_state_dict(sd)
        print(f"Loaded init checkpoint: {init_checkpoint}")
    val_metrics = train_bc(model, samples, cfg.bc, device,
                           trainable_param_patterns=trainable_param_patterns)

    if save_path:
        _save_bc_checkpoint(model, cfg, save_path)

    print("\nBC validation complete!")
    print(f"  Final val loss: {val_metrics.get('val_loss', float('nan')):.4f}")
    return val_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="",
                        help="Path to heuristic agent file (collect trajectories on-the-fly)")
    parser.add_argument("--samples", action="append", default=[],
                        help="Path to pre-extracted samples .pkl. Repeatable: "
                             "--samples teacher.pkl --samples replays.pkl to mix.")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="checkpoints/bc_warmstart.pt",
                        help="Where to save the BC-pretrained model")
    parser.add_argument("--init-checkpoint", type=str, default="",
                        help="Optional checkpoint to load before BC training. "
                             "Useful for targeted BC fine-tuning from an existing warmstart.")
    parser.add_argument("--trainable-param", action="append", default=[],
                        help="Optional substring filter for trainable params. Repeatable. "
                             "If set, only parameters whose names contain one of these "
                             "substrings are updated.")
    parser.add_argument("--lr", type=float, default=0.0,
                        help="Learning rate override (default: use BCConfig.learning_rate=3e-4). "
                             "For fine-tuning from a strong checkpoint, use 1e-4.")
    parser.add_argument("--game-phase-features", action="store_true",
                        help="Train a 15-global model (11->15: phase one-hot + comet-cycle). The "
                             "--samples pkl MUST be built with the same flag (build_snowball_bc.py "
                             "--game-phase-features) so the stored global tensors are 15-dim.")
    parser.add_argument("--entity-dim", type=int, default=0,
                        help="Override model entity_dim (default 96). Use 128 for the capacity-probe arm.")
    parser.add_argument("--device", type=str, default="",
                        help="Override device (e.g. 'mps' for Apple GPU, 'cuda', 'cpu'). Default: cfg.device.")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Override BC batch size (default 128). Larger (512) = fewer steps + better CPU/MPS amortization.")
    parser.add_argument("--fire-pos-weight", type=float, default=1.0,
                        help="pos_weight on the fire-head BCE to counter class imbalance "
                             "(winners fire ~9.5%% of slots). >1 = the BC fires more. Try ~5-9.")
    args = parser.parse_args()

    _FIRE_POS_WEIGHT = args.fire_pos_weight  # module-level rebind; bc_loss reads this global

    cfg = Config()
    cfg.seed = args.seed
    cfg.bc.num_trajectories = args.num_games
    cfg.bc.num_steps = args.steps
    if args.lr > 0:
        cfg.bc.learning_rate = args.lr
    if args.device:
        cfg.device = args.device
    if args.batch_size > 0:
        cfg.bc.batch_size = args.batch_size
    if args.game_phase_features:
        from features import set_game_phase_features
        cfg.model.game_phase_features = True
        cfg.model.global_feature_dim = 15
        set_game_phase_features(True)   # so any on-the-fly extraction emits 15-global too
    if args.entity_dim > 0:
        cfg.model.entity_dim = args.entity_dim

    if args.samples:
        validate_bc_from_samples(cfg, args.samples, save_path=args.save,
                                 init_checkpoint=args.init_checkpoint,
                                 trainable_param_patterns=args.trainable_param or None)
    else:
        if not args.agent:
            raise SystemExit("--agent or --samples required")
        validate_bc(cfg, agent_path=args.agent, save_path=args.save)
