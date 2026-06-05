"""Train the shadow joint action scorer on producer preference pairs."""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from orbit_wars_rl.config import Config
from orbit_wars_rl.eval import load_checkpoint
from orbit_wars_rl.joint_action_ranker import JointActionRanker
from orbit_wars_rl.model import EntityTransformer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", action="append", required=True)
    ap.add_argument("--init-checkpoint", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--device", default="")
    return ap.parse_args()


def _collate(samples: list[dict], device: torch.device) -> dict:
    keys = [
        "planet_features", "fleet_features", "global_features",
        "planet_mask", "fleet_mask", "fire_mask", "angle_mask",
        "slot_valid", "owned_indices", "pairwise_features",
        "pos_slot", "pos_ship_bin", "pos_target_idx",
        "pos_action_extra",
        "neg_slot", "neg_ship_bin", "neg_target_idx", "neg_action_extra", "weight",
    ]
    return {k: torch.stack([s[k] for s in samples]).to(device) for k in keys}


def main() -> None:
    args = parse_args()
    cfg = Config()
    sd, _ = load_checkpoint(args.init_checkpoint, cfg)
    device = torch.device(args.device or cfg.device)
    backbone = EntityTransformer(cfg.model)
    backbone.load_state_dict(sd)
    model = JointActionRanker(backbone).to(device)

    samples = []
    for path in args.samples:
        with open(path, "rb") as f:
            chunk = pickle.load(f)
        print(f"Loaded {len(chunk)} samples from {path}")
        samples.extend(chunk)
    print(f"Total samples: {len(samples)}")
    if not samples:
        raise SystemExit("No preference samples loaded")

    for p in model.backbone.parameters():
        p.requires_grad = False
    params = list(model.ship_emb.parameters()) + list(model.scorer.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate / 10
    )

    n_val = max(1, int(len(samples) * 0.1))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"Joint ranker training: {len(train_samples)} train / {len(val_samples)} val")
    best_val = float("inf")
    best_state = None
    step = 0

    while step < args.steps:
        np.random.shuffle(train_samples)
        batch_start = 0
        while batch_start < len(train_samples) and step < args.steps:
            batch = _collate(train_samples[batch_start: batch_start + args.batch_size], device)
            pos = model.score_actions(batch, batch["pos_slot"], batch["pos_ship_bin"], batch["pos_target_idx"], batch["pos_action_extra"])
            neg = model.score_actions(batch, batch["neg_slot"], batch["neg_ship_bin"], batch["neg_target_idx"], batch["neg_action_extra"])
            margin = pos - neg
            loss = (F.softplus(-margin) * batch["weight"]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            scheduler.step()
            step += 1
            batch_start += args.batch_size
            if step % 100 == 0:
                acc = (margin > 0).float().mean().item()
                print(f"  step {step:4d} | loss {loss.item():.4f} | pref_acc {acc:.3f} | margin {margin.mean().item():.3f} | lr {optimizer.param_groups[0]['lr']:.2e}")

        with torch.no_grad():
            val_losses = []
            val_acc = []
            for bs in range(0, len(val_samples), args.batch_size):
                batch = _collate(val_samples[bs: bs + args.batch_size], device)
                pos = model.score_actions(batch, batch["pos_slot"], batch["pos_ship_bin"], batch["pos_target_idx"], batch["pos_action_extra"])
                neg = model.score_actions(batch, batch["neg_slot"], batch["neg_ship_bin"], batch["neg_target_idx"], batch["neg_action_extra"])
                margin = pos - neg
                val_losses.append((F.softplus(-margin) * batch["weight"]).mean().item())
                val_acc.append((margin > 0).float().mean().item())
            val_loss = float(np.mean(val_losses))
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best weights (val_loss={best_val:.4f})")

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({
        "joint_ranker": model.state_dict(),
        "backbone_checkpoint": args.init_checkpoint,
    }, args.save)
    print(f"Joint ranker saved -> {args.save}")


if __name__ == "__main__":
    main()
