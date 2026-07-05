"""Train pairwise producer-action preferences on top of an existing checkpoint."""

from __future__ import annotations

import argparse
import math
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
from orbit_wars_rl.model import EntityTransformer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", action="append", required=True)
    ap.add_argument("--init-checkpoint", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--trainable-param", action="append", default=[])
    ap.add_argument("--device", default="")
    return ap.parse_args()


def _collate(samples: list[dict], device: torch.device) -> dict:
    keys = [
        "planet_features", "fleet_features", "global_features",
        "planet_mask", "fleet_mask", "fire_mask", "angle_mask",
        "slot_valid", "owned_indices", "pairwise_features",
        "pos_slot", "pos_ship_bin", "pos_target_idx",
        "neg_slot", "neg_ship_bin", "neg_target_idx", "weight",
    ]
    return {k: torch.stack([s[k] for s in samples]).to(device) for k in keys}


def _action_score(outputs: dict, slot: torch.Tensor, ship_bin: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    B = slot.shape[0]
    b = torch.arange(B, device=slot.device)
    fire = outputs["fire_logits"][b, slot]
    ship = outputs["ship_logits"][b, slot, ship_bin]
    target = outputs["target_logits"][b, slot, target_idx]
    return fire + ship + target


def _pref_metrics(pos_score: torch.Tensor, neg_score: torch.Tensor) -> dict[str, float]:
    margin = pos_score - neg_score
    acc = (margin > 0).float().mean().item()
    return {
        "pref_acc": acc,
        "mean_margin": margin.mean().item(),
    }


def _save_checkpoint(model: EntityTransformer, cfg: Config, save_path: str):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {
            "action_decode": "target",
            "ship_bin_mode": cfg.model.ship_bin_mode,
            "num_ship_bins": cfg.model.num_ship_bins,
            "min_ship_bin": cfg.model.min_ship_bin,
        },
    }, save_path)
    print(f"Preference model saved -> {save_path}")


def main() -> None:
    args = parse_args()
    cfg = Config()
    sd, _ = load_checkpoint(args.init_checkpoint, cfg)
    device = torch.device(args.device or cfg.device)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd)
    model = model.to(device)

    samples = []
    for path in args.samples:
        with open(path, "rb") as f:
            chunk = pickle.load(f)
        print(f"Loaded {len(chunk)} samples from {path}")
        samples.extend(chunk)
    print(f"Total samples: {len(samples)}")
    if not samples:
        raise SystemExit("No preference samples loaded")

    if args.trainable_param:
        for p in model.parameters():
            p.requires_grad = False
        trainable = []
        for name, p in model.named_parameters():
            if any(pattern in name for pattern in args.trainable_param):
                p.requires_grad = True
                trainable.append((name, p))
        if not trainable:
            raise ValueError(f"No trainable params matched {args.trainable_param}")
        params = [p for _, p in trainable]
        print("Preference trainable params:")
        for name, _ in trainable:
            print(f"  - {name}")
    else:
        params = list(model.parameters())

    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate / 10
    )

    n_val = max(1, int(len(samples) * 0.1))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    if not train_samples:
        raise ValueError("No training samples after split")

    print(f"Preference training: {len(train_samples)} train / {len(val_samples)} val")
    best_val = float("inf")
    best_state = None
    step = 0

    while step < args.steps:
        np.random.shuffle(train_samples)
        batch_start = 0
        while batch_start < len(train_samples) and step < args.steps:
            batch = _collate(train_samples[batch_start: batch_start + args.batch_size], device)
            outputs = model(
                batch["planet_features"], batch["fleet_features"], batch["global_features"],
                batch["planet_mask"], batch["fleet_mask"],
                fire_mask=batch["fire_mask"], angle_mask=batch["angle_mask"],
                slot_valid=batch["slot_valid"], owned_indices=batch["owned_indices"],
                pairwise_features=batch["pairwise_features"],
            )
            pos_score = _action_score(outputs, batch["pos_slot"], batch["pos_ship_bin"], batch["pos_target_idx"])
            neg_score = _action_score(outputs, batch["neg_slot"], batch["neg_ship_bin"], batch["neg_target_idx"])
            margin = pos_score - neg_score
            loss = (F.softplus(-margin) * batch["weight"]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            step += 1
            batch_start += args.batch_size
            if step % 100 == 0:
                m = _pref_metrics(pos_score.detach(), neg_score.detach())
                print(f"  step {step:4d} | loss {loss.item():.4f} | pref_acc {m['pref_acc']:.3f} | margin {m['mean_margin']:.3f} | lr {optimizer.param_groups[0]['lr']:.2e}")

        model.eval()
        with torch.no_grad():
            val_losses = []
            val_acc = []
            for bs in range(0, len(val_samples), args.batch_size):
                batch = _collate(val_samples[bs: bs + args.batch_size], device)
                outputs = model(
                    batch["planet_features"], batch["fleet_features"], batch["global_features"],
                    batch["planet_mask"], batch["fleet_mask"],
                    fire_mask=batch["fire_mask"], angle_mask=batch["angle_mask"],
                    slot_valid=batch["slot_valid"], owned_indices=batch["owned_indices"],
                    pairwise_features=batch["pairwise_features"],
                )
                pos_score = _action_score(outputs, batch["pos_slot"], batch["pos_ship_bin"], batch["pos_target_idx"])
                neg_score = _action_score(outputs, batch["neg_slot"], batch["neg_ship_bin"], batch["neg_target_idx"])
                margin = pos_score - neg_score
                val_losses.append((F.softplus(-margin) * batch["weight"]).mean().item())
                val_acc.append((margin > 0).float().mean().item())
            val_loss = float(np.mean(val_losses))
            val_pref_acc = float(np.mean(val_acc))
        model.train()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best weights (val_loss={best_val:.4f})")

    model.eval()
    with torch.no_grad():
        val_losses = []
        val_acc = []
        val_margin = []
        for bs in range(0, len(val_samples), args.batch_size):
            batch = _collate(val_samples[bs: bs + args.batch_size], device)
            outputs = model(
                batch["planet_features"], batch["fleet_features"], batch["global_features"],
                batch["planet_mask"], batch["fleet_mask"],
                fire_mask=batch["fire_mask"], angle_mask=batch["angle_mask"],
                slot_valid=batch["slot_valid"], owned_indices=batch["owned_indices"],
                pairwise_features=batch["pairwise_features"],
            )
            pos_score = _action_score(outputs, batch["pos_slot"], batch["pos_ship_bin"], batch["pos_target_idx"])
            neg_score = _action_score(outputs, batch["neg_slot"], batch["neg_ship_bin"], batch["neg_target_idx"])
            margin = pos_score - neg_score
            val_losses.append((F.softplus(-margin) * batch["weight"]).mean().item())
            val_acc.append((margin > 0).float().mean().item())
            val_margin.append(margin.mean().item())
    print({
        "val_loss": float(np.mean(val_losses)),
        "val_pref_acc": float(np.mean(val_acc)),
        "val_mean_margin": float(np.mean(val_margin)),
    })
    _save_checkpoint(model, cfg, args.save)


if __name__ == "__main__":
    main()
