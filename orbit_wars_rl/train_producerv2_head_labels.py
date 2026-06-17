"""Fine-tune fire/target heads on Producer-v2 head labels.

This is an offline probe before any PPO run. It answers: can a narrow supervised
objective move overlap with Producer-v2 labels, and which heads move first?
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbit_wars_rl.config import Config
from orbit_wars_rl.eval import load_checkpoint
from orbit_wars_rl.features import set_game_phase_features
from orbit_wars_rl.model import EntityTransformer


def _load_model(checkpoint: str, device: torch.device):
    cfg = Config()
    state_dict, _ = load_checkpoint(checkpoint, cfg)
    cfg.model.game_phase_features = bool(cfg.model.game_phase_features or cfg.model.global_feature_dim >= 15)
    set_game_phase_features(cfg.model.game_phase_features)
    model = EntityTransformer(cfg.model).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"target_head.weight", "target_head.bias"}
    bad_missing = [k for k in missing if k not in allowed_missing]
    bad_unexpected = [k for k in unexpected if not k.startswith("value_pp_")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={bad_missing} unexpected={bad_unexpected}")
    return model, cfg


def _save_checkpoint(model, cfg, path: str, source_checkpoint: str, label_source: str) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {
            "action_decode": "target",
            "ship_bin_mode": cfg.model.ship_bin_mode,
            "num_ship_bins": cfg.model.num_ship_bins,
            "min_ship_bin": cfg.model.min_ship_bin,
            "game_phase_features": cfg.model.game_phase_features,
            "allow_reinforce": bool(getattr(cfg.model, "allow_reinforce", False)),
            "reinforce_gate_min_planets": int(getattr(cfg.model, "reinforce_gate_min_planets", 0)),
            "reinforce_forward_only": bool(getattr(cfg.model, "reinforce_forward_only", False)),
            "reverse_edge_cooldown": int(getattr(cfg.model, "reverse_edge_cooldown", 0)),
            "reinforce_garrison_floor": float(getattr(cfg.model, "reinforce_garrison_floor", 0.0)),
            "sufficient_commit_factor": float(getattr(cfg.model, "sufficient_commit_factor", 0.0)),
            "producer_v2_head_labels": {
                "source_checkpoint": source_checkpoint,
                "label_source": label_source,
            },
        },
    }, out)


def _collate(samples: list[dict], device: torch.device) -> dict:
    keys = [
        "planet_features", "fleet_features", "global_features", "planet_mask", "fleet_mask",
        "fire_mask", "slot_valid", "owned_indices", "pairwise_features",
        "target_legal_mask",
        "candidate_kind", "candidate_target_idx", "candidate_ship_count",
        "selected_kind", "selected_target_idx", "selected_ship_count",
    ]
    out = {}
    for key in keys:
        if key in samples[0]:
            out[key] = torch.stack([s[key] for s in samples]).to(device)
    return out


def _forward(model, batch: dict) -> dict:
    return model(
        batch["planet_features"],
        batch["fleet_features"],
        batch["global_features"],
        batch["planet_mask"],
        batch["fleet_mask"],
        fire_mask=batch["fire_mask"],
        slot_valid=batch["slot_valid"],
        owned_indices=batch["owned_indices"],
        pairwise_features=batch.get("pairwise_features"),
    )


def _loss(outputs: dict, batch: dict, label_source: str, fire_pos_weight: float,
          fire_coef: float, target_coef: float, ship_coef: float) -> tuple[torch.Tensor, dict]:
    kind = batch[f"{label_source}_kind"]
    target_idx = batch[f"{label_source}_target_idx"]
    ship_count = batch[f"{label_source}_ship_count"]
    slot_valid = batch["slot_valid"].float()
    positive = (kind > 0).float() * slot_valid
    fire_target = (kind > 0).float()

    fire_loss = F.binary_cross_entropy_with_logits(
        outputs["fire_logits"].clamp(-30, 30),
        fire_target,
        reduction="none",
        pos_weight=torch.tensor(float(fire_pos_weight), device=outputs["fire_logits"].device),
    )
    fire_loss = (fire_loss * slot_valid).sum() / slot_valid.sum().clamp(min=1)

    target_logits = outputs["target_logits"]
    if "target_legal_mask" in batch:
        target_mask = batch["target_legal_mask"].bool()
        target_logits = target_logits.masked_fill(~target_mask, -1e9)
    B, MO, MP = target_logits.shape
    safe_target = target_idx.clamp(min=0)
    target_raw = F.cross_entropy(
        target_logits.reshape(B * MO, MP),
        safe_target.reshape(B * MO),
        reduction="none",
    ).view(B, MO)
    target_loss = (target_raw * positive).sum() / positive.sum().clamp(min=1)

    ship_loss = outputs["fire_logits"].new_zeros(())
    if ship_coef > 0.0:
        # Optional coarse adequacy loss: target the first absolute bin >= Producer-v2's ship count.
        from orbit_wars_rl.model import SHIP_COUNTS
        bins = torch.full_like(ship_count, len(SHIP_COUNTS) - 1)
        for i, c in enumerate(SHIP_COUNTS):
            bins = torch.where(ship_count <= int(c), torch.minimum(bins, torch.full_like(bins, i)), bins)
        ship_raw = F.cross_entropy(
            outputs["ship_logits"].reshape(B * MO, -1),
            bins.reshape(B * MO).clamp(min=0, max=outputs["ship_logits"].shape[-1] - 1),
            reduction="none",
        ).view(B, MO)
        ship_loss = (ship_raw * positive).sum() / positive.sum().clamp(min=1)

    loss = fire_coef * fire_loss + target_coef * target_loss + ship_coef * ship_loss
    with torch.no_grad():
        fire_ready = ((torch.sigmoid(outputs["fire_logits"]) >= 0.5).float() * positive).sum() / positive.sum().clamp(min=1)
        top = target_logits.topk(min(3, MP), dim=-1).indices
        tgt_top1 = ((top[..., 0] == safe_target).float() * positive).sum() / positive.sum().clamp(min=1)
        tgt_top3 = ((top == safe_target.unsqueeze(-1)).any(dim=-1).float() * positive).sum() / positive.sum().clamp(min=1)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "fire_loss": float(fire_loss.detach().cpu()),
        "target_loss": float(target_loss.detach().cpu()),
        "ship_loss": float(ship_loss.detach().cpu()),
        "positives": float(positive.sum().detach().cpu()),
        "fire_ready": float(fire_ready.detach().cpu()),
        "target_top1": float(tgt_top1.detach().cpu()),
        "target_top3": float(tgt_top3.detach().cpu()),
    }


def _audit(model, samples: list[dict], device: torch.device, label_source: str, batch_size: int) -> dict:
    model.eval()
    totals = {k: 0.0 for k in ("positive", "attack", "save", "fire", "target1", "target3", "ship1", "ship3", "joint1", "joint3")}
    from orbit_wars_rl.action_mask import _def_ship_adequacy_rank
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            batch = _collate(batch_samples, device)
            out = _forward(model, batch)
            target_logits = out["target_logits"]
            if "target_legal_mask" in batch:
                target_logits = target_logits.masked_fill(~batch["target_legal_mask"].bool(), -1e9)
            kind = batch[f"{label_source}_kind"]
            target_idx = batch[f"{label_source}_target_idx"]
            ship_count = batch[f"{label_source}_ship_count"]
            pos = kind > 0
            if not bool(pos.any()):
                continue
            fire = torch.sigmoid(out["fire_logits"]) >= 0.5
            top = target_logits.topk(min(3, target_logits.shape[-1]), dim=-1).indices
            safe_target = target_idx.clamp(min=0)
            target1 = top[..., 0] == safe_target
            target3 = (top == safe_target.unsqueeze(-1)).any(dim=-1)
            ship1 = torch.zeros_like(pos)
            ship3 = torch.zeros_like(pos)
            for b, sample in enumerate(batch_samples):
                for slot in range(int(sample["owned_count"])):
                    if int(kind[b, slot]) <= 0:
                        continue
                    scores = out["ship_logits"][b, slot].detach().cpu().tolist()
                    rank = _def_ship_adequacy_rank(scores, int(ship_count[b, slot].item()), 420, "absolute")
                    if rank is not None and rank <= 1:
                        ship1[b, slot] = True
                    if rank is not None and rank <= 3:
                        ship3[b, slot] = True
            p = pos.float()
            totals["positive"] += float(p.sum().cpu())
            totals["attack"] += float(((kind == 1).float()).sum().cpu())
            totals["save"] += float(((kind == 2).float()).sum().cpu())
            totals["fire"] += float((fire.float() * p).sum().cpu())
            totals["target1"] += float((target1.float() * p).sum().cpu())
            totals["target3"] += float((target3.float() * p).sum().cpu())
            totals["ship1"] += float((ship1.float() * p).sum().cpu())
            totals["ship3"] += float((ship3.float() * p).sum().cpu())
            totals["joint1"] += float(((fire & target1 & ship1).float() * p).sum().cpu())
            totals["joint3"] += float(((fire & target3 & ship3).float() * p).sum().cpu())
    n = max(totals["positive"], 1.0)
    return {
        "n": int(totals["positive"]),
        "attack": int(totals["attack"]),
        "save": int(totals["save"]),
        "fire_ready": totals["fire"] / n,
        "target_top1": totals["target1"] / n,
        "target_top3": totals["target3"] / n,
        "ship_top1": totals["ship1"] / n,
        "ship_top3": totals["ship3"] / n,
        "joint_top1": totals["joint1"] / n,
        "joint_top3": totals["joint3"] / n,
    }


def _set_trainable(model, mode: str) -> None:
    for _name, param in model.named_parameters():
        param.requires_grad_(mode == "all")
    if mode == "heads":
        prefixes = ("fire_head.", "target_scorer.", "target_head.")
        for name, param in model.named_parameters():
            if name.startswith(prefixes):
                param.requires_grad_(True)
    elif mode == "heads_ship":
        prefixes = ("fire_head.", "target_scorer.", "target_head.", "ship_head.")
        for name, param in model.named_parameters():
            if name.startswith(prefixes):
                param.requires_grad_(True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--label-source", choices=["candidate", "selected"], default="selected")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--fire-pos-weight", type=float, default=8.0)
    ap.add_argument("--fire-coef", type=float, default=1.0)
    ap.add_argument("--target-coef", type=float, default=1.0)
    ap.add_argument("--ship-coef", type=float, default=0.0)
    ap.add_argument("--trainable", choices=["heads", "heads_ship", "all"], default="heads")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--summary-out", default="gpu_run_artifacts/head_audit/producerv2_head_ft_summary.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device("cpu")
    with Path(args.samples).open("rb") as f:
        samples = pickle.load(f)
    model, cfg = _load_model(args.checkpoint, device)
    baseline = _audit(model, samples, device, args.label_source, args.batch_size)

    _set_trainable(model, args.trainable)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(args.lr), eps=1e-5)
    logs = []
    model.train()
    for step in range(1, int(args.steps) + 1):
        batch_samples = random.sample(samples, min(int(args.batch_size), len(samples)))
        batch = _collate(batch_samples, device)
        out = _forward(model, batch)
        loss, metrics = _loss(
            out, batch, args.label_source, args.fire_pos_weight,
            args.fire_coef, args.target_coef, args.ship_coef,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % max(1, int(args.steps) // 10) == 0:
            row = {"step": step, **metrics}
            logs.append(row)
            print(json.dumps(row))

    after = _audit(model, samples, device, args.label_source, args.batch_size)
    if args.save:
        _save_checkpoint(model, cfg, args.save, args.checkpoint, args.label_source)
    payload = {
        "checkpoint": args.checkpoint,
        "samples": args.samples,
        "label_source": args.label_source,
        "steps": int(args.steps),
        "trainable": args.trainable,
        "lr": float(args.lr),
        "fire_pos_weight": float(args.fire_pos_weight),
        "fire_coef": float(args.fire_coef),
        "target_coef": float(args.target_coef),
        "ship_coef": float(args.ship_coef),
        "baseline": baseline,
        "after": after,
        "logs": logs,
        "save": args.save,
    }
    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"baseline": baseline, "after": after}, indent=2))
    print(f"summary saved -> {out}")
    if args.save:
        print(f"checkpoint saved -> {args.save}")


if __name__ == "__main__":
    main()
