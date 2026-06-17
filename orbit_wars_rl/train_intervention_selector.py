"""Train a lightweight selector from paired defense intervention records."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.build_defense_selector import FEATURE_NAMES, train_selector  # noqa: E402


def _record_label(record: dict, label: str) -> int:
    if label == "helped":
        return int(record.get("helped", 0))
    if label == "hold_advantage":
        return int(record.get("hold_advantage", int(record.get("hold_delta", 0) > 0)))
    if label == "nonhurt_hold_advantage":
        return int(record.get("hold_delta", 0) > 0 and not record.get("hurt", 0))
    raise ValueError(f"Unknown label: {label}")


def _record_target(record: dict, target: str, hurt_penalty: float) -> float:
    if target == "hold_delta":
        return float(record.get("hold_delta", 0))
    if target == "hold_delta_minus_hurt":
        return float(record.get("hold_delta", 0)) - float(hurt_penalty) * float(record.get("hurt", 0))
    raise ValueError(f"Unknown regression target: {target}")


def _train_regressor(records: list[dict], steps: int, lr: float, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(records), generator=gen).tolist()
    shuffled = [records[i] for i in order]
    n_val = max(1, int(0.2 * len(shuffled)))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    if not train:
        raise ValueError("No training records")

    x_train = torch.tensor([r["features"] for r in train], dtype=torch.float32)
    y_train = torch.tensor([r["target"] for r in train], dtype=torch.float32)
    x_val = torch.tensor([r["features"] for r in val], dtype=torch.float32)
    y_val = torch.tensor([r["target"] for r in val], dtype=torch.float32)

    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp(min=1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    w = torch.zeros(x_train.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        pred = x_train @ w + b
        loss = F.mse_loss(pred, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        val_pred = x_val @ w + b
        mse = F.mse_loss(val_pred, y_val).item()
        base = F.mse_loss(torch.full_like(y_val, y_train.mean()), y_val).item()
        centered_pred = val_pred - val_pred.mean()
        centered_target = y_val - y_val.mean()
        denom = centered_pred.norm() * centered_target.norm()
        corr = 0.0 if denom.item() <= 1e-12 else float((centered_pred * centered_target).sum().item() / denom.item())
        binary = (y_val > 0).float()
        auc = 0.5
        if binary.min().item() != binary.max().item():
            from orbit_wars_rl.build_defense_selector import _auc
            auc = _auc(val_pred, binary)

    return {
        "weights": w.detach(),
        "bias": b.detach(),
        "mean": mean,
        "std": std,
        "metrics": {
            "val_mse": mse,
            "val_base_mse": base,
            "val_corr": corr,
            "val_positive_auc": auc,
            "train_records": len(train),
            "val_records": len(val),
            "train_target_mean": y_train.mean().item(),
            "val_target_mean": y_val.mean().item(),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", action="append", required=True)
    ap.add_argument("--objective", choices=["classification", "regression"], default="classification")
    ap.add_argument("--label", choices=["helped", "hold_advantage", "nonhurt_hold_advantage"], default="helped")
    ap.add_argument("--regression-target", choices=["hold_delta", "hold_delta_minus_hurt"], default="hold_delta_minus_hurt")
    ap.add_argument("--hurt-penalty", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selector-out", required=True)
    ap.add_argument("--summary-out", default="")
    args = ap.parse_args()

    records = []
    raw_count = helped = hurt = hold_advantage = 0
    for path in args.records:
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        for r in loaded:
            label = _record_label(r, args.label)
            target = _record_target(r, args.regression_target, args.hurt_penalty)
            raw_count += 1
            helped += int(r.get("helped", 0))
            hurt += int(r.get("hurt", 0))
            hold_advantage += int(r.get("hold_advantage", int(r.get("hold_delta", 0) > 0)))
            records.append({
                "features": r["features"],
                "label": label,
                "target": target,
            })

    if not records:
        raise SystemExit("No records loaded")

    if args.objective == "regression":
        selector = _train_regressor(records, args.steps, args.lr, args.seed)
        activation = "linear"
        metric_key = "regression_metrics"
        payload_label = args.regression_target
    else:
        selector = train_selector(records, args.steps, args.lr, args.seed)
        activation = "sigmoid"
        metric_key = "selector_metrics"
        payload_label = args.label
    payload = {
        "feature_names": FEATURE_NAMES[: int(selector["weights"].numel())],
        "weights": selector["weights"],
        "bias": selector["bias"],
        "mean": selector["mean"],
        "std": selector["std"],
        "activation": activation,
        "objective": args.objective,
        "label": payload_label,
        "metrics": selector["metrics"],
    }
    out = Path(args.selector_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

    summary = {
        "records": len(records),
        "raw_records": raw_count,
        "helped": helped,
        "hurt": hurt,
        "hold_advantage": hold_advantage,
        "positive": sum(r["label"] for r in records),
        "target_mean": sum(r["target"] for r in records) / max(raw_count, 1),
        "target_positive": sum(1 for r in records if r["target"] > 0),
        "help_rate": helped / max(raw_count, 1),
        "hurt_rate": hurt / max(raw_count, 1),
        "hold_advantage_rate": hold_advantage / max(raw_count, 1),
        "positive_rate": sum(r["label"] for r in records) / max(raw_count, 1),
        "selector_out": str(out),
        metric_key: selector["metrics"],
    }
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
