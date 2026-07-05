#!/usr/bin/env python3
"""Warm up the new pairwise feature columns before PPO.

pair_kv.weight[:, 108:111] and target_scorer[0].weight[:, 204:207] are near-zero
because the seed checkpoint was zero-padded from a pairwise=12 model. PPO barely
moves them because the input signal is too small to produce gradient.

This script:
  1. Initialises those columns with noise matching the scale of the trained columns.
  2. Freezes everything except those columns + two tiny regression heads.
  3. Regresses: contribution_new -> predict roi_20, roi_50, enemy_contest.
  4. Saves the modified checkpoint (heads discarded).

Data: synthetic samples from the known feature ranges.
  roi_20, roi_50  in [-1, 1]   (clipped ROI values in features.py)
  enemy_contest   in [ 0, 5]   (contested ships / 100, capped at 5)
"""

import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_batch(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample a batch of the 3 new feature values from their known ranges."""
    roi_20 = torch.empty(batch_size, 1, device=device).uniform_(-1.0, 1.0)
    roi_50 = torch.empty(batch_size, 1, device=device).uniform_(-1.0, 1.0)
    contest = torch.empty(batch_size, 1, device=device).uniform_(0.0, 5.0)
    return torch.cat([roi_20, roi_50, contest], dim=-1)  # (B, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"]

    # --- pair_kv new columns ---
    w_kv = sd["pair_kv.weight"]              # [192, 111]
    trained_scale = w_kv[:, :108].std().item()
    print(f"Trained pair_kv column std: {trained_scale:.4f}")
    print(f"New columns before: {w_kv[:, 108:111].abs().mean():.6f}")
    with torch.no_grad():
        nn.init.normal_(w_kv[:, 108:111], std=trained_scale)
    print(f"New columns after init: {w_kv[:, 108:111].abs().mean():.6f}")

    # --- target_scorer new columns ---
    ts_w = sd["target_scorer.0.weight"]      # [96, 207]
    ts_scale = ts_w[:, :204].std().item()
    print(f"Trained target_scorer column std: {ts_scale:.4f}")
    print(f"TS new columns before: {ts_w[:, 204:207].abs().mean():.6f}")
    with torch.no_grad():
        nn.init.normal_(ts_w[:, 204:207], std=ts_scale)
    print(f"TS new columns after init: {ts_w[:, 204:207].abs().mean():.6f}")

    # --- isolate trainable parameters ---
    # We train only the new columns — no other model weights move.
    w_kv_new  = w_kv[:, 108:111].detach().clone().requires_grad_(True)   # [192, 3]
    ts_w_new  = ts_w[:, 204:207].detach().clone().requires_grad_(True)   # [96, 3]

    # Regression heads (discarded after training)
    head_kv = nn.Linear(192, 3, bias=False).to(device)
    head_ts = nn.Linear(96,  3, bias=False).to(device)

    params = [w_kv_new, ts_w_new] + list(head_kv.parameters()) + list(head_ts.parameters())
    optimiser = torch.optim.Adam(params, lr=args.lr)

    w_kv_new = w_kv_new.to(device)
    ts_w_new = ts_w_new.to(device)

    print(f"\nTraining {args.steps} steps  (batch={args.batch_size})...")
    for step in range(1, args.steps + 1):
        x = sample_batch(args.batch_size, device)          # (B, 3)

        # pair_kv branch: contribution from new columns only
        contrib_kv = F.linear(x, w_kv_new)                # (B, 192)
        pred_kv    = head_kv(contrib_kv)                   # (B, 3)
        loss_kv    = F.mse_loss(pred_kv, x)

        # target_scorer branch
        contrib_ts = F.linear(x, ts_w_new)                # (B, 96)
        pred_ts    = head_ts(contrib_ts)                   # (B, 3)
        loss_ts    = F.mse_loss(pred_ts, x)

        loss = loss_kv + loss_ts
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:4d}  loss_kv={loss_kv.item():.5f}  loss_ts={loss_ts.item():.5f}")

    # Write trained weights back into state dict
    with torch.no_grad():
        sd["pair_kv.weight"][:, 108:111]         = w_kv_new.cpu()
        sd["target_scorer.0.weight"][:, 204:207] = ts_w_new.cpu()

    print(f"\nFinal pair_kv new col |w|:  {sd['pair_kv.weight'][:, 108:111].abs().mean():.4f}")
    print(f"Final ts new col |w|:       {sd['target_scorer.0.weight'][:, 204:207].abs().mean():.4f}")
    print(f"Trained pair_kv col |w|:    {sd['pair_kv.weight'][:, :108].abs().mean():.4f}")

    ckpt["model"] = sd
    torch.save(ckpt, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
