"""Migrate a checkpoint trained with pairwise_feature_dim=12 to dim=15.

The three new features (roi_20, roi_50, enemy_contest) are appended as the
last three columns of the pairwise input.  In both layers that embed pairwise
features the pairwise columns are at the END of the concatenation:

    pair_kv:         cat([planet_emb(D), pairwise(F)])  →  Linear(D+F, 2D)
    target_scorer[0]: cat([q_tgt(D), k_tgt(D), pairwise(F)])  →  Linear(2D+F, D)

Zero-padding the last 3 columns of those weight matrices is correct and
produces a model whose first few forward passes are identical to the old model.
BC / PPO then gradient-updates the new columns into usefulness.

Usage:
    python orbit_wars_rl/migrate_pairwise_dim.py \\
        --input  seed_checkpoints/rev32b_6M_resume.pt \\
        --output seed_checkpoints/rev32b_6M_pairwise15.pt
"""

import argparse
import torch


OLD_DIM = 12
NEW_DIM = 15
DELTA   = NEW_DIM - OLD_DIM   # 3


def migrate(src: str, dst: str) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    # Checkpoints may store the state dict directly or under a 'model' key.
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "pair_kv.weight" in ckpt:
        sd = ckpt
    else:
        # Try to find state dict among top-level keys
        sd = None
        for v in ckpt.values():
            if isinstance(v, dict) and "pair_kv.weight" in v:
                sd = v
                break
        if sd is None:
            raise ValueError(
                f"Cannot locate model state dict in {src}. "
                "Keys found: " + str(list(ckpt.keys()))
            )

    # Detect current dim from the weight shape
    if "pair_kv.weight" not in sd:
        raise KeyError("'pair_kv.weight' not found — is this a pairwise-enabled checkpoint?")

    pair_kv_w = sd["pair_kv.weight"]          # (2D, D+F)
    D = pair_kv_w.shape[0] // 2               # entity_dim
    current_F = pair_kv_w.shape[1] - D        # current pairwise dim

    if current_F == NEW_DIM:
        print(f"Checkpoint already has pairwise_dim={NEW_DIM}. Nothing to do.")
        return
    if current_F != OLD_DIM:
        raise ValueError(
            f"Expected pairwise_dim={OLD_DIM}, found {current_F}. "
            "This script only migrates 12→15."
        )

    print(f"Migrating {src}: pairwise_dim {OLD_DIM} → {NEW_DIM}  (D={D})")

    # --- pair_kv.weight: (2D, D+12) → (2D, D+15) ---
    zeros_kv = torch.zeros(2 * D, DELTA, dtype=pair_kv_w.dtype)
    sd["pair_kv.weight"] = torch.cat([pair_kv_w, zeros_kv], dim=1)

    # pair_kv.bias is (2D,) — shape unchanged, no touch needed.

    # --- target_scorer.0.weight: (D, 2D+12) → (D, 2D+15) ---
    ts0_w = sd["target_scorer.0.weight"]      # (D, 2D+F)
    zeros_ts = torch.zeros(D, DELTA, dtype=ts0_w.dtype)
    sd["target_scorer.0.weight"] = torch.cat([ts0_w, zeros_ts], dim=1)

    # target_scorer.0.bias is (D,) — unchanged.
    # target_scorer.2 (Linear(D,1)) — unchanged.

    # Update config inside the checkpoint if present
    for cfg_key in ("cfg", "config", "model_cfg"):
        if cfg_key in ckpt and hasattr(ckpt[cfg_key], "pairwise_feature_dim"):
            ckpt[cfg_key].pairwise_feature_dim = NEW_DIM
            print(f"  Updated ckpt['{cfg_key}'].pairwise_feature_dim → {NEW_DIM}")

    # Persist the mutated state dict back
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt["model"] = sd
    else:
        ckpt = sd

    torch.save(ckpt, dst)
    print(f"Saved migrated checkpoint → {dst}")

    # Quick verification
    verify = torch.load(dst, map_location="cpu", weights_only=False)
    if isinstance(verify, dict) and "model" in verify:
        verify = verify["model"]
    new_F = verify["pair_kv.weight"].shape[1] - D
    assert new_F == NEW_DIM, f"Verification failed: got pairwise_dim={new_F}"
    new_ts_F = verify["target_scorer.0.weight"].shape[1] - 2 * D
    assert new_ts_F == NEW_DIM, f"Verification failed (target_scorer): got {new_ts_F}"
    print(f"Verification passed: pair_kv={verify['pair_kv.weight'].shape}, "
          f"target_scorer.0={verify['target_scorer.0.weight'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  required=True, help="Source checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Destination checkpoint (.pt)")
    args = parser.parse_args()
    migrate(args.input, args.output)
