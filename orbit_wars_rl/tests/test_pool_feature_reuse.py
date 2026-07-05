"""Regression test for the SPS pool-self optimization (feature reuse + subset forward).

Context: the per-EPISODE pool assignment made `compute_pool_actions` recompute
`get_features()` over ALL N envs and forward the frozen model over all N envs for EACH
(member, seat) group, then discard all but a few rows — correctness-clean but SPS-hostile.
The optimization reuses the features already computed for the learning model this step and
forwards the frozen model on ONLY its assigned envs (`index_features`).

This asserts that selecting envs BEFORE the forward (`index_features(feats, ids)` then
model) yields outputs IDENTICAL to forwarding the full batch then indexing the rows — i.e.
the opponent's action distribution is unchanged (the model has no cross-env coupling, so
the optimization changes only RNG alignment of the never-trained opponent samples, not the
distribution they are drawn from). Also covers non-contiguous and reordered id sets.

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_pool_feature_reuse.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from config import ModelConfig
from model import EntityTransformer
from torch_env import VecTorchEnv, MAX_OWNED
from train_torch import index_features


def _forward(model, feats):
    with torch.no_grad():
        return model(
            feats["planet_features"], feats["fleet_features"],
            feats["global_features"], feats["planet_mask"], feats["fleet_mask"],
            fire_mask=feats["fire_mask"],
            slot_valid=feats["slot_valid"], owned_indices=feats["owned_indices"],
            owned_count=feats["owned_count"],
            pairwise_features=feats.get("pairwise_features"),
        )


def _setup(num_envs=8, steps=30):
    torch.manual_seed(0)
    env = VecTorchEnv(num_envs=num_envs, num_players=2, device="cpu", action_decode="target")
    env.reset(seeds=list(range(num_envs)))
    for _ in range(steps):
        env.step({0: torch.randint(0, 2, (num_envs, MAX_OWNED, 3)),
                  1: torch.randint(0, 2, (num_envs, MAX_OWNED, 3))})
    model = EntityTransformer(ModelConfig()).eval()
    return env, model


def test_index_then_forward_matches_forward_then_index():
    """index_features(feats, ids) -> forward  ==  forward(all) then index rows."""
    env, model = _setup()
    for player in (0, 1):
        feats = env.get_features(player, max_planets=48, max_fleets=128)
        outs_full = _forward(model, feats)
        # Non-contiguous + reordered id set (exercises the per-episode grouping case).
        ids = torch.tensor([5, 1, 6, 2], dtype=torch.long)
        sub = index_features(feats, ids)
        outs_sub = _forward(model, sub)
        for key in ("fire_logits", "ship_logits", "target_logits", "value"):
            ref = outs_full[key][ids]
            got = outs_sub[key]
            assert torch.allclose(ref, got, atol=1e-5), (
                f"player {player} key {key}: max diff "
                f"{(ref - got).abs().max().item():.2e}")
    print("ok: subset-forward outputs identical to full-forward-then-index")


def test_index_features_handles_list_and_none():
    """owned_count is a python list (len N); pairwise/target_mask may be absent/None."""
    env, _ = _setup(num_envs=4, steps=10)
    feats = env.get_features(0, max_planets=48, max_fleets=128)
    assert isinstance(feats["owned_count"], list)
    ids = torch.tensor([3, 0], dtype=torch.long)
    sub = index_features(feats, ids)
    assert sub["owned_count"] == [feats["owned_count"][3], feats["owned_count"][0]]
    assert sub["planet_features"].shape[0] == 2
    # None passthrough
    feats["pairwise_features"] = None
    sub2 = index_features(feats, ids)
    assert sub2["pairwise_features"] is None
    print("ok: index_features handles list owned_count + None passthrough")


if __name__ == "__main__":
    test_index_then_forward_matches_forward_then_index()
    test_index_features_handles_list_and_none()
    print("ALL PASS")
