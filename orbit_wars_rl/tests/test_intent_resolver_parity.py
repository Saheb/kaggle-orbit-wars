"""Intent-sizing resolver parity: the numpy (eval/export) twin must produce BIT-IDENTICAL
integer ship counts to the torch (training) twin, incl. at integer boundaries with float noise.
If this drifts, the policy is calibrated to a resolver that no longer exists at submission time."""
import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_mask import (resolve_intent_sizes_np, NUM_INTENTS,
                         INTENT_CAPTURE, INTENT_CAPTURE_DEFEND, INTENT_MAINTAIN, INTENT_ALL_IN)
from torch_env import _resolve_intent_sizes


def _both(cap_cost, reach_em, mass_soon, src_ships, is_own):
    npv = resolve_intent_sizes_np(cap_cost, reach_em, mass_soon, src_ships, is_own)
    tv = _resolve_intent_sizes(
        torch.tensor(cap_cost, dtype=torch.float32),
        torch.tensor(reach_em, dtype=torch.float32),
        torch.tensor(mass_soon, dtype=torch.float32),
        torch.tensor(src_ships, dtype=torch.float32),
        torch.tensor(is_own, dtype=torch.bool),
    ).numpy()
    return npv, tv


def test_fuzz_parity():
    rng = np.random.default_rng(0)
    n = 20000
    cap = rng.uniform(0, 400, n).astype(np.float32)
    reach = rng.uniform(0, 200, n).astype(np.float32)
    soon = rng.uniform(0, 150, n).astype(np.float32)
    S = rng.uniform(0, 300, n).astype(np.float32)
    own = rng.integers(0, 2, n).astype(bool)
    npv, tv = _both(cap, reach, soon, S, own)
    assert np.array_equal(npv, tv), f"max diff {np.abs(npv - tv).max()}"


def test_integer_boundary_noise():
    # costs exactly ON integers, perturbed by float32-scale noise both ways → must still match.
    base = np.arange(1, 300, dtype=np.float32)
    for delta in (0.0, 1e-5, -1e-5, 2e-6, -2e-6):
        cap = base + delta
        reach = np.zeros_like(base)
        soon = base + delta
        S = np.full_like(base, 500.0)
        own = np.ones_like(base, dtype=bool)
        npv, tv = _both(cap, reach, soon, S, own)
        assert np.array_equal(npv, tv), f"delta={delta} mismatch"


def test_semantics():
    # capture = ceil(cap_cost); all-in = S; capture clamped to S; maintain 0 on enemy targets.
    cap = np.array([30.0, 30.0, 80.0], dtype=np.float32)
    reach = np.array([10.0, 10.0, 0.0], dtype=np.float32)
    soon = np.array([12.0, 12.0, 5.0], dtype=np.float32)
    S = np.array([60.0, 25.0, 60.0], dtype=np.float32)     # mid, source-starved, mid
    own = np.array([False, False, True], dtype=bool)
    v = resolve_intent_sizes_np(cap, reach, soon, S, own)
    assert v[0, INTENT_CAPTURE] == 30 and v[0, INTENT_CAPTURE_DEFEND] == 40 and v[0, INTENT_ALL_IN] == 60
    assert v[0, INTENT_MAINTAIN] == 0                       # enemy target → no maintain
    # source-starved: capture (30) and all-in collapse to S=25 (the "can't take it from here" signal)
    assert v[1, INTENT_CAPTURE] == 25 and v[1, INTENT_ALL_IN] == 25
    # own target → maintain = ceil(5)+1 = 6, capture-defend uses cap_cost=80 clamped to 60
    assert v[2, INTENT_MAINTAIN] == 6
    assert v.shape[-1] == NUM_INTENTS
