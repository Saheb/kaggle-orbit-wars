"""Pinned RL champions: added as fixed 'self' opponents, never FIFO-evicted,
and the pin survives a save/load round-trip."""
from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opponent_pool import OpponentPool


def _sd(v):
    return {"w": torch.tensor([float(v)])}


def test_pinned_rl_is_never_fifo_evicted():
    pool = OpponentPool(max_self_members=2)
    pool.add_pinned_rl("rev38", _sd(99))

    # Flood with organic self-snapshots well past the cap.
    for step in range(1, 8):
        pool.add_self_checkpoint(step, _sd(step))

    names = [m.name for m in pool.members]
    pinned = [m for m in pool.members if m.pinned]
    organic = [m for m in pool.members if m.kind == "self" and not m.pinned]

    assert "seed_rev38" in names                      # champion survived the flood
    assert len(pinned) == 1
    assert len(organic) == 2                           # FIFO cap applies to organic only
    # the pin runs through the GPU 'self' forward path
    assert pinned[0].kind == "self" and pinned[0].state_dict is not None


def test_pinned_flag_survives_save_load():
    pool = OpponentPool(max_self_members=3)
    pool.add_pinned_rl("rev53b", _sd(7))
    pool.add_self_checkpoint(100, _sd(1))

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "pool.pt")
        pool.save(p)
        loaded = OpponentPool.load(p, reload_externals=False)

    pinned = [m for m in loaded.members if m.pinned]
    assert len(pinned) == 1 and pinned[0].name == "seed_rev53b"
    # organic self-snapshot is restored un-pinned
    assert any(m.kind == "self" and not m.pinned and m.name == "self_step_100"
               for m in loaded.members)
