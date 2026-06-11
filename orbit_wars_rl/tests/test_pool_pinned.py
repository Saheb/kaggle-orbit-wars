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


def test_pinned_uses_ema_regular_self_uses_lifetime():
    """uses_ema: pinned RL champions + externals use the recent EMA win-rate (so PFSP
    doesn't go stale as the policy improves); transient self-snapshots use lifetime."""
    pool = OpponentPool(max_self_members=5)
    pool.add_pinned_rl("rev38", _sd(1))
    pool.add_self_checkpoint(100, _sd(2))
    pinned = next(m for m in pool.members if m.pinned)
    organic = next(m for m in pool.members if m.kind == "self" and not m.pinned)
    assert pinned.uses_ema is True
    assert organic.uses_ema is False


def test_pinned_pfsp_weight_tracks_recent_not_stale_lifetime():
    """The fix: a pinned champion the agent USED to lose to but now beats should get a LOW
    PFSP weight (recent EMA), not a high one dragged up by stale early-run losses."""
    pool = OpponentPool(pfsp_min_games=5, ema_alpha=0.5)
    pool.add_pinned_rl("rev38", _sd(1))      # pinned -> EMA
    pool.add_self_checkpoint(100, _sd(2))    # organic -> lifetime
    pinned = next(m for m in pool.members if m.pinned)
    organic = next(m for m in pool.members if m.kind == "self" and not m.pinned)

    # identical history on both: 30 early losses, then 10 recent wins (policy improved).
    for m in (pinned, organic):
        for _ in range(30):
            pool.record_result(m, "loss")
        for _ in range(10):
            pool.record_result(m, "win")

    # lifetime wr is the same (10/40 = 0.25) for both...
    assert abs(pinned.win_rate - 0.25) < 1e-9
    assert abs(organic.win_rate - 0.25) < 1e-9
    # ...but the pinned EMA reflects the recent win streak (high), the organic has no EMA.
    assert pinned.ema_win_rate > 0.8
    assert organic.ema_games == 0
    # so PFSP down-weights the (now-beaten) pinned champion, but still over-weights the
    # organic snapshot on its stale lifetime 0.25.
    w_pinned = pool._pfsp_weight(pinned)
    w_organic = pool._pfsp_weight(organic)
    assert w_pinned < w_organic


def test_pinned_ema_survives_save_load():
    pool = OpponentPool(pfsp_min_games=5, ema_alpha=0.5)
    pool.add_pinned_rl("rev53b", _sd(7))
    pinned = next(m for m in pool.members if m.pinned)
    for _ in range(10):
        pool.record_result(pinned, "win")
    ema_before, games_before = pinned.ema_win_rate, pinned.ema_games

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "pool.pt")
        pool.save(p)
        loaded = OpponentPool.load(p, reload_externals=False)

    lp = next(m for m in loaded.members if m.pinned)
    assert abs(lp.ema_win_rate - ema_before) < 1e-9 and lp.ema_games == games_before
