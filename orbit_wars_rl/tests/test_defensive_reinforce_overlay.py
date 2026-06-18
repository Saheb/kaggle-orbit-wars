"""Eval-time defensive reinforce overlay.

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_defensive_reinforce_overlay.py
"""

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action_mask import actions_from_target_policy, compute_action_masks


def test_overlay_forces_nearest_safe_source_and_logs_original_no_fire():
    obs = {
        "step": 12,
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 50.0, 50.0, 2.0, 5.0, 2.0],    # threatened owned target
            [1, 0, 40.0, 50.0, 2.0, 50.0, 2.0],   # nearby safe support source
            [2, 1, 80.0, 50.0, 2.0, 20.0, 2.0],   # enemy support mass
        ],
        "fleets": [
            [0, 1, 60.0, 50.0, math.pi, 9, 30.0], # enemy inbound to planet 0
        ],
    }
    masks = compute_action_masks(obs, player=0)
    n_planets = len(obs["planets"])
    fire_logits = torch.full((1, 16, n_planets), -10.0)       # policy wants no-fire everywhere
    target_logits = torch.zeros((1, 16, n_planets))
    ship_logits = torch.zeros((1, 16, n_planets, 32))
    stats = {}

    moves = actions_from_target_policy(
        fire_logits,
        target_logits,
        ship_logits,
        masks,
        obs,
        player=0,
        allow_reinforce=True,
        defensive_reinforce_k=1,
        defensive_reinforce_beta=2.2,
        defensive_reinforce_max_targets=1,
        defensive_reinforce_stats=stats,
    )

    assert len(moves) == 1
    assert int(moves[0][0]) == 1, moves              # support source reinforces target
    assert abs(float(moves[0][1])) < 0.2, moves      # east toward planet 0
    assert int(moves[0][2]) >= 38, moves             # fills the defensive deficit
    assert stats["forced_moves"] == 1
    assert stats["forced_targets"] == 1
    assert stats["orig_no_fire"] == 1
    assert stats["deficit_after"] <= 1.0
    assert stats["head_fire_lt_01"] == 1
    assert stats["head_target_top1"] == 1
    assert stats["head_ship_rank_n"] == 1


def test_overlay_skips_hopeless_target():
    obs = {
        "step": 12,
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 50.0, 50.0, 2.0, 5.0, 2.0],
            [1, 0, 40.0, 50.0, 2.0, 10.0, 2.0],   # not enough spare
            [2, 1, 80.0, 50.0, 2.0, 80.0, 2.0],
        ],
        "fleets": [
            [0, 1, 60.0, 50.0, math.pi, 9, 80.0],
        ],
    }
    masks = compute_action_masks(obs, player=0)
    n_planets = len(obs["planets"])
    stats = {}

    moves = actions_from_target_policy(
        torch.full((1, 16, n_planets), -10.0),
        torch.zeros((1, 16, n_planets)),
        torch.zeros((1, 16, n_planets, 32)),
        masks,
        obs,
        player=0,
        allow_reinforce=True,
        defensive_reinforce_k=1,
        defensive_reinforce_stats=stats,
    )

    assert moves == []
    assert stats["hopeless_targets"] == 1
    assert "forced_moves" not in stats


def test_overlay_value_gate_can_skip_fillable_target():
    obs = {
        "step": 12,
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 50.0, 50.0, 2.0, 5.0, 1.0],
            [1, 0, 40.0, 50.0, 2.0, 80.0, 1.0],
            [2, 1, 80.0, 50.0, 2.0, 10.0, 5.0],
        ],
        "fleets": [
            [0, 1, 60.0, 50.0, math.pi, 9, 20.0],
        ],
    }
    masks = compute_action_masks(obs, player=0)
    n_planets = len(obs["planets"])
    stats = {}

    moves = actions_from_target_policy(
        torch.full((1, 16, n_planets), -10.0),
        torch.zeros((1, 16, n_planets)),
        torch.zeros((1, 16, n_planets, 32)),
        masks,
        obs,
        player=0,
        allow_reinforce=True,
        defensive_reinforce_k=1,
        defensive_reinforce_beta=0.0,
        defensive_reinforce_value_margin=999.0,
        defensive_reinforce_stats=stats,
    )

    assert moves == []
    assert stats["value_gate_checked"] == 1
    assert stats["value_gate_skipped_targets"] == 1
    assert "forced_moves" not in stats


def test_overlay_overfill_scales_requested_mass_and_logs_realized_fill():
    obs = {
        "step": 12,
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 50.0, 50.0, 2.0, 5.0, 1.0],
            [1, 0, 40.0, 50.0, 2.0, 80.0, 1.0],
            [2, 1, 80.0, 50.0, 2.0, 10.0, 1.0],
        ],
        "fleets": [
            [0, 1, 60.0, 50.0, math.pi, 9, 20.0],
        ],
    }
    masks = compute_action_masks(obs, player=0)
    n_planets = len(obs["planets"])
    stats = {}

    moves = actions_from_target_policy(
        torch.full((1, 16, n_planets), -10.0),
        torch.zeros((1, 16, n_planets)),
        torch.zeros((1, 16, n_planets, 32)),
        masks,
        obs,
        player=0,
        allow_reinforce=True,
        defensive_reinforce_k=1,
        defensive_reinforce_beta=0.0,
        defensive_reinforce_max_targets=1,
        defensive_reinforce_overfill=1.5,
        defensive_reinforce_stats=stats,
    )

    assert len(moves) == 1
    assert stats["deficit_before"] == 16.0
    assert stats["requested_deficit"] == 24.0
    assert stats["forced_ships"] >= 24.0
    assert stats["realized_fill_forced_sum"] == stats["forced_ships"]
    assert stats["realized_fill_requested_sum"] == 24.0
    assert stats["realized_fill_full_targets"] == 1


def test_natural_head_audit_logs_attack_and_save_candidate_readiness():
    obs = {
        "step": 20,
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 30.0, 50.0, 2.0, 100.0, 2.0],  # source with enough mass
            [1, 0, 50.0, 50.0, 2.0, 5.0, 4.0],    # threatened own target
            [2, 1, 70.0, 50.0, 2.0, 5.0, 4.0],    # cheap enemy attack target
        ],
        "fleets": [
            [0, 1, 60.0, 50.0, math.pi, 9, 15.0], # enemy inbound to planet 1
        ],
    }
    masks = compute_action_masks(obs, player=0)
    n_planets = len(obs["planets"])
    fire_logits = torch.full((1, 16, n_planets), -10.0)
    fire_logits[0, 0, 2] = 10.0
    target_logits = torch.zeros((1, 16, n_planets))
    target_logits[0, 0, 2] = 10.0                  # slot 0 naturally attacks planet 2
    ship_logits = torch.zeros((1, 16, n_planets, 32))
    ship_logits[:, :, :, 20] = 10.0                # large-enough bin near the top
    stats = {}

    actions_from_target_policy(
        fire_logits,
        target_logits,
        ship_logits,
        masks,
        obs,
        player=0,
        allow_reinforce=True,
        natural_head_audit_stats=stats,
        natural_head_audit_beta=0.0,
    )

    assert stats["natural_all_slots"] == 2
    assert stats["natural_open_slots"] == 2
    assert stats["natural_all_attack_n"] >= 1
    assert stats["natural_all_attack_fire_ready"] >= 1
    assert stats["natural_all_attack_target_top1"] >= 1
    assert stats["natural_all_save_n"] >= 1
    assert stats["natural_all_save_fire_ready"] >= 1


if __name__ == "__main__":
    test_overlay_forces_nearest_safe_source_and_logs_original_no_fire()
    test_overlay_skips_hopeless_target()
    test_overlay_value_gate_can_skip_fillable_target()
    test_overlay_overfill_scales_requested_mass_and_logs_realized_fill()
    test_natural_head_audit_logs_attack_and_save_candidate_readiness()
    print("PASS: defensive reinforce overlay + natural head audit")
