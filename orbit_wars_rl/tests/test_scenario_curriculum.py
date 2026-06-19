"""Tiny scenario-curriculum reset states.

Run:  /Users/saheb/home/.venv/bin/python orbit_wars_rl/tests/test_scenario_curriculum.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from torch_env import (
    CENTER,
    MAX_OWNED,
    _SCENARIO_AGG_ATTACK,
    _SCENARIO_HOLD_UNDER_PEEL,
    _SCENARIO_STAGE_ATTACK,
    SUN_RADIUS,
    VecTorchEnv,
)


def _env(kind: str, deadline: int = 8) -> VecTorchEnv:
    env = VecTorchEnv(
        num_envs=1,
        num_players=2,
        device="cpu",
        action_decode="target",
        allow_reinforce=True,
        scenario_curriculum=kind,
        scenario_fraction=1.0,
        scenario_deadline=deadline,
    )
    env.reset(seeds=[123])
    return env


def _segment_sun_distance(ax: float, ay: float, bx: float, by: float) -> float:
    vx = bx - ax
    vy = by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return ((ax - CENTER) ** 2 + (ay - CENTER) ** 2) ** 0.5
    t = ((CENTER - ax) * vx + (CENTER - ay) * vy) / denom
    t = max(0.0, min(1.0, t))
    px = ax + t * vx
    py = ay + t * vy
    return ((px - CENTER) ** 2 + (py - CENTER) ** 2) ** 0.5


def test_scenario_target_and_main_routes_are_sun_safe():
    for kind in ("agg_attack", "stage_attack", "hold_under_peel"):
        env = _env(kind)
        adv = int(env.scenario_adv_player[0])
        target = int(env.scenario_target[0])
        p = env.planets[0].cpu()
        tx, ty = float(p[target, 2]), float(p[target, 3])
        assert ((tx - CENTER) ** 2 + (ty - CENTER) ** 2) ** 0.5 > SUN_RADIUS + 1.0
        owner = p[:, 1].long()
        source_ids = [i for i in (0, 1) if int(owner[i]) == adv]
        for src in source_ids:
            sx, sy = float(p[src, 2]), float(p[src, 3])
            assert _segment_sun_distance(sx, sy, tx, ty) > SUN_RADIUS + 1.0, (
                kind, src, target
            )


def test_agg_attack_requires_multiple_sources():
    env = _env("agg_attack")
    assert int(env.scenario_id[0]) == _SCENARIO_AGG_ATTACK
    adv = int(env.scenario_adv_player[0])
    target = int(env.scenario_target[0])
    target_ships = float(env.planets[0, target, 5])
    owner = env.planets[0, :, 1].long()
    ships = env.planets[0, :, 5]
    adv_sources = ships[(owner == adv) & env.planet_alive[0]]
    top2 = torch.topk(adv_sources, 2).values
    assert float(top2.max()) <= target_ships + 1.0
    assert float(top2.sum()) > target_ships + 1.0


def test_agg_attack_terminal_success_and_failure():
    env = _env("agg_attack", deadline=5)
    adv = int(env.scenario_adv_player[0])
    opp = 1 - adv
    target = int(env.scenario_target[0])
    env.planets[0, target, 1] = adv
    rewards, done = env._check_done()
    assert bool(done[0])
    assert float(rewards[0, adv]) == 1.0
    assert float(rewards[0, opp]) == -1.0

    env = _env("agg_attack", deadline=5)
    adv = int(env.scenario_adv_player[0])
    opp = 1 - adv
    env.step_count[0] = 5
    rewards, done = env._check_done()
    assert bool(done[0])
    assert float(rewards[0, adv]) == -1.0
    assert float(rewards[0, opp]) == 1.0


def test_stage_attack_seeds_friendly_inbound():
    env = _env("stage_attack")
    assert int(env.scenario_id[0]) == _SCENARIO_STAGE_ATTACK
    adv = int(env.scenario_adv_player[0])
    target = int(env.scenario_target[0])
    assert bool(env.fleet_alive[0, 0])
    assert int(env.fleets[0, 0, 1]) == adv
    assert int(env._fleet_target_idx()[0, 0]) == target


def test_hold_under_peel_terminal_success_and_failure():
    env = _env("hold_under_peel", deadline=6)
    assert int(env.scenario_id[0]) == _SCENARIO_HOLD_UNDER_PEEL
    rewards, done = env._check_done()
    assert not bool(done[0])
    assert float(rewards.abs().sum()) == 0.0

    env.step_count[0] = 6
    adv = int(env.scenario_adv_player[0])
    opp = 1 - adv
    rewards, done = env._check_done()
    assert bool(done[0])
    assert float(rewards[0, adv]) == 1.0
    assert float(rewards[0, opp]) == -1.0

    env = _env("hold_under_peel", deadline=6)
    adv = int(env.scenario_adv_player[0])
    opp = 1 - adv
    target = int(env.scenario_target[0])
    env.planets[0, target, 1] = opp
    rewards, done = env._check_done()
    assert bool(done[0])
    assert float(rewards[0, adv]) == -1.0
    assert float(rewards[0, opp]) == 1.0


def test_hold_under_peel_noop_fails_by_deadline():
    env = _env("hold_under_peel", deadline=20)
    adv = int(env.scenario_adv_player[0])
    noop = torch.zeros((1, MAX_OWNED, 4), dtype=torch.long)
    done = torch.tensor([False])
    steps = 0
    while not bool(done[0]) and steps <= 25:
        _, _, done = env.step({adv: noop})
        steps += 1
    assert bool(done[0])
    assert not bool(env._last_scenario_success[0])


def test_auto_reset_reapplies_scenario():
    env = _env("stage_attack")
    env._auto_reset(torch.tensor([True]))
    assert int(env.scenario_id[0]) == _SCENARIO_STAGE_ATTACK
    assert bool(env.fleet_alive[0, 0])


if __name__ == "__main__":
    test_scenario_target_and_main_routes_are_sun_safe()
    test_agg_attack_requires_multiple_sources()
    test_agg_attack_terminal_success_and_failure()
    test_stage_attack_seeds_friendly_inbound()
    test_hold_under_peel_terminal_success_and_failure()
    test_hold_under_peel_noop_fails_by_deadline()
    test_auto_reset_reapplies_scenario()
    print("PASS: scenario curriculum reset states, terminal outcomes, and auto-reset")
