"""Reinforcement target-mask: own planets become legal targets (except the launch
source) when allow_reinforce=True, and stay illegal when False — in BOTH the train
env (torch_env) and the eval/export path (action_mask.actions_from_target_policy).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from orbit_wars_rl.torch_env import VecTorchEnv
from orbit_wars_rl.action_mask import actions_from_target_policy, compute_action_masks


def _give_player0_a_second_planet(te):
    owner = te.planets[0, :, 1]
    neutral = [p for p in range(te.planets.shape[1]) if te.planet_alive[0, p] and owner[p] == -1]
    second = neutral[0]
    te.planets[0, second, 1] = 0
    te.planets[0, second, 5] = 10
    return second


def test_torch_env_target_mask_reinforce_toggle():
    for allow in (False, True):
        te = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                         action_decode="target", allow_reinforce=allow)
        te.reset([7])
        second = _give_player0_a_second_planet(te)
        f = te.get_features(player=0)
        tm, sv, oi = f["target_mask"], f["slot_valid"], f["owned_indices"]
        owner = te.planets[0, :, 1]
        for s in range(tm.shape[1]):
            if not sv[0, s]:
                continue
            src = int(oi[0, s])
            # the source planet is NEVER a legal target of itself
            assert not bool(tm[0, s, src]), "source must never target itself"
            # the OTHER own planet: legal iff reinforcement is on
            other = [p for p in range(tm.shape[2]) if owner[p] == 0 and p != src]
            for p in other:
                assert bool(tm[0, s, p]) == allow, (
                    f"own-target legality should equal allow_reinforce={allow}")


def test_empire_gate_blocks_own_targets_below_threshold():
    """Empire-size gate: with allow_reinforce=True AND reinforce_gate_min_planets=3, own
    planets are ILLEGAL reinforce targets while the player owns < 3 planets, and become
    legal at >= 3. Enemy/neutral targets are never gated; the source is never a target."""
    # threshold 3: at 2 owned planets -> own targets blocked; at 3 -> allowed
    for n_extra, expect_own_legal in ((1, False), (2, True)):  # 1+1=2 planets, 1+2=3 planets
        te = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                         action_decode="target", allow_reinforce=True,
                         reinforce_gate_min_planets=3)
        te.reset([7])
        owner = te.planets[0, :, 1]
        neutral = [p for p in range(te.planets.shape[1])
                   if te.planet_alive[0, p] and owner[p] == -1]
        for p in neutral[:n_extra]:
            te.planets[0, p, 1] = 0
            te.planets[0, p, 5] = 10
        f = te.get_features(player=0)
        tm, sv, oi = f["target_mask"], f["slot_valid"], f["owned_indices"]
        owner = te.planets[0, :, 1]
        enemy = [p for p in range(tm.shape[2]) if owner[p] == 1]
        for s in range(tm.shape[1]):
            if not sv[0, s]:
                continue
            src = int(oi[0, s])
            other_own = [p for p in range(tm.shape[2]) if owner[p] == 0 and p != src]
            for p in other_own:
                assert bool(tm[0, s, p]) == expect_own_legal, (
                    f"own-target legality should be {expect_own_legal} at gate=3 "
                    f"with {1 + n_extra} planets")
            # enemy targets are NEVER gated
            for p in enemy:
                assert bool(tm[0, s, p]), "enemy targets must stay legal under the gate"


def test_action_mask_eval_reinforce_toggle():
    # source planet 0 at center. Own reinforce candidate (planet 1) due EAST,
    # enemy (planet 2) due NORTH — orthogonal so the chosen launch angle reveals which
    # target was selected. [id, owner, x, y, r, ships, prod]
    planets = [[0, 0, 50.0, 50.0, 2.0, 40, 3],   # mine, source
               [1, 0, 85.0, 50.0, 2.0, 5, 2],    # mine, EAST  (reinforce candidate)
               [2, 1, 50.0, 85.0, 2.0, 5, 3]]    # enemy, NORTH
    obs = {"planets": planets, "fleets": [], "step": 0, "player": 0,
           "angular_velocity": 0.0}  # 0 angular_velocity → static planets, clean aim
    masks = compute_action_masks(obs, player=0)
    n_p = len(planets)
    src_slot = [s for s in range(masks["owned_count"])
                if int(masks["owned_indices"][s]) == 0][0]
    fire_logits = torch.full((1, masks["owned_count"], n_p), -10.0)
    fire_logits[0, src_slot, 1] = 10.0
    fire_logits[0, src_slot, 2] = 10.0
    ship_logits = torch.zeros(1, masks["owned_count"], n_p, 32)
    ship_logits[0, src_slot, 1, 4] = 10.0  # bin 4 = a few ships
    ship_logits[0, src_slot, 2, 4] = 10.0

    def chosen_angle(allow):
        tl = torch.full((1, masks["owned_count"], n_p), -5.0)
        tl[0, src_slot, 1] = 10.0   # strongly prefer OWN planet 1 (east)
        tl[0, src_slot, 2] = 8.0    # enemy planet 2 (north) second
        acts = actions_from_target_policy(
            fire_logits.clone(), tl, ship_logits, masks, obs, player=0,
            ship_bin_mode="absolute", allow_reinforce=allow)
        mv = [m for m in acts if int(m[0]) == 0]
        assert mv, "source planet 0 should have launched"
        return float(mv[0][1])  # angle

    a_on = chosen_angle(True)    # reinforce ON  → aims EAST at own planet 1 (~0 rad)
    a_off = chosen_angle(False)  # reinforce OFF → own planet 1 masked → aims NORTH (~pi/2)
    assert abs(a_on) < 0.4, f"reinforce ON should aim east (~0), got {a_on}"
    assert abs(a_off - np.pi / 2) < 0.4, f"reinforce OFF should aim north (~pi/2), got {a_off}"


def test_reinforce_rate_counts_reinforce_vs_attack_launches():
    """reinforce_rate metric: after reset_reinforce_stats, the env counts realized
    launches per (env,player) and how many were reinforcement. Fire two sources for
    player 0 — one reinforce (target own), one attack (target enemy) — and expect
    fire_count=2, reinforce_count=1 (rate 0.5)."""
    from orbit_wars_rl.torch_env import MAX_OWNED
    te = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                     action_decode="target", allow_reinforce=True)
    te.reset([7])
    B = _give_player0_a_second_planet(te)
    te.planets[0, :, 5] = torch.clamp(te.planets[0, :, 5], min=30)  # ensure ships
    enemy = next(p for p in range(te.planets.shape[1])
                 if te.planet_alive[0, p] and int(te.planets[0, p, 1]) == 1)
    oi, _ = te.owned_indices_for(0)
    home_slot = next(s for s in range(MAX_OWNED) if int(oi[0, s]) == 0)
    b_slot = next(s for s in range(MAX_OWNED) if int(oi[0, s]) == B)
    act = torch.zeros(1, MAX_OWNED, 4)
    act[0, home_slot, 0] = 1; act[0, home_slot, 2] = 8; act[0, home_slot, 3] = enemy  # attack
    act[0, b_slot, 0] = 1;    act[0, b_slot, 2] = 8;    act[0, b_slot, 3] = 0          # reinforce home

    te.reset_reinforce_stats()
    te.step({0: act})
    assert float(te._fire_launch_count[0, 0]) == 2.0, "both launches should be counted"
    assert float(te._reinforce_launch_count[0, 0]) == 1.0, "exactly one was reinforcement"
    # target-owner share diagnostic: the two launches were own + enemy, neither neutral
    assert float(te._neutral_launch_count[0, 0]) == 0.0, "no launch targeted a neutral"


def test_torch_env_reinforce_launch_creates_fleet():
    """The training-side decode (_apply_actions) must actually CREATE a fleet for a
    reinforce launch when ON, and drop it when OFF (else reinforcement silently
    vanishes in training and the agent can never learn it)."""
    from orbit_wars_rl.torch_env import MAX_OWNED
    for allow in (False, True):
        te = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                         action_decode="target", allow_reinforce=allow)
        te.reset([7])
        B = _give_player0_a_second_planet(te)
        te.planets[0, :, 5] = torch.clamp(te.planets[0, :, 5], min=20)  # ensure ships
        oi, _ = te.owned_indices_for(0)
        a_slot = next(s for s in range(MAX_OWNED) if int(oi[0, s]) == 0)  # home (source)
        act = torch.zeros(1, MAX_OWNED, 4)
        act[0, a_slot, 0] = 1     # fire
        act[0, a_slot, 2] = 8     # ship bin
        act[0, a_slot, 3] = B     # target = own planet B (reinforce)
        n_before = int(te.fleet_alive[0].sum())
        te.step({0: act})
        created = int(te.fleet_alive[0].sum()) - n_before
        assert created == (1 if allow else 0), (
            f"reinforce launch should create {1 if allow else 0} fleet, got {created}")
