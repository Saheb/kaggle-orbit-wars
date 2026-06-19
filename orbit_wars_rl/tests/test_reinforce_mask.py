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
            other = [p for p in range(te.planets.shape[1]) if owner[p] == 0 and p != src]
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
        enemy = [p for p in range(te.planets.shape[1]) if owner[p] == 1]
        for s in range(tm.shape[1]):
            if not sv[0, s]:
                continue
            src = int(oi[0, s])
            other_own = [p for p in range(te.planets.shape[1]) if owner[p] == 0 and p != src]
            for p in other_own:
                assert bool(tm[0, s, p]) == expect_own_legal, (
                    f"own-target legality should be {expect_own_legal} at gate=3 "
                    f"with {1 + n_extra} planets")
            # enemy targets are NEVER gated
            for p in enemy:
                assert bool(tm[0, s, p]), "enemy targets must stay legal under the gate"


def test_forward_staging_gate_blocks_rear_reinforcement():
    """Forward-staging gate (#4): with allow_reinforce + reinforce_forward_only, an own
    planet is a legal reinforce target only if it is CLOSER to the nearest enemy planet
    than the launch source. A rearward own planet (farther from the enemy) is masked; a
    forward one (closer) stays legal; enemy targets are never constrained. With the gate
    OFF the rear target is legal — proving the flag is what blocks it."""
    def setup(forward_only):
        te = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                         action_decode="target", allow_reinforce=True,
                         reinforce_forward_only=forward_only)
        te.reset([7])
        # Controlled 1-D layout on the x-axis; everything else neutral and far away so
        # the designated enemy (planet 3) is unambiguously the nearest enemy.
        te.planet_alive[0, :] = True
        te.planets[0, :, 1] = -1          # all neutral
        te.planets[0, :, 2] = 1000.0      # x far
        te.planets[0, :, 3] = 0.0         # y
        te.planets[0, :, 5] = 20.0        # ships
        #             idx  owner   x       role
        for idx, own, px in [(0, 0, 40.0),   # S  source (mine)
                             (1, 0, 70.0),   # F  front  (mine, closer to enemy)
                             (2, 0, 10.0),   # R  rear   (mine, farther from enemy)
                             (3, 1, 100.0)]: # E  enemy
            te.planets[0, idx, 1] = own
            te.planets[0, idx, 2] = px
        return te

    for forward_only, rear_legal in ((True, False), (False, True)):
        te = setup(forward_only)
        f = te.get_features(player=0)
        tm, sv, oi = f["target_mask"], f["slot_valid"], f["owned_indices"]
        s = next(i for i in range(tm.shape[1]) if sv[0, i] and int(oi[0, i]) == 0)
        assert bool(tm[0, s, 1]), "front own target (closer to enemy) must stay legal"
        assert bool(tm[0, s, 2]) == rear_legal, (
            f"rear own-target legality should be {rear_legal} (forward_only={forward_only})")
        assert bool(tm[0, s, 3]), "enemy target must never be constrained by forward-staging"
        assert not bool(tm[0, s, 0]), "source must never target itself"


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


def test_garrison_floor_vetoes_drain_but_spares_attacks():
    """#1 Garrison floor: a REINFORCE launch that would drain its source below the
    floor is vetoed (no fleet); a launch that stays at/above the floor fires; and an
    ATTACK that drains below the floor is UNAFFECTED (floor only governs reinforcement).
    SHIP_COUNTS: bin 3 = 4 ships, bin 9 = 10 ships."""
    from orbit_wars_rl.torch_env import MAX_OWNED

    def run(ship_bin, target_idx_fn):
        te = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                         action_decode="target", allow_reinforce=True,
                         reinforce_garrison_floor=25.0)
        te.reset([7])
        B = _give_player0_a_second_planet(te)
        te.planets[0, 0, 5] = 30.0          # source (home, planet 0) has 30 ships
        oi, _ = te.owned_indices_for(0)
        a_slot = next(s for s in range(MAX_OWNED) if int(oi[0, s]) == 0)
        act = torch.zeros(1, MAX_OWNED, 4)
        act[0, a_slot, 0] = 1
        act[0, a_slot, 2] = ship_bin
        act[0, a_slot, 3] = target_idx_fn(te, B)
        n_before = int(te.fleet_alive[0].sum())
        te.step({0: act})
        return int(te.fleet_alive[0].sum()) - n_before

    enemy = lambda te, B: next(p for p in range(te.planets.shape[1])
                               if te.planet_alive[0, p] and int(te.planets[0, p, 1]) == 1)
    own = lambda te, B: B

    # reinforce sending 10 ships: 30-10=20 < floor 25 → VETOED
    assert run(9, own) == 0, "reinforce draining below floor must be vetoed"
    # reinforce sending 4 ships: 30-4=26 >= floor 25 → fires
    assert run(3, own) == 1, "reinforce staying above floor must fire"
    # ATTACK sending 10 ships: 30-10=20 < floor, but it's an attack → UNAFFECTED
    assert run(9, enemy) == 1, "garrison floor must not touch attacks"


def test_reinforce_transit_cost_charges_only_reinforcement():
    """#2 Transit cost: the launching player's step reward drops by cost × ships sent
    to OWN planets; an attack of the same size incurs no cost."""
    from orbit_wars_rl.torch_env import MAX_OWNED
    COST = 0.01

    def reward_for(target_idx_fn):
        te = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                         action_decode="target", allow_reinforce=True,
                         reinforce_cost=COST)
        te.reset([7])
        B = _give_player0_a_second_planet(te)
        te.planets[0, 0, 5] = 50.0
        oi, _ = te.owned_indices_for(0)
        a_slot = next(s for s in range(MAX_OWNED) if int(oi[0, s]) == 0)
        act = torch.zeros(1, MAX_OWNED, 4)
        act[0, a_slot, 0] = 1
        act[0, a_slot, 2] = 9          # bin 9 = 10 ships
        act[0, a_slot, 3] = target_idx_fn(te, B)
        _, rewards, done = te.step({0: act})
        assert not bool(done[0]), "env should not terminate on this step"
        return float(rewards[0, 0])

    enemy = lambda te, B: next(p for p in range(te.planets.shape[1])
                               if te.planet_alive[0, p] and int(te.planets[0, p, 1]) == 1)
    own = lambda te, B: B

    assert abs(reward_for(own) - (-COST * 10.0)) < 1e-5, "reinforce must be charged cost×ships"
    assert abs(reward_for(enemy)) < 1e-5, "attacks must incur no transit cost"


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
