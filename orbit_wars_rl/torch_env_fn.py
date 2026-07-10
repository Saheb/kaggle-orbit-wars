"""Functional / immutable twin of torch_env.VecTorchEnv (Stage 0 scaffold).

WHY: torch_env.step mutates self.* in place and calls numpy/set/.tolist() in the comet
and reset paths — untraceable for whole-program fusion, so torch.compile gets ~0% (see
docs/perf.md). This module re-expresses the env as pure functions over an immutable
EnvState so (a) torch.compile can fuse the whole step, and (b) a later JAX port is
near-mechanical (swap torch→jnp/lax). Comets are dropped in v1 (user-approved: kills the
Python-serial spawn path; re-add vectorized later for final training).

Validation contract: torch_env.py is the ORACLE. The differential harness
(tests/test_env_fn_parity or scratchpad) initializes an EnvState from a live VecTorchEnv,
then steps both on identical actions and asserts state equality every tick. Nothing here
is trusted until it matches the oracle.

Tensor layout (identical to torch_env):
    planets (N, 48, 7) → [id, owner, x, y, radius, ships, production]
    fleets  (N, 256, 7) → [id, owner, x, y, angle, from_pid, ships]
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch

from torch_env import (
    MAX_PLANETS, MAX_FLEETS, CENTER, BOARD_SIZE, SUN_RADIUS, ROTATION_RADIUS_LIMIT,
    MAX_SHIP_SPEED, MAX_OWNED, ANGLE_BIN_WIDTH, _ship_speed,
)


class EnvState(NamedTuple):
    """Immutable env state. Every field is a batched tensor with leading dim N.
    Orbital params (initial_angle/orbital_r/is_orbiting) are cached derived quantities
    recomputed only at reset — carried in-state so step_fn stays a pure function of it."""
    planets: torch.Tensor          # (N, 48, 7) float
    planet_alive: torch.Tensor     # (N, 48) bool
    fleets: torch.Tensor           # (N, 256, 7) float
    fleet_alive: torch.Tensor      # (N, 256) bool
    step_count: torch.Tensor       # (N,) long
    angular_velocity: torch.Tensor # (N,) float
    next_fleet_id: torch.Tensor    # (N,) long
    done: torch.Tensor             # (N,) bool
    # cached orbital params (function of init planet positions; recomputed at reset)
    init_planets: torch.Tensor     # (N, 48, 7) float — frozen initial layout (orbit reference)
    planet_initial_angle: torch.Tensor  # (N, 48) float
    planet_orbital_r: torch.Tensor      # (N, 48) float
    planet_is_orbiting: torch.Tensor    # (N, 48) bool


def compute_orbital(init_planets: torch.Tensor, planet_alive: torch.Tensor):
    """Derive (initial_angle, orbital_r, is_orbiting) from the frozen initial layout.
    Mirrors VecTorchEnv._precompute_orbital_params exactly."""
    ix = init_planets[:, :, 2]
    iy = init_planets[:, :, 3]
    r = init_planets[:, :, 4]
    dx = ix - CENTER
    dy = iy - CENTER
    orbital_r = torch.sqrt(dx * dx + dy * dy)
    initial_angle = torch.atan2(dy, dx)
    is_orbiting = ((orbital_r + r) < ROTATION_RADIUS_LIMIT) & planet_alive
    return initial_angle, orbital_r, is_orbiting


def state_from_torch_env(env) -> EnvState:
    """Extract an EnvState from a live VecTorchEnv (the oracle). Clones so the functional
    env and the oracle don't alias. Assumes comets are OFF on the oracle."""
    ia, orb_r, is_orb = compute_orbital(env.init_planets, env.planet_alive)
    return EnvState(
        planets=env.planets.clone(),
        planet_alive=env.planet_alive.clone(),
        fleets=env.fleets.clone(),
        fleet_alive=env.fleet_alive.clone(),
        step_count=env.step_count.clone(),
        angular_velocity=env.angular_velocity.clone(),
        next_fleet_id=env.next_fleet_id.clone(),
        done=env.done.clone(),
        init_planets=env.init_planets.clone(),
        planet_initial_angle=ia.clone(),
        planet_orbital_r=orb_r.clone(),
        planet_is_orbiting=is_orb.clone(),
    )


def make_board_pool(k: int, num_players: int, device, seed: int = 0):
    """Pre-generate k fresh boards with the REAL kaggle generate_planets (once, at startup).
    Vectorized reset then gathers from this pool instead of calling numpy per-env — exact
    parity with the generator, zero numpy in the hot loop. Returns (planets, alive) tensors
    of shape (k, 48, 7) / (k, 48). angular_velocity is drawn per-reset, not pooled."""
    from kaggle_environments.envs.orbit_wars.orbit_wars import generate_planets
    import random as _random
    rng = _random.Random(seed)
    planets = np.zeros((k, MAX_PLANETS, 7), dtype=np.float32)
    alive = np.zeros((k, MAX_PLANETS), dtype=bool)
    for b in range(k):
        raw = generate_planets(rng)
        n = len(raw)
        for i, p in enumerate(raw):
            planets[b, i] = p
        alive[b, :n] = True
        planets[b, n:, 1] = -1
        num_groups = n // 4
        if num_groups > 0:
            home = rng.randint(0, num_groups - 1)
            base = home * 4
            if num_players == 2:
                planets[b, base, 1] = 0;     planets[b, base, 5] = 10
                planets[b, base + 3, 1] = 1; planets[b, base + 3, 5] = 10
            elif num_players == 4:
                for j in range(4):
                    planets[b, base + j, 1] = j; planets[b, base + j, 5] = 10
    return (torch.from_numpy(planets).to(device), torch.from_numpy(alive).to(device))


def reset_masked(state: EnvState, done_mask: torch.Tensor, pool_planets: torch.Tensor,
                 pool_alive: torch.Tensor, pool_idx: torch.Tensor,
                 ang_vel_new: torch.Tensor) -> EnvState:
    """Vectorized reset (replaces the numpy/.tolist per-env loop). For each done env, gather
    a fresh board from the pre-generated pool and clear fleets/step/done; non-done envs are
    untouched. All tensor ops — no numpy, no Python loop, no host sync → torch.compile-safe.
    `pool_idx` (N,) and `ang_vel_new` (N,) are caller-supplied randomness (keeps this pure)."""
    N = state.planets.shape[0]
    fresh_planets = pool_planets[pool_idx]              # (N, 48, 7)
    fresh_alive = pool_alive[pool_idx]                  # (N, 48)
    dm3 = done_mask.view(N, 1, 1)
    dm2 = done_mask.view(N, 1)
    new_planets = torch.where(dm3, fresh_planets, state.planets)
    new_init = torch.where(dm3, fresh_planets, state.init_planets)
    new_alive = torch.where(dm2, fresh_alive, state.planet_alive)
    new_fleets = torch.where(dm3, torch.zeros_like(state.fleets), state.fleets)
    new_fleet_alive = state.fleet_alive & ~dm2
    new_step = torch.where(done_mask, torch.zeros_like(state.step_count), state.step_count)
    new_angvel = torch.where(done_mask, ang_vel_new, state.angular_velocity)
    new_nfid = torch.where(done_mask, torch.zeros_like(state.next_fleet_id), state.next_fleet_id)
    new_done = state.done & ~done_mask
    ia, orb_r, is_orb = compute_orbital(new_init, new_alive)
    return EnvState(new_planets, new_alive, new_fleets, new_fleet_alive, new_step,
                    new_angvel, new_nfid, new_done, new_init, ia, orb_r, is_orb)


def _owned_indices(planets, planet_alive, player: int):
    """Top-MAX_OWNED owned planets by garrison (ties → lowest idx). Mirrors
    VecTorchEnv.owned_indices_for. Returns (owned_idx (N,MO), slot_valid (N,MO) bool)."""
    owner = planets[:, :, 1]
    ships = planets[:, :, 5]
    is_mine = (owner.long() == player) & planet_alive
    N, P = is_mine.shape
    idx_grid = torch.arange(P, device=planets.device).expand(N, P)
    mine_key = -torch.round(ships).long() * P + idx_grid
    scores = torch.where(is_mine, mine_key, torch.full_like(idx_grid, 1 << 40))
    _, owned_idx = torch.topk(scores, MAX_OWNED, dim=1, largest=False)
    return owned_idx, torch.gather(is_mine, 1, owned_idx)


def _intercept_angle(planets, angular_velocity, src_x, src_y, src_r, ship_count, target_idx):
    """8-iteration lead intercept aimer. Mirrors VecTorchEnv._target_intercept_angle."""
    P = planets.shape[1]
    tgt = planets.gather(1, target_idx.clamp(0, P - 1).unsqueeze(-1).expand(-1, -1, 7))
    tx, ty, tgt_r = tgt[:, :, 2], tgt[:, :, 3], tgt[:, :, 4]
    speed = _ship_speed(ship_count)
    ang_vel = angular_velocity.unsqueeze(1)
    dx0, dy0 = tx - CENTER, ty - CENTER
    orbit_r = torch.sqrt(dx0 * dx0 + dy0 * dy0)
    static = (orbit_r + tgt_r) >= ROTATION_RADIUS_LIMIT
    phase0 = torch.atan2(dy0, dx0)
    gap = src_r + 0.1 + tgt_r
    t = ((torch.sqrt((tx - src_x) ** 2 + (ty - src_y) ** 2) - gap) / speed).clamp(min=0.0)
    for _ in range(8):
        aa = phase0 + ang_vel * t
        px = torch.where(static, tx, CENTER + orbit_r * torch.cos(aa))
        py = torch.where(static, ty, CENTER + orbit_r * torch.sin(aa))
        t = ((torch.sqrt((px - src_x) ** 2 + (py - src_y) ** 2) - gap) / speed).clamp(min=0.0)
    aa = phase0 + ang_vel * t
    px = torch.where(static, tx, CENTER + orbit_r * torch.cos(aa))
    py = torch.where(static, ty, CENTER + orbit_r * torch.sin(aa))
    return torch.atan2(py - src_y, px - src_x) % (2 * math.pi)


def apply_actions_core(planets, planet_alive, fleets, fleet_alive, next_fleet_id,
                       angular_velocity, actions, owner_id: int, ship_counts):
    """Functional launch application for one player (attack-only target-decode, clamp
    overflow). Deferred vs the oracle: reinforce discipline, sufficient-commit, diagnostics,
    fleet_tgt cache — additive/separable, don't affect physics or throughput. Fleet writes use
    a MASKLESS scratch-slot scatter (slot F = throwaway) instead of the oracle's boolean-masked
    advanced index → torch.compile-traceable. Returns (planets, fleets, fleet_alive, next_fleet_id)."""
    N, F, _ = fleets.shape
    P = planets.shape[1]
    dev = planets.device
    owned_idx, slot_valid = _owned_indices(planets, planet_alive, owner_id)
    fire = actions[:, :, 0].bool() & slot_valid
    src = planets.gather(1, owned_idx.unsqueeze(-1).expand(-1, -1, 7))
    src_x, src_y, src_r, src_ships, src_owner = src[:, :, 2], src[:, :, 3], src[:, :, 4], src[:, :, 5], src[:, :, 1].long()
    ship_count = ship_counts[actions[:, :, 2].long().clamp(0, ship_counts.shape[0] - 1)]
    raw_target = actions[:, :, 3].long()
    use_target = raw_target >= 0
    target_idx = raw_target.clamp(0, P - 1)
    tgt = planets.gather(1, target_idx.unsqueeze(-1).expand(-1, -1, 7))
    target_owner = tgt[:, :, 1].long()
    target_alive = planet_alive.gather(1, target_idx)
    tgt_ok = target_alive & (target_owner != owner_id)                       # attack-only
    target_ok = torch.where(use_target, tgt_ok, torch.ones_like(target_alive))
    # Angle: intercept aimer for target-decode slots, angle-bin center for the fallback (target<0).
    angle_binned = (actions[:, :, 1].float() + 0.5) * ANGLE_BIN_WIDTH
    angle_intercept = _intercept_angle(planets, angular_velocity, src_x, src_y, src_r, ship_count, target_idx)
    angle = torch.where(use_target, angle_intercept, angle_binned)
    ship_count = torch.minimum(ship_count, src_ships)                        # clamp overflow
    can_fire = fire & (src_owner == owner_id) & slot_valid & (src_ships >= ship_count) & target_ok & (ship_count > 0)

    start_x = src_x + torch.cos(angle) * (src_r + 0.1)
    start_y = src_y + torch.sin(angle) * (src_r + 0.1)

    # Free-slot allocation (topk-smallest of dead-slot indices).
    dead = ~fleet_alive
    slot_grid = torch.arange(F, device=dev).expand(N, F)
    slot_scores = torch.where(dead, slot_grid, torch.full_like(slot_grid, F + 1))
    rank_to_slot, _ = torch.topk(slot_scores, MAX_OWNED, dim=1, largest=False)
    rank_has = rank_to_slot < F
    rank_to_slot = rank_to_slot.clamp(max=F - 1)
    fire_rank = (can_fire.long().cumsum(dim=1) - 1).clamp(min=0)
    target_slot = rank_to_slot.gather(1, fire_rank)
    target_valid = rank_has.gather(1, fire_rank) & can_fire

    # Debit ships (maskless: debit=0 for non-launches).
    debit = ship_count * target_valid.float()
    new_ships = planets[:, :, 5].scatter_add(1, owned_idx, -debit)
    new_planets = torch.stack([planets[:, :, 0], planets[:, :, 1], planets[:, :, 2],
                               planets[:, :, 3], planets[:, :, 4], new_ships, planets[:, :, 6]], dim=2)

    new_ids = next_fleet_id.unsqueeze(1) + (target_valid.long().cumsum(dim=1) - 1).clamp(min=0)
    # MASKLESS scatter: invalid launches write to scratch slot F (dropped); valid → distinct free slots.
    safe_slot = torch.where(target_valid, target_slot, torch.full_like(target_slot, F))
    zpad = torch.zeros(N, 1, 7, device=dev)
    fpad = torch.cat([fleets, zpad], dim=1)
    cols = [new_ids.float(), torch.full_like(start_x, float(owner_id)), start_x, start_y,
            angle, src[:, :, 0], ship_count]
    new_fpad = torch.stack([fpad[:, :, c].scatter(1, safe_slot, cols[c]) for c in range(7)], dim=2)
    apad = torch.cat([fleet_alive, torch.zeros(N, 1, dtype=torch.bool, device=dev)], dim=1)
    apad = apad.scatter(1, safe_slot, torch.ones_like(safe_slot, dtype=torch.bool))
    new_next = next_fleet_id + target_valid.long().sum(dim=1)
    return new_planets, new_fpad[:, :F], apad[:, :F], new_next


def step_full_core(planets, planet_alive, fleets, fleet_alive, step_count, angular_velocity,
                   next_fleet_id, done, planet_initial_angle, planet_orbital_r, planet_is_orbiting,
                   actions0, actions1, ship_counts, num_players: int, episode_steps: int):
    """One full tick: apply both players' launches, then physics. Tensor-in/out (compilable).
    Returns the tensors needed to continue: (planets, fleets, fleet_alive, step_count,
    next_fleet_id, done, rewards)."""
    planets, fleets, fleet_alive, next_fleet_id = apply_actions_core(
        planets, planet_alive, fleets, fleet_alive, next_fleet_id, angular_velocity, actions0, 0, ship_counts)
    planets, fleets, fleet_alive, next_fleet_id = apply_actions_core(
        planets, planet_alive, fleets, fleet_alive, next_fleet_id, angular_velocity, actions1, 1, ship_counts)
    (new_planets, _, new_fleets, survives, new_step, new_done, rewards, _) = physics_core(
        planets, planet_alive, fleets, fleet_alive, step_count, angular_velocity,
        planet_initial_angle, planet_orbital_r, planet_is_orbiting, done, num_players, episode_steps)
    return new_planets, new_fleets, survives, new_step, next_fleet_id, new_done, rewards


def physics_core(planets, planet_alive, fleets, fleet_alive, step_count,
                 angular_velocity, planet_initial_angle, planet_orbital_r,
                 planet_is_orbiting, done, num_players: int, episode_steps: int):
    """Pure tensor-in / tensor-out physics tick — the compilable core. NamedTuple wrapping
    is done OUTSIDE (physics_step) because dynamo (torch 2.11) chokes on NamedTuple-typed
    inputs; plain tensors also map cleanly to a future JAX jit. Mirrors VecTorchEnv.step's
    physics + _check_done (all shaping coefs = 0) exactly, out-of-place. Combat uses a
    MASKLESS dense scatter (non-combat fleets add 0 ships) so there's no data-dependent
    index → fully traceable. Returns
    (new_planets, planet_alive, new_fleets, survives, new_step, new_done, rewards, newly_done)."""
    N, P, _ = planets.shape
    F = fleets.shape[1]
    NP = num_players
    dev = planets.device

    # 1. Production (owner != -1 and alive) — feeds combat this same tick.
    owner = planets[:, :, 1]
    is_owned = (owner != -1) & planet_alive
    ships_p = planets[:, :, 5] + planets[:, :, 6] * is_owned.float()

    # 2. Planet orbital motion (non-orbiting planets stay put).
    step_f = step_count.float().unsqueeze(-1)
    cur_angle = planet_initial_angle + angular_velocity.unsqueeze(-1) * step_f
    p_old_x, p_old_y = planets[:, :, 2], planets[:, :, 3]
    p_new_x = torch.where(planet_is_orbiting,
                          CENTER + planet_orbital_r * torch.cos(cur_angle), p_old_x)
    p_new_y = torch.where(planet_is_orbiting,
                          CENTER + planet_orbital_r * torch.sin(cur_angle), p_old_y)

    # 3. Fleet movement.
    speed = _ship_speed(fleets[:, :, 6]) * fleet_alive.float()
    f_old_x, f_old_y = fleets[:, :, 2], fleets[:, :, 3]
    f_new_x = f_old_x + torch.cos(fleets[:, :, 4]) * speed
    f_new_y = f_old_y + torch.sin(fleets[:, :, 4]) * speed

    # 4. Swept-pair collision (fleet old→new vs planet old→new): ||d0 + t·dv||² = r².
    fx0, fy0 = f_old_x.unsqueeze(2), f_old_y.unsqueeze(2)
    fx1, fy1 = f_new_x.unsqueeze(2), f_new_y.unsqueeze(2)
    px0, py0 = p_old_x.unsqueeze(1), p_old_y.unsqueeze(1)
    px1, py1 = p_new_x.unsqueeze(1), p_new_y.unsqueeze(1)
    pr = planets[:, :, 4].unsqueeze(1)
    d0x, d0y = fx0 - px0, fy0 - py0
    dvx, dvy = (fx1 - fx0) - (px1 - px0), (fy1 - fy0) - (py1 - py0)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - pr * pr
    disc = b * b - 4.0 * a * c
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    safe_a = torch.where(a > 1e-12, a, torch.ones_like(a))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    hit_moving = (disc >= 0) & (t2 >= 0.0) & (t1 <= 1.0)
    hit_degenerate = (a < 1e-12) & (c <= 0.0)
    hit = (hit_moving & (a >= 1e-12)) | hit_degenerate
    hit = hit & (fleet_alive.unsqueeze(2) & planet_alive.unsqueeze(1))
    hit_any = hit.any(dim=2)
    hit_planet_idx = hit.float().argmax(dim=2)

    # 5. Sun crossing (point-to-segment distance from CENTER).
    seg_dx, seg_dy = f_new_x - f_old_x, f_new_y - f_old_y
    seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
    seg_len2_safe = torch.where(seg_len2 > 0, seg_len2, torch.ones_like(seg_len2))
    tt = torch.clamp(((CENTER - f_old_x) * seg_dx + (CENTER - f_old_y) * seg_dy) / seg_len2_safe, 0.0, 1.0)
    sun_dist = torch.sqrt((f_old_x + tt * seg_dx - CENTER) ** 2 + (f_old_y + tt * seg_dy - CENTER) ** 2)
    sun_dist = torch.where(seg_len2 > 0, sun_dist, torch.full_like(sun_dist, 1000.0))
    crosses_sun = sun_dist < SUN_RADIUS

    # 6. Out-of-bounds.
    in_bounds = (f_new_x >= 0) & (f_new_x <= BOARD_SIZE) & (f_new_y >= 0) & (f_new_y <= BOARD_SIZE)

    combat_mask = hit_any & fleet_alive
    survives = fleet_alive & ~combat_mask & in_bounds & ~crosses_sun

    # 7. Combat — MASKLESS dense scatter (non-combat fleets add 0 ships → same result, but
    # no data-dependent boolean index → torch.compile-traceable, unlike the oracle).
    fleet_owner = torch.clamp(fleets[:, :, 1].long(), 0, NP - 1)
    ships_contrib = fleets[:, :, 6] * combat_mask.float()
    env_idx = torch.arange(N, device=dev).unsqueeze(1).expand(N, F)
    flat_idx = (env_idx * P + hit_planet_idx) * NP + fleet_owner
    attacker = torch.zeros(N * P * NP, device=dev).scatter_add(
        0, flat_idx.reshape(-1), ships_contrib.reshape(-1)).view(N, P, NP)
    top_ships, top_owner = attacker.max(dim=2)
    second_ships = attacker.scatter(2, top_owner.unsqueeze(2), 0.0).max(dim=2).values
    tie = (top_ships == second_ships) & (top_ships > 0)
    survivor = torch.where(tie, torch.zeros_like(top_ships), top_ships - second_ships)
    any_combat = top_ships > 0

    p_owner = planets[:, :, 1]
    same_owner = (top_owner.float() == p_owner) & any_combat & ~tie
    diff_owner = (top_owner.float() != p_owner) & any_combat & ~tie
    ns = torch.where(same_owner, ships_p + survivor, ships_p)
    ns_attack = ships_p - survivor
    do_flip = diff_owner & (ns_attack < 0)
    ns = torch.where(diff_owner, ns_attack.abs(), ns)
    no = torch.where(do_flip, top_owner.float(), p_owner)
    upd = planet_alive & any_combat
    final_ships = torch.where(upd, ns, ships_p)
    final_owner = torch.where(upd, no, p_owner)

    # Reassemble planets/fleets out-of-place: [id, owner, x, y, radius, ships, production].
    new_planets = torch.stack([planets[:, :, 0], final_owner, p_new_x, p_new_y,
                               planets[:, :, 4], final_ships, planets[:, :, 6]], dim=2)
    new_fleets = torch.stack([fleets[:, :, 0], fleets[:, :, 1], f_new_x, f_new_y,
                              fleets[:, :, 4], fleets[:, :, 5], fleets[:, :, 6]], dim=2)
    new_step = step_count + 1

    # 8. Termination (mirrors _check_done with win_margin_coeff=0).
    owner_p = final_owner.long()
    owner_f = new_fleets[:, :, 1].long()
    alive_pl = torch.zeros(N, NP, dtype=torch.bool, device=dev)
    scores = torch.zeros(N, NP, device=dev)
    ships_pl = final_ships * planet_alive.float()
    ships_fl = new_fleets[:, :, 6] * survives.float()
    for pl in range(NP):
        has_p = ((owner_p == pl) & planet_alive).any(dim=1)
        has_f = ((owner_f == pl) & survives).any(dim=1)
        alive_pl[:, pl] = has_p | has_f
        scores[:, pl] = ((owner_p == pl).float() * ships_pl).sum(1) + ((owner_f == pl).float() * ships_fl).sum(1)
    n_alive = alive_pl.long().sum(dim=1)
    time_up = new_step >= (episode_steps - 1)
    newly_done = ((time_up | (n_alive <= 1)) & ~done)
    max_score = scores.max(dim=1, keepdim=True).values
    wins = (scores == max_score) & (max_score > 0)
    rewards = torch.where(wins, torch.ones_like(scores), -torch.ones_like(scores)) * newly_done.unsqueeze(1).float()
    new_done = done | newly_done

    return (new_planets, planet_alive, new_fleets, survives, new_step, new_done,
            rewards, newly_done)


def physics_step(state: EnvState, num_players: int, episode_steps: int):
    """NamedTuple wrapper over physics_core (kept out of the compiled region). Returns
    (new_state, rewards (N,P), newly_done (N,))."""
    (new_planets, planet_alive, new_fleets, survives, new_step, new_done,
     rewards, newly_done) = physics_core(
        state.planets, state.planet_alive, state.fleets, state.fleet_alive,
        state.step_count, state.angular_velocity, state.planet_initial_angle,
        state.planet_orbital_r, state.planet_is_orbiting, state.done,
        num_players, episode_steps)
    new_state = state._replace(planets=new_planets, fleets=new_fleets,
                               fleet_alive=survives, step_count=new_step, done=new_done)
    return new_state, rewards, newly_done
