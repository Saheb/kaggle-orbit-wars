"""Projected-future timeline features.

Per planet, a K-step projection of owner + garrison assuming no new
launches: resolve each in-flight fleet to its target + arrival step, then run a vectorized
K-step recurrence (production + engine combat/flip). This is NOT "run the env K times" — it's
a single scatter + a K-iteration tensor recurrence, pure/compile-friendly.

The combat rule mirrors torch_env_fn.physics_core EXACTLY (attacker survivor = top-second;
reinforce if top owner == current owner, else attack/flip) so it parity-checks against
stepping the functional env K times with no actions.
"""
from __future__ import annotations

import math

import torch

# Self-contained on purpose (constants + speed formula duplicated from torch_env):
# export_agent.py inlines this module's source into submissions with imports stripped,
# so it must not depend on torch_env. Values must match torch_env / the kaggle env.
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SHIP_SPEED = 6.0

TIMELINE_K = 24                     # projection horizon (steps)
TIMELINE_DIM = 4 * TIMELINE_K       # mine/enemy/neutral one-hot + log-garrison, per step
CANDIDATE_TARGET_TIMELINE_DIM = 6
CANDIDATE_SOURCE_TIMELINE_DIM = 4
CANDIDATE_TIMELINE_DIM = CANDIDATE_TARGET_TIMELINE_DIM + CANDIDATE_SOURCE_TIMELINE_DIM
PROJECTED_HOLD_SEARCH_STEPS = 12


def _ship_speed(ships: torch.Tensor) -> torch.Tensor:
    """Speed formula from kaggle env, vectorized (mirror of torch_env._ship_speed)."""
    ships_clamped = torch.clamp(ships, min=1.0)
    base = (torch.log(ships_clamped) / math.log(1000.0)) ** 1.5
    speed = 1.0 + (MAX_SHIP_SPEED - 1.0) * base
    return torch.clamp(speed, max=MAX_SHIP_SPEED)


def resolve_target_eta(planets, planet_alive, fleets, angular_velocity):
    """Per fleet → (target_idx (N,F) long, eta (N,F) float). target -1 = hits nothing.
    Mirrors VecTorchEnv._resolve_targets_at but also returns the ETA to the resolved target."""
    fx = fleets[:, :, 2].unsqueeze(2)
    fy = fleets[:, :, 3].unsqueeze(2)
    fcos = torch.cos(fleets[:, :, 4]).unsqueeze(2)
    fsin = torch.sin(fleets[:, :, 4]).unsqueeze(2)
    speed = _ship_speed(fleets[:, :, 6]).clamp(min=1e-6).unsqueeze(2)
    px = planets[:, :, 2].unsqueeze(1)
    py = planets[:, :, 3].unsqueeze(1)
    pr = planets[:, :, 4].unsqueeze(1)
    angvel = angular_velocity.view(-1, 1, 1)
    dx0, dy0 = px - CENTER, py - CENTER
    orbit_r = torch.sqrt(dx0 * dx0 + dy0 * dy0)
    static = (orbit_r + pr) >= ROTATION_RADIUS_LIMIT
    phase0 = torch.atan2(dy0, dx0)
    eta = ((torch.sqrt((px - fx) ** 2 + (py - fy) ** 2) - pr) / speed).clamp(min=0.0)
    for _ in range(4):
        a = phase0 + angvel * eta
        lx = torch.where(static, px, CENTER + orbit_r * torch.cos(a))
        ly = torch.where(static, py, CENTER + orbit_r * torch.sin(a))
        eta = ((torch.sqrt((lx - fx) ** 2 + (ly - fy) ** 2) - pr) / speed).clamp(min=0.0)
    a = phase0 + angvel * eta
    lx = torch.where(static, px, CENTER + orbit_r * torch.cos(a))
    ly = torch.where(static, py, CENTER + orbit_r * torch.sin(a))
    vx, vy = lx - fx, ly - fy
    along = vx * fcos + vy * fsin
    perp = torch.abs(vx * fsin - vy * fcos)
    candidate = (along > 0) & (perp < pr + 0.5) & planet_alive.unsqueeze(1)
    has = candidate.any(dim=2)
    eta_masked = eta.masked_fill(~candidate, 1e6)
    tgt = eta_masked.argmin(dim=2)                                   # (N,F)
    eta_at = eta_masked.gather(2, tgt.unsqueeze(2)).squeeze(2)       # (N,F)
    tgt = torch.where(has, tgt, torch.full_like(tgt, -1))
    return tgt, eta_at


def project_timeline(planets, planet_alive, fleets, fleet_alive, angular_velocity,
                     num_players: int, K: int = TIMELINE_K,
                     return_arrivals: bool = False):
    """Returns (owner_ts (N,P,K) float in {-1,0..NP-1}, garr_ts (N,P,K) float): the projected
    owner and garrison of each planet at each of the next K steps, assuming no new launches."""
    N, P, _ = planets.shape
    F = fleets.shape[1]
    NP = num_players
    dev = planets.device

    tgt, eta = resolve_target_eta(planets, planet_alive, fleets, angular_velocity)
    arr_step = torch.ceil(eta).clamp(1, K).long()                   # (N,F) arrival step 1..K
    valid = fleet_alive & (tgt >= 0)
    f_owner = torch.clamp(fleets[:, :, 1].long(), 0, NP - 1)
    contrib = fleets[:, :, 6] * valid.float()

    # Scatter arriving fleet ships into (N, K, P, NP) by (arrival_step-1, target, owner).
    k_idx = (arr_step - 1).clamp(0, K - 1)
    tgt_c = tgt.clamp(0, P - 1)
    env_ar = torch.arange(N, device=dev).view(N, 1)
    flat = ((env_ar * K + k_idx) * P + tgt_c) * NP + f_owner        # (N,F)
    arrivals = torch.zeros(N * K * P * NP, device=dev).scatter_add(
        0, flat.reshape(-1), contrib.reshape(-1)).view(N, K, P, NP)

    owner = planets[:, :, 1].clone()                                # (N,P) float
    garr = planets[:, :, 5].clone()
    prod = planets[:, :, 6]
    owner_ts = torch.empty(N, P, K, device=dev)
    garr_ts = torch.empty(N, P, K, device=dev)
    for k in range(K):
        # Production (owned & alive).
        is_owned = (owner != -1) & planet_alive
        garr = garr + prod * is_owned.float()
        # Combat with this step's arrivals (attackers only; defender = current garrison).
        arr_k = arrivals[:, k]                                      # (N,P,NP)
        top_ships, top_owner = arr_k.max(dim=2)
        second = arr_k.scatter(2, top_owner.unsqueeze(2), 0.0).max(dim=2).values
        any_arr = top_ships > 0
        tie = (top_ships == second) & (top_ships > 0)
        survivor = torch.where(tie, torch.zeros_like(top_ships), top_ships - second)
        same = (top_owner.float() == owner) & any_arr & ~tie
        diff = (top_owner.float() != owner) & any_arr & ~tie
        g_re = garr + survivor
        g_at = garr - survivor
        flip = diff & (g_at < 0)
        new_garr = torch.where(same, g_re, torch.where(diff, g_at.abs(), garr))
        new_owner = torch.where(flip, top_owner.float(), owner)
        upd = planet_alive & any_arr
        garr = torch.where(upd, new_garr, garr)
        owner = torch.where(upd, new_owner, owner)
        owner_ts[:, :, k] = owner
        garr_ts[:, :, k] = garr
    if return_arrivals:
        return owner_ts, garr_ts, arrivals
    return owner_ts, garr_ts


def candidate_timeline_features(planets, planet_alive, arrivals, owner_ts, garr_ts,
                                player: int, candidate_ships, candidate_eta,
                                source_indices, slot_valid=None):
    """Summarize the target outcome if each source-target candidate launches.

    ``candidate_ships`` and ``candidate_eta`` are (N, S, P). Existing in-flight fleets use
    the exact arrivals tensor from :func:`project_timeline`; only the hypothetical fleet is
    added. The recurrence is source-target-local, so it evaluates all S×P candidates
    without duplicating the full P-planet state.

    Target channels: mine-at-arrival, signed arrival margin / 200, owned fraction from
    arrival through K, held-through-K, production delta / 100, terminal signed-margin
    delta / 200. Source channels replay the same no-new-launch future after deducting the
    candidate ships: owned fraction, held-through-K, production delta / 100, and terminal
    signed-margin delta / 200.
    """
    N, P, _ = planets.shape
    K = arrivals.shape[1]
    NP = arrivals.shape[3]
    S = candidate_ships.shape[1]
    dev = planets.device

    owner = planets[:, :, 1].unsqueeze(1).expand(-1, S, -1).clone()
    garr = planets[:, :, 5].unsqueeze(1).expand(-1, S, -1).clone()
    prod = planets[:, :, 6].unsqueeze(1).expand(-1, S, -1)
    alive = planet_alive.unsqueeze(1).expand(-1, S, -1)
    arrival_idx = torch.ceil(candidate_eta).clamp(1, K).long() - 1
    candidate_ships = candidate_ships.clamp(min=0.0)

    source_idx = source_indices.long().clamp(0, P - 1)
    source_gather = source_idx.unsqueeze(-1).expand(-1, -1, planets.shape[2])
    source_planets = torch.gather(planets, 1, source_gather)
    source_owner = source_planets[:, :, 1].unsqueeze(2).expand(-1, -1, P).clone()
    source_garr = (
        source_planets[:, :, 5].unsqueeze(2).expand(-1, -1, P) - candidate_ships
    ).clamp(min=0.0)
    source_prod = source_planets[:, :, 6].unsqueeze(2).expand(-1, -1, P)
    source_alive = torch.gather(planet_alive, 1, source_idx).unsqueeze(2).expand(-1, -1, P)
    source_arrival_idx = source_idx[:, None, :, None].expand(-1, K, -1, NP)
    source_arrivals = torch.gather(arrivals, 2, source_arrival_idx)
    baseline_source_owner = torch.gather(
        owner_ts, 1, source_idx.unsqueeze(-1).expand(-1, -1, K))
    baseline_source_garr = torch.gather(
        garr_ts, 1, source_idx.unsqueeze(-1).expand(-1, -1, K))

    mine_at_arrival = torch.zeros(N, S, P, device=dev)
    margin_at_arrival = torch.zeros_like(mine_at_arrival)
    mine_steps = torch.zeros_like(mine_at_arrival)
    held = torch.ones(N, S, P, dtype=torch.bool, device=dev)
    production_delta = torch.zeros_like(mine_at_arrival)
    source_mine_steps = torch.zeros_like(mine_at_arrival)
    source_held = torch.ones(N, S, P, dtype=torch.bool, device=dev)
    source_production_delta = torch.zeros_like(mine_at_arrival)

    for k in range(K):
        active = k >= arrival_idx
        mine_before = owner == float(player)
        if k == 0:
            base_mine_before = planets[:, :, 1].unsqueeze(1) == float(player)
        else:
            base_mine_before = owner_ts[:, :, k - 1].unsqueeze(1) == float(player)
        production_delta += prod * (mine_before.float() - base_mine_before.float())

        garr = garr + prod * (mine_before & alive).float()
        arr_k = arrivals[:, k].unsqueeze(1).expand(-1, S, -1, -1).clone()
        add_now = (arrival_idx == k) & alive
        arr_k[..., player] += candidate_ships * add_now.float()

        top_ships, top_owner = arr_k.max(dim=3)
        second = arr_k.scatter(3, top_owner.unsqueeze(3), 0.0).max(dim=3).values
        any_arr = top_ships > 0
        tie = (top_ships == second) & any_arr
        survivor = torch.where(tie, torch.zeros_like(top_ships), top_ships - second)
        same = (top_owner.float() == owner) & any_arr & ~tie
        diff = (top_owner.float() != owner) & any_arr & ~tie
        g_at = garr - survivor
        flip = diff & (g_at < 0)
        garr = torch.where(same, garr + survivor,
                           torch.where(diff, g_at.abs(), garr))
        owner = torch.where(flip, top_owner.float(), owner)

        mine_now = owner == float(player)
        at_arrival = arrival_idx == k
        signed_margin = torch.where(mine_now, garr, -garr)
        mine_at_arrival = torch.where(at_arrival, mine_now.float(), mine_at_arrival)
        margin_at_arrival = torch.where(at_arrival, signed_margin, margin_at_arrival)
        mine_steps += (active & mine_now).float()
        held &= (~active) | mine_now

        source_mine_before = source_owner == float(player)
        if k == 0:
            baseline_source_mine_before = source_planets[:, :, 1] == float(player)
        else:
            baseline_source_mine_before = baseline_source_owner[:, :, k - 1] == float(player)
        source_production_delta += source_prod * (
            source_mine_before.float()
            - baseline_source_mine_before.unsqueeze(2).float()
        )
        source_garr = source_garr + source_prod * (
            source_mine_before & source_alive).float()
        source_arr_k = source_arrivals[:, k].unsqueeze(2).expand(-1, -1, P, -1)
        source_top_ships, source_top_owner = source_arr_k.max(dim=3)
        source_second = source_arr_k.scatter(
            3, source_top_owner.unsqueeze(3), 0.0).max(dim=3).values
        source_any_arr = source_top_ships > 0
        source_tie = (source_top_ships == source_second) & source_any_arr
        source_survivor = torch.where(
            source_tie, torch.zeros_like(source_top_ships), source_top_ships - source_second)
        source_same = (
            (source_top_owner.float() == source_owner) & source_any_arr & ~source_tie)
        source_diff = (
            (source_top_owner.float() != source_owner) & source_any_arr & ~source_tie)
        source_g_at = source_garr - source_survivor
        source_flip = source_diff & (source_g_at < 0)
        source_garr = torch.where(
            source_same, source_garr + source_survivor,
            torch.where(source_diff, source_g_at.abs(), source_garr))
        source_owner = torch.where(source_flip, source_top_owner.float(), source_owner)
        source_mine_now = source_owner == float(player)
        source_mine_steps += source_mine_now.float()
        source_held &= source_mine_now

    horizon = (K - arrival_idx).clamp(min=1).float()
    baseline_terminal_margin = torch.where(
        owner_ts[:, :, -1].unsqueeze(1) == float(player),
        garr_ts[:, :, -1].unsqueeze(1),
        -garr_ts[:, :, -1].unsqueeze(1),
    )
    terminal_margin = torch.where(owner == float(player), garr, -garr)
    baseline_source_terminal_margin = torch.where(
        baseline_source_owner[:, :, -1].unsqueeze(2) == float(player),
        baseline_source_garr[:, :, -1].unsqueeze(2),
        -baseline_source_garr[:, :, -1].unsqueeze(2),
    )
    source_terminal_margin = torch.where(
        source_owner == float(player), source_garr, -source_garr)
    out = torch.stack([
        mine_at_arrival,
        (margin_at_arrival / 200.0).clamp(-5.0, 5.0),
        mine_steps / horizon,
        held.float(),
        (production_delta / 100.0).clamp(-2.0, 2.0),
        ((terminal_margin - baseline_terminal_margin) / 200.0).clamp(-5.0, 5.0),
        source_mine_steps / float(K),
        source_held.float(),
        (source_production_delta / 100.0).clamp(-2.0, 2.0),
        ((source_terminal_margin - baseline_source_terminal_margin) / 200.0).clamp(-5.0, 5.0),
    ], dim=-1)
    valid = alive.unsqueeze(-1).float()
    if slot_valid is not None:
        valid = valid * slot_valid.unsqueeze(-1).unsqueeze(-1).float()
    return out * valid


def projected_hold_sizes(planets, planet_alive, arrivals, owner_ts, garr_ts,
                         player: int, max_ships, candidate_distance,
                         source_indices, slot_valid=None,
                         min_ships: int = 5,
                         search_steps: int = PROJECTED_HOLD_SEARCH_STEPS):
    """Find a verified hold-sized attack for every source-target pair.

    A candidate succeeds only when it captures the target and owns it at every projected
    step from arrival through the no-new-launch horizon. It is rejected if deducting the
    fleet makes a source fall that stays ours in the baseline projection.

    The bounded search keeps only explicitly successful upper bounds and re-evaluates its
    final answer. Fleet speed makes the predicate potentially non-monotonic, so this is the
    smallest *found* verified fleet, not a mathematical global minimum. Unverified pairs
    fall back to all-in; callers can therefore use this as a sizing-only intervention
    without changing the policy's fire or target decision.
    """
    upper = max_ships.float().unsqueeze(-1).expand_as(candidate_distance).floor().clamp(min=0.0)
    eligible = upper >= float(min_ships)

    def evaluate(ships):
        eta = torch.ceil(
            candidate_distance / _ship_speed(ships).clamp(min=1e-6)
        ).clamp(min=1.0)
        feats = candidate_timeline_features(
            planets, planet_alive, arrivals, owner_ts, garr_ts, player,
            ships, eta, source_indices, slot_valid,
        )
        target_holds = (feats[..., 0] > 0.5) & (feats[..., 3] > 0.5)
        return target_holds, feats

    all_in_holds, _ = evaluate(upper)
    target_feasible = eligible & all_in_holds
    low = torch.full_like(upper, float(min_ships - 1))
    high = upper.clone()
    for _ in range(search_steps):
        active = target_feasible & ((high - low) > 1.0)
        if not active.any():
            break
        mid = torch.floor((low + high) / 2.0).clamp(min=float(min_ships))
        mid_holds, _ = evaluate(mid)
        high = torch.where(active & mid_holds, mid, high)
        low = torch.where(active & ~mid_holds, mid, low)

    final_holds, final_feats = evaluate(high)
    K = owner_ts.shape[-1]
    baseline_source_owner = torch.gather(
        owner_ts, 1, source_indices.long().unsqueeze(-1).expand(-1, -1, K))
    baseline_source_held = (baseline_source_owner == float(player)).all(dim=-1).unsqueeze(-1)
    candidate_source_held = final_feats[..., 7] > 0.5
    source_preserved = ~baseline_source_held | candidate_source_held
    feasible = target_feasible & final_holds & source_preserved

    feasible &= planet_alive.unsqueeze(1)
    if slot_valid is not None:
        feasible &= slot_valid.unsqueeze(-1)
    return torch.where(feasible, high, upper), feasible


def timeline_features(owner_ts, garr_ts, player: int):
    """Encode a projection into model input channels: (N, P, TIMELINE_DIM).

    Channel-major layout [mine(K) | enemy(K) | neutral(K) | log1p(garrison)/8 (K)]:
    projected ownership plus the garrison trace. Shared by torch_env.get_features
    (training) and features.extract_features (eval/export) so both paths encode
    identically by construction.
    """
    mine = (owner_ts == float(player)).float()
    enemy = ((owner_ts != float(player)) & (owner_ts >= 0)).float()
    neutral = (owner_ts < 0).float()
    garr = torch.log1p(garr_ts.clamp(min=0.0)) / 8.0
    return torch.cat([mine, enemy, neutral, garr], dim=2)
