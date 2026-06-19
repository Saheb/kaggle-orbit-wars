"""Phase 5 sufficient-prefix WAVE planner (spec §7) — deterministic BC label generator.

Calls the shared `wave_primitives` (classify_holds / choose_attack_anchor / ready_wave_quota /
ship_choice_for_quota) — the SAME functions the features and eval use (invariants I1-I3). For a
single observation it produces, per owned source planet, exactly ONE decision (MAX_LANES=1):

  - DEFENSE : reinforce a HOLDABLE planet this source was claimed by (most-urgent claim wins).
  - ATTACK  : contribute to a feasible offensive wave, sized by the ready-wave quota.
  - NO_OP   : do nothing (the source is not needed / cannot land on time).

v1 arbitration priority (§6): imminent HOLDABLE defense BEFORE opportunistic attack. Targets the
REACTIVE floor (never the static garrison) so labels never teach undercommit. Per-step recompute
is automatic: the planner is re-run on every observation.

Outputs both kaggle-format moves (for self-play state generation) and the per-slot (fire, target,
ship-bin) label needed for BC, plus per-wave diagnostics for the §9 poisoning checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import wave_primitives as wp
from wave_primitives import (
    EPS,
    HOLD_HOLDABLE,
    WAVE_MARGIN,
    WAVE_TOL_STEPS,
    VALUE_HORIZON,
)


@dataclass
class SourceDecision:
    src_pid: int
    kind: str                     # "attack" | "defense" | "noop"
    target_pid: int | None = None
    ship_count: int = 0
    ship_bin: int | None = None
    tau: float | None = None
    arrival_tau: float | None = None
    quota_target: float = 0.0     # the share we tried to send (for §9 bin-resolution check)


@dataclass
class WaveDiag:
    target_pid: int
    kind: str                     # "attack" | "defense"
    tau: float
    floor: float                  # reactive floor at the anchor
    static_floor: float           # garrison + margin only (no reactive/prod)
    cover_inflight: float         # friendly mass already inbound in the window
    launched: float               # mass this planner launches into the wave this step
    remaining: float              # floor - cover at decision time
    ready_safe: float
    arrival_taus: list[float] = field(default_factory=list)   # per-source arrival, for spread


@dataclass
class PlanResult:
    decisions: dict[int, SourceDecision]      # src_pid -> decision (one per owned planet)
    waves: list[WaveDiag]
    holds: dict = field(default_factory=dict)  # src_pid -> wave_primitives.HoldInfo


def _by_pid(planets):
    return {wp.planet_id(p): p for p in planets}


def _target_value(target, current_step, episode_steps, value_horizon=VALUE_HORIZON):
    """Production-weighted holdability proxy for ATTACK target ordering (planner-internal only;
    NOT a model feature). Mirrors hold_value so attacks prefer durable, productive targets."""
    remaining_steps = max(0, int(episode_steps) - int(current_step))
    return float(target[6]) * min(remaining_steps, int(value_horizon))


def _static_floor(target):
    return float(target[5]) + WAVE_MARGIN


def plan(
    obs: dict,
    player: int,
    episode_steps: int = 500,
    min_ship_bin: int = 0,
    beta: float = wp.DEFAULT_REACTIVE_BETA,
    tol: float = WAVE_TOL_STEPS,
) -> PlanResult:
    planets = obs["planets"]
    fleets = obs.get("fleets") or []
    step = int(obs.get("step", 0))
    by_pid = _by_pid(planets)

    holds = wp.classify_holds(planets, fleets, player, current_step=step,
                              episode_steps=episode_steps, tol=tol)
    # Final per-source spare available for ATTACK (post defense-claim subtraction).
    safe_by_pid = {pid: max(0.0, info.safe_sendable) for pid, info in holds.items()}

    decisions: dict[int, SourceDecision] = {
        pid: SourceDecision(src_pid=pid, kind="noop") for pid in holds
    }
    waves: list[WaveDiag] = []
    assigned: set[int] = set()

    # ---- DEFENSE first (most-urgent HOLDABLE wins each claimed source) ----
    holdables = sorted(
        (info for info in holds.values() if info.hold_class == HOLD_HOLDABLE),
        key=lambda i: (i.d_def_tau if i.d_def_tau is not None else float("inf"), i.planet_id),
    )
    for hinfo in holdables:
        tgt = by_pid.get(hinfo.planet_id)
        if tgt is None:
            continue
        tau = float(hinfo.d_def_tau or 0.0)
        launched = 0.0
        arrivals: list[float] = []
        for qid, take in sorted(hinfo.claims.items(), key=lambda kv: kv[0]):
            if qid in assigned or take <= EPS:
                continue
            src = by_pid.get(qid)
            if src is None:
                continue
            # the claiming source's full spare is its base pool capacity; size to the claim
            cap = max(safe_by_pid.get(qid, 0.0), float(take))
            choice = wp.ship_choice_for_quota(src, tgt, cap, float(take), tau,
                                              tol=tol, min_ship_bin=min_ship_bin)
            if not choice.viable or choice.chosen_count <= 0:
                continue
            decisions[qid] = SourceDecision(
                src_pid=qid, kind="defense", target_pid=hinfo.planet_id,
                ship_count=choice.chosen_count, ship_bin=choice.chosen_bin,
                tau=tau, arrival_tau=choice.arrival_tau, quota_target=choice.ship_target,
            )
            assigned.add(qid)
            launched += choice.chosen_count
            if choice.arrival_tau is not None:
                arrivals.append(choice.arrival_tau)
        if launched > 0:
            floor, cover, remaining = wp.defense_remaining(tgt, planets, fleets, player, tau, tol=tol)
            waves.append(WaveDiag(
                target_pid=hinfo.planet_id, kind="defense", tau=tau,
                floor=floor, static_floor=_static_floor(tgt), cover_inflight=cover,
                launched=launched, remaining=remaining, ready_safe=0.0, arrival_taus=arrivals,
            ))

    # ---- ATTACK waves over feasible enemy/neutral targets, value-ordered ----
    def _attack_pool():
        return {pid: safe_by_pid[pid] for pid in safe_by_pid
                if pid not in assigned and safe_by_pid[pid] > EPS}

    targets = [p for p in planets if wp.planet_owner(p) != player and wp.planet_owner(p) != -2]
    targets.sort(key=lambda p: (-_target_value(p, step, episode_steps), wp.planet_id(p)))

    for tgt in targets:
        pool = _attack_pool()
        if not pool:
            break
        anchor = wp.choose_attack_anchor(tgt, planets, fleets, player, pool, beta=beta, tol=tol)
        if anchor is None or anchor.remaining <= EPS:
            continue
        src_planets = [by_pid[pid] for pid in pool]
        quota = wp.ready_wave_quota(src_planets, tgt, pool, anchor.remaining, anchor.tau,
                                    tol=tol, min_ship_bin=min_ship_bin)
        # §6 commit gate: only launch a wave whose READY mass can actually cross the reactive
        # floor (don't fritter sources into a wave that under-fills). Sources that can't form a
        # crossing wave stay home and remain available for a fillable target this step / next step.
        if not quota.crosses_if_all_ready_send:
            continue
        launched = 0.0
        arrivals: list[float] = []
        for qid in quota.ready_source_ids:
            if qid in assigned:
                continue
            src = by_pid[qid]
            choice = wp.ship_choice_for_quota(src, tgt, safe_by_pid[qid], quota.quotas.get(qid, 0.0),
                                              anchor.tau, tol=tol, min_ship_bin=min_ship_bin)
            if not choice.viable or choice.chosen_count <= 0:
                continue
            decisions[qid] = SourceDecision(
                src_pid=qid, kind="attack", target_pid=wp.planet_id(tgt),
                ship_count=choice.chosen_count, ship_bin=choice.chosen_bin,
                tau=anchor.tau, arrival_tau=choice.arrival_tau, quota_target=choice.ship_target,
            )
            assigned.add(qid)
            launched += choice.chosen_count
            if choice.arrival_tau is not None:
                arrivals.append(choice.arrival_tau)
        if launched > 0:
            waves.append(WaveDiag(
                target_pid=wp.planet_id(tgt), kind="attack", tau=anchor.tau,
                floor=anchor.floor, static_floor=_static_floor(tgt),
                cover_inflight=anchor.cover, launched=launched, remaining=anchor.remaining,
                ready_safe=quota.ready_safe, arrival_taus=arrivals,
            ))

    return PlanResult(decisions=decisions, waves=waves, holds=holds)


def plan_agent(obs, config=None, episode_steps: int = 500, min_ship_bin: int = 0):
    """Kaggle-format agent wrapper (for self-play state generation). Emits intercept-aimed moves."""
    from action_mask import _target_intercept_angle
    planets = obs["planets"]
    by_pid = _by_pid(planets)
    player = int(obs["player"])
    res = plan(obs, player, episode_steps=episode_steps, min_ship_bin=min_ship_bin)
    moves = []
    for d in res.decisions.values():
        if d.kind == "noop" or d.target_pid is None or d.ship_count <= 0:
            continue
        src, tgt = by_pid.get(d.src_pid), by_pid.get(d.target_pid)
        if src is None or tgt is None:
            continue
        angle = _target_intercept_angle(src, tgt, int(d.ship_count), obs)
        moves.append([int(d.src_pid), float(angle), int(d.ship_count)])
    return moves
