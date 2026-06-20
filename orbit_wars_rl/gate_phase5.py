"""Phase 5 Step 6 — pre-PPO no-veto gate on synthetic concentration states (spec §11, §14.6).

The cheap CPU go/no-go BEFORE spending a GPU-hour on PPO. It checks that a policy, on hand-built
load-bearing states and with NO veto crutch (`sufficient_commit_factor = 0`):

  attack         : CROSSES the REACTIVE floor of an enemy target (sufficiency; arrival
                   synchronization was dropped by the 2026-06-19 audit — docs/phase5-blocked.md).
  defense_hold   : reinforces a HOLDABLE owned planet enough to cross its hold deficit.
  defense_doomed : does NOT feed a DOOMED planet, and DRAINS it into a counterattack.

The states are validated to be LOAD-BEARING: the oracle (`wave_planner`) passes every state and a
no-op policy fails every state. Policies are scored by the SAME shared `wave_primitives` floors the
planner / features / eval use (invariant I1) — never a private re-derivation.

A policy is any callable `(obs, player) -> {src_pid: wave_planner.SourceDecision}`. `oracle_policy`
wraps the deterministic wave planner; `noop_policy` does nothing. The model (BC) policy is supplied
by the Step 7 BC pipeline, which owns the feature/decode path and MUST decode with
`sufficient_commit_factor=0` (no veto) to honor this gate's contract.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))

import wave_planner as WPL
import wave_primitives as wp
from wave_primitives import EPS, HOLD_DOOMED, HOLD_HOLDABLE

EPISODE_STEPS = 500


@dataclass
class GateState:
    name: str
    category: str        # "attack" | "defense_hold" | "defense_doomed"
    player: int
    target_pid: int      # the planet the criterion is evaluated on
    obs: dict


@dataclass
class StateResult:
    name: str
    category: str
    passed: bool
    detail: str


# --------------------------------------------------------------------------------------
# Synthetic battery. Each state is load-bearing: the oracle solves it, no-op fails it.
# Planet schema: [pid, owner, x, y, _, ships, prod]. owner 0 = us, 1 = enemy.
# --------------------------------------------------------------------------------------
def battery() -> list[GateState]:
    return [
        # ---- ATTACK: a multi-source bundle must combine to cross a reactive floor ----
        GateState(
            name="attack_two_source_cross",
            category="attack", player=0, target_pid=10,
            obs={"step": 20, "planets": [
                [10, 1, 50, 50, 2, 30, 1],   # T enemy, 30 ships
                [0,  0, 30, 50, 2, 50, 1],   # S1 owned
                [1,  0, 70, 50, 2, 50, 1],   # S2 owned
            ], "fleets": []},
        ),
        # ---- ATTACK: a single sufficient source must cross alone ----
        GateState(
            name="attack_single_source_cross",
            category="attack", player=0, target_pid=10,
            obs={"step": 20, "planets": [
                [10, 1, 50, 50, 2, 20, 1],   # T enemy, 20 ships
                [0,  0, 40, 50, 2, 90, 1],   # S1 owned, large garrison
            ], "fleets": []},
        ),
        # ---- DEFENSE (hold): a HOLDABLE planet must be reinforced across its deficit ----
        GateState(
            name="defense_hold_reinforce",
            category="defense_hold", player=0, target_pid=5,
            obs={"step": 20, "planets": [
                [5, 0, 50, 50, 2, 10, 1],    # P owned, under threat
                [6, 0, 40, 50, 2, 50, 1],    # R owned reinforcement, close
                [9, 1, 90, 90, 2, 80, 1],    # distant enemy
            ], "fleets": [
                [100, 1, 50, 20, math.pi / 2, 0, 30],   # enemy fleet inbound to P
            ]},
        ),
        # ---- DEFENSE (doomed): must NOT feed P; must drain it into a counterattack ----
        GateState(
            name="defense_doomed_drain",
            category="defense_doomed", player=0, target_pid=5,
            obs={"step": 20, "planets": [
                [5, 0, 50, 50, 2, 40, 1],    # P owned, doomed
                [8, 1, 52, 50, 2, 20, 1],    # nearby enemy target the drained garrison can hit
            ], "fleets": [
                [100, 1, 50, 10, math.pi / 2, 0, 200],   # overwhelming enemy wave inbound to P
            ]},
        ),
    ]


# --------------------------------------------------------------------------------------
# Scorers — all reactive floors come from shared wave_primitives (invariant I1).
# --------------------------------------------------------------------------------------
def _holds(state: GateState):
    return wp.classify_holds(state.obs["planets"], state.obs.get("fleets") or [], state.player,
                             current_step=int(state.obs.get("step", 0)), episode_steps=EPISODE_STEPS)


def _score_attack(state: GateState, decisions: dict) -> tuple[bool, str]:
    planets = state.obs["planets"]
    fleets = state.obs.get("fleets") or []
    holds = _holds(state)
    pool = {pid: max(0.0, info.safe_sendable) for pid, info in holds.items()}
    target = next((p for p in planets if wp.planet_id(p) == state.target_pid), None)
    if target is None:
        return False, "target planet missing"
    anchor = wp.choose_attack_anchor(target, planets, fleets, state.player, pool)
    if anchor is None:
        return False, "no feasible wave (state ill-posed)"
    launched = sum(d.ship_count for d in decisions.values()
                   if d.kind == "attack" and d.target_pid == state.target_pid)
    ok = launched + EPS >= anchor.floor
    return ok, f"launched {launched:.1f} vs reactive floor {anchor.floor:.1f}"


def _score_defense_hold(state: GateState, decisions: dict) -> tuple[bool, str]:
    info = _holds(state).get(state.target_pid)
    if info is None or info.hold_class != HOLD_HOLDABLE:
        return False, f"target not HOLDABLE (got {info.hold_class if info else 'missing'})"
    reinforced = sum(d.ship_count for d in decisions.values()
                     if d.kind == "defense" and d.target_pid == state.target_pid)
    ok = reinforced + EPS >= info.remaining0
    return ok, f"reinforced {reinforced:.1f} vs hold deficit {info.remaining0:.1f}"


def _score_defense_doomed(state: GateState, decisions: dict) -> tuple[bool, str]:
    info = _holds(state).get(state.target_pid)
    if info is None or info.hold_class != HOLD_DOOMED:
        return False, f"target not DOOMED (got {info.hold_class if info else 'missing'})"
    fed = any(d.kind == "defense" and d.target_pid == state.target_pid for d in decisions.values())
    drained = any(d.src_pid == state.target_pid and d.kind == "attack" and d.ship_count > 0
                  for d in decisions.values())
    return (not fed) and drained, f"fed={fed} drained={drained}"


_SCORERS = {
    "attack": _score_attack,
    "defense_hold": _score_defense_hold,
    "defense_doomed": _score_defense_doomed,
}


# --------------------------------------------------------------------------------------
# Policies. A policy is (obs, player) -> {src_pid: SourceDecision}.
# --------------------------------------------------------------------------------------
def oracle_policy(obs: dict, player: int) -> dict:
    return WPL.plan(obs, player, episode_steps=EPISODE_STEPS).decisions


def noop_policy(obs: dict, player: int) -> dict:
    return {}


def run_gate(policy, states: list[GateState] | None = None) -> tuple[bool, list[StateResult]]:
    """Score `policy` on the battery. Returns (all_passed, per-state results)."""
    states = states if states is not None else battery()
    results: list[StateResult] = []
    for st in states:
        decisions = policy(st.obs, st.player)
        passed, detail = _SCORERS[st.category](st, decisions)
        results.append(StateResult(st.name, st.category, passed, detail))
    return all(r.passed for r in results), results


def _print(label: str, passed: bool, results: list[StateResult]) -> None:
    print(f"\n{label}: {'PASS' if passed else 'FAIL'}")
    for r in results:
        print(f"  [{'ok' if r.passed else 'XX'}] {r.category:15s} {r.name:28s} {r.detail}")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Phase 5 pre-PPO no-veto gate (§11)")
    ap.add_argument("--checkpoint", default=None,
                    help="BC checkpoint to gate (Step 7 supplies the model policy decode; "
                         "not wired here — run the load-bearing validation instead).")
    args = ap.parse_args()

    if args.checkpoint is not None:
        print("Model-policy decode is owned by the Step 7 BC pipeline (sufficient_commit_factor=0, "
              "no veto). Import gate_phase5.run_gate and pass your decoded-policy callable.")
        return 2

    # Load-bearing self-validation: the states must discriminate before they gate a real policy.
    oracle_ok, oracle_res = run_gate(oracle_policy)
    noop_ok, noop_res = run_gate(noop_policy)
    _print("ORACLE (must pass every state)", oracle_ok, oracle_res)
    _print("NO-OP (must fail every state)", noop_ok, noop_res)

    load_bearing = oracle_ok and not any(r.passed for r in noop_res)
    print(f"\nStates load-bearing (oracle passes, noop fails all): "
          f"{'YES' if load_bearing else 'NO'}")
    return 0 if load_bearing else 1


if __name__ == "__main__":
    sys.exit(main())
