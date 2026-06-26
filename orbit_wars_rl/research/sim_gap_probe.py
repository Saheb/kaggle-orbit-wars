"""Trajectory-diff harness: torch_env vs the real kaggle env.

Drives BOTH engines from the SAME seed (they share generate_planets + RNG order, so the
initial board is identical) and diffs state step-by-step to localize the torch_env<->kaggle
fidelity gap (docs/train-eval.md, suspect: geometry/timing).

Phase A: orbital motion only (no fleets) — isolates the planet-position `step` off-by-one.
Phase B: + a scripted fleet launch — isolates fleet movement / arrival / combat.

Run: orbit_wars_rl/.venv/bin/python orbit_wars_rl/sim_gap_probe.py
"""
import math
import numpy as np
import torch
from kaggle_environments import make
from orbit_wars_rl.torch_env import VecTorchEnv, CENTER


def kaggle_planets_by_id(obs):
    return {int(p[0]): p for p in obs.planets}


def torch_planets_by_id(env, e=0):
    out = {}
    P = env.planets.shape[1]
    for i in range(P):
        if bool(env.planet_alive[e, i]):
            pid = int(env.planets[e, i, 0].item())
            out[pid] = env.planets[e, i].tolist()
    return out


def run_orbital_diff(seed, n_steps=25):
    print(f"\n===== PHASE A: orbital motion diff (seed={seed}, no fleets) =====")
    # --- kaggle ---
    kenv = make("orbit_wars", configuration={"seed": seed, "episodeSteps": n_steps + 5}, debug=False)
    kenv.reset(num_agents=2)
    # --- torch ---
    tenv = VecTorchEnv(num_envs=1, num_players=2, device="cpu")
    tenv.reset([seed])

    # Sanity: same initial board?
    ko = kenv.state[0].observation
    kp0 = kaggle_planets_by_id(ko)
    tp0 = torch_planets_by_id(tenv)
    print(f"kaggle planets: {len(kp0)}  torch planets: {len(tp0)}  ang_vel k={ko.angular_velocity:.5f} t={tenv.angular_velocity[0].item():.5f}")
    common = sorted(set(kp0) & set(tp0))
    max_init = max(abs(kp0[i][2] - tp0[i][2]) + abs(kp0[i][3] - tp0[i][3]) for i in common)
    print(f"  initial-board max |Δpos| over {len(common)} planets: {max_init:.6f}")

    first_div = None
    for t in range(n_steps):
        kenv.step([[], []])  # no-op both players
        # torch: step with no actions
        tenv.step()
        ko = kenv.state[0].observation
        kstep = getattr(ko, "step", "NA")
        kp = kaggle_planets_by_id(ko)
        tp = torch_planets_by_id(tenv)
        tstep = int(tenv.step_count[0].item())
        common = sorted(set(kp) & set(tp))
        # max position divergence among orbiting planets
        diffs = []
        for i in common:
            dk = math.hypot(kp[i][2] - tp[i][2], kp[i][3] - tp[i][3])
            diffs.append((dk, i))
        diffs.sort(reverse=True)
        maxd, worst = diffs[0]
        if maxd > 1e-3 and first_div is None:
            first_div = t
        flag = "  <-- DIVERGES" if (maxd > 1e-3 and first_div == t) else ""
        if t < 6 or maxd > 1e-3:
            print(f"  step {t}: kaggle.step={kstep} torch.step={tstep}  max|Δpos|={maxd:.4f} (planet {worst}){flag}")
            if maxd > 1e-3 and t == (first_div):
                pk, pt = kp[worst], tp[worst]
                print(f"      planet {worst}: kaggle=({pk[2]:.3f},{pk[3]:.3f})  torch=({pt[2]:.3f},{pt[3]:.3f})")
    if first_div is None:
        print("  orbital motion MATCHES (no divergence > 1e-3 over", n_steps, "steps)")
    else:
        print(f"  >>> orbital motion DIVERGES first at step {first_div}")
    return first_div


def torch_inject_launch(tenv, owner, from_id, angle, ships, e=0):
    """Replicate kaggle phase-0 launch directly into torch_env tensors:
    subtract ships from source, spawn fleet just outside the planet radius."""
    P = tenv.planets.shape[1]
    slot = next(i for i in range(P)
                if bool(tenv.planet_alive[e, i]) and int(tenv.planets[e, i, 0].item()) == from_id)
    src_x = tenv.planets[e, slot, 2].item()
    src_y = tenv.planets[e, slot, 3].item()
    src_r = tenv.planets[e, slot, 4].item()
    tenv.planets[e, slot, 5] -= ships
    start_x = src_x + math.cos(angle) * (src_r + 0.1)
    start_y = src_y + math.sin(angle) * (src_r + 0.1)
    fslot = next(i for i in range(tenv.fleets.shape[1]) if not bool(tenv.fleet_alive[e, i]))
    nid = int(tenv.next_fleet_id[e].item())
    tenv.fleets[e, fslot] = torch.tensor([nid, owner, start_x, start_y, angle, from_id, ships],
                                         dtype=torch.float32)
    tenv.fleet_alive[e, fslot] = True
    tenv.next_fleet_id[e] += 1


def kaggle_fleets_by_id(obs):
    return {int(f[0]): f for f in obs.fleets}


def torch_fleets_by_id(env, e=0):
    out = {}
    for i in range(env.fleets.shape[1]):
        if bool(env.fleet_alive[e, i]):
            out[int(env.fleets[e, i, 0].item())] = env.fleets[e, i].tolist()
    return out


def run_fleet_diff(seed, n_steps=40):
    print(f"\n===== PHASE B: fleet movement/arrival/combat diff (seed={seed}) =====")
    kenv = make("orbit_wars", configuration={"seed": seed, "episodeSteps": n_steps + 5}, debug=False)
    kenv.reset(num_agents=2)
    tenv = VecTorchEnv(num_envs=1, num_players=2, device="cpu")
    tenv.reset([seed])

    ko = kenv.state[0].observation
    kp = kaggle_planets_by_id(ko)
    # source = player-0 planet with the MOST ships; launch a VALID amount (ships-1).
    p0 = [int(p[0]) for p in ko.planets if p[1] == 0]
    if not p0:
        print("  no p0 planet; skip"); return
    src = max(p0, key=lambda i: kp[i][5])
    sx, sy = kp[src][2], kp[src][3]
    ships = int(kp[src][5]) - 1
    cands = [(math.hypot(p[2]-sx, p[3]-sy), int(p[0])) for p in ko.planets if p[1] != 0]
    cands.sort()
    tgt = cands[0][1]
    tx, ty = kp[tgt][2], kp[tgt][3]
    angle = math.atan2(ty - sy, tx - sx)
    print(f"  launch: from planet {src} ({ships+1} ships) -> target {tgt} (dist {cands[0][0]:.1f}), angle {angle:.4f}, send {ships} ships")

    # Apply identical launch to both (player 0), no-op player 1.
    kenv.step([[[src, angle, ships]], []])
    torch_inject_launch(tenv, owner=0, from_id=src, angle=angle, ships=ships)
    tenv.step()

    first_fleet_div = None
    for t in range(1, n_steps):
        ko = kenv.state[0].observation
        kf = kaggle_fleets_by_id(ko)
        tf = torch_fleets_by_id(tenv)
        # fleet position divergence
        common_f = sorted(set(kf) & set(tf))
        fmsg = ""
        if common_f:
            fdiffs = [(math.hypot(kf[i][2]-tf[i][2], kf[i][3]-tf[i][3]), i) for i in common_f]
            fdiffs.sort(reverse=True)
            fmax, fworst = fdiffs[0]
            if fmax > 1e-2 and first_fleet_div is None:
                first_fleet_div = t
            fmsg = f"fleets[{len(common_f)}] max|Δpos|={fmax:.4f}"
        # planet-state divergence (ownership/ships) — detects combat differences
        kp = kaggle_planets_by_id(ko); tp = torch_planets_by_id(tenv)
        common_p = sorted(set(kp) & set(tp))
        own_diff = [(i, int(kp[i][1]), int(tp[i][1]), round(kp[i][5],1), round(tp[i][5],1))
                    for i in common_p if int(kp[i][1]) != int(tp[i][1]) or abs(kp[i][5]-tp[i][5]) > 0.5]
        nk, nt = len(kf), len(tf)
        line = f"  step {t}: kf={nk} tf={nt}  {fmsg}"
        if nk != nt:
            line += f"  FLEET-COUNT MISMATCH"
        if own_diff:
            line += f"  PLANET-DIFF {own_diff[:3]}"
        if (nk != nt) or own_diff or (common_f and fdiffs[0][0] > 1e-2) or t <= 2:
            print(line)
        # step forward
        kenv.step([[], []])
        tenv.step()
    print(f"  first fleet-position divergence: {'step '+str(first_fleet_div) if first_fleet_div else 'NONE (<1e-2)'}")


def run_full_game_replay(seed, agentA, agentB, max_steps=120):
    """Run a real kaggle game (agentA vs agentB), capture the EXACT per-step action
    stream, then replay those identical actions in torch_env and diff states.
    MATCH => physics is faithful given same actions => the gap is obs/action construction.
    DIVERGE => a physics bug at the first divergent step."""
    print(f"\n===== FULL-GAME REPLAY (seed={seed}, {agentA.split('/')[-1]} vs {agentB.split('/')[-1]}) =====")
    kenv = make("orbit_wars", configuration={"seed": seed, "episodeSteps": max_steps}, debug=False)
    kenv.run([agentA, agentB])
    steps = kenv.steps
    print(f"  kaggle game ran {len(steps)} steps")

    tenv = VecTorchEnv(num_envs=1, num_players=2, device="cpu")
    tenv.reset([seed])

    def kp_of(obs_dict):
        return {int(p[0]): p for p in obs_dict["planets"]}
    def kf_of(obs_dict):
        return {int(f[0]): f for f in obs_dict["fleets"]}

    first_div = None
    for t in range(len(steps) - 1):
        # action applied to steps[t] to produce steps[t+1] is stored in steps[t+1][pid].action
        for pid in (0, 1):
            cell = steps[t + 1][pid]
            act = cell.get("action") if isinstance(cell, dict) else cell.action
            if not act:
                continue
            for mv in act:
                if not mv or len(mv) != 3:
                    continue
                from_id, angle, ships = int(mv[0]), float(mv[1]), int(mv[2])
                # validate like kaggle: planet owned by pid with enough ships
                P = tenv.planets.shape[1]
                slot = next((i for i in range(P) if bool(tenv.planet_alive[0, i])
                             and int(tenv.planets[0, i, 0].item()) == from_id), None)
                if slot is None:
                    continue
                if int(tenv.planets[0, slot, 1].item()) == pid and tenv.planets[0, slot, 5].item() >= ships and ships > 0:
                    torch_inject_launch(tenv, owner=pid, from_id=from_id, angle=angle, ships=ships)
        # stop at the natural game end: kaggle marks status DONE on the terminal step.
        # torch auto-resets on elimination, so comparing past game-end is meaningless.
        nxt = steps[t + 1][0]
        status = nxt.get("status") if isinstance(nxt, dict) else getattr(nxt, "status", "ACTIVE")
        if status and status != "ACTIVE":
            print(f"  >>> GAME ENDED cleanly at step {t+1} (kaggle status={status}); "
                  f"physics FAITHFUL for the full {t+1}-step game incl. comets.")
            first_div = "END"
            break
        tenv.step()
        # compare to kaggle steps[t+1] observation
        kobs = steps[t + 1][0]["observation"] if isinstance(steps[t + 1][0], dict) else steps[t + 1][0].observation
        kobs = kobs if isinstance(kobs, dict) else dict(kobs)
        kp = {int(p[0]): p for p in kobs["planets"]}
        kf = {int(f[0]): f for f in kobs["fleets"]}
        tp = torch_planets_by_id(tenv); tf = torch_fleets_by_id(tenv)
        # planet divergence: ownership or ships
        pdiff = [(i, int(kp[i][1]), int(tp[i][1]), round(kp[i][5],1), round(tp[i][5],1))
                 for i in sorted(set(kp) & set(tp))
                 if int(kp[i][1]) != int(tp[i][1]) or abs(kp[i][5]-tp[i][5]) > 0.5]
        fcount = (len(kf), len(tf))
        fposdiff = 0.0
        for i in set(kf) & set(tf):
            fposdiff = max(fposdiff, math.hypot(kf[i][2]-tf[i][2], kf[i][3]-tf[i][3]))
        diverged = pdiff or abs(fcount[0]-fcount[1]) > 0 or fposdiff > 0.05
        if diverged and first_div is None:
            first_div = t + 1
            print(f"  >>> FIRST DIVERGENCE at step {t+1}:")
            print(f"      kaggle fleets={fcount[0]} torch fleets={fcount[1]}  max fleet Δpos={fposdiff:.3f}")
            if pdiff:
                print(f"      planet diffs (id, k_own, t_own, k_ships, t_ships): {pdiff[:6]}")
            only_t = sorted(set(tf) - set(kf))
            only_k = sorted(set(kf) - set(tf))
            for fid in only_t:
                f = tf[fid]
                print(f"      fleet {fid} in TORCH only: owner={int(f[1])} pos=({f[2]:.2f},{f[3]:.2f}) ang={f[4]:.3f} from={int(f[5])} ships={int(f[6])}")
            for fid in only_k:
                f = kf[fid]
                print(f"      fleet {fid} in KAGGLE only: owner={int(f[1])} pos=({f[2]:.2f},{f[3]:.2f}) ang={f[4]:.3f} from={int(f[5])} ships={int(f[6])}")
            # What did that fleet do the PREVIOUS step? print its prior position from kaggle step t
            pobs = steps[t][0]["observation"] if isinstance(steps[t][0], dict) else steps[t][0].observation
            pobs = pobs if isinstance(pobs, dict) else dict(pobs)
            pkf = {int(f[0]): f for f in pobs["fleets"]}
            for fid in only_t + only_k:
                if fid in pkf:
                    f = pkf[fid]
                    print(f"      fleet {fid} prev-step (kaggle t={t}): pos=({f[2]:.2f},{f[3]:.2f}) ang={f[4]:.3f} ships={int(f[6])}")
            break
    if first_div is None:
        print(f"  >>> NO DIVERGENCE over {len(steps)-1} steps — torch_env physics is FAITHFUL given identical actions.")
        print(f"      => the train/eval gap is NOT the simulator physics; it's OBSERVATION/ACTION construction.")
    return first_div


if __name__ == "__main__":
    for s in [12345, 777, 2024]:
        run_orbital_diff(s, n_steps=25)
    for s in [12345, 777, 2024]:
        run_fleet_diff(s, n_steps=45)
    HAMMER = "opponents/orbit-wars-heuristic-bots/14_main_k_v2_lb1152_LAST_HEURISTIC.py"
    HAMMER2 = "opponents/orbit-wars-heuristic-bots/11_v14_1n_lb1138_doom_evac_mega_hammer.py"
    for s in [12345, 777, 2024, 99, 4242, 8888, 31337, 5]:
        run_full_game_replay(s, HAMMER, HAMMER2, max_steps=260)
