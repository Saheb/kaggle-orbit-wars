"""Teacher BC-ceiling audit.

For each (state, fired-source, chosen-target) sample we ask:

  (A) Label-recovery ambiguity: when bc._find_target_planet_index() decodes the
      teacher's (angle, ships) launch back to a planet index, how close is the
      2nd-best candidate in circular angle? If two planets are within a few
      degrees of bearing, the recovered label may not be what the teacher meant
      — i.e., the BC label itself is noisy.

  (B) Teacher score margin: re-run the teacher's internal candidate scoring for
      the chosen source and compare top-1 vs top-2 score. If many sources have
      a tiny gap, the chosen target is an arbitrary tiebreak — a model that
      picks any near-top planet should be considered "correct", which means
      top1 has a real ceiling below 1.0.

Usage: python audit_teacher.py [--num-games 60]
"""

from __future__ import annotations
import argparse
import math
import sys
from collections import Counter

sys.path.insert(0, ".")

from bc import (
    collect_heuristic_trajectories,
    _find_target_planet_index,
    _bc_fleet_speed,
    _BC_CENTER,
    _ROTATION_LIMIT,
)

import teacher as T


def all_target_angles(src_xy, ship_count, planets, initial_planets,
                      angular_velocity, current_step, max_planets=48):
    """Return list of (planet_idx, predicted_angle_rad) for every planet."""
    sx, sy = src_xy
    speed = _bc_fleet_speed(ship_count)
    init_by_id = {int(p[0]): p for p in initial_planets}
    out = []
    for j, tgt in enumerate(planets[:max_planets]):
        pid = int(tgt[0])
        tx, ty, tr = float(tgt[2]), float(tgt[3]), float(tgt[4])
        ax, ay = tx, ty
        ip = init_by_id.get(pid)
        if ip is not None:
            irx = float(ip[2]) - _BC_CENTER
            iry = float(ip[3]) - _BC_CENTER
            init_angle = math.atan2(iry, irx)
            orbital_r = math.hypot(irx, iry)
            is_orbiting = (orbital_r + tr) < _ROTATION_LIMIT
        else:
            is_orbiting = False
            init_angle, orbital_r = 0.0, 0.0
        for _ in range(4):
            dist = math.hypot(ax - sx, ay - sy)
            eta = max(1, int(math.ceil(dist / speed)))
            if is_orbiting:
                ang = init_angle + angular_velocity * (current_step + eta)
                nax = _BC_CENTER + orbital_r * math.cos(ang)
                nay = _BC_CENTER + orbital_r * math.sin(ang)
            else:
                nax, nay = tx, ty
            if abs(nax - ax) < 0.5 and abs(nay - ay) < 0.5:
                ax, ay = nax, nay
                break
            ax, ay = nax, nay
        out.append((j, math.atan2(ay - sy, ax - sx)))
    return out


def teacher_scores_for_source(obs, src_pid: int):
    """Re-run the teacher's candidate scoring for one source; return sorted
    [(score, tgt_pid, ships, angle), ...] descending."""
    me = int(obs["player"])
    planets = obs["planets"]
    fleets = obs["fleets"]
    initial_planets = obs.get("initial_planets") or planets
    angular_velocity = float(obs.get("angular_velocity", 0.0))
    current_step = int(obs["step"])

    orbit_table = T._build_orbit_table(initial_planets)
    arrivals = T._project_arrivals(fleets, planets)
    by_id = {int(p[0]): p for p in planets}

    src_p = by_id[src_pid]
    sx, sy, src_radius = float(src_p[2]), float(src_p[3]), float(src_p[4])
    src_owner, src_ships, src_prod = int(src_p[1]), int(src_p[5]), float(src_p[6])

    inbound_enemy = [a for a in arrivals.get(src_pid, []) if a[2] != me]
    reserve = T._defense_reserve(inbound_enemy, src_prod)
    available = src_ships - reserve
    if available < T.MIN_SHIP_FLOOR:
        return []

    def already_winning(tgt_pid: int) -> bool:
        ours = sorted([a for a in arrivals.get(tgt_pid, []) if a[2] == me])
        if not ours:
            return False
        tgt = by_id[tgt_pid]
        tgt_ships, tgt_owner, tgt_prod = float(tgt[5]), int(tgt[1]), float(tgt[6])
        my_eta = ours[0][0]
        my_total = sum(a[1] for a in ours if a[0] <= my_eta + 2)
        their_total = sum(a[1] for a in arrivals.get(tgt_pid, [])
                          if a[2] != me and a[0] <= my_eta + 2)
        needed = tgt_ships + (tgt_prod if tgt_owner != -1 else 0.0) * my_eta
        return (my_total - their_total) > needed

    out = []
    for tgt_p in planets:
        tgt_pid = int(tgt_p[0])
        if tgt_pid == src_pid:
            continue
        tgt_owner = int(tgt_p[1])
        if tgt_owner == me:
            continue
        tgt_x, tgt_y = float(tgt_p[2]), float(tgt_p[3])
        tgt_ships, tgt_prod = float(tgt_p[5]), float(tgt_p[6])
        if already_winning(tgt_pid):
            continue
        ships_try = max(T.MIN_SHIP_FLOOR, min(available, 30))
        eta, aim_x, aim_y = T._solve_eta(
            (sx, sy), tgt_pid, ships_try, current_step,
            orbit_table, angular_velocity, (tgt_x, tgt_y),
        )
        need = T._capture_cost(tgt_ships, tgt_prod, tgt_owner, eta)
        ships_final = min(available, max(T.MIN_SHIP_FLOOR, need))
        if ships_final < T.MIN_SHIP_FLOOR or ships_final < T.MIN_COVERAGE_FRAC * need:
            continue
        angle = math.atan2(aim_y - sy, aim_x - sx)
        start_x = sx + math.cos(angle) * (src_radius + 0.1)
        start_y = sy + math.sin(angle) * (src_radius + 0.1)
        if T._segment_crosses_sun(start_x, start_y, aim_x, aim_y):
            continue
        k = 5.0
        neutral_bonus = 1.2 if tgt_owner == -1 else 1.0
        score = (tgt_prod * neutral_bonus) / (eta + k)
        out.append((score, tgt_pid, ships_final, angle))
    out.sort(key=lambda c: -c[0])
    return out


def circ_diff(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 2 * math.pi - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-games", type=int, default=60)
    args = ap.parse_args()

    print(f"Collecting {args.num_games} games of teacher vs random...")
    trajs = collect_heuristic_trajectories(
        "teacher.py", num_games=args.num_games, opponent="random", verbose=False,
    )
    print(f"  {len(trajs)} transitions (turns with at least one move)\n")

    # Aggregators
    n_moves = 0
    n_decoded = 0
    n_decode_fail = 0
    n_decode_ambig_5deg = 0       # 2nd-best within 5° of chosen
    n_decode_ambig_15deg = 0      # 2nd-best within 15°
    angle_gap_2nd = []            # circular angle gap from chosen to 2nd-best (deg)

    n_score_solo = 0              # only 1 valid candidate from this source
    score_gap_pct = []            # (top1 - top2) / top1  *  100   (when ≥2 cands)
    teacher_top1_position = Counter()  # position of teacher's recovered tgt in score-ranked list

    for tr in trajs:
        obs = tr["obs"]
        planets = obs["planets"]
        initial_planets = obs.get("initial_planets") or planets
        angular_velocity = float(obs.get("angular_velocity", 0.0))
        current_step = int(obs["step"])
        max_planets = 48
        pid_to_idx = {int(p[0]): i for i, p in enumerate(planets[:max_planets])}

        for mv in tr["action"]:
            src_pid = int(mv[0])
            angle_rad = float(mv[1])
            ships = int(mv[2])
            src_p = next((p for p in planets if int(p[0]) == src_pid), None)
            if src_p is None:
                continue
            n_moves += 1
            src_xy = (float(src_p[2]), float(src_p[3]))

            # ---- (A) decode ambiguity ----
            all_ang = all_target_angles(src_xy, ships, planets, initial_planets,
                                        angular_velocity, current_step, max_planets)
            errs = sorted([(circ_diff(pa, angle_rad), j) for j, pa in all_ang])
            best_err, best_idx = errs[0]
            second_err = errs[1][0] if len(errs) > 1 else float("inf")

            if best_err > math.radians(15.0):
                n_decode_fail += 1
            else:
                n_decoded += 1
                gap = second_err - best_err
                angle_gap_2nd.append(math.degrees(gap))
                if gap < math.radians(5.0):
                    n_decode_ambig_5deg += 1
                if gap < math.radians(15.0):
                    n_decode_ambig_15deg += 1

            # ---- (B) teacher score margin ----
            cands = teacher_scores_for_source(obs, src_pid)
            if len(cands) == 0:
                continue
            if len(cands) == 1:
                n_score_solo += 1
                continue
            top1 = cands[0][0]
            top2 = cands[1][0]
            gap_pct = (top1 - top2) / max(top1, 1e-9) * 100.0
            score_gap_pct.append(gap_pct)

            # Where did the teacher's recovered target land in the score ranking?
            if best_err <= math.radians(15.0):
                recovered_pid = int(planets[best_idx][0])
                for rank, c in enumerate(cands):
                    if c[1] == recovered_pid:
                        teacher_top1_position[rank] += 1
                        break
                else:
                    teacher_top1_position["not-in-cands"] += 1

    # ---- report ----
    def pct(n, d):
        return 100.0 * n / max(d, 1)

    print("=" * 70)
    print("(A) LABEL-RECOVERY AMBIGUITY")
    print("=" * 70)
    print(f"Total teacher moves:           {n_moves}")
    print(f"Decoded (best <15°):           {n_decoded}  ({pct(n_decoded, n_moves):.1f}%)")
    print(f"Decode failed (best >15°):     {n_decode_fail}  ({pct(n_decode_fail, n_moves):.1f}%)")
    print(f"Decoded but ambiguous <5°:     {n_decode_ambig_5deg}  ({pct(n_decode_ambig_5deg, n_decoded):.1f}% of decoded)")
    print(f"Decoded but ambiguous <15°:    {n_decode_ambig_15deg}  ({pct(n_decode_ambig_15deg, n_decoded):.1f}% of decoded)")
    if angle_gap_2nd:
        angle_gap_2nd.sort()
        n = len(angle_gap_2nd)
        print(f"Angle gap to 2nd-best (deg):  p10={angle_gap_2nd[n//10]:.1f}  "
              f"p25={angle_gap_2nd[n//4]:.1f}  p50={angle_gap_2nd[n//2]:.1f}  "
              f"p75={angle_gap_2nd[3*n//4]:.1f}")

    print()
    print("=" * 70)
    print("(B) TEACHER SCORE-MARGIN (chosen vs 2nd-best target from same source)")
    print("=" * 70)
    print(f"Sources with only 1 candidate (trivial choice): {n_score_solo}  "
          f"({pct(n_score_solo, n_moves):.1f}% of moves)")
    if score_gap_pct:
        sg = sorted(score_gap_pct)
        n = len(sg)
        print(f"Score gap (top1 - top2) / top1, when ≥2 candidates  (n={n}):")
        print(f"  p10={sg[n//10]:.1f}%  p25={sg[n//4]:.1f}%  p50={sg[n//2]:.1f}%  "
              f"p75={sg[3*n//4]:.1f}%  p90={sg[9*n//10]:.1f}%")
        below_5 = sum(1 for x in sg if x < 5.0)
        below_10 = sum(1 for x in sg if x < 10.0)
        print(f"  Fraction with gap <5%:  {pct(below_5, n):.1f}%   (near-tie)")
        print(f"  Fraction with gap <10%: {pct(below_10, n):.1f}%")

    print()
    print("Recovered-label's rank in teacher's own score order:")
    total_rank = sum(v for k, v in teacher_top1_position.items() if isinstance(k, int))
    for rank in sorted(k for k in teacher_top1_position if isinstance(k, int))[:8]:
        n = teacher_top1_position[rank]
        print(f"  rank {rank}:  {n}  ({pct(n, total_rank):.1f}%)")
    if "not-in-cands" in teacher_top1_position:
        print(f"  not-in-cands: {teacher_top1_position['not-in-cands']}")

    print()
    print("Implied BC top1 ceiling:")
    if total_rank:
        rank0 = teacher_top1_position.get(0, 0)
        print(f"  If model perfectly mimics teacher and labels were perfect, top1 = {pct(rank0, total_rank):.1f}%")
        print(f"  (Lower = label noise + score-tie ambiguity reduce the achievable top1.)")


if __name__ == "__main__":
    main()
