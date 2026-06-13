"""Comet-aware feature-parity probe: VecTorchEnv.get_features() vs features.extract_features().

STEP 1 of the post-comet-fix sequence. The comet PHYSICS fix is byte-validated (sim_gap_probe),
but byte-replay feeds fixed actions — it never exercises the OBSERVATION pipeline (get_features).
This probe closes that seam: take ONE torch_env state (past step 50, so comets are live planets in
slots 44-47), run BOTH feature extractors on it, and diff every feature element-wise, aligned by
planet/fleet id (NOT row index — comets sit after dead padding slots, so row order differs).

Decisive because the reference obs is given the TRUE comet ids (to_legacy_obs hardcodes []). So:
  - extract_features (kaggle-faithful)  -> is_comet=1 on comet planets
  - get_features (training path)         -> is_comet=0 (torch_env.py:716 stub)
Expected verdict: planet feature 7 (is_comet) is the ONLY divergence; everything else — incl.
pairwise (roi/contest/cap_gap on comets-as-neutral-planets) and global — matches within float tol.
That confirms the obs gap is exactly is_comet (+ no hidden pairwise breakage from comets), which is
what the feature-fix phase then closes.

Run: orbit_wars_rl/.venv/bin/python orbit_wars_rl/feature_parity_comet_probe.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import extract_features
from torch_env import (VecTorchEnv, to_legacy_obs, MAX_OWNED, MAX_PLANETS,
                       COMET_SLOT_START, N_COMET_SLOTS)

TOL = 0.05  # generous — float32 pairwise/trig precision differs between the np and torch ports


def true_comet_ids(env, e):
    """Real comet planet ids for env e: alive planets in the comet slots [44, 48)."""
    ids = []
    for s in range(COMET_SLOT_START, COMET_SLOT_START + N_COMET_SLOTS):
        if bool(env.planet_alive[e, s]):
            ids.append(int(env.planets[e, s, 0].item()))
    return ids


def id_to_row_ref(obs):
    """planet id -> row index in extract_features output (compacted alive order)."""
    return {int(p[0]): j for j, p in enumerate(obs["planets"])}


def id_to_row_vec(env, e):
    """planet id -> raw slot index in get_features output (slot order, dead slots zeroed)."""
    out = {}
    for s in range(MAX_PLANETS):
        if bool(env.planet_alive[e, s]):
            out[int(env.planets[e, s, 0].item())] = s
    return out


def compare_planets(vec_pf, ref_pf, ids, v_row, r_row):
    """Return per-column max abs diff over shared planet ids. (20 columns.)"""
    col_max = np.zeros(20, dtype=np.float64)
    col_cnt = np.zeros(20, dtype=np.int64)
    for pid in ids:
        d = np.abs(vec_pf[v_row[pid]] - ref_pf[r_row[pid]])
        over = d > TOL
        col_max = np.maximum(col_max, d)
        col_cnt += over.astype(np.int64)
    return col_max, col_cnt


def main():
    seeds = [0, 1, 2, 3]
    after_steps = 80  # > 50 so the first comet group (spawn @50) is live
    env = VecTorchEnv(num_envs=len(seeds), num_players=2, device="cpu")
    env.reset(seeds=seeds)
    for _ in range(after_steps):
        env.step({0: torch.randint(0, 2, (len(seeds), MAX_OWNED, 3)),
                  1: torch.randint(0, 2, (len(seeds), MAX_OWNED, 3))})

    print("=" * 78)
    print(f"COMET-AWARE FEATURE PARITY  (after {after_steps} steps, seeds={seeds})")
    print("=" * 78)

    n_comet_states = 0
    planet_col_max = np.zeros(20)
    planet_col_cnt = np.zeros(20, dtype=np.int64)
    fleet_col_max = np.zeros(13)
    fleet_col_cnt = np.zeros(13, dtype=np.int64)
    global_max = np.zeros(11)
    global_cnt = np.zeros(11, dtype=np.int64)
    pair_col_max = np.zeros(15)
    pair_col_cnt = np.zeros(15, dtype=np.int64)
    comet_is_comet_checked = 0

    for player in (0, 1):
        vec = env.get_features(player, max_planets=48, max_fleets=128)
        for e in range(len(seeds)):
            cids = true_comet_ids(env, e)
            if cids:
                n_comet_states += 1
            obs = to_legacy_obs(env, env_idx=e, player=player)  # now surfaces comets + ids
            ref = extract_features(obs, player, num_players=2, max_planets=48, max_fleets=128)

            ids = sorted(set(id_to_row_ref(obs)) & set(id_to_row_vec(env, e)))
            v_row = id_to_row_vec(env, e)
            r_row = id_to_row_ref(obs)

            # --- planets (20) ---
            cm, cc = compare_planets(vec["planet_features"][e].numpy(),
                                     ref["planet_features"].numpy(), ids, v_row, r_row)
            planet_col_max = np.maximum(planet_col_max, cm)
            planet_col_cnt += cc
            # explicit is_comet sanity on comet planets
            for pid in cids:
                vi = vec["planet_features"][e].numpy()[v_row[pid], 7]
                ri = ref["planet_features"].numpy()[r_row[pid], 7]
                comet_is_comet_checked += 1
                if e == 0 and player == 0:
                    print(f"  comet pid={pid}: is_comet vec={vi:.1f} ref={ri:.1f}")

            # --- fleets (13): align by fleet id ---
            f = env.fleets[e].cpu().numpy(); fa = env.fleet_alive[e].cpu().numpy()
            vrow_f = {int(f[s, 0]): s for s in range(f.shape[0]) if fa[s]}
            rrow_f = {int(fl[0]): j for j, fl in enumerate(obs["fleets"])}
            ff_v = vec["fleet_features"][e].numpy(); ff_r = ref["fleet_features"].numpy()
            for fid in set(vrow_f) & set(rrow_f):
                d = np.abs(ff_v[vrow_f[fid]] - ff_r[rrow_f[fid]])
                fleet_col_max = np.maximum(fleet_col_max, d)
                fleet_col_cnt += (d > TOL).astype(np.int64)

            # --- global (11) ---
            gd = np.abs(vec["global_features"][e].numpy() - ref["global_features"].numpy())
            global_max = np.maximum(global_max, gd)
            global_cnt += (gd > TOL).astype(np.int64)

            # --- pairwise (max_owned, max_planets, 15): align src & tgt by id ---
            pw_v = vec["pairwise_features"][e].numpy()
            pw_r = ref["pairwise_features"].numpy()
            v_oidx = vec["owned_indices"][e].numpy()
            v_ocnt = vec["owned_count"][e]
            r_oidx = ref["owned_indices"].numpy()
            r_ocnt = ref["owned_count"]
            # source id -> owned-slot, each side
            v_src = {int(env.planets[e, int(v_oidx[k]), 0].item()): k for k in range(v_ocnt)}
            r_src = {int(obs["planets"][int(r_oidx[k])][0]): k for k in range(r_ocnt)}
            for sid in set(v_src) & set(r_src):
                vk, rk = v_src[sid], r_src[sid]
                for pid in ids:
                    d = np.abs(pw_v[vk, v_row[pid]] - pw_r[rk, r_row[pid]])
                    pair_col_max = np.maximum(pair_col_max, d)
                    pair_col_cnt += (d > TOL).astype(np.int64)

    def report(name, col_max, col_cnt, labels=None):
        bad = [(i, col_max[i], int(col_cnt[i])) for i in range(len(col_max)) if col_cnt[i] > 0]
        print(f"\n{name}: {'CLEAN' if not bad else 'DIVERGES'}")
        for i, m, c in bad:
            lbl = f" ({labels[i]})" if labels and i < len(labels) else ""
            print(f"    feat[{i}]{lbl}: max|Δ|={m:.4f}  n_over_tol={c}")

    planet_labels = ["x", "y", "owner", "radius", "log_ships", "prod", "is_orbiting",
                     "IS_COMET", "dist_sun", "orb_r", "pred_x", "pred_y", "friend_press",
                     "enemy_press", "cap_cost", "min_owned_dist", "is_home", "conn15",
                     "conn30", "active"]
    pair_labels = ["sin", "cos", "dist", "1/(eta+1)", "sun_safe", "is_mine", "is_enemy",
                   "is_neutral", "tgt_prod", "valid", "ships@arr", "cap_gap", "roi20",
                   "roi50", "enemy_contest"]
    print(f"\ncomet-bearing (env,player) states: {n_comet_states}/8   "
          f"is_comet checks on comet planets: {comet_is_comet_checked}")
    report("PLANET (20)", planet_col_max, planet_col_cnt, planet_labels)
    report("FLEET (13)", fleet_col_max, fleet_col_cnt)
    report("GLOBAL (11)", global_max, global_cnt)
    report("PAIRWISE (15)", pair_col_max, pair_col_cnt, pair_labels)

    # planet feat[15] (min_owned_dist) has a PRE-EXISTING, comet-independent divergence: torch
    # caps it at BOARD_SIZE while extract_features reports true distances >BOARD_SIZE on sparse
    # boards. Present before the comet fix; excluded from the comet-parity verdict (tracked separately).
    planet_cnt_comet = planet_col_cnt.copy(); planet_cnt_comet[15] = 0
    any_bad = any(c > 0 for c in
                  list(planet_cnt_comet) + list(fleet_col_cnt) + list(global_cnt) + list(pair_col_cnt))
    print("\n" + "=" * 78)
    if not any_bad:
        print("VERDICT: CLEAN — train (get_features) and eval (extract_features) obs pipelines")
        print("MATCH on comet states, including is_comet, the path-aware comet position/expiry")
        print("channels, and pairwise. The comet obs-pipeline gap is closed.")
        if planet_col_cnt[15] > 0:
            print("(planet feat[15] min_owned_dist shows a PRE-EXISTING, comet-independent residual")
            print(" — torch caps at BOARD_SIZE on sparse boards; tracked separately, not a comet gap.)")
    else:
        print("VERDICT: DIVERGES — an obs-pipeline gap remains (see flagged columns). Expected")
        print("CLEAN after the comet feature fix; investigate any column with n_over_tol > 0.")
        print("(feat[15] min_owned_dist may show a tiny low-count residual unrelated to comets.)")
    print("=" * 78)


if __name__ == "__main__":
    main()
