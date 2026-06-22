"""V(s)-vs-spare-opportunity diagnostic (2026-06-21).

WHY: the disposition decomposition on policy-short contested launches (vs Ajay)
showed 80% of spare sources are IDLE when they could pile on a contested neutral.
Two candidate root causes:

  (a) CRITIC bottleneck: V(s) is flat across fire-spare / don't-fire-spare states.
      Then the PPO advantage A = r + gamma*V(s') - V(s) carries NO gradient toward
      firing the spare, and the fire head defaults to its conservative "don't fire"
      prior. Fix: attack the DATA (permanent external aggregating opponent so the
      critic learns that sitting idle = losing), not the policy.

  (b) POLICY bottleneck: V(s) correctly prices the spare-launch state higher, but
      the fire head is stuck in a low-entropy basin and won't fire regardless.
      Fix: crank entropy_coef_fire, or add a per-step bonus for firing a spare at
      a takeable neutral.

This script distinguishes (a) from (b) by recording, for every spare-opportunity
state in steps 10-50, V(s_t) and whether the policy actually fired a spare source
at that neutral. If mean V(s) | fired ~= mean V(s) | didn't-fire (within outcome)
-> critic is flat -> bottleneck (a). If V separates -> policy bottleneck (b).

Reuses expansion_autopsy._classify_neutrals for the spare-source floor (so the
opportunity set matches the panel decomposition), and value_analysis's
V-tracking agent structure.

Run from repo ROOT:
  orbit_wars_rl/.venv/bin/python orbit_wars_rl/value_spare_diagnostic.py \
      --checkpoint <ck.pt> \
      --opponent opponents/candidate_ajay_1200.py \
      --seeds 24 --output /tmp/v_spare.png
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from kaggle_environments import make

import eval as E
from transition_autopsy import _load_model, _obs_to_dict
from expansion_autopsy import _classify_neutrals  # noqa: F401  (parity-validated floor)
from action_mask import compute_action_masks, _head_threat_maps, _ship_bin_to_count
from features import extract_features

# PBRS pre-test: γ for the single-step potential term γΦ(s')−Φ(s). γ≈1, so the
# discount is negligible for one step; we report the raw potential rise Φ'−Φ.
_GAMMA = 0.995


# Steps where the expansion race is decided (matches expansion_autopsy.SAMPLES early tail).
STEP_LO, STEP_HI = 10, 50
# max_eta for spare-source reach (matches _classify_neutrals default).
_MAX_ETA = 25


def _spare_sources_for_neutral(neutral, planets, fleets, seat):
    """Owned planets that (i) can reach `neutral` within _MAX_ETA and (ii) hold
    spare garrison (ships - own enemy-inbound threat > 0). Returns list of
    (planet_idx_in_obs, spare_ships, cost_to_capture). Mirrors the reach-set
    construction inside expansion_autopsy._classify_neutrals so the opportunity
    definition matches the panel."""
    out = []
    nx, ny = float(neutral[2]), float(neutral[3])
    for pidx, p in enumerate(planets):
        if int(p[1]) != seat:
            continue
        dist = math.hypot(nx - float(p[2]), ny - float(p[3]))
        eta = max(1.0, math.ceil(dist / E._ETA_PROBE_SPEED))
        if eta > _MAX_ETA:
            continue
        threat, _ = E._enemy_threat(planets, fleets, p, seat)
        spare = max(0.0, float(p[5]) - threat)
        if spare <= 0.0:
            continue
        cost = E._cap_cost_at_arrival(p, neutral, seat)
        out.append((pidx, spare, cost))
    return out


def _decode_slot_targets(outputs, masks, obs, player, allow_reinforce, ship_bin_mode="absolute"):
    """Mirror actions_from_target_policy's per-slot (target, fire) decode so the
    (source -> target) pairing is EXACTLY what the policy decided. Returns
    dict: slot_idx -> {"fired": bool, "target_planet_idx": int|None,
                       "target_id": int|None, "target_owner": int|None}.

    We deliberately do NOT apply the garrison-floor / sufficient-commit /
    reverse-cooldown post-vetoes here — those veto launches, but the question is
    what the policy WANTED to fire. Vetoed launches are still "intent to fire"
    and count as the spare source NOT being idle. (The 80%-idle decomposition
    used the same floor.)"""
    planets = obs["planets"]
    owned_indices = masks["owned_indices"].cpu().numpy()
    target_logits = outputs["target_logits"].cpu()
    fire_logits = outputs["fire_logits"].cpu()
    owned_count = int(masks["owned_count"])

    # Mask illegal targets exactly like actions_from_target_policy (lines 868-879):
    # source self + own-planet-if-no-reinforce.
    for slot in range(min(owned_count, target_logits.shape[1])):
        pidx = int(owned_indices[slot])
        if pidx >= len(planets):
            continue
        for tidx, tgt in enumerate(planets[:target_logits.shape[-1]]):
            is_source = int(tgt[0]) == int(planets[pidx][0])
            is_own = int(tgt[1]) == player
            illegal = is_source or (is_own and not allow_reinforce)
            if illegal:
                target_logits[:, slot, tidx] = -1e9

    target_indices = torch.argmax(target_logits, dim=-1).squeeze(0).numpy()
    # fire deciscion = sigmoid(fire_logit_at_chosen_target) > 0.5 (eval default
    # fire_threshold=0.5; matches build_agent_fn).
    chosen_fire_logits = torch.gather(
        fire_logits, -1,
        torch.as_tensor(target_indices).unsqueeze(0).unsqueeze(-1)
    ).squeeze(-1).squeeze(0)
    fire_decisions = (torch.sigmoid(chosen_fire_logits) > 0.5).numpy()

    # ship-size decode: gather ship_logits at the chosen target, argmax bin, then
    # _ship_bin_to_count — EXACTLY as actions_from_target_policy (action_mask.py:925-942).
    # Lets us split fired-spare into 1-ship probes vs floor-reaching sends, and check
    # whether the source had the garrison to send big (the min-ship-bin headroom test).
    ship_logits = outputs["ship_logits"].cpu()  # [1, slots, targets, bins]
    max_ships = masks["max_ships"].cpu().numpy().squeeze(0)
    tgt_idx_t = torch.as_tensor(target_indices).unsqueeze(0)  # [1, slots]
    chosen_ship_logits = torch.gather(
        ship_logits, 2,
        tgt_idx_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, ship_logits.shape[-1]),
    ).squeeze(2).squeeze(0)  # [slots, bins]
    ship_bins = torch.argmax(chosen_ship_logits, dim=-1).numpy()

    slot_map = {}
    for slot in range(min(owned_count, target_indices.shape[0])):
        pidx = int(owned_indices[slot])
        if pidx >= len(planets):
            slot_map[slot] = {"fired": False, "target_planet_idx": None,
                              "target_id": None, "target_owner": None,
                              "src_planet_idx": pidx, "ships": 0, "max_ships": 0}
            continue
        tidx = int(target_indices[slot])
        tgt_ok = 0 <= tidx < len(planets)
        ms = int(max_ships[slot])
        slot_map[slot] = {
            "fired": bool(fire_decisions[slot]),
            "target_planet_idx": tidx if tgt_ok else None,
            "target_id": int(planets[tidx][0]) if tgt_ok else None,
            "target_owner": int(planets[tidx][1]) if tgt_ok else None,
            "src_planet_idx": pidx,
            "ships": int(_ship_bin_to_count(int(ship_bins[slot]), ms, mode=ship_bin_mode)),
            "max_ships": ms,
        }
    return slot_map


def build_tracking_agent(model, device, target_decode, ship_bin_mode, allow_reinforce,
                         step_records):
    """Like eval.build_agent_fn but records per-step:
       {"step", "v_s", "planets", "fleets", "slot_targets"} into step_records."""
    model.eval()

    def agent_fn(obs):
        obs = _obs_to_dict(obs)
        player = obs["player"]
        features = extract_features(obs, player, num_players=2)
        masks = compute_action_masks(obs, player)
        with torch.no_grad():
            outputs = model(
                features["planet_features"].unsqueeze(0).to(device),
                features["fleet_features"].unsqueeze(0).to(device),
                features["global_features"].unsqueeze(0).to(device),
                features["planet_mask"].unsqueeze(0).to(device),
                features["fleet_mask"].unsqueeze(0).to(device),
                fire_mask=masks["fire_mask"].to(device),
                angle_mask=masks["angle_mask"].to(device),
                slot_valid=masks["slot_valid"].to(device),
                owned_indices=masks["owned_indices"].to(device),
                owned_count=masks["owned_count"],
                pairwise_features=features["pairwise_features"].unsqueeze(0).to(device)
                    if "pairwise_features" in features else None,
            )
        v_s = float(outputs["value"][0].cpu())
        slot_targets = _decode_slot_targets(outputs, masks, obs, player, allow_reinforce,
                                            ship_bin_mode=ship_bin_mode)

        step_records.append({
            "step": int(obs["step"]),
            "v_s": v_s,
            "planets": obs["planets"],
            "fleets": obs["fleets"],
            "slot_targets": slot_targets,
            "owned_indices": masks["owned_indices"].cpu().numpy(),
            "owned_count": int(masks["owned_count"]),
        })

        # Use the real eval agent for the actual moves so the game proceeds at
        # eval strength (with all post-vetoes). Re-derive from outputs/masks.
        action_fn = E.actions_from_target_policy if target_decode else E.actions_from_policy
        if not target_decode:
            raise NotImplementedError("angle-decode path removed; use target_decode.")
        moves = action_fn(
            outputs["fire_logits"].cpu(), outputs["target_logits"].cpu(),
            outputs["ship_logits"].cpu(),
            {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
            obs, player, ship_bin_mode=ship_bin_mode,
            allow_reinforce=allow_reinforce,
            reinforce_gate_min_planets=getattr(model, "reinforce_gate_min_planets", 0),
            reinforce_forward_only=getattr(model, "reinforce_forward_only", False),
            reinforce_garrison_floor=getattr(model, "reinforce_garrison_floor", 0.0),
            sufficient_commit_factor=getattr(model, "sufficient_commit_factor", 0.0),
            reverse_edge_cooldown=int(getattr(model, "reverse_edge_cooldown", 0)),
        )
        return moves

    return agent_fn


def _scan_opportunities(step_records, seat):
    """For each step in [STEP_LO, STEP_HI], enumerate contested neutrals that have
    >=1 spare source (the af1 + agg buckets), and record:
       (step, v_s, neutral_id, fired_at_neutral, n_spare_sources, bucket)
    where fired_at_neutral = any spare source's slot decided to fire at this neutral.

    A spare source is "idle on this neutral" if its slot either didn't fire at all
    or fired at a different target. The user's 80%-idle number counts idle sources;
    here we aggregate to the neutral level (did ANY spare source pile on?)."""
    rows = []
    for rec in step_records:
        t = rec["step"]
        if t < STEP_LO or t > STEP_HI:
            continue
        planets = rec["planets"]
        fleets = rec["fleets"]
        v_s = rec["v_s"]
        slot_targets = rec["slot_targets"]
        owned_indices = rec["owned_indices"]
        owned_count = rec["owned_count"]

        # Build slot lookup by src planet_idx for quick "what did source P do".
        by_src = {info["src_planet_idx"]: info
                  for info in slot_targets.values()}
        # Friendly inbound per planet-id (already-staged mass) for the PBRS potential.
        _, _, fr_in = _head_threat_maps(planets, fleets, seat)

        neutrals = [(pi, p) for pi, p in enumerate(planets) if int(p[1]) < 0]
        for nidx, neutral in neutrals:
            sources = _spare_sources_for_neutral(neutral, planets, fleets, seat)
            if not sources:
                continue
            total_spare = sum(s for _, s, _ in sources)
            cheapest_cost = min(c for _, _, c in sources)
            # bucket: af1 = single source can solo; agg = needs pooling; either way
            # a spare source exists (this is the "opportunity" set).
            solo = any(s >= c for _, s, c in sources)
            bucket = "af1" if solo else ("agg" if total_spare >= cheapest_cost else "cap")
            if bucket == "cap":
                continue  # no amount of pooling captures it; not an opportunity

            # Did any spare source's slot fire at THIS neutral? If so, capture that
            # source's send size, its solo capture-cost (the floor), and its garrison
            # (max_ships) — for the min-ship-bin headroom test downstream.
            fired_at = False
            n_id = int(neutral[0])
            fired_ships = fired_cost = fired_maxships = None
            for pidx, spare, src_cost in sources:
                info = by_src.get(pidx)
                if info is None:
                    continue
                if info["fired"] and info["target_id"] == n_id:
                    fired_at = True
                    fired_ships = int(info["ships"])
                    fired_cost = float(src_cost)
                    fired_maxships = int(info["max_ships"])
                    break

            # ---- PBRS counterfactual: the per-target potential rise firing the spare
            # WOULD inject, at exactly this A≈0 spot. Φ = min(1, inbound/floor), capped.
            # This is the gradient PBRS adds; the actual (idle) trajectory has Φ'−Φ≈0,
            # so we must compute the COUNTERFACTUAL (send the pooled spare to floor), not
            # the realized step — else we'd wrongly conclude the lever is dead.
            floor = max(cheapest_cost, 1.0)
            fr_inb = sum(s for s, _e in fr_in.get(n_id, []))
            phi_before = min(1.0, fr_inb / floor)
            send = min(total_spare, max(0.0, floor - fr_inb))   # send just enough to reach floor
            phi_after = min(1.0, (fr_inb + send) / floor)
            shaping_raw = _GAMMA * phi_after - phi_before        # single-step PBRS term (pre-α)

            rows.append({
                "step": t,
                "v_s": v_s,
                "neutral_id": n_id,
                "fired_at": fired_at,
                "n_spare_sources": len(sources),
                "bucket": bucket,
                "shaping_raw": shaping_raw,
                "fired_ships": fired_ships,
                "fired_cost": fired_cost,
                "fired_maxships": fired_maxships,
            })
    return rows


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _stderr(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def _pbrs_pretest(all_rows, alpha):
    """Does PBRS inject a gradient where the diagnostic just showed A≈0? Reports the
    counterfactual per-target potential rise Φ'−Φ (from firing the pooled spare to floor)
    at every spare-opportunity, scaled by α, vs the measured advantage scale. Also reports
    the SUM-vs-MAX surface (opportunities per state) to decide Φ aggregation: many big-gain
    targets per state -> Σ favours spread (use top-k/max for pile-on); few -> Σ is safe."""
    sh = [r["shaping_raw"] for r in all_rows]
    sh_af1 = [r["shaping_raw"] for r in all_rows if r["bucket"] == "af1"]
    sh_agg = [r["shaping_raw"] for r in all_rows if r["bucket"] == "agg"]
    vw = [r["v_s"] for r in all_rows if r["outcome"] == "won"]
    vl = [r["v_s"] for r in all_rows if r["outcome"] == "lost"]
    v_spread = _mean(vw) - _mean(vl)
    # opportunities per (seed,seat,step) state = Σ-aggregation breadth.
    per_state = defaultdict(int)
    for r in all_rows:
        per_state[(r["seed"], r["seat"], r["step"])] += 1
    opp_per_state = _mean(list(per_state.values()))

    print("\n" + "=" * 92)
    print(f"PBRS PRE-TEST  (α={alpha})  — does shaping inject the missing gradient?")
    print("=" * 92)
    print(f"  counterfactual potential rise Φ'−Φ from firing pooled spare to floor:")
    print(f"    all opportunities : mean {_mean(sh):+.3f}  (n={len(sh)})")
    print(f"    af1 (solo-takeable): mean {_mean(sh_af1):+.3f}  (n={len(sh_af1)})")
    print(f"    agg (needs pooling): mean {_mean(sh_agg):+.3f}  (n={len(sh_agg)})")
    print(f"  α·(Φ'−Φ) added advantage @α={alpha}: mean {alpha*_mean(sh):+.3f} per staging step")
    print(f"  reference scales: selection ΔV ≈ −0.13..−0.17 (the noise floor) | "
          f"V(won)−V(lost) spread = {v_spread:+.2f}")
    print(f"  Σ-vs-MAX surface: mean {opp_per_state:.1f} contestable targets per state "
          f"(high → Σ rewards spread → prefer top-k/max; low → Σ safe)")
    pulls = alpha * _mean(sh) > 0.05
    print(f"\n  VERDICT: {'LEVER PULLS' if pulls else 'lever weak'} — α·shaping "
          f"{'≫' if pulls else '≈/<'} the A≈0 floor the diagnostic measured "
          f"→ {'PBRS injects the gradient. Launch with α-sweep.' if pulls else 'raise α or refix Φ.'}")
    print("  (caveat: shaping is positive by construction at below-floor targets; the test")
    print("   confirms the gradient EXISTS and calibrates α — it is not a Nash-vs-fixedpoint test.)")


def _ship_size_split(all_rows):
    """min-ship-bin headroom test. Of the fired-spare events, how many are 1-ship /
    sub-min-bin probes, and CRUCIALLY did those probes come from sources that HAD the
    garrison to send a floor-reaching amount but chose a probe anyway?

      - probe (<=4 ships): min-ship-bin 4 would force these up.
      - of probes, 'had-mass' = max_ships >= solo capture-cost: the source could have
        soloed the neutral but sent a probe -> min-ship-bin has real headroom.
      - 'garrison-limited' = max_ships < cost: clamps to garrison anyway, min-ship-bin
        can't manufacture ships -> no help (this is an AGGREGATION problem, not size).

    Decision: min-ship-bin helps idleness->mass conversion only if a large share of
    fired-spare are had-mass probes (esp. in the af1/solo bucket). If probes are mostly
    garrison-limited or on agg neutrals (need pooling), min-ship-bin is dead weight."""
    fired = [r for r in all_rows if r["fired_at"] and r.get("fired_ships") is not None]
    print("\n" + "=" * 92)
    print("SHIP-SIZE SPLIT of fired-spare  (min-ship-bin headroom test)")
    print("=" * 92)
    if not fired:
        print("  no fired-spare events with ship-size recorded.")
        return
    n = len(fired)
    ships = [r["fired_ships"] for r in fired]
    probes = [r for r in fired if r["fired_ships"] <= 4]       # min-ship-bin 4 target
    one_ship = [r for r in fired if r["fired_ships"] == 1]
    reached = [r for r in fired if r["fired_ships"] >= r["fired_cost"]]  # real solo-capture send
    had_mass_probe = [r for r in probes if r["fired_maxships"] >= r["fired_cost"]]
    garr_lim_probe = [r for r in probes if r["fired_maxships"] < r["fired_cost"]]

    def pct(x):
        return 100.0 * len(x) / n if n else 0.0

    print(f"  fired-spare events: {n}   mean ship-size {_mean(ships):.1f}  "
          f"(median {sorted(ships)[n//2]})")
    print(f"    1-ship probes     : {len(one_ship):>4} ({pct(one_ship):.0f}%)")
    print(f"    <=4-ship probes   : {len(probes):>4} ({pct(probes):.0f}%)   <- min-ship-bin 4 would force these up")
    print(f"    reached solo floor: {len(reached):>4} ({pct(reached):.0f}%)   <- already a real capture-attempt send")
    print(f"\n  Of the <=4-ship probes (n={len(probes)}):")
    if probes:
        pp = lambda x: 100.0 * len(x) / len(probes)
        print(f"    had-mass (max_ships>=cost): {len(had_mass_probe):>4} ({pp(had_mass_probe):.0f}%)   "
              f"<- COULD have soloed, sent a probe => min-ship-bin headroom")
        print(f"    garrison-limited (<cost)  : {len(garr_lim_probe):>4} ({pp(garr_lim_probe):.0f}%)   "
              f"<- clamps to garrison anyway => min-ship-bin can't help (aggregation, not size)")
    # bucket split: af1 (solo-takeable) vs agg (needs pooling)
    for bk in ("af1", "agg"):
        fb = [r for r in fired if r["bucket"] == bk]
        if not fb:
            continue
        pb = [r for r in fb if r["fired_ships"] <= 4]
        hm = [r for r in pb if r["fired_maxships"] >= r["fired_cost"]]
        print(f"    [{bk}] fired {len(fb)}  probe {len(pb)}  had-mass-probe {len(hm)}")
    headroom = len(had_mass_probe)
    verdict = (headroom >= 0.30 * n)
    print(f"\n  VERDICT: min-ship-bin {'HAS headroom' if verdict else 'WEAK'} — "
          f"had-mass probes = {100.0*headroom/n:.0f}% of fired-spare "
          f"{'>=' if verdict else '<'} 30% bar.")
    print("  (had-mass probe = source could solo the neutral but sent <=4 ships. High => the un-")
    print("   trapped firing is being discharged as junk that min-ship-bin would convert to mass.")
    print("   Low => the probes are garrison-limited / pooling problems min-ship-bin cannot fix.)")


def _report_and_plot(all_rows, output_path, shaping_coef=None):
    """all_rows: list of dicts with keys: step, v_s, fired_at, outcome, bucket, seed, seat."""
    if not all_rows:
        print("\nNO SPARE-OPPORTUNITY ROWS collected. Check that the checkpoint/opponent")
        print("produce contested neutrals in steps 10-50. Try more seeds or a different opponent.")
        return

    print("\n" + "=" * 92)
    print("V(s) vs SPARE-OPPORTUNITY DIAGNOSTIC")
    print("=" * 92)
    n_fired_total = sum(1 for r in all_rows if r["fired_at"])
    print(f"rows total: {len(all_rows)}  (steps {STEP_LO}-{STEP_HI}, af1+agg opportunities)")
    print(f"fired-spare rate: {n_fired_total}/{len(all_rows)} = {100*n_fired_total/len(all_rows):.1f}%  "
          f"(idle {100*(1-n_fired_total/len(all_rows)):.1f}%)")

    by_out = defaultdict(list)
    for r in all_rows:
        by_out[r["outcome"]].append(r)

    # ---- CALIBRATION CHECK (is the critic even right about outcome?) ----
    # This is the cleanest signal and free of the fire-selection confound. If mean V(s) on
    # LOST games is ~0 or positive, the critic is over-optimistic on the exact trajectories
    # where the policy needs a signal to change behavior -> no advantage gradient -> the
    # wall. This is necessary-condition evidence for the critic bottleneck.
    print(f"\n  CALIBRATION (mean V(s) over steps {STEP_LO}-{STEP_HI} by realized outcome):")
    for outcome in ("won", "lost"):
        vs = [r["v_s"] for r in by_out[outcome]]
        m, s = _mean(vs), _stderr(vs)
        tag = "  <-- over-optimistic (critic fails to predict loss)" if (
            outcome == "lost" and m > 0.0) else ""
        print(f"    {outcome:>4}: mean V = {m:+.3f} +/- {s:.3f}  (n={len(vs)}){tag}")

    # ---- delta_V (fired vs idle) — with selection confound noted ----
    # NOTE: delta_V is CONFOUNDED. The policy fires spares preferentially in worse states
    # (selection effect), so V|fired < V|idle doesn't mean "critic thinks firing is bad."
    # The calibration check above is the cleaner signal. delta_V is reported for completeness
    # and to show the policy doesn't fire in high-V (comfortable) states.
    print(f"\n  delta_V = mean(V|fired-spare) - mean(V|idle-spare), by outcome "
          f"[CONFOUNDED by fire-selection]:")
    print(f"  {'outcome':>8} | {'V|fired (n)':>20} | {'V|idle (n)':>20} | {'delta_V':>9}")
    print("  " + "-" * 80)
    summary = {}
    for outcome in ("won", "lost"):
        rows = by_out[outcome]
        fired = [r["v_s"] for r in rows if r["fired_at"]]
        idle = [r["v_s"] for r in rows if not r["fired_at"]]
        mf, mi = _mean(fired), _mean(idle)
        sf, si = _stderr(fired), _stderr(idle)
        d = mf - mi
        summary[outcome] = {"mf": mf, "mi": mi, "sf": sf, "si": si, "d": d,
                            "nf": len(fired), "ni": len(idle)}
        low_n = "  [low n_fired, collect more seeds]" if len(fired) < 30 else ""
        print(f"  {outcome:>8} | {mf:+.3f} +/- {sf:.3f} ({len(fired):>3}) | "
              f"{mi:+.3f} +/- {si:.3f} ({len(idle):>3}) | {d:+.3f}   {low_n}")
    print("  " + "-" * 80)
    print("\n  HOW TO READ THIS:")
    print("  (1) CALIBRATION = is the critic even right about outcome? Clean signal, no selection")
    print("      confound. If mean V on LOST trajectories > 0, critic is over-optimistic where it")
    print("      needs to signal trouble -> CRITIC bottleneck (attack the data).")
    print("  (2) delta_V = does the critic price the spare-launch DECISION, given the outcome?")
    print("      NOTE: confounded by selection (policy fires in worse states), so a small negative")
    print("      delta_V is expected and benign. The diagnostic case is delta_V ~= 0 WITH good")
    print("      calibration: critic knows you're losing but can't tell that firing the spare")
    print("      would help -> advantage A = r + gamma*V(s') - V(s) ~= 0 on spare-fire actions")
    print("      -> no gradient -> POLICY stays at 'don't fire' -> critic never sees fire data")
    print("      -> self-reinforcing loop. Fix: break the loop with off-policy pressure (Door 1:")
    print("      permanent aggregating opponent so firing predicts winning), OR exploration")
    print("      (crank entropy_coef_fire to force fire-sampling), OR a shaped per-step bonus")
    print("      for firing a spare (creates immediate reward signal independent of V).")
    print("  (3) delta_V clearly POSITIVE (and n_fired >= 30): critic prices spare-firing higher")
    print("      but policy still idles -> true POLICY bottleneck. Fix: entropy_coef_fire, or")
    print("      a shaped per-step bonus for firing a spare at a takeable neutral.")

    # ---- ship-size split (min-ship-bin headroom test) ----
    _ship_size_split(all_rows)

    # ---- PBRS pre-test (counterfactual shaping gradient) ----
    if shaping_coef is not None and all("shaping_raw" in r for r in all_rows):
        _pbrs_pretest(all_rows, shaping_coef)

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (1) V(s) histograms: fired vs idle, faceted by outcome.
    for ax, outcome in zip(axes[:2], ("won", "lost")):
        rows = by_out[outcome]
        fired_v = [r["v_s"] for r in rows if r["fired_at"]]
        idle_v = [r["v_s"] for r in rows if not r["fired_at"]]
        bins = np.linspace(min(min(fired_v, default=0), min(idle_v, default=0)),
                           max(max(fired_v, default=0), max(idle_v, default=0)), 40)
        ax.hist(idle_v, bins=bins, alpha=0.55, label=f"idle (n={len(idle_v)})", color="#d62728")
        ax.hist(fired_v, bins=bins, alpha=0.55, label=f"fired (n={len(fired_v)})", color="#2ca02c")
        s = summary[outcome]
        ax.axvline(s["mi"], color="#d62728", ls="--", lw=1.5,
                   label=f"mean idle {s['mi']:+.3f}")
        ax.axvline(s["mf"], color="#2ca02c", ls="--", lw=1.5,
                   label=f"mean fired {s['mf']:+.3f}")
        ax.set_title(f"{outcome.upper()}  delta_V = {s['d']:+.3f}")
        ax.set_xlabel("V(s)")
        ax.set_ylabel("spare-opportunity count")
        ax.legend(fontsize=8)

    # (2) by-step: mean V | fired vs mean V | idle, with the gap visible.
    ax = axes[2]
    for outcome, color, marker in (("won", "#2ca02c", "o"), ("lost", "#d62728", "x")):
        rows = by_out[outcome]
        by_step_f = defaultdict(list)
        by_step_i = defaultdict(list)
        for r in rows:
            (by_step_f if r["fired_at"] else by_step_i)[r["step"]].append(r["v_s"])
        steps = sorted(set(list(by_step_f) + list(by_step_i)))
        f_means = [_mean(by_step_f.get(s, [])) for s in steps]
        i_means = [_mean(by_step_i.get(s, [])) for s in steps]
        ax.plot(steps, i_means, f"{marker}--", color=color, alpha=0.6,
                label=f"{outcome} idle")
        ax.plot(steps, f_means, f"{marker}-", color=color, alpha=0.9,
                label=f"{outcome} fired")
    ax.set_xlabel("step")
    ax.set_ylabel("mean V(s)")
    ax.set_title("V(s) by step: fired vs idle")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    print(f"\nplot -> {output_path}")


def run(ckpt_path, opponent, n_seeds, seed_set, output_path, shaping_coef=None):
    model, cfg, target_decode, ship_bin_mode = _load_model(ckpt_path)
    device = torch.device("cpu")
    allow_reinforce = bool(getattr(model, "allow_reinforce", False))

    from expansion_autopsy import _seed_list
    seeds = _seed_list(seed_set, n_seeds)
    print(f"seed_set={seed_set}  {len(seeds)} seeds x2 seats = {len(seeds)*2} games  vs {opponent}",
          flush=True)

    all_rows = []
    for seed in seeds:
        for our_seat in (0, 1):
            step_records = []
            agent = build_tracking_agent(model, device, target_decode, ship_bin_mode,
                                         allow_reinforce, step_records)
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run([agent, opponent] if our_seat == 0 else [opponent, agent])
            final = env.steps[-1]
            r_us = final[our_seat].reward or 0.0
            r_opp = final[1 - our_seat].reward or 0.0
            outcome = "won" if r_us >= r_opp else "lost"

            rows = _scan_opportunities(step_records, our_seat)
            for r in rows:
                r["outcome"] = outcome
                r["seed"] = seed
                r["seat"] = our_seat
            all_rows.extend(rows)
            n_fired = sum(1 for r in rows if r["fired_at"])
            print(f"  seed={seed} seat={our_seat} {outcome:>4}: "
                  f"{len(rows)} opportunities, {n_fired} fired-spare "
                  f"({100*n_fired/max(len(rows),1):.0f}%)", flush=True)

    _report_and_plot(all_rows, output_path, shaping_coef=shaping_coef)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed-set", choices=["random", "static", "rotating"], default="random")
    ap.add_argument("--output", default="/tmp/v_spare_diagnostic.png")
    ap.add_argument("--shaping-coef", type=float, default=None,
                    help="α for the PBRS pre-test: report α·(Φ'−Φ) counterfactual gradient at spare-opps.")
    args = ap.parse_args()
    run(args.checkpoint, args.opponent, args.seeds, args.seed_set, args.output,
        shaping_coef=args.shaping_coef)


if __name__ == "__main__":
    main()
