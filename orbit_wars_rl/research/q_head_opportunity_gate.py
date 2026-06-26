"""CONCLUSIVE per-opportunity action-sensitivity gate for the COMA Q-head (docs/q-head.md step 1).

q_head_offline_probe measured q_fire−q_idle over ALL idle slots — but most idle slots are CORRECTLY
idle (no good target), so that population dilutes the signal. The gate q-head.md actually specified is
A_i on the SPARE-FIRE OPPORTUNITY set: idle owned planets that are a reachable, below-floor spare
source for a contested neutral (the exact 80%-idle population from multi_source_why). This script:

  1. loads the eval-faithful model (transition_autopsy._load_model → reinforce gates / ship-mode set),
  2. trains the SAME Q-head on the dumped torch_env+GAE rollout batch (q_head_offline_probe.train_q_head),
  3. plays vs Ajay, and at every spare-fire opportunity reads the spare SOURCE slot's q_fire/q_idle/p,
  4. reports, restricted to those slots:
       q_fire − q_idle  (>0 ⇒ Q says firing the idle spare helps — the delta_V≈0 gap, on-target),
       A_i = q_sa − (p·q_fire+(1−p)·q_idle)  (on an idle slot = −p·(q_fire−q_idle); <0 = recruitment).

Run from repo root (slow — plays Ajay games):
  ORBIT_GAME_PHASE_FEATURES=1 <venv>/bin/python orbit_wars_rl/q_head_opportunity_gate.py \
      --rollout /tmp/qhead_rollout.pt \
      --checkpoint gpu_run_artifacts/r32_stage_hlr/checkpoints/torch_step_2097152_*.pt \
      --opponent opponents/candidate_ajay_1200.py --seeds 24
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from kaggle_environments import make

import orbit_wars_rl.eval as E

from orbit_wars_rl.research.transition_autopsy import _load_model, _obs_to_dict
from orbit_wars_rl.features import extract_features
from orbit_wars_rl.action_mask import compute_action_masks
from orbit_wars_rl.research.value_spare_diagnostic import (
    _spare_sources_for_neutral, _decode_slot_targets, STEP_LO, STEP_HI, _mean, _stderr,
)
from orbit_wars_rl.research.q_head_offline_probe import load_rollout, split_train_val, train_q_head


def _decode_action_tensors(outputs, masks, obs, player, allow_reinforce, device):
    """Per-slot TAKEN action (fire_bit, ship_bin, target_idx) as (1, MO) tensors, with the same
    illegal-target masking as value_spare_diagnostic._decode_slot_targets, for q_counterfactual."""
    planets = obs["planets"]
    owned_indices = masks["owned_indices"].cpu().numpy()
    owned_count = int(masks["owned_count"])
    target_logits = outputs["target_logits"].cpu().clone()      # (1, MO, P)
    fire_logits = outputs["fire_logits"].cpu()                  # (1, MO, P)
    ship_logits = outputs["ship_logits"].cpu()                  # (1, MO, P, bins)
    P = target_logits.shape[-1]
    for slot in range(min(owned_count, target_logits.shape[1])):
        pidx = int(owned_indices[slot])
        if pidx >= len(planets):
            continue
        for tidx, tgt in enumerate(planets[:P]):
            is_source = int(tgt[0]) == int(planets[pidx][0])
            is_own = int(tgt[1]) == player
            if is_source or (is_own and not allow_reinforce):
                target_logits[:, slot, tidx] = -1e9
    target_idx = target_logits.argmax(dim=-1)                   # (1, MO)
    fire_logit_at = torch.gather(fire_logits, -1, target_idx.unsqueeze(-1)).squeeze(-1)  # (1, MO)
    fire_bit = (torch.sigmoid(fire_logit_at) > 0.5).long()
    ship_at = torch.gather(
        ship_logits, 2,
        target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, ship_logits.shape[-1]),
    ).squeeze(2)                                                # (1, MO, bins)
    ship_bin = ship_at.argmax(dim=-1)                          # (1, MO)
    p = torch.sigmoid(fire_logit_at)                           # (1, MO)
    return fire_bit.to(device), ship_bin.to(device), target_idx.to(device), p


def build_q_tracking_agent(model, device, ship_bin_mode, allow_reinforce, step_records):
    model.eval()

    def agent_fn(obs):
        obs = _obs_to_dict(obs)
        player = obs["player"]
        features = extract_features(obs, player, num_players=2)
        masks = compute_action_masks(obs, player)

        def f(name):
            return features[name].unsqueeze(0).to(device)
        slot_valid = masks["slot_valid"].to(device)
        if slot_valid.dim() == 1:
            slot_valid = slot_valid.unsqueeze(0)
        with torch.no_grad():
            fwd_kw = dict(
                fire_mask=masks["fire_mask"].to(device), angle_mask=masks["angle_mask"].to(device),
                slot_valid=slot_valid, owned_indices=masks["owned_indices"].to(device),
                owned_count=masks["owned_count"],
                pairwise_features=f("pairwise_features") if "pairwise_features" in features else None,
            )
            base = (f("planet_features"), f("fleet_features"), f("global_features"),
                    f("planet_mask"), f("fleet_mask"))
            outputs = model(*base, **fwd_kw)
            encoded = model.encode_state(
                *base, slot_valid=slot_valid, owned_indices=masks["owned_indices"].to(device),
                pairwise_features=fwd_kw["pairwise_features"])
            fire_bit, ship_bin, target_idx, p = _decode_action_tensors(
                outputs, masks, obs, player, allow_reinforce, device)
            q = model.q_counterfactual(encoded, fire_bit, ship_bin, target_idx, slot_valid)

        slot_targets = _decode_slot_targets(outputs, masks, obs, player, allow_reinforce,
                                            ship_bin_mode=ship_bin_mode)
        step_records.append({
            "step": int(obs["step"]),
            "planets": obs["planets"], "fleets": obs["fleets"],
            "slot_targets": slot_targets,
            "owned_indices": masks["owned_indices"].cpu().numpy(),
            "owned_count": int(masks["owned_count"]),
            "q_fire": q["q_fire"][0].cpu().numpy(), "q_idle": q["q_idle"][0].cpu().numpy(),
            "q_sa": float(q["q_sa"][0].cpu()), "p": p[0].cpu().numpy(),
        })

        moves = E.actions_from_target_policy(
            outputs["fire_logits"].cpu(), outputs["target_logits"].cpu(), outputs["ship_logits"].cpu(),
            {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
            obs, player, ship_bin_mode=ship_bin_mode, allow_reinforce=allow_reinforce,
            reinforce_gate_min_planets=getattr(model, "reinforce_gate_min_planets", 0),
            reinforce_forward_only=getattr(model, "reinforce_forward_only", False),
            reinforce_garrison_floor=getattr(model, "reinforce_garrison_floor", 0.0),
            sufficient_commit_factor=getattr(model, "sufficient_commit_factor", 0.0),
            reverse_edge_cooldown=int(getattr(model, "reverse_edge_cooldown", 0)),
        )
        return moves

    return agent_fn


def _scan(step_records, seat, outcome):
    rows = []
    for rec in step_records:
        t = rec["step"]
        if t < STEP_LO or t > STEP_HI:
            continue
        planets, fleets = rec["planets"], rec["fleets"]
        owned_indices, owned_count = rec["owned_indices"], rec["owned_count"]
        q_fire, q_idle, p, q_sa = rec["q_fire"], rec["q_idle"], rec["p"], rec["q_sa"]
        by_src = {info["src_planet_idx"]: info for info in rec["slot_targets"].values()}
        slot_by_pidx = {}
        for s in range(min(owned_count, len(owned_indices))):
            slot_by_pidx.setdefault(int(owned_indices[s]), s)

        for nidx, neutral in [(i, p_) for i, p_ in enumerate(planets) if int(p_[1]) < 0]:
            sources = _spare_sources_for_neutral(neutral, planets, fleets, seat)
            if not sources:
                continue
            total_spare = sum(s for _, s, _ in sources)
            cheapest = min(c for _, _, c in sources)
            solo = any(s >= c for _, s, c in sources)
            bucket = "af1" if solo else ("agg" if total_spare >= cheapest else "cap")
            if bucket == "cap":
                continue
            for pidx, spare, cost in sources:
                slot = slot_by_pidx.get(pidx)
                if slot is None or slot >= len(q_fire):
                    continue
                info = by_src.get(pidx)
                fired_any = bool(info and info["fired"])
                qf, qi, pi = float(q_fire[slot]), float(q_idle[slot]), float(p[slot])
                rows.append({
                    "qf_minus_qi": qf - qi,
                    "A_i": q_sa - (pi * qf + (1 - pi) * qi),
                    "p": pi, "is_idle": not fired_any, "bucket": bucket, "outcome": outcome,
                })
    return rows


def _report(rows):
    def stat(sel):
        d = [r["qf_minus_qi"] for r in sel]
        a = [r["A_i"] for r in sel]
        return len(sel), _mean(d), _stderr(d), _mean(a)

    print("\n" + "=" * 92)
    print("CONCLUSIVE PER-OPPORTUNITY Q-HEAD GATE  (spare-source slots at spare-fire opportunities)")
    print("=" * 92)
    idle = [r for r in rows if r["is_idle"]]
    fired = [r for r in rows if not r["is_idle"]]
    print(f"opportunities scanned: {len(rows)} spare-source slots  "
          f"(idle {len(idle)} / fired {len(fired)})")
    print(f"\n{'population':>26} | {'n':>5} | {'q_fire−q_idle (±se)':>24} | {'mean A_i':>9}")
    print("-" * 78)
    for name, sel in [("IDLE spares (THE GATE)", idle), ("  idle af1 (solo-takeable)",
                       [r for r in idle if r["bucket"] == "af1"]),
                      ("  idle agg (needs pooling)", [r for r in idle if r["bucket"] == "agg"]),
                      ("  idle | our WON games", [r for r in idle if r["outcome"] == "won"]),
                      ("  idle | our LOST games", [r for r in idle if r["outcome"] == "lost"]),
                      ("fired spares", fired)]:
        if not sel:
            continue
        n, md, se, ma = stat(sel)
        print(f"{name:>26} | {n:>5} | {md:+.4f} ± {se:.4f}        | {ma:+.4f}")
    if idle:
        _, md, se, _ = stat(idle)
        gate = md - 2 * se > 0
        print(f"\nGATE READ: mean q_fire−q_idle on idle spares = {md:+.4f} ± {se:.4f}  →  "
              f"{'PASS (Q prices firing the idle spare as helping, >2se)' if gate else 'FAIL/≈0 (Q does not price the idle spare as helping)'}")
        print("  (this is the on-target version of the broad −0.015; >0 ⇒ the signal delta_V≈0 lacked is present)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model, cfg, target_decode, ship_bin_mode = _load_model(args.checkpoint)
    for n, pr in model.named_parameters():
        pr.requires_grad_(n.startswith("q_"))
    allow_reinforce = bool(getattr(model, "allow_reinforce", False))
    device = torch.device("cpu")

    batch = load_rollout(args.rollout)
    train_idx, val_idx = split_train_val(batch, 0.2, args.seed)
    print(f"training Q-head on dumped rollout (TN={batch['returns'].shape[0]})", flush=True)
    train_q_head(model, batch, train_idx, val_idx, epochs=args.epochs)

    from orbit_wars_rl.research.expansion_autopsy import _seed_list
    seeds = _seed_list("random", args.seeds)
    print(f"\nplaying {len(seeds)*2} games vs {args.opponent}", flush=True)
    all_rows = []
    for sd in seeds:
        for our_seat in (0, 1):
            recs = []
            agent = build_q_tracking_agent(model, device, ship_bin_mode, allow_reinforce, recs)
            env = make("orbit_wars", configuration={"seed": sd}, debug=False)
            env.run([agent, args.opponent] if our_seat == 0 else [args.opponent, agent])
            final = env.steps[-1]
            r_us = final[our_seat].reward or 0.0
            r_opp = final[1 - our_seat].reward or 0.0
            outcome = "won" if r_us >= r_opp else "lost"
            rows = _scan(recs, our_seat, outcome)
            all_rows.extend(rows)
            print(f"  seed={sd} seat={our_seat} {outcome:>4}: {len(rows)} spare-source slots", flush=True)
    _report(all_rows)


if __name__ == "__main__":
    main()
