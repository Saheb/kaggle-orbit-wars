"""Veto-rate probe — is a reinforce-discipline mask blocking the holds the policy WANTS?

For each fire-positive slot, looks at the policy's UNMASKED argmax target (source excluded).
If that target is an own planet (= the policy wants to reinforce it), attribute whether the
reinforce is blocked by the empire GATE, the FORWARD-only mask, or the GARRISON-FLOOR — or
allowed. This separates two very different worlds:
  - high block rate (esp. FLOOR/FORWARD)  → the mask is suppressing reinforces the policy would
    otherwise make → relaxing it could let it hold.
  - low block rate                        → the policy doesn't even want those reinforces → the
    mask is NOT the constraint; the hold failure is elsewhere (targeting / no defense incentive).

Reuses the exact production mask logic (eval.build_agent_fn → action_mask.actions_from_target_policy
with veto_stats), so the counts can't diverge from real behaviour.

Usage (from repo root):
  python orbit_wars_rl/garrison_floor_probe.py \
    --checkpoint gpu_run_artifacts/p2rev3/checkpoints/torch_step_4194304_p2rev3_20260611_153903.pt \
    --opponent opponents/candidate_debatreya_1300.py --games 30 \
    --reinforce-gate-min-planets 3 --reinforce-forward-only --reinforce-garrison-floor 10
"""
import argparse

import torch
import kaggle_environments as ke

from orbit_wars_rl.config import Config
from orbit_wars_rl.model import EntityTransformer
from orbit_wars_rl.eval import load_checkpoint, build_agent_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_debatreya_1300.py")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--reinforce-gate-min-planets", type=int, default=3)
    ap.add_argument("--reinforce-forward-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reinforce-garrison-floor", type=float, default=10.0)
    args = ap.parse_args()

    cfg = Config()
    cfg.device = "cpu"
    device = torch.device("cpu")
    sd, _ = load_checkpoint(args.checkpoint, cfg)
    model = EntityTransformer(cfg.model).to(device)
    model.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    model.reinforce_gate_min_planets = args.reinforce_gate_min_planets
    model.reinforce_forward_only = args.reinforce_forward_only
    model.reinforce_garrison_floor = args.reinforce_garrison_floor
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"allow_reinforce={model.allow_reinforce}  gate>={args.reinforce_gate_min_planets}  "
          f"forward_only={args.reinforce_forward_only}  garrison_floor={args.reinforce_garrison_floor}")

    veto = {}
    agent = build_agent_fn(model, device, ship_bin_mode=cfg.model.ship_bin_mode,
                           target_decode=True, allow_reinforce=model.allow_reinforce,
                           veto_stats=veto)

    def our(obs, config=None):
        return agent(obs)

    for i in range(args.games):
        seed = args.seed_start + i
        env = ke.make("orbit_wars", {"seed": seed})
        agents = [our, args.opponent] if (i % 2 == 0) else [args.opponent, our]
        env.run(agents)
        print(f"  game {i + 1}/{args.games} (seat {i % 2}) done", end="\r")

    fs = veto.get("fire_slots", 0)
    ai = veto.get("attack_intent", 0)
    ri = veto.get("reinforce_intent", 0)
    bg = veto.get("blocked_gate", 0)
    bf = veto.get("blocked_forward", 0)
    bl = veto.get("blocked_floor", 0)
    ok = veto.get("reinforce_allowed", 0)
    pct = lambda x, d: (f"{x/d:.1%}" if d else "—")
    print("\n" + "=" * 60)
    print(f"vs {args.opponent}  ({args.games} games, both seats)")
    print(f"fire-positive slots : {fs}")
    print(f"  attack-intent     : {ai}  ({pct(ai, fs)})")
    print(f"  reinforce-intent  : {ri}  ({pct(ri, fs)})")
    print(f"  of reinforce-intent — which mask blocks the WANTED reinforce:")
    print(f"    blocked GATE     : {bg}  ({pct(bg, ri)})")
    print(f"    blocked FORWARD  : {bf}  ({pct(bf, ri)})")
    print(f"    blocked FLOOR    : {bl}  ({pct(bl, ri)})")
    print(f"    ALLOWED          : {ok}  ({pct(ok, ri)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
