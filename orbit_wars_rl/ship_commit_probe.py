"""Ship-commitment audit: per launch, how big a fraction of the SOURCE garrison the trained
ship head chooses to send. The key bucket is send/garrison >= 1.0 — the policy picked a ship_bin
whose nominal count >= the source's current ships. In EVAL that CLAMPS to the full garrison
(action_mask `_ship_bin_to_count` = min(count, max_ships)); in TORCH_ENV TRAINING that SAME pick is
SILENTLY DROPPED (`valid_ships = src_ships >= ship_count`, torch_env.py:1346). So this measures both
"how often it commits the whole garrison" and (as a proxy) "how often a launch would be dropped in
training". Run from repo ROOT:
  orbit_wars_rl/.venv/bin/python orbit_wars_rl/ship_commit_probe.py
"""
import argparse
import torch
from kaggle_environments import make

import action_mask
from config import Config
from model import EntityTransformer
from eval import load_checkpoint, build_agent_fn


def load_ckpt_agent(ckpt_path, gate, device="cpu", fire_threshold=0.5):
    cfg = Config(); cfg.device = device
    sd, dec = load_checkpoint(ckpt_path, cfg)
    target_decode = (dec == "target")
    model = EntityTransformer(cfg.model).to(device)
    model.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    model.reinforce_gate_min_planets = int(gate)
    model.reinforce_forward_only = False
    model.reinforce_garrison_floor = 0.0
    model.sufficient_commit_factor = 0.0
    model.load_state_dict(sd, strict=False); model.eval()
    return build_agent_fn(model, torch.device(device), fire_threshold=fire_threshold,
                          sample=False, ship_bin_mode=cfg.model.ship_bin_mode,
                          target_decode=target_decode)


def run(agent, opp, games):
    for s in range(games):
        env = make("orbit_wars", configuration={"seed": s}, debug=False)
        env.run([agent, opp])


def report(label):
    print(f"\n=== {label} ===")
    for cat in ("attack", "reinforce"):
        d = action_mask._SHIP_AUDIT_DATA[cat]; n = d["n"]
        if n == 0:
            print(f"  {cat}: no launches"); continue
        print(f"  {cat}: {n} launches | full-commit (bin>=garrison) {100*d['full']/n:5.1f}% "
              f"| overflow (bin>garrison → DROPPED in train) {100*d['overflow']/n:5.1f}% "
              f"| mean send/garrison {d['ratio_sum']/n:.2f}")
        h = d["hist"]; tot = sum(h) or 1
        labs = ["0-.1", ".1-.2", ".2-.3", ".3-.4", ".4-.5", ".5-.6", ".6-.7", ".7-.8", ".8-.9", ".9-1", ">=1"]
        print("    send/garrison dist: " + " ".join(f"{labs[i]}:{100*h[i]/tot:.0f}" for i in range(11)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="opponents/candidate_debatreya_1300.py")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--gate", type=int, default=2)
    args = ap.parse_args()
    action_mask._SHIP_AUDIT["on"] = True
    ckpts = [
        ("consol_6M_final", "gpu_run_artifacts/consol/checkpoints/torch_step_6012928_consol_20260614_212437_final.pt"),
        ("rev38_5M",        "seed_checkpoints/rev38_5M_15g.pt"),
    ]
    print(f"Ship-commitment audit vs {args.opponent}, {args.games} games, gate={args.gate}")
    print("send/garrison >= 1.0 = picks a bin >= source garrison → CLAMPED to full in EVAL, DROPPED in torch_env TRAINING")
    for label, ck in ckpts:
        action_mask._reset_ship_audit()
        agent = load_ckpt_agent(ck, args.gate)
        run(agent, args.opponent, args.games)
        report(label)


if __name__ == "__main__":
    main()
