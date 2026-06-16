"""Probe A: dump our agent's games as kaggle replays for the action-list
aggregation analyzer (analyze_action_list_replays.py).

Runs a checkpoint vs an opponent in the real kaggle env, both seats, and writes
each game's replay JSON into <out>/won or <out>/lost (split by our result), with
our seat tagged "OURS" in info.TeamNames so the analyzer's --player OURS isolates
our seat. Then:

  python3 analyze_action_list_replays.py <out>/won  --player OURS
  python3 analyze_action_list_replays.py <out>/lost --player OURS
  python3 analyze_action_list_replays.py <out>      --player OURS   # combined

Reuses eval.load_checkpoint / eval.build_agent_fn so loading + decode match the
panel exactly (discipline masks auto-load from the checkpoint).
"""
import argparse
import json
import os

import torch

from config import Config
from model import EntityTransformer
from eval import load_checkpoint, build_agent_fn


def _build_model(ckpt_path: str, cfg: Config):
    state_dict, ckpt_action_decode = load_checkpoint(ckpt_path, cfg)
    target_decode = (ckpt_action_decode == "target")
    m = cfg.model
    if bool(m.allow_reinforce) and not bool(getattr(m, "_discipline_persisted", False)):
        raise SystemExit("Checkpoint has allow_reinforce but no persisted discipline; this probe "
                         "targets post-2026-06-15 ckpts (revedge1) that carry it.")
    device = torch.device("cpu")
    model = EntityTransformer(m).to(device)
    model.allow_reinforce = bool(getattr(m, "allow_reinforce", False))
    model.reinforce_gate_min_planets = int(getattr(m, "reinforce_gate_min_planets", 0))
    model.reinforce_forward_only = bool(getattr(m, "reinforce_forward_only", False))
    model.reinforce_garrison_floor = float(getattr(m, "reinforce_garrison_floor", 0.0))
    model.sufficient_commit_factor = float(getattr(m, "sufficient_commit_factor", 0.0))
    model.reverse_edge_cooldown = int(getattr(m, "reverse_edge_cooldown", 0))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if k not in {"target_head.weight", "target_head.bias"}]
    bad_unexpected = [k for k in unexpected if not k.startswith("value_pp_")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"ckpt/model mismatch: missing={bad_missing} unexpected={bad_unexpected}")
    model.eval()
    print(f"loaded {ckpt_path}  target_decode={target_decode}  allow_reinforce={model.allow_reinforce} "
          f"gate={model.reinforce_gate_min_planets} forward={model.reinforce_forward_only} "
          f"floor={model.reinforce_garrison_floor} revedge_cd={model.reverse_edge_cooldown} "
          f"ship_bin_mode={m.ship_bin_mode}", flush=True)
    return model, device, target_decode, m.ship_bin_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", required=True, help="path to opponent agent .py")
    ap.add_argument("--games", type=int, default=64, help="seeds; each played BOTH seats")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--out", required=True, help="output dir; won/ and lost/ created under it")
    args = ap.parse_args()

    from kaggle_environments import make

    cfg = Config()
    cfg.env.num_players = 2
    model, device, target_decode, ship_bin_mode = _build_model(args.checkpoint, cfg)
    agent_fn = build_agent_fn(model, device, fire_threshold=0.5, sample=False,
                              ship_bin_mode=ship_bin_mode, target_decode=target_decode)

    won_dir = os.path.join(args.out, "won")
    lost_dir = os.path.join(args.out, "lost")
    os.makedirs(won_dir, exist_ok=True)
    os.makedirs(lost_dir, exist_ok=True)

    n_won = n_lost = 0
    opp_name = os.path.splitext(os.path.basename(args.opponent))[0]
    for seed in range(args.seed_start, args.seed_start + args.games):
        for my_seat in (0, 1):
            agents = [agent_fn, args.opponent] if my_seat == 0 else [args.opponent, agent_fn]
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(agents)
            final = env.steps[-1]
            rewards = [s.reward if s.reward is not None else 0.0 for s in final]
            is_win = rewards[my_seat] > rewards[1 - my_seat]

            replay = env.toJSON()
            names = ["OURS", "OURS"]
            names[my_seat] = "OURS"
            names[1 - my_seat] = opp_name
            replay.setdefault("info", {})["TeamNames"] = names
            replay["rewards"] = rewards

            dest = won_dir if is_win else lost_dir
            with open(os.path.join(dest, f"seed{seed}_seat{my_seat}.json"), "w") as f:
                json.dump(replay, f)
            n_won += int(is_win); n_lost += int(not is_win)
        if (seed - args.seed_start + 1) % 8 == 0:
            tot = n_won + n_lost
            print(f"  seed {seed - args.seed_start + 1}/{args.games}  WR {n_won}/{tot} "
                  f"({100*n_won/max(tot,1):.0f}%)", flush=True)

    print(f"DONE vs {opp_name}: won={n_won} lost={n_lost}  ->  {args.out}", flush=True)


if __name__ == "__main__":
    main()
