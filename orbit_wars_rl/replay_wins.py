"""
Run a small number of games vs an opponent and save HTML replays for wins.
Useful for diagnosing whether wins are meaningful or accidental.

Usage:
  python3 orbit_wars_rl/replay_wins.py \
    --checkpoint <path.pt> \
    --opponent opponents/candidate_zach_public.py \
    --seeds 417 451 2663 8782 \
    --output-dir /tmp/replays \
    --target-decode
"""
import argparse, json, os, sys
import torch
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from orbit_wars_rl.eval import load_checkpoint, build_agent_fn
from orbit_wars_rl.config import Config
from orbit_wars_rl.model import EntityTransformer


def run_replay(checkpoint, opponent_path, seeds, output_dir, target_decode=False):
    from kaggle_environments import make

    os.makedirs(output_dir, exist_ok=True)

    cfg = Config()
    sd, _ = load_checkpoint(checkpoint, cfg)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    agent_fn = build_agent_fn(model, device,
                               ship_bin_mode=cfg.model.ship_bin_mode,
                               target_decode=target_decode)

    wins, losses = [], []

    for seat in [0, 1]:
        for seed in seeds:
            agents = [agent_fn, opponent_path] if seat == 0 else [opponent_path, agent_fn]
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(agents)
            final = env.steps[-1]
            rewards = [s.reward for s in final]

            my_idx = seat
            my_reward = rewards[my_idx] if rewards[my_idx] is not None else 0.0
            opp_reward = rewards[1 - my_idx] if rewards[1 - my_idx] is not None else 0.0
            is_win = my_reward > opp_reward

            result = {
                "seed": seed, "seat": seat, "win": is_win,
                "my_reward": my_reward, "opp_reward": opp_reward,
                "steps": len(env.steps),
            }

            label = f"seed{seed}_seat{seat}_{'WIN' if is_win else 'LOSS'}"
            print(f"  {label}: my={my_reward:+.3f} opp={opp_reward:+.3f} steps={len(env.steps)}")

            # Save HTML replay
            html = env.render(mode="html")
            html_path = os.path.join(output_dir, f"{label}.html")
            with open(html_path, "w") as f:
                f.write(html)

            # Save raw episode JSON for analysis
            episode = [{"step": i, "obs": str(s[0].observation)[:500]}
                       for i, s in enumerate(env.steps[::10])]  # every 10 steps
            json_path = os.path.join(output_dir, f"{label}.json")
            with open(json_path, "w") as f:
                json.dump({"result": result, "sampled_steps": episode}, f, indent=2)

            if is_win:
                wins.append(label)
            else:
                losses.append(label)

    print(f"\nWins ({len(wins)}/{len(wins)+len(losses)}): {wins}")
    print(f"Replays saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[417, 451, 2663, 8782])
    parser.add_argument("--output-dir", default="/tmp/orbit_replays")
    parser.add_argument("--target-decode", action="store_true")
    args = parser.parse_args()

    run_replay(args.checkpoint, args.opponent,
               args.seeds, args.output_dir, args.target_decode)
