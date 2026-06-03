"""
Plot value function trajectories V(s) over game steps for wins vs losses.

Runs N games vs an opponent, captures the model's value estimate at every step,
then plots trajectories coloured by outcome. Tells us whether the critic is
actually predictive (V rises before wins, falls before losses) or just noisy.

Usage:
  python3 orbit_wars_rl/value_analysis.py \
    --checkpoint <path.pt> \
    --opponent opponents/candidate_zach_public.py \
    --seeds 585 1217 6462 9984 1 1725 1948 2419 1467 2412 3154 8717 \
    --output /tmp/value_trajectories.png \
    --target-decode
"""
import argparse, sys, collections
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from orbit_wars_rl.config import Config
from orbit_wars_rl.model import EntityTransformer
from orbit_wars_rl.eval import load_checkpoint, build_agent_fn
from orbit_wars_rl.features import extract_features
from orbit_wars_rl.action_mask import compute_action_masks


def build_value_tracking_agent(model, device, target_decode, value_log, player_idx):
    """Like build_agent_fn but also appends value estimates to value_log."""
    from orbit_wars_rl.eval import actions_from_target_policy, actions_from_policy

    model.eval()

    def agent_fn(obs):
        if not isinstance(obs, dict):
            obs = {
                "step": int(getattr(obs, "step", 0)),
                "player": int(getattr(obs, "player", 0)),
                "planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                            for p in obs.planets],
                "fleets": [[f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships]
                           for f in obs.fleets],
                "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
                "initial_planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                                    for p in getattr(obs, "initial_planets", obs.planets)],
                "comet_planet_ids": list(getattr(obs, "comet_planet_ids", [])),
            }

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

        # Record value at this step
        value_log.append((obs["step"], float(outputs["value"][0].cpu())))

        action_fn = actions_from_target_policy if target_decode else actions_from_policy
        if target_decode:
            return action_fn(
                outputs["fire_logits"].cpu(), outputs["target_logits"].cpu(),
                outputs["ship_logits"].cpu(),
                {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
                obs, player, ship_bin_mode="absolute",
            )
        return action_fn(
            outputs["fire_logits"].cpu(), outputs["angle_logits"].cpu(),
            outputs["ship_logits"].cpu(),
            {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
            obs, player, ship_bin_mode="absolute",
        )

    return agent_fn


def run(checkpoint, opponent_path, seeds, output, target_decode):
    from kaggle_environments import make

    cfg = Config()
    sd, _ = load_checkpoint(checkpoint, cfg)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    wins, losses = [], []

    for seat in [0, 1]:
        for seed in seeds:
            value_log = []
            agent_fn = build_value_tracking_agent(
                model, device, target_decode, value_log, player_idx=seat)

            agents = [agent_fn, opponent_path] if seat == 0 else [opponent_path, agent_fn]
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(agents)

            final = env.steps[-1]
            rewards = [s.reward for s in final]
            my_reward = rewards[seat] if rewards[seat] is not None else 0.0
            opp_reward = rewards[1 - seat] if rewards[1 - seat] is not None else 0.0
            is_win = my_reward > opp_reward

            steps = [s for s, v in value_log]
            values = [v for s, v in value_log]
            label = f"s{seed}/seat{seat} ({'WIN' if is_win else 'LOSS'}, {len(env.steps)} steps)"
            print(f"  {label}: V_start={values[0]:+.3f} V_end={values[-1]:+.3f}")

            if is_win:
                wins.append((steps, values, label))
            else:
                losses.append((steps, values, label))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Value Function V(s) Trajectories — wins vs losses", fontsize=13)

    for ax, trajectories, title, color in [
        (axes[0], wins,   f"WINS ({len(wins)})",   "green"),
        (axes[1], losses, f"LOSSES ({len(losses)})", "red"),
    ]:
        cmap = cm.Greens if color == "green" else cm.Reds
        for i, (steps, values, label) in enumerate(trajectories):
            alpha = 0.6
            c = cmap(0.4 + 0.5 * i / max(len(trajectories), 1))
            ax.plot(steps, values, color=c, alpha=alpha, linewidth=1.2, label=label)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Game step")
        ax.set_ylabel("V(s)")
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"\nPlot saved to: {output}")
    print(f"Wins: {len(wins)}  Losses: {len(losses)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[585, 1217, 6462, 9984, 1, 1725, 1948, 2419,
                                 1467, 2412, 3154, 8717])
    parser.add_argument("--output", default="/tmp/value_trajectories.png")
    parser.add_argument("--target-decode", action="store_true")
    args = parser.parse_args()

    run(args.checkpoint, args.opponent, args.seeds, args.output, args.target_decode)
