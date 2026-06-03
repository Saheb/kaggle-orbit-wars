"""
Run a small number of games vs an opponent and save HTML replays + analysis.

For each game saves:
  - HTML replay (with title showing which color you are)
  - JSON with planet-lead trajectory per step
  - PNG plot of planet lead over game steps

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from orbit_wars_rl.eval import load_checkpoint, build_agent_fn
from orbit_wars_rl.config import Config
from orbit_wars_rl.model import EntityTransformer

# seat 0 = blue, seat 1 = orange (orbit wars renderer convention)
SEAT_COLOR = {0: "BLUE", 1: "ORANGE"}


def extract_planet_lead(env_steps, my_seat):
    """Return list of (step, my_planets, opp_planets, lead) across all game steps."""
    trajectory = []
    for i, step_obs in enumerate(env_steps):
        obs = step_obs[my_seat].observation
        if obs is None:
            continue
        my_planets = sum(1 for p in obs.planets if p[1] == my_seat)
        opp_planets = sum(1 for p in obs.planets if p[1] == (1 - my_seat))
        trajectory.append({
            "step": i,
            "my_planets": my_planets,
            "opp_planets": opp_planets,
            "lead": my_planets - opp_planets,
        })
    return trajectory


def plot_planet_lead(trajectory, label, output_path, is_win):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [t["step"] for t in trajectory]
        lead  = [t["lead"] for t in trajectory]
        mine  = [t["my_planets"] for t in trajectory]
        opp   = [t["opp_planets"] for t in trajectory]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        outcome = "WIN" if is_win else "LOSS"
        fig.suptitle(f"{label} — {outcome}", fontsize=12)

        ax1.plot(steps, mine, color="steelblue", label="My planets", linewidth=1.5)
        ax1.plot(steps, opp,  color="tomato",    label="Opp planets", linewidth=1.5)
        ax1.set_ylabel("Planet count")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(steps, lead, 0,
                         where=[l >= 0 for l in lead], color="steelblue", alpha=0.4, label="Ahead")
        ax2.fill_between(steps, lead, 0,
                         where=[l < 0 for l in lead],  color="tomato",    alpha=0.4, label="Behind")
        ax2.plot(steps, lead, color="black", linewidth=1.0)
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax2.axvline(400, color="orange", linewidth=1.0, linestyle=":", label="Capture reward cliff (400)")
        ax2.set_ylabel("Planet lead (me - opp)")
        ax2.set_xlabel("Game step")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=130)
        plt.close()
        return True
    except Exception as e:
        print(f"    (plot failed: {e})")
        return False


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
        color = SEAT_COLOR[seat]
        for seed in seeds:
            agents = [agent_fn, opponent_path] if seat == 0 else [opponent_path, agent_fn]
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(agents)
            final = env.steps[-1]
            rewards = [s.reward for s in final]

            my_reward  = rewards[seat]       if rewards[seat]       is not None else 0.0
            opp_reward = rewards[1 - seat]   if rewards[1 - seat]   is not None else 0.0
            is_win = my_reward > opp_reward
            n_steps = len(env.steps)

            label = f"seed{seed}_seat{seat}_{color}_{'WIN' if is_win else 'LOSS'}"
            print(f"  {label}: my={my_reward:+.3f} opp={opp_reward:+.3f} steps={n_steps}")

            # Planet lead trajectory
            trajectory = extract_planet_lead(env.steps, seat)
            final_lead = trajectory[-1]["lead"] if trajectory else 0

            # HTML replay — inject banner via script (runs after Preact mounts)
            html = env.render(mode="html")
            html = html.replace("<title>", f"<title>[You={color} seat{seat}] ", 1)
            bg = "#1a5276" if color == "BLUE" else "#7d3c00"
            border = "#3498db" if color == "BLUE" else "#e67e22"
            banner_script = f"""
<script>
window.addEventListener('load', function() {{
  var b = document.createElement('div');
  b.innerHTML = 'YOU = <b>{color}</b> (seat {seat}) &nbsp;|&nbsp; Seed {seed} &nbsp;|&nbsp; {"WIN ✓" if is_win else "LOSS ✗"} in {n_steps} steps &nbsp;|&nbsp; Final planet lead: {final_lead:+d}';
  b.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:2147483647;background:{bg};color:white;padding:7px 14px;font-size:14px;font-family:monospace;border-bottom:3px solid {border};pointer-events:none;';
  document.body.appendChild(b);
}});
</script>"""
            html = html.replace("</body>", banner_script + "</body>", 1)
            html_path = os.path.join(output_dir, f"{label}.html")
            with open(html_path, "w") as f:
                f.write(html)

            # JSON with trajectory
            json_path = os.path.join(output_dir, f"{label}.json")
            with open(json_path, "w") as f:
                json.dump({
                    "result": {
                        "seed": seed, "seat": seat, "color": color,
                        "win": is_win, "steps": n_steps,
                        "my_reward": my_reward, "opp_reward": opp_reward,
                        "final_planet_lead": final_lead,
                    },
                    "planet_trajectory": trajectory,
                }, f, indent=2)

            # Planet lead plot
            plot_path = os.path.join(output_dir, f"{label}_planet_lead.png")
            plot_planet_lead(trajectory, label, plot_path, is_win)

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
