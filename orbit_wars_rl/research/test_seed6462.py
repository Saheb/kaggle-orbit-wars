"""
Isolation test for seeds that were losses at Rev28 27M peak.

Losing seeds at 27M (from replay analysis):
  seed6462 seat0: LOSS 473 steps (passive standoff — primary test case)
  seed1467 seat0: LOSS 214 steps
  seed3154 seat0: LOSS 500 steps (timeout)
  seed1467 seat1: LOSS 191 steps
  seed3154 seat1: LOSS 156 steps

Reports step count, outcome, and whether game ended before step 400.
Shorter = more decisive. Rev28 27M baselines shown for comparison.

Usage:
  python3 orbit_wars_rl/test_seed6462.py --checkpoint <path.pt> --target-decode
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from orbit_wars_rl.config import Config
from orbit_wars_rl.model import EntityTransformer
from orbit_wars_rl.eval import load_checkpoint, build_agent_fn

# (seed, seat) pairs that were losses at Rev28 27M — baselines for comparison
LOSING_CASES = [
    (6462, 0, 473),   # passive standoff — primary test
    (1467, 0, 214),
    (3154, 0, 500),   # timeout
    (1467, 1, 191),
    (3154, 1, 156),
    # Weak archetypes from Rev30 10M panel (<62.5%) — added to track improvement
    # low_prod__mostly_static__big_static (37.5%)
    (2560, 0, 500), (4676, 0, 500), (6475, 0, 500), (8526, 0, 500),
    (2560, 1, 500), (4676, 1, 500), (6475, 1, 500), (8526, 1, 500),
    # low_prod__mixed_rotating__big_static (50.0%)
    (78,   0, 500), (1750, 0, 500), (7166, 0, 500), (8866, 0, 500),
    (78,   1, 500), (1750, 1, 500), (7166, 1, 500), (8866, 1, 500),
]


def test(checkpoint, target_decode=False):
    from kaggle_environments import make

    cfg = Config()
    sd, _ = load_checkpoint(checkpoint, cfg)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    agent_fn = build_agent_fn(model, device, ship_bin_mode=cfg.model.ship_bin_mode,
                               target_decode=target_decode)

    # Use absolute path so it works regardless of cwd
    opp = str(Path(__file__).parent.parent / "opponents" / "candidate_zach_public.py")
    print(f"\nLoss-seed isolation test — {Path(checkpoint).name}")
    print(f"{'Seed/Seat':<16} {'Result':<8} {'Steps':>6}  {'vs baseline':>12}  {'Note'}")
    print("-" * 70)

    wins = 0
    for seed, seat, baseline_steps in LOSING_CASES:
        agents = [agent_fn, opp] if seat == 0 else [opp, agent_fn]
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)
        final = env.steps[-1]
        rewards = [s.reward for s in final]
        my_r = rewards[seat] if rewards[seat] is not None else 0.0
        opp_r = rewards[1 - seat] if rewards[1 - seat] is not None else 0.0
        is_win = my_r > opp_r
        steps = len(env.steps)
        delta = steps - baseline_steps
        decisive = steps < 400
        wins += int(is_win)

        direction = f"{delta:+d} vs {baseline_steps}" if delta != 0 else f"= {baseline_steps}"
        note = "✓ decisive" if decisive else "⚠ passive (>400)"
        print(f"  seed{seed} seat{seat}:  {'WIN' if is_win else 'LOSS'}    {steps:>6}  {direction:>12}  {note}")

    print(f"\n  Wins: {wins}/{len(LOSING_CASES)} (was 0/{len(LOSING_CASES)} at Rev28 27M)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-decode", action="store_true")
    args = parser.parse_args()
    test(args.checkpoint, args.target_decode)
