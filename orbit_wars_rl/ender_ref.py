"""Ender-vs-Ajay reference — the top-10 target line for our conversion metrics.

Runs Ender and Ajay against each other as kaggle path-agents (both loaded by file
path, no torch checkpoint) and computes Ender's conversion metrics from the replay
via eval.game_conversion (agent-agnostic — it reads env.steps for a seat). Ender
alternates seats for balance. The printed block is directly comparable to any
`eval.py` run, so it is a standing reference for "what a top-10 agent's conversion
looks like vs Ajay" — the target to close toward.

Why this exists: held-out WR + loss-depth + decisive-mass cross vs Ender are the
progress signals that DON'T saturate (unlike out-massed%). This produces the
reference column. See docs/metrics.md "Ender reference" + [[project-ender-opponent-calibration]].

    CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/ender_ref.py --seeds 128   # 256 games
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)          # import eval (sibling module)
os.chdir(_REPO)                    # opponent paths are relative to repo root

import eval as ev                  # noqa: E402
from kaggle_environments import make  # noqa: E402

ENDER = "opponents/candidate_ender.py"
AJAY = "opponents/candidate_ajay_1200.py"


def _own_material(obs, seat):
    return (sum(p[5] for p in obs.planets if p[1] == seat)
            + sum(f[6] for f in obs.fleets if f[1] == seat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=128,
                    help="seeds; total games = 2*seeds (both seatings)")
    ap.add_argument("--agent", default=ENDER, help="agent to measure (path)")
    ap.add_argument("--opponent", default=AJAY, help="opponent (path)")
    args = ap.parse_args()

    acc = ev.new_conversion_acc()
    wins = games = 0
    for seed in range(args.seeds):
        for my_seat in (0, 1):
            agents = ([args.agent, args.opponent] if my_seat == 0
                      else [args.opponent, args.agent])
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(agents)
            rewards = [s.reward for s in env.steps[-1]]
            my_r = rewards[my_seat] if rewards[my_seat] is not None else 0.0
            op_r = rewards[1 - my_seat] if rewards[1 - my_seat] is not None else 0.0
            is_win = my_r > op_r
            material = _own_material(env.steps[-1][my_seat].observation, my_seat)
            ev.add_conversion(acc, ev.game_conversion(env.steps, my_seat),
                              won=is_win, material=material)
            wins += int(is_win)
            games += 1
            print(f"  seed={seed} seat={my_seat} win={is_win} material={material}", flush=True)

    print(f"\n{args.agent} win-rate vs {args.opponent}: {wins}/{games} ({100*wins/games:.1f}%)")
    print(ev._fmt_conversion(acc))
    print(ev._fmt_tier_summary(acc))


if __name__ == "__main__":
    main()
