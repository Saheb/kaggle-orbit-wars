"""Count how many distinct source planets Ajay fires from each step."""
import sys
import os
import collections

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opponents"))

from kaggle_environments import make
import candidate_ajay_1200 as ajay_mod

NUM_GAMES = 20
SEEDS = range(NUM_GAMES)

srcs_counts = collections.Counter()  # n_sources → count of steps

original_agent = ajay_mod.agent
step_srcs = []

def counting_agent(obs):
    moves = original_agent(obs)
    # moves is a list of [from_planet_id, angle, ships]
    n_sources = len(set(m[0] for m in moves)) if moves else 0
    step_srcs.append(n_sources)
    return moves

total_steps = 0
for seed in SEEDS:
    step_srcs.clear()
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    # Ajay plays as player 1 (opponent seat); we need a dummy agent for player 0
    env.run([counting_agent, counting_agent])  # both seats = Ajay vs Ajay for pure firing stats
    for n in step_srcs:
        srcs_counts[n] += 1
    total_steps += len(step_srcs)

print(f"\nAjay firing source distribution over {NUM_GAMES} self-play games ({total_steps} agent steps):\n")
print(f"  {'n_sources':>10}  {'steps':>8}  {'pct':>7}")
print("  " + "-" * 30)
for n in sorted(srcs_counts):
    pct = 100 * srcs_counts[n] / total_steps
    print(f"  {n:>10}  {srcs_counts[n]:>8}  {pct:>6.1f}%")

firing_steps = sum(v for k, v in srcs_counts.items() if k > 0)
multi_steps  = sum(v for k, v in srcs_counts.items() if k > 1)
multi3_steps = sum(v for k, v in srcs_counts.items() if k > 2)
print(f"\n  Steps with any firing:     {firing_steps} ({100*firing_steps/total_steps:.1f}%)")
print(f"  Steps firing >1 source:    {multi_steps}  ({100*multi_steps/total_steps:.1f}%)")
print(f"  Steps firing >2 sources:   {multi3_steps}  ({100*multi3_steps/total_steps:.1f}%)")
print(f"  Mean sources per step:     {sum(k*v for k,v in srcs_counts.items())/total_steps:.3f}")
print(f"  Mean sources when firing:  {sum(k*v for k,v in srcs_counts.items() if k>0)/max(firing_steps,1):.3f}")
