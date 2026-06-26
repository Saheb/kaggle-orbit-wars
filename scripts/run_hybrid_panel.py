"""Run a 256-game panel for a standalone agent .py file vs an opponent."""
import sys
import kaggle_environments as ke

AGENT = sys.argv[1]
OPPONENT = sys.argv[2]

env = ke.make("orbit_wars")
wins = 0
total = 0

for seed in range(128):
    for slot in [0, 1]:
        agents = [AGENT, OPPONENT] if slot == 0 else [OPPONENT, AGENT]
        env.reset()
        result = env.run(agents)
        r = [s["reward"] for s in result[-1]]
        won = r[slot] > r[1 - slot]
        wins += int(won)
        total += 1
        if total % 16 == 0:
            print(f"  panel progress: {total}/256  overall {wins}/{total} ({100*wins/total:.1f}%)", flush=True)

print(f"  panel progress: 256/256  overall {wins}/256 ({100*wins/256:.1f}%)")
