"""One-shot: collect teacher trajectories, convert to BC samples, save .pkl."""
import argparse
import pickle
from bc import collect_heuristic_trajectories, trajectory_to_training_sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--num-games", type=int, default=200)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"Collecting {args.num_games} games from {args.agent}...")
    trajs = collect_heuristic_trajectories(args.agent, num_games=args.num_games,
                                            opponent="random", verbose=False)
    print(f"  {len(trajs)} raw transitions")
    samples = []
    for t in trajs:
        s = trajectory_to_training_sample(t)
        if s is not None:
            samples.append(s)
    print(f"  {len(samples)} usable samples")
    with open(args.output, "wb") as f:
        pickle.dump(samples, f)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
