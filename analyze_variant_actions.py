"""Compare candidate action choices on one observation without running a full match."""

from __future__ import annotations

import argparse
from contextlib import contextmanager

from kaggle_environments import make

import main
from evaluate_ablation import VARIANTS


@contextmanager
def variant_flags(settings):
    old = {name: getattr(main, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(main, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(main, name, value)


def variant_agent(name):
    settings = VARIANTS[name]

    def agent(obs):
        with variant_flags(settings):
            return main.agent(obs)

    return agent


def action_summary(obs, action):
    planets = {planet[0]: planet for planet in obs["planets"]}
    rows = []
    for move in action or []:
        if len(move) < 3:
            continue
        src_id, angle, ships = move
        src = planets.get(src_id)
        rows.append(
            {
                "src": int(src_id),
                "src_owner": None if src is None else int(src[1]),
                "src_ships": None if src is None else int(src[5]),
                "src_prod": None if src is None else int(src[6]),
                "angle": round(float(angle), 3),
                "ships": int(ships),
            }
        )
    return rows


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--candidate", default="wait_strict", choices=sorted(VARIANTS))
    parser.add_argument("--baseline", default="legacy", choices=sorted(VARIANTS))
    parser.add_argument("--until", type=int, default=160)
    args = parser.parse_args()

    agents = [variant_agent(args.baseline) for _ in range(4)]
    agents[args.slot] = variant_agent(args.candidate)
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=True)
    env.run(agents)

    previous = None
    for step in env.steps:
        obs = step[args.slot].observation
        turn = int(obs.get("step", 0))
        if turn > args.until:
            break
        candidate_action = step[args.slot].action or []
        with variant_flags(VARIANTS[args.baseline]):
            baseline_action = main.agent(obs)
        if candidate_action == baseline_action:
            continue
        key = (turn, tuple(tuple(m) for m in candidate_action), tuple(tuple(m) for m in baseline_action))
        if key == previous:
            continue
        previous = key
        print(f"step={turn}")
        print(f"  {args.candidate}: {action_summary(obs, candidate_action)}")
        print(f"  {args.baseline}: {action_summary(obs, baseline_action)}")


if __name__ == "__main__":
    main_cli()
