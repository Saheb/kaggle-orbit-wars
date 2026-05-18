"""Compare local Orbit Wars heuristic variants on identical seeds."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from statistics import mean, median

from kaggle_environments import make

import main


CHECKPOINTS = (25, 50, 75, 100, 150, 200)

VARIANTS = {
    "legacy": {
        "ENABLE_UNIFIED_SELECTION": False,
        "ENABLE_WAIT_GATE": False,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 160.0,
        "ATTACK_TRAVEL_PENALTY": 0.55,
        "OPENING_ATTACK_TRAVEL_LIMIT": 999.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 999.0,
    },
    "wait": {
        "ENABLE_UNIFIED_SELECTION": False,
        "ENABLE_WAIT_GATE": True,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 160.0,
        "ATTACK_TRAVEL_PENALTY": 0.55,
        "OPENING_ATTACK_TRAVEL_LIMIT": 999.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 999.0,
    },
    "wait_strict": {
        "ENABLE_UNIFIED_SELECTION": False,
        "ENABLE_WAIT_GATE": True,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 120,
        "WAIT_GATE_MIN_PRODUCTION": 3,
        "WAIT_GATE_MARGIN": 20.0,
        "WAIT_GATE_MAX_WAIT": 6,
        "ATTACK_VALUE_HORIZON": 160.0,
        "ATTACK_TRAVEL_PENALTY": 0.55,
        "OPENING_ATTACK_TRAVEL_LIMIT": 999.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 999.0,
    },
    "tempo": {
        "ENABLE_UNIFIED_SELECTION": False,
        "ENABLE_WAIT_GATE": False,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 120.0,
        "ATTACK_TRAVEL_PENALTY": 1.35,
        "OPENING_ATTACK_TRAVEL_LIMIT": 22.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 32.0,
    },
    "tempo_opening": {
        "ENABLE_UNIFIED_SELECTION": False,
        "ENABLE_WAIT_GATE": False,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 135.0,
        "ATTACK_TRAVEL_PENALTY": 1.1,
        "OPENING_ATTACK_TRAVEL_LIMIT": 22.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 999.0,
    },
    "tempo_opening_light": {
        "ENABLE_UNIFIED_SELECTION": False,
        "ENABLE_WAIT_GATE": False,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 145.0,
        "ATTACK_TRAVEL_PENALTY": 0.9,
        "OPENING_ATTACK_TRAVEL_LIMIT": 24.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 999.0,
    },
    "tempo_soft": {
        "ENABLE_UNIFIED_SELECTION": False,
        "ENABLE_WAIT_GATE": False,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 140.0,
        "ATTACK_TRAVEL_PENALTY": 1.0,
        "OPENING_ATTACK_TRAVEL_LIMIT": 28.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 42.0,
    },
    "tempo_light": {
        "ENABLE_UNIFIED_SELECTION": False,
        "ENABLE_WAIT_GATE": False,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 150.0,
        "ATTACK_TRAVEL_PENALTY": 0.8,
        "OPENING_ATTACK_TRAVEL_LIMIT": 35.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 55.0,
    },
    "tempo_wait_strict": {
        "ENABLE_UNIFIED_SELECTION": False,
        "ENABLE_WAIT_GATE": True,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 120,
        "WAIT_GATE_MIN_PRODUCTION": 3,
        "WAIT_GATE_MARGIN": 20.0,
        "WAIT_GATE_MAX_WAIT": 6,
        "ATTACK_VALUE_HORIZON": 120.0,
        "ATTACK_TRAVEL_PENALTY": 1.35,
        "OPENING_ATTACK_TRAVEL_LIMIT": 22.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 32.0,
    },
    "v6_public": {
        "ENABLE_UNIFIED_SELECTION": False,
        "ENABLE_WAIT_GATE": False,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": True,
        "ENABLE_ORBITING_DEFENSE_V6": True,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 160.0,
        "ATTACK_TRAVEL_PENALTY": 0.55,
        "OPENING_ATTACK_TRAVEL_LIMIT": 999.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 999.0,
    },
    "opening_v6": {
        "ENABLE_UNIFIED_SELECTION": False,
        "ENABLE_WAIT_GATE": False,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": True,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 160.0,
        "ATTACK_TRAVEL_PENALTY": 0.55,
        "OPENING_ATTACK_TRAVEL_LIMIT": 999.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 999.0,
    },
    "orbit_defense_v6": {
        "ENABLE_UNIFIED_SELECTION": False,
        "ENABLE_WAIT_GATE": False,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": True,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 160.0,
        "ATTACK_TRAVEL_PENALTY": 0.55,
        "OPENING_ATTACK_TRAVEL_LIMIT": 999.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 999.0,
    },
    "unified": {
        "ENABLE_UNIFIED_SELECTION": True,
        "ENABLE_WAIT_GATE": False,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 160.0,
        "ATTACK_TRAVEL_PENALTY": 0.55,
        "OPENING_ATTACK_TRAVEL_LIMIT": 999.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 999.0,
    },
    "unified_wait": {
        "ENABLE_UNIFIED_SELECTION": True,
        "ENABLE_WAIT_GATE": True,
        "ENABLE_LATENT_THREAT_RESERVE": False,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 160.0,
        "ATTACK_TRAVEL_PENALTY": 0.55,
        "OPENING_ATTACK_TRAVEL_LIMIT": 999.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 999.0,
    },
    "unified_wait_threat": {
        "ENABLE_UNIFIED_SELECTION": True,
        "ENABLE_WAIT_GATE": True,
        "ENABLE_LATENT_THREAT_RESERVE": True,
        "ENABLE_OPENING_RESERVE_V6": False,
        "ENABLE_ORBITING_DEFENSE_V6": False,
        "WAIT_GATE_START": 90,
        "WAIT_GATE_MIN_PRODUCTION": 2,
        "WAIT_GATE_MARGIN": 8.0,
        "WAIT_GATE_MAX_WAIT": 15,
        "ATTACK_VALUE_HORIZON": 160.0,
        "ATTACK_TRAVEL_PENALTY": 0.55,
        "OPENING_ATTACK_TRAVEL_LIMIT": 999.0,
        "MIDGAME_ATTACK_TRAVEL_LIMIT": 999.0,
    },
}


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


def nearest_step(env, checkpoint: int):
    return min(env.steps, key=lambda s: abs(s[0].observation.get("step", 0) - checkpoint))


def player_metrics(env, checkpoint: int):
    step = nearest_step(env, checkpoint)
    obs = step[0].observation
    material = 0
    fleet_ships = 0
    production = 0
    planets = 0
    for planet in obs["planets"]:
        if planet[1] == 0:
            material += int(planet[5])
            production += int(planet[6])
            planets += 1
    for fleet in obs["fleets"]:
        if fleet[1] == 0:
            ships = int(fleet[6])
            material += ships
            fleet_ships += ships
    launched = 0
    for env_step in env.steps:
        if env_step[0].observation.get("step", 0) > checkpoint:
            continue
        launched += sum(int(move[2]) for move in (env_step[0].action or []) if len(move) >= 3)
    return {
        "production": production,
        "planets": planets,
        "material": material,
        "fleet_ratio": 0.0 if material <= 0 else fleet_ships / material,
        "launched": launched,
    }


def final_material(env, players):
    obs = env.steps[-1][0].observation
    totals = [0 for _ in range(players)]
    for planet in obs["planets"]:
        owner = planet[1]
        if 0 <= owner < players:
            totals[owner] += int(planet[5])
    for fleet in obs["fleets"]:
        owner = fleet[1]
        if 0 <= owner < players:
            totals[owner] += int(fleet[6])
    return totals


def run_variant(seed, players, opponent):
    opponent_agent = "random" if opponent == "random" else "starter_nearest.py"
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run([main.agent, *([opponent_agent] * (players - 1))])
    rewards = [state.reward for state in env.steps[-1]]
    winner = max(range(players), key=lambda i: rewards[i])
    checkpoints = {cp: player_metrics(env, cp) for cp in CHECKPOINTS}
    return {
        "winner": winner,
        "rewards": rewards,
        "material": final_material(env, players),
        "checkpoints": checkpoints,
    }


def summarize(name, rows):
    print(f"\n== {name} ==")
    print(f"p0_wins={sum(row['winner'] == 0 for row in rows)}/{len(rows)}")
    final = [row["material"][0] for row in rows]
    print(f"final_material avg/med={mean(final):.1f}/{median(final):.1f}")
    for cp in CHECKPOINTS:
        prod = [row["checkpoints"][cp]["production"] for row in rows]
        material = [row["checkpoints"][cp]["material"] for row in rows]
        launched = [row["checkpoints"][cp]["launched"] for row in rows]
        print(
            f"t={cp:3d} "
            f"prod={mean(prod):5.1f}/{median(prod):4.1f} "
            f"mat={mean(material):6.1f}/{median(material):5.1f} "
            f"launch={mean(launched):6.1f}/{median(launched):5.1f}"
        )


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--players", type=int, choices=[2, 4], default=4)
    parser.add_argument("--opponent", choices=["random", "starter"], default="starter")
    args = parser.parse_args()

    for variant in args.variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant {variant!r}; choices={sorted(VARIANTS)}")
        rows = []
        with variant_flags(VARIANTS[variant]):
            for seed in range(args.start_seed, args.start_seed + args.seeds):
                row = run_variant(seed, args.players, args.opponent)
                rows.append(row)
                print(
                    f"{variant} seed={seed:03d} winner={row['winner']} "
                    f"material={row['material']}"
                )
        summarize(variant, rows)


if __name__ == "__main__":
    main_cli()
