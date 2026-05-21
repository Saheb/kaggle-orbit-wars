"""Orbit Wars environment wrapper using kaggle_environments.

Provides a clean RL interface: reset() -> obs, step(actions) -> obs, reward, done, info.
The kaggle_environments package handles all game logic; we just convert observations.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet,
    Fleet,
    CENTER,
    ROTATION_RADIUS_LIMIT,
    random_agent,
    starter_agent,
)

BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0


class OrbitWarsEnv:
    def __init__(self, num_players=2, seed=None, debug=False, opponent_policy="random"):
        self.num_players = num_players
        self.default_seed = seed
        self.debug = debug
        self.opponent_policy = opponent_policy
        self.env = None

    def reset(self, seed=None):
        if seed is None:
            seed = self.default_seed if self.default_seed is not None else np.random.randint(0, 2**31)
        config = {"seed": int(seed)}
        self.env = make("orbit_wars", configuration=config, debug=self.debug)
        self.env.reset(self.num_players)
        self.done = False
        return self._get_obs(0)

    def step(self, agent_actions, opponent_actions=None):
        """agent_actions: list of [from_planet_id, angle, ships] for player 0.
        opponent_actions: optional list of actions for players 1..N-1.
        If None, opponents use this wrapper's configured opponent policy."""
        if opponent_actions is not None:
            actions = [agent_actions] + list(opponent_actions)
        else:
            actions = [agent_actions] + [
                self._opponent_actions(player)
                for player in range(1, self.num_players)
            ]
        self.env.step(actions)
        self.done = self.env.steps[-1][0].status == "DONE"
        obs = self._get_obs(0)
        reward = self._compute_reward()
        return obs, reward, self.done, {}

    def _opponent_actions(self, player):
        obs = self._get_obs(player)
        policy = self.opponent_policy
        if policy in (None, "none"):
            return []
        if policy == "random":
            return random_agent(obs)
        if policy == "starter":
            return starter_agent(obs)
        if callable(policy):
            return policy(obs)
        raise ValueError(f"Unknown opponent_policy: {policy!r}")

    def _get_obs(self, player):
        obs = self.env.steps[-1][player].observation
        planets = []
        for p in (obs.planets if hasattr(obs, 'planets') else obs.get("planets", [])):
            if isinstance(p, Planet):
                planets.append([p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production])
            else:
                planets.append(list(p))

        fleets = []
        for f in (obs.fleets if hasattr(obs, 'fleets') else obs.get("fleets", [])):
            if isinstance(f, Fleet):
                fleets.append([f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships])
            else:
                fleets.append(list(f))

        initial_planets = []
        for p in (obs.initial_planets if hasattr(obs, 'initial_planets') else obs.get("initial_planets", [])):
            if isinstance(p, Planet):
                initial_planets.append([p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production])
            else:
                initial_planets.append(list(p))

        comet_planet_ids = list(obs.comet_planet_ids if hasattr(obs, 'comet_planet_ids') else obs.get("comet_planet_ids", []))
        angular_velocity = float(obs.angular_velocity if hasattr(obs, 'angular_velocity') else obs.get("angular_velocity", 0.0))
        step = int(obs.step if hasattr(obs, 'step') else obs.get("step", 0))

        return {
            "step": step,
            "player": player,
            "planets": planets,
            "fleets": fleets,
            "angular_velocity": angular_velocity,
            "initial_planets": initial_planets,
            "comet_planet_ids": comet_planet_ids,
        }

    def _compute_reward(self):
        if not self.done:
            return 0.0
        final = self.env.steps[-1]
        rewards = [s.reward for s in final]
        return float(rewards[0]) if rewards[0] is not None else 0.0

    def get_obs_for_player(self, player):
        return self._get_obs(player)

    def compute_material(self, player):
        obs = self._get_obs(player)
        total = 0.0
        for p in obs["planets"]:
            if p[1] == player:
                total += p[5]
        for f in obs["fleets"]:
            if f[1] == player:
                total += f[6]
        return total

    def run_with_agent(self, agent_fn, opponent="random", num_episodes=1):
        results = []
        for seed in range(num_episodes):
            self.reset(seed=seed)
            done = False
            while not done:
                obs = self._get_obs(0)
                actions = agent_fn(obs)
                _, reward, done, info = self.step(actions)
            final_reward = self._compute_reward()
            results.append({"seed": seed, "reward": final_reward, "material": self.compute_material(0)})
        return results
