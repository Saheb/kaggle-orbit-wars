"""Fast in-process Orbit Wars simulator.

This mirrors the official Kaggle interpreter while avoiding the generic
`kaggle_environments.make(...).step(...)` machinery in PPO rollouts. Keep
Kaggle evaluation as the source of truth; this backend is for faster training
feedback only and should stay covered by parity tests.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    generate_comet_paths,
    generate_planets,
    point_to_segment_distance,
    random_agent,
    starter_agent,
    swept_pair_hit,
)


class FastOrbitWarsEnv:
    """Minimal RL wrapper with the same public surface as `OrbitWarsEnv`."""

    def __init__(
        self,
        num_players: int = 2,
        seed: int | None = None,
        debug: bool = False,
        opponent_policy: str | Callable = "random",
        ship_speed: float = 6.0,
        comet_speed: float = 4.0,
        episode_steps: int = 500,
    ):
        self.num_players = num_players
        self.default_seed = seed
        self.debug = debug
        self.opponent_policy = opponent_policy
        self.ship_speed = ship_speed
        self.comet_speed = comet_speed
        self.episode_steps = episode_steps
        self.done = False
        self.rewards = [0.0] * num_players

    def reset(self, seed: int | None = None):
        if seed is None:
            seed = self.default_seed if self.default_seed is not None else np.random.randint(0, 2**31)
        self.seed = int(seed)
        init_rng = random.Random(self.seed)

        self.step_count = 0
        self.angular_velocity = init_rng.uniform(0.025, 0.05)
        self.planets = generate_planets(init_rng)
        self.initial_planets = [p.copy() for p in self.planets]
        self.fleets = []
        self.next_fleet_id = 0
        self.comets = []
        self.comet_planet_ids = []
        self.done = False
        self.rewards = [0.0] * self.num_players

        num_groups = len(self.planets) // 4
        if num_groups > 0:
            home_group = init_rng.randint(0, num_groups - 1)
            base = home_group * 4
            if self.num_players == 2:
                self.planets[base][1] = 0
                self.planets[base][5] = 10
                self.planets[base + 3][1] = 1
                self.planets[base + 3][5] = 10
            elif self.num_players == 4:
                for j in range(4):
                    self.planets[base + j][1] = j
                    self.planets[base + j][5] = 10

        return self._get_obs(0)

    def step(self, agent_actions, opponent_actions=None):
        if self.done:
            return self._get_obs(0), self._compute_reward(), self.done, {}

        if opponent_actions is not None:
            actions = [agent_actions] + list(opponent_actions)
        else:
            actions = [agent_actions] + [
                self._opponent_actions(player)
                for player in range(1, self.num_players)
            ]

        self._expire_comets_before_launch()
        self._spawn_comets_if_due()
        for player_id, action in enumerate(actions):
            self._process_moves(player_id, action)
        self._produce()
        self._advance_world_and_resolve_combat()
        self.step_count += 1
        self._maybe_terminate()

        return self._get_obs(0), self._compute_reward(), self.done, {}

    def _get_obs(self, player: int):
        return {
            "step": self.step_count,
            "player": player,
            "planets": [p.copy() for p in self.planets],
            "fleets": [f.copy() for f in self.fleets],
            "angular_velocity": self.angular_velocity,
            "initial_planets": [p.copy() for p in self.initial_planets],
            "comet_planet_ids": list(self.comet_planet_ids),
        }

    def get_obs_for_player(self, player: int):
        return self._get_obs(player)

    def _opponent_actions(self, player: int):
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

    def _expire_comets_before_launch(self):
        expired = []
        for group in self.comets:
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                if idx >= len(group["paths"][i]):
                    expired.append(pid)
        if expired:
            self._remove_planets(set(expired))

    def _spawn_comets_if_due(self):
        next_step = self.step_count + 1
        if next_step not in COMET_SPAWN_STEPS:
            return

        comet_rng = random.Random(f"orbit_wars-comet-{self.seed}-{next_step}")
        comet_paths = generate_comet_paths(
            self.initial_planets,
            self.angular_velocity,
            next_step,
            self.comet_planet_ids,
            self.comet_speed,
            rng=comet_rng,
        )
        if not comet_paths:
            return

        next_id = max(p[0] for p in self.planets) + 1
        comet_ships = min(
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
        )
        group = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
        for i in range(4):
            pid = next_id + i
            group["planet_ids"].append(pid)
            self.comet_planet_ids.append(pid)
            planet = [pid, -1, -99, -99, COMET_RADIUS, comet_ships, COMET_PRODUCTION]
            self.planets.append(planet)
            self.initial_planets.append(planet[:])
        self.comets.append(group)

    def _process_moves(self, player_id: int, action):
        if not action or not isinstance(action, list):
            return
        by_id = {p[0]: p for p in self.planets}
        for move in action:
            if not isinstance(move, (list, tuple)) or len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)
            from_planet = by_id.get(from_id)
            if from_planet and from_planet[1] == player_id and from_planet[5] >= ships and ships > 0:
                from_planet[5] -= ships
                start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
                start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
                self.fleets.append([
                    self.next_fleet_id,
                    player_id,
                    start_x,
                    start_y,
                    angle,
                    from_id,
                    ships,
                ])
                self.next_fleet_id += 1

    def _produce(self):
        for planet in self.planets:
            if planet[1] != -1:
                planet[5] += planet[6]

    def _advance_world_and_resolve_combat(self):
        current_step = self.step_count
        comet_pid_set = set(self.comet_planet_ids)
        initial_by_id = {p[0]: p for p in self.initial_planets}
        planet_paths = {}
        expired_comet_pids = []

        for planet in self.planets:
            if planet[0] in comet_pid_set:
                continue
            old_pos = (planet[2], planet[3])
            new_pos = old_pos
            initial_p = initial_by_id.get(planet[0])
            if initial_p is not None:
                dx = initial_p[2] - CENTER
                dy = initial_p[3] - CENTER
                orbital_r = math.sqrt(dx * dx + dy * dy)
                if orbital_r + planet[4] < ROTATION_RADIUS_LIMIT:
                    initial_angle = math.atan2(dy, dx)
                    current_angle = initial_angle + self.angular_velocity * current_step
                    new_pos = (
                        CENTER + orbital_r * math.cos(current_angle),
                        CENTER + orbital_r * math.sin(current_angle),
                    )
            planet_paths[planet[0]] = (old_pos, new_pos, True)

        for group in self.comets:
            group["path_index"] += 1
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                planet = next((p for p in self.planets if p[0] == pid), None)
                if planet is None:
                    continue
                p_path = group["paths"][i]
                old_pos = (planet[2], planet[3])
                if idx >= len(p_path):
                    expired_comet_pids.append(pid)
                    planet_paths[pid] = (old_pos, old_pos, True)
                else:
                    new_pos = (p_path[idx][0], p_path[idx][1])
                    planet_paths[pid] = (old_pos, new_pos, old_pos[0] >= 0)

        fleets_to_remove = []
        combat_lists = {p[0]: [] for p in self.planets}
        for fleet in self.fleets:
            angle = fleet[4]
            ships = fleet[6]
            speed = 1.0 + (self.ship_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
            speed = min(speed, self.ship_speed)
            old_pos = (fleet[2], fleet[3])
            fleet[2] += math.cos(angle) * speed
            fleet[3] += math.sin(angle) * speed
            new_pos = (fleet[2], fleet[3])

            hit_planet = False
            for planet in self.planets:
                path = planet_paths.get(planet[0])
                if path is None or not path[2]:
                    continue
                p_old, p_new, _ = path
                if swept_pair_hit(old_pos, new_pos, p_old, p_new, planet[4]):
                    combat_lists[planet[0]].append(fleet)
                    fleets_to_remove.append(fleet)
                    hit_planet = True
                    break
            if hit_planet:
                continue

            if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
                fleets_to_remove.append(fleet)
                continue
            if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
                fleets_to_remove.append(fleet)

        for planet in self.planets:
            path = planet_paths.get(planet[0])
            if path is not None:
                planet[2], planet[3] = path[1]

        if expired_comet_pids:
            self._remove_planets(set(expired_comet_pids))

        remove_ids = {id(f) for f in fleets_to_remove}
        self.fleets = [f for f in self.fleets if id(f) not in remove_ids]
        self._resolve_combat(combat_lists)

    def _resolve_combat(self, combat_lists):
        by_id = {p[0]: p for p in self.planets}
        for pid, planet_fleets in combat_lists.items():
            planet = by_id.get(pid)
            if not planet or not planet_fleets:
                continue

            player_ships = {}
            for fleet in planet_fleets:
                owner = fleet[1]
                player_ships[owner] = player_ships.get(owner, 0) + fleet[6]
            sorted_players = sorted(player_ships.items(), key=lambda item: item[1], reverse=True)
            top_player, top_ships = sorted_players[0]

            if len(sorted_players) > 1:
                survivor_ships = top_ships - sorted_players[1][1]
                if sorted_players[0][1] == sorted_players[1][1]:
                    survivor_ships = 0
                survivor_owner = top_player if survivor_ships > 0 else -1
            else:
                survivor_owner = top_player
                survivor_ships = top_ships

            if survivor_ships > 0:
                if planet[1] == survivor_owner:
                    planet[5] += survivor_ships
                else:
                    planet[5] -= survivor_ships
                    if planet[5] < 0:
                        planet[1] = survivor_owner
                        planet[5] = abs(planet[5])

    def _remove_planets(self, expired_set: set[int]):
        self.planets = [p for p in self.planets if p[0] not in expired_set]
        self.initial_planets = [p for p in self.initial_planets if p[0] not in expired_set]
        self.comet_planet_ids = [pid for pid in self.comet_planet_ids if pid not in expired_set]
        for group in self.comets:
            group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired_set]
        self.comets = [g for g in self.comets if g["planet_ids"]]

    def _maybe_terminate(self):
        terminated = self.step_count >= self.episode_steps - 1

        alive_players = set()
        for planet in self.planets:
            if planet[1] != -1:
                alive_players.add(planet[1])
        for fleet in self.fleets:
            alive_players.add(fleet[1])
        if len(alive_players) <= 1:
            terminated = True

        if not terminated:
            return

        self.done = True
        scores = [0] * self.num_players
        for planet in self.planets:
            if planet[1] != -1:
                scores[planet[1]] += planet[5]
        for fleet in self.fleets:
            scores[fleet[1]] += fleet[6]
        max_score = max(scores)
        self.rewards = [1.0 if score == max_score and max_score > 0 else -1.0 for score in scores]

    def _compute_reward(self):
        return float(self.rewards[0]) if self.done else 0.0

    def compute_material(self, player: int):
        total = 0.0
        for planet in self.planets:
            if planet[1] == player:
                total += planet[5]
        for fleet in self.fleets:
            if fleet[1] == player:
                total += fleet[6]
        return total

    def run_with_agent(self, agent_fn, opponent="random", num_episodes=1):
        old_policy = self.opponent_policy
        self.opponent_policy = opponent
        results = []
        try:
            for seed in range(num_episodes):
                self.reset(seed=seed)
                done = False
                while not done:
                    obs = self._get_obs(0)
                    actions = agent_fn(obs)
                    _, _, done, _ = self.step(actions)
                results.append({
                    "seed": seed,
                    "reward": self._compute_reward(),
                    "material": self.compute_material(0),
                })
        finally:
            self.opponent_policy = old_policy
        return results
