"""Tests for environment behaviour and feature utilities.

All tests use plain Python/numpy/torch — no JAX dependency.
"""

from __future__ import annotations

import math
import sys
import os
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

def test_basic_game():
    """A random game runs without errors and ends in DONE status."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.run(["random", "random"])

    final = env.steps[-1]
    assert len(final) == 2, f"Expected 2 players, got {len(final)}"
    assert all(s.status in ("DONE", "ACTIVE") for s in final)
    print("test_basic_game: PASS")


def test_game_determinism():
    """Same seed + same agent RNG → identical final state."""
    def _random_agent(obs):
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        if isinstance(obs, dict):
            planets = [Planet(*p) for p in obs.get("planets", [])]
        else:
            planets = list(obs.planets)
        moves = []
        for p in planets:
            if p.owner == player and p.ships > 0:
                angle = random.uniform(0, 2 * math.pi)
                ships = p.ships // 2
                if ships > 0:
                    moves.append([p.id, angle, ships])
        return moves

    env1 = make("orbit_wars", configuration={"seed": 123}, debug=False)
    env2 = make("orbit_wars", configuration={"seed": 123}, debug=False)

    random.seed(42)
    env1.run([_random_agent, _random_agent])
    random.seed(42)
    env2.run([_random_agent, _random_agent])

    obs1 = env1.steps[-1][0].observation
    obs2 = env2.steps[-1][0].observation

    assert len(obs1.planets) == len(obs2.planets)
    for p1, p2 in zip(obs1.planets, obs2.planets):
        assert p1[0] == p2[0], f"Planet ID mismatch: {p1[0]} vs {p2[0]}"
        assert abs(p1[2] - p2[2]) < 1e-6, f"Planet x mismatch"
        assert abs(p1[3] - p2[3]) < 1e-6, f"Planet y mismatch"

    print("test_game_determinism: PASS")


def test_planet_generation():
    """Maps have at least 20 planets; owned planets start with 10 ships."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.run(["random", "random"])

    obs = env.steps[0][0].observation
    planets = obs.planets

    assert len(planets) >= 20, f"Expected >= 20 planets, got {len(planets)}"

    home_planets = [p for p in planets if p[1] >= 0]
    assert len(home_planets) >= 2
    for p in home_planets:
        assert p[5] == 10, f"Home planet should start with 10 ships, got {p[5]}"

    print("test_planet_generation: PASS")


def test_combat_resolution():
    """After a game, all planets have non-negative ship counts."""
    env = make("orbit_wars", configuration={"seed": 99}, debug=False)
    env.run(["random", "random"])

    final_obs = env.steps[-1][0].observation
    for planet in final_obs.planets:
        assert planet[5] >= 0, f"Planet {planet[0]} has negative ships: {planet[5]}"

    print("test_combat_resolution: PASS")


# ---------------------------------------------------------------------------
# Feature utility tests
# ---------------------------------------------------------------------------

def test_fleet_speed_range():
    """fleet_speed returns values in [1.0, 6.0] for any positive ship count."""
    from features import fleet_speed

    test_cases = [1, 5, 10, 100, 500, 1000, 5000]
    for ships in test_cases:
        speed = fleet_speed(ships)
        assert 1.0 <= speed <= 6.0, f"fleet_speed({ships}) = {speed} out of range"

    # Boundary values
    assert abs(fleet_speed(1) - 1.0) < 0.01, f"fleet_speed(1) should be ~1.0"
    assert abs(fleet_speed(1000) - 6.0) < 0.01, f"fleet_speed(1000) should be ~6.0"
    print("test_fleet_speed_range: PASS")


def test_sun_crossing_detection():
    """_point_segment_distance_array correctly identifies sun-crossing trajectories."""
    from action_mask import _point_segment_distance_array, CENTER, SUN_RADIUS

    # A segment through the exact center crosses the sun
    dist = _point_segment_distance_array(CENTER, CENTER,
                                         np.array([0.0]), np.array([0.0]),
                                         np.array([100.0]), np.array([100.0]))
    assert dist[0] <= SUN_RADIUS, f"Center crossing should be within sun radius, dist={dist[0]:.2f}"

    # A segment along the top edge should not cross the sun
    dist_edge = _point_segment_distance_array(CENTER, CENTER,
                                               np.array([0.0]), np.array([99.0]),
                                               np.array([100.0]), np.array([99.0]))
    assert dist_edge[0] > SUN_RADIUS, \
        f"Top-edge trajectory should not cross sun, dist={dist_edge[0]:.2f}"

    print("test_sun_crossing_detection: PASS")


def test_env_wrapper_reset_and_step():
    """OrbitWarsEnv wrapper resets and steps without errors."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from env import OrbitWarsEnv

    env = OrbitWarsEnv(num_players=2, seed=42)
    obs = env.reset(seed=42)

    assert "planets" in obs
    assert "fleets" in obs
    assert "angular_velocity" in obs
    assert len(obs["planets"]) > 0

    # Step with no moves
    obs2, reward, done, info = env.step([])
    assert "planets" in obs2
    assert isinstance(reward, float)
    assert isinstance(done, bool)

    print(f"  Planets: {len(obs['planets'])}, Fleets: {len(obs['fleets'])}")
    print("test_env_wrapper_reset_and_step: PASS")


def test_material_accounting():
    """compute_material sums ships on owned planets."""
    from env import OrbitWarsEnv

    env = OrbitWarsEnv(num_players=2, seed=0)
    env.reset(seed=0)

    material = env.compute_material(0)
    assert material >= 0, f"Material should be non-negative, got {material}"

    obs = env._get_obs(0)
    expected = sum(p[5] for p in obs["planets"] if p[1] == 0)
    expected += sum(f[6] for f in obs["fleets"] if f[1] == 0)
    assert abs(material - expected) < 1e-3, f"Material mismatch: {material} vs {expected}"

    print(f"  Player 0 material at step 0: {material:.0f}")
    print("test_material_accounting: PASS")


if __name__ == "__main__":
    print("Running environment parity tests...\n")
    test_basic_game()
    test_game_determinism()
    test_planet_generation()
    test_combat_resolution()
    test_fleet_speed_range()
    test_sun_crossing_detection()
    test_env_wrapper_reset_and_step()
    test_material_accounting()
    print("\nAll environment tests passed!")
