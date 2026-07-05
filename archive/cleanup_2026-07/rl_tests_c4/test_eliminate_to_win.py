"""Tests for --eliminate-to-win (torch_env._check_done).

Default (off): timeout winner = most ships (+1 / -1), legacy behaviour preserved.
On: a win counts ONLY on an elimination (few_left); a timeout without one is a draw
(reward 0 for both). Elimination still pays ±1. This removes the hoard-to-timeout
"most ships wins" attractor that pins self-play in the under-mass Nash.
project_force_concentration_wall.

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_eliminate_to_win.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch_env import VecTorchEnv


def _build(eliminate_to_win):
    """Two envs: [0] = timeout, both alive (P0 has more ships); [1] = P1 eliminated."""
    env = VecTorchEnv(num_envs=2, num_players=2, device="cpu", action_decode="angle",
                      eliminate_to_win=eliminate_to_win)
    env.reset(seeds=[0, 1])           # scenario_fraction 0 -> scenario_id all OFF (inert)
    env.planet_alive[:, :] = False
    env.fleet_alive[:, :] = False
    env.done[:] = False

    # env 0 — timeout, no elimination: P0 owns 2 planets (100 ships), P1 owns 1 (30).
    env.planets[0, 0, 1], env.planets[0, 0, 5], env.planet_alive[0, 0] = 0, 60.0, True
    env.planets[0, 1, 1], env.planets[0, 1, 5], env.planet_alive[0, 1] = 0, 40.0, True
    env.planets[0, 2, 1], env.planets[0, 2, 5], env.planet_alive[0, 2] = 1, 30.0, True
    env.step_count[0] = env.episode_steps - 1     # time_up, n_alive = 2 (not few_left)

    # env 1 — elimination: only P0 alive (50 ships), P1 has nothing.
    env.planets[1, 0, 1], env.planets[1, 0, 5], env.planet_alive[1, 0] = 0, 50.0, True
    env.step_count[1] = 10                         # not time_up; few_left drives the done

    rewards, done = env._check_done()
    return rewards, done


def test_default_off_is_legacy():
    rewards, done = _build(eliminate_to_win=False)
    assert done[0] and done[1], f"both envs should be done, got {done.tolist()}"
    # timeout: most ships wins (legacy ±1)
    assert torch.allclose(rewards[0], torch.tensor([1.0, -1.0])), f"env0 {rewards[0].tolist()}"
    # elimination: P0 wins
    assert torch.allclose(rewards[1], torch.tensor([1.0, -1.0])), f"env1 {rewards[1].tolist()}"
    print("OK default-off legacy: timeout +1/-1, elim +1/-1")


def test_on_timeout_draws_elim_wins():
    rewards, done = _build(eliminate_to_win=True)
    assert done[0] and done[1], f"both envs should be done, got {done.tolist()}"
    # timeout without elimination = draw (0/0) — the hoarding attractor is gone
    assert torch.allclose(rewards[0], torch.tensor([0.0, 0.0])), f"env0 {rewards[0].tolist()}"
    # elimination still pays ±1
    assert torch.allclose(rewards[1], torch.tensor([1.0, -1.0])), f"env1 {rewards[1].tolist()}"
    print("OK eliminate-to-win: timeout draw 0/0, elim +1/-1")


def test_on_last_wins_attribution():
    env = VecTorchEnv(num_envs=2, num_players=2, device="cpu", action_decode="angle",
                      eliminate_to_win=True)
    env.reset(seeds=[0, 1])
    env.planet_alive[:, :] = False
    env.fleet_alive[:, :] = False
    env.done[:] = False
    env.planets[0, 0, 1], env.planets[0, 0, 5], env.planet_alive[0, 0] = 0, 60.0, True
    env.planets[0, 2, 1], env.planets[0, 2, 5], env.planet_alive[0, 2] = 1, 30.0, True
    env.step_count[0] = env.episode_steps - 1
    env.planets[1, 0, 1], env.planets[1, 0, 5], env.planet_alive[1, 0] = 0, 50.0, True
    env.step_count[1] = 10
    env._check_done()
    # timeout draw -> no winner credited; elimination -> P0 credited
    assert not env._last_wins[0].any(), f"draw should credit no winner: {env._last_wins[0].tolist()}"
    assert env._last_wins[1, 0] and not env._last_wins[1, 1], f"elim: {env._last_wins[1].tolist()}"
    print("OK _last_wins (PFSP credit): draw=no winner, elim=P0")


def _build_pc(coef):
    """Like _build but eliminate_to_win + timeout_planet_coef=coef.
    env0 timeout: P0 owns 2 planets, P1 owns 1 (planet margin, NOT ship-driven)."""
    env = VecTorchEnv(num_envs=2, num_players=2, device="cpu", action_decode="angle",
                      eliminate_to_win=True, timeout_planet_coef=coef)
    env.reset(seeds=[0, 1])
    env.planet_alive[:, :] = False
    env.fleet_alive[:, :] = False
    env.done[:] = False
    env.planets[0, 0, 1], env.planets[0, 0, 5], env.planet_alive[0, 0] = 0, 60.0, True
    env.planets[0, 1, 1], env.planets[0, 1, 5], env.planet_alive[0, 1] = 0, 40.0, True
    env.planets[0, 2, 1], env.planets[0, 2, 5], env.planet_alive[0, 2] = 1, 30.0, True
    env.step_count[0] = env.episode_steps - 1
    env.planets[1, 0, 1], env.planets[1, 0, 5], env.planet_alive[1, 0] = 0, 50.0, True
    env.step_count[1] = 10
    rewards, done = env._check_done()
    return env, rewards, done


def test_timeout_planet_margin():
    # P0 2 planets, P1 1 planet → margin P0=(2*2-3)/3=1/3, coef 0.5 → +1/6 / -1/6 (zero-sum).
    _, rewards, done = _build_pc(0.5)
    assert done[0] and done[1]
    assert torch.allclose(rewards[0], torch.tensor([1.0/6, -1.0/6]), atol=1e-5), f"env0 {rewards[0].tolist()}"
    # elimination still pays the full ±1 (strictly better than any planet-margin timeout)
    assert torch.allclose(rewards[1], torch.tensor([1.0, -1.0])), f"env1 {rewards[1].tolist()}"
    print("OK timeout-planet-margin: timeout +1/6 / -1/6 (planet-share), elim +1/-1")


def test_timeout_planet_leader_credited():
    env, _, _ = _build_pc(0.5)
    # strict planet-leader (P0, 2>1) credited as PFSP winner on the timeout; elim P0 too.
    assert env._last_wins[0, 0] and not env._last_wins[0, 1], f"env0 {env._last_wins[0].tolist()}"
    assert env._last_wins[1, 0] and not env._last_wins[1, 1], f"env1 {env._last_wins[1].tolist()}"
    print("OK timeout-planet-margin _last_wins: planet-leader credited")


def test_timeout_planet_tie_is_draw():
    # Equal planets at timeout → margin 0, reward 0/0, no winner (not a strict lead).
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu", action_decode="angle",
                      eliminate_to_win=True, timeout_planet_coef=0.5)
    env.reset(seeds=[0]); env.planet_alive[:, :] = False; env.fleet_alive[:, :] = False; env.done[:] = False
    env.planets[0, 0, 1], env.planets[0, 0, 5], env.planet_alive[0, 0] = 0, 90.0, True  # P0 1 planet, many ships
    env.planets[0, 1, 1], env.planets[0, 1, 5], env.planet_alive[0, 1] = 1, 10.0, True  # P1 1 planet, few ships
    env.step_count[0] = env.episode_steps - 1
    rewards, _ = env._check_done()
    assert torch.allclose(rewards[0], torch.tensor([0.0, 0.0])), f"tie {rewards[0].tolist()}"
    assert not env._last_wins[0].any(), f"tie should credit no winner: {env._last_wins[0].tolist()}"
    print("OK timeout-planet-margin tie: equal planets = 0/0 draw (ships ignored)")


if __name__ == "__main__":
    test_default_off_is_legacy()
    test_on_timeout_draws_elim_wins()
    test_on_last_wins_attribution()
    test_timeout_planet_margin()
    test_timeout_planet_leader_credited()
    test_timeout_planet_tie_is_draw()
    print("\nall eliminate-to-win tests passed")
