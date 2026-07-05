"""Regression test for the RAW winner exposed for per-episode pool attribution.

Context: the per-EPISODE pool assignment fix (train_torch) credits each finished pool
env to its assigned member using the RAW (pre-shaping) winner, NOT the shaped `rewards`
tensor returned by step() (which by then carries material/expansion/early-capture/etc.
shaping). The primitive that makes that possible is `env._last_wins`, stashed in
_check_done from the score comparison BEFORE any win-margin/speed/shaping bonus.

This asserts `_last_wins` matches the score-based winner on terminating envs, and that it
is independent of the shaping coefficients (win_margin / expansion / early-capture) — i.e.
turning shaping on does not change who `_last_wins` says won.

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_pool_episode_attribution.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch_env import VecTorchEnv, MAX_OWNED


def _build(**env_kwargs):
    """2-env env: env0 -> seat 0 out-ships seat 1; env1 -> seat 1 out-ships seat 0.
    Both forced to terminate this step (step_count = episode_steps - 1)."""
    env = VecTorchEnv(num_envs=2, num_players=2, device="cpu", action_decode="angle",
                      **env_kwargs)
    env.reset(seeds=[0, 1])
    # Wipe to a controlled 2-planet board per env.
    env.planet_alive[:] = False
    env.fleet_alive[:] = False
    env.fleets[:] = 0
    # env 0: planet0 = seat0 w/ 50 ships, planet1 = seat1 w/ 10 ships  -> seat0 wins
    env.planets[0, 0, 1] = 0; env.planets[0, 0, 5] = 50; env.planet_alive[0, 0] = True
    env.planets[0, 1, 1] = 1; env.planets[0, 1, 5] = 10; env.planet_alive[0, 1] = True
    # env 1: mirror -> seat1 wins
    env.planets[1, 0, 1] = 0; env.planets[1, 0, 5] = 10; env.planet_alive[1, 0] = True
    env.planets[1, 1, 1] = 1; env.planets[1, 1, 5] = 50; env.planet_alive[1, 1] = True
    env.step_count[:] = env.episode_steps - 1   # force time_up this step
    return env


def _noop_actions():
    fire = torch.zeros(2, MAX_OWNED, dtype=torch.long)
    rest = torch.zeros(2, MAX_OWNED, dtype=torch.long)
    a = torch.stack([fire, rest, rest], dim=-1)
    return {0: a, 1: a}


def test_last_wins_matches_scores():
    env = _build()
    _, rewards, done = env.step(_noop_actions())
    assert done.all(), f"both envs should terminate (time_up), got {done}"
    lw = env._last_wins
    assert lw is not None, "_last_wins not set by _check_done"
    # env0 -> seat0 won; env1 -> seat1 won
    assert bool(lw[0, 0]) and not bool(lw[0, 1]), f"env0 winner wrong: {lw[0]}"
    assert bool(lw[1, 1]) and not bool(lw[1, 0]), f"env1 winner wrong: {lw[1]}"
    print("ok: _last_wins matches the score-based winner")


def test_last_wins_independent_of_shaping():
    # Heavy shaping on; the raw winner must be unchanged (it's read pre-bonus).
    env = _build(win_margin_coeff=5.0, expansion_coef=1.0, early_capture_coef=1.0)
    _, rewards, done = env.step(_noop_actions())
    assert done.all()
    lw = env._last_wins
    assert bool(lw[0, 0]) and not bool(lw[0, 1]), f"env0 winner changed under shaping: {lw[0]}"
    assert bool(lw[1, 1]) and not bool(lw[1, 0]), f"env1 winner changed under shaping: {lw[1]}"
    # And the SHAPED reward really did diverge from raw +-1 (proving the bonus is live,
    # i.e. comparing shaped reward would have been the wrong primitive).
    assert rewards[0, 0].item() > 1.0, f"expected win_margin bonus on winner, got {rewards[0,0].item()}"
    print("ok: _last_wins is independent of win_margin/expansion/early-capture shaping")


if __name__ == "__main__":
    test_last_wins_matches_scores()
    test_last_wins_independent_of_shaping()
    print("ALL PASS")
