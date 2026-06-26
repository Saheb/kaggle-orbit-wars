"""Regression test: torch_env must stay byte-faithful to the real kaggle engine,
INCLUDING comets (spawned at steps 50/150/... — collidable, capturable, moving).

This guards the 2026-06-13 fix that closed the train/eval simulator gap (docs/train-eval.md):
torch_env previously did not simulate comets, so every game diverged from kaggle after step 50.
The probe replays a real kaggle game's exact action stream in torch_env and asserts the states
match every step until the game's natural end.

Run: orbit_wars_rl/.venv/bin/python -m orbit_wars_rl.tests.test_comet_fidelity
"""
import orbit_wars_rl.research.sim_gap_probe as P

HAMMER = "opponents/orbit-wars-heuristic-bots/14_main_k_v2_lb1152_LAST_HEURISTIC.py"
HAMMER2 = "opponents/orbit-wars-heuristic-bots/11_v14_1n_lb1138_doom_evac_mega_hammer.py"


def test_comet_fidelity():
    # Seeds whose hammer-vs-hammer games run past the step-50 (and step-150) comet spawns.
    for seed in [12345, 777, 2024, 99, 4242]:
        res = P.run_full_game_replay(seed, HAMMER, HAMMER2, max_steps=260)
        # "END" = matched every step until kaggle declared the game DONE; None = matched the
        # full window with no game end. An int = a real physics divergence at that step (FAIL).
        assert res in ("END", None), (
            f"seed {seed}: torch_env DIVERGED from kaggle at step {res} "
            f"(comet/physics fidelity regression)"
        )
    print("PASS: torch_env is byte-faithful to kaggle (comets included) across all seeds")


def test_orbital_and_fleet_match():
    for seed in [12345, 2024]:
        assert P.run_orbital_diff(seed, n_steps=25) is None, f"orbital motion diverged (seed {seed})"
    print("PASS: orbital motion matches")


def test_fast_comet_paths_byte_identical():
    """The vectorized _comet_paths_fast must produce output BYTE-IDENTICAL to kaggle's
    generate_comet_paths (else the resample boundaries / collisions drift)."""
    import random
    from kaggle_environments.envs.orbit_wars.orbit_wars import generate_comet_paths, generate_planets
    from orbit_wars_rl.torch_env import _comet_paths_fast
    mism = 0
    for seed in range(40):
        rng = random.Random(seed)
        av = rng.uniform(0.025, 0.05)
        planets = generate_planets(rng)
        for S in (50, 150, 250, 350, 450):
            r1 = random.Random(f"orbit_wars-comet-{seed}-{S}")
            r2 = random.Random(f"orbit_wars-comet-{seed}-{S}")
            pk = generate_comet_paths(planets, av, S, comet_planet_ids=set(), comet_speed=4.0, rng=r1)
            pf = _comet_paths_fast(planets, av, S, comet_planet_ids=set(), comet_speed=4.0, rng=r2)
            if pk != pf:
                mism += 1
    assert mism == 0, f"fast comet paths diverged from kaggle in {mism} cases"
    print("PASS: _comet_paths_fast is byte-identical to kaggle generate_comet_paths")


if __name__ == "__main__":
    test_orbital_and_fleet_match()
    test_fast_comet_paths_byte_identical()
    test_comet_fidelity()
