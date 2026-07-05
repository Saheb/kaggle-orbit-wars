"""Unit test for the PBRS staging shaping reward (torch_env._staging_potential + step()).

The idle fire head is an escapable PPO fixed point: A≈0 on the spare-fire action because
self-play never generates fire→win data. PBRS injects a DIRECTED gradient:
    r_t += coef · (gamma·Φ(s_{t+1}) − Φ(s_t)),   Φ = top-k Σ min(1, our_inflight/capture_floor)
over NEUTRAL targets only. Potential-based → telescopes → can't be farmed by spray.
project_undermass_by_choice.

Asserts: (1) Φ is neutral-only (enemy/own targets gated out) and capped at 1/target;
(2) Φ sums the top-k neutrals (breadth bounded); (3) the per-step reward equals
coef·(γΦ'−Φ) exactly (isolated via a coef=0 vs coef=α diff); (4) the telescoping identity
Σ γᵗ·shaping = coef·(γᵀΦ_T − Φ_0) holds → spray-safe; (5) terminal Φ(s')=0 and prev resets
to 0 across the episode boundary (no cross-game spike).

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_staging_shaping.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch_env import VecTorchEnv

ALPHA = 0.2


# --------------------------------------------------------------------------------------
# (1)(2) Φ = top-k Σ min(1, mass/floor) over NEUTRAL targets — direct unit test.
# --------------------------------------------------------------------------------------
def _phi_env(topk=2):
    env = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                      action_decode="target", staging_shaping_coef=ALPHA, staging_topk=topk)
    env.reset(seeds=[0])
    env.planet_alive[0, :] = False
    return env


def _fields(env, ratios_owner):
    """Build (mass, floor, eta, is_enemy) so player-0's ratio at planet i = ratios_owner[i][0],
    and set planet i's owner = ratios_owner[i][1], alive. floor=10 everywhere; mass = ratio*floor."""
    P = env.planets.shape[1]
    mass = torch.zeros(1, P, 2)
    floor = torch.full((1, P, 2), 10.0)
    for i, (ratio, owner) in enumerate(ratios_owner):
        env.planets[0, i, 1] = owner
        env.planet_alive[0, i] = True
        mass[0, i, 0] = ratio * 10.0
    return mass, floor, None, None


def test_neutral_only_and_cap():
    env = _phi_env(topk=2)
    # neutral ratio 0.8 (counts) | enemy ratio 10→capped (GATED) | own ratio 10 (GATED)
    fields = _fields(env, [(0.8, -1), (10.0, 1), (10.0, 0)])
    phi = env._staging_potential(fields)
    assert abs(phi[0, 0].item() - 0.8) < 1e-5, \
        f"only the neutral should count (enemy+own gated); got Φ={phi[0,0].item():.4f}"
    print("OK neutral-only + enemy/own gated + cap")


def test_topk_sum_and_bound():
    env = _phi_env(topk=2)
    # three neutrals: ratios 1.0 (over-floor→cap 1.0), 1.0, 0.8 → top-2 = 1.0+1.0 = 2.0
    fields = _fields(env, [(2.0, -1), (1.0, -1), (0.8, -1)])
    phi = env._staging_potential(fields)
    assert abs(phi[0, 0].item() - 2.0) < 1e-5, \
        f"top-2 of capped neutral ratios should be 2.0; got {phi[0,0].item():.4f}"
    # with k=1 the same board gives 1.0 (single best neutral, capped)
    env1 = _phi_env(topk=1)
    fields1 = _fields(env1, [(2.0, -1), (1.0, -1), (0.8, -1)])
    assert abs(env1._staging_potential(fields1)[0, 0].item() - 1.0) < 1e-5
    print("OK top-k sum + per-target cap + k bound")


# --------------------------------------------------------------------------------------
# (3)(4) PBRS reward formula + telescoping — isolated via coef=0 vs coef=α diff.
# --------------------------------------------------------------------------------------
def _build_pair():
    """Two identical envs (same seed, same hand-placed fleets) differing ONLY in
    staging_shaping_coef. reward(env1)-reward(env0) = the shaping term in isolation."""
    envs = []
    for coef in (0.0, ALPHA):
        e = VecTorchEnv(num_envs=1, num_players=2, device="cpu",
                        action_decode="target", staging_shaping_coef=coef, staging_topk=2)
        e.reset(seeds=[7])
        e.planet_alive[0, :] = False
        e.fleet_alive[0, :] = False
        # planet 0: ours (home), planet 1: NEUTRAL target at (300,300) garr 10
        e.planets[0, 0, 1], e.planets[0, 0, 2], e.planets[0, 0, 3] = 0, 100.0, 300.0
        e.planets[0, 0, 4], e.planets[0, 0, 5], e.planets[0, 0, 6] = 8.0, 50.0, 1.0
        e.planet_alive[0, 0] = True
        e.planets[0, 1, 1], e.planets[0, 1, 2], e.planets[0, 1, 3] = -1, 300.0, 300.0
        e.planets[0, 1, 4], e.planets[0, 1, 5], e.planets[0, 1, 6] = 8.0, 10.0, 0.0
        e.planet_alive[0, 1] = True
        # an enemy planet far away so neither side is eliminated (no premature done)
        e.planets[0, 2, 1], e.planets[0, 2, 2], e.planets[0, 2, 3] = 1, 700.0, 700.0
        e.planets[0, 2, 4], e.planets[0, 2, 5], e.planets[0, 2, 6] = 8.0, 50.0, 1.0
        e.planet_alive[0, 2] = True
        # our fleet inflight toward the neutral (far → stays in flight several steps)
        e.fleets[0, 0, 1], e.fleets[0, 0, 2], e.fleets[0, 0, 3] = 0, 130.0, 300.0
        e.fleets[0, 0, 4], e.fleets[0, 0, 6] = 0.0, 8.0
        e.fleet_alive[0, 0] = True
        e.step_count[0] = 0
        # hand-placed fleets bypass _apply_actions → resolve their targets from current state
        e.refresh_fleet_targets()
        # re-arm prev_staging_phi from this hand-set state (reset() saw the seed board, not ours)
        e.prev_staging_phi[:] = e._staging_potential(e._decisive_mass_fields())
        envs.append(e)
    return envs


def test_pbrs_formula_and_telescoping():
    env0, env1 = _build_pair()
    gamma = env1.staging_gamma
    phi0 = env1.prev_staging_phi.clone()          # Φ(s_0)
    G = torch.zeros(1, 2)                          # discounted Σ γᵗ·shaping
    last_phi = phi0
    T = 6
    for t in range(T):
        phi_before = env1.prev_staging_phi.clone()
        _, r0, d0 = env0.step()
        _, r1, d1 = env1.step()
        assert not bool(d1[0]), "test trajectory must stay non-terminal"
        phi_after = env1.prev_staging_phi.clone()  # Φ(s_{t+1})
        shaping = r1 - r0
        expected = ALPHA * (gamma * phi_after - phi_before)
        assert torch.allclose(shaping, expected, atol=1e-4), \
            f"step {t}: shaping {shaping.tolist()} != coef·(γΦ'−Φ) {expected.tolist()}"
        G += (gamma ** t) * shaping
        last_phi = phi_after
    # Telescoping: Σ γᵗ·coef·(γΦ_{t+1}−Φ_t) = coef·(γᵀΦ_T − Φ_0). Spray can't farm it.
    telescoped = ALPHA * (gamma ** T * last_phi - phi0)
    assert torch.allclose(G, telescoped, atol=1e-4), \
        f"telescoping broken: ΣγᵗΔ {G.tolist()} != coef·(γᵀΦ_T−Φ_0) {telescoped.tolist()}"
    # And Φ must have actually moved (else the test is vacuous).
    assert abs(last_phi[0, 0].item() - phi0[0, 0].item()) > 1e-3, "Φ should change as the fleet advances"
    print(f"OK PBRS formula per-step + telescoping (Φ0={phi0[0,0]:.3f} → Φ_T={last_phi[0,0]:.3f})")


# --------------------------------------------------------------------------------------
# (5) terminal Φ(s')=0 + prev resets to 0 across the boundary.
# --------------------------------------------------------------------------------------
def test_terminal_potential_zero_and_rearm():
    env0, env1 = _build_pair()
    phi_before = env1.prev_staging_phi.clone()
    assert phi_before[0, 0].item() > 1e-3, "need nonzero Φ pre-terminal for a meaningful check"
    # force time_up termination on the next step
    env0.step_count[0] = int(env0.episode_steps)
    env1.step_count[0] = int(env1.episode_steps)
    _, r0, d0 = env0.step()
    _, r1, d1 = env1.step()
    assert bool(d1[0]), "env should terminate (time_up)"
    shaping = r1 - r0                              # terminal: Φ(s')=0 → shaping = coef·(0 − Φ(s))
    expected = ALPHA * (0.0 - phi_before)
    assert torch.allclose(shaping, expected, atol=1e-4), \
        f"terminal shaping {shaping.tolist()} != coef·(−Φ) {expected.tolist()}"
    assert torch.allclose(env1.prev_staging_phi, torch.zeros_like(env1.prev_staging_phi)), \
        "prev_staging_phi must reset to 0 for the fresh post-reset board"
    print("OK terminal Φ=0 + boundary re-arm")


if __name__ == "__main__":
    test_neutral_only_and_cap()
    test_topk_sum_and_bound()
    test_pbrs_formula_and_telescoping()
    test_terminal_potential_zero_and_rearm()
    print("PASS: PBRS staging shaping — neutral-only/cap/top-k, formula, telescoping, boundary")
