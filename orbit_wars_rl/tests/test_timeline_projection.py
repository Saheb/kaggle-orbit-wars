"""Parity test for the projected-future timeline (writeup lesson 1).

project_timeline() must agree with the ground truth of actually stepping the
functional engine (torch_env_fn.physics_step) K times with no actions: same
per-planet owner at every step, near-identical garrison. Fleets are launched
first (via apply_actions_core) so there are real in-flight arrivals to resolve.

Known, accepted slack: resolve_target_eta's 4-iteration moving-target solve can
be one step off on a marginal intercept, so owner agreement is gated at >=99.5%
(measured 100.0% / 99.8% at step 24) and garrison MAE at <=0.1 (measured 0.01).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import timeline as tl
import torch_env_fn as fn
from torch_env import MAX_OWNED, VecTorchEnv

N, P, K = 16, 48, tl.TIMELINE_K


def _make_state():
    torch.manual_seed(0)
    env = VecTorchEnv(num_envs=N, num_players=2, device=torch.device("cpu"),
                      episode_steps=500, enable_comets=False,
                      expansion_coef=0.0, early_capture_coef=0.0,
                      staging_shaping_coef=0.0, win_margin_coeff=0.0,
                      first_strike_steps=0, reinforce_cost=0.0,
                      action_decode="target", ship_bin_mode="absolute",
                      ship_overflow_mode="clamp", allow_reinforce=False)
    env.reset(seeds=list(range(N)))
    sc = env._ship_counts_t.clone()
    st = fn.state_from_torch_env(env)

    # Launch fleets from both players at random targets → in-flight arrivals.
    def actions():
        fire = torch.ones(N, MAX_OWNED, dtype=torch.long)
        angle = torch.zeros(N, MAX_OWNED, dtype=torch.long)
        ships = torch.full((N, MAX_OWNED), 20, dtype=torch.long)
        tgt = torch.randint(0, P, (N, MAX_OWNED))
        return torch.stack([fire, angle, ships, tgt], dim=2)

    for pl in range(2):
        np_, nf, nfa, nn = fn.apply_actions_core(
            st.planets, st.planet_alive, st.fleets, st.fleet_alive,
            st.next_fleet_id, st.angular_velocity, actions(), pl, sc)
        st = st._replace(planets=np_, fleets=nf, fleet_alive=nfa, next_fleet_id=nn)
    assert st.fleet_alive.sum().item() > 0, "no in-flight fleets — test is vacuous"
    return st


def test_timeline_parity():
    st = _make_state()

    own_proj, garr_proj = tl.project_timeline(
        st.planets, st.planet_alive, st.fleets, st.fleet_alive,
        st.angular_velocity, num_players=2, K=K)

    # Ground truth: step the functional engine K times with no actions.
    gt_owner = torch.empty(N, P, K)
    gt_garr = torch.empty(N, P, K)
    s = st
    for k in range(K):
        s, _, _ = fn.physics_step(s, 2, 500)
        gt_owner[:, :, k] = s.planets[:, :, 1]
        gt_garr[:, :, k] = s.planets[:, :, 5]

    alive = st.planet_alive.unsqueeze(-1).expand(N, P, K)
    own_match = ((own_proj == gt_owner) & alive).float().sum() / alive.float().sum()
    garr_mae = ((garr_proj - gt_garr).abs() * alive.float()).sum() / alive.float().sum()
    print(f"owner agreement (all K, alive): {own_match.item() * 100:.2f}%")
    print(f"garrison MAE (alive): {garr_mae.item():.3f}")
    assert own_match.item() >= 0.995, f"owner agreement {own_match.item():.4f} < 0.995"
    assert garr_mae.item() <= 0.1, f"garrison MAE {garr_mae.item():.3f} > 0.1"


def test_timeline_feature_encoding():
    """timeline_features layout: [mine(K)|enemy(K)|neutral(K)|log-garrison(K)],
    one-hot sums to 1, and get_features appends it zeroed on dead slots."""
    st = _make_state()
    own_ts, garr_ts = tl.project_timeline(
        st.planets, st.planet_alive, st.fleets, st.fleet_alive,
        st.angular_velocity, num_players=2, K=K)
    feats = tl.timeline_features(own_ts, garr_ts, player=0)
    assert feats.shape == (N, P, tl.TIMELINE_DIM)
    grouped = feats.view(N, P, 4, K)
    onehot_sum = grouped[:, :, 0] + grouped[:, :, 1] + grouped[:, :, 2]
    assert torch.all(onehot_sum == 1.0), "owner one-hot must sum to 1 per (planet, step)"
    # mine channel at k must match owner_ts == 0 for player 0
    assert torch.equal(grouped[:, :, 0], (own_ts == 0.0).float())
    assert torch.equal(grouped[:, :, 3], torch.log1p(garr_ts.clamp(min=0.0)) / 8.0)


def test_candidate_timeline_captures_and_values_future_production():
    planets = torch.tensor([[[0.0, -1.0, 20.0, 20.0, 1.0, 10.0, 2.0]]])
    alive = torch.ones(1, 1, dtype=torch.bool)
    fleets = torch.zeros(1, 1, 7)
    fleet_alive = torch.zeros(1, 1, dtype=torch.bool)
    owner_ts, garr_ts, arrivals = tl.project_timeline(
        planets, alive, fleets, fleet_alive, torch.zeros(1),
        num_players=2, return_arrivals=True,
    )
    feats = tl.candidate_timeline_features(
        planets, alive, arrivals, owner_ts, garr_ts, player=0,
        candidate_ships=torch.tensor([[[15.0]]]),
        candidate_eta=torch.tensor([[[1.0]]]),
        source_indices=torch.tensor([[0]]),
        slot_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    assert feats.shape == (1, 1, 1, tl.CANDIDATE_TIMELINE_DIM)
    actual = feats[0, 0, 0]
    assert actual[0].item() == 1.0
    assert actual[1].item() == pytest.approx(5.0 / 200.0)
    assert actual[2].item() == 1.0
    assert actual[3].item() == 1.0
    assert actual[4].item() == pytest.approx(2.0 * (K - 1) / 100.0)
    terminal_margin_delta = (5.0 + 2.0 * (K - 1) + 10.0) / 200.0
    assert actual[5].item() == pytest.approx(terminal_margin_delta)


def test_candidate_timeline_prices_source_loss_after_launch():
    planets = torch.tensor([[[0.0, 0.0, 20.0, 20.0, 1.0, 20.0, 3.0],
                             [1.0, 1.0, 30.0, 20.0, 1.0, 10.0, 1.0]]])
    alive = torch.ones(1, 2, dtype=torch.bool)
    fleets = torch.tensor([[[0.0, 1.0, 10.0, 20.0, 0.0, 1.0, 20.0]]])
    fleet_alive = torch.ones(1, 1, dtype=torch.bool)
    owner_ts, garr_ts, arrivals = tl.project_timeline(
        planets, alive, fleets, fleet_alive, torch.zeros(1),
        num_players=2, return_arrivals=True,
    )
    feats = tl.candidate_timeline_features(
        planets, alive, arrivals, owner_ts, garr_ts, player=0,
        candidate_ships=torch.tensor([[[20.0, 20.0]]]),
        candidate_eta=torch.tensor([[[1.0, 1.0]]]),
        source_indices=torch.tensor([[0]]),
        slot_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    source = feats[0, 0, 1, 6:]
    assert source[0].item() < 1.0
    assert source[1].item() == 0.0
    assert source[2].item() < 0.0
    assert source[3].item() < 0.0


def _hold_fixture(*, target_enemy_step=None, source_enemy_step=None):
    planets = torch.tensor([[[0.0, 0.0, 20.0, 20.0, 1.0, 20.0, 0.0],
                             [1.0, -1.0, 21.0, 20.0, 1.0, 10.0, 0.0]]])
    alive = torch.ones(1, 2, dtype=torch.bool)
    arrivals = torch.zeros(1, K, 2, 2)
    if target_enemy_step is not None:
        arrivals[0, target_enemy_step, 1, 1] = 5.0
    if source_enemy_step is not None:
        arrivals[0, source_enemy_step, 0, 1] = 10.0
    owner_ts = torch.tensor([[[0.0] * K, [-1.0] * K]])
    garr_ts = torch.tensor([[[20.0] * K, [10.0] * K]])
    return planets, alive, arrivals, owner_ts, garr_ts


def _resolve_hold(**fixture_kwargs):
    planets, alive, arrivals, owner_ts, garr_ts = _hold_fixture(**fixture_kwargs)
    return tl.projected_hold_sizes(
        planets, alive, arrivals, owner_ts, garr_ts, player=0,
        max_ships=torch.tensor([[20.0]]),
        candidate_distance=torch.tensor([[[0.0, 1.0]]]),
        source_indices=torch.tensor([[0]]),
        slot_valid=torch.ones(1, 1, dtype=torch.bool),
    )


def test_projected_hold_finds_minimum_verified_capture():
    sizes, feasible = _resolve_hold()
    assert feasible[0, 0, 1]
    assert sizes[0, 0, 1].item() == 11.0


def test_projected_hold_includes_known_future_counterattack():
    sizes, feasible = _resolve_hold(target_enemy_step=2)
    assert feasible[0, 0, 1]
    assert sizes[0, 0, 1].item() == 15.0


def test_projected_hold_falls_back_to_all_in_when_source_would_newly_fall():
    sizes, feasible = _resolve_hold(source_enemy_step=1)
    assert not feasible[0, 0, 1]
    assert sizes[0, 0, 1].item() == 20.0


def test_global_economy_parity():
    """The projected economy series must match stepping the engine K times with no actions:
    production delta exactly (integer, ownership-driven), material delta to within the
    projection's own garrison slack. Material counts in-flight ships, so it is conserved
    across a launch — the property that makes an emptied source visible as a source cost."""
    st = _make_state()

    own_ts, garr_ts, arrivals = tl.project_timeline(
        st.planets, st.planet_alive, st.fleets, st.fleet_alive,
        st.angular_velocity, num_players=2, K=K, return_arrivals=True)
    econ = tl.global_economy_features(
        st.planets, st.planet_alive, st.fleets, st.fleet_alive,
        own_ts, garr_ts, arrivals, player=0, num_players=2)
    assert econ.shape == (N, tl.GLOBAL_ECON_DIM)

    def _deltas(planets, planet_alive, fleets, fleet_alive):
        owner, garr, prod = planets[:, :, 1], planets[:, :, 5], planets[:, :, 6]
        mine = (owner == 0) & planet_alive
        enemy = (owner > 0) & planet_alive
        p = (mine.float() * prod).sum(1) - (enemy.float() * prod).sum(1)
        m = (mine.float() * garr).sum(1) - (enemy.float() * garr).sum(1)
        fs = fleets[:, :, 6] * fleet_alive.float()
        f_mine = (fleets[:, :, 1] == 0).float() * fs
        m = m + f_mine.sum(1) - (fs - f_mine).sum(1)
        return p, m

    # Tolerances mirror test_timeline_parity: the projection's 4-iteration intercept solve can
    # be a step off on a marginal fleet, which misattributes that planet's whole production for
    # a step. Measured: production exact on 99.7% of (env, step) cells, MAE 0.01; material MAE
    # 0.41. So assert on the distribution, not a worst case that one slow fleet can set.
    s = st
    prod_err, mat_err = [], []
    for k in range(K):
        s, _, _ = fn.physics_step(s, 2, 500)
        gt_p, gt_m = _deltas(s.planets, s.planet_alive, s.fleets, s.fleet_alive)
        # Invert the signed-log encoding to compare in raw units.
        got_p = torch.expm1(econ[:, k].abs() * 4.0) * torch.sign(econ[:, k])
        got_m = torch.expm1(econ[:, K + k].abs() * 8.0) * torch.sign(econ[:, K + k])
        prod_err.append((got_p - gt_p).abs())
        mat_err.append((got_m - gt_m).abs())
    prod_err, mat_err = torch.stack(prod_err), torch.stack(mat_err)
    prod_exact = (prod_err < 0.5).float().mean().item()
    print(f"economy series — production MAE {prod_err.mean():.3f} (exact on "
          f"{prod_exact * 100:.1f}% of cells)  material MAE {mat_err.mean():.3f}")
    assert prod_exact >= 0.99, f"production delta exact on only {prod_exact:.4f} of cells"
    assert prod_err.mean().item() <= 0.1, f"production delta MAE {prod_err.mean():.3f}"
    assert mat_err.mean().item() <= 2.0, f"material delta MAE {mat_err.mean():.3f}"


if __name__ == "__main__":
    test_timeline_parity()
    test_timeline_feature_encoding()
    test_global_economy_parity()
    print("OK")
