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


if __name__ == "__main__":
    test_timeline_parity()
    test_timeline_feature_encoding()
    print("OK")
