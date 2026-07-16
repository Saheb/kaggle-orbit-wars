"""Vectorized Orbit Wars environment in PyTorch.

Runs complete games in parallel as batched tensor operations for self-play
training, including action decoding, orbital motion, combat, rewards, and resets.

State layout (batched over N envs):
    planets       (N, P, 7)  → [id, owner, x, y, radius, ships, production]
    init_planets  (N, P, 7)  → snapshot at t=0 for orbital calculation
    planet_alive  (N, P)     → bool
    fleets        (N, F, 7)  → [id, owner, x, y, angle, from_pid, ships]
    fleet_alive   (N, F)     → bool
    step          (N,)       → game tick
    angular_vel   (N,)       → per-env orbit rate

Constants follow kaggle env exactly.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import torch

# Constants — must match kaggle_environments.envs.orbit_wars.orbit_wars
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SHIP_SPEED = 6.0

# Lever A (decisive-mass reward) — capture-floor constants, mirroring producer_v2's
# ProducerLiteConfig + orbit_lite.capture_floor (opponents/candidate_producer_v2.py):
# floor = projected_defenders_at_arrival + beta*rho(eta)*reachable_enemy_mass + overhead.
# The beta*rho*enemy_mass margin = the ETA-aware REACTIVE reinforcement v2 anticipates
# (the not-yet-launched defense that out-masses us); deb uses the bare floor (margin off).
_DM_BETA = 2.2          # reinforce_size_beta
_DM_ETA_FREE = 3.0      # reinforce_eta_free  (eta below which the enemy can't react → rho=0)
_DM_ETA_SCALE = 12.0    # reinforce_eta_scale
_DM_HORIZON = 18.0      # config.horizon — reach cap for enemy_mass AND eta cap

# Threat-timing window (ch 20-21): enemy fleet mass landing within this many steps is "soon".
# Matches the age 0-5 post-capture under-defense window + a nearby reinforce ETA (~3-5 steps).
# Must match features._THREAT_ETA_WINDOW. ch14 (enemy_contest) is ETA-agnostic, so this pair is
# the only observation of WHEN pressure lands — the timing signal the policy needs to defend.
_THREAT_ETA_WINDOW = 6.0
_DM_OVERHEAD = 1.0      # capture_overhead
_VALUE_HORIZON = 40.0   # capped production-value lookahead for target value / keepability channels

# Reverse-edge reinforce cooldown — "edge never fired" sentinel; must match
# reinforce_cooldown.NEVER so the train mask and the eval/export canonical rule agree.
_REINF_CD_NEVER = -(1 << 30)

# Tensor sizes (worst-case bounds from kaggle env)
MAX_PLANETS = 48
MAX_FLEETS = 256
MAX_OWNED = 16
# fleet_tgt sentinel: fleet exists but its target was never resolved (created outside
# _apply_actions, or invalidated by a comet alive-transition) → lazy re-resolve in
# _fleet_targets(). Distinct from -1 = resolved, hits nothing.
_TGT_UNRESOLVED = -2

# Comets (extra-solar objects) — spawn mid-game, are collidable + capturable + moving.
# Match kaggle_environments.envs.orbit_wars: 4-comet symmetric groups spawn at these steps,
# carry ships + production 1, radius 1, move along precomputed elliptical paths at 4 units/turn.
# One group is active at a time (spawns 100 steps apart, paths <=40 steps), so 4 reserved
# planet slots suffice. They live in the LAST N_COMET_SLOTS slots so the policy observes them.
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1.0
COMET_SPEED = 4.0
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
N_COMET_SLOTS = 4
COMET_SLOT_START = MAX_PLANETS - N_COMET_SLOTS  # 44
# Comet observation features (overload the orbital channels of comet slots, gated by is_comet).
# Comets don't orbit circularly, so the generic orb_r/pred_x/pred_y channels are meaningless for
# them; for comet slots we instead expose the comet's path position COMET_FEAT_LOOKAHEAD steps
# ahead (pred_x/pred_y) and normalized steps-to-departure (orb_r channel). MUST stay in sync with
# the comet branch in features.extract_features / compute_pairwise_features (parity-tested in
# tests/feature_parity_comet_probe.py).
COMET_FEAT_LOOKAHEAD = 5    # path-position lookahead (matches the planet 5-turn orbit lookahead)
COMET_LIFE_NORM = 40.0      # normalize steps-to-departure (comet paths are <= ~40 steps)

# Discrete action bins (match action_mask.py / model.py)
from action_mask import (SHIP_COUNTS, FRACTION_BIN_VALUES, NUM_INTENTS,
                         MIN_BINARY_COMMIT_SHIPS)  # single source of truth (re-exported)
from timeline import (candidate_timeline_features, global_economy_features,
                      project_timeline, timeline_features)

NUM_ANGLE_BINS = 144
ANGLE_BIN_WIDTH = 2 * math.pi / NUM_ANGLE_BINS
NUM_SHIP_BINS = len(SHIP_COUNTS)


_COMET_T_ARR = None  # cached dense-sample parameter grid (t), built on first use


def _comet_paths_fast(initial_planets, angular_velocity, spawn_step,
                      comet_planet_ids=None, comet_speed=4.0, rng=None):
    """Vectorized drop-in for kaggle's generate_comet_paths with byte-identical output.
    Only the two 5000-iteration loops (dense ellipse sampling + arc-length resample) are
    vectorized with numpy; the RNG draw order (e, a, phi per attempt) and the validity check are
    preserved exactly, so `comet_ships` (drawn after) and accept/reject decisions are unchanged.
    np.cos/np.sin match math.cos/math.sin bit-for-bit here (verified); cross-checked against
    kaggle on 250 seeds in tests/test_comet_fidelity.py. Segments (arc/5000 ≈ 0.05) << comet_speed
    so the resample never skips a target → searchsorted matches kaggle's sequential append."""
    global _COMET_T_ARR
    import math as _m
    if rng is None:
        rng = random
    comet_planet_ids = set() if comet_planet_ids is None else set(comet_planet_ids)
    num = 5000
    if _COMET_T_ARR is None:
        _COMET_T_ARR = 0.3 * _m.pi + 1.4 * _m.pi * np.arange(num) / (num - 1)
    t = _COMET_T_ARR

    def _dist(p1, p2):
        return _m.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue
        b = a * _m.sqrt(1 - e ** 2)
        c_val = a * e
        phi = rng.uniform(_m.pi / 6, _m.pi / 3)
        cphi, sphi = _m.cos(phi), _m.sin(phi)
        ex = c_val + a * np.cos(t)
        ey = b * np.sin(t)
        x = CENTER + ex * cphi - ey * sphi
        y = CENTER + ex * sphi + ey * cphi
        # Resample at constant comet_speed arc-length intervals (searchsorted == kaggle's
        # sequential cum>=target since per-segment length << comet_speed).
        seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        cum = np.cumsum(seg)                                  # cum[j] = arc length to dense[j+1]
        K = int(cum[-1] // comet_speed)
        if K >= 1:
            sel = np.searchsorted(cum, comet_speed * np.arange(1, K + 1), side="left") + 1
            px = np.concatenate(([x[0]], x[sel]))
            py = np.concatenate(([y[0]], y[sel]))
        else:
            px, py = x[:1], y[:1]
        onboard = (px >= 0) & (px <= BOARD_SIZE) & (py >= 0) & (py <= BOARD_SIZE)
        if not onboard.any():
            continue
        bi = np.where(onboard)[0]
        visible = [(float(px[i]), float(py[i])) for i in range(bi[0], bi[-1] + 1)]
        if not (5 <= len(visible) <= 40):
            continue
        paths = [
            [[yv, xv] for xv, yv in visible],
            [[BOARD_SIZE - xv, yv] for xv, yv in visible],
            [[xv, BOARD_SIZE - yv] for xv, yv in visible],
            [[BOARD_SIZE - yv, BOARD_SIZE - xv] for xv, yv in visible],
        ]
        static_planets, orbiting_planets = [], []
        for planet in initial_planets:
            if planet[0] in comet_planet_ids:
                continue
            pr = _dist((planet[2], planet[3]), (CENTER, CENTER))
            (orbiting_planets if pr + planet[4] < ROTATION_RADIUS_LIMIT else static_planets).append(planet)
        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if _dist((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break
            sym_pts = [(cy, cx), (BOARD_SIZE - cx, cy), (cx, BOARD_SIZE - cy), (BOARD_SIZE - cy, BOARD_SIZE - cx)]
            for planet in static_planets:
                for sp in sym_pts:
                    if _dist(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = _m.sqrt(dx ** 2 + dy ** 2)
                init_angle = _m.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                pxp = CENTER + orb_r * _m.cos(cur_angle)
                pyp = CENTER + orb_r * _m.sin(cur_angle)
                for sp in sym_pts:
                    if _dist(sp, (pxp, pyp)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return paths
    return None


def _ship_speed(ships: torch.Tensor) -> torch.Tensor:
    """Speed formula from kaggle env, vectorized.

    speed = 1 + (max_speed - 1) * (log(ships) / log(1000)) ** 1.5
    clamped to [1, max_speed].
    """
    ships_clamped = torch.clamp(ships, min=1.0)
    base = (torch.log(ships_clamped) / math.log(1000.0)) ** 1.5
    speed = 1.0 + (MAX_SHIP_SPEED - 1.0) * base
    return torch.clamp(speed, max=MAX_SHIP_SPEED)


_INTENT_CEIL_EPS = 1e-3   # matches action_mask._INTENT_CEIL_EPS (integer-parity snap)


def _resolve_intent_sizes(cap_cost, reach_em, mass_soon, src_ships, is_own):
    """Torch twin of action_mask.resolve_intent_sizes_np — raw integer ships per intent,
    shape (..., 4), each clamped to [0, src_ships]. capture / capture-defend / maintain / all-in.
    Same formula + ceil-snap so GPU float32 and numpy produce identical integers (fuzz-parity test)."""
    S = src_ships.clamp(min=0.0)
    ceil_snap = lambda x: torch.ceil(x - _INTENT_CEIL_EPS)
    capture = ceil_snap(cap_cost).clamp(min=0.0).minimum(S)
    cap_def = ceil_snap(cap_cost + reach_em).clamp(min=0.0).minimum(S)
    maintain = torch.where(is_own, (ceil_snap(mass_soon) + 1.0).clamp(min=0.0).minimum(S),
                           torch.zeros_like(S))
    all_in = S
    return torch.stack([capture, cap_def, maintain, all_in], dim=-1)


def _resolve_binary_commit(pairwise_features, src_ships, gates="full"):
    """Torch twin of action_mask.resolve_binary_commit_np (see it for the gates rationale)."""
    S = src_ships.unsqueeze(-1).float()
    is_own = pairwise_features[..., 5] > 0.5
    is_enemy = pairwise_features[..., 6] > 0.5
    if gates == "minimal":
        feasible = (S >= MIN_BINARY_COMMIT_SHIPS).expand_as(is_own)
        ships = S.expand_as(is_own).float()
        return torch.where(feasible, ships, torch.zeros_like(ships)), feasible
    capture_required = (pairwise_features[..., 10] * 200.0
                        + is_enemy.float() * pairwise_features[..., 8] * 5.0 * 3.0 + 1.0)
    defend = torch.round(pairwise_features[..., 24] * 200.0)
    attack_ok = (S >= MIN_BINARY_COMMIT_SHIPS) & (S + 1e-3 >= capture_required)
    defend_ok = (defend >= MIN_BINARY_COMMIT_SHIPS) & (S + 1e-3 >= defend)
    feasible = torch.where(is_own, defend_ok, attack_ok)
    ships = torch.where(is_own, defend, S.expand_as(defend))
    return torch.where(feasible, ships, torch.zeros_like(ships)), feasible


class VecTorchEnv:
    """Vectorized Orbit Wars environment running N games in parallel."""

    def __init__(
        self,
        num_envs: int,
        num_players: int = 2,
        device: str | torch.device = "cpu",
        episode_steps: int = 500,
        ship_bin_mode: str = "absolute",
        # "full" = capture_required + maintain/defend_ok gates (legacy); "minimal" = COMMIT is
        # all-in at any target, gated only on having MIN_BINARY_COMMIT_SHIPS. Binary mode only.
        binary_commit_gates: str = "full",
        global_econ: bool = False,   # append the 48 projected economy-delta globals (15 -> 63)
        ship_overflow_mode: str = "clamp",   # matches eval (_ship_bin_to_count clamps); "drop"=legacy bug
        action_decode: str = "angle",
        win_margin_coeff: float = 0.0,
        expansion_coef: float = 0.0,
        early_capture_coef: float = 0.0,
        early_capture_steps: int = 100,
        first_strike_steps: int = 0,
        first_strike_mult: float = 2.0,
        staging_shaping_coef: float = 0.0,
        staging_topk: int = 2,
        staging_gamma: float = 0.995,
        allow_reinforce: bool = False,
        reinforce_garrison_floor: float = 0.0,
        reinforce_cost: float = 0.0,
        reinforce_gate_min_planets: int = 0,
        reinforce_forward_only: bool = False,
        reverse_edge_cooldown: int = 0,
        sufficient_commit_factor: float = 0.0,
        enable_comets: bool = True,
        fleet_target_refresh_every: int = 4,
    ):
        self.num_envs = num_envs
        # Training may disable comets
        # entirely (no spawns, no schedule compute). The real kaggle game HAS comets —
        # eval/export always keep them — so this trades a training-distribution gap
        # for throughput. Default ON.
        self.enable_comets = bool(enable_comets)
        # Launch-cache staleness bound: every K ticks step() re-resolves ALL cached fleet
        # targets from current state (see _fleet_targets). Measured accuracy vs the true
        # swept collision (64 envs x 300 random steps, comets on): K=1 95.6%, K=2 95.0%,
        # K=4 93.7%, K=8 91.9%, K=0 (launch-only, comet staleness never fixed) 79.2% —
        # long-range lead error vs rotating planets drifts. K=4 = ~1/4 the resolver cost
        # for a 2pp accuracy dip. 1 ≈ fresh every tick, 0 = never refresh.
        self.fleet_target_refresh_every = int(fleet_target_refresh_every)
        # Canonical feature config: game-phase globals (dim 15) always on;
        # pressure channels always routed through the lead-aware swept-collision resolver
        # (_fleet_target_idx, one fleet → one target); threat ETA to planet CENTER; friendly
        # roi-deflation always on; enemy-deflate/zero-roi removed. Must match features.py
        # (parity test test_torch_env_features).
        # Reinforcement: when True, own planets (except the launch source) are LEGAL
        # targets — ships arriving at a friendly planet add to its garrison (physics
        # already implemented in step()). Default False = attack-only.
        self.allow_reinforce = bool(allow_reinforce)
        # Reinforcement discipline prevents costless reinforcement floods.
        #   #1 GARRISON FLOOR: a reinforce launch may not drain its source below this
        #      many ships. Pure training-time mask (veto), NOT a penalty → no Nash risk.
        #      Kills the "drain a planet, then lose it" regression. The real Kaggle env
        #      has no floor, so inference is unconstrained — the policy internalises it.
        #   #2 TRANSIT COST: subtract reinforce_cost × ships_reinforced from the
        #      launching player's per-step reward. Scales with WASTE (a flood of
        #      thousands of ships is expensive; one useful staging move is cheap), so it
        #      prunes the wasteful tail rather than zeroing reinforcement — PROVIDED
        #      credit connects a useful stage to its payoff. reinforce_cost is the
        #      calibration knob; reinforce_rate is the dial (target ~0.4-0.6, not 0/0.8).
        # Both only act on launches whose target is OUR OWN planet — attacks
        # (enemy/neutral) are untouched, so neither lever can distort the attack Nash.
        self.reinforce_garrison_floor = float(reinforce_garrison_floor)
        self.reinforce_cost = float(reinforce_cost)
        #   #3 EMPIRE-SIZE GATE: own planets become legal reinforce targets only once the
        #      player owns >= this many planets. Below it, attack-only (must expand first).
        #      Reinforcement should ramp with empire size. A pure action mask that makes
        #      the early flood impossible by construction. 0 = off (no gate). Training-only,
        #      like the garrison floor — the policy internalises it.
        self.reinforce_gate_min_planets = int(reinforce_gate_min_planets)
        #   #4 FORWARD-STAGING GATE: an own (reinforce) target is legal only if it sits
        #      closer to the nearest enemy planet than the launch source — reinforcement
        #      flows rear→front (staging), never into a safe rear hoard. A rear hoard is the costless
        #      safe-fire outlet that floods symmetric self-play. Pure mask (no Nash risk),
        #      training-only, internalised at inference. 0/False = off. Enemy/neutral
        #      targets are never constrained.
        self.reinforce_forward_only = bool(reinforce_forward_only)
        #   REVERSE-EDGE COOLDOWN: after an own-target reinforce A→B, the reverse B→A reinforce
        #   is illegal for K steps (block the A→B→A ping-pong; rank1 recip<=3st <0.01 vs our
        #   0.06-0.10). Canonical rule in reinforce_cooldown.py; ownership-change & episode resets
        #   clear stale edges. Pure mask, training-internalised. 0 = off. project_reinforce_pingpong.
        self.reverse_edge_cooldown = int(reverse_edge_cooldown)
        self.reinf_cd = None   # (N, num_players, P, P) long: last step each reinforce edge fired
        #   SUFFICIENT-COMMIT MASK: veto an ATTACK launch (enemy/neutral target) whose
        #   ship_count <= target's current defense × this factor → fragments fired under a
        #   target's garrison become impossible by construction, forcing concentration
        #   (attack only a target you can actually take, else accumulate first). Fixes the
        #   opening under-commitment that caps conversion. 1.0 = strict (need strictly
        #   more than current defense); 0.6 = relaxed
        #   fallback if it over-constrains; 0.0 = off. Exact for neutrals (they don't regrow),
        #   approximate for enemy planets (reinforce in transit). Pure training-time mask
        #   (no reward tax → no fire=0 Nash); the policy internalises it, parity at eval/export.
        self.sufficient_commit_factor = float(sufficient_commit_factor)
        # reinforce_rate metric accumulators (N, num_players), allocated/zeroed per
        # rollout via reset_reinforce_stats(). None = not collecting (no overhead).
        self._reinforce_launch_count = None
        self._fire_launch_count = None
        # reinf-by-step: reinforce & all launches binned by episode step-window
        # [<50, 50-100, >100] — surfaces the back-loaded reinforce timing (session 06-12).
        self._reinf_step = None        # (N, num_players, 3)
        self._fire_step = None         # (N, num_players, 3)
        # target-owner share diagnostic: launches whose target is a NEUTRAL planet
        # (own = _reinforce_launch_count; enemy = fire − own − neutral). Phase-2
        # target-head health (is the "where" head selective or uniform?).
        self._neutral_launch_count = None
        self._overask_step = None      # (N, num_players, 3) attacks+reinf where bin>garrison
        self._attempt_step = None      # (N, num_players, 3) all intended launches (= requested moves)
        self._emitted_step = None      # (N, num_players, 3) launches that created a fleet (emitted)
        self._slotstarve_step = None   # (N, num_players, 3) can_fire dropped: fleet storage full
        self._last_wins = None         # (N, num_players) bool: raw winner mask from the last _check_done
        self._obs_trunc = None         # (num_players,) get_features calls with live fleets > obs cap
        self._obs_calls = None         # (num_players,) get_features calls total (denom)
        # richer truncation severity (how much mass is hidden, not just whether any is):
        self._obs_trunc_fleets = None  # (num_players,) live fleets in slots >= obs cap (omitted)
        self._obs_total_fleets = None  # (num_players,) all live fleets (denom)
        self._obs_trunc_ships = None   # (num_players,) ship mass in omitted fleets
        self._obs_total_ships = None   # (num_players,) ship mass in all live fleets (denom)
        # ENEMY-only severity (the high-signal cut for the obs256 decision: is the hidden mass
        # enemy fleets we're blind to, not our own?). Enemy = fleet owner != the obs player.
        self._obs_trunc_enemy_ships = None  # (num_players,) enemy ship mass in omitted fleets
        self._obs_total_enemy_ships = None  # (num_players,) enemy ship mass in all live fleets
        self.num_players = num_players
        self.episode_steps = episode_steps
        self.device = torch.device(device)
        # Ship-bin decode lookup tables, built once (were re-created + copied to device
        # on every _apply_actions call).
        self._ship_counts_t = torch.tensor(SHIP_COUNTS, dtype=torch.float32, device=self.device)
        self._frac_bins_t = torch.tensor(FRACTION_BIN_VALUES, dtype=torch.float32, device=self.device)
        # See ModelConfig.ship_bin_mode. "absolute" uses SHIP_COUNTS lookup;
        # "fraction" uses round(FRAC_VALUES[bin] * src_ships).
        self.ship_bin_mode = ship_bin_mode
        if binary_commit_gates not in ("full", "minimal"):
            raise ValueError(f"unknown binary_commit_gates: {binary_commit_gates}")
        self.binary_commit_gates = binary_commit_gates
        self.global_econ = bool(global_econ)
        # Intent sizing (#4): per-player raw resolved-size table {player: (N,MO,P,4)}, stashed by
        # get_features and read at decode (_apply_actions) to turn a chosen intent → exact ships.
        self._intent_sizes = {}
        self._binary_commit_sizes = {}
        # What to do when a launch's ship_count exceeds the source garrison:
        #   "drop"  — legacy: void the whole launch (valid_ships = src_ships >= ship_count)
        #   "clamp" — send min(ship_count, src_ships), i.e. the whole garrison — MATCHES EVAL
        #             (action_mask `_ship_bin_to_count` = min(count, max_ships)). Fixes the
        #             train/eval gap where ~35% of attacks were silently dropped in training.
        if ship_overflow_mode not in {"drop", "clamp"}:
            raise ValueError(f"unknown ship_overflow_mode={ship_overflow_mode!r}")
        self.ship_overflow_mode = ship_overflow_mode
        if action_decode not in {"angle", "target"}:
            raise ValueError(f"unknown action_decode={action_decode!r}")
        self.action_decode = action_decode
        # Terminal bonus for winners: +win_margin_coeff * (my_score / total_score).
        # 0.0 = pure ±1 reward (default, backward-compatible).
        self.win_margin_coeff = float(win_margin_coeff)
        # Expansion shaping: potential-based reward on OWNED PRODUCTION (sum of
        # planet production rates owned). Unlike material (ships), production only
        # changes when planets change hands, so a passive hoarder gets 0 from it.
        # Rewards winning the planet/economy
        # race that decides snowball games. 0.0 = off (default).
        self.expansion_coef = float(expansion_coef)
        # Early capture shaping: per-step bonus for each net new planet owned above
        # starting count (1), decayed linearly from 1.0→0.0 over early_capture_steps.
        # Gives gradient signal for the opening probe that the terminal reward cannot see.
        # Coeff math: sum(coeff*(1-t/100), t=4..100) ≈ 97*0.48*coeff per planet captured
        # at step 3. Keep cumulative bonus ≤ 10-15% of terminal win → coeff 0.002-0.003.
        self.early_capture_coef = float(early_capture_coef)
        self.first_strike_steps = int(first_strike_steps)
        self.first_strike_mult = float(first_strike_mult)
        self.early_capture_steps = int(early_capture_steps)
        # PBRS staging shaping (project_undermass_by_choice): potential-based reward that injects a
        # DIRECTED gradient for the idle fire head to STAGE inflight toward NEUTRAL captures.
        # Φ = top-k Σ min(1, our_inflight/capture_floor) over neutral targets; r += coef·(γΦ' − Φ).
        # Telescoping → spray-safe (can't farm by cycling). Neutral-ONLY (enemy = decmass, which failed).
        self.staging_shaping_coef = float(staging_shaping_coef)
        self.staging_topk = int(staging_topk)
        self.staging_gamma = float(staging_gamma)
        self.prev_staging_phi = None         # (N, num_players) — allocated in reset()
        self._staging_phi_acc = 0.0          # rollout mean Φ accumulator — reset_reinforce_stats()
        self._staging_phi_n = 0

        # State tensors — allocated in reset()
        self.planets: torch.Tensor = None       # (N, P, 7)
        self.init_planets: torch.Tensor = None  # (N, P, 7) — pristine snapshot
        self.planet_alive: torch.Tensor = None  # (N, P) bool
        self.fleets: torch.Tensor = None        # (N, F, 7)
        self.fleet_alive: torch.Tensor = None   # (N, F) bool
        self.step_count: torch.Tensor = None    # (N,) long
        self.angular_velocity: torch.Tensor = None  # (N,) float
        self.next_fleet_id: torch.Tensor = None     # (N,) long
        self._has_comets = False                    # set in _precompute_comets
        self._comet_xy = None                       # (N, T+1, 4, 2) precomputed comet positions
        self.done: torch.Tensor = None              # (N,) bool
        self.rewards: torch.Tensor = None           # (N, num_players) float
        self.prev_production: torch.Tensor = None   # (N, num_players) float — owned production for expansion shaping
        self.prev_owned: torch.Tensor = None        # (N, num_players) float — owned planet count for delta-capture shaping
        # Seeds (per-env) so we can deterministically auto-reset
        self.seeds: list[int] = []

        # Pre-computed once per reset for orbital motion
        self._planet_initial_angle: torch.Tensor = None  # (N, P) float
        self._planet_orbital_r: torch.Tensor = None      # (N, P) float
        self._planet_is_orbiting: torch.Tensor = None    # (N, P) bool


    # ---------------------------------------------------------------------
    # Reset — generates N games using the kaggle env's seed-based generator,
    # then stacks into batched tensors. This is the only non-vectorized op,
    # and it only runs once per episode.
    # ---------------------------------------------------------------------

    def reset(self, seeds: Optional[list[int]] = None) -> dict:
        """Reset N environments. Returns initial state dict."""
        from kaggle_environments.envs.orbit_wars.orbit_wars import generate_planets

        if seeds is None:
            seeds = [random.randint(0, 2**31) for _ in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        planets_list = []
        planet_alive_list = []
        angular_velocities = []

        for seed_idx, seed in enumerate(seeds):
            init_rng = random.Random(seed)
            ang_vel = init_rng.uniform(0.025, 0.05)
            angular_velocities.append(ang_vel)

            raw_planets = generate_planets(init_rng)  # list of [id, owner, x, y, r, ships, prod]
            n = len(raw_planets)
            pad = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
            for i, p in enumerate(raw_planets):
                pad[i] = p
            planets_list.append(pad)

            alive = np.zeros(MAX_PLANETS, dtype=bool)
            alive[:n] = True

            # Home planet assignment (matches fast_env reset)
            num_groups = n // 4
            if num_groups > 0:
                home_group = init_rng.randint(0, num_groups - 1)
                base = home_group * 4
                if self.num_players == 2:
                    pad[base, 1] = 0;     pad[base, 5] = 10
                    pad[base + 3, 1] = 1; pad[base + 3, 5] = 10
                elif self.num_players == 4:
                    for j in range(4):
                        pad[base + j, 1] = j; pad[base + j, 5] = 10
                else:
                    pad[base, 1] = 0; pad[base, 5] = 10

            # Mark unused slots as neutral (-1)
            pad[n:, 1] = -1
            planet_alive_list.append(alive)

        planets_np = np.stack(planets_list, axis=0)
        alive_np = np.stack(planet_alive_list, axis=0)

        self.planets = torch.from_numpy(planets_np).to(self.device)
        self.init_planets = self.planets.clone()
        self.planet_alive = torch.from_numpy(alive_np).to(self.device)
        self.fleets = torch.zeros(self.num_envs, MAX_FLEETS, 7, dtype=torch.float32, device=self.device)
        self.fleet_alive = torch.zeros(self.num_envs, MAX_FLEETS, dtype=torch.bool, device=self.device)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.angular_velocity = torch.tensor(angular_velocities, dtype=torch.float32, device=self.device)
        self.next_fleet_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rewards = torch.zeros(self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        self.seeds = list(seeds)

        self._precompute_orbital_params()
        self._init_comets()
        # Launch-time fleet-target cache (see _fleet_targets). No fleets yet → all "none";
        # checked=True keeps the training hot path a plain read (only refresh_fleet_targets
        # arms the lazy sweep).
        self.fleet_tgt = torch.full((self.num_envs, MAX_FLEETS), -1,
                                    dtype=torch.long, device=self.device)
        self._fleet_tgt_checked = True
        self._tick_counter = 0   # global tick for the periodic fleet-target refresh
        self.prev_production = self._compute_production()
        owner_p = self.planets[:, :, 1].long()
        self.prev_owned = torch.zeros(self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        for pl in range(self.num_players):
            self.prev_owned[:, pl] = ((owner_p == pl) & self.planet_alive).float().sum(dim=1)
        P = self.planets.shape[1]
        # PBRS staging: Φ of the previous state. Fresh boards have no inflight → Φ(s_0)=0.
        self.prev_staging_phi = torch.zeros(
            self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        # Reverse-edge reinforce cooldown: last step each (player, src, tgt) reinforce edge fired.
        # _DM-style NEVER sentinel so untouched edges never trip the (step - last) <= K test.
        if self.reverse_edge_cooldown > 0:
            self.reinf_cd = torch.full(
                (self.num_envs, self.num_players, P, P), _REINF_CD_NEVER,
                dtype=torch.long, device=self.device)
        return self._state_dict()

    def _fleet_target_idx(self) -> torch.Tensor:
        """Lead-aware swept-collision target planet index per fleet, (N, F), or -1 if it hits nothing.

        A launched fleet flies straight and captures whatever it physically collides with, so the
        true target depends on distance, planet radius AND the target's orbital motion over the
        flight — not just the launch-instant heading. For each candidate planet we project it to its
        orbit position at the fleet's ETA (the engine advances phase by angular_velocity*step),
        converge the ETA in 4 iterations, then keep the min-ETA planet the heading reaches within
        radius. This mirrors the agent's own intercept aimer (`_target_intercept_angle`).

        Scalar mirror: eval._dm_fleet_target / _lead_collision_target, validated at 98.4% vs the true
        swept-collision on replay (the old along/perp-r+2 nearest-distance heuristic was ~85% — it
        ignored orbital lead and over-loosely matched). Player-independent (geometry only). Feeds the
        capture-floor machinery via _decisive_mass_fields (staging potential, sufficient-commit
        veto); dead fleets are masked by the caller (valid_f)."""
        return self._resolve_targets_at(
            self.fleets[:, :, 2], self.fleets[:, :, 3],
            self.fleets[:, :, 4], self.fleets[:, :, 6])

    def _resolve_targets_at(self, fleet_x, fleet_y, fleet_angle, fleet_ships) -> torch.Tensor:
        """Resolver core for arbitrary fleet states (each arg (N, K)) → (N, K) target idx or -1.
        Used per-launch on the (N, MAX_OWNED) grid and by _fleet_target_idx for
        full-fleet re-resolution (comet transitions, externally-poked fleets, direct test calls)."""
        fx = fleet_x.unsqueeze(2)                                      # (N, K, 1)
        fy = fleet_y.unsqueeze(2)
        fcos = torch.cos(fleet_angle).unsqueeze(2)                     # (N, K, 1)
        fsin = torch.sin(fleet_angle).unsqueeze(2)
        speed = _ship_speed(fleet_ships).clamp(min=1e-6).unsqueeze(2)  # (N, K, 1)
        px = self.planets[:, :, 2].unsqueeze(1)                        # (N, 1, P)
        py = self.planets[:, :, 3].unsqueeze(1)
        pr = self.planets[:, :, 4].unsqueeze(1)
        angvel = self.angular_velocity.view(-1, 1, 1)                  # (N, 1, 1)
        dx0 = px - CENTER
        dy0 = py - CENTER
        orbit_r = torch.sqrt(dx0 * dx0 + dy0 * dy0)                    # rotation-invariant radius
        static = (orbit_r + pr) >= ROTATION_RADIUS_LIMIT
        phase0 = torch.atan2(dy0, dx0)
        eta = ((torch.sqrt((px - fx) ** 2 + (py - fy) ** 2) - pr) / speed).clamp(min=0.0)   # (N, F, P)
        for _ in range(4):                                             # converge ETA vs the moving target
            a = phase0 + angvel * eta
            lx = torch.where(static, px, CENTER + orbit_r * torch.cos(a))
            ly = torch.where(static, py, CENTER + orbit_r * torch.sin(a))
            eta = ((torch.sqrt((lx - fx) ** 2 + (ly - fy) ** 2) - pr) / speed).clamp(min=0.0)
        a = phase0 + angvel * eta
        lx = torch.where(static, px, CENTER + orbit_r * torch.cos(a))
        ly = torch.where(static, py, CENTER + orbit_r * torch.sin(a))
        vx = lx - fx
        vy = ly - fy
        along = vx * fcos + vy * fsin
        perp = torch.abs(vx * fsin - vy * fcos)
        candidate = (along > 0) & (perp < pr + 0.5) & self.planet_alive.unsqueeze(1)
        has_candidate = candidate.any(dim=2)
        tgt_idx = eta.masked_fill(~candidate, 1e6).argmin(dim=2)        # (N, F) — min-ETA hit
        return torch.where(has_candidate, tgt_idx, torch.full_like(tgt_idx, -1))

    def _fleet_targets(self) -> torch.Tensor:
        """(N, F) resolved target planet per fleet slot; -1 = hits nothing.

        A fleet's ray and every planet's
        orbit are fixed at launch, so the target is resolved ONCE in _apply_actions on the
        (N, MAX_OWNED) launch grid instead of the (N, 256, P) full cube every tick (was >50% of
        env wall time). Mid-flight the cached answer can drift from a fresh per-tick resolve
        (features.py on the kaggle side re-resolves from current obs each step; the acceptance
        test perp < r+0.5 is evaluated from the current fleet position).

        Staleness bound: step() re-resolves ALL targets every fleet_target_refresh_every ticks
        (unconditional, sync-free), which also covers comet spawn/expiry changing who a ray
        hits. The lazy sweep below exists ONLY for fleets poked into existence outside
        _apply_actions — tests must call refresh_fleet_targets() after direct pokes; the
        training loop never sets the flag, so the hot path is a plain attribute read."""
        if not self._fleet_tgt_checked:
            self._fleet_tgt_checked = True
            unresolved = self.fleet_alive & (self.fleet_tgt == _TGT_UNRESOLVED)
            if bool(unresolved.any()):
                fresh = self._fleet_target_idx()
                self.fleet_tgt = torch.where(unresolved, fresh, self.fleet_tgt)
        return self.fleet_tgt

    def refresh_fleet_targets(self):
        """Force a from-current-state re-resolution of every fleet's cached target on the
        next consumer. For parity tests (features.py resolves from current obs — see
        _fleet_targets for the accepted drift) and after poking env.fleets directly.
        Training never needs this."""
        self.fleet_tgt.fill_(_TGT_UNRESOLVED)
        self._fleet_tgt_checked = False

    def _decisive_mass_fields(self):
        """Per-(N,P,num_players) inflight mass, capture floor, max-ETA and is_enemy mask — the
        EXACT quantities producer_v2's capture floor uses. Consumed by the PBRS staging
        potential (_staging_potential).

        floor_t = garrison + prod*eta + enemy_inbound_now      (projected defenders at arrival)
                + beta*rho(eta)*reachable_enemy_mass           (v2's reactive-reinforcement margin)
                + overhead
        mass_t = our alive inflight fleet ships converging on t. eta = MAX ETA of that mass
        (capped at the horizon — the floor must hold at the LAST arrival, when all counted mass is
        present, and rho(eta) then gives the enemy the full reaction window). enemy_mass =
        cheap_enemy_pressure (reachable enemy PLANET mass); enemy_inbound_now = enemy FLEET ships
        already racing to defend t."""
        N, P = self.planets.shape[0], self.planets.shape[1]
        owner = self.planets[:, :, 1].long()                            # (N, P)
        garr = self.planets[:, :, 5]
        prod = self.planets[:, :, 6]
        px, py = self.planets[:, :, 2], self.planets[:, :, 3]
        alive = self.planet_alive
        # Reachable-enemy-mass scaffolding (player-independent): pairwise planet distance +
        # per-source reach decay (cheap_enemy_pressure: closer enemy planets reinforce more).
        dx = px.unsqueeze(2) - px.unsqueeze(1)                          # (N, P_src, P_tgt)
        dy = py.unsqueeze(2) - py.unsqueeze(1)
        pdist = torch.sqrt(dx * dx + dy * dy)
        src_reach = (_ship_speed(garr) * _DM_HORIZON).clamp(min=1e-6)   # (N, P_src)
        decay = (1.0 - pdist / src_reach.unsqueeze(2)).clamp(min=0.0)   # (N, P_src, P_tgt)
        not_self = ~torch.eye(P, dtype=torch.bool, device=self.device).unsqueeze(0)
        # Our converging fleets: target, ships, and ETA to that target.
        tgt_idx = self._fleet_targets()                                # (N, F)
        f_owner = self.fleets[:, :, 1].long()
        f_ships_raw = self.fleets[:, :, 6]
        fx, fy = self.fleets[:, :, 2], self.fleets[:, :, 3]
        tgt_safe = tgt_idx.clamp(min=0)
        tx = torch.gather(px, 1, tgt_safe)                             # (N, F) target planet x
        ty = torch.gather(py, 1, tgt_safe)
        f_eta = (torch.sqrt((fx - tx) ** 2 + (fy - ty) ** 2)
                 / _ship_speed(f_ships_raw).clamp(min=1e-6)).clamp(max=_DM_HORIZON)
        f_ships = f_ships_raw * self.fleet_alive.float()
        valid_f = (tgt_idx >= 0) & self.fleet_alive
        mass = torch.zeros(N, P, self.num_players, dtype=garr.dtype, device=self.device)
        floor = torch.zeros_like(mass)
        eta_out = torch.zeros_like(mass)
        is_enemy = torch.zeros(N, P, self.num_players, dtype=torch.bool, device=self.device)
        for pl in range(self.num_players):
            mine = (valid_f & (f_owner == pl)).float()                 # (N, F)
            w = f_ships * mine
            m = torch.zeros(N, P, dtype=garr.dtype, device=self.device)
            m.scatter_add_(1, tgt_safe, w)                             # our inflight per target
            # MAX ETA over our contributing fleets (non-mine get -1 so they never win the max;
            # the 0-init is harmless — targets with no mine fleet have mass=0).
            eta_src = torch.where(mine.bool(), f_eta, torch.full_like(f_eta, -1.0))
            eta = torch.zeros(N, P, dtype=garr.dtype, device=self.device)
            eta.scatter_reduce_(1, tgt_safe, eta_src, reduce='amax', include_self=True)
            # enemy fleets already inbound to the target (current reinforcements en route)
            enemy_f = (valid_f & (f_owner != pl) & (f_owner >= 0)).float()
            inbound = torch.zeros(N, P, dtype=garr.dtype, device=self.device)
            inbound.scatter_add_(1, tgt_safe, f_ships * enemy_f)
            # reachable enemy PLANET mass for each target (producer_v2 cheap_enemy_pressure)
            enemy_src = (alive & (owner != pl) & (owner >= 0)).unsqueeze(2)   # (N, P_src, 1)
            valid_sp = enemy_src & alive.unsqueeze(1) & not_self
            enemy_mass = torch.where(valid_sp, garr.unsqueeze(2) * decay,
                                     torch.zeros_like(decay)).sum(dim=1)      # (N, P_tgt)
            rho = ((eta - _DM_ETA_FREE) / _DM_ETA_SCALE).clamp(0.0, 1.0)
            mass[:, :, pl] = m
            # NEUTRALS DON'T REGROW (engine applies production only to owner != -1) -> no prod
            # accrual during flight. The staging potential is neutral-only, so the phantom term
            # under-credited staging toward cheap neutrals. Enemy targets keep prod*eta.
            prod_floor = torch.where(owner == -1, torch.zeros_like(prod), prod)
            floor[:, :, pl] = (garr + prod_floor * eta + inbound
                               + _DM_BETA * rho * enemy_mass + _DM_OVERHEAD)
            eta_out[:, :, pl] = eta
            is_enemy[:, :, pl] = alive & (owner != pl) & (owner >= 0)
        return mass, floor, eta_out, is_enemy

    def _staging_potential(self, fields):
        """PBRS potential Φ(s) for the staging shaping reward (project_undermass_by_choice):
        Φ = top-k Σ min(1, our_inflight_mass / capture_floor) over NEUTRAL targets (owner<0, alive),
        per (N, num_players). Rewards building inflight toward neutrals we can take-and-hold (the
        floor folds in enemy reactive defense, so contested neutrals count). NEUTRAL-ONLY: enemy
        targets are excluded (staging into reactive enemy defense = the out-mass contest = decmass,
        which failed). Reuses _decisive_mass_fields' mass/floor — no new geometry."""
        mass, floor, _eta, _is_enemy = fields
        owner = self.planets[:, :, 1].long()                          # (N, P)
        neutral = ((owner < 0) & self.planet_alive).unsqueeze(-1)     # (N, P, 1)
        ratio = (mass / floor.clamp(min=1e-6)).clamp(min=0.0, max=1.0)   # (N, P, players), capped
        ratio = ratio * neutral.float()                              # neutral targets only
        k = min(self.staging_topk, ratio.shape[1])
        phi = ratio.topk(k, dim=1).values.sum(dim=1)                 # (N, players): top-k per player
        return phi

    def _compute_production(self) -> torch.Tensor:
        """Total production rate of planets owned by each player. (N, num_players)"""
        owner_p = self.planets[:, :, 1].long()
        prod_p = self.planets[:, :, 6] * self.planet_alive.float()
        production = torch.zeros(self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        for pl in range(self.num_players):
            production[:, pl] = ((owner_p == pl).float() * prod_p).sum(dim=1)
        return production

    def _precompute_orbital_params(self):
        """Cache initial_angle, orbital_r, is_orbiting per planet (shape (N, P))."""
        ix = self.init_planets[:, :, 2]
        iy = self.init_planets[:, :, 3]
        r  = self.init_planets[:, :, 4]
        dx = ix - CENTER
        dy = iy - CENTER
        self._planet_orbital_r = torch.sqrt(dx * dx + dy * dy)
        self._planet_initial_angle = torch.atan2(dy, dx)
        self._planet_is_orbiting = (
            (self._planet_orbital_r + r) < ROTATION_RADIUS_LIMIT
        ) & self.planet_alive

    def _init_comets(self, env_indices=None):
        """Allocate / clear the per-step comet schedule buffers. Comets are computed LAZILY
        (`_lazy_comets`) the first time an env reaches a spawn step, because generate_comet_paths
        is expensive (5000-pt sampling) and most games end before the later spawns — so an upfront
        precompute of all 5 spawns × N envs would dominate reset/auto-reset cost. `_has_comets`
        stays True (the integration is always wired); an un-computed schedule is simply all-dead."""
        if not self.enable_comets:
            self._has_comets = False
            return
        N, T1 = self.num_envs, self.episode_steps + 1
        if getattr(self, "_comet_xy", None) is None:
            self._comet_xy = torch.zeros(N, T1, N_COMET_SLOTS, 2, device=self.device)
            self._comet_alive = torch.zeros(N, T1, N_COMET_SLOTS, dtype=torch.bool, device=self.device)
            self._comet_check = torch.zeros(N, T1, N_COMET_SLOTS, dtype=torch.bool, device=self.device)
            self._comet_ships0 = torch.zeros(N, T1, N_COMET_SLOTS, device=self.device)
            self._comet_spawn_done = torch.zeros(N, len(COMET_SPAWN_STEPS), dtype=torch.bool, device=self.device)
            self._comet_ids = torch.zeros(N, N_COMET_SLOTS, device=self.device)
        self._has_comets = True
        idx = torch.arange(N, device=self.device) if env_indices is None else \
            torch.tensor(env_indices, device=self.device)
        # Clear schedule for these envs (auto-reset gives them new seeds/comets).
        self._comet_alive[idx] = False
        self._comet_ships0[idx] = 0.0
        self._comet_check[idx] = False
        self._comet_spawn_done[idx] = False
        # Comet ids match kaggle (max regular id + 1 + c = n_regular + c; ids are contiguous
        # 0..n-1 and prior comets are removed before each spawn, so ids repeat).
        nreg = self.planet_alive[idx][:, :COMET_SLOT_START].sum(dim=1).float()
        self._comet_ids[idx] = nreg.unsqueeze(1) + torch.arange(N_COMET_SLOTS, device=self.device).float()

    @torch._dynamo.disable
    def _lazy_comets(self):
        """Compute any comet spawn an env reaches THIS step (step_count == spawn-1). Cheap check
        (5 scalar compares); the heavy generate_comet_paths runs only for the few envs crossing a
        spawn boundary, and only for spawns games actually reach.

        @torch._dynamo.disable: this path is Python-serial (.tolist()/float()/set()/numpy in
        _compute_spawn) and torch.compile HARD-ERRORS trying to trace it (not a clean graph
        break). Disabling dynamo here lets --compile-env compile the surrounding physics while
        this runs eager (it's rare — only the few envs crossing a spawn boundary)."""
        if self.episode_steps <= COMET_SPAWN_STEPS[0]:
            return
        sc = self.step_count
        if getattr(self, "_spawn_prev_t", None) is None:
            self._spawn_si = [si for si, S in enumerate(COMET_SPAWN_STEPS)
                              if S < self.episode_steps]
            self._spawn_prev_t = torch.tensor(
                [COMET_SPAWN_STEPS[si] - 1 for si in self._spawn_si],
                dtype=sc.dtype, device=self.device)
        if not self._spawn_si:
            return
        # (k, N) need matrix, read back once — was one .any() sync per spawn index.
        need = (sc.unsqueeze(0) == self._spawn_prev_t.unsqueeze(1)) \
            & ~self._comet_spawn_done[:, self._spawn_si].T
        need_cpu = need.cpu().numpy()
        for row, si in enumerate(self._spawn_si):
            envs = np.where(need_cpu[row])[0]
            if envs.size:
                self._compute_spawn(envs.tolist(), si)

    def _compute_spawn(self, env_indices, si):
        """Fill the comet schedule for spawn COMET_SPAWN_STEPS[si] for the given envs. Reuses
        kaggle's generate_comet_paths so the ellipse math + comet RNG order (paths THEN ships)
        are byte-identical; folds spawn+advance into the per-step lookup (position, alive,
        check=collision-tested, ships0 at activation)."""
        S, T1 = COMET_SPAWN_STEPS[si], self.episode_steps + 1
        # One batched device-to-CPU readback for all spawning environments.
        idx_t = torch.tensor(env_indices, dtype=torch.long, device=self.device)
        self._comet_spawn_done[idx_t, si] = True
        ip_cpu = self.init_planets[idx_t, :COMET_SLOT_START].cpu().numpy()
        alive_cpu = self.planet_alive[idx_t, :COMET_SLOT_START].cpu().numpy()
        av_cpu = self.angular_velocity[idx_t].cpu().numpy()
        for j, e in enumerate(env_indices):
            init_planets = [
                [int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4]),
                 float(p[5]), float(p[6])]
                for p, a in zip(ip_cpu[j], alive_cpu[j]) if a
            ]
            rng = random.Random(f"orbit_wars-comet-{self.seeds[e]}-{S}")
            paths = _comet_paths_fast(init_planets, float(av_cpu[j]), S, comet_planet_ids=set(),
                                      comet_speed=COMET_SPEED, rng=rng)
            if not paths:
                continue
            comet_ships = min(rng.randint(1, 99), rng.randint(1, 99),
                              rng.randint(1, 99), rng.randint(1, 99))
            # path_index k = step_count-(S-1). k in [0,L-1]=on-path; k==L = kaggle's
            # stay-put expiry tick (still collidable), then gone. All 4 paths are
            # symmetries of the same visible arc, so they share one length L.
            # Stage the whole schedule in NumPy and write per-environment slices.
            L = len(paths[0])
            t0 = S - 1
            span = min(L + 1, T1 - t0)
            if span <= 0:
                continue
            kk = np.minimum(np.arange(span), L - 1)
            xy = np.empty((span, N_COMET_SLOTS, 2), dtype=np.float32)
            for c in range(N_COMET_SLOTS):
                xy[:, c, :] = np.asarray(paths[c], dtype=np.float32)[kk]
            self._comet_xy[e, t0:t0 + span] = torch.from_numpy(xy).to(self.device)
            self._comet_alive[e, t0:t0 + span] = True
            chk = torch.ones(span, N_COMET_SLOTS, dtype=torch.bool)
            chk[0] = False                      # k == 0: not collision-tested
            self._comet_check[e, t0:t0 + span] = chk.to(self.device)
            self._comet_ships0[e, t0, :] = float(comet_ships)

    def _apply_comet_state(self):
        """Set comet slots for the current step_count: activate new comets (owner=-1,
        ships, production) and set alive. Position is applied in step() phase 2/8.
        Returns the per-(N,P) collision-check mask for comet slots this tick."""
        t = self.step_count.clamp(max=self.episode_steps).long()          # (N,)
        bidx = torch.arange(self.num_envs, device=self.device)
        c0 = COMET_SLOT_START
        alive_t = self._comet_alive[bidx, t]                              # (N, 4)
        ships0_t = self._comet_ships0[bidx, t]                            # (N, 4)
        activating = ships0_t > 0                                         # (N, 4)
        sl = slice(c0, c0 + N_COMET_SLOTS)
        # Activate: seed id, owner=-1, ships, production at the comet's first tick.
        self.planets[:, sl, 0] = torch.where(activating, self._comet_ids, self.planets[:, sl, 0])
        self.planets[:, sl, 1] = torch.where(activating, torch.full_like(ships0_t, -1.0),
                                             self.planets[:, sl, 1])
        self.planets[:, sl, 5] = torch.where(activating, ships0_t, self.planets[:, sl, 5])
        self.planets[:, sl, 6] = torch.where(activating, torch.full_like(ships0_t, COMET_PRODUCTION),
                                             self.planets[:, sl, 6])
        self.planets[:, sl, 4] = torch.where(activating, torch.full_like(ships0_t, COMET_RADIUS),
                                             self.planets[:, sl, 4])
        self.planet_alive[:, sl] = alive_t
        return t, bidx

    def _state_dict(self) -> dict:
        return {
            "planets": self.planets,
            "fleets": self.fleets,
            "planet_alive": self.planet_alive,
            "fleet_alive": self.fleet_alive,
            "step": self.step_count,
            "angular_velocity": self.angular_velocity,
        }

    # ---------------------------------------------------------------------
    # Vectorized feature extraction (matches features.extract_features).
    # Returns batched tensors for all N envs in one pass — no Python loops.
    # ---------------------------------------------------------------------

    def get_features(self, player: int, max_planets: int = 48, max_fleets: int = 128) -> dict:
        """Returns dict of batched tensors for model input.

        Output shapes (batched over N envs):
            planet_features:  (N, max_planets, 116) — 20 base + 96 projected-timeline
            fleet_features:   (N, max_fleets, 13)
            global_features:  (N, 11)
            planet_mask:      (N, max_planets) bool
            fleet_mask:       (N, max_fleets) bool
            fire_mask:        (N, MAX_OWNED) bool — can fire (owned planet)
            target_mask:      (N, MAX_OWNED, max_planets) bool — legal target planets
            slot_valid:       (N, MAX_OWNED) bool
            owned_indices:    (N, MAX_OWNED) long
            max_ships:        (N, MAX_OWNED) float — ships available
            owned_count:      list[int] of length N
        """
        N = self.num_envs
        P = max_planets   # truncate to model's expected dim
        F = max_fleets    # model expects 128 even though env stores up to 256

        # Truncate state to model dimensions
        planets = self.planets[:, :P, :]
        planet_alive = self.planet_alive[:, :P]
        init_planets = self.init_planets[:, :P, :]
        fleets = self.fleets[:, :F, :]
        fleet_alive = self.fleet_alive[:, :F]
        # obs_fleet_truncation diagnostic: an env whose LIVE fleets spill past the obs cap F (slots
        # >= F) is partly unseen by the model. Rate = such (env,call) / all (env,call). The rate only
        # says SOME mass is hidden; the *_frac severity metrics below say HOW MUCH (fleet count + ship
        # mass omitted vs total) — the real signal for whether raising max_fleets is worth it.
        if self._obs_calls is not None and 0 <= player < self.num_players:
            self._obs_calls[player] += N
            if F < self.fleet_alive.shape[1]:
                self._obs_trunc[player] += self.fleet_alive[:, F:].any(dim=1).sum().float()
                alive_f = self.fleet_alive.float()
                ships_all = self.fleets[:, :, 6] * alive_f
                self._obs_trunc_fleets[player] += alive_f[:, F:].sum()
                self._obs_total_fleets[player] += alive_f.sum()
                self._obs_trunc_ships[player] += ships_all[:, F:].sum()
                self._obs_total_ships[player] += ships_all.sum()
                # ENEMY-only (owner != this obs player): the obs256-relevant cut — how much of
                # the fleet mass the model is blind to belongs to the OPPONENT (e.g. an inbound
                # strike), not our own in-flight reinforcements.
                enemy_ships = ships_all * (self.fleets[:, :, 1].long() != player).float()
                self._obs_trunc_enemy_ships[player] += enemy_ships[:, F:].sum()
                self._obs_total_enemy_ships[player] += enemy_ships.sum()

        owner = planets[:, :, 1].long()
        x  = planets[:, :, 2];  y  = planets[:, :, 3]
        r  = planets[:, :, 4]
        ships = planets[:, :, 5]
        prod  = planets[:, :, 6]

        is_mine    = (owner == player) & planet_alive
        is_enemy   = (owner != player) & (owner != -1) & planet_alive
        is_neutral = (owner == -1) & planet_alive

        # Owner encoding: mine=1, enemy=-1, neutral=0
        owner_emb = torch.where(is_mine, torch.ones_like(x),
                     torch.where(is_enemy, -torch.ones_like(x), torch.zeros_like(x)))

        dist_to_sun = torch.sqrt((x - CENTER) ** 2 + (y - CENTER) ** 2)
        is_orbiting = ((dist_to_sun + r) < ROTATION_RADIUS_LIMIT) & planet_alive

        # Predicted position 5 turns ahead
        step_f = self.step_count.float().unsqueeze(-1)
        ang_vel = self.angular_velocity.unsqueeze(-1)
        ix = init_planets[:, :, 2]; iy = init_planets[:, :, 3]
        dx = ix - CENTER; dy = iy - CENTER
        orb_r = torch.sqrt(dx * dx + dy * dy)
        init_ang = torch.atan2(dy, dx)
        future_ang = init_ang + ang_vel * (step_f + 5)
        pred_x = torch.where(is_orbiting, CENTER + orb_r * torch.cos(future_ang), x)
        pred_y = torch.where(is_orbiting, CENTER + orb_r * torch.sin(future_ang), y)

        # Incoming fleet pressure (vectorized).
        fx = fleets[:, :, 2]; fy = fleets[:, :, 3]
        fa = fleets[:, :, 4]
        f_owner = fleets[:, :, 1].long()
        f_ships = fleets[:, :, 6]
        fcos = torch.cos(fa); fsin = torch.sin(fa)
        # Planet↔fleet deltas (N, P, F) — used by the threat-ETA channels below.
        vx = x.unsqueeze(2) - fx.unsqueeze(1)
        vy = y.unsqueeze(2) - fy.unsqueeze(1)

        # Pressure-channel attribution: one fleet → its single lead-aware swept-collision target.
        # Mirrors features._resolve_fleet_targets (parity-tested).
        # _fleet_target_idx runs over ALL self.fleets (256); the feature path caps to the first
        # F (=max_fleets). Resolution is per-fleet (geometry only) so slicing is exact.
        _f_tgt = self._fleet_targets()[:, :fleets.shape[1]]                 # (N, F), -1 if none
        _p_ar = torch.arange(x.shape[1], device=self.device).view(1, -1, 1)  # (1, P, 1)
        incoming_pw = (_f_tgt.unsqueeze(1) == _p_ar) & fleet_alive.unsqueeze(1)  # (N, P, F)

        # Planet ch12/13: friendly / enemy inbound mass per planet, attributed via incoming_pw.
        friend = incoming_pw & (f_owner.unsqueeze(1) == player)
        enemy  = incoming_pw & (f_owner.unsqueeze(1) != player) & (f_owner.unsqueeze(1) >= 0)
        friendly_pressure = (f_ships.unsqueeze(1) * friend.float()).sum(dim=2)  # (N, P)
        enemy_pressure    = (f_ships.unsqueeze(1) * enemy.float()).sum(dim=2)
        friendly_pressure_pw = friendly_pressure  # pairwise friendly_contest reuses the same attribution

        # Capture cost
        capture_cost = torch.where(
            is_neutral, ships + 1,
            torch.where(is_enemy, ships + prod * 3 + 1, torch.zeros_like(ships)),
        )

        # Distance from each planet to nearest mine planet.
        # Pairwise (P, P) distance, mask non-mine columns to inf, take min.
        dxp = x.unsqueeze(2) - x.unsqueeze(1)     # (N, P, P)
        dyp = y.unsqueeze(2) - y.unsqueeze(1)
        dpp = torch.sqrt(dxp * dxp + dyp * dyp)
        mine_col = is_mine.unsqueeze(1)            # (N, 1, P)
        # Sentinel must exceed the max possible board distance (diagonal ~141 on
        # a 100x100 board) so it never wins the min over a real owned distance;
        # BOARD_SIZE alone capped feat15 on sparse boards (train/eval mismatch).
        big = torch.full_like(dpp, BOARD_SIZE * 4.0)
        dpp_masked = torch.where(mine_col, dpp, big)
        min_owned_dist = dpp_masked.min(dim=2).values  # (N, P)
        any_mine = is_mine.any(dim=1, keepdim=True)
        min_owned_dist = torch.where(any_mine.expand_as(min_owned_dist),
                                      min_owned_dist, torch.zeros_like(min_owned_dist))

        # is_home heuristic
        is_home = is_mine & (ships <= 10 + prod * 5) & (ships >= 10 - prod)

        # is_comet — alive planets occupying the reserved comet slots (44-47).
        is_comet = torch.zeros_like(is_orbiting)
        if self._has_comets:
            c0 = COMET_SLOT_START
            is_comet[:, c0:c0 + N_COMET_SLOTS] = planet_alive[:, c0:c0 + N_COMET_SLOTS]

        # Connectivity features: owned planets within r=15 / r=30 of each planet.
        # dpp[n,i,j] = dist from planet i to j; mine_row[n,i,1] = is i mine?
        mine_row = is_mine.unsqueeze(2)                                  # (N, P, 1)
        owned_within_15 = ((dpp < 15.0) & mine_row.expand_as(dpp)).sum(dim=1).float()  # (N, P)
        owned_within_30 = ((dpp < 30.0) & mine_row.expand_as(dpp)).sum(dim=1).float()

        # Assemble planet features (N, P, 20). Dead slots get zero — matches
        # legacy extract_features which only iterates over alive planets.
        pf = torch.stack([
            (x - CENTER) / CENTER,           # 0
            (y - CENTER) / CENTER,           # 1
            owner_emb,                       # 2
            r / 2.0,                         # 3
            torch.log1p(ships) / 8.0,        # 4
            prod / 5.0,                      # 5
            is_orbiting.float(),             # 6
            is_comet.float(),                # 7
            dist_to_sun / CENTER,            # 8
            orb_r / CENTER,                  # 9
            (pred_x - CENTER) / CENTER,      # 10
            (pred_y - CENTER) / CENTER,      # 11
            friendly_pressure / 100.0,       # 12
            enemy_pressure / 100.0,          # 13
            torch.log1p(capture_cost) / 8.0, # 14
            min_owned_dist / BOARD_SIZE,     # 15
            is_home.float(),                 # 16
            (owned_within_15 / 8.0).clamp(max=1.0),  # 17 — connectivity r=15
            (owned_within_30 / 12.0).clamp(max=1.0), # 18 — connectivity r=30
            planet_alive.float(),            # 19 — active mask (was 17)
        ], dim=2)  # (N, P, 20)

        # Comet overlay: for alive comet slots, replace the meaningless orbital channels
        # (9 orb_r, 10 pred_x, 11 pred_y — comets don't orbit circularly) with path-aware
        # values that MATCH features.extract_features' comet branch:
        #   ch 9  -> normalized steps-to-departure  (clamp((last_alive_step - now)/NORM, 0, 1))
        #   ch 10 -> comet path position LOOKAHEAD steps ahead, x  (normalized)
        #   ch 11 -> same, y
        if self._has_comets:
            c0 = COMET_SLOT_START
            ns = N_COMET_SLOTS
            T1 = self.episode_steps + 1
            bidx = torch.arange(N, device=self.device)
            t_now = self.step_count.clamp(max=self.episode_steps).long()          # (N,)
            step_ids = torch.arange(T1, device=self.device).view(1, T1, 1)        # (1,T1,1)
            # last step index at which each comet slot is alive (0 if never)
            last_alive = (self._comet_alive.float() * step_ids).amax(dim=1)       # (N, ns)
            remaining = (last_alive - t_now.unsqueeze(1)).clamp(min=0.0)          # (N, ns)
            ahead_t = torch.minimum(t_now.unsqueeze(1) + COMET_FEAT_LOOKAHEAD,
                                    last_alive.long()).clamp(0, self.episode_steps)
            cslot = torch.arange(ns, device=self.device).view(1, ns)
            ahead_xy = self._comet_xy[bidx.view(N, 1), ahead_t, cslot]            # (N, ns, 2)
            sl = slice(c0, c0 + ns)
            pf[:, sl, 9] = (remaining / COMET_LIFE_NORM).clamp(0.0, 1.0)
            pf[:, sl, 10] = (ahead_xy[..., 0] - CENTER) / CENTER
            pf[:, sl, 11] = (ahead_xy[..., 1] - CENTER) / CENTER

        # Zero out dead slots so output matches legacy (which leaves them zero).
        pf = pf * planet_alive.unsqueeze(-1).float()

        # Projected-future timeline (writeup lesson 1): per planet, owner one-hot +
        # log-garrison over the next TIMELINE_K steps assuming no new launches — the raw
        # resolved timeline the winners let the model read instead of untimed aggregates.
        # Projects over the FULL fleet set (all slots, not the max_fleets view) so training
        # matches eval, which sees every fleet in the obs. Comet slots are approximate
        # (projection assumes circular orbits / no expiry); everywhere else it parity-checks
        # against stepping the engine K times with no actions (tests/test_timeline_projection).
        own_ts, garr_ts, timeline_arrivals = project_timeline(
            planets, planet_alive, self.fleets, self.fleet_alive,
            self.angular_velocity, num_players=self.num_players,
            return_arrivals=True)
        tf = timeline_features(own_ts, garr_ts, player)
        pf = torch.cat([pf, tf * planet_alive.unsqueeze(-1).float()], dim=2)  # (N, P, 116)

        # Fleet features (N, F, 13): add destination decoding (gap 1 from roadmap).
        # For each fleet, find the target planet by projecting all planets onto
        # the fleet heading and taking the one with minimum perpendicular distance
        # among those "ahead" (along > 0) and alive.
        f_speed = _ship_speed(f_ships)
        f_dist_sun = torch.sqrt((fx - CENTER) ** 2 + (fy - CENTER) ** 2)
        f_owner_emb = torch.where(
            f_owner == player, torch.ones_like(fx),
            torch.where(f_owner >= 0, -torch.ones_like(fx), torch.full_like(fx, -0.5)),
        )

        # Fleet→planet vectors: transposed views of the (N, P, F) deltas computed above
        # (vx[n,p,f] = x[p] − fx[f] — identical values, no second subtraction cube).
        vx_fp = vx.transpose(1, 2)                 # (N, F, P) planet_x − fleet_x
        vy_fp = vy.transpose(1, 2)
        along_fp = vx_fp * fcos.unsqueeze(2) + vy_fp * fsin.unsqueeze(2)
        perp_fp  = torch.abs(vx_fp * fsin.unsqueeze(2) - vy_fp * fcos.unsqueeze(2))
        alive_expand = planet_alive.unsqueeze(1).expand(-1, F, -1)  # (N, F, P)
        candidate = (along_fp > 0) & (perp_fp < r.unsqueeze(1) + 2.0) & alive_expand
        has_candidate = candidate.any(dim=2)
        # One distance cube shared with the threat-ETA channels below (dist_pf).
        dist_pf = torch.sqrt((vx * vx + vy * vy).clamp(min=1e-9))   # (N, P, F)
        dists_fp = dist_pf.transpose(1, 2)                          # (N, F, P) view
        dists_masked = dists_fp.masked_fill(~candidate, 1e6)
        tgt_idx = dists_masked.argmin(dim=2)  # (N, F) — target planet index per fleet

        # Gather target planet properties
        _gi = tgt_idx.unsqueeze(-1)                                    # (N, F, 1)
        best_dist = dists_fp.gather(2, _gi).squeeze(2)
        dist_to_target = torch.where(
            has_candidate,
            best_dist,
            torch.full_like(best_dist, BOARD_SIZE),
        )
        eta_to_target = (best_dist / f_speed.clamp(min=1e-3)).clamp(min=1.0)
        tgt_owner_f  = owner.unsqueeze(1).expand(-1, F, -1).gather(2, _gi).squeeze(2).long()
        tgt_prod_f   = prod.unsqueeze(1).expand(-1, F, -1).gather(2, _gi).squeeze(2)
        threatens_owned = ((tgt_owner_f == player) & has_candidate & fleet_alive).float()
        eta_feat = torch.where(has_candidate, 1.0 / (eta_to_target + 1.0), torch.zeros_like(eta_to_target))
        tgt_prod_feat = torch.where(has_candidate, tgt_prod_f / 5.0, torch.zeros_like(tgt_prod_f))

        ff = torch.stack([
            (fx - CENTER) / CENTER,                   # 0
            (fy - CENTER) / CENTER,                   # 1
            f_owner_emb,                              # 2
            fcos,                                     # 3
            fsin,                                     # 4
            torch.log1p(f_ships) / 8.0,              # 5
            f_speed / MAX_SHIP_SPEED,                 # 6
            f_dist_sun / CENTER,                      # 7
            eta_feat,                                # 8  urgency
            dist_to_target / BOARD_SIZE,             # 9  distance remaining
            threatens_owned,                          # 10 heads toward player planet
            tgt_prod_feat,                            # 11 target production
            fleet_alive.float(),                      # 12 active mask (was 8)
        ], dim=2)  # (N, F, 13)
        # Zero out dead fleet slots
        ff = ff * fleet_alive.unsqueeze(-1).float()

        # Global features (N, 11): split enemy ships into on_planets vs in_fleets.
        total_owned_ships = (ships * is_mine.float()).sum(dim=1)
        total_owned_prod  = (prod  * is_mine.float()).sum(dim=1)
        num_owned         = is_mine.float().sum(dim=1)
        enemy_planet_ships = (ships * is_enemy.float()).sum(dim=1)
        is_enemy_fleet = (f_owner != player) & (f_owner >= 0) & fleet_alive
        enemy_fleet_ships = (f_ships * is_enemy_fleet.float()).sum(dim=1)
        is_my_fleet = (f_owner == player) & fleet_alive
        my_fleet_ships = (f_ships * is_my_fleet.float()).sum(dim=1)
        fleet_commit = my_fleet_ships / torch.clamp(total_owned_ships + my_fleet_ships, min=1)

        mode_2p = float(self.num_players == 2)
        mode_4p = float(self.num_players == 4)
        player_norm = player / max(self.num_players - 1, 1)

        gf_list = [
            torch.full_like(total_owned_ships, player_norm),  # 0
            torch.clamp(self.step_count.float() / 500.0, 0.0, 1.0),  # 1
            torch.clamp(self.angular_velocity / 0.05, -1.0, 1.0),    # 2
            num_owned / float(MAX_OWNED),                     # 3  normalised to [0,1] within cap
            torch.clamp(total_owned_ships / 500.0, 0.0, 1.0),        # 4
            torch.clamp(total_owned_prod / 20.0, 0.0, 1.0),          # 5
            torch.clamp(enemy_planet_ships / 2000.0, 0.0, 1.0),      # 6  on-planet only
            torch.clamp(enemy_fleet_ships  / 2000.0, 0.0, 1.0),      # 7  in-flight (new)
            torch.clamp(fleet_commit, 0.0, 1.0),                     # 8  (was 7)
            torch.full_like(total_owned_ships, mode_2p),      # 9  (was 8)
            torch.full_like(total_owned_ships, mode_4p),      # 10 (was 9)
        ]
        # 11-13: phase one-hot (early<50 / mid50-100 / late>=100); 14: norm steps-to-next
        # comet spawn = (next_spawn - step)/100 in (0,1], or 1.0 if none remain. Mirrors
        # features.game_phase_channels exactly (parity-tested).
        sc = self.step_count.float()
        early = (sc < 50).float()
        mid = ((sc >= 50) & (sc < 100)).float()
        late = (sc >= 100).float()
        comet_cycle = torch.ones_like(sc)
        for S in reversed(COMET_SPAWN_STEPS):       # descending → smallest S>step wins
            comet_cycle = torch.where(sc < S, (S - sc) / 100.0, comet_cycle)
        gf_list += [early, mid, late, comet_cycle]
        gf = torch.stack(gf_list, dim=1)  # (N, 15)

        # 15-62 (opt-in): the passive rollout's economy series (Yijie's global token carries the
        # same per-turn ship/production differences). Reuses the projection already computed
        # above for the planet channels — no second rollout.
        if self.global_econ:
            gf = torch.cat([gf, global_economy_features(
                planets, planet_alive, self.fleets, self.fleet_alive,
                own_ts, garr_ts, timeline_arrivals, player, self.num_players)], dim=1)  # (N, 63)

        # Action masks
        owned_idx, slot_valid = self.owned_indices_for(player)
        # max_ships per slot: ships at the owned planet
        gather_idx = owned_idx.unsqueeze(-1).expand(-1, -1, 7)
        owned_ships = self.planets.gather(1, gather_idx)[:, :, 5]
        max_ships = owned_ships * slot_valid.float()
        # Fire mask: can fire iff slot is valid and has at least 1 ship
        fire_mask = slot_valid & (max_ships >= 1.0)
        # Angle mask: all angles legal (no sun-blocking for now)
        # Target mask: target-conditioned rollouts sample a live planet. Per-source
        # mask keeps padded owned slots off. With reinforcement OFF (default) only
        # non-own planets are legal; with reinforcement ON, own planets are legal too
        # (friendly arrival reinforces the garrison — see step()), EXCLUDING the launch
        # source planet itself (degenerate self-launch).
        target_owner = owner.unsqueeze(1).expand(-1, MAX_OWNED, -1)
        target_alive = planet_alive.unsqueeze(1).expand(-1, MAX_OWNED, -1)
        if self.allow_reinforce:
            P_idx = torch.arange(owner.shape[1], device=self.device).view(1, 1, -1)
            is_source = (P_idx == owned_idx.unsqueeze(-1))  # (N, MAX_OWNED, P)
            target_mask = target_alive & slot_valid.unsqueeze(-1) & ~is_source
            # Empire-size gate: own planets are legal reinforce targets only when the
            # player owns >= reinforce_gate_min_planets. Enemy/neutral targets are never
            # gated. Below the threshold the agent is attack-only (must expand first).
            if self.reinforce_gate_min_planets > 0:
                is_own = (target_owner == player)  # (N, MAX_OWNED, P)
                # num_owned = true owned-planet count (uncapped, computed above)
                gate_ok = (num_owned >= self.reinforce_gate_min_planets).view(-1, 1, 1)
                # disallow own targets where the empire is too small
                target_mask = target_mask & (~is_own | gate_ok)
            # Forward-staging gate: an own (reinforce) target is legal only if it is
            # closer to the nearest enemy planet than the launch source. Reinforcement
            # flows rear→front (staging), never into a safe rear hoard — the outlet that
            # floods symmetric self-play. Enemy/neutral targets are never constrained.
            if self.reinforce_forward_only:
                is_own = (target_owner == player)  # (N, MAX_OWNED, P)
                enemy_planet = (owner != player) & (owner >= 0) & planet_alive  # (N, P)
                # dpp (N, P, P) pairwise dists computed above
                INF = torch.finfo(dpp.dtype).max
                d2e = torch.where(enemy_planet.unsqueeze(1), dpp,
                                  torch.full_like(dpp, INF)).min(dim=2).values  # (N, P)
                # gather source planet's enemy-distance per owned slot (owned_idx is
                # clamped gather-safe; padded slots are dropped by slot_valid anyway)
                src_d2e = torch.gather(d2e, 1, owned_idx)              # (N, MAX_OWNED)
                forward_ok = d2e.unsqueeze(1) < src_d2e.unsqueeze(-1)  # (N, MAX_OWNED, P)
                # envs with no live enemy planet: forward-staging is moot → don't constrain
                forward_ok = forward_ok | (~enemy_planet.any(dim=1)).view(-1, 1, 1)
                target_mask = target_mask & (~is_own | forward_ok)
            # Reverse-edge cooldown: an own (reinforce) target d is illegal from source s if the
            # REVERSE edge d→s reinforced within K steps (block A→B→A ping-pong). reinf_cd[n,p,u,v]
            # = last step reinforce u→v fired; reverse edge active iff (step - reinf_cd) <= K, so
            # candidate s→d is blocked by reinf_cd[d,s] (the transpose). Ownership/episode resets
            # keep it from mis-blocking recaptured planets. reinforce_cooldown.py is the canon.
            if self.reverse_edge_cooldown > 0 and self.reinf_cd is not None:
                cd = self.reinf_cd[:, player]                            # (N, P_u, P_v) = u→v last step
                rev_active = (self.step_count.view(-1, 1, 1) - cd) <= self.reverse_edge_cooldown
                blocked = rev_active.transpose(1, 2)                     # (N, P_src, P_tgt): s→d blocked by d→s
                blocked_slot = torch.gather(
                    blocked, 1, owned_idx.unsqueeze(-1).expand(-1, -1, owner.shape[1]))  # (N, MAX_OWNED, P)
                is_own_cd = (target_owner == player)
                target_mask = target_mask & ~(is_own_cd & blocked_slot)
        else:
            target_mask = target_alive & (target_owner != player) & slot_valid.unsqueeze(-1)
        # Per-env owned_count for the model
        owned_count = slot_valid.long().sum(dim=1).tolist()

        # ----- Pairwise (src, tgt) features for the model's cross-attention -----
        # Matches features.compute_pairwise_features() in the kaggle path.
        # Output: (N, MAX_OWNED, P, 15) — same channel order as features.py.
        # enemy_contest[n, p] = total enemy fleet ships racing toward planet p in env n.
        # `incoming_pw` (N, P, F) and enemy mask reuse tensors already computed above.
        enemy_fleet = (f_owner != player) & (f_owner >= 0) & fleet_alive   # (N, F)
        enemy_contest = (f_ships.unsqueeze(1) * (incoming_pw & enemy_fleet.unsqueeze(1)).float()).sum(dim=2)  # (N, P)
        # Threat timing (ch 20-21): ETA-profiled enemy pressure. enemy_contest (ch14) sums ALL
        # inbound enemy mass with no ETA; these add the WHEN. eta = Euclidean planet-fleet dist /
        # fleet speed (matches _fleet_target_idx eta_to_target, torch_env.py:1478). Mirrors
        # features.py compute_pairwise_features ch20-21 byte-for-byte.
        enemy_inc = incoming_pw & enemy_fleet.unsqueeze(1)                # (N, P, F)
        # dist_pf (N, P, F) computed once in the fleet-features block above.
        f_speed_pf = _ship_speed(f_ships).unsqueeze(1)                     # (N, 1, F)
        eta_pf = (dist_pf / f_speed_pf.clamp(min=1e-3)).clamp(min=1.0)     # (N, P, F)
        soon = enemy_inc & (eta_pf <= _THREAT_ETA_WINDOW)                  # (N, P, F)
        enemy_mass_soon = (f_ships.unsqueeze(1) * soon.float()).sum(dim=2)  # (N, P)
        eta_for_min = torch.where(enemy_inc, eta_pf, torch.full_like(eta_pf, 1e6))
        threat_imminence = 1.0 / (eta_for_min.min(dim=2).values + 1.0)     # (N, P) in (0, 0.5], 0 if none
        pairwise = self._compute_pairwise(
            planets=planets, planet_alive=planet_alive, P=P,
            owned_idx=owned_idx, slot_valid=slot_valid, player=player,
            enemy_contest=enemy_contest,
            friendly_contest=friendly_pressure_pw,   # (N, P): own ships inbound per target (resolved)
            enemy_mass_soon=enemy_mass_soon,
            threat_imminence=threat_imminence,
            planet_dist=dpp,                         # (N, P, P) pairwise dists computed above
        )
        candidate_ships = torch.where(
            pairwise[..., 5] > 0.5,
            torch.round(pairwise[..., 24] * 200.0),
            max_ships.unsqueeze(-1).expand_as(pairwise[..., 5]),
        )
        candidate_eta = torch.ceil(
            pairwise[..., 2] * BOARD_SIZE /
            _ship_speed(candidate_ships).clamp(min=1e-6)
        ).clamp(min=1.0)
        candidate = candidate_timeline_features(
            planets, planet_alive, timeline_arrivals, own_ts, garr_ts, player,
            candidate_ships, candidate_eta, owned_idx, slot_valid,
        )
        pairwise = torch.cat([pairwise, candidate], dim=-1)
        # Stash this player's raw resolved-size table for intent decode (_apply_actions reads it).
        if self.ship_bin_mode == "intent":
            self._intent_sizes[player] = self._pw_intent_sizes
        elif self.ship_bin_mode == "binary":
            commit_sizes, commit_feasible = _resolve_binary_commit(
                pairwise, max_ships, gates=self.binary_commit_gates)
            legal_commit = target_mask & commit_feasible
            has_commit = legal_commit.any(dim=-1)
            # Rows with no feasible commit retain a valid but unused target categorical;
            # fire_mask=False makes the only executed action NOOP.
            target_mask = torch.where(has_commit.unsqueeze(-1), legal_commit, target_mask)
            fire_mask = fire_mask & has_commit
            self._binary_commit_sizes[player] = commit_sizes

        return {
            "planet_features": pf,
            "fleet_features":  ff,
            "global_features": gf,
            "planet_mask":     planet_alive,
            "fleet_mask":      fleet_alive,
            "fire_mask":       fire_mask,
            "target_mask":     target_mask,
            "slot_valid":      slot_valid,
            "owned_indices":   owned_idx,
            "max_ships":       max_ships,
            "owned_count":     owned_count,
            "pairwise_features": pairwise,
        }

    def _compute_pairwise(self, planets, planet_alive, P, owned_idx, slot_valid, player,
                          planet_dist,
                          enemy_contest=None, friendly_contest=None,
                          enemy_mass_soon=None, threat_imminence=None):
        # planet_dist: (N, P, P) pairwise planet distances from get_features (its dpp),
        # passed to avoid recomputing the distance cube.
        """Vectorized counterpart of features.compute_pairwise_features().

        Returns (N, MAX_OWNED, P, 22) float32 on self.device. Channel order:
          0  sin(angle src→tgt)
          1  cos(angle src→tgt)
          2  distance / BOARD_SIZE
          3  1 / (eta@~20ships + 1)
          4  sun-safe flag
          5  target is mine
          6  target is enemy
          7  target is neutral
          8  target production / 5
          9  valid flag (slot_valid AND target_alive)
          10 ships_at_arrival / 200  — tgt_ships + tgt_prod * eta (capped 500)
          11 capture gap at arrival — (ships_at_arrival - capture_cost) / 200, clipped [-1,5]
             capture_cost = tgt_ships+1 (neutral) or tgt_ships+3*prod+1 (enemy planet)
             positive = more ships at arrival than needed to capture (hard to take)
          12 roi_20  — (prod*20 - cap_cost_at_arrival) / cap_cost_at_arrival, clipped [-1,1]
          13 roi_50  — same at horizon 50
          14 enemy_contest / 100  — total enemy fleet ships racing toward this target
          15 reachable_enemy_mass / 100 — distance-decayed enemy garrison reachable to this target
          16 capture_value_40 — production value over a capped 40-step horizon / board production
          17 reactive_roi_40 — value vs cap_cost_at_arrival + contest + reachable enemy mass
          18 friendly_reachable_mass / 100 — distance-decayed friendly support mass
          19 keepability_margin / 100 — friendly support minus enemy contest/reaction
          20 enemy_mass_soon / 100 — enemy fleet mass landing within _THREAT_ETA_WINDOW steps (clamp 5)
          21 threat_imminence — 1/(min_enemy_eta+1); urgency in (0,0.5], 0 if no enemy inbound
        """
        N = self.num_envs
        device = self.device

        # Source positions per (env, slot): gather along the planet axis
        gather_idx = owned_idx.unsqueeze(-1).expand(-1, -1, 7)  # (N, MO, 7)
        src = self.planets[:, :, :7].gather(1, gather_idx)       # (N, MO, 7)
        sx = src[:, :, 2]                                        # (N, MO)
        sy = src[:, :, 3]

        # All target positions: (N, 1, P) broadcastable against (N, MO, 1)
        tx = planets[:, :, 2].unsqueeze(1)                       # (N, 1, P)
        ty = planets[:, :, 3].unsqueeze(1)                       # (N, 1, P)
        owner_t = planets[:, :, 1].unsqueeze(1).long()           # (N, 1, P)
        ships_t = planets[:, :, 5].unsqueeze(1)                  # (N, 1, P) — current ships
        prod_t = planets[:, :, 6].unsqueeze(1)                   # (N, 1, P)
        alive_t = planet_alive.unsqueeze(1)                      # (N, 1, P)
        regular_t_1d = (torch.arange(P, device=device) < COMET_SLOT_START).view(1, P)
        regular_alive = planet_alive & regular_t_1d

        sx_b = sx.unsqueeze(-1)                                  # (N, MO, 1)
        sy_b = sy.unsqueeze(-1)

        dx0 = tx - sx_b                                          # (N, MO, P)
        dy0 = ty - sy_b
        dist2_0 = dx0 * dx0 + dy0 * dy0
        dist0 = torch.sqrt(dist2_0.clamp(min=1e-9))
        ETA_SPEED = 1.0 + (MAX_SHIP_SPEED - 1.0) * (math.log(20.0) / math.log(1000.0)) ** 1.5
        eta0 = (dist0 / ETA_SPEED).ceil().clamp(min=1.0)

        init_ang_t = self._planet_initial_angle[:, :P].unsqueeze(1)
        orb_r_t = self._planet_orbital_r[:, :P].unsqueeze(1)
        is_orb_t = self._planet_is_orbiting[:, :P].unsqueeze(1)
        step_f = self.step_count.float().view(N, 1, 1)
        ang_vel = self.angular_velocity.view(N, 1, 1)
        future_ang = init_ang_t + ang_vel * (step_f + eta0)
        arr_x = torch.where(is_orb_t, CENTER + orb_r_t * torch.cos(future_ang), tx)
        arr_y = torch.where(is_orb_t, CENTER + orb_r_t * torch.sin(future_ang), ty)

        dx = arr_x - sx_b
        dy = arr_y - sy_b
        dist2 = dx * dx + dy * dy
        dist = torch.sqrt(dist2.clamp(min=1e-9))
        sin_a = dy / dist
        cos_a = dx / dist
        eta = (dist / ETA_SPEED).ceil().clamp(min=1.0)

        # Sun-cross check: point-to-segment distance from (CENTER, CENTER) to src→tgt
        seg_len2 = dist2_0.clamp(min=1e-9)
        t_param = ((CENTER - sx_b) * dx0 + (CENTER - sy_b) * dy0) / seg_len2
        t_param = t_param.clamp(0.0, 1.0)
        proj_x = sx_b + t_param * dx0
        proj_y = sy_b + t_param * dy0
        sun_d = torch.sqrt((proj_x - CENTER) ** 2 + (proj_y - CENTER) ** 2)
        sun_safe = (sun_d >= SUN_RADIUS).float()

        is_mine_t = (owner_t == player).float() * alive_t.float()        # (N, 1, P)
        is_enemy_t = ((owner_t != player) & (owner_t != -1)).float() * alive_t.float()
        is_neutral_t = (owner_t == -1).float() * alive_t.float()

        # Broadcast scalar / 1-along-MO channels to (N, MO, P)
        MO = owned_idx.shape[1]
        is_mine_b = is_mine_t.expand(-1, MO, -1)
        is_enemy_b = is_enemy_t.expand(-1, MO, -1)
        is_neutral_b = is_neutral_t.expand(-1, MO, -1)
        prod_b = (prod_t / 5.0).expand(-1, MO, -1)
        valid_b = alive_t.float().expand(-1, MO, -1)

        total_board_prod = (planets[:, :, 6] * regular_alive.float()).sum(dim=1).clamp(min=1.0)  # (N,)
        value_h = (self.episode_steps - self.step_count.float()).clamp(min=0.0, max=_VALUE_HORIZON)  # (N,)
        owner_t_2d = planets[:, :, 1].long()
        owner_weight = torch.where(
            owner_t_2d == player,
            torch.ones_like(planets[:, :, 6]),
            torch.where(owner_t_2d == -1,
                        torch.ones_like(planets[:, :, 6]),
                        torch.full_like(planets[:, :, 6], 2.0)),
        )
        capture_value_mass = owner_weight * planets[:, :, 6] * value_h.unsqueeze(1)
        capture_value_mass = torch.where(regular_alive, capture_value_mass, torch.zeros_like(capture_value_mass))
        capture_value = (capture_value_mass / (total_board_prod * _VALUE_HORIZON).unsqueeze(1)).clamp(0.0, 2.0)

        # Ships-at-arrival features (ch 10-11)
        ships_b = ships_t.expand(-1, MO, -1)                     # (N, MO, P)
        # NEUTRALS DON'T REGROW (engine applies production only to owner != -1) -> no prod
        # accrual during flight. Phantom neutral production priced cheap rotating neutrals as
        # far more expensive than they are. Mirrors features.py compute_pairwise_features.
        prod_growth = torch.where(owner_t == -1, torch.zeros_like(prod_b), prod_b * 5.0 * eta)
        ships_at_arr = (ships_b + prod_growth).clamp(max=500.0) / 200.0
        cap_cost = torch.where(
            owner_t == -1,
            ships_t + 1.0,
            torch.where(owner_t != player, ships_t + prod_t * 3.0 + 1.0, torch.zeros_like(ships_t)),
        ).expand(-1, MO, -1)
        cap_gap = ((ships_at_arr * 200.0 - cap_cost) / 200.0).clamp(-1.0, 5.0)

        # ROI features (ch 12-13): (prod*H - cap_cost_at_arrival) / cap_cost_at_arrival
        prod_actual = prod_b * 5.0                               # (N, MO, P) unnormalized
        ships_actual = ships_at_arr * 200.0                      # (N, MO, P) unnormalized
        owner_exp = owner_t.expand(-1, MO, -1)                   # (N, MO, P)
        cap_at_arr = torch.where(
            owner_exp == -1,
            ships_actual + 1.0,
            torch.where(owner_exp != player,
                        ships_actual + prod_actual * 3.0 + 1.0,
                        torch.zeros_like(ships_actual))
        )
        safe_cap = cap_at_arr.clamp(min=1.0)
        roi_20 = ((prod_actual * 20.0 - cap_at_arr) / safe_cap).clamp(-1.0, 1.0)
        roi_50 = ((prod_actual * 50.0 - cap_at_arr) / safe_cap).clamp(-1.0, 1.0)

        # Deflate capture roi by friendly ships already inbound (symmetric to enemy_contest):
        # a target we already have a fleet en route to offers ~0 marginal return to a NEW
        # launch. coverage in [0,1] = inbound / capture-cost; own (reinforce) targets are never
        # deflated. Matches features.py compute_pairwise_features (parity-tested).
        if friendly_contest is not None:
            fc_b = friendly_contest.unsqueeze(1).expand(-1, MO, -1)          # (N, MO, P)
            coverage = torch.where(owner_exp != player,
                                   (fc_b / safe_cap).clamp(max=1.0),
                                   torch.zeros_like(safe_cap))
            roi_20 = roi_20 * (1.0 - coverage)
            roi_50 = roi_50 * (1.0 - coverage)

        # Enemy contest feature (ch 14): broadcast (N, P) → (N, MO, P)
        if enemy_contest is not None:
            contest_b = (enemy_contest / 100.0).clamp(max=5.0).unsqueeze(1).expand(-1, MO, -1)
            enemy_contest_raw = enemy_contest
        else:
            contest_b = torch.zeros(N, MO, P, device=device)
            enemy_contest_raw = torch.zeros(N, P, device=device)
        if friendly_contest is not None:
            friendly_contest_raw = friendly_contest
        else:
            friendly_contest_raw = torch.zeros(N, P, device=device)

        # Reachable enemy planet mass (ch 15): distance-decayed enemy garrison that could
        # reinforce/contest each target within the horizon. Source-slot independent, so
        # computed per-target and broadcast. Raw (no rho/eta scaling) — the per-target head
        # learns its own reaction coefficient. Mirrors features.compute_pairwise_features ch15
        # and the dm-floor enemy_mass term.
        gar_a = planets[:, :, 5]
        own_a = planets[:, :, 1].long()
        # Pairwise planet distances: the caller's dpp. Identical values where consumed —
        # the diagonal (where the previously-computed clamped sqrt differed by ~3e-5)
        # is masked by not_self below.
        pde = planet_dist                                       # (N, P_src, P_tgt)
        src_reach_e = (_ship_speed(gar_a) * _DM_HORIZON).clamp(min=1e-6)   # (N, P_src)
        dec = (1.0 - pde / src_reach_e.unsqueeze(2)).clamp(min=0.0)        # (N, P_src, P_tgt)
        not_self = ~torch.eye(P, dtype=torch.bool, device=device).unsqueeze(0)
        enemy_src = (planet_alive & (own_a != player) & (own_a >= 0)).unsqueeze(2)  # (N,P_src,1)
        valid_e = enemy_src & planet_alive.unsqueeze(1) & not_self
        reach_em = torch.where(valid_e, gar_a.unsqueeze(2) * dec,
                               torch.zeros_like(dec)).sum(dim=1)           # (N, P_tgt)
        reach_b = (reach_em / 100.0).clamp(max=5.0).unsqueeze(1).expand(-1, MO, -1)

        # Reachable friendly planet mass and keepability pressure (ch 18-19). This mirrors the
        # enemy reach calculation but uses our planets, excluding transient comet slots. It tells
        # the target head whether a captured/saved planet has nearby friendly support.
        friend_src = (regular_alive & (own_a == player)).unsqueeze(2)       # (N,P_src,1)
        valid_f = friend_src & regular_alive.unsqueeze(1) & not_self
        reach_fm = torch.where(valid_f, gar_a.unsqueeze(2) * dec,
                               torch.zeros_like(dec)).sum(dim=1)           # (N, P_tgt)
        friendly_reach_b = (reach_fm / 100.0).clamp(max=5.0).unsqueeze(1).expand(-1, MO, -1)

        enemy_pressure = reach_em + enemy_contest_raw
        friendly_support = reach_fm + friendly_contest_raw
        keepability_margin = ((friendly_support - enemy_pressure) / 100.0).clamp(-5.0, 5.0)
        keepability_margin = torch.where(regular_alive, keepability_margin, torch.zeros_like(keepability_margin))
        keepability_b = keepability_margin.unsqueeze(1).expand(-1, MO, -1)

        target_value_b = capture_value.unsqueeze(1).expand(-1, MO, -1)
        target_value_mass_b = capture_value_mass.unsqueeze(1).expand(-1, MO, -1)
        enemy_pressure_b = enemy_pressure.unsqueeze(1).expand(-1, MO, -1)
        reactive_cost = cap_at_arr + enemy_pressure_b
        non_mine_regular = ((owner_exp != player) & regular_alive.unsqueeze(1).expand(-1, MO, -1))
        reactive_roi_40 = ((target_value_mass_b - reactive_cost) / reactive_cost.clamp(min=1.0)).clamp(-1.0, 1.0)
        reactive_roi_40 = torch.where(non_mine_regular, reactive_roi_40, torch.zeros_like(reactive_roi_40))

        # Threat-timing channels (ch 20-21): ETA-profiled enemy pressure. Computed in get_features
        # (eta = dist/speed per (target,fleet)); here just normalize+broadcast like ch14.
        if enemy_mass_soon is not None:
            mass_soon_b = (enemy_mass_soon / 100.0).clamp(max=5.0).unsqueeze(1).expand(-1, MO, -1)
        else:
            mass_soon_b = torch.zeros(N, MO, P, device=device)
        if threat_imminence is not None:
            imminence_b = threat_imminence.unsqueeze(1).expand(-1, MO, -1)
        else:
            imminence_b = torch.zeros(N, MO, P, device=device)

        # Intent-sizing resolved sizes (ch 22-25): exact ships for capture / capture-defend /
        # maintain / all-in, clamped to source garrison. Torch twin of features.py; read back at
        # decode. reach_em / enemy_mass_soon are used RAW here (not the /100 normalized channels).
        src_ships_b = src[:, :, 5].unsqueeze(-1).expand(-1, -1, P)               # (N, MO, P)
        reach_em_raw_b = reach_em.unsqueeze(1).expand(-1, MO, -1)                # (N, MO, P)
        mass_soon_raw = (enemy_mass_soon if enemy_mass_soon is not None
                         else torch.zeros(N, P, device=device)).unsqueeze(1).expand(-1, MO, -1)
        intent_sizes = _resolve_intent_sizes(
            cap_at_arr, reach_em_raw_b, mass_soon_raw, src_ships_b, owner_exp == player)  # (N,MO,P,4)
        intent_sizes_n = (intent_sizes.clamp(max=500.0) / 200.0)                 # (N, MO, P, 4)

        # Stack channels
        out = torch.stack([
            sin_a, cos_a, dist / BOARD_SIZE, 1.0 / (eta + 1.0),
            sun_safe, is_mine_b, is_enemy_b, is_neutral_b, prod_b, valid_b,
            ships_at_arr, cap_gap, roi_20, roi_50, contest_b, reach_b,
            target_value_b, reactive_roi_40, friendly_reach_b, keepability_b,
            mass_soon_b, imminence_b,
        ], dim=-1)  # (N, MO, P, 22)
        out = torch.cat([out, intent_sizes_n], dim=-1)  # (N, MO, P, 26) — + intent resolved sizes
        self._pw_intent_sizes = intent_sizes            # (N, MO, P, 4) raw — for decode read-back

        # Zero out invalid owned slots AND invalid target planets (match kaggle path)
        slot_valid_b = slot_valid.unsqueeze(-1).unsqueeze(-1).float()    # (N, MO, 1, 1)
        target_valid_b = alive_t.float().expand(-1, MO, -1).unsqueeze(-1)  # (N, MO, P, 1)
        out = out * slot_valid_b * target_valid_b
        return out

    # ---------------------------------------------------------------------
    # Owned-planet indices per player — vectorized.
    # For each env, returns the planet-array indices of the highest-GARRISON
    # MAX_OWNED owned planets (ties → lowest array index). `slot_valid` masks
    # the empty slots when fewer than MAX_OWNED are owned. See the docstring.
    # ---------------------------------------------------------------------

    def owned_indices_for(self, player: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (owned_idx: (N, MAX_OWNED) long, slot_valid: (N, MAX_OWNED) bool).

        When more than MAX_OWNED planets are owned, the 16
        source slots are the highest-GARRISON owned planets (ties → lowest array index), NOT
        the first 16 by array index. Owning >16 happens ~16% of steps (up to 30 owned), and
        firing only from the lowest-index 16 left up to 14 force-bearing planets inert. Ranking
        by garrison keeps the planets that can actually contribute ships (attack/reinforce/hold).
        Integer key `-round(ships)*P + idx` == sort by (-ships, idx): a 1-ship difference (=P)
        always outweighs any index difference (idx < P), so it is parity-exact with the eval/
        export selection in features.py / action_mask.py. No-op at <=16 owned (all get a slot).
        """
        owner = self.planets[:, :, 1]
        ships = self.planets[:, :, 5]
        is_mine = (owner.long() == player) & self.planet_alive          # (N, P)
        N, P = is_mine.shape
        idx_grid = torch.arange(P, device=self.device).expand(N, P)
        mine_key = -torch.round(ships).long() * P + idx_grid            # smaller = higher garrison
        SENTINEL = 1 << 40                                              # > any mine_key
        scores = torch.where(is_mine, mine_key, torch.full_like(idx_grid, SENTINEL))
        # topk INDICES (not values) → the selected planets' array indices.
        _, owned_idx = torch.topk(scores, MAX_OWNED, dim=1, largest=False)  # (N, MAX_OWNED)
        slot_valid = torch.gather(is_mine, 1, owned_idx)                # masked-out where filled by a non-mine
        return owned_idx, slot_valid

    # ---------------------------------------------------------------------
    def _target_intercept_angle(
        self,
        src_x: torch.Tensor,
        src_y: torch.Tensor,
        src_r: torch.Tensor,
        ship_count: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorised target->launch-angle intercept.

        Mirrors action_mask._target_intercept_angle so training and inference aim
        identically: predict the target
        from its CURRENT orbit position, subtract the src+tgt surface gap, and run
        8 continuous (non-quantised) lead iterations.
        """
        P = self.planets.shape[1]
        target_idx = target_idx.long().clamp(0, P - 1)
        gather_idx = target_idx.unsqueeze(-1).expand(-1, -1, 7)
        tgt = self.planets.gather(1, gather_idx)
        tx = tgt[:, :, 2]
        ty = tgt[:, :, 3]
        tgt_r = tgt[:, :, 4]

        speed = _ship_speed(ship_count)
        ang_vel = self.angular_velocity.unsqueeze(1)

        # Orbit (radius + phase) from the target's CURRENT position; static if at/
        # beyond the rotation-radius limit (engine leaves it fixed).
        dx0 = tx - CENTER
        dy0 = ty - CENTER
        orbit_r = torch.sqrt(dx0 * dx0 + dy0 * dy0)
        static = (orbit_r + tgt_r) >= ROTATION_RADIUS_LIMIT
        phase0 = torch.atan2(dy0, dx0)

        gap = src_r + 0.1 + tgt_r  # source surface + launch offset + target surface
        dist0 = torch.sqrt((tx - src_x) ** 2 + (ty - src_y) ** 2)
        t = ((dist0 - gap) / speed).clamp(min=0.0)
        for _ in range(8):
            a = phase0 + ang_vel * t
            px = torch.where(static, tx, CENTER + orbit_r * torch.cos(a))
            py = torch.where(static, ty, CENTER + orbit_r * torch.sin(a))
            dist = torch.sqrt((px - src_x) ** 2 + (py - src_y) ** 2)
            t = ((dist - gap) / speed).clamp(min=0.0)
        a = phase0 + ang_vel * t
        px = torch.where(static, tx, CENTER + orbit_r * torch.cos(a))
        py = torch.where(static, ty, CENTER + orbit_r * torch.sin(a))

        return torch.atan2(py - src_y, px - src_x) % (2 * math.pi)

    # ---------------------------------------------------------------------
    # Apply actions for one player. Launches fleets from owned planets.
    # actions: (N, MAX_OWNED, 3) int — [fire, angle_bin, ship_bin]
    #      or (N, MAX_OWNED, 4) int — [fire, angle_bin, ship_bin, target_idx]
    # ---------------------------------------------------------------------

    def _apply_actions(self, actions: torch.Tensor, owner_id: int,
                       angle_override: torch.Tensor = None):
        if actions is None:
            return
        owned_idx, slot_valid = self.owned_indices_for(owner_id)
        N = self.num_envs

        # Decode action components
        fire = actions[:, :, 0].bool() & slot_valid                # (N, MAX_OWNED)
        angle_bin = actions[:, :, 1].long().clamp(0, NUM_ANGLE_BINS - 1)

        # Gather source planet state: (N, MAX_OWNED, 7). Done early so the
        # fraction-mode decode can scale by src_ships.
        gather_idx = owned_idx.unsqueeze(-1).expand(-1, -1, 7)
        src = self.planets.gather(1, gather_idx)                  # (N, MAX_OWNED, 7)
        src_x = src[:, :, 2]; src_y = src[:, :, 3]; src_r = src[:, :, 4]
        src_ships = src[:, :, 5]; src_owner = src[:, :, 1].long()

        # Decode ship_bin -> ship count. "absolute" uses fixed table; "fraction"
        # scales by max sendable ships, matching compute_action_masks() and
        # bc_frac.py labels: keep one ship behind when possible.
        if self.ship_bin_mode == "binary":
            ship_count = src_ships
        elif self.ship_bin_mode == "intent":
            # Intent index; the exact ship count is resolved AFTER the target is decoded (needs
            # the chosen target's resolved-size row). Placeholder = source garrison until then.
            ship_count = src_ships.clamp(min=1.0)
        elif self.ship_bin_mode == "fraction":
            num_bins = len(FRACTION_BIN_VALUES)
            ship_bin = actions[:, :, 2].long().clamp(0, num_bins - 1)
            frac = self._frac_bins_t[ship_bin]                    # (N, MAX_OWNED)
            max_sendable = (src_ships - 1.0).clamp(min=1.0)
            ship_count = torch.round(frac * max_sendable).clamp(min=1.0)
        else:
            ship_bin = actions[:, :, 2].long().clamp(0, NUM_SHIP_BINS - 1)
            ship_count = self._ship_counts_t[ship_bin]            # (N, MAX_OWNED)

        # Angle-bin decode uses the BIN CENTER (external-opponent action fallback).
        # Target mode executes the target head by converting target_idx to an
        # intercept angle while keeping angle_bin in storage for compatibility.
        angle = (angle_bin.float() + 0.5) * ANGLE_BIN_WIDTH       # (N, MAX_OWNED)
        target_valid = torch.ones_like(fire, dtype=torch.bool)
        if self.action_decode == "target" and actions.shape[-1] >= 4:
            raw_target_idx = actions[:, :, 3].long()
            use_target_decode = raw_target_idx >= 0
            target_idx = raw_target_idx.clamp(0, self.planets.shape[1] - 1)
            # Intent sizing (#4): resolve intent → exact ships from the chosen target's row of the
            # per-player resolved-size table stashed by get_features. Overrides the placeholder.
            if self.ship_bin_mode == "binary":
                bsz = self._binary_commit_sizes.get(owner_id)
                if bsz is not None:
                    ti = target_idx.clamp(0, bsz.shape[2] - 1)
                    ship_count = torch.gather(bsz, 2, ti.unsqueeze(-1)).squeeze(-1)
            elif self.ship_bin_mode == "intent":
                isz = self._intent_sizes.get(owner_id)
                if isz is not None:
                    intent = actions[:, :, 2].long().clamp(0, NUM_INTENTS - 1)          # (N, MO)
                    ti = target_idx.clamp(0, isz.shape[2] - 1)
                    row = torch.gather(
                        isz, 2, ti.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, NUM_INTENTS)
                    ).squeeze(2)                                                          # (N, MO, 4)
                    ship_count = torch.gather(row, 2, intent.unsqueeze(-1)).squeeze(2).clamp(min=1.0)
            target_gather = target_idx.unsqueeze(-1).expand(-1, -1, 7)
            tgt = self.planets.gather(1, target_gather)
            target_owner = tgt[:, :, 1].long()
            target_ships = tgt[:, :, 5]                            # current defense at the target
            target_alive = self.planet_alive.gather(1, target_idx)
            if self.allow_reinforce:
                # Own planets are valid targets (friendly arrival reinforces the
                # garrison); only the launch source planet itself is invalid.
                tgt_ok = target_alive & (target_idx != owned_idx)
            else:
                tgt_ok = target_alive & (target_owner != owner_id)
            target_valid = torch.where(
                use_target_decode,
                tgt_ok,
                torch.ones_like(target_alive, dtype=torch.bool),
            )
            target_angle = self._target_intercept_angle(src_x, src_y, src_r, ship_count, target_idx)
            angle = torch.where(use_target_decode, target_angle, angle)

        # Continuous-angle override (NaN = none). External heuristics emit a precise
        # continuous intercept angle; the real engine uses it directly, but our 144-bin
        # angle decode quantizes it (±~1.25-2.5°) and handicaps their aiming in-sim. Apply
        # the raw angle here so aiming-heavy opponents play at true strength (matches eval).
        if angle_override is not None:
            has_cont = ~torch.isnan(angle_override)
            angle = torch.where(has_cont, angle_override, angle)

        # Validate: planet still owned by this player AND has enough ships
        valid_owner = (src_owner == owner_id) & slot_valid
        # overask audit (nominal = pre-clamp ship_count): an intended launch whose ask exceeds
        # the source garrison. In "drop" mode it's voided; in "clamp" mode it sends the whole
        # garrison (matching eval). Measured regardless of mode so the A/B sees the same intent.
        _attempted = fire & valid_owner & target_valid & (ship_count > 0)
        _overask = _attempted & (ship_count > src_ships)
        if self.ship_overflow_mode == "clamp":
            ship_count = torch.minimum(ship_count, src_ships)
        valid_ships = src_ships >= ship_count
        can_fire = fire & valid_owner & valid_ships & target_valid & (ship_count > 0)  # (N, MAX_OWNED)
        if self._attempt_step is not None:
            _sc = self.step_count
            _w0 = (_sc < 50).float(); _w1 = ((_sc >= 50) & (_sc < 100)).float(); _w2 = (_sc >= 100).float()
            _oa = _overask.sum(dim=1).float(); _at = _attempted.sum(dim=1).float()    # (N,)
            self._overask_step[:, owner_id, 0] += _oa * _w0
            self._overask_step[:, owner_id, 1] += _oa * _w1
            self._overask_step[:, owner_id, 2] += _oa * _w2
            self._attempt_step[:, owner_id, 0] += _at * _w0
            self._attempt_step[:, owner_id, 1] += _at * _w1
            self._attempt_step[:, owner_id, 2] += _at * _w2

        # Reinforcement discipline (#1 garrison floor + #2 transit cost) + reinforce_rate
        # metric. is_reinforce: a launch whose target is one of OUR OWN planets — only
        # possible with allow_reinforce + target decode. Attacks (enemy/neutral) untouched.
        if (self.allow_reinforce and self.action_decode == "target"
                and actions.shape[-1] >= 4):
            is_reinforce = use_target_decode & (target_owner == owner_id)  # (N, MAX_OWNED)
            # #1 Garrison floor: veto any reinforce launch that would drain its source
            # below the floor (mask, not penalty → no Nash risk).
            if self.reinforce_garrison_floor > 0.0:
                would_underflow = is_reinforce & ((src_ships - ship_count) < self.reinforce_garrison_floor)
                can_fire = can_fire & ~would_underflow
            # Reverse-edge cooldown bookkeeping (after the floor veto, so only REALIZED reinforces
            # arm the reverse block): (1) clear edges touching any planet we don't currently own —
            # recapture starts clean (the ownership-change reset, better than a static exception);
            # (2) record this step's executed reinforces src→tgt at the current step.
            if self.reverse_edge_cooldown > 0 and self.reinf_cd is not None:
                cur_owner = self.planets[:, :, 1].long()                 # (N, P)
                no = (cur_owner != owner_id)                             # (N, P): not ours now
                cd = self.reinf_cd[:, owner_id]                          # (N, P, P)
                clear = no.unsqueeze(-1) | no.unsqueeze(1)               # edge touches a non-owned planet
                cd = torch.where(clear, torch.full_like(cd, _REINF_CD_NEVER), cd)
                rec = can_fire & is_reinforce                            # (N, MAX_OWNED)
                step_now = self.step_count                               # (N,)
                # Record executed edges src→tgt in one gather/scatter (the old per-slot
                # loop cost up to MAX_OWNED CPU-GPU syncs per player per step). Slots are
                # distinct sources (topk indices), so the flattened edge ids are unique
                # per env; non-recording slots write their current value back (no-op).
                Pp = cd.shape[1]
                cd_flat = cd.view(N, Pp * Pp)
                edge = owned_idx * Pp + target_idx                       # (N, MAX_OWNED)
                cur = cd_flat.gather(1, edge)
                val = torch.where(rec, step_now.unsqueeze(1).expand_as(edge), cur)
                cd_flat.scatter_(1, edge, val)
                self.reinf_cd[:, owner_id] = cd
            # #2 Per-ship transit cost: accumulate ships sent to own planets this step
            # for the launching player; the penalty is applied to the reward in step().
            # Counts only launches that actually fire (post-floor-veto).
            if self.reinforce_cost > 0.0:
                reinforce_ships = (ship_count * (can_fire & is_reinforce).float()).sum(dim=1)  # (N,)
                self._reinforce_ships[:, owner_id] = self._reinforce_ships[:, owner_id] + reinforce_ships
            # reinforce_rate metric: per-(env,player) counts of realized launches (post
            # floor-veto) and how many were reinforcement. train_torch combines these with
            # train_mask used for the current policy's reinforce-rate telemetry.
            if self._fire_launch_count is not None:
                fire_per = can_fire.sum(dim=1).float()                       # (N,)
                reinf_per = (can_fire & is_reinforce).sum(dim=1).float()     # (N,)
                self._fire_launch_count[:, owner_id] += fire_per
                self._reinforce_launch_count[:, owner_id] += reinf_per
                # reinf-by-step: bin this step's launches by episode step-window [<50,50-100,>100]
                sc = self.step_count
                w0 = (sc < 50).float(); w1 = ((sc >= 50) & (sc < 100)).float(); w2 = (sc >= 100).float()
                self._fire_step[:, owner_id, 0] += fire_per * w0
                self._fire_step[:, owner_id, 1] += fire_per * w1
                self._fire_step[:, owner_id, 2] += fire_per * w2
                self._reinf_step[:, owner_id, 0] += reinf_per * w0
                self._reinf_step[:, owner_id, 1] += reinf_per * w1
                self._reinf_step[:, owner_id, 2] += reinf_per * w2
                is_neutral = use_target_decode & (target_owner < 0)  # neutral planet owner = -1
                self._neutral_launch_count[:, owner_id] += (can_fire & is_neutral).sum(dim=1).float()

        # SUFFICIENT-COMMIT MASK (arrival-aware): veto a NEUTRAL attack launch whose
        # ship count + friendly inbound can't beat the target's PROJECTED defense at
        # arrival (current garrison + production×ETA + enemy inbound arriving before
        # us). Enemy targets are exempt — under-strength attacks on enemies can soften,
        # feint, or arrive as a second wave. Reinforces (own targets) are untouched.
        if (self.sufficient_commit_factor > 0.0
                and self.action_decode == "target" and actions.shape[-1] >= 4):
            is_neutral_attack = use_target_decode & (target_owner < 0)
            if is_neutral_attack.any():
                tgt_x = tgt[:, :, 2]; tgt_y = tgt[:, :, 3]
                # ETA from source to target (fleet speed depends on launched ship_count)
                dist = torch.sqrt((src_x - tgt_x) ** 2 + (src_y - tgt_y) ** 2)
                speed = _ship_speed(ship_count).clamp(min=1e-6)
                eta = torch.ceil(dist / speed).clamp(min=1.0)           # (N, MAX_OWNED)
                # Projected defense = current garrison. Neutrals DON'T regrow (engine applies
                # production only to owner != -1), so there is NO production×ETA term — adding
                # one was phantom defense that vetoed ~all deterministic-decode launches.
                projected_defense = target_ships                       # (N, MAX_OWNED)
                # Enemy inbound: fleets (owner != us, owner >= 0) heading to the same
                # target, arriving before or at our ETA. Friendly inbound (our fleets
                # already en route) counts as added offense — subtract from the cost.
                # Resolved via the vectorized fleet-target resolver to match the real
                # swept-collision target (not just angle alignment).
                fi = self.fleets                                       # (N, F, 7)
                fa = self.fleet_alive                                  # (N, F)
                f_owner = fi[:, :, 1].long()
                f_ships = fi[:, :, 6] * fa.float()
                f_tgt = self._fleet_targets()                          # (N, F)
                f_fx, f_fy = fi[:, :, 2], fi[:, :, 3]
                f_speed = _ship_speed(fi[:, :, 6]).clamp(min=1e-6)
                # Per-(slot, fleet) ETA: distance from fleet to the slot's target / fleet speed
                # tgt per slot (N, MAX_OWNED); broadcast fleets (N, F) → (N, MAX_OWNED, F)
                tx = tgt_x.unsqueeze(2); ty = tgt_y.unsqueeze(2)       # (N, MAX_OWNED, 1)
                fxp = f_fx.unsqueeze(1); fyp = f_fy.unsqueeze(1)       # (N, 1, F)
                f_eta_to_tgt = (torch.sqrt((fxp - tx) ** 2 + (fyp - ty) ** 2)
                                / f_speed.unsqueeze(1)).clamp(max=100.0)  # (N, MAX_OWNED, F)
                hits_tgt = (f_tgt.unsqueeze(1) == target_idx.unsqueeze(2))  # (N, MAX_OWNED, F)
                arrives_before = f_eta_to_tgt <= eta.unsqueeze(2)      # (N, MAX_OWNED, F)
                # Enemy inbound (adds to defense); friendly inbound (adds to offense)
                enemy_mask = (f_owner.unsqueeze(1) != owner_id) & (f_owner.unsqueeze(1) >= 0)  # (N, 1, F)
                friendly_mask = (f_owner.unsqueeze(1) == owner_id)     # (N, 1, F)
                valid = hits_tgt & arrives_before & fa.unsqueeze(1)    # (N, MAX_OWNED, F)
                enemy_inbound = (f_ships.unsqueeze(1) * (valid & enemy_mask).float()).sum(dim=2)
                friendly_inbound = (f_ships.unsqueeze(1) * (valid & friendly_mask).float()).sum(dim=2)
                total_offense = ship_count + friendly_inbound
                # SUFFICIENT-COMMIT: veto if (ship_count + friendly_inbound) can't beat the floor.
                insufficient = is_neutral_attack & (total_offense <= projected_defense * self.sufficient_commit_factor)
                can_fire = can_fire & ~insufficient

        # Compute launch positions (just outside planet radius along angle)
        start_x = src_x + torch.cos(angle) * (src_r + 0.1)
        start_y = src_y + torch.sin(angle) * (src_r + 0.1)

        # Find first MAX_OWNED dead fleet slots per env via topk-smallest trick. Done BEFORE the
        # ship debit so a slot-starved launch doesn't burn ships (see target_valid below).
        dead = ~self.fleet_alive                                  # (N, F) bool
        F = dead.shape[1]
        SENTINEL_F = F + 1
        slot_grid = torch.arange(F, device=self.device).expand(N, F)
        slot_scores = torch.where(dead, slot_grid, torch.full_like(slot_grid, SENTINEL_F))
        rank_to_slot, _ = torch.topk(slot_scores, MAX_OWNED, dim=1, largest=False)  # (N, MAX_OWNED)
        rank_has = rank_to_slot < F
        rank_to_slot = rank_to_slot.clamp(max=F - 1)

        # fire_rank: among `can_fire` slots in an env, what's the rank of this slot?
        fire_rank = (can_fire.long().cumsum(dim=1) - 1).clamp(min=0)   # (N, MAX_OWNED)
        target_slot = rank_to_slot.gather(1, fire_rank)                # (N, MAX_OWNED)
        # target_valid = a launch that BOTH wants to fire AND got a free fleet slot. If fleet
        # storage is saturated (no free slot) the launch is DROPPED — not fired with ships burned.
        target_valid = rank_has.gather(1, fire_rank) & can_fire        # (N, MAX_OWNED)

        # Debit ships from source planets — ONLY for launches that actually created a fleet
        # (target_valid), so a slot-starved launch keeps its ships instead of burning them.
        # scatter_add with negative values; one slot per planet so the scatter is well-defined.
        debit = ship_count * target_valid.float()                 # (N, MAX_OWNED)
        ships_col = self.planets[:, :, 5]
        new_ships = ships_col.scatter_add(1, owned_idx, -debit)
        self.planets[:, :, 5] = new_ships

        # Launch diagnostics: emitted = fleets actually created (target_valid); slot-starved =
        # can_fire dropped because fleet storage was full. fleet_slot_saturation = slotstarve/can_fire
        # (can_fire = emitted ⊔ slotstarve). Together with _attempt_step (requested) this exposes the
        # requested→emitted gap. Split by episode window [<50/50-100/>100].
        if self._emitted_step is not None:
            _sc2 = self.step_count
            _v0 = (_sc2 < 50).float(); _v1 = ((_sc2 >= 50) & (_sc2 < 100)).float(); _v2 = (_sc2 >= 100).float()
            _em = target_valid.sum(dim=1).float()                          # (N,)
            _ss = (can_fire & ~target_valid).sum(dim=1).float()            # (N,)
            self._emitted_step[:, owner_id, 0] += _em * _v0
            self._emitted_step[:, owner_id, 1] += _em * _v1
            self._emitted_step[:, owner_id, 2] += _em * _v2
            self._slotstarve_step[:, owner_id, 0] += _ss * _v0
            self._slotstarve_step[:, owner_id, 1] += _ss * _v1
            self._slotstarve_step[:, owner_id, 2] += _ss * _v2

        # Per-env IDs for new fleets: next_fleet_id + 0, 1, 2, ... within env.
        new_id_within_env = (target_valid.long().cumsum(dim=1) - 1).clamp(min=0)
        new_ids = self.next_fleet_id.unsqueeze(1) + new_id_within_env  # (N, MAX_OWNED)

        # Maskless scatter: invalid launches write to scratch slot F, which is dropped.
        # Valid target slots are unique within each env by construction above.
        safe_slot = torch.where(target_valid, target_slot, torch.full_like(target_slot, F))
        fleet_pad = torch.cat([
            self.fleets,
            torch.zeros(N, 1, 7, dtype=self.fleets.dtype, device=self.device),
        ], dim=1)
        new_fleets = torch.stack([
            new_ids.float(),
            torch.full_like(start_x, float(owner_id)),
            start_x,
            start_y,
            angle,
            src[..., 0],
            ship_count,
        ], dim=2)
        fleet_pad = fleet_pad.scatter(
            1, safe_slot.unsqueeze(2).expand(-1, -1, 7), new_fleets)
        self.fleets = fleet_pad[:, :F]
        alive_pad = torch.cat([
            self.fleet_alive,
            torch.zeros(N, 1, dtype=torch.bool, device=self.device),
        ], dim=1)
        self.fleet_alive = alive_pad.scatter(
            1, safe_slot, torch.ones_like(safe_slot, dtype=torch.bool))[:, :F]
        self.next_fleet_id = self.next_fleet_id + target_valid.long().sum(dim=1)
        # Resolve the new fleets' targets ONCE, on the (N, MAX_OWNED) launch grid — the ray
        # and the planets' orbits are fixed at launch (see _fleet_targets). Uses the final
        # angle (post intercept/override) and final ship_count (speed), i.e. exactly what
        # was written into the fleet slots above.
        new_tgt = self._resolve_targets_at(start_x, start_y, angle, ship_count)  # (N, MAX_OWNED)
        target_pad = torch.cat([
            self.fleet_tgt,
            torch.full((N, 1), -1, dtype=self.fleet_tgt.dtype, device=self.device),
        ], dim=1)
        self.fleet_tgt = target_pad.scatter(1, safe_slot, new_tgt)[:, :F]

    def reset_reinforce_stats(self):
        """Zero the reinforce_rate accumulators. Call once per rollout (before the
        step loop); read _reinforce_launch_count / _fire_launch_count after, combine
        with train_mask to get the current policy's reinforce_rate."""
        self._reinforce_launch_count = torch.zeros(
            self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        self._fire_launch_count = torch.zeros(
            self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        self._neutral_launch_count = torch.zeros(
            self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        self._reinf_step = torch.zeros(
            self.num_envs, self.num_players, 3, dtype=torch.float32, device=self.device)
        self._fire_step = torch.zeros(
            self.num_envs, self.num_players, 3, dtype=torch.float32, device=self.device)
        self._overask_step = torch.zeros(
            self.num_envs, self.num_players, 3, dtype=torch.float32, device=self.device)
        self._attempt_step = torch.zeros(
            self.num_envs, self.num_players, 3, dtype=torch.float32, device=self.device)
        self._emitted_step = torch.zeros(
            self.num_envs, self.num_players, 3, dtype=torch.float32, device=self.device)
        self._slotstarve_step = torch.zeros(
            self.num_envs, self.num_players, 3, dtype=torch.float32, device=self.device)
        self._obs_trunc = torch.zeros(self.num_players, dtype=torch.float32, device=self.device)
        self._obs_calls = torch.zeros(self.num_players, dtype=torch.float32, device=self.device)
        self._obs_trunc_fleets = torch.zeros(self.num_players, dtype=torch.float32, device=self.device)
        self._obs_total_fleets = torch.zeros(self.num_players, dtype=torch.float32, device=self.device)
        self._obs_trunc_ships = torch.zeros(self.num_players, dtype=torch.float32, device=self.device)
        self._obs_total_ships = torch.zeros(self.num_players, dtype=torch.float32, device=self.device)
        self._obs_trunc_enemy_ships = torch.zeros(self.num_players, dtype=torch.float32, device=self.device)
        self._obs_total_enemy_ships = torch.zeros(self.num_players, dtype=torch.float32, device=self.device)
        self._staging_phi_acc = 0.0
        self._staging_phi_n = 0

    # ---------------------------------------------------------------------
    # Step — pure tensor ops, runs all N envs in one pass.
    def step(self, actions=None, angle_overrides=None) -> dict:
        """Advance all N envs by one tick.

        actions: optional dict {player_id: (N, MAX_OWNED, 3) tensor}.
                 Each player's fleets are launched before physics.
        angle_overrides: optional dict {player_id: (N, MAX_OWNED) float tensor};
                 NaN = no override, else a continuous launch angle that bypasses the
                 144-bin quantization (used for external heuristics — see _apply_actions).
        """
        # Per-step buffer for the reinforcement transit cost (#2): ships each player
        # sent to its own planets this step. Zeroed before launches accumulate into it.
        if self.allow_reinforce and self.reinforce_cost > 0.0:
            self._reinforce_ships = torch.zeros(
                self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        if actions is not None:
            for pid, act in actions.items():
                ovr = angle_overrides.get(pid) if angle_overrides else None
                self._apply_actions(act, pid, angle_override=ovr)

        # 0b. Comets: lazily compute any spawn reached this step, then activate this tick's
        # comet group (owner=-1, ships, prod) + set alive, BEFORE production (a neutral comet
        # doesn't produce; a captured one does).
        if self._has_comets:
            self._lazy_comets()
            comet_t, comet_bidx = self._apply_comet_state()

        # 1. Production: planet[5] += planet[6] for owned planets (owner != -1)
        owner = self.planets[:, :, 1]
        prod  = self.planets[:, :, 6]
        is_owned = (owner != -1) & self.planet_alive
        self.planets[:, :, 5] = self.planets[:, :, 5] + prod * is_owned.float()

        # 2. Planet path: compute (old_pos, new_pos) for swept collision.
        step_f = self.step_count.float().unsqueeze(-1)   # (N, 1)
        ang_vel = self.angular_velocity.unsqueeze(-1)    # (N, 1)
        cur_angle = self._planet_initial_angle + ang_vel * step_f   # (N, P)

        planet_old_x = self.planets[:, :, 2].clone()
        planet_old_y = self.planets[:, :, 3].clone()
        planet_new_x = CENTER + self._planet_orbital_r * torch.cos(cur_angle)
        planet_new_y = CENTER + self._planet_orbital_r * torch.sin(cur_angle)
        is_orb = self._planet_is_orbiting
        planet_new_x = torch.where(is_orb, planet_new_x, planet_old_x)
        planet_new_y = torch.where(is_orb, planet_new_y, planet_old_y)

        # 2b. Comets follow their precomputed path (not orbital motion): new pos = path[t].
        # old pos stays the current position so the swept-pair check uses the real chord.
        if self._has_comets:
            c0 = COMET_SLOT_START
            cxy = self._comet_xy[comet_bidx, comet_t]                  # (N, 4, 2)
            planet_new_x[:, c0:c0 + N_COMET_SLOTS] = cxy[:, :, 0]
            planet_new_y[:, c0:c0 + N_COMET_SLOTS] = cxy[:, :, 1]

        # 3. Fleet movement
        fleet_angle = self.fleets[:, :, 4]
        fleet_ships = self.fleets[:, :, 6]
        speed = _ship_speed(fleet_ships)
        speed = speed * self.fleet_alive.float()
        fleet_old_x = self.fleets[:, :, 2].clone()
        fleet_old_y = self.fleets[:, :, 3].clone()
        fleet_new_x = fleet_old_x + torch.cos(fleet_angle) * speed
        fleet_new_y = fleet_old_y + torch.sin(fleet_angle) * speed
        self.fleets[:, :, 2] = fleet_new_x
        self.fleets[:, :, 3] = fleet_new_y

        # 4. Swept-pair collision detection: (fleet old→new) vs (planet old→new).
        # Quadratic form: ||A + t·(B-A) - (P0 + t·(P1-P0))||^2 = r^2
        # Coefficients:
        #     a = ||dv||^2       where dv = (B-A) - (P1-P0)
        #     b = 2·(d0 · dv)    where d0 = A - P0
        #     c = ||d0||^2 - r^2
        # Hit iff disc >= 0 AND any t in [0,1] satisfies the quadratic <= 0,
        # i.e. t1 <= 1 and t2 >= 0 where t1,t2 = (-b ± sqrt(disc)) / (2a).
        # Degenerate case a≈0: hit iff c <= 0 (already overlapping).
        N, F, _ = self.fleets.shape
        _, P, _ = self.planets.shape

        # Reshape for broadcast: fleet (N, F, 1), planet (N, 1, P)
        fx0 = fleet_old_x.unsqueeze(2);  fy0 = fleet_old_y.unsqueeze(2)
        fx1 = fleet_new_x.unsqueeze(2);  fy1 = fleet_new_y.unsqueeze(2)
        px0 = planet_old_x.unsqueeze(1); py0 = planet_old_y.unsqueeze(1)
        px1 = planet_new_x.unsqueeze(1); py1 = planet_new_y.unsqueeze(1)
        pr  = self.planets[:, :, 4].unsqueeze(1)  # (N, 1, P)

        d0x = fx0 - px0;       d0y = fy0 - py0
        dvx = (fx1 - fx0) - (px1 - px0)
        dvy = (fy1 - fy0) - (py1 - py0)
        a = dvx * dvx + dvy * dvy
        b = 2.0 * (d0x * dvx + d0y * dvy)
        c = d0x * d0x + d0y * d0y - pr * pr

        # Numerically safe disc (clamp negative to 0 so sqrt is finite — these
        # cases get masked out by `disc_ok` below).
        disc = b * b - 4.0 * a * c
        disc_ok = disc >= 0
        sq = torch.sqrt(torch.clamp(disc, min=0.0))
        safe_a = torch.where(a > 1e-12, a, torch.ones_like(a))
        t1 = (-b - sq) / (2.0 * safe_a)
        t2 = (-b + sq) / (2.0 * safe_a)
        hit_moving = disc_ok & (t2 >= 0.0) & (t1 <= 1.0)
        hit_degenerate = (a < 1e-12) & (c <= 0.0)
        hit = (hit_moving & (a >= 1e-12)) | hit_degenerate  # (N, F, P)

        # Comets are not collision-tested on their first (off-board) placement tick.
        if self._has_comets:
            c0 = COMET_SLOT_START
            chk = self._comet_check[comet_bidx, comet_t]              # (N, 4) bool
            hit[:, :, c0:c0 + N_COMET_SLOTS] &= chk.unsqueeze(1)

        # Mask out dead fleets / dead planets
        valid = self.fleet_alive.unsqueeze(2) & self.planet_alive.unsqueeze(1)
        hit = hit & valid                                              # (N, F, P)

        # Each fleet hits at most one planet — pick first true index per (N,F).
        # If no hit, hit_planet_idx will be 0 but hit_any will be False.
        hit_any = hit.any(dim=2)                                       # (N, F)
        # Use argmax on bool-as-int to find first true index along P axis
        hit_planet_idx = hit.float().argmax(dim=2)                     # (N, F)
        # Diagnostic stash: the TRUE swept-collision outcome this tick (ground truth for
        # resolver-accuracy measurements; see gpu_run_artifacts/envperf/).
        self._last_hit_any = hit_any
        self._last_hit_idx = hit_planet_idx

        # 5. Sun-crossing: fleet path crosses sun (point-to-segment distance <= SUN_RADIUS).
        # Project (CENTER, CENTER) onto segment (fleet_old → fleet_new) and check distance.
        seg_dx = fleet_new_x - fleet_old_x
        seg_dy = fleet_new_y - fleet_old_y
        seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
        seg_len2_safe = torch.where(seg_len2 > 0, seg_len2, torch.ones_like(seg_len2))
        t = ((CENTER - fleet_old_x) * seg_dx + (CENTER - fleet_old_y) * seg_dy) / seg_len2_safe
        t = torch.clamp(t, 0.0, 1.0)
        proj_x = fleet_old_x + t * seg_dx
        proj_y = fleet_old_y + t * seg_dy
        sun_dist = torch.sqrt((proj_x - CENTER) ** 2 + (proj_y - CENTER) ** 2)
        # Zero-length segments (dead fleets that didn't move): set sun_dist large
        sun_dist = torch.where(seg_len2 > 0, sun_dist, torch.full_like(sun_dist, 1000.0))
        crosses_sun = sun_dist < SUN_RADIUS

        # 6. Out-of-bounds removal
        in_bounds = (
            (fleet_new_x >= 0) & (fleet_new_x <= BOARD_SIZE)
            & (fleet_new_y >= 0) & (fleet_new_y <= BOARD_SIZE)
        )

        # Order of removal in kaggle env: planet-hit takes priority, then OOB, then sun.
        # Combat list contains only fleets that hit a planet (NOT OOB or sun-killed).
        combat_mask = hit_any & self.fleet_alive  # (N, F)
        # Fleets that survive this tick: alive, in bounds, no sun, no planet hit
        survives = (
            self.fleet_alive
            & ~combat_mask
            & in_bounds
            & ~crosses_sun
        )

        # 7. Combat resolution.
        # For each (env, planet), sum ships per owner using scatter_add.
        # Build (N, P, num_players) tensor of ship contributions.
        attacker_ships = torch.zeros(N, P, self.num_players, device=self.device)
        # Fleet contribution: scatter add fleet.ships into attacker_ships[env, hit_planet, owner]
        fleet_owner_long = self.fleets[:, :, 1].long()                # (N, F)
        # Clamp owner to [0, num_players) for safety (dead fleets may have 0)
        fleet_owner_long = torch.clamp(fleet_owner_long, 0, self.num_players - 1)
        ships_contrib = self.fleets[:, :, 6] * combat_mask.float()    # (N, F)

        # Flatten (N, F) → indices into (N*P*num_players) for scatter_add
        env_idx = torch.arange(N, device=self.device).unsqueeze(1).expand(N, F)
        flat_idx = (env_idx * P + hit_planet_idx) * self.num_players + fleet_owner_long
        attacker_ships = attacker_ships.view(-1).scatter_add(
            0, flat_idx.reshape(-1),
            ships_contrib.reshape(-1),
        ).view(N, P, self.num_players)

        # Top owner per (N, P): top_ships, top_owner; second_ships
        top_ships, top_owner = attacker_ships.max(dim=2)               # (N, P)
        # Zero out the top to find second
        attacker_minus_top = attacker_ships.clone()
        attacker_minus_top.scatter_(2, top_owner.unsqueeze(2), 0.0)
        second_ships = attacker_minus_top.max(dim=2).values            # (N, P)

        tie = (top_ships == second_ships) & (top_ships > 0)
        survivor_ships = torch.where(tie, torch.zeros_like(top_ships), top_ships - second_ships)
        any_combat = top_ships > 0

        # Apply to planet ownership
        planet_owner = self.planets[:, :, 1]
        planet_ships = self.planets[:, :, 5]

        # If survivor_owner == planet_owner: reinforce (planet.ships += survivor)
        # Else: attack (planet.ships -= survivor; if < 0, flip ownership to abs)
        same_owner = (top_owner.float() == planet_owner) & any_combat & ~tie
        diff_owner = (top_owner.float() != planet_owner) & any_combat & ~tie

        new_ships = torch.where(same_owner, planet_ships + survivor_ships, planet_ships)
        # Attack case
        new_ships_attack = planet_ships - survivor_ships
        flipped = new_ships_attack < 0
        do_flip = diff_owner & flipped
        new_ships = torch.where(diff_owner, new_ships_attack.abs(), new_ships)
        new_owner = torch.where(do_flip, top_owner.float(), planet_owner)

        # Only update alive planets
        update_mask = self.planet_alive & any_combat
        self.planets[:, :, 5] = torch.where(update_mask, new_ships, planet_ships)
        self.planets[:, :, 1] = torch.where(update_mask, new_owner, planet_owner)

        # 8. Apply planet new positions (after collision detection used old pos)
        self.planets[:, :, 2] = planet_new_x
        self.planets[:, :, 3] = planet_new_y

        # 9. Update fleet alive flags
        self.fleet_alive = survives

        # 10. Advance step
        self.step_count = self.step_count + 1

        # 11. Termination + reward (terminal_rewards is non-zero only for newly-done envs)
        terminal_rewards, done = self._check_done()
        # Expansion shaping: potential-based reward on the change in owned-production
        # lead. Dense per-step signal for winning the planet/economy race (the thing
        # that decides snowball games). Telescopes, so passive play nets ~0.
        if self.expansion_coef != 0.0:
            production = self._compute_production()
            prod_delta = production - self.prev_production
            if self.num_players == 2:
                d = prod_delta[:, 0] - prod_delta[:, 1]
                expansion_rewards = torch.stack([d, -d], dim=1)
            else:
                others = prod_delta.sum(dim=1, keepdim=True) - prod_delta
                expansion_rewards = prod_delta - others / max(self.num_players - 1, 1)
            terminal_rewards = terminal_rewards + self.expansion_coef * expansion_rewards
        # Delta-capture shaping: time-decayed reward for CAPTURING planets (delta in owned
        # count), NOT for holding them. Fires as a spike when a planet changes hands.
        # Symmetric delta + exponential decay with a permanent floor:
        #   1. SYMMETRIC: planet_delta tracks both gains (+) and losses (-), clamped to [-1, 1].
        #      Losing a planet now costs as much as gaining one. This eliminates the "planet
        #      tennis" arbitrage in self-play where both agents trade planets for free reward.
        #      With losses penalised, trading is net-zero → farming is structurally impossible.
        #
        #   2. EXPONENTIAL DECAY + FLOOR: replaces the hard linear cliff at step 400.
        #      decay = exp(-2.5 × t/500) + 0.10 → stabilises at ~10% of initial coeff.
        #      At step 450: ~12% of coeff survives. Keeps a navigational beacon alive on
        #      rotating boards (e.g. seed6462) when orbital alignment opens at step 430.
        #      Never hits absolute zero, so the late-game gradient desert is eliminated.
        #
        #   early_capture_steps parameter is now unused (kept for CLI compat), decay runs
        #   to episode end.
        if self.early_capture_coef != 0.0:
            ec_owner = self.planets[:, :, 1].long()  # (N, P)
            t = self.step_count.float()  # (N,)
            # Exponential decay with 10% permanent floor — never hits zero
            decay = torch.exp(-2.5 * t / self.episode_steps) + 0.10  # (N,)
            owned = torch.zeros(self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
            for pl in range(self.num_players):
                owned[:, pl] = ((ec_owner == pl) & self.planet_alive).float().sum(dim=1)
            # Symmetric delta: gains positive, losses negative. Capped at ±1/step.
            planet_delta = (owned - self.prev_owned).clamp(min=-1.0, max=1.0)
            # Each player's reward = their net delta (no additional zero-sum netting;
            # symmetric delta is already self-correcting: capturing from opponent gives
            # +1 to attacker and -1 to defender automatically).
            ec_rewards = planet_delta
            # First Strike bonus: multiply capture reward by first_strike_mult for t < first_strike_steps.
            # Overcomes value critic's "home invasion fear" — makes early captures so lucrative
            # that the policy fires at step 0 instead of waiting.
            if self.first_strike_steps > 0:
                # Linear decay from first_strike_mult at t=0 to 1.0 at t=first_strike_steps.
                frac = (t.float() / self.first_strike_steps).clamp(max=1.0)
                fs_mult = 1.0 + (self.first_strike_mult - 1.0) * (1.0 - frac)  # (N,)
                effective_coef = self.early_capture_coef * decay * fs_mult  # (N,)
            else:
                effective_coef = self.early_capture_coef * decay  # (N,)
            terminal_rewards = terminal_rewards + effective_coef.unsqueeze(1) * ec_rewards
            self.prev_owned = owned
        if self.staging_shaping_coef != 0.0:
            # PBRS: r += coef·(γΦ(s') − Φ(s)). Φ(s') from the post-step pre-reset state.
            # Terminal Φ(s')=0 (absorbing) on done envs so the potential collapses cleanly.
            phi_now = self._staging_potential(self._decisive_mass_fields())   # (N, players)
            gamma_phi = torch.where(done.unsqueeze(1),
                                    torch.zeros_like(phi_now),
                                    self.staging_gamma * phi_now)
            terminal_rewards = terminal_rewards + self.staging_shaping_coef * (
                gamma_phi - self.prev_staging_phi)
            self.prev_staging_phi = phi_now                      # done envs fixed post-reset
            self._staging_phi_acc += float(phi_now.mean().item())
            self._staging_phi_n += 1
        # Reinforcement transit cost: price the ships each player sent to its own
        # planets this step, pruning the wasteful tail. Calibrate reinforce_cost so
        # reinforcement is neither suppressed nor allowed to flood.
        if self.allow_reinforce and self.reinforce_cost > 0.0:
            terminal_rewards = terminal_rewards - self.reinforce_cost * self._reinforce_ships
        # 12. Auto-reset done envs in-place — must come AFTER capturing rewards
        if done.any():
            self._auto_reset(done)
        # Refresh prev_production / prev_owned AFTER auto-reset so done envs telescope
        # from their fresh post-reset state (avoids spurious spike on episode boundary).
        if self.expansion_coef != 0.0:
            self.prev_production = self._compute_production()
        if self.early_capture_coef != 0.0 and done.any():
            ec_owner = self.planets[:, :, 1].long()
            for pl in range(self.num_players):
                self.prev_owned[done, pl] = ((ec_owner[done] == pl) & self.planet_alive[done]).float().sum(dim=1)
        # PBRS staging: fresh post-reset boards have no inflight → Φ=0; reset prev so the next step's
        # γΦ(s')−Φ(s) telescopes from 0 (no spurious spike across the episode boundary).
        if self.staging_shaping_coef != 0.0 and done.any():
            self.prev_staging_phi[done] = 0.0
        # Reverse-edge cooldown: clear the per-edge history for fresh games (else a prior game's
        # reinforce edges mis-block the new board — same boundary bug class as decmass re-arm).
        if self.reverse_edge_cooldown > 0 and self.reinf_cd is not None and done.any():
            self.reinf_cd[done] = _REINF_CD_NEVER
        # Periodic refresh (staleness bound, see fleet_target_refresh_every): every K
        # ticks re-resolve ALL fleet targets from the post-step state, UNCONDITIONALLY —
        # the cadence is known CPU-side, so the hot loop stays free of data-dependent
        # syncs (a mark-unresolved variant paid a pipeline flush inside get_features on
        # every refresh tick, costing more than it saved). Comet spawn/expiry staleness
        # is bounded by the same cadence (the K-accuracy sweep ran with comets on).
        self._tick_counter += 1
        if (self.fleet_target_refresh_every > 0
                and self._tick_counter % self.fleet_target_refresh_every == 0):
            self.fleet_tgt = self._fleet_target_idx()
        return self._state_dict(), terminal_rewards, done

    # ---------------------------------------------------------------------
    # Termination logic — matches fast_env._maybe_terminate
    # Episode ends if step_count >= episode_steps - 1 OR <= 1 player alive
    # (alive = has any owned planet or any in-flight fleet).
    # Reward: per-env, per-player. +1 if that player has the max score AND
    # max > 0, else -1.
    # ---------------------------------------------------------------------

    @torch._dynamo.disable
    def _check_done(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (rewards: (N, num_players), done: (N,) bool).

        rewards are 0 for envs not done, ±1 for envs that just terminated.
        """
        N = self.num_envs
        P_ = self.num_players

        # Per-player alive check: any planet owned OR any fleet owned
        owner_p = self.planets[:, :, 1].long()         # (N, P)
        owner_f = self.fleets[:, :, 1].long()          # (N, F)
        # Build (N, num_players) "has any" masks
        alive_mask = torch.zeros(N, P_, dtype=torch.bool, device=self.device)
        for pl in range(P_):
            has_planet = ((owner_p == pl) & self.planet_alive).any(dim=1)
            has_fleet  = ((owner_f == pl) & self.fleet_alive).any(dim=1)
            alive_mask[:, pl] = has_planet | has_fleet
        n_alive = alive_mask.long().sum(dim=1)         # (N,)

        time_up = self.step_count >= (self.episode_steps - 1)
        few_left = n_alive <= 1
        newly_done = (time_up | few_left) & ~self.done

        # Scores: ships on owned planets + ships in fleets, per player.
        scores = torch.zeros(N, P_, dtype=torch.float32, device=self.device)
        ships_p = self.planets[:, :, 5] * self.planet_alive.float()
        ships_f = self.fleets[:, :, 6] * self.fleet_alive.float()
        for pl in range(P_):
            sp = (owner_p == pl).float() * ships_p
            sf = (owner_f == pl).float() * ships_f
            scores[:, pl] = sp.sum(dim=1) + sf.sum(dim=1)
        max_score, _ = scores.max(dim=1, keepdim=True)  # (N, 1)
        wins = (scores == max_score) & (max_score > 0)
        # Expose the RAW winner mask (pre-shaping/bonus) so callers (PFSP result
        # attribution) don't have to infer win/loss from the shaped reward tensor —
        # which already carries expansion/early-capture/etc. shaping by the
        # time step() returns it. Valid for envs that are newly-done THIS step.
        self._last_wins = wins
        rewards = torch.where(wins, torch.ones_like(scores), -torch.ones_like(scores))
        # Optional win-margin bonus: winner gets +α*(my_score/total_score).
        # Losers stay at -1; coefficient 0 = pure ±1 (default).
        if self.win_margin_coeff != 0.0:
            total_score = scores.sum(dim=1, keepdim=True).clamp(min=1.0)
            margin = scores / total_score          # (N, P) fraction in [0, 1]
            bonus = self.win_margin_coeff * margin
            rewards = torch.where(wins, rewards + bonus, rewards)
        # Only return rewards for newly-done envs; zero otherwise
        rewards = rewards * newly_done.unsqueeze(1).float()
        self.rewards = torch.where(newly_done.unsqueeze(1), rewards, self.rewards)
        self.done = self.done | newly_done
        return rewards, newly_done

    # ---------------------------------------------------------------------
    # Auto-reset: pick new seeds for done envs and regenerate their state.
    # Keeps the running training loop simple — no need for caller-side resets.
    # ---------------------------------------------------------------------

    @torch._dynamo.disable
    def _auto_reset(self, done_mask: torch.Tensor):
        """Re-generate state for envs where done_mask is True (Python-serial: .cpu().tolist()
        + per-env numpy rebuild → @torch._dynamo.disable so --compile-env graph-breaks here
        cleanly instead of hard-erroring while tracing it)."""
        from kaggle_environments.envs.orbit_wars.orbit_wars import generate_planets

        done_idx = torch.where(done_mask)[0].cpu().tolist()
        for env_i in done_idx:
            seed = random.randint(0, 2**31)
            self.seeds[env_i] = seed
            init_rng = random.Random(seed)
            ang_vel = init_rng.uniform(0.025, 0.05)
            raw_planets = generate_planets(init_rng)
            n = len(raw_planets)
            pad = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
            for i, p in enumerate(raw_planets):
                pad[i] = p
            alive = np.zeros(MAX_PLANETS, dtype=bool)
            alive[:n] = True
            num_groups = n // 4
            if num_groups > 0:
                home_group = init_rng.randint(0, num_groups - 1)
                base = home_group * 4
                if self.num_players == 2:
                    pad[base, 1] = 0;     pad[base, 5] = 10
                    pad[base + 3, 1] = 1; pad[base + 3, 5] = 10
                elif self.num_players == 4:
                    for j in range(4):
                        pad[base + j, 1] = j; pad[base + j, 5] = 10
            pad[n:, 1] = -1

            self.planets[env_i] = torch.from_numpy(pad).to(self.device)
            self.init_planets[env_i] = self.planets[env_i].clone()
            self.planet_alive[env_i] = torch.from_numpy(alive).to(self.device)
            self.fleets[env_i] = 0
            self.fleet_alive[env_i] = False
            self.step_count[env_i] = 0
            self.angular_velocity[env_i] = ang_vel
            self.next_fleet_id[env_i] = 0
            self.done[env_i] = False
            self.rewards[env_i] = 0.0
        # Re-precompute orbital params for changed envs
        self._precompute_orbital_params()
        # Reset the comet schedule for the reset envs (new seeds → recomputed lazily).
        if done_idx:
            self._init_comets(done_idx)
            self.fleet_tgt[torch.tensor(done_idx, device=self.device)] = -1


# -------------------------------------------------------------------------
# Helper: extract a single-env dict in fast_env format (for parity testing)
# -------------------------------------------------------------------------

def to_legacy_obs(env: VecTorchEnv, env_idx: int = 0, player: int = 0) -> dict:
    """Convert one env's state to the fast_env obs dict format."""
    p = env.planets[env_idx].cpu().numpy()
    a = env.planet_alive[env_idx].cpu().numpy()
    f = env.fleets[env_idx].cpu().numpy()
    fa = env.fleet_alive[env_idx].cpu().numpy()

    planets = []
    for i in range(MAX_PLANETS):
        if a[i]:
            planets.append([
                int(p[i, 0]), int(p[i, 1]),
                float(p[i, 2]), float(p[i, 3]), float(p[i, 4]),
                float(p[i, 5]), float(p[i, 6]),
            ])
    fleets = []
    for i in range(MAX_FLEETS):
        if fa[i]:
            fleets.append([
                int(f[i, 0]), int(f[i, 1]),
                float(f[i, 2]), float(f[i, 3]), float(f[i, 4]),
                int(f[i, 5]), float(f[i, 6]),
            ])
    ip = env.init_planets[env_idx].cpu().numpy()
    # Exclude comet slots (>= COMET_SLOT_START): their init_planets entry is unpopulated
    # (id 0, pos 0,0) and would collide with real planet id 0 in init_by_id, corrupting that
    # planet's orbit features. Comets carry no meaningful initial-board entry (they spawn
    # mid-game and get path-aware features instead).
    initial_planets = [
        [int(ip[i, 0]), int(ip[i, 1]),
         float(ip[i, 2]), float(ip[i, 3]), float(ip[i, 4]),
         float(ip[i, 5]), float(ip[i, 6])]
        for i in range(COMET_SLOT_START) if a[i]
    ]
    # Comets: surface the live comet group (ids + full per-comet paths + path_index) so the
    # kaggle-side feature path (features.extract_features) sees them exactly as the real env's
    # obs does. Reconstructed from the byte-faithful comet schedule buffers.
    comet_planet_ids: list[int] = []
    comets: list[dict] = []
    if getattr(env, "_has_comets", False):
        c0 = COMET_SLOT_START
        ca = env._comet_alive[env_idx].cpu().numpy()        # (T1, ns)
        cxy = env._comet_xy[env_idx].cpu().numpy()           # (T1, ns, 2)
        cids = env._comet_ids[env_idx].cpu().numpy()         # (ns,)
        cur = int(env.step_count[env_idx].item())
        planet_ids: list[int] = []
        paths: list[list] = []
        path_index = -1
        for c in range(N_COMET_SLOTS):
            if not bool(a[c0 + c]):
                continue  # comet slot not currently alive (no live comet there)
            alive_steps = [t for t in range(ca.shape[0]) if ca[t, c]]
            if not alive_steps:
                continue
            first = alive_steps[0]
            full_path = [[float(cxy[t, c, 0]), float(cxy[t, c, 1])]
                         for t in range(first, alive_steps[-1] + 1)]
            planet_ids.append(int(cids[c]))
            paths.append(full_path)
            path_index = cur - first
        if planet_ids:
            comets.append({"planet_ids": planet_ids, "paths": paths, "path_index": path_index})
            comet_planet_ids = list(planet_ids)
    return {
        "step": int(env.step_count[env_idx].item()),
        "player": player,
        "planets": planets,
        "fleets": fleets,
        "angular_velocity": float(env.angular_velocity[env_idx].item()),
        "initial_planets": initial_planets,
        "comet_planet_ids": comet_planet_ids,
        "comets": comets,
    }


def to_legacy_obs_batch(env: VecTorchEnv, env_ids, player: int = 0) -> list[dict]:
    """Batched `to_legacy_obs` for a group of envs (1-D LongTensor or list of ids).

    Produces the SAME per-env obs dicts as ``[to_legacy_obs(env, e, player) for e in ids]``
    but with one GPU-to-CPU copy per field over all ids. Pure transport: identical bytes out
    (verified by tests/test_to_legacy_obs_batch.py)."""
    # Keep the ids ON-DEVICE for indexing — a .tolist() here (then re-tensoring) would force
    # an extra GPU->CPU->GPU round-trip per external group; the only sync we want is the
    # batched .cpu() copies below.
    if torch.is_tensor(env_ids):
        idx = env_ids.to(device=env.device, dtype=torch.long)
    else:
        idx = torch.as_tensor(list(env_ids), dtype=torch.long, device=env.device)
    n = int(idx.shape[0])
    # One batched copy per field (vs per-env in to_legacy_obs).
    P = env.planets[idx].cpu().numpy()              # (n, MAX_PLANETS, 7)
    A = env.planet_alive[idx].cpu().numpy()         # (n, MAX_PLANETS)
    F = env.fleets[idx].cpu().numpy()               # (n, MAX_FLEETS, 7)
    FA = env.fleet_alive[idx].cpu().numpy()         # (n, MAX_FLEETS)
    IP = env.init_planets[idx].cpu().numpy()        # (n, MAX_PLANETS, 7)
    STEP = env.step_count[idx].cpu().numpy()        # (n,)
    ANG = env.angular_velocity[idx].cpu().numpy()   # (n,)
    has_comets = getattr(env, "_has_comets", False)
    if has_comets:
        CA = env._comet_alive[idx].cpu().numpy()    # (n, T1, ns)
        CXY = env._comet_xy[idx].cpu().numpy()      # (n, T1, ns, 2)
        CIDS = env._comet_ids[idx].cpu().numpy()    # (n, ns)

    out = []
    for j in range(n):
        p, a, f, fa, ip = P[j], A[j], F[j], FA[j], IP[j]
        planets = [
            [int(p[i, 0]), int(p[i, 1]), float(p[i, 2]), float(p[i, 3]),
             float(p[i, 4]), float(p[i, 5]), float(p[i, 6])]
            for i in range(MAX_PLANETS) if a[i]
        ]
        fleets = [
            [int(f[i, 0]), int(f[i, 1]), float(f[i, 2]), float(f[i, 3]),
             float(f[i, 4]), int(f[i, 5]), float(f[i, 6])]
            for i in range(MAX_FLEETS) if fa[i]
        ]
        initial_planets = [
            [int(ip[i, 0]), int(ip[i, 1]), float(ip[i, 2]), float(ip[i, 3]),
             float(ip[i, 4]), float(ip[i, 5]), float(ip[i, 6])]
            for i in range(COMET_SLOT_START) if a[i]
        ]
        comet_planet_ids: list[int] = []
        comets: list[dict] = []
        if has_comets:
            ca, cxy, cids = CA[j], CXY[j], CIDS[j]
            cur = int(STEP[j])
            planet_ids: list[int] = []
            paths: list[list] = []
            path_index = -1
            for c in range(N_COMET_SLOTS):
                if not bool(a[COMET_SLOT_START + c]):
                    continue
                alive_steps = [t for t in range(ca.shape[0]) if ca[t, c]]
                if not alive_steps:
                    continue
                first = alive_steps[0]
                full_path = [[float(cxy[t, c, 0]), float(cxy[t, c, 1])]
                             for t in range(first, alive_steps[-1] + 1)]
                planet_ids.append(int(cids[c]))
                paths.append(full_path)
                path_index = cur - first
            if planet_ids:
                comets.append({"planet_ids": planet_ids, "paths": paths, "path_index": path_index})
                comet_planet_ids = list(planet_ids)
        out.append({
            "step": int(STEP[j]),
            "player": player,
            "planets": planets,
            "fleets": fleets,
            "angular_velocity": float(ANG[j]),
            "initial_planets": initial_planets,
            "comet_planet_ids": comet_planet_ids,
            "comets": comets,
        })
    return out
