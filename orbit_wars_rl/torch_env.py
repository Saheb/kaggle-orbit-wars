"""Vectorized Orbit Wars environment in PyTorch.

Runs N games in parallel as batched tensor operations. Designed for self-play
training at 5,000+ SPS — replaces the per-env Python loop in fast_env.py.

PHASE 0 (this file): planet orbital motion + fleet movement (no combat yet).
Validates the tensor-physics approach against fast_env via parity test.

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

_SCENARIO_OFF = 0
_SCENARIO_AGG_ATTACK = 1
_SCENARIO_STAGE_ATTACK = 2
_SCENARIO_HOLD_UNDER_PEEL = 3
_SCENARIO_NAME_TO_ID = {
    "off": _SCENARIO_OFF,
    "agg_attack": _SCENARIO_AGG_ATTACK,
    "stage_attack": _SCENARIO_STAGE_ATTACK,
    "hold_under_peel": _SCENARIO_HOLD_UNDER_PEEL,
}
_SCENARIO_MIXED = "mixed"

# Tensor sizes (worst-case bounds from kaggle env)
MAX_PLANETS = 48
MAX_FLEETS = 256
MAX_OWNED = 16

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
NUM_ANGLE_BINS = 144
ANGLE_BIN_WIDTH = 2 * math.pi / NUM_ANGLE_BINS
NUM_SHIP_BINS = 32
SHIP_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 26, 30, 35, 42, 50, 60, 72, 86, 102, 122, 145, 173, 206, 245, 290, 350, 420]
# Fraction-mode decode (10 bins): bin i → (i+1)/10 * src_ships
FRACTION_BIN_VALUES = [(i + 1) / 10 for i in range(10)]


_COMET_T_ARR = None  # cached dense-sample parameter grid (t), built on first use


def _comet_paths_fast(initial_planets, angular_velocity, spawn_step,
                      comet_planet_ids=None, comet_speed=4.0, rng=None):
    """Vectorized drop-in for kaggle's generate_comet_paths — byte-identical output, ~30x faster.
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


class VecTorchEnv:
    """Vectorized Orbit Wars environment running N games in parallel."""

    def __init__(
        self,
        num_envs: int,
        num_players: int = 2,
        device: str | torch.device = "cpu",
        episode_steps: int = 500,
        ship_bin_mode: str = "absolute",
        ship_overflow_mode: str = "clamp",   # matches eval (_ship_bin_to_count clamps); "drop"=legacy bug
        action_decode: str = "angle",
        win_margin_coeff: float = 0.0,
        shaping_coef: float = 0.0,
        expansion_coef: float = 0.0,
        defense_coef: float = 0.0,
        early_capture_coef: float = 0.0,
        prod_share_coef: float = 0.0,
        early_capture_steps: int = 100,
        first_strike_steps: int = 0,
        first_strike_mult: float = 2.0,
        speed_coef: float = 0.0,
        consolidation_coef: float = 0.0,
        consolidation_steps: int = 40,
        capture_utility_coef: float = 0.0,
        capture_utility_window: int = 30,
        capture_idle_penalty: float = 0.0,
        decisive_mass_coef: float = 0.0,
        decisive_mass_beta: float = _DM_BETA,
        decisive_diag: bool = False,
        handicap_frac: float = 0.0,
        handicap_ships: int = 5,
        ssdr_frac: float = 0.0,
        ssdr_max_steps: int = 20,
        neutral_garrison_scale: float = 1.0,
        scenario_curriculum: str = "off",
        scenario_fraction: float = 0.0,
        scenario_deadline: int = 20,
        allow_reinforce: bool = False,
        reinforce_garrison_floor: float = 0.0,
        reinforce_cost: float = 0.0,
        reinforce_gate_min_planets: int = 0,
        reinforce_forward_only: bool = False,
        reverse_edge_cooldown: int = 0,
        sufficient_commit_factor: float = 0.0,
        game_phase_features: bool = False,
    ):
        self.num_envs = num_envs
        # Game-phase observation channels (Stage B): append 4 globals (dim 11->15). Must match
        # features.game_phase_channels (parity test feature_parity_gamephase_probe.py).
        self.game_phase_features = bool(game_phase_features)
        # Reinforcement: when True, own planets (except the launch source) are LEGAL
        # targets — ships arriving at a friendly planet add to its garrison (physics
        # already implemented in step()). EDA of top players: ~57% of fleets reinforce;
        # beginner agents 0%. Default False = attack-only (backward-compatible).
        self.allow_reinforce = bool(allow_reinforce)
        # Reinforcement discipline (rev56 lesson: costless reinforcement floods — the
        # curriculum times availability but adds no cost, so any fire incentive → flood).
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
        #      Grounded in top-player replays: reinforce_rate ≈0 at 1 planet, ~0.1 at 2,
        #      then ramps with empire size. A pure action mask (no Nash risk) that makes
        #      the early flood impossible by construction. 0 = off (no gate). Training-only,
        #      like the garrison floor — the policy internalises it.
        self.reinforce_gate_min_planets = int(reinforce_gate_min_planets)
        #   #4 FORWARD-STAGING GATE: an own (reinforce) target is legal only if it sits
        #      closer to the nearest enemy planet than the launch source — reinforcement
        #      flows rear→front (staging), never into a safe rear hoard. Top-player
        #      replays stage forward 66-70% of the time; a rear hoard is the costless
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
        #   opening under-commitment that caps conversion (open<50 cap/atk ~0.38 vs winner
        #   0.51). 1.0 = strict (need strictly more than current defense); 0.6 = relaxed
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
        self._last_scenario_id = None  # (N,) long: scenario that just terminated, 0 otherwise
        self._last_scenario_success = None  # (N,) bool: advantaged player won the scenario terminal
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
        # See ModelConfig.ship_bin_mode. "absolute" uses SHIP_COUNTS lookup;
        # "fraction" uses round(FRAC_VALUES[bin] * src_ships).
        self.ship_bin_mode = ship_bin_mode
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
        self.shaping_coef = float(shaping_coef)
        # Expansion shaping: potential-based reward on OWNED PRODUCTION (sum of
        # planet production rates owned). Unlike material (ships), production only
        # changes when planets change hands — so a passive hoarder gets 0 from it
        # (avoids the rev8 material-shaping trap). Rewards winning the planet/economy
        # race that decides snowball games. 0.0 = off (default).
        self.expansion_coef = float(expansion_coef)
        # Defense shaping: per-step penalty for losing owned production (consolidation
        # incentive — rewards HOLDING planets, complements expansion's GRAB). 0.0 = off.
        self.defense_coef = float(defense_coef)
        # Early capture shaping: per-step bonus for each net new planet owned above
        # starting count (1), decayed linearly from 1.0→0.0 over early_capture_steps.
        # Gives gradient signal for the opening probe that the terminal reward cannot see.
        # Coeff math: sum(coeff*(1-t/100), t=4..100) ≈ 97*0.48*coeff per planet captured
        # at step 3. Keep cumulative bonus ≤ 10-15% of terminal win → coeff 0.002-0.003.
        self.early_capture_coef = float(early_capture_coef)
        # Production-share capture reward (the unified term). Symmetric, capture-time-ANCHORED,
        # value-weighted by share of total board production: r = coef·decay(t_cap)·Δ(prod/total).
        # Capturing pays +coef·decay(now)·(prod/total); losing pays −coef·decay(t_cap)·(prod/total)
        # with the SAME anchor → capture-then-lose nets 0 (no tennis farm; losing drives holding).
        # Absolute (no opponent subtraction → mirror-safe) and bounded (≤1.1·coef → loss always
        # negative under pure ±1 terminal). Replaces early_capture(count)+expansion(prod-lead).
        self.prod_share_coef = float(prod_share_coef)
        self.first_strike_steps = int(first_strike_steps)
        self.first_strike_mult = float(first_strike_mult)
        self.early_capture_steps = int(early_capture_steps)
        # Time-to-victory velocity bonus: winners get an extra reward scaled by how early
        # they won. reward_win = 1.0 + (episode_steps - T) / episode_steps * speed_coef.
        # A win at step 150 of 500 earns +0.70*speed_coef extra vs +0.02*speed_coef at 490.
        # Creates constant pressure to close games fast; grinding passive wins penalised.
        # speed_coef=0.5 means a step-0 win scores 1.5, a timeout win scores ~1.0.
        # Keep ≤ 0.5 so a slow win still beats a fast loss.
        self.speed_coef = float(speed_coef)
        # Consolidation bonus (force-concentration lever, 2026-06-15): a ONE-TIME +coef when a
        # NET-NEW captured planet SURVIVES consolidation_steps. Unlike defense_coef (per-step
        # penalty for losing production → hoard-to-avoid → FLOOD), this is success-GATED (paid
        # only when a capture sticks), EVENT-based + capped (one-time → can't farm by sitting),
        # and NEW-captures-only (home/initial excluded) → rewards expand-AND-consolidate, not
        # blanket hoarding. Prices "commit enough to hold" → concentration. See
        # project_force_concentration_wall; KILL if reinforce-rate/garr floods like defense_coef.
        self.consolidation_coef = float(consolidation_coef)
        self.consolidation_steps = int(consolidation_steps)
        # Capture follow-through reward (project_capture_quality): the triage diagnostic showed
        # the wall is not "reinforce more"; captured planets are often born/left unproductive.
        # A net-new capture gets one window to prove utility: either it launches an ATTACK from
        # that planet, or it is still one of the holder's top-3 frontline planets at window end.
        # Optional idle penalty prices captures that do neither. Off by default.
        self.capture_utility_coef = float(capture_utility_coef)
        self.capture_utility_window = int(capture_utility_window)
        self.capture_idle_penalty = float(capture_idle_penalty)
        self.capture_utility_active = (
            self.capture_utility_coef != 0.0 or self.capture_idle_penalty != 0.0
        )
        # Lever A — decisive-mass reward: +coef once when our INFLIGHT force converging on an
        # ENEMY target first reaches the capture floor (projected defenders + 3-turn reaction +
        # overhead — deb's capture_floor). Board-grounded, NOT outcome-tied → injects the force-
        # concentration gradient symmetric self-play structurally cannot price (the wall: we get
        # out-massed ~2.3x, planets@50=6 invariant). One credit per crossing (capped, no over-mass
        # scaling); re-arms when no longer sufficient. project_force_concentration_wall. 0.0 = off.
        self.decisive_mass_coef = float(decisive_mass_coef)
        # Weight on producer_v2's reactive-reinforcement margin (beta*rho(eta)*enemy_mass). v2
        # uses 2.2 (planner-conservative); for a TRAINING reward a high beta makes crossings rare
        # (sparse signal) — lower it if `decis` stays ~0 on the resumed (trained) policy.
        self.decisive_mass_beta = float(decisive_mass_beta)
        # Decisive-mass GAP diagnostic: measure how far our inflight attacks fall short of the
        # capture floor (dm_gap/dm_cross/...) using the EXACT reward floor, EVEN when the reward
        # itself is off (decisive_mass_coef=0) — tells us whether the policy is moving toward the
        # decmass target vs only improving adjacent competence. project_force_concentration_wall.
        self.decisive_diag = bool(decisive_diag)
        self.prev_decisive_suff = None       # (N, P, num_players) bool — allocated in reset()
        self._decisive_credit = None         # (N, num_players) diag accumulator — reset_reinforce_stats()
        # dm_* phase-split (early<50 / mid50-100 / late>=100) accumulators — reset_reinforce_stats()
        self._dm_targets = None              # (N, num_players, 3) enemy targets w/ our inflight mass>0
        self._dm_cross = None                # of those, mass >= floor
        self._dm_ratio_sum = None            # sum mass/floor over targets
        self._dm_gap_sum = None              # sum max(0,floor-mass)/floor over targets
        self._dm_overkill_sum = None         # sum mass/floor over CROSSED targets (denom = _dm_cross)
        self._dm_nearmiss = None             # of targets, ratio in [0.75, 1.0)
        # Handicap curriculum: fraction of games where player 0 starts with fewer ships.
        # Forces the agent to practise fighting from behind — the bimodal collapse state
        # that pure symmetric self-play never generates enough gradient for.
        self.handicap_frac = float(handicap_frac)
        self.handicap_ships = int(handicap_ships)
        # Start-State Domain Randomisation (SSDR): with probability ssdr_frac,
        # fast-forward a freshly-reset env by U(1, ssdr_max_steps) random steps
        # before handing it to the learner. Both players take random actions during
        # the warmup so the learner wakes up in a messy, asymmetric mid-game state.
        # This shatters the symmetric-start passive Nash equilibrium.
        self.ssdr_frac = float(ssdr_frac)
        self.ssdr_max_steps = int(ssdr_max_steps)  # now = max extra planets granted to opponent
        # Neutral garrison scale (board-curriculum): multiply neutral planet ships by
        # this factor at reset, symmetrically (both players face the same board). >1.0
        # makes captures expensive → single-source can't capture → must aggregate
        # multiple sources (concentration). Applied BEFORE home assignment so home
        # planets (overwritten to 10 ships) are unaffected. Training-only; eval/LB
        # use default boards (scale 1.0) — the transfer test is whether the
        # concentration habit carries to normal-garrison boards.
        self.neutral_garrison_scale = float(neutral_garrison_scale)
        if scenario_curriculum not in _SCENARIO_NAME_TO_ID and scenario_curriculum != _SCENARIO_MIXED:
            raise ValueError(f"unknown scenario_curriculum={scenario_curriculum!r}")
        self.scenario_curriculum = scenario_curriculum
        self.scenario_fraction = float(scenario_fraction)
        self.scenario_deadline = int(scenario_deadline)
        self.scenario_id = None          # (N,) long; 0 = normal generated board
        self.scenario_adv_player = None  # (N,) long; player whose tactic is being tested
        self.scenario_target = None      # (N,) long; focal target planet index
        self.scenario_done_step = None   # (N,) long; scenario deadline
        # Asymmetric Planet SSDR: with probability ssdr_frac, grant opponent 1..ssdr_max_steps
        # extra neutral planets at reset. No random play, no fleet explosion.
        # Breaks symmetric-start Nash cleanly.
        #
        # ssdr_self_only_mask: bool tensor (N,) set by training loop each rollout.
        # True = self-play env (SSDR active), False = pool env (symmetric start).
        # If None, SSDR applies to all envs.
        self._ssdr_self_mask: torch.Tensor | None = None  # set via set_ssdr_mask()

        # Self-boost (handicapped-real-planner curriculum): the INVERSE of SSDR — grant
        # OUR seat (_self_boost_seat) _self_boost_k extra neutral planets at reset, in the
        # envs flagged by _self_boost_mask (the pool/planner envs). The training loop tapers
        # k -> 0 so a strong pool planner (deb) is beatable early (win-gradient for holding)
        # then weans off the head-start. Set via set_self_boost(); inert when k<=0.
        self._self_boost_k = 0
        self._self_boost_seat = 0
        self._self_boost_mask: torch.Tensor | None = None

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
        self.prev_material: torch.Tensor = None     # (N, num_players) float
        self.prev_production: torch.Tensor = None   # (N, num_players) float — owned production for expansion shaping
        self.prev_owned: torch.Tensor = None        # (N, num_players) float — owned planet count for delta-capture shaping
        self.prev_planet_owner: torch.Tensor = None  # (N, P) long — per-planet owner last step (prod-share term)
        self.capture_time: torch.Tensor = None       # (N, P) long — step the current owner acquired each planet
        self.total_board_prod: torch.Tensor = None   # (N,) float — Σ production of all planets at reset (normalizer)
        # Seeds (per-env) so we can deterministically auto-reset
        self.seeds: list[int] = []

        # Pre-computed once per reset for orbital motion
        self._planet_initial_angle: torch.Tensor = None  # (N, P) float
        self._planet_orbital_r: torch.Tensor = None      # (N, P) float
        self._planet_is_orbiting: torch.Tensor = None    # (N, P) bool

    def set_ssdr_mask(self, self_play_mask: torch.Tensor) -> None:
        """Mark which envs are self-play (SSDR active) vs pool (symmetric start).

        Call once per rollout from the training loop when pool assignments change:
            env.set_ssdr_mask(torch.arange(N) < N_self)  # first N_self = self-play

        If never called, SSDR applies to all envs (original behaviour).
        """
        self._ssdr_self_mask = self_play_mask.bool().cpu()

    def set_self_boost(self, k: int, seat: int, env_mask: torch.Tensor | None) -> None:
        """Grant OUR seat `seat` `k` extra neutral planets at reset in envs where env_mask
        is True (handicapped-real-planner curriculum). Call once per rollout; k tapers to 0."""
        self._self_boost_k = int(k)
        self._self_boost_seat = int(seat)
        self._self_boost_mask = env_mask.bool().cpu() if env_mask is not None else None

    def _maybe_self_boost(self, pad, n: int, base: int, env_i: int) -> None:
        """Inverse of SSDR: give OUR seat extra neutral planets in boosted (pool) envs."""
        if self._self_boost_k <= 0 or self._self_boost_mask is None:
            return
        if not bool(self._self_boost_mask[env_i]):
            return
        seat = self._self_boost_seat
        neutral_idx = [i for i in range(n)
                       if pad[i, 1] == -1 and i != base and i != base + 3]
        random.shuffle(neutral_idx)
        for ni in neutral_idx[:self._self_boost_k]:
            pad[ni, 1] = seat
            pad[ni, 5] = max(10, int(pad[ni, 6] * 3))

    def _ssdr_active_for(self, env_i: int) -> bool:
        """Return True if SSDR should apply to env index env_i."""
        if self.ssdr_frac <= 0.0:
            return False
        if self._ssdr_self_mask is None:
            return True  # no mask set → apply to all
        return bool(self._ssdr_self_mask[env_i].item())

    def _scale_neutrals(self, pad: np.ndarray, n: int) -> None:
        if self.neutral_garrison_scale <= 1.0:
            return
        for i in range(n):
            if pad[i, 1] == -1:
                pad[i, 5] = float(int(pad[i, 5] * self.neutral_garrison_scale))

    def _choose_scenario(self, rng: random.Random) -> int:
        if self.scenario_fraction <= 0.0 or rng.random() >= self.scenario_fraction:
            return _SCENARIO_OFF
        if self.scenario_curriculum == _SCENARIO_MIXED:
            # Attack-side lessons are the main intended pressure; defensive peel
            # remains in the mix, but at lower weight.
            return rng.choice([
                _SCENARIO_AGG_ATTACK,
                _SCENARIO_STAGE_ATTACK,
                _SCENARIO_AGG_ATTACK,
                _SCENARIO_STAGE_ATTACK,
                _SCENARIO_HOLD_UNDER_PEEL,
            ])
        return _SCENARIO_NAME_TO_ID[self.scenario_curriculum]

    def _apply_scenario(
        self,
        pad: np.ndarray,
        alive: np.ndarray,
        rng: random.Random,
    ) -> tuple[int, int, int, int]:
        """Install a tiny concentration scenario in one env.

        The scenarios are deliberately small. They are not meant to mimic full games;
        they create a short terminal lesson where the advantaged player wins only by
        concentrating or staging enough mass on the focal target.
        """
        scenario_id = self._choose_scenario(rng)
        if scenario_id == _SCENARIO_OFF:
            return _SCENARIO_OFF, 0, -1, 0

        adv = rng.randint(0, 1)
        opp = 1 - adv
        mirror = adv == 1

        def mx(x: float) -> float:
            return 100.0 - x if mirror else x

        pad[:, :] = 0.0
        pad[:, 1] = -1.0
        alive[:] = False

        def planet(idx: int, owner: int, x: float, y: float,
                   ships: float, prod: float, radius: float = 2.2) -> None:
            pad[idx] = [idx, owner, mx(x), y, radius, ships, prod]
            alive[idx] = True

        target = 2
        deadline = self.scenario_deadline if self.scenario_deadline > 0 else 20

        if scenario_id == _SCENARIO_AGG_ATTACK:
            # No single advantaged source can take the neutral target; two sources can.
            # If the target is not taken by the deadline, the larger opponent economy wins.
            planet(0, adv, 28.0, 63.0, 55.0, 2.0)
            planet(1, adv, 28.0, 77.0, 55.0, 2.0)
            planet(target, -1, 50.0, 70.0, 80.0, 5.0)
            planet(3, opp, 84.0, 70.0, 130.0, 4.0)
            planet(4, opp, 76.0, 84.0, 35.0, 1.0)
            planet(5, adv, 16.0, 70.0, 25.0, 1.0)
        elif scenario_id == _SCENARIO_STAGE_ATTACK:
            # A prior friendly fleet is already committed but stops short. The winning
            # move is to add one more source to the same target before the deadline.
            planet(0, adv, 30.0, 70.0, 45.0, 2.0)
            planet(1, adv, 31.0, 82.0, 42.0, 2.0)
            planet(target, -1, 50.0, 70.0, 75.0, 5.0)
            planet(3, opp, 84.0, 70.0, 125.0, 4.0)
            planet(4, opp, 76.0, 84.0, 35.0, 1.0)
            planet(5, adv, 17.0, 70.0, 25.0, 1.0)
        elif scenario_id == _SCENARIO_HOLD_UNDER_PEEL:
            # The focal planet starts ours but thin; an enemy peel is inbound. The
            # winning move is defensive concentration from both nearby sources.
            planet(0, adv, 38.0, 64.0, 45.0, 2.0)
            planet(1, adv, 38.0, 76.0, 45.0, 2.0)
            planet(target, adv, 50.0, 70.0, 15.0, 5.0)
            planet(3, opp, 84.0, 70.0, 125.0, 4.0)
            planet(4, opp, 75.0, 84.0, 40.0, 1.0)
            planet(5, adv, 28.0, 70.0, 20.0, 1.0)
        else:
            raise AssertionError(f"unhandled scenario id {scenario_id}")

        return scenario_id, adv, target, deadline

    def _scenario_fleet_seed(self, env_i: int) -> None:
        """Seed existing inbound fleets for scenarios that test staged/defensive follow-up."""
        sid = int(self.scenario_id[env_i].item()) if self.scenario_id is not None else _SCENARIO_OFF
        if sid not in (_SCENARIO_STAGE_ATTACK, _SCENARIO_HOLD_UNDER_PEEL):
            return
        adv = int(self.scenario_adv_player[env_i].item())
        opp = 1 - adv
        target = int(self.scenario_target[env_i].item())
        if sid == _SCENARIO_STAGE_ATTACK:
            owner, ships, src_pid = adv, 45.0, 0
            x = 30.0 if adv == 0 else 70.0
            y = 70.0
        else:
            owner, ships, src_pid = opp, 130.0, 3
            # Keep the peel already committed, but not so close that an immediate
            # correct reinforce cannot arrive first.
            x = 84.0 if adv == 0 else 16.0
            y = 70.0

        # Seeded scenario fleets must use the same intercept aimer as normal
        # launches. Straight current-position aim can harmlessly miss an orbiting
        # target and invalidate the lesson.
        src_x = torch.tensor([[x]], dtype=torch.float32, device=self.device)
        src_y = torch.tensor([[y]], dtype=torch.float32, device=self.device)
        src_r = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        ship_count = torch.tensor([[ships]], dtype=torch.float32, device=self.device)
        target_idx = torch.tensor([[target]], dtype=torch.long, device=self.device)
        angle = float(self._target_intercept_angle(src_x, src_y, src_r, ship_count, target_idx)[0, 0].item())
        src_px = float(self.planets[env_i, src_pid, 2].item())
        src_py = float(self.planets[env_i, src_pid, 3].item())
        src_pr = float(self.planets[env_i, src_pid, 4].item())
        start_x = src_px + math.cos(angle) * (src_pr + 0.1)
        start_y = src_py + math.sin(angle) * (src_pr + 0.1)
        self.fleets[env_i, 0, 0] = 0.0
        self.fleets[env_i, 0, 1] = float(owner)
        self.fleets[env_i, 0, 2] = start_x
        self.fleets[env_i, 0, 3] = start_y
        self.fleets[env_i, 0, 4] = angle
        self.fleets[env_i, 0, 5] = float(src_pid)
        self.fleets[env_i, 0, 6] = ships
        self.fleet_alive[env_i, 0] = True
        self.next_fleet_id[env_i] = 1

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
        scenario_ids = []
        scenario_adv = []
        scenario_targets = []
        scenario_deadlines = []

        for seed_idx, seed in enumerate(seeds):
            init_rng = random.Random(seed)
            ang_vel = init_rng.uniform(0.025, 0.05)
            angular_velocities.append(ang_vel)

            raw_planets = generate_planets(init_rng)  # list of [id, owner, x, y, r, ships, prod]
            n = len(raw_planets)
            pad = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
            for i, p in enumerate(raw_planets):
                pad[i] = p
            # Board-curriculum: scale neutral garrison symmetrically. Applied BEFORE
            # home assignment (next block overwrites home planets' ships to 10), so
            # only the neutrals that REMAIN neutral after assignment are scaled.
            self._scale_neutrals(pad, n)
            planets_list.append(pad)

            alive = np.zeros(MAX_PLANETS, dtype=bool)
            alive[:n] = True

            # Home planet assignment (matches fast_env reset)
            num_groups = n // 4
            if num_groups > 0:
                home_group = init_rng.randint(0, num_groups - 1)
                base = home_group * 4
                if self.num_players == 2:
                    p0_ships = (self.handicap_ships
                                if self.handicap_frac > 0 and random.random() < self.handicap_frac
                                else 10)
                    pad[base, 1] = 0;     pad[base, 5] = p0_ships
                    pad[base + 3, 1] = 1; pad[base + 3, 5] = 10
                    # SSDR: grant opponent 1..ssdr_max_steps extra neutral planets
                    # Only applies to self-play envs (not pool envs) per mask.
                    if self._ssdr_active_for(seed_idx) and random.random() < self.ssdr_frac:
                        k = random.randint(1, max(1, self.ssdr_max_steps))
                        # find neutral planets (owner=-1, alive) excluding home slots
                        neutral_idx = [i for i in range(n)
                                       if pad[i, 1] == -1 and i != base and i != base + 3]
                        random.shuffle(neutral_idx)
                        for ni in neutral_idx[:k]:
                            prod = pad[ni, 6]
                            pad[ni, 1] = 1  # give to opponent
                            pad[ni, 5] = max(10, int(prod * 3))  # realistic ships
                    # Self-boost (handicapped-real-planner): grant OUR seat extra planets
                    self._maybe_self_boost(pad, n, base, seed_idx)
                elif self.num_players == 4:
                    for j in range(4):
                        pad[base + j, 1] = j; pad[base + j, 5] = 10
                else:
                    pad[base, 1] = 0; pad[base, 5] = 10

            # Mark unused slots as neutral (-1)
            pad[n:, 1] = -1
            sid, adv, tgt, deadline = self._apply_scenario(
                pad, alive, random.Random(f"orbit-wars-scenario-{seed}")
            )
            scenario_ids.append(sid)
            scenario_adv.append(adv)
            scenario_targets.append(tgt)
            scenario_deadlines.append(deadline)
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
        self.scenario_id = torch.tensor(scenario_ids, dtype=torch.long, device=self.device)
        self.scenario_adv_player = torch.tensor(scenario_adv, dtype=torch.long, device=self.device)
        self.scenario_target = torch.tensor(scenario_targets, dtype=torch.long, device=self.device)
        self.scenario_done_step = torch.tensor(scenario_deadlines, dtype=torch.long, device=self.device)
        self._last_scenario_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_scenario_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._precompute_orbital_params()
        self._init_comets()
        for env_i in range(self.num_envs):
            self._scenario_fleet_seed(env_i)
        self.prev_material = self._compute_material()
        self.prev_production = self._compute_production()
        owner_p = self.planets[:, :, 1].long()
        self.prev_owned = torch.zeros(self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        for pl in range(self.num_players):
            self.prev_owned[:, pl] = ((owner_p == pl) & self.planet_alive).float().sum(dim=1)
        # Prod-share term state. Initial ownership is pre-existing state, not a capture: holding
        # a home pays nothing, while losing it is a negative delta anchored at t=0.
        self.prev_planet_owner = torch.where(
            self.planet_alive,
            owner_p,
            torch.full_like(owner_p, -1),
        )
        self.capture_time = torch.zeros(self.num_envs, self.planets.shape[1],
                                        dtype=torch.long, device=self.device)
        self.total_board_prod = self._prod_share_total_board_prod()
        # Consolidation-bonus per-planet state (only used when consolidation_coef != 0): track,
        # per planet, the owner being held + how long since this holding episode began, whether
        # it began as a CAPTURE (initial owners are NOT captures), and whether already credited.
        P = self.planets.shape[1]
        self.cap_owner = owner_p.clone()                                              # (N, P)
        self.cap_age = torch.zeros(self.num_envs, P, dtype=torch.long, device=self.device)
        self.cap_credited = torch.zeros(self.num_envs, P, dtype=torch.bool, device=self.device)
        self.cap_is_capture = torch.zeros(self.num_envs, P, dtype=torch.bool, device=self.device)
        # Capture-utility state mirrors cap_* but tracks whether the current holding episode has
        # used the captured planet as an attack source within the utility window.
        self.cu_owner = owner_p.clone()
        self.cu_age = torch.zeros(self.num_envs, P, dtype=torch.long, device=self.device)
        self.cu_credited = torch.zeros(self.num_envs, P, dtype=torch.bool, device=self.device)
        self.cu_is_capture = torch.zeros(self.num_envs, P, dtype=torch.bool, device=self.device)
        self.cu_used_attack = torch.zeros(self.num_envs, P, dtype=torch.bool, device=self.device)
        # Lever A: per (env, planet, player) "inflight force already sufficient to take this
        # enemy planet" — so the bonus fires only on the crossing (assembly), not every step.
        self.prev_decisive_suff = torch.zeros(
            self.num_envs, P, self.num_players, dtype=torch.bool, device=self.device)
        # Reverse-edge reinforce cooldown: last step each (player, src, tgt) reinforce edge fired.
        # _DM-style NEVER sentinel so untouched edges never trip the (step - last) <= K test.
        if self.reverse_edge_cooldown > 0:
            self.reinf_cd = torch.full(
                (self.num_envs, self.num_players, P, P), _REINF_CD_NEVER,
                dtype=torch.long, device=self.device)
        return self._state_dict()

    def _compute_material(self) -> torch.Tensor:
        owner_p = self.planets[:, :, 1].long()
        owner_f = self.fleets[:, :, 1].long()
        ships_p = self.planets[:, :, 5] * self.planet_alive.float()
        ships_f = self.fleets[:, :, 6] * self.fleet_alive.float()
        material = torch.zeros(self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        for pl in range(self.num_players):
            material[:, pl] = (
                ((owner_p == pl).float() * ships_p).sum(dim=1)
                + ((owner_f == pl).float() * ships_f).sum(dim=1)
            )
        return material

    def _prod_share_regular_alive(self) -> torch.Tensor:
        regular = torch.arange(self.planets.shape[1], device=self.device) < COMET_SLOT_START
        return self.planet_alive & regular.unsqueeze(0)

    def _prod_share_total_board_prod(self) -> torch.Tensor:
        return (self.planets[:, :, 6] * self._prod_share_regular_alive().float()).sum(dim=1).clamp(min=1.0)

    def _prod_share_bonus(self, terminal_rewards: torch.Tensor) -> torch.Tensor:
        """Unified capture reward: symmetric, capture-time-ANCHORED, value-weighted by share of
        total board production, excluding transient comet slots from both numerator and denominator.
        For each player pl and each regular planet whose owner CHANGED this step:
            gain (captured by pl) : +coef · decay(NOW)        · prod/total_board
            loss (lost from pl)   : −coef · decay(capture_time) · prod/total_board   (same anchor)
        so a capture and its eventual loss cancel exactly (no tennis farm; losing drives holding).
        Mutates capture_time (changed planets -> now) and prev_planet_owner (dead slots -> -1)."""
        owner_now = self.planets[:, :, 1].long()                          # (N, P)
        prod_p = self.planets[:, :, 6]                                    # (N, P)
        regular_alive = self._prod_share_regular_alive()
        changed = (owner_now != self.prev_planet_owner) & regular_alive
        decay_now = (torch.exp(-2.5 * self.step_count.float() / self.episode_steps) + 0.10).unsqueeze(1)
        decay_cap = torch.exp(-2.5 * self.capture_time.float() / self.episode_steps) + 0.10
        share = prod_p / self.total_board_prod.unsqueeze(1)               # (N, P) fraction of regular economy
        for pl in range(self.num_players):
            gain = (changed & (owner_now == pl)).float() * decay_now * share
            loss = (changed & (self.prev_planet_owner == pl)).float() * decay_cap * share
            terminal_rewards[:, pl] = terminal_rewards[:, pl] + self.prod_share_coef * (gain - loss).sum(dim=1)
        step_b = self.step_count.long().unsqueeze(1).expand_as(self.capture_time)
        self.capture_time = torch.where(changed, step_b, self.capture_time)
        self.prev_planet_owner = torch.where(self.planet_alive, owner_now,
                                             torch.full_like(owner_now, -1))
        return terminal_rewards

    def _consolidation_bonus(self, terminal_rewards: torch.Tensor) -> torch.Tensor:
        """One-time +consolidation_coef per NET-NEW captured planet that survives
        consolidation_steps. Drives the consolidation state machine (cap_owner/age/credited/
        is_capture) from the CURRENT planet owners and credits the holder. See __init__ note."""
        cur_owner = self.planets[:, :, 1].long()                       # (N, P)
        changed = cur_owner != self.cap_owner
        # reset the holding episode on any ownership change; else age it one step
        self.cap_age = torch.where(changed, torch.zeros_like(self.cap_age), self.cap_age + 1)
        self.cap_credited = self.cap_credited & ~changed
        # a mid-episode change to a real player = a capture (initial owners never "change")
        self.cap_is_capture = torch.where(changed, cur_owner >= 0, self.cap_is_capture)
        self.cap_owner = cur_owner
        ready = (self.cap_is_capture & (self.cap_age >= self.consolidation_steps)
                 & ~self.cap_credited & (cur_owner >= 0) & self.planet_alive)
        if ready.any():
            for pl in range(self.num_players):
                cnt = (ready & (cur_owner == pl)).float().sum(dim=1)    # (N,) planets consolidated
                terminal_rewards[:, pl] = terminal_rewards[:, pl] + self.consolidation_coef * cnt
            self.cap_credited = self.cap_credited | ready
        return terminal_rewards

    def _capture_frontline_mask(self, cur_owner: torch.Tensor) -> torch.Tensor:
        """Top-3 owned planets nearest to any enemy planet for each player/env."""
        P = cur_owner.shape[1]
        x = self.planets[:, :, 2]
        y = self.planets[:, :, 3]
        alive = self.planet_alive
        dx = x.unsqueeze(2) - x.unsqueeze(1)
        dy = y.unsqueeze(2) - y.unsqueeze(1)
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)                 # (N, P, P)
        frontline = torch.zeros(self.num_envs, P, dtype=torch.bool, device=self.device)
        big = torch.full((self.num_envs, P), 1e9, dtype=torch.float32, device=self.device)
        k = min(3, P)
        for pl in range(self.num_players):
            mine = (cur_owner == pl) & alive
            enemy = (cur_owner >= 0) & (cur_owner != pl) & alive
            enemy_any = enemy.any(dim=1, keepdim=True)
            nearest_enemy = torch.where(enemy.unsqueeze(1), dist, 1e9).min(dim=2).values
            scores = torch.where(mine & enemy_any, nearest_enemy, big)
            idx = torch.topk(scores, k, dim=1, largest=False).indices
            picked = torch.zeros_like(frontline)
            picked.scatter_(1, idx, True)
            picked &= scores < 1e9
            frontline |= picked & mine
        return frontline

    def _capture_utility_bonus(self, terminal_rewards: torch.Tensor) -> torch.Tensor:
        """One-time reward/penalty for whether a net-new capture becomes useful within K steps."""
        # First credit attack utility against the PRE-change holder. Actions launch before combat;
        # if the source is lost on the same tick, the attack still happened and should be credited.
        attack_ready = (
            self.cu_is_capture
            & self.cu_used_attack
            & ~self.cu_credited
            & (self.cu_owner >= 0)
            & self.planet_alive
        )
        if attack_ready.any():
            for pl in range(self.num_players):
                cnt = (attack_ready & (self.cu_owner == pl)).float().sum(dim=1)
                terminal_rewards[:, pl] = terminal_rewards[:, pl] + self.capture_utility_coef * cnt
            self.cu_credited = self.cu_credited | attack_ready

        cur_owner = self.planets[:, :, 1].long()
        changed = cur_owner != self.cu_owner
        self.cu_age = torch.where(changed, torch.zeros_like(self.cu_age), self.cu_age + 1)
        self.cu_credited = self.cu_credited & ~changed
        self.cu_is_capture = torch.where(changed, cur_owner >= 0, self.cu_is_capture)
        self.cu_used_attack = self.cu_used_attack & ~changed
        self.cu_owner = cur_owner

        window_ready = (
            self.cu_is_capture
            & ~self.cu_credited
            & (self.cu_age >= self.capture_utility_window)
            & (cur_owner >= 0)
            & self.planet_alive
        )
        if window_ready.any():
            frontline = self._capture_frontline_mask(cur_owner)
            useful = window_ready & frontline
            idle = window_ready & ~frontline
            for pl in range(self.num_players):
                useful_cnt = (useful & (cur_owner == pl)).float().sum(dim=1)
                idle_cnt = (idle & (cur_owner == pl)).float().sum(dim=1)
                terminal_rewards[:, pl] = (
                    terminal_rewards[:, pl]
                    + self.capture_utility_coef * useful_cnt
                    - self.capture_idle_penalty * idle_cnt
                )
            self.cu_credited = self.cu_credited | window_ready
        return terminal_rewards

    def _fleet_target_idx(self) -> torch.Tensor:
        """Geometry-only target planet index per fleet, (N, F), or -1 if none.
        Mirrors the heading-projection in get_features (nearest alive planet ahead of
        the fleet along its heading). Player-independent — depends only on geometry."""
        fx = self.fleets[:, :, 2]
        fy = self.fleets[:, :, 3]
        fang = self.fleets[:, :, 4]
        fcos = torch.cos(fang)
        fsin = torch.sin(fang)
        x = self.planets[:, :, 2]
        y = self.planets[:, :, 3]
        r = self.planets[:, :, 4]
        F = self.fleets.shape[1]
        vx_fp = x.unsqueeze(1) - fx.unsqueeze(2)                 # (N, F, P)
        vy_fp = y.unsqueeze(1) - fy.unsqueeze(2)
        along_fp = vx_fp * fcos.unsqueeze(2) + vy_fp * fsin.unsqueeze(2)
        perp_fp = torch.abs(vx_fp * fsin.unsqueeze(2) - vy_fp * fcos.unsqueeze(2))
        alive_expand = self.planet_alive.unsqueeze(1).expand(-1, F, -1)
        candidate = (along_fp > 0) & (perp_fp < r.unsqueeze(1) + 2.0) & alive_expand
        has_candidate = candidate.any(dim=2)
        dists_fp = torch.sqrt(vx_fp * vx_fp + vy_fp * vy_fp)
        tgt_idx = dists_fp.masked_fill(~candidate, 1e6).argmin(dim=2)    # (N, F)
        return torch.where(has_candidate, tgt_idx, torch.full_like(tgt_idx, -1))

    def _decisive_mass_fields(self):
        """Per-(N,P,num_players) inflight mass, capture floor, max-ETA and is_enemy mask — the
        EXACT quantities producer_v2's capture floor uses. Shared by the Lever-A reward
        (_decisive_mass_bonus) and the dm_* gap diagnostic (_accumulate_decisive_diag) so the two
        can never drift. project_force_concentration_wall.

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
        tgt_idx = self._fleet_target_idx()                             # (N, F)
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
            floor[:, :, pl] = (garr + prod * eta + inbound
                               + self.decisive_mass_beta * rho * enemy_mass + _DM_OVERHEAD)
            eta_out[:, :, pl] = eta
            is_enemy[:, :, pl] = alive & (owner != pl) & (owner >= 0)
        return mass, floor, eta_out, is_enemy

    def _decisive_mass_bonus(self, terminal_rewards: torch.Tensor, fields=None) -> torch.Tensor:
        """Lever A: +decisive_mass_coef the step our inflight force converging on an ENEMY target
        first reaches producer_v2's capture floor (one credit per crossing, no over-mass scaling;
        prev_decisive_suff re-arms when mass drops below floor / the planet stops being enemy).
        `fields` = a precomputed (mass, floor, eta, is_enemy) from _decisive_mass_fields() so
        step() shares one computation with the diagnostic; None → compute fresh."""
        mass, floor, _eta, is_enemy = fields if fields is not None else self._decisive_mass_fields()
        for pl in range(self.num_players):
            suff = is_enemy[:, :, pl] & (mass[:, :, pl] >= floor[:, :, pl])
            crossing = suff & ~self.prev_decisive_suff[:, :, pl]
            cnt = crossing.float().sum(dim=1)
            terminal_rewards[:, pl] = terminal_rewards[:, pl] + self.decisive_mass_coef * cnt
            self.prev_decisive_suff[:, :, pl] = suff
            if self._decisive_credit is not None:
                self._decisive_credit[:, pl] += cnt
        return terminal_rewards

    def _accumulate_decisive_diag(self, mass, floor, eta, is_enemy):
        """Phase-split dm_* GAP diagnostic from the EXACT reward floor (decisive_diag). Per
        (env, player): targets = enemy planets with our inflight mass>0; ratio = mass/floor;
        cross = ratio>=1; gap = max(0,1-ratio); near-miss = ratio in [0.75,1); overkill = ratio
        on crossed. Reward-side OFF is fine — this reads whether the policy moves toward the
        decmass target regardless. project_force_concentration_wall."""
        sc = self.step_count.float()                                   # (N,)
        w = torch.stack([(sc < 50).float(),
                         ((sc >= 50) & (sc < 100)).float(),
                         (sc >= 100).float()], dim=1)                  # (N, 3) phase one-hot
        fl = floor.clamp(min=1e-6)
        ratio = mass / fl                                              # (N, P, players)
        gap = (fl - mass).clamp(min=0.0) / fl
        crossed = (mass >= floor).float()
        nearmiss = ((ratio >= 0.75) & (ratio < 1.0)).float()
        att = (is_enemy & (mass > 0)).float()                         # attacked enemy targets
        we = w.unsqueeze(1)                                            # (N, 1, 3) → over players
        def acc(per_target):                                          # (N,P,players) → (N,players,3)
            return (att * per_target).sum(dim=1).unsqueeze(-1) * we
        self._dm_targets += att.sum(dim=1).unsqueeze(-1) * we
        self._dm_cross += acc(crossed)
        self._dm_ratio_sum += acc(ratio)
        self._dm_gap_sum += acc(gap)
        self._dm_overkill_sum += acc(crossed * ratio)
        self._dm_nearmiss += acc(nearmiss)

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

    def _lazy_comets(self):
        """Compute any comet spawn an env reaches THIS step (step_count == spawn-1). Cheap check
        (5 scalar compares); the heavy generate_comet_paths runs only for the few envs crossing a
        spawn boundary, and only for spawns games actually reach."""
        if self.episode_steps <= COMET_SPAWN_STEPS[0]:
            return
        sc = self.step_count
        for si, S in enumerate(COMET_SPAWN_STEPS):
            if S >= self.episode_steps:
                continue
            need = (sc == (S - 1)) & ~self._comet_spawn_done[:, si]
            if need.any():
                self._compute_spawn(torch.where(need)[0].cpu().tolist(), si)

    def _compute_spawn(self, env_indices, si):
        """Fill the comet schedule for spawn COMET_SPAWN_STEPS[si] for the given envs. Reuses
        kaggle's generate_comet_paths so the ellipse math + comet RNG order (paths THEN ships)
        are byte-identical; folds spawn+advance into the per-step lookup (position, alive,
        check=collision-tested, ships0 at activation)."""
        S, T1 = COMET_SPAWN_STEPS[si], self.episode_steps + 1
        for e in env_indices:
            self._comet_spawn_done[e, si] = True
            ang_vel = float(self.angular_velocity[e].item())
            init_planets = []
            for i in range(COMET_SLOT_START):
                if bool(self.planet_alive[e, i]):
                    p = self.init_planets[e, i].tolist()
                    init_planets.append([int(p[0]), int(p[1]), p[2], p[3], p[4], p[5], p[6]])
            rng = random.Random(f"orbit_wars-comet-{self.seeds[e]}-{S}")
            paths = _comet_paths_fast(init_planets, ang_vel, S, comet_planet_ids=set(),
                                      comet_speed=COMET_SPEED, rng=rng)
            if not paths:
                continue
            comet_ships = min(rng.randint(1, 99), rng.randint(1, 99),
                              rng.randint(1, 99), rng.randint(1, 99))
            for c in range(N_COMET_SLOTS):
                path = paths[c]
                L = len(path)
                # path_index k = step_count-(S-1). k in [0,L-1]=on-path; k==L = kaggle's
                # stay-put expiry tick (still collidable), then gone.
                for k in range(L + 1):
                    t = (S - 1) + k
                    if t >= T1:
                        break
                    kk = min(k, L - 1)
                    self._comet_xy[e, t, c, 0] = path[kk][0]
                    self._comet_xy[e, t, c, 1] = path[kk][1]
                    self._comet_alive[e, t, c] = True
                    self._comet_check[e, t, c] = (k >= 1)
                    if k == 0:
                        self._comet_ships0[e, t, c] = comet_ships

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
            planet_features:  (N, max_planets, 20)
            fleet_features:   (N, max_fleets, 13)
            global_features:  (N, 11)
            planet_mask:      (N, max_planets) bool
            fleet_mask:       (N, max_fleets) bool
            fire_mask:        (N, MAX_OWNED) bool — can fire (owned planet)
            angle_mask:       (N, MAX_OWNED, NUM_ANGLE_BINS) bool — all True for now
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
        # For each (planet, fleet): compute "along" and "perp" projections.
        # along > 0 AND perp < r + 1.5 → fleet is incoming.
        fx = fleets[:, :, 2]; fy = fleets[:, :, 3]
        fa = fleets[:, :, 4]
        f_owner = fleets[:, :, 1].long()
        f_ships = fleets[:, :, 6]
        fcos = torch.cos(fa); fsin = torch.sin(fa)

        # Broadcast: planet (N, P, 1) vs fleet (N, 1, F)
        vx = x.unsqueeze(2) - fx.unsqueeze(1)
        vy = y.unsqueeze(2) - fy.unsqueeze(1)
        along = vx * fcos.unsqueeze(1) + vy * fsin.unsqueeze(1)
        perp  = torch.abs(vx * fsin.unsqueeze(1) - vy * fcos.unsqueeze(1))
        incoming = (along > 0) & (perp < r.unsqueeze(2) + 1.5) & fleet_alive.unsqueeze(1)

        friend = incoming & (f_owner.unsqueeze(1) == player)
        enemy  = incoming & (f_owner.unsqueeze(1) != player) & (f_owner.unsqueeze(1) >= 0)
        friendly_pressure = (f_ships.unsqueeze(1) * friend.float()).sum(dim=2)  # (N, P)
        enemy_pressure    = (f_ships.unsqueeze(1) * enemy.float()).sum(dim=2)

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

        # Fleet→planet vectors: (N, F, P)
        vx_fp = x.unsqueeze(1) - fx.unsqueeze(2)   # planet_x − fleet_x
        vy_fp = y.unsqueeze(1) - fy.unsqueeze(2)
        along_fp = vx_fp * fcos.unsqueeze(2) + vy_fp * fsin.unsqueeze(2)
        perp_fp  = torch.abs(vx_fp * fsin.unsqueeze(2) - vy_fp * fcos.unsqueeze(2))
        alive_expand = planet_alive.unsqueeze(1).expand(-1, F, -1)  # (N, F, P)
        candidate = (along_fp > 0) & (perp_fp < r.unsqueeze(1) + 2.0) & alive_expand
        has_candidate = candidate.any(dim=2)
        dists_fp = torch.sqrt(vx_fp * vx_fp + vy_fp * vy_fp)
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
        if self.game_phase_features:
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
        gf = torch.stack(gf_list, dim=1)  # (N, 11) or (N, 15) with game-phase

        # Action masks
        owned_idx, slot_valid = self.owned_indices_for(player)
        # max_ships per slot: ships at the owned planet
        gather_idx = owned_idx.unsqueeze(-1).expand(-1, -1, 7)
        owned_ships = self.planets.gather(1, gather_idx)[:, :, 5]
        max_ships = owned_ships * slot_valid.float()
        # Fire mask: can fire iff slot is valid and has at least 1 ship
        fire_mask = slot_valid & (max_ships >= 1.0)
        # Angle mask: all angles legal (no sun-blocking for now)
        angle_mask = torch.ones(N, MAX_OWNED, NUM_ANGLE_BINS, dtype=torch.bool, device=self.device)
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
                dx = x.unsqueeze(2) - x.unsqueeze(1)   # (N, P, P): planet i vs planet j
                dy = y.unsqueeze(2) - y.unsqueeze(1)
                pdist = torch.sqrt(dx * dx + dy * dy)
                INF = torch.finfo(pdist.dtype).max
                d2e = torch.where(enemy_planet.unsqueeze(1), pdist,
                                  torch.full_like(pdist, INF)).min(dim=2).values  # (N, P)
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
        # `incoming` (N, P, F) and enemy mask reuse tensors already computed above.
        enemy_fleet = (f_owner != player) & (f_owner >= 0) & fleet_alive   # (N, F)
        enemy_contest = (f_ships.unsqueeze(1) * (incoming & enemy_fleet.unsqueeze(1)).float()).sum(dim=2)  # (N, P)
        # Threat timing (ch 20-21): ETA-profiled enemy pressure. enemy_contest (ch14) sums ALL
        # inbound enemy mass with no ETA; these add the WHEN. eta = Euclidean planet-fleet dist /
        # fleet speed (matches _fleet_target_idx eta_to_target, torch_env.py:1478). Mirrors
        # features.py compute_pairwise_features ch20-21 byte-for-byte.
        enemy_inc = incoming & enemy_fleet.unsqueeze(1)                    # (N, P, F)
        dist_pf = torch.sqrt((vx * vx + vy * vy).clamp(min=1e-9))          # (N, P, F)
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
            friendly_contest=friendly_pressure,   # (N, P): own ships already inbound per target
            enemy_mass_soon=enemy_mass_soon,
            threat_imminence=threat_imminence,
        )

        return {
            "planet_features": pf,
            "fleet_features":  ff,
            "global_features": gf,
            "planet_mask":     planet_alive,
            "fleet_mask":      fleet_alive,
            "fire_mask":       fire_mask,
            "angle_mask":      angle_mask,
            "target_mask":     target_mask,
            "slot_valid":      slot_valid,
            "owned_indices":   owned_idx,
            "max_ships":       max_ships,
            "owned_count":     owned_count,
            "pairwise_features": pairwise,
        }

    def _compute_pairwise(self, planets, planet_alive, P, owned_idx, slot_valid, player,
                          enemy_contest=None, friendly_contest=None,
                          enemy_mass_soon=None, threat_imminence=None):
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
        ships_at_arr = (ships_b + prod_b * 5.0 * eta).clamp(max=500.0) / 200.0
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
        px_a = planets[:, :, 2]                                  # (N, P)
        py_a = planets[:, :, 3]
        gar_a = planets[:, :, 5]
        own_a = planets[:, :, 1].long()
        dxe = px_a.unsqueeze(2) - px_a.unsqueeze(1)             # (N, P_src, P_tgt)
        dye = py_a.unsqueeze(2) - py_a.unsqueeze(1)
        pde = torch.sqrt((dxe * dxe + dye * dye).clamp(min=1e-9))
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

        # Stack channels
        out = torch.stack([
            sin_a, cos_a, dist / BOARD_SIZE, 1.0 / (eta + 1.0),
            sun_safe, is_mine_b, is_enemy_b, is_neutral_b, prod_b, valid_b,
            ships_at_arr, cap_gap, roi_20, roi_50, contest_b, reach_b,
            target_value_b, reactive_roi_40, friendly_reach_b, keepability_b,
            mass_soon_b, imminence_b,
        ], dim=-1)  # (N, MO, P, 22)

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

        SOURCE SELECTION (2026-06-15): when more than MAX_OWNED planets are owned, the 16
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

        Mirrors action_mask._target_intercept_angle (the aim-benchmark-validated
        ~95% aimer) so TRAINING and INFERENCE aim identically: predict the target
        from its CURRENT orbit position, subtract the src+tgt surface gap, and run
        8 continuous (non-quantised) lead iterations. The old version over-led
        (centre-to-centre distance, integer-ceil ETA, 4 iters) — ~73% on the
        benchmark.
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
        if self.ship_bin_mode == "fraction":
            num_bins = len(FRACTION_BIN_VALUES)
            ship_bin = actions[:, :, 2].long().clamp(0, num_bins - 1)
            frac_t = torch.tensor(FRACTION_BIN_VALUES, dtype=torch.float32, device=self.device)
            frac = frac_t[ship_bin]                                # (N, MAX_OWNED)
            max_sendable = (src_ships - 1.0).clamp(min=1.0)
            ship_count = torch.round(frac * max_sendable).clamp(min=1.0)
        else:
            ship_bin = actions[:, :, 2].long().clamp(0, NUM_SHIP_BINS - 1)
            ship_counts_t = torch.tensor(SHIP_COUNTS, dtype=torch.float32, device=self.device)
            ship_count = ship_counts_t[ship_bin]                  # (N, MAX_OWNED)

        # Angle-bin mode uses BIN center (matches actions_from_policy).
        # Target mode executes the target head by converting target_idx to an
        # intercept angle while keeping angle_bin in storage for compatibility.
        angle = (angle_bin.float() + 0.5) * ANGLE_BIN_WIDTH       # (N, MAX_OWNED)
        target_valid = torch.ones_like(fire, dtype=torch.bool)
        if self.action_decode == "target" and actions.shape[-1] >= 4:
            raw_target_idx = actions[:, :, 3].long()
            use_target_decode = raw_target_idx >= 0
            target_idx = raw_target_idx.clamp(0, self.planets.shape[1] - 1)
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
                for slot in range(rec.shape[1]):
                    m = rec[:, slot]
                    if bool(m.any()):
                        ni = m.nonzero(as_tuple=True)[0]
                        cd[ni, owned_idx[ni, slot], target_idx[ni, slot]] = step_now[ni]
                self.reinf_cd[:, owner_id] = cd
            # #2 Per-ship transit cost: accumulate ships sent to own planets this step
            # for the launching player; the penalty is applied to the reward in step().
            # Counts only launches that actually fire (post-floor-veto).
            if self.reinforce_cost > 0.0:
                reinforce_ships = (ship_count * (can_fire & is_reinforce).float()).sum(dim=1)  # (N,)
                self._reinforce_ships[:, owner_id] = self._reinforce_ships[:, owner_id] + reinforce_ships
            # reinforce_rate metric: per-(env,player) counts of realized launches (post
            # floor-veto) and how many were reinforcement. train_torch combines these with
            # train_mask → the current policy's reinforce_rate (target 0.4-0.6, Vadasz 0.57).
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

        # SUFFICIENT-COMMIT MASK: veto an attack launch (enemy/neutral target) whose ship
        # count can't beat the target's current defense × factor → fragments impossible by
        # construction, forcing concentration. Reinforces (own targets) are untouched —
        # they have their own garrison-floor discipline. Pure veto, no reward tax.
        if (self.sufficient_commit_factor > 0.0 and self.action_decode == "target"
                and actions.shape[-1] >= 4):
            is_attack = use_target_decode & (target_owner != owner_id)
            insufficient = is_attack & (ship_count <= target_ships * self.sufficient_commit_factor)
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

        # Capture-utility reward: mark newly captured planets that became useful by
        # launching a REAL emitted attack (not reinforce, not vetoed, not slot-starved).
        if (self.capture_utility_active and self.action_decode == "target"
                and actions.shape[-1] >= 4):
            emitted_attack = target_valid & use_target_decode & (target_owner != owner_id)
            for slot in range(emitted_attack.shape[1]):
                m = emitted_attack[:, slot]
                if bool(m.any()):
                    ni = m.nonzero(as_tuple=True)[0]
                    self.cu_used_attack[ni, owned_idx[ni, slot]] = True

        # Per-env IDs for new fleets: next_fleet_id + 0, 1, 2, ... within env.
        new_id_within_env = (target_valid.long().cumsum(dim=1) - 1).clamp(min=0)
        new_ids = self.next_fleet_id.unsqueeze(1) + new_id_within_env  # (N, MAX_OWNED)

        # Vectorized scatter into fleets tensor — advanced indexing in one shot.
        env_arange = torch.arange(N, device=self.device).unsqueeze(1).expand(N, MAX_OWNED)
        flat_env = env_arange[target_valid]                            # (K,)
        flat_slot = target_slot[target_valid]                          # (K,)
        self.fleets[flat_env, flat_slot, 0] = new_ids[target_valid].float()
        self.fleets[flat_env, flat_slot, 1] = float(owner_id)
        self.fleets[flat_env, flat_slot, 2] = start_x[target_valid]
        self.fleets[flat_env, flat_slot, 3] = start_y[target_valid]
        self.fleets[flat_env, flat_slot, 4] = angle[target_valid]
        self.fleets[flat_env, flat_slot, 5] = src[..., 0][target_valid]
        self.fleets[flat_env, flat_slot, 6] = ship_count[target_valid]
        self.fleet_alive[flat_env, flat_slot] = True
        self.next_fleet_id = self.next_fleet_id + target_valid.long().sum(dim=1)

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
        # Lever A: count of decisive-mass crossings credited this rollout (per env, per player).
        self._decisive_credit = torch.zeros(
            self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        # dm_* GAP diagnostic: phase-split (early/mid/late) per-(env,player) sums over the rollout,
        # accumulated every step (when decisive_diag) from the EXACT reward floor so the diag and
        # the Lever-A reward can never drift. Read train_frac-weighted in train_torch.
        z3 = lambda: torch.zeros(self.num_envs, self.num_players, 3,
                                 dtype=torch.float32, device=self.device)
        self._dm_targets = z3()
        self._dm_cross = z3()
        self._dm_ratio_sum = z3()
        self._dm_gap_sum = z3()
        self._dm_overkill_sum = z3()
        self._dm_nearmiss = z3()

    # ---------------------------------------------------------------------
    # Step — pure tensor ops, runs all N envs in one pass.
    # Phase 2 scope: orbital motion + collision/combat + action processing.
    # ---------------------------------------------------------------------

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
            0, flat_idx[combat_mask].view(-1),
            ships_contrib[combat_mask].view(-1),
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
        if self.shaping_coef != 0.0:
            material = self._compute_material()
            material_delta = material - self.prev_material
            if self.num_players == 2:
                delta = material_delta[:, 0] - material_delta[:, 1]
                shaping_rewards = torch.stack([delta, -delta], dim=1)
            else:
                others = material_delta.sum(dim=1, keepdim=True) - material_delta
                shaping_rewards = material_delta - others / max(self.num_players - 1, 1)
            shaping_rewards = self.shaping_coef * torch.tanh(shaping_rewards / 50.0)
            terminal_rewards = terminal_rewards + shaping_rewards
            self.prev_material = material
        # Expansion shaping: potential-based reward on the change in owned-production
        # lead. Dense per-step signal for winning the planet/economy race (the thing
        # that decides snowball games). Telescopes, so passive play nets ~0.
        # Defense shaping: per-step PENALTY for losing owned production (a planet
        # captured from us). Targets the consolidation gap — agent grabs planets but
        # won't hold/reinforce them (reinforce_rate ~0.05). Asymmetric: only the
        # negative (loss) side, so it rewards HOLDING, distinct from expansion's GRAB.
        if self.expansion_coef != 0.0 or self.defense_coef != 0.0:
            production = self._compute_production()
            prod_delta = production - self.prev_production
            if self.expansion_coef != 0.0:
                if self.num_players == 2:
                    d = prod_delta[:, 0] - prod_delta[:, 1]
                    expansion_rewards = torch.stack([d, -d], dim=1)
                else:
                    others = prod_delta.sum(dim=1, keepdim=True) - prod_delta
                    expansion_rewards = prod_delta - others / max(self.num_players - 1, 1)
                terminal_rewards = terminal_rewards + self.expansion_coef * expansion_rewards
            if self.defense_coef != 0.0:
                # penalize each player's own production lost this step (clamp to losses)
                prod_lost = (-prod_delta).clamp(min=0.0)   # (N, num_players)
                terminal_rewards = terminal_rewards - self.defense_coef * prod_lost
        # Delta-capture shaping: time-decayed reward for CAPTURING planets (delta in owned
        # count), NOT for holding them. Fires as a spike when a planet changes hands.
        # Rev30 capture reward: symmetric delta + exponential decay with permanent floor.
        #
        # Key changes from Rev28:
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
        # Production-share capture reward (unified term): symmetric, capture-time-anchored,
        # value-weighted. r[pl] = coef · Σ_planets [ Δown(pl)·decay_anchor · prod/total_board ].
        # gain anchored to NOW (decay at current step), loss anchored to the LOSING owner's
        # capture_time → a capture and its eventual loss cancel exactly (no farm; drives holding).
        if self.prod_share_coef != 0.0:
            terminal_rewards = self._prod_share_bonus(terminal_rewards)
        # Consolidation bonus: ONE-TIME +coef when a net-new CAPTURED planet survives K steps.
        if self.consolidation_coef != 0.0:
            terminal_rewards = self._consolidation_bonus(terminal_rewards)
        if self.capture_utility_active:
            terminal_rewards = self._capture_utility_bonus(terminal_rewards)
        if self.decisive_mass_coef != 0.0 or self.decisive_diag:
            dm_fields = self._decisive_mass_fields()
            if self.decisive_mass_coef != 0.0:
                terminal_rewards = self._decisive_mass_bonus(terminal_rewards, dm_fields)
            if self.decisive_diag and self._dm_targets is not None:
                self._accumulate_decisive_diag(*dm_fields)
        # Reinforcement transit cost (#2): price the ships each player sent to its own
        # planets this step. Costless reinforcement floods (rev56, ~30× launch volume);
        # this prunes the wasteful tail. Calibrate reinforce_cost so reinforce_rate
        # settles ~0.4-0.6 (Vadasz-like), not 0 (over-suppressed) and not 0.8 (flood).
        if self.allow_reinforce and self.reinforce_cost > 0.0:
            terminal_rewards = terminal_rewards - self.reinforce_cost * self._reinforce_ships
        # 12. Auto-reset done envs in-place — must come AFTER capturing rewards
        if done.any():
            self._auto_reset(done)
        # Refresh prev_production / prev_owned AFTER auto-reset so done envs telescope
        # from their fresh post-reset state (avoids spurious spike on episode boundary).
        if self.expansion_coef != 0.0 or self.defense_coef != 0.0:
            self.prev_production = self._compute_production()
        if self.early_capture_coef != 0.0 and done.any():
            ec_owner = self.planets[:, :, 1].long()
            for pl in range(self.num_players):
                self.prev_owned[done, pl] = ((ec_owner[done] == pl) & self.planet_alive[done]).float().sum(dim=1)
        # Re-arm prod-share state for the fresh post-reset boards: initial ownership is already held
        # state, capture_time=0, recompute the fixed regular-planet normalizer.
        if self.prod_share_coef != 0.0 and done.any():
            ps_owner = self.planets[:, :, 1].long()
            self.prev_planet_owner[done] = torch.where(
                self.planet_alive[done],
                ps_owner[done],
                torch.full_like(ps_owner[done], -1),
            )
            self.capture_time[done] = 0
            self.total_board_prod[done] = self._prod_share_total_board_prod()[done]
        # Reset consolidation state for done envs to their fresh post-reset ownership (initial
        # owners are NOT captures), so the bonus telescopes cleanly across the episode boundary.
        if self.consolidation_coef != 0.0 and done.any():
            fresh_owner = self.planets[:, :, 1].long()
            self.cap_owner[done] = fresh_owner[done]
            self.cap_age[done] = 0
            self.cap_credited[done] = False
            self.cap_is_capture[done] = False
        if self.capture_utility_active and done.any():
            fresh_owner = self.planets[:, :, 1].long()
            self.cu_owner[done] = fresh_owner[done]
            self.cu_age[done] = 0
            self.cu_credited[done] = False
            self.cu_is_capture[done] = False
            self.cu_used_attack[done] = False
        # Re-arm the decisive-mass crossing detector for done envs: fresh boards have no
        # fleets (mass=0), so terminal-step sufficiency must NOT carry over — else a decisive
        # opening launch landing on a previously-armed index would be suppressed (uncredited).
        if self.decisive_mass_coef != 0.0 and done.any():
            self.prev_decisive_suff[done] = False
        # Reverse-edge cooldown: clear the per-edge history for fresh games (else a prior game's
        # reinforce edges mis-block the new board — same boundary bug class as decmass re-arm).
        if self.reverse_edge_cooldown > 0 and self.reinf_cd is not None and done.any():
            self.reinf_cd[done] = _REINF_CD_NEVER
        return self._state_dict(), terminal_rewards, done

    # ---------------------------------------------------------------------
    # Termination logic — matches fast_env._maybe_terminate
    # Episode ends if step_count >= episode_steps - 1 OR <= 1 player alive
    # (alive = has any owned planet or any in-flight fleet).
    # Reward: per-env, per-player. +1 if that player has the max score AND
    # max > 0, else -1.
    # ---------------------------------------------------------------------

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
        scenario_success = torch.zeros(N, dtype=torch.bool, device=self.device)
        scenario_failure = torch.zeros(N, dtype=torch.bool, device=self.device)
        if self.scenario_id is not None:
            scenario_active = self.scenario_id != _SCENARIO_OFF
            target_idx = self.scenario_target.clamp(0, self.planets.shape[1] - 1)
            target_owner = owner_p.gather(1, target_idx.view(-1, 1)).squeeze(1)
            adv = self.scenario_adv_player
            deadline = self.step_count >= self.scenario_done_step.clamp(min=1)
            attack_scenario = (
                (self.scenario_id == _SCENARIO_AGG_ATTACK)
                | (self.scenario_id == _SCENARIO_STAGE_ATTACK)
            )
            hold_scenario = self.scenario_id == _SCENARIO_HOLD_UNDER_PEEL
            scenario_success = scenario_active & (
                (attack_scenario & (target_owner == adv))
                | (hold_scenario & deadline & (target_owner == adv))
            )
            scenario_failure = scenario_active & (
                (attack_scenario & deadline & (target_owner != adv))
                | (hold_scenario & (target_owner != adv))
            )
        scenario_done = (scenario_success | scenario_failure) & ~self.done
        newly_done = (time_up | few_left | scenario_done) & ~self.done
        if self._last_scenario_id is not None:
            self._last_scenario_id.zero_()
            self._last_scenario_success.zero_()
            self._last_scenario_id[scenario_done] = self.scenario_id[scenario_done]
            self._last_scenario_success[scenario_done] = scenario_success[scenario_done]

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
        # which already carries material/expansion/early-capture/etc. shaping by the
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
        # Time-to-victory velocity bonus: winners get extra reward for winning early.
        # reward_win += (episode_steps - T) / episode_steps * speed_coef
        # Gradient constantly pressures the agent to close games faster.
        if self.speed_coef != 0.0:
            # step_count is (N,) — clamp to episode_steps to handle edge cases
            t = self.step_count.float().clamp(max=self.episode_steps)
            velocity = (self.episode_steps - t) / self.episode_steps  # (N,) in [0, 1]
            speed_bonus = self.speed_coef * velocity.unsqueeze(1)     # (N, 1)
            rewards = torch.where(wins, rewards + speed_bonus, rewards)
        if scenario_done.any():
            env_idx = torch.where(scenario_done)[0]
            rewards[env_idx] = -1.0
            adv_idx = self.scenario_adv_player[env_idx]
            opp_idx = 1 - adv_idx
            succ = scenario_success[env_idx]
            rewards[env_idx, adv_idx] = torch.where(
                succ, torch.ones_like(rewards[env_idx, adv_idx]), -torch.ones_like(rewards[env_idx, adv_idx])
            )
            rewards[env_idx, opp_idx] = torch.where(
                succ, -torch.ones_like(rewards[env_idx, opp_idx]), torch.ones_like(rewards[env_idx, opp_idx])
            )
            self._last_wins[env_idx] = False
            self._last_wins[env_idx, adv_idx] = succ
            self._last_wins[env_idx, opp_idx] = ~succ
        # Only return rewards for newly-done envs; zero otherwise
        rewards = rewards * newly_done.unsqueeze(1).float()
        self.rewards = torch.where(newly_done.unsqueeze(1), rewards, self.rewards)
        self.done = self.done | newly_done
        return rewards, newly_done

    # ---------------------------------------------------------------------
    # Auto-reset: pick new seeds for done envs and regenerate their state.
    # Keeps the running training loop simple — no need for caller-side resets.
    # ---------------------------------------------------------------------

    def _ssdr_warmup(self, env_indices: list):
        """Fast-forward a random subset of envs by 1..ssdr_max_steps random steps.

        Both players fire randomly so the learner wakes in a messy, asymmetric
        mid-game state — shattering the symmetric-start passive Nash.
        Called after reset() and after each auto-reset.
        """
        if not env_indices or self.ssdr_frac <= 0.0 or self._ssdr_active:
            return
        self._ssdr_active = True
        try:
            self._ssdr_warmup_inner(env_indices)
        finally:
            self._ssdr_active = False

    def _ssdr_warmup_inner(self, env_indices: list):
        # Pick which envs get warmed up this time
        warmup_envs = [i for i in env_indices if random.random() < self.ssdr_frac]
        if not warmup_envs:
            return

        # For each chosen env, sample how many steps to fast-forward
        steps_per_env = {i: random.randint(1, self.ssdr_max_steps) for i in warmup_envs}
        max_steps = max(steps_per_env.values())

        # Mask: which envs still need more warmup steps at each timestep t
        warmup_set = set(warmup_envs)
        # Build fire mask: 1 for warmup envs, 0 for non-warmup (they get no-op)
        warmup_mask = torch.tensor(
            [1 if i in warmup_set else 0 for i in range(self.num_envs)],
            dtype=torch.long, device=self.device
        ).unsqueeze(1)  # (N, 1)

        for t in range(max_steps):
            # Only fire for envs that still have warmup steps remaining
            active_mask = torch.tensor(
                [1 if (i in warmup_set and t < steps_per_env[i]) else 0
                 for i in range(self.num_envs)],
                dtype=torch.long, device=self.device
            ).unsqueeze(1)  # (N, 1)

            actions = {}
            for pid in range(self.num_players):
                _, slot_valid = self.owned_indices_for(pid)
                fire = (torch.rand(self.num_envs, MAX_OWNED, device=self.device) < 0.6).long()
                fire = fire * slot_valid.long() * active_mask  # zero fire for non-warmup envs
                angle_bin = torch.randint(0, NUM_ANGLE_BINS, (self.num_envs, MAX_OWNED), device=self.device)
                ship_bin = torch.randint(10, 15, (self.num_envs, MAX_OWNED), device=self.device)
                actions[pid] = torch.stack([fire, angle_bin, ship_bin], dim=-1)

            self.step(actions)

        # After warmup: reset step_count to 0 for non-warmup envs (physics advanced
        # but no fleets launched — equivalent to a different random orbital offset)
        for i in range(self.num_envs):
            if i not in warmup_set:
                self.step_count[i] = 0
        # Also reset step_count to actual warmup length for warmup envs
        for i, n in steps_per_env.items():
            self.step_count[i] = n

    def _auto_reset(self, done_mask: torch.Tensor):
        """Re-generate state for envs where done_mask is True."""
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
            self._scale_neutrals(pad, n)
            alive = np.zeros(MAX_PLANETS, dtype=bool)
            alive[:n] = True
            num_groups = n // 4
            if num_groups > 0:
                home_group = init_rng.randint(0, num_groups - 1)
                base = home_group * 4
                if self.num_players == 2:
                    pad[base, 1] = 0;     pad[base, 5] = 10
                    pad[base + 3, 1] = 1; pad[base + 3, 5] = 10
                    # SSDR asymmetric planet assignment — self-play envs only
                    if self._ssdr_active_for(env_i) and random.random() < self.ssdr_frac:
                        k = random.randint(1, max(1, self.ssdr_max_steps))
                        neutral_idx = [i for i in range(n)
                                       if pad[i, 1] == -1 and i != base and i != base + 3]
                        random.shuffle(neutral_idx)
                        for ni in neutral_idx[:k]:
                            pad[ni, 1] = 1
                            pad[ni, 5] = max(10, int(pad[ni, 6] * 3))
                    # Self-boost (handicapped-real-planner): grant OUR seat extra planets
                    self._maybe_self_boost(pad, n, base, env_i)
                elif self.num_players == 4:
                    for j in range(4):
                        pad[base + j, 1] = j; pad[base + j, 5] = 10
            pad[n:, 1] = -1
            sid, adv, tgt, deadline = self._apply_scenario(
                pad, alive, random.Random(f"orbit-wars-scenario-{seed}")
            )

            self.planets[env_i] = torch.from_numpy(pad).to(self.device)
            self.init_planets[env_i] = self.planets[env_i].clone()
            self.planet_alive[env_i] = torch.from_numpy(alive).to(self.device)
            self.fleets[env_i] = 0
            self.fleet_alive[env_i] = False
            self.scenario_id[env_i] = sid
            self.scenario_adv_player[env_i] = adv
            self.scenario_target[env_i] = tgt
            self.scenario_done_step[env_i] = deadline
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
            for env_i in done_idx:
                self._scenario_fleet_seed(env_i)
        self.prev_material[done_mask] = self._compute_material()[done_mask]


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
    but with ONE GPU->CPU copy per field (over all ids) instead of ~8 tiny syncs per env.
    At ~96 external-opponent envs/step that collapses hundreds of per-step syncs into a
    handful — the dominant per-step transport tax. Pure transport: identical bytes out
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
