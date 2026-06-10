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

# Tensor sizes (worst-case bounds from kaggle env)
MAX_PLANETS = 48
MAX_FLEETS = 256
MAX_OWNED = 16

# Discrete action bins (match action_mask.py / model.py)
NUM_ANGLE_BINS = 144
ANGLE_BIN_WIDTH = 2 * math.pi / NUM_ANGLE_BINS
NUM_SHIP_BINS = 32
SHIP_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 26, 30, 35, 42, 50, 60, 72, 86, 102, 122, 145, 173, 206, 245, 290, 350, 420]
# Fraction-mode decode (10 bins): bin i → (i+1)/10 * src_ships
FRACTION_BIN_VALUES = [(i + 1) / 10 for i in range(10)]


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
        action_decode: str = "angle",
        win_margin_coeff: float = 0.0,
        shaping_coef: float = 0.0,
        expansion_coef: float = 0.0,
        defense_coef: float = 0.0,
        early_capture_coef: float = 0.0,
        early_capture_steps: int = 100,
        first_strike_steps: int = 0,
        first_strike_mult: float = 2.0,
        speed_coef: float = 0.0,
        handicap_frac: float = 0.0,
        handicap_ships: int = 5,
        ssdr_frac: float = 0.0,
        ssdr_max_steps: int = 20,
        allow_reinforce: bool = False,
        reinforce_garrison_floor: float = 0.0,
        reinforce_cost: float = 0.0,
        reinforce_gate_min_planets: int = 0,
        reinforce_forward_only: bool = False,
    ):
        self.num_envs = num_envs
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
        # reinforce_rate metric accumulators (N, num_players), allocated/zeroed per
        # rollout via reset_reinforce_stats(). None = not collecting (no overhead).
        self._reinforce_launch_count = None
        self._fire_launch_count = None
        # target-owner share diagnostic: launches whose target is a NEUTRAL planet
        # (own = _reinforce_launch_count; enemy = fire − own − neutral). Phase-2
        # target-head health (is the "where" head selective or uniform?).
        self._neutral_launch_count = None
        self.num_players = num_players
        self.episode_steps = episode_steps
        self.device = torch.device(device)
        # See ModelConfig.ship_bin_mode. "absolute" uses SHIP_COUNTS lookup;
        # "fraction" uses round(FRAC_VALUES[bin] * src_ships).
        self.ship_bin_mode = ship_bin_mode
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
        # Asymmetric Planet SSDR: with probability ssdr_frac, grant opponent 1..ssdr_max_steps
        # extra neutral planets at reset. No random play, no fleet explosion.
        # Breaks symmetric-start Nash cleanly.
        #
        # ssdr_self_only_mask: bool tensor (N,) set by training loop each rollout.
        # True = self-play env (SSDR active), False = pool env (symmetric start).
        # If None, SSDR applies to all envs.
        self._ssdr_self_mask: torch.Tensor | None = None  # set via set_ssdr_mask()

        # State tensors — allocated in reset()
        self.planets: torch.Tensor = None       # (N, P, 7)
        self.init_planets: torch.Tensor = None  # (N, P, 7) — pristine snapshot
        self.planet_alive: torch.Tensor = None  # (N, P) bool
        self.fleets: torch.Tensor = None        # (N, F, 7)
        self.fleet_alive: torch.Tensor = None   # (N, F) bool
        self.step_count: torch.Tensor = None    # (N,) long
        self.angular_velocity: torch.Tensor = None  # (N,) float
        self.next_fleet_id: torch.Tensor = None     # (N,) long
        self.done: torch.Tensor = None              # (N,) bool
        self.rewards: torch.Tensor = None           # (N, num_players) float
        self.prev_material: torch.Tensor = None     # (N, num_players) float
        self.prev_production: torch.Tensor = None   # (N, num_players) float — owned production for expansion shaping
        self.prev_owned: torch.Tensor = None        # (N, num_players) float — owned planet count for delta-capture shaping
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

    def _ssdr_active_for(self, env_i: int) -> bool:
        """Return True if SSDR should apply to env index env_i."""
        if self.ssdr_frac <= 0.0:
            return False
        if self._ssdr_self_mask is None:
            return True  # no mask set → apply to all
        return bool(self._ssdr_self_mask[env_i].item())

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
        self.prev_material = self._compute_material()
        self.prev_production = self._compute_production()
        owner_p = self.planets[:, :, 1].long()
        self.prev_owned = torch.zeros(self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        for pl in range(self.num_players):
            self.prev_owned[:, pl] = ((owner_p == pl) & self.planet_alive).float().sum(dim=1)
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
        big = torch.full_like(dpp, BOARD_SIZE)
        dpp_masked = torch.where(mine_col, dpp, big)
        min_owned_dist = dpp_masked.min(dim=2).values  # (N, P)
        any_mine = is_mine.any(dim=1, keepdim=True)
        min_owned_dist = torch.where(any_mine.expand_as(min_owned_dist),
                                      min_owned_dist, torch.zeros_like(min_owned_dist))

        # is_home heuristic
        is_home = is_mine & (ships <= 10 + prod * 5) & (ships >= 10 - prod)

        # is_comet — skip for now (comets not implemented in Phase 3a)
        is_comet = torch.zeros_like(is_orbiting)

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

        gf = torch.stack([
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
        ], dim=1)  # (N, 11)

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
        pairwise = self._compute_pairwise(
            planets=planets, planet_alive=planet_alive, P=P,
            owned_idx=owned_idx, slot_valid=slot_valid, player=player,
            enemy_contest=enemy_contest,
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
                          enemy_contest=None):
        """Vectorized counterpart of features.compute_pairwise_features().

        Returns (N, MAX_OWNED, P, 15) float32 on self.device. Channel order:
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

        # Enemy contest feature (ch 14): broadcast (N, P) → (N, MO, P)
        if enemy_contest is not None:
            contest_b = (enemy_contest / 100.0).clamp(max=5.0).unsqueeze(1).expand(-1, MO, -1)
        else:
            contest_b = torch.zeros(N, MO, P, device=device)

        # Stack channels
        out = torch.stack([
            sin_a, cos_a, dist / BOARD_SIZE, 1.0 / (eta + 1.0),
            sun_safe, is_mine_b, is_enemy_b, is_neutral_b, prod_b, valid_b,
            ships_at_arr, cap_gap, roi_20, roi_50, contest_b,
        ], dim=-1)  # (N, MO, P, 15)

        # Zero out invalid owned slots AND invalid target planets (match kaggle path)
        slot_valid_b = slot_valid.unsqueeze(-1).unsqueeze(-1).float()    # (N, MO, 1, 1)
        target_valid_b = alive_t.float().expand(-1, MO, -1).unsqueeze(-1)  # (N, MO, P, 1)
        out = out * slot_valid_b * target_valid_b
        return out

    # ---------------------------------------------------------------------
    # Owned-planet indices per player — vectorized.
    # For each env, returns the planet-array indices of the first MAX_OWNED
    # planets where owner == player. Pad with 0 and use a `slot_valid` mask.
    # ---------------------------------------------------------------------

    def owned_indices_for(self, player: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (owned_idx: (N, MAX_OWNED) long, slot_valid: (N, MAX_OWNED) bool).

        Single-pass topk: planets are scored by their array index (lowest first)
        with non-mine slots assigned a sentinel > P. Top-K smallest = first K mine.
        """
        owner = self.planets[:, :, 1]
        is_mine = (owner.long() == player) & self.planet_alive          # (N, P)
        N, P = is_mine.shape
        SENTINEL = P + 1
        idx_grid = torch.arange(P, device=self.device).expand(N, P)
        scores = torch.where(is_mine, idx_grid, torch.full_like(idx_grid, SENTINEL))
        owned_idx, _ = torch.topk(scores, MAX_OWNED, dim=1, largest=False)  # (N, MAX_OWNED)
        slot_valid = owned_idx < P
        owned_idx = owned_idx.clamp(max=P - 1)  # keep gather-safe
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

    def _apply_actions(self, actions: torch.Tensor, owner_id: int):
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

        # Validate: planet still owned by this player AND has enough ships
        valid_owner = (src_owner == owner_id) & slot_valid
        valid_ships = src_ships >= ship_count
        can_fire = fire & valid_owner & valid_ships & target_valid & (ship_count > 0)  # (N, MAX_OWNED)

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
                self._fire_launch_count[:, owner_id] += can_fire.sum(dim=1).float()
                self._reinforce_launch_count[:, owner_id] += (can_fire & is_reinforce).sum(dim=1).float()
                is_neutral = use_target_decode & (target_owner < 0)  # neutral planet owner = -1
                self._neutral_launch_count[:, owner_id] += (can_fire & is_neutral).sum(dim=1).float()

        # Compute launch positions (just outside planet radius along angle)
        start_x = src_x + torch.cos(angle) * (src_r + 0.1)
        start_y = src_y + torch.sin(angle) * (src_r + 0.1)

        # Debit ships from source planets. Use scatter_add with negative values.
        # Multiple fires from same planet aren't possible (one slot per planet),
        # so scatter is well-defined.
        debit = ship_count * can_fire.float()                     # (N, MAX_OWNED)
        ships_col = self.planets[:, :, 5]
        new_ships = ships_col.scatter_add(1, owned_idx, -debit)
        self.planets[:, :, 5] = new_ships

        # Find first MAX_OWNED dead fleet slots per env via topk-smallest trick.
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
        target_valid = rank_has.gather(1, fire_rank) & can_fire        # (N, MAX_OWNED)

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

    # ---------------------------------------------------------------------
    # Step — pure tensor ops, runs all N envs in one pass.
    # Phase 2 scope: orbital motion + collision/combat + action processing.
    # ---------------------------------------------------------------------

    def step(self, actions=None) -> dict:
        """Advance all N envs by one tick.

        actions: optional dict {player_id: (N, MAX_OWNED, 3) tensor}.
                 Each player's fleets are launched before physics.
        """
        # Per-step buffer for the reinforcement transit cost (#2): ships each player
        # sent to its own planets this step. Zeroed before launches accumulate into it.
        if self.allow_reinforce and self.reinforce_cost > 0.0:
            self._reinforce_ships = torch.zeros(
                self.num_envs, self.num_players, dtype=torch.float32, device=self.device)
        if actions is not None:
            for pid, act in actions.items():
                self._apply_actions(act, pid)

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
    initial_planets = [
        [int(ip[i, 0]), int(ip[i, 1]),
         float(ip[i, 2]), float(ip[i, 3]), float(ip[i, 4]),
         float(ip[i, 5]), float(ip[i, 6])]
        for i in range(MAX_PLANETS) if a[i]
    ]
    return {
        "step": int(env.step_count[env_idx].item()),
        "player": player,
        "planets": planets,
        "fleets": fleets,
        "angular_velocity": float(env.angular_velocity[env_idx].item()),
        "initial_planets": initial_planets,
        "comet_planet_ids": [],
    }
