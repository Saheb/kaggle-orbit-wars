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
    ):
        self.num_envs = num_envs
        self.num_players = num_players
        self.episode_steps = episode_steps
        self.device = torch.device(device)

        # State tensors — allocated in reset()
        self.planets: torch.Tensor = None       # (N, P, 7)
        self.init_planets: torch.Tensor = None  # (N, P, 7) — pristine snapshot
        self.planet_alive: torch.Tensor = None  # (N, P) bool
        self.fleets: torch.Tensor = None        # (N, F, 7)
        self.fleet_alive: torch.Tensor = None   # (N, F) bool
        self.step_count: torch.Tensor = None    # (N,) long
        self.angular_velocity: torch.Tensor = None  # (N,) float
        self.next_fleet_id: torch.Tensor = None     # (N,) long

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

        for seed in seeds:
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

        self._precompute_orbital_params()
        return self._state_dict()

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
    # Step — pure tensor ops, runs all N envs in one pass.
    # Phase 1 scope: orbital motion + fleet movement + collision/combat.
    # ---------------------------------------------------------------------

    def step(self, actions=None) -> dict:
        """Advance all N envs by one tick.

        Phase 1: actions still ignored (no launch yet). Adds collision and combat.
        """
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
        return self._state_dict()


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
    return {
        "step": int(env.step_count[env_idx].item()),
        "player": player,
        "planets": planets,
        "fleets": fleets,
        "angular_velocity": float(env.angular_velocity[env_idx].item()),
        "initial_planets": [],
        "comet_planet_ids": [],
    }
