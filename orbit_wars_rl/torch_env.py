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
    # Phase 0 scope: orbital motion + fleet movement only (no launch/combat).
    # ---------------------------------------------------------------------

    def step(self, actions=None) -> dict:
        """Advance all N envs by one tick.

        Phase 0: actions are ignored. Just runs orbital motion + fleet movement.
        """
        # 1. Production: planet[5] += planet[6] for owned planets (owner != -1)
        owner = self.planets[:, :, 1]
        prod  = self.planets[:, :, 6]
        is_owned = (owner != -1) & self.planet_alive
        self.planets[:, :, 5] = self.planets[:, :, 5] + prod * is_owned.float()

        # 2. Orbital motion — update planet positions
        # current_angle = initial_angle + angular_velocity * step
        # new_x = CENTER + orbital_r * cos(current_angle)  if is_orbiting else x
        step_f = self.step_count.float().unsqueeze(-1)  # (N, 1)
        ang_vel = self.angular_velocity.unsqueeze(-1)   # (N, 1)
        cur_angle = self._planet_initial_angle + ang_vel * step_f  # (N, P)

        new_x = CENTER + self._planet_orbital_r * torch.cos(cur_angle)
        new_y = CENTER + self._planet_orbital_r * torch.sin(cur_angle)
        is_orb = self._planet_is_orbiting
        self.planets[:, :, 2] = torch.where(is_orb, new_x, self.planets[:, :, 2])
        self.planets[:, :, 3] = torch.where(is_orb, new_y, self.planets[:, :, 3])

        # 3. Fleet movement — fleet[2,3] += speed * cos/sin(angle)
        fleet_angle = self.fleets[:, :, 4]
        fleet_ships = self.fleets[:, :, 6]
        speed = _ship_speed(fleet_ships)  # (N, F)
        speed = speed * self.fleet_alive.float()  # zero out dead fleets
        self.fleets[:, :, 2] = self.fleets[:, :, 2] + torch.cos(fleet_angle) * speed
        self.fleets[:, :, 3] = self.fleets[:, :, 3] + torch.sin(fleet_angle) * speed

        # 4. Mark out-of-bounds fleets as dead (no collision yet — Phase 1)
        in_bounds = (
            (self.fleets[:, :, 2] >= 0) & (self.fleets[:, :, 2] <= BOARD_SIZE)
            & (self.fleets[:, :, 3] >= 0) & (self.fleets[:, :, 3] <= BOARD_SIZE)
        )
        self.fleet_alive = self.fleet_alive & in_bounds

        # 5. Advance step
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
