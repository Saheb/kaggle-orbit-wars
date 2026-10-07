"""Opponent pool for self-play training with PFSP sampling.

A pool holds past-self checkpoints and pinned RL champions (e.g. the anchor). Training
rollouts can sample from the pool to diversify the policy's training distribution beyond
current-vs-current self-play, which prevents narrow-equilibrium cycling. (External .py
heuristic opponents were removed in the 2026-10 cleanup — tag pre-cleanup-2026-10.)

PFSP (Prioritized Fictitious Self-Play): opponents are sampled with weight
``(1 - win_rate_against_them) ** alpha``. As you master an opponent, its weight
shrinks → you stop training against it.

Self-checkpoint storage uses FIFO eviction when the pool exceeds
``max_self_members``, keeping a rolling window of past selves.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PoolMember:
    name: str
    kind: str            # 'self' (the only kind since the 2026-10 cleanup)
    state_dict: Optional[dict] = None
    step_saved: int = 0                      # training step when added
    pinned: bool = False                      # fixed RL opponent (seeded champion): never FIFO-evicted
    wins: int = 0
    losses: int = 0
    draws: int = 0
    # EMA win-rate — updated per game, decoupled from the lifetime win/loss counters.
    # Starts at 0.5 (uninformative). Used for opponents that LIVE THE WHOLE RUN, where a
    # lifetime rate goes stale as the policy improves (early-run losses averaged in forever):
    # pinned RL champions (see `uses_ema`). Transient self-snapshots are
    # FIFO-evicted within a bounded window, so their lifetime rate stays fresh and is used.
    ema_win_rate: float = 0.5
    ema_games: int = 0   # number of EMA updates (games) so far

    @property
    def n_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def uses_ema(self) -> bool:
        """Whether PFSP reads the EMA (recent) win-rate vs the lifetime rate. True for
        long-lived FIXED opponents (pinned RL champions) whose lifetime rate goes stale as the
        policy improves; False for transient (evictable) self-snapshots."""
        return self.pinned

    @property
    def win_rate(self) -> float:
        # Win-rate of *current model* vs this opponent (so high = opponent mastered).
        if self.n_games == 0:
            return 0.5  # uninformative prior
        return self.wins / self.n_games


class OpponentPool:
    def __init__(self, max_self_members: int = 20, pfsp_alpha: float = 2.0,
                 pfsp_min_games: int = 30, ema_alpha: float = 0.01):
        self.members: list[PoolMember] = []
        self.max_self_members = max_self_members
        self.pfsp_alpha = pfsp_alpha
        # Minimum games before trusting win-rate for PFSP weighting.
        # Until this threshold, wr=0.5 is used so early lucky streaks don't
        # sand-bag an opponent (e.g. Hellburner getting 0.003 weight after 17 games).
        self.pfsp_min_games = pfsp_min_games
        # EMA smoothing for long-lived (pinned) opponents' win-rate. A lifetime win/loss count
        # goes stale once the denominator is large — early-training wins dilute recent
        # performance. ema_alpha ≈ 0.01 keeps an effective window of ~100 games.
        # Self-checkpoints still use lifetime win rate (fresh given their bounded lifespan).
        self.ema_alpha: float = float(ema_alpha)

    def __len__(self) -> int:
        return len(self.members)

    # ---- adding members ----------------------------------------------------

    def add_self_checkpoint(self, step: int, state_dict: dict) -> None:
        """Add a snapshot of current model. FIFO-evicts oldest self if over cap."""
        # Detach state-dict to CPU so we don't hold GPU memory hostage
        cpu_sd = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        self.members.append(PoolMember(
            name=f"self_step_{step}", kind="self",
            state_dict=cpu_sd, step_saved=step,
        ))
        # Evict oldest self if over cap (pinned champions are untouched)
        self_members = [m for m in self.members if m.kind == "self" and not m.pinned]
        if len(self_members) > self.max_self_members:
            oldest = min(self_members, key=lambda m: m.step_saved)
            self.members.remove(oldest)

    def add_pinned_rl(self, name: str, state_dict: dict) -> None:
        """Add a fixed RL champion as a never-evicted 'self'
        opponent. Runs through the same GPU 'self' forward path; pinned so organic
        self-snapshot FIFO never drops it. step_saved=-1 keeps it out of FIFO order."""
        cpu_sd = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        self.members.append(PoolMember(
            name=f"seed_{name}", kind="self", state_dict=cpu_sd,
            step_saved=-1, pinned=True,
        ))

    # ---- sampling ----------------------------------------------------------

    def sample(self, rng: Optional[random.Random] = None,
               pinned_fraction: Optional[float] = None) -> Optional[PoolMember]:
        """Sample a member by PFSP weight. Returns None if the pool is empty.

        ``pinned_fraction`` engages **ramp mode** (2-way split): a **pinned-RL slice** /
        PFSP over ORGANIC (non-pinned) selves. This pulls pinned RL champions out of PFSP
        into their own fixed ramped fraction — necessary because PFSP weight ``(1-wr)^α``
        *up-samples* an opponent you lose to, so a weak from-scratch policy would otherwise
        see a strong pinned opponent more often early. When ``pinned_fraction is None``,
        pinned members compete inside PFSP with the selves.
        Returns None when the chosen budget falls to PFSP but no organic snapshot
        exists yet (early from-scratch) — the caller then falls back to self-play.
        """
        if not self.members:
            return None
        r = rng or random

        if pinned_fraction is not None:
            # --- Ramp mode: pinned-RL / PFSP-over-organic (non-pinned selves) ---
            pinned = [m for m in self.members if m.pinned]
            organic = [m for m in self.members if m.kind == "self" and not m.pinned]
            roll = r.random()
            if pinned and roll < pinned_fraction:
                return r.choice(pinned)
            if not organic:
                # No organic snapshots yet (early from-scratch) → caller does self-play.
                return None
            candidates = organic
        else:
            # No fixed slices: PFSP over all members together.
            candidates = self.members

        weights = [self._pfsp_weight(m) for m in candidates]
        total = sum(weights)
        if total <= 0:
            return r.choice(candidates)
        return r.choices(candidates, weights=weights, k=1)[0]

    def _pfsp_weight(self, m: PoolMember) -> float:
        # Long-lived fixed opponents (pinned RL champions): EMA win-rate, so the
        # weight tracks RECENT performance and doesn't go stale as the policy improves.
        # Transient self-snapshots: lifetime win-rate (fresh given their bounded lifespan).
        # Either way, use the uninformative 0.5 prior until enough games to trust the estimate.
        if m.uses_ema:
            wr = 0.5 if m.ema_games < self.pfsp_min_games else m.ema_win_rate
        else:
            wr = 0.5 if m.n_games < self.pfsp_min_games else m.win_rate
        return max(1.0 - wr, 1e-6) ** self.pfsp_alpha

    # ---- bookkeeping -------------------------------------------------------

    def record_result(self, member: PoolMember, result: str) -> None:
        """result in {'win', 'loss', 'draw'} from *current model's* perspective."""
        if result == "win":   member.wins += 1
        elif result == "loss": member.losses += 1
        else:                  member.draws += 1
        # Update EMA win-rate for long-lived fixed opponents (pinned RL champions)
        # so PFSP stays responsive to recent performance rather than the full accumulated history.
        if member.uses_ema:
            win_val = 1.0 if result == "win" else 0.0
            member.ema_win_rate = (
                (1.0 - self.ema_alpha) * member.ema_win_rate + self.ema_alpha * win_val
            )
            member.ema_games += 1

    # ---- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist pool to disk so it survives spot interruption / restart."""
        import torch  # local import: avoid forcing torch on test-only imports
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self_members = [{
            "name": m.name, "wins": m.wins, "losses": m.losses, "draws": m.draws,
            "step_saved": m.step_saved, "state_dict": m.state_dict, "pinned": m.pinned,
            "ema_win_rate": m.ema_win_rate, "ema_games": m.ema_games,
        } for m in self.members]
        payload = {
            "self_members": self_members,
            "config": {
                "max_self_members": self.max_self_members,
                "pfsp_alpha": self.pfsp_alpha,
                "pfsp_min_games": self.pfsp_min_games,
                "ema_alpha": self.ema_alpha,
            },
        }
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @classmethod
    def load(cls, path: str) -> "OpponentPool":
        """Recreate a pool from a saved file. Pool files saved before the 2026-10 cleanup may
        also list external heuristic members and their config keys; those are ignored."""
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        cfg = data.get("config", {})
        pool = cls(
            max_self_members=cfg.get("max_self_members", 20),
            pfsp_alpha=cfg.get("pfsp_alpha", 2.0),
            pfsp_min_games=cfg.get("pfsp_min_games", 30),
            ema_alpha=cfg.get("ema_alpha", 0.01),
        )
        for m in data.get("self_members", []):
            pool.members.append(PoolMember(
                name=m["name"], kind="self",
                state_dict=m["state_dict"], step_saved=m["step_saved"],
                pinned=m.get("pinned", False),
                wins=m["wins"], losses=m["losses"], draws=m["draws"],
                ema_win_rate=m.get("ema_win_rate", 0.5), ema_games=m.get("ema_games", 0),
            ))
        if data.get("external_members"):
            print(f"  NOTE: ignoring {len(data['external_members'])} external heuristic pool "
                  f"member(s) (removed in the 2026-10 cleanup)")
        return pool

    # ---- diagnostics -------------------------------------------------------

    def summary(self, max_rows: int = 8) -> str:
        if not self.members:
            return "  (pool empty)"
        rows = sorted(self.members, key=lambda m: -self._pfsp_weight(m))[:max_rows]
        lines = [f"  pool size={len(self.members)}  alpha={self.pfsp_alpha}"]
        for m in rows:
            w = self._pfsp_weight(m)
            if m.uses_ema:
                # Show both EMA (recent) and lifetime win-rate so drift is visible.
                ema_str = f" ema_wr={m.ema_win_rate:.2f}(n={m.ema_games})"
                lines.append(
                    f"    {m.kind:20s} {m.name:30s} "
                    f"wr={m.win_rate:.2f}(n={m.n_games}){ema_str}  pfsp_w={w:.3f}"
                )
            else:
                lines.append(
                    f"    {m.kind:20s} {m.name:30s} "
                    f"wr={m.win_rate:.2f} (n={m.n_games})  pfsp_w={w:.3f}"
                )
        return "\n".join(lines)
