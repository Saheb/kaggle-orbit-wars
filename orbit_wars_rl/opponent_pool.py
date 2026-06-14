"""Opponent pool for self-play training with PFSP sampling.

A pool holds past-self checkpoints and (optionally) frozen external opponents
(.py heuristics like candidate_suneet_lb1200.py). Training rollouts can sample
from the pool to diversify the policy's training distribution beyond
current-vs-current self-play, which prevents narrow-equilibrium cycling.

PFSP (Prioritized Fictitious Self-Play): opponents are sampled with weight
``(1 - win_rate_against_them) ** alpha``. As you master an opponent, its weight
shrinks → you stop training against it. External opponents whose sustained
win-rate passes a "mastered" threshold are auto-evicted; self-checkpoints stay
and just get low sampling weight.

Self-checkpoint storage uses FIFO eviction when the pool exceeds
``max_self_members`` — keeps a rolling window of past selves. With a 1M-step
cadence and cap=20, the pool spans ~20M steps of training history.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class PoolMember:
    name: str
    kind: str            # 'self' | 'external_heuristic'
    state_dict: Optional[dict] = None        # for 'self'
    agent_fn: Optional[Callable] = None      # for 'external_heuristic'
    step_saved: int = 0                      # training step when added (self only)
    pinned: bool = False                      # fixed RL opponent (seeded champion): never FIFO-evicted
    wins: int = 0
    losses: int = 0
    draws: int = 0
    # EMA win-rate — updated per game, decoupled from the lifetime win/loss counters.
    # Starts at 0.5 (uninformative). Used for opponents that LIVE THE WHOLE RUN, where a
    # lifetime rate goes stale as the policy improves (early-run losses averaged in forever):
    # fixed externals AND pinned RL champions (see `uses_ema`). Transient self-snapshots are
    # FIFO-evicted within a bounded window, so their lifetime rate stays fresh and is used.
    ema_win_rate: float = 0.5
    ema_games: int = 0   # number of EMA updates (games) so far

    @property
    def n_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def uses_ema(self) -> bool:
        """Whether PFSP reads the EMA (recent) win-rate vs the lifetime rate. True for
        long-lived FIXED opponents — externals and pinned RL champions — whose lifetime
        rate goes stale as the policy improves; False for transient (evictable) self-snapshots."""
        return self.kind == "external_heuristic" or self.pinned

    @property
    def win_rate(self) -> float:
        # Win-rate of *current model* vs this opponent (so high = opponent mastered).
        if self.n_games == 0:
            return 0.5  # uninformative prior
        return self.wins / self.n_games


class OpponentPool:
    def __init__(self, max_self_members: int = 20, pfsp_alpha: float = 2.0,
                 mastered_winrate: float = 0.9, mastered_min_games: int = 50,
                 pfsp_min_games: int = 30, external_fraction: float = 0.0,
                 ema_alpha: float = 0.01):
        self.members: list[PoolMember] = []
        self.max_self_members = max_self_members
        self.pfsp_alpha = pfsp_alpha
        self.mastered_winrate = mastered_winrate
        self.mastered_min_games = mastered_min_games
        # Minimum games before trusting win-rate for PFSP weighting.
        # Until this threshold, wr=0.5 is used so early lucky streaks don't
        # sand-bag an opponent (e.g. Hellburner getting 0.003 weight after 17 games).
        self.pfsp_min_games = pfsp_min_games
        # Fixed fraction of pool samples that go to external heuristics,
        # bypassing PFSP. Guarantees Hellburner exposure regardless of win-rate.
        # Remaining (1 - external_fraction) is governed by PFSP over self-members.
        # 0.0 = legacy behaviour (externals compete in PFSP with everyone else).
        self.external_fraction = external_fraction
        # EMA smoothing for external-opponent win-rate tracking.  Using a lifetime
        # win/loss count for externals causes the PFSP weight to go stale once the
        # denominator is large — early-training wins dilute recent performance and
        # make a now-dominant opponent look "almost mastered".  An EMA with
        # ema_alpha ≈ 0.01 keeps an effective window of ~100 games, so PFSP
        # reflects the last ~1–2M training steps rather than the full run history.
        # Self-checkpoints still use lifetime win rate (stable enough given their
        # smaller n and shorter lifespan).
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
        # Evict oldest self if over cap (externals AND pinned champions are untouched)
        self_members = [m for m in self.members if m.kind == "self" and not m.pinned]
        if len(self_members) > self.max_self_members:
            oldest = min(self_members, key=lambda m: m.step_saved)
            self.members.remove(oldest)

    def add_pinned_rl(self, name: str, state_dict: dict) -> None:
        """Add a FIXED RL champion (e.g. rev38, rev53b) as a never-evicted 'self'
        opponent. Runs through the same GPU 'self' forward path; pinned so organic
        self-snapshot FIFO never drops it. step_saved=-1 keeps it out of FIFO order."""
        cpu_sd = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        self.members.append(PoolMember(
            name=f"seed_{name}", kind="self", state_dict=cpu_sd,
            step_saved=-1, pinned=True,
        ))

    def add_external_heuristic(self, name: str, py_path: str) -> None:
        """Load a .py file (must define `agent(obs)` or `agent(obs, config)`)."""
        py_path = os.fspath(py_path)
        spec = importlib.util.spec_from_file_location(f"opp_{name}", py_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load opponent .py: {py_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # required for @dataclass __module__ resolution
        spec.loader.exec_module(mod)
        if not hasattr(mod, "agent"):
            raise RuntimeError(f"{py_path} has no `agent` function")
        member = PoolMember(name=name, kind="external_heuristic", agent_fn=mod.agent)
        # Stash the source path so save() can persist it. Not a dataclass field
        # because it's a transient implementation detail for round-tripping.
        member._source_path = py_path
        self.members.append(member)

    # ---- sampling ----------------------------------------------------------

    def sample(self, rng: Optional[random.Random] = None,
               external_fraction: Optional[float] = None,
               pinned_fraction: Optional[float] = None) -> Optional[PoolMember]:
        """Sample a member by PFSP weight. Returns None if the pool is empty.

        ``external_fraction`` overrides the instance default (lets a caller ramp it
        per-rollout). If > 0, external heuristics are guaranteed that fraction of
        samples regardless of their PFSP win-rate, keeping targeted opponents (e.g.
        the peeler) in the mix even as the rollout win-rate climbs.

        ``pinned_fraction`` engages **ramp mode** (3-way split):
        external slice / **pinned-RL slice** / PFSP over ORGANIC (non-pinned) selves.
        This pulls pinned RL champions (e.g. rev38) OUT of PFSP into their own fixed
        ramped fraction — necessary because PFSP weight ``(1-wr)^α`` *up-samples* an
        opponent you lose to, so a weak from-scratch policy would otherwise see MORE
        rev38 early (backwards). When ``pinned_fraction is None`` the legacy 2-way
        behaviour is preserved (pinned members compete inside PFSP with the selves).
        Returns None when the chosen budget falls to PFSP but no organic snapshot
        exists yet (early from-scratch) — the caller then falls back to self-play.
        """
        if not self.members:
            return None
        r = rng or random
        ext_frac = self.external_fraction if external_fraction is None else external_fraction

        externals = [m for m in self.members if m.kind == "external_heuristic"]

        if pinned_fraction is not None:
            # --- Ramp mode: external / pinned-RL / PFSP-over-organic (non-pinned selves) ---
            pinned = [m for m in self.members if m.pinned]
            organic = [m for m in self.members if m.kind == "self" and not m.pinned]
            roll = r.random()
            if externals and roll < ext_frac:
                return r.choice(externals)
            if pinned and roll < ext_frac + pinned_fraction:
                return r.choice(pinned)
            if not organic:
                # No organic snapshots yet (early from-scratch) → caller does self-play.
                return None
            candidates = organic
        elif externals and ext_frac > 0.0:
            # Legacy 2-way: fixed external slice vs PFSP over all self-members (incl. pinned).
            if r.random() < ext_frac:
                return r.choice(externals)
            self_members = [m for m in self.members if m.kind == "self"]
            candidates = self_members if self_members else self.members
        else:
            # Legacy path: PFSP over all members together.
            candidates = self.members

        weights = [self._pfsp_weight(m) for m in candidates]
        total = sum(weights)
        if total <= 0:
            return r.choice(candidates)
        return r.choices(candidates, weights=weights, k=1)[0]

    def _pfsp_weight(self, m: PoolMember) -> float:
        # Long-lived fixed opponents (externals + pinned RL champions): EMA win-rate, so the
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
        # Update EMA win-rate for long-lived fixed opponents (externals + pinned RL champions)
        # so PFSP stays responsive to recent performance rather than the full accumulated history.
        if member.uses_ema:
            win_val = 1.0 if result == "win" else 0.0
            member.ema_win_rate = (
                (1.0 - self.ema_alpha) * member.ema_win_rate + self.ema_alpha * win_val
            )
            member.ema_games += 1

    def maybe_evict_mastered(self) -> list[str]:
        """Drop external opponents whose sustained win-rate is past threshold.
        Returns names of evicted members. Self-checkpoints are never evicted
        here (FIFO handles them when adding)."""
        evicted = []
        keep = []
        for m in self.members:
            mastered = (
                m.kind == "external_heuristic"
                and m.n_games >= self.mastered_min_games
                and m.win_rate >= self.mastered_winrate
            )
            if mastered:
                evicted.append(m.name)
            else:
                keep.append(m)
        self.members = keep
        return evicted

    # ---- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist pool to disk so it survives spot interruption / restart.

        Self-checkpoint state_dicts are saved in full. External heuristics save
        their .py path only (re-imported on load) — closures/agent functions
        themselves are not picklable across module reloads.
        """
        import torch  # local import: avoid forcing torch on test-only imports
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self_members = []
        ext_members = []
        for m in self.members:
            base = {
                "name": m.name, "wins": m.wins, "losses": m.losses, "draws": m.draws,
            }
            if m.kind == "self":
                self_members.append({**base, "step_saved": m.step_saved,
                                     "state_dict": m.state_dict, "pinned": m.pinned,
                                     "ema_win_rate": m.ema_win_rate, "ema_games": m.ema_games})
            elif m.kind == "external_heuristic":
                # We need the path to re-import on load. Resolution happens at
                # add time; we store it as an attribute when loading externals.
                path_attr = getattr(m, "_source_path", None)
                if path_attr is None:
                    # Skip — can't reconstruct without the source path
                    continue
                ext_members.append({
                    **base,
                    "source_path": path_attr,
                    "ema_win_rate": m.ema_win_rate,
                    "ema_games": m.ema_games,
                })
        payload = {
            "self_members": self_members,
            "external_members": ext_members,
            "config": {
                "max_self_members": self.max_self_members,
                "pfsp_alpha": self.pfsp_alpha,
                "mastered_winrate": self.mastered_winrate,
                "mastered_min_games": self.mastered_min_games,
                "pfsp_min_games": self.pfsp_min_games,
                "external_fraction": self.external_fraction,
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
    def load(cls, path: str, reload_externals: bool = True) -> "OpponentPool":
        """Recreate a pool from a saved file. Externals are re-imported from
        their stored .py path; pass reload_externals=False to skip them."""
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        cfg = data.get("config", {})
        pool = cls(
            max_self_members=cfg.get("max_self_members", 20),
            pfsp_alpha=cfg.get("pfsp_alpha", 2.0),
            mastered_winrate=cfg.get("mastered_winrate", 0.9),
            mastered_min_games=cfg.get("mastered_min_games", 50),
            pfsp_min_games=cfg.get("pfsp_min_games", 30),
            external_fraction=cfg.get("external_fraction", 0.0),
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
        if reload_externals:
            for m in data.get("external_members", []):
                try:
                    pool.add_external_heuristic(m["name"], m["source_path"])
                    pool.members[-1].wins = m["wins"]
                    pool.members[-1].losses = m["losses"]
                    pool.members[-1].draws = m["draws"]
                    pool.members[-1].ema_win_rate = m.get("ema_win_rate", 0.5)
                    pool.members[-1].ema_games = m.get("ema_games", 0)
                except Exception as e:
                    print(f"  WARN: could not re-import external {m['name']} "
                          f"from {m['source_path']}: {e}")
        return pool

    # ---- diagnostics -------------------------------------------------------

    def summary(self, max_rows: int = 8) -> str:
        if not self.members:
            return "  (pool empty)"
        # External heuristics are always shown (they can fall off the top-N
        # display once many self-checkpoints accumulate, creating the false
        # impression they were evicted when they're still being sampled).
        externals = [m for m in self.members if m.kind == "external_heuristic"]
        self_members = [m for m in self.members if m.kind != "external_heuristic"]
        top_self = sorted(self_members, key=lambda m: -self._pfsp_weight(m))[:max_rows]
        rows = top_self + externals
        # NOTE: this is the CONFIGURED target/cap; under a ramp (train_torch) the LIVE
        # per-rollout fraction is lower — see the "pool hard-ramp" line for the live value.
        ext_note = (f"  external_target_frac={self.external_fraction:.2f}"
                    if self.external_fraction > 0 else "")
        lines = [f"  pool size={len(self.members)}  alpha={self.pfsp_alpha}{ext_note}"]
        for m in rows:
            w = self._pfsp_weight(m)
            tag = " [fixed]" if (m.kind == "external_heuristic"
                                 and self.external_fraction > 0) else ""
            if m.uses_ema:
                # Show both EMA (recent) and lifetime win-rate so drift is visible.
                # uses_ema = externals AND pinned RL champions (both PFSP-weight off EMA).
                ema_str = f" ema_wr={m.ema_win_rate:.2f}(n={m.ema_games})"
                lines.append(
                    f"    {m.kind:20s} {m.name:30s} "
                    f"wr={m.win_rate:.2f}(n={m.n_games}){ema_str}  pfsp_w={w:.3f}{tag}"
                )
            else:
                lines.append(
                    f"    {m.kind:20s} {m.name:30s} "
                    f"wr={m.win_rate:.2f} (n={m.n_games})  pfsp_w={w:.3f}{tag}"
                )
        return "\n".join(lines)
