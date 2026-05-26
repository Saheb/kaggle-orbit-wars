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
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def n_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        # Win-rate of *current model* vs this opponent (so high = opponent mastered).
        if self.n_games == 0:
            return 0.5  # uninformative prior
        return self.wins / self.n_games


class OpponentPool:
    def __init__(self, max_self_members: int = 20, pfsp_alpha: float = 2.0,
                 mastered_winrate: float = 0.9, mastered_min_games: int = 50,
                 pfsp_min_games: int = 30, external_fraction: float = 0.0):
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
        # Evict oldest self if over cap (externals are untouched here)
        self_members = [m for m in self.members if m.kind == "self"]
        if len(self_members) > self.max_self_members:
            oldest = min(self_members, key=lambda m: m.step_saved)
            self.members.remove(oldest)

    def add_external_heuristic(self, name: str, py_path: str) -> None:
        """Load a .py file (must define `agent(obs)` or `agent(obs, config)`)."""
        py_path = os.fspath(py_path)
        spec = importlib.util.spec_from_file_location(f"opp_{name}", py_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load opponent .py: {py_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "agent"):
            raise RuntimeError(f"{py_path} has no `agent` function")
        member = PoolMember(name=name, kind="external_heuristic", agent_fn=mod.agent)
        # Stash the source path so save() can persist it. Not a dataclass field
        # because it's a transient implementation detail for round-tripping.
        member._source_path = py_path
        self.members.append(member)

    # ---- sampling ----------------------------------------------------------

    def sample(self, rng: Optional[random.Random] = None) -> Optional[PoolMember]:
        """Sample a member by PFSP weight. Returns None if the pool is empty.

        If external_fraction > 0, external heuristics are guaranteed that
        fraction of samples regardless of their PFSP win-rate. This keeps
        targeted opponents (e.g. Hellburner) in the training mix even when
        the rollout win-rate climbs high. PFSP governs only the self-members
        within the remaining (1 - external_fraction) budget.
        """
        if not self.members:
            return None
        r = rng or random

        externals = [m for m in self.members if m.kind == "external_heuristic"]
        self_members = [m for m in self.members if m.kind == "self"]

        if externals and self.external_fraction > 0.0:
            if r.random() < self.external_fraction:
                # Fixed external slice: uniform among all external heuristics.
                return r.choice(externals)
            # Remaining budget: PFSP over self-members only (fall through below).
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
        # Use uninformative prior until we have enough games to trust the estimate.
        wr = 0.5 if m.n_games < self.pfsp_min_games else m.win_rate
        return max(1.0 - wr, 1e-6) ** self.pfsp_alpha

    # ---- bookkeeping -------------------------------------------------------

    def record_result(self, member: PoolMember, result: str) -> None:
        """result in {'win', 'loss', 'draw'} from *current model's* perspective."""
        if result == "win":   member.wins += 1
        elif result == "loss": member.losses += 1
        else:                  member.draws += 1

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
                self_members.append({**base, "step_saved": m.step_saved, "state_dict": m.state_dict})
            elif m.kind == "external_heuristic":
                # We need the path to re-import on load. Resolution happens at
                # add time; we store it as an attribute when loading externals.
                path_attr = getattr(m, "_source_path", None)
                if path_attr is None:
                    # Skip — can't reconstruct without the source path
                    continue
                ext_members.append({**base, "source_path": path_attr})
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
        )
        for m in data.get("self_members", []):
            pool.members.append(PoolMember(
                name=m["name"], kind="self",
                state_dict=m["state_dict"], step_saved=m["step_saved"],
                wins=m["wins"], losses=m["losses"], draws=m["draws"],
            ))
        if reload_externals:
            for m in data.get("external_members", []):
                try:
                    pool.add_external_heuristic(m["name"], m["source_path"])
                    pool.members[-1].wins = m["wins"]
                    pool.members[-1].losses = m["losses"]
                    pool.members[-1].draws = m["draws"]
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
        ext_note = (f"  external_fraction={self.external_fraction:.2f}"
                    if self.external_fraction > 0 else "")
        lines = [f"  pool size={len(self.members)}  alpha={self.pfsp_alpha}{ext_note}"]
        for m in rows:
            w = self._pfsp_weight(m)
            tag = " [fixed]" if (m.kind == "external_heuristic"
                                 and self.external_fraction > 0) else ""
            lines.append(
                f"    {m.kind:20s} {m.name:30s} "
                f"wr={m.win_rate:.2f} (n={m.n_games})  pfsp_w={w:.3f}{tag}"
            )
        return "\n".join(lines)
