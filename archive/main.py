"""Adaptive strong heuristic entrypoint for Orbit Wars.

This wraps the strongest local public-kernel references currently in the repo:

- Suneet is the default policy because it is strong into Zach-style slow opens.
- Zach is used as an early anti-rush response when hostile fleets appear before
  turn 30, which fixes Suneet's observed Marco failure mode.

For Kaggle single-file submission this must be inlined, or submitted as a
multi-file tarball with ``candidate_zach_public.py`` and
``candidate_suneet_lb1200.py`` at the root.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
EARLY_RUSH_DETECT_UNTIL = 30
ZACH_RESPONSE_UNTIL = 90

_rush_mode_by_player = {}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_zach = _load("candidate_zach_public.py", "main_adaptive_zach_public")
_suneet = _load("candidate_suneet_lb1200.py", "main_adaptive_suneet_public")


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _has_early_hostile_fleet(obs, player):
    for fleet in _read(obs, "fleets", []) or []:
        owner = fleet[1]
        ships = fleet[6]
        if owner not in (-1, player) and ships >= 8:
            return True
    return False


def agent(obs, config=None):
    player = int(_read(obs, "player", 0) or 0)
    step = int(_read(obs, "step", 0) or 0)

    if step <= 0:
        _rush_mode_by_player[player] = False
    if step <= EARLY_RUSH_DETECT_UNTIL and _has_early_hostile_fleet(obs, player):
        _rush_mode_by_player[player] = True

    if _rush_mode_by_player.get(player, False) and step < ZACH_RESPONSE_UNTIL:
        return _zach.agent(obs)
    return _suneet.agent(obs, config)
