"""Adaptive review candidate: Suneet default with Zach anti-rush opening.

This is a local teacher-candidate wrapper, not submission-ready as a single
file. It imports the strongest pulled heuristic agents so we can validate the
policy switch before inlining or packaging it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
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


_zach = _load("candidate_zach_public.py", "adaptive_zach_public")
_suneet = _load("candidate_suneet_lb1200.py", "adaptive_suneet_public")


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
