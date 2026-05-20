"""Review-only hybrid: Zach public opening, Suneet public mid/late planner.

This file imports local candidate modules for rapid local validation. It is not
submission-ready as-is; if it proves useful, inline the selected agents before
submitting.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OPENING_UNTIL = 80


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_zach = _load("candidate_zach_public.py", "hybrid_zach_opening")
_suneet = _load("candidate_suneet_lb1200.py", "hybrid_suneet_midlate")


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def agent(obs, config=None):
    step = int(_read(obs, "step", 0) or 0)
    if step < OPENING_UNTIL:
        return _zach.agent(obs)
    return _suneet.agent(obs, config)
