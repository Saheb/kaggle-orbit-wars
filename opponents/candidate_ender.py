"""Ender (provisional top-10 open-source Orbit Wars agent) as an eval opponent.

Wraps sinking-point/ender's published "launch-halt-search" Kaggle submission bundle
(github.com/sinking-point/ender, release tag `submission`). Inference is torch-only
(JAX was only their training/env backend); the slim bundle lives in
`opponents/ender_bundle/` alongside this file — same sibling-package pattern as
`opponents/orbit_lite/` for Ajay.

This mirrors the bundle's own `main.py`: it sets the submitted runtime env-var config
verbatim (so we face the ACTUAL submitted agent, search + all), only overriding the two
checkpoint paths to absolute so eval can run from any cwd, then re-exports the kaggle
`agent(obs, config=None)` entry point.
"""
from __future__ import annotations

import os
import sys

def _find_bundle() -> str:
    """Locate opponents/ender_bundle robustly.

    kaggle_environments execs a path-agent WITHOUT a usable ``__file__``, so we
    cannot rely on it; search the file's dir (when available), cwd, cwd/opponents,
    and every sys.path entry for a dir containing ``ender_bundle/orbit_wars_pt``.
    """
    bases = []
    try:
        bases.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    cwd = os.getcwd()
    bases += [cwd, os.path.join(cwd, "opponents")]
    bases += [p for p in sys.path if p]
    for base in bases:
        for cand in (os.path.join(base, "ender_bundle"),
                     os.path.join(base, "opponents", "ender_bundle")):
            if os.path.isdir(os.path.join(cand, "orbit_wars_pt")):
                return cand
    raise ImportError("candidate_ender: could not locate opponents/ender_bundle")


_BUNDLE = _find_bundle()
if _BUNDLE not in sys.path:
    sys.path.insert(0, _BUNDLE)

# Absolute checkpoint paths (bundle's main.py uses cwd-relative names).
os.environ.setdefault("ORBIT_WARS_CHECKPOINT_4P", os.path.join(_BUNDLE, "checkpoint_4p.pt"))
os.environ.setdefault("ORBIT_WARS_CHECKPOINT_2P", os.path.join(_BUNDLE, "checkpoint_2p.pt"))

# Submitted runtime config, copied verbatim from the bundle's main.py so behaviour
# matches the real top-10 submission (do not "tune" these — that changes the opponent).
os.environ.setdefault("ORBIT_WARS_DEVICE", "cpu")
os.environ.setdefault("ORBIT_WARS_CPU_THREADS", "1")
os.environ.setdefault("ORBIT_WARS_GREEDY", "0")
os.environ.setdefault("ORBIT_WARS_LOG_TIMING", "0")  # quiet: bundle default was 1 (per-step spam)
os.environ.setdefault("ORBIT_WARS_USE_STUDENT_FOR_SEARCH_4P", "1")
os.environ.setdefault("ORBIT_WARS_USE_STUDENT_FOR_SEARCH_2P", "1")
os.environ.setdefault("ORBIT_WARS_SEARCH_MAIN_POLICY_FOR_EGO_STEPS_4P", "10")
os.environ.setdefault("ORBIT_WARS_SEARCH_MAIN_POLICY_FOR_EGO_STEPS_2P", "10")
os.environ.setdefault("ORBIT_WARS_SAMPLING_MODE", "mixed")
os.environ.setdefault("ORBIT_WARS_SAMPLING_MODE_4P", "mixed")
os.environ.setdefault("ORBIT_WARS_SAMPLING_MODE_2P", "mixed")
os.environ.setdefault("ORBIT_WARS_TARGET_METHOD", "interval")
os.environ.setdefault("ORBIT_WARS_INTERVAL_GEOMETRY", "tangent")
os.environ.setdefault("ORBIT_WARS_MODEL_SEARCH_ADAPTIVE_HORIZON", "1")
os.environ.setdefault("ORBIT_WARS_MODEL_SEARCH_ADAPTIVE_HORIZON_OFFSET", "3")
os.environ.setdefault("ORBIT_WARS_MODEL_SEARCH_MIN_OVERAGE_S", "10.0")
os.environ.setdefault("ORBIT_WARS_MODEL_SEARCH_LAUNCH_PROB_THRESHOLD", "0.05")

# Kaggle path-agent entry point. Must be the last callable defined in this module.
from orbit_wars_pt.kaggle_adapter import agent  # noqa: E402

__all__ = ["agent"]
