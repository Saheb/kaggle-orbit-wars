"""yusa (Rank 166, imitation learning of 13 top teams) as an eval opponent.

Wraps github.com/yusa-bot/Kaggle_Orbit_Wars_public `submission/` — a 7.2M entity
transformer trained by behavioral cloning of 13 top teams (torch-only inference,
2p/4p auto-detected). The submission's bare modules (model/features/physics) are
vendored under the `yusa_il` PACKAGE in opponents/yusa_bundle/ so their generic
names don't collide with our own model.py/features.py in sys.modules. Re-exports the
submission's `agent(obs)` as the kaggle path-agent hook. Same sibling-package pattern
as opponents/ender_bundle/.

⚠ WEIGHTS ARE GITIGNORED (*.pth, ~58MB). To run this opponent, place the two files in
opponents/yusa_bundle/yusa_il/weights/ :
  joint13_d384_best_2pbest.pth   joint13_d384_best_4pbest.pth
from github.com/yusa-bot/Kaggle_Orbit_Wars_public → submission/weights/.
"""
import os
import sys


def _find_bundle() -> str:
    """Locate opponents/yusa_bundle robustly — kaggle path-loads an agent file
    WITHOUT a usable __file__, so search the file dir (when available), cwd,
    cwd/opponents, and every sys.path entry for a dir containing yusa_bundle/yusa_il."""
    bases = []
    try:
        bases.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    cwd = os.getcwd()
    bases += [cwd, os.path.join(cwd, "opponents")]
    bases += [p for p in sys.path if p]
    for base in bases:
        for cand in (os.path.join(base, "yusa_bundle"),
                     os.path.join(base, "opponents", "yusa_bundle")):
            if os.path.isdir(os.path.join(cand, "yusa_il")):
                return cand
    raise ImportError("candidate_yusa: could not locate opponents/yusa_bundle")


_BUNDLE = _find_bundle()
if _BUNDLE not in sys.path:
    sys.path.insert(0, _BUNDLE)

from yusa_il.main import agent as _yusa_agent  # noqa: E402


def agent(obs, config=None):
    """kaggle path-agent entry → the submission's agent (obs-only). yusa's main.py
    already resets its 2p/4p detection on step rewind, so it is multi-game safe."""
    return _yusa_agent(obs)
