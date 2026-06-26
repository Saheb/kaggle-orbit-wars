"""One-off research, analysis, probe, audit and dataset-builder scripts.

Not part of the core train/eval pipeline. Run as modules from the repo root, e.g.
    python -m orbit_wars_rl.scripts.hold_autopsy --help

These scripts import the core pipeline as ``orbit_wars_rl.<module>``. The core
modules in turn use sibling imports (e.g. ``from config import Config``), which
resolve only when ``orbit_wars_rl/`` itself is on sys.path — so we add it here
(the parent of this package) for the ``python -m`` invocation above.
"""
import os as _os
import sys as _sys

_pkg_parent = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # orbit_wars_rl/
if _pkg_parent not in _sys.path:
    _sys.path.insert(0, _pkg_parent)
