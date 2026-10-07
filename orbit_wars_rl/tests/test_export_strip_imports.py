"""export_agent inlines features.py / action_mask.py with their imports stripped. A parenthesized
multi-line import used to leave its continuation line behind, so the exported agent failed to
compile (IndentationError) — every timeline-era export since 2026-07-16 was broken."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_agent import _read_module_body, _strip_imports

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_multiline_import_fully_stripped():
    src = ("import math\n"
           "from timeline import (a, b,\n"
           "                      c, d)\n"
           "from x import (\n"
           "    e,\n"
           ")\n"
           "X = 1\n")
    assert _strip_imports(src).strip() == "X = 1"


def test_inlined_modules_compile():
    for name in ("features.py", "action_mask.py", "timeline.py", "reinforce_cooldown.py",
                 "binary_policy.py"):
        compile(_read_module_body(os.path.join(_SRC_DIR, name)), name, "exec")
