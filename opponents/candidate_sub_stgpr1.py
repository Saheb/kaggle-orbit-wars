"""Final submitted stgpr1 0.5M 2p payload, loaded from the tracked tarball."""
import os
import tarfile

_ARCHIVE = os.path.abspath("final_submissions/submission_stgpr1.tar.gz")
with tarfile.open(_ARCHIVE, "r:gz") as _tar:
    _SOURCE = _tar.extractfile("neural_agent.py").read()
_GLOBALS = {
    "__name__": "_submitted_stgpr1_neural",
    "__file__": f"{_ARCHIVE}:neural_agent.py",
}
exec(compile(_SOURCE, _GLOBALS["__file__"], "exec"), _GLOBALS)
agent = _GLOBALS["agent"]

__all__ = ["agent"]
