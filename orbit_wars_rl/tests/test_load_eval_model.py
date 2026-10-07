"""Probes must build the model exactly like evaluate_checkpoint (Key Lesson 14).

build_agent_fn reads the mask contract off the MODEL OBJECT. The 2026-10 review found
ender_sizing / coord_overkill_probe / peel_diagnosis hand-copying that setup and forgetting
binary_commit_gates, so the minimal-gates champion was probed under the legacy "full" gates.
All of them now go through eval.load_eval_model; these tests pin that.
"""
import os
import sys

import torch

_RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RL)

import eval as ev  # noqa: E402
from config import Config  # noqa: E402
from model import EntityTransformer  # noqa: E402
from ppo import PPOLearner  # noqa: E402


def _save_minimal_ckpt(tmp_path):
    cfg = Config()
    m = cfg.model
    m.ship_bin_mode = "binary"
    m.binary_commit_gates = "minimal"
    m.action_decode = "target"
    m.allow_reinforce = True
    m.reinforce_gate_min_planets = 2
    m.reverse_edge_cooldown = 3
    path = tmp_path / "ckpt.pt"
    torch.save(PPOLearner(EntityTransformer(m), cfg).state_dict(), path)
    return str(path)


def test_load_eval_model_carries_the_mask_contract(tmp_path):
    cfg = Config()
    cfg.device = "cpu"
    model, decode = ev.load_eval_model(_save_minimal_ckpt(tmp_path), cfg)
    assert decode == "target"
    assert model.binary_commit_gates == "minimal"
    assert model.allow_reinforce is True
    assert model.reinforce_gate_min_planets == 2
    assert model.reverse_edge_cooldown == 3
    assert not model.training


def test_probe_agent_runs_under_the_checkpoints_gates(tmp_path, monkeypatch):
    monkeypatch.chdir(os.getcwd())          # ender_sizing chdirs to the repo on import
    import ender_sizing
    seen = {}
    monkeypatch.setattr(ev, "build_agent_fn", lambda model, *a, **k: seen.setdefault("m", model))
    ender_sizing._checkpoint_agent(_save_minimal_ckpt(tmp_path))
    assert seen["m"].binary_commit_gates == "minimal"


def test_probes_do_not_hand_build_models():
    for name in ("ender_sizing.py", "peel_diagnosis.py", "coord_overkill_probe.py"):
        with open(os.path.join(_RL, name)) as f:
            src = f.read()
        assert "EntityTransformer(" not in src and "load_checkpoint(" not in src, (
            f"{name} builds its own model — use eval.load_eval_model")


def test_legacy_discipline_checkpoint_is_refused(tmp_path):
    """presres1/stgpr1-style checkpoints (sufficient_commit_factor=1.0) would silently play a
    different policy now that the legacy masks are gone — loading must refuse instead."""
    import pytest
    path = _save_minimal_ckpt(tmp_path)
    ckpt = torch.load(path, weights_only=False)
    ckpt["config"]["sufficient_commit_factor"] = 1.0
    torch.save(ckpt, path)
    cfg = Config()
    cfg.device = "cpu"
    with pytest.raises(RuntimeError, match="legacy discipline masks"):
        ev.load_eval_model(path, cfg)
