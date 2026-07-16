from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import _assert_game_completed, validate_opponent_assets


def test_bundled_opponent_missing_asset_fails_closed(tmp_path):
    opponent = tmp_path / "candidate_ender.py"
    opponent.write_text("def agent(obs, config=None): return []\n")
    with pytest.raises(FileNotFoundError, match="Do not trust this eval"):
        validate_opponent_assets(str(opponent), num_players=2)


def test_bundled_opponent_manifest_hashes_required_asset(tmp_path):
    opponent = tmp_path / "candidate_ender.py"
    opponent.write_text("def agent(obs, config=None): return []\n")
    asset = tmp_path / "ender_bundle" / "checkpoint_2p.pt"
    asset.parent.mkdir()
    asset.write_bytes(b"checkpoint")
    manifest = validate_opponent_assets(str(opponent), num_players=2)
    assert set(manifest) == {"ender_bundle/checkpoint_2p.pt"}
    assert len(next(iter(manifest.values()))) == 64


def test_submitted_opponent_missing_payload_fails_closed(tmp_path):
    opponent = tmp_path / "candidate_sub_presres05.py"
    opponent.write_text("def agent(obs, config=None): return []\n")
    with pytest.raises(FileNotFoundError, match="Do not trust this eval"):
        validate_opponent_assets(str(opponent), num_players=2)


def test_game_error_status_aborts_eval():
    env = SimpleNamespace(steps=[[SimpleNamespace(status="DONE"),
                                  SimpleNamespace(status="ERROR")]])
    with pytest.raises(RuntimeError, match="statuses"):
        _assert_game_completed(env, "seed=7")
