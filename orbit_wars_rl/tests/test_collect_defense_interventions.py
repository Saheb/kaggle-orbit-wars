import json
import pickle
import tempfile
from pathlib import Path

from orbit_wars_rl.collect_defense_interventions import write_outputs


def test_write_outputs_overwrites_partial_with_final(tmp_path):
    records_out = tmp_path / "records.pkl"
    summary_out = tmp_path / "summary.json"
    config = {"games": 4, "flush_every": 1}

    first = [{"seed": 1, "helped": 0}]
    first_stats = {"branch_records": 1, "helped": 0}
    payload = write_outputs(first, first_stats, records_out, summary_out, config, complete=False)

    assert payload["complete"] is False
    assert pickle.load(records_out.open("rb")) == first
    summary = json.loads(summary_out.read_text())
    assert summary["records"] == 1
    assert summary["complete"] is False

    final = first + [{"seed": 2, "helped": 1}]
    final_stats = {"branch_records": 2, "helped": 1}
    payload = write_outputs(final, final_stats, records_out, summary_out, config, complete=True)

    assert payload["complete"] is True
    assert pickle.load(records_out.open("rb")) == final
    summary = json.loads(summary_out.read_text())
    assert summary["records"] == 2
    assert summary["stats"] == final_stats
    assert summary["complete"] is True
    print("test_write_outputs_overwrites_partial_with_final: PASS")


if __name__ == "__main__":
    print("Running collect_defense_interventions tests...\n")
    with tempfile.TemporaryDirectory() as d:
        test_write_outputs_overwrites_partial_with_final(Path(d))
    print("\nAll collect_defense_interventions tests passed!")
