from collections import Counter

from orbit_wars_rl.research.audit_ar_action_lists import _merge_stats, _phase


def test_phase_bucket_skips_all_bucket():
    assert _phase(1) == "open"
    assert _phase(49) == "open"
    assert _phase(50) == "mid"
    assert _phase(99) == "mid"
    assert _phase(100) == "late"


def test_merge_stats_keeps_max_list_len_as_max_not_sum():
    dst = Counter({"list_len_max": 3})
    dst["_list_lens"] = [1, 3]
    src = Counter({"list_len_max": 5, "action_turns": 2})
    src["_list_lens"] = [2, 5]

    _merge_stats(dst, src)

    assert dst["list_len_max"] == 5
    assert dst["action_turns"] == 2
    assert dst["_list_lens"] == [1, 3, 2, 5]


if __name__ == "__main__":
    test_phase_bucket_skips_all_bucket()
    test_merge_stats_keeps_max_list_len_as_max_not_sum()
    print("test_audit_ar_action_lists: PASS")
