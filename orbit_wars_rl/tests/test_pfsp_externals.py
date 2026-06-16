"""Unit test for --pfsp-externals: the external pool slice is sampled PFSP-weighted (toward the
rung we lose to most) when ON, and UNIFORM when OFF (legacy).

Context: with external_fraction=1.0 the pool serves only the external slice. Historically that slice
was `r.choice(externals)` = uniform, so a 3-rung league (h10/h12/h14) got ~1/3 each REGARDLESS of the
displayed pfsp_w (which only governed the self-snapshot branch). --pfsp-externals routes the external
pick through _pfsp_weight so games concentrate on the low-win-rate / matched-difficulty rungs.

Run:  orbit_wars_rl/.venv/bin/python orbit_wars_rl/tests/test_pfsp_externals.py
"""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opponent_pool import OpponentPool, PoolMember


def _pool(pfsp_externals):
    # external_fraction=1.0 → every sample falls to the external slice (legacy 2-way branch).
    pool = OpponentPool(pfsp_alpha=2.0, pfsp_min_games=30,
                        external_fraction=1.0, pfsp_externals=pfsp_externals)
    for name, ema in (("h10", 0.85), ("h12", 0.50), ("h14", 0.25)):
        m = PoolMember(name=name, kind="external_heuristic", state_dict=None, step_saved=-1)
        m.ema_win_rate = ema
        m.ema_games = 100          # > pfsp_min_games → win-rate is trusted (not the 0.5 prior)
        pool.members.append(m)
    return pool


def _counts(pool, n=30000):
    rng = random.Random(0)
    c = {"h10": 0, "h12": 0, "h14": 0}
    for _ in range(n):
        c[pool.sample(rng=rng).name] += 1
    return {k: v / n for k, v in c.items()}


def test_uniform_when_off():
    f = _counts(_pool(pfsp_externals=False))
    for k in ("h10", "h12", "h14"):
        assert abs(f[k] - 1 / 3) < 0.03, f"OFF must be ~uniform, got {f}"


def test_pfsp_weighted_when_on():
    # weights = (1-wr)^2: h10 .0225, h12 .25, h14 .5625 → normalized ~0.027 / 0.299 / 0.674
    f = _counts(_pool(pfsp_externals=True))
    assert f["h14"] > f["h12"] > f["h10"], f"ON must tilt toward low-wr rung, got {f}"
    assert f["h14"] > 0.55, f"h14 (lose-most) should dominate, got {f['h14']:.3f}"
    assert f["h10"] < 0.10, f"h10 (mastered) should be starved, got {f['h10']:.3f}"


def test_single_external_is_uniform_noop():
    pool = OpponentPool(external_fraction=1.0, pfsp_externals=True)
    m = PoolMember(name="solo", kind="external_heuristic", state_dict=None, step_saved=-1)
    m.ema_win_rate, m.ema_games = 0.99, 100      # high wr → weight ~0, but it's the only option
    pool.members.append(m)
    assert pool.sample(rng=random.Random(0)).name == "solo", "1 external must still be picked"


def test_external_fraction_leaves_pool_self_slice():
    pool = _pool(pfsp_externals=True)
    pool.external_fraction = 0.8
    pool.members.append(PoolMember(name="self_step_0", kind="self", step_saved=0))
    rng = random.Random(1)
    n = 20000
    external = 0
    self_snap = 0
    for _ in range(n):
        m = pool.sample(rng=rng)
        if m.kind == "external_heuristic":
            external += 1
        elif m.kind == "self":
            self_snap += 1
    ef = external / n
    sf = self_snap / n
    assert abs(ef - 0.8) < 0.03, f"external slice should be ~0.8, got {ef:.3f}"
    assert abs(sf - 0.2) < 0.03, f"pool-self slice should be ~0.2, got {sf:.3f}"


def test_persists_across_save_load():
    import torch  # noqa
    for val in (True, False):
        pool = _pool(pfsp_externals=val)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "pool.pt")
            pool.save(p)
            loaded = OpponentPool.load(p, reload_externals=False)
            assert loaded.pfsp_externals == val, f"pfsp_externals must round-trip ({val})"


if __name__ == "__main__":
    test_uniform_when_off()
    test_pfsp_weighted_when_on()
    test_single_external_is_uniform_noop()
    test_external_fraction_leaves_pool_self_slice()
    test_persists_across_save_load()
    print("PASS: pfsp-externals — uniform off, PFSP-weighted on, self slice, single-noop, save/load round-trip")
