"""v2 core gates: the simulator must be an exact bounded martingale and
the observation knobs must corrupt it in the documented directions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import core


def _budget_ratio(mkts):
    num = den = 0.0
    for m in mkts:
        num += float(np.sum(np.diff(m.closes) ** 2)
                     + (m.final - m.closes[-1]) ** 2)
        den += m.closes[0] * (1 - m.closes[0])
    return num / den


def test_budget_theorem_clean():
    mkts = core.simulate(3000, seed=1)
    assert abs(_budget_ratio(mkts) - 1.0) < 0.05


def test_budget_theorem_stochvol():
    mkts = core.simulate(3000, stochvol=True, seed=2)
    assert abs(_budget_ratio(mkts) - 1.0) < 0.06


def test_staleness_leaves_budget_unbiased():
    mkts = core.simulate(3000, stale=0.5, seed=3)
    assert abs(_budget_ratio(mkts) - 1.0) < 0.06


def test_noise_inflates_budget():
    # 2% additive noise adds ~2*noise^2 to each dp^2 (MA(1) bounce), which
    # inflates the ratio well clear of the exact-theorem tolerance of 0.05.
    assert _budget_ratio(core.simulate(1500, noise=0.02, seed=4)) > 1.2


def test_one_step_martingale():
    pan = core.panel(core.simulate(2000, seed=5))
    se = pan["dp"].std() / np.sqrt(len(pan))
    assert abs(pan["dp"].mean()) < 4 * se


def test_outcome_frequency_matches_price():
    mkts = core.simulate(4000, seed=6)
    p0 = np.array([m.closes[0] for m in mkts])
    y = np.array([m.final for m in mkts])
    hi = p0 > 0.6
    lo = p0 < 0.4
    assert abs(y[hi].mean() - p0[hi].mean()) < 0.04
    assert abs(y[lo].mean() - p0[lo].mean()) < 0.04


def test_stochvol_clusters_squared_moves():
    pan = core.panel(core.simulate(1500, stochvol=True, seed=7))
    pan = pan.sort_values(["key", "tau"], ascending=[True, False])
    x = pan["dp2"].to_numpy()
    same = (pan["key"] == pan["key"].shift(-1)).to_numpy()[:-1]
    a, b = x[:-1][same], x[1:][same]
    r = np.corrcoef(np.log(a + 1e-12), np.log(b + 1e-12))[0, 1]
    assert r > 0.05


def test_grid_helpers():
    assert core.bucket(np.array([0.0, 0.049, 0.5, 0.999])).tolist() == \
        [0, 0, 4, 8]
    assert core.tau_band(np.array([1.0, 2.9, 100.0, 1e6])).tolist() == \
        [0, 0, 5, 5]
    assert core.bucket_label(0) == "[0.00,0.05)"
    assert core.tau_label(5) == "[90,inf)d"
