"""Oracle gates for analysis/race_v4.py.

Block A (ZI-SLV): the filter must be a proper probability in the tick
measure (masses sum to one), the batched and single-path filters must
agree step for step, the fit must recover pi0 on a stale simulator and
must not invent pi0 on a clean one, and zero inflation must buy
likelihood exactly when the truth is zero inflated. Block B: the race
must rank conditional-variance models above static baselines on a
stochvol simulator, and the regime composite must match the global fit
when the simulator has only one regime. Block C: the realized budget
ratio is a theorem (== 1 in expectation) and the gate holds the MC
plumbing to it. Fits use reduced sizes; thresholds bracket the seed-7
reference run with margin for drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.config import UNIT
from analysis.core import simulate
from analysis.race_v4 import (run_budget, run_race, simulate_zi,
                              zi_batches, zi_bin_prob, zi_fit, zi_forecast,
                              zi_loglik, zi_start)

ZI_TRUTH = {"c": 0.02, "alpha": 0.5, "eta": 0.02, "phi": 0.9, "s": 0.6,
            "pi0": 0.25}
CLEAN_TRUTH = {**ZI_TRUTH, "pi0": 0.0}


@pytest.fixture(scope="module")
def zi_oracle():
    batches = zi_batches(simulate_zi(150, ZI_TRUTH))
    start = zi_start(batches)
    full = zi_fit(batches, "zi", start, maxiter=250)
    off = zi_fit(batches, "zi-off", start, maxiter=250)
    return full, off, batches


# ------------------------------------------------------------- filter math
def test_tick_masses_sum_to_one():
    seg = simulate_zi(1, ZI_TRUTH, seed=11)[0]
    fores, _ = zi_forecast(ZI_TRUTH, seg)
    from scipy.stats import norm
    unit = 0.01
    for j in (0, len(fores) // 2, len(fores) - 1):
        # previous print for step j+1 (seg[0] is z; invert the probit)
        p_prev = float(norm.cdf(seg[0][j]))
        p_prev = float(np.clip(np.round(p_prev / unit) * unit,
                               unit, 1 - unit))
        ks = np.arange(int(-p_prev / unit) - 1, int((1 - p_prev) / unit) + 2)
        tot = sum(zi_bin_prob(fores[j], ZI_TRUTH["pi0"], p_prev, int(k),
                              unit) for k in ks)
        assert 0.985 <= tot <= 1.002, (j, tot)


def test_batched_matches_single():
    segs = simulate_zi(8, ZI_TRUTH, seed=3)
    ll_b, n_b = zi_loglik(ZI_TRUTH, zi_batches(segs))
    ll_s = sum(float(zi_forecast(ZI_TRUTH, s)[1].sum()) for s in segs)
    n_s = sum(len(s[0]) - 1 for s in segs)
    assert n_b == n_s
    assert abs(ll_b - ll_s) <= 1e-6 * max(1.0, abs(ll_s)), (ll_b, ll_s)


# --------------------------------------------------------------- recovery
def test_zi_recovers_pi0(zi_oracle):
    th = zi_oracle[0]["theta"]
    assert 0.18 <= th["pi0"] <= 0.32, th["pi0"]
    assert 0.012 <= th["c"] <= 0.032
    assert 0.30 <= th["alpha"] <= 0.70
    assert th["phi"] >= 0.80
    assert 0.40 <= th["s"] <= 0.85


def test_zi_beats_zi_off_on_stale(zi_oracle):
    full, off, _ = zi_oracle
    assert full["ll_per_obs"] - off["ll_per_obs"] > 0.02


def test_zi_null_on_clean():
    batches = zi_batches(simulate_zi(120, CLEAN_TRUTH, seed=5))
    r = zi_fit(batches, "zi", zi_start(batches), maxiter=200)
    assert r["theta"]["pi0"] <= 0.10, r["theta"]["pi0"]


# -------------------------------------------------------------- horse race
@pytest.fixture(scope="module")
def race_df():
    mkts = simulate(400, stochvol=True)
    return run_race("simtest", mkts, UNIT["sim"], max_obs=4000,
                    max_test=100, maxiter=80, write=False)


def test_race_slv_beats_static(race_df):
    assert race_df["global-slv"].mean() >= race_df["bucket"].mean()
    assert race_df["global-slv"].mean() >= race_df["garch"].mean()


def test_race_composite_matches_global_single_regime(race_df):
    # one-regime simulator: the regime split cannot help or hurt much
    gap = abs(race_df["composite"].mean() - race_df["global-slv"].mean())
    assert gap <= 0.10, gap


# ------------------------------------------------------------------ budget
def test_realized_budget_ratio_is_one():
    mkts = simulate(800)
    thetas = {e: dict(ZI_TRUTH) for e in ("full", "zi", "clam")}
    thetas["clam"]["s"] = 1e-6
    df = run_budget(mkts, "simtest", UNIT["sim"], thetas=thetas,
                    n_paths=50, write=False)
    ratio = (df["evd"].sum() + df["evj"].sum()) / df["budget"].sum()
    assert 0.85 <= ratio <= 1.15, ratio
