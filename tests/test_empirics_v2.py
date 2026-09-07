"""Oracle gates for analysis/empirics_v2.py.

Each estimator must recover the known truth on the exact-martingale
simulator before its venue numbers are believed. Thresholds bracket the
n=2000 seed-7 reference run with margin for seed drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.core import simulate
from analysis.empirics_v2 import (budget, diagnostics, local_rates,
                                  move_panel, power_fit)


@pytest.fixture(scope="module")
def clean():
    return simulate(2000)


@pytest.fixture(scope="module")
def noisy():
    return simulate(2000, noise=0.02)


@pytest.fixture(scope="module")
def stochvol():
    return simulate(2000, stochvol=True)


def test_budget_clean(clean):
    a, _ = budget(clean)
    assert 0.93 <= a["slope"] <= 1.07


def test_budget_noise_inflates(noisy):
    a, _ = budget(noisy)
    assert a["slope"] > 1.15


def test_budget_stochvol_kept(stochvol):
    a, _ = budget(stochvol)
    assert 0.90 <= a["slope"] <= 1.10


def test_gauss_rate_is_the_right_one(clean):
    slopes, _ = local_rates(move_panel(clean))
    assert 0.90 <= slopes["v_gauss"]["slope"] <= 1.15
    assert slopes["v_binom"]["slope"] < 0.80


def test_power_fit_tau_clock(clean):
    pf = power_fit(move_panel(clean))
    assert 0.85 <= pf["alpha"] <= 1.20
    assert 1.30 <= pf["gamma"] <= 1.90     # probit rate: between arcsin=1
    #                                        and logit=2 over this p range


def test_acf_clean_flat(clean):
    _, acf = diagnostics(move_panel(clean))
    assert abs(acf.acf_z.iloc[0]) < 0.05
    assert abs(acf.acf_lambda.iloc[0]) < 0.05


def test_acf_noise_bounce(noisy):
    _, acf = diagnostics(move_panel(noisy))
    assert acf.acf_z.iloc[0] < -0.10       # MA(1) reversal
    assert acf.acf_lambda.iloc[0] > 0.02   # squares of MA(1) cluster


def test_acf_stochvol_persistence(stochvol):
    _, acf = diagnostics(move_panel(stochvol))
    assert acf.acf_lambda.iloc[0] > 0.08   # true vol clustering
    assert abs(acf.acf_z.iloc[0]) < 0.06   # with no signed reversal
