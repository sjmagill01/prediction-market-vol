"""Oracle gates for analysis.bridge_v5 on synthetic panels + the frozen
pair table. Synthetic tests use known-answer generators; the pair-table
test pins the signed-off match table (2026-09-07)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import bridge_v5
from analysis.bridge_v5 import (_within_ar1, build_pairs, gap_stats,
                                lead_lag, paired_budget, paired_power)

RNG = np.random.default_rng(7)


def _panel(p_k: np.ndarray, p_pm: np.ndarray, n_per: int) -> pd.DataFrame:
    n = len(p_k)
    return pd.DataFrame({
        "pair_id": np.repeat(np.arange(n // n_per), n_per),
        "ts": pd.date_range("2025-01-01", periods=n_per, freq="D",
                            tz="UTC").to_numpy().reshape(1, -1).repeat(
                                n // n_per, 0).ravel(),
        "p_k": p_k, "p_pm": p_pm})


def test_pair_table_pinned():
    pairs = build_pairs()
    assert len(pairs) == 82
    assert (pairs.quality == "exact").sum() == 64
    assert set(pairs.domain) == {"fed_decision", "fed_chair_nom",
                                 "btc_monthly", "btc_yearly", "election"}
    # every key is bar-backed by construction
    kk = bridge_v5._bar_keys("kalshi")
    pk = bridge_v5._bar_keys("polymarket")
    assert pairs.k_key.isin(kk).all()
    assert pairs.pm_key.isin(pk).all()
    # no duplicate matches on either side within a domain
    assert not pairs.duplicated(["domain", "k_key"]).any()
    assert not pairs.duplicated(["domain", "pm_key"]).any()


def test_within_ar1_recovers_rho():
    rho = 0.7
    n_pair, n_per = 40, 400
    g = np.zeros((n_pair, n_per))
    eps = RNG.normal(size=(n_pair, n_per))
    for t in range(1, n_per):
        g[:, t] = rho * g[:, t - 1] + eps[:, t]
    panel = _panel(g.ravel(), g.ravel(), n_per).assign(gap=g.ravel())
    est = _within_ar1(panel, "gap")
    assert abs(est - rho) < 0.03


def test_paired_budget_martingale_near_one():
    # binary-settling martingale: p diffuses and is absorbed at 0/1;
    # realized sum dp^2 should average p0(1-p0) => ratios ~ 1.
    n_pair, n_per = 300, 250
    p0 = RNG.uniform(0.2, 0.8, size=n_pair)
    p = np.empty((n_pair, n_per))
    p[:, 0] = p0
    for t in range(1, n_per):
        step = RNG.normal(0, 0.05, size=n_pair) * np.sqrt(
            p[:, t - 1] * (1 - p[:, t - 1]))
        p[:, t] = np.clip(p[:, t - 1] + step, 0.0, 1.0)
    # force settlement at the end
    p[:, -1] = (RNG.uniform(size=n_pair) < p[:, -2]).astype(float)
    panel = _panel(p.ravel(), p.ravel(), n_per)
    pairs = pd.DataFrame({"domain": "sim", "unit": "s", "quality": "exact"},
                         index=np.arange(n_pair))
    stats, agg = paired_budget(panel, pairs)
    assert stats["n_pairs"] == n_pair
    # budget identity holds in EXPECTATION; the ratio is right-skewed so
    # the median sits below 1 even for a true martingale.
    assert abs(agg.ratio_k.mean() - 1.0) < 0.15
    assert 0.5 < stats["median_ratio_k"] < 1.1
    assert abs(stats["median_ratio_of_ratios"] - 1.0) < 1e-9  # identical cols


def test_paired_power_recovers_gamma():
    gamma = 1.4
    n_pair, n_per = 200, 300
    p = np.empty((n_pair, n_per))
    p[:, 0] = RNG.uniform(0.15, 0.85, size=n_pair)
    for t in range(1, n_per):
        v = 0.004 * (p[:, t - 1] * (1 - p[:, t - 1])) ** gamma
        p[:, t] = np.clip(p[:, t - 1] + RNG.normal(size=n_pair) * np.sqrt(v),
                          0.01, 0.99)
    panel = _panel(p.ravel(), p.ravel(), n_per)
    out = paired_power(panel)
    for venue in ("kalshi", "polymarket"):
        assert abs(out[venue]["gamma"] - gamma) < 0.15


def test_lead_lag_detects_direction():
    # p_pm is a random walk; p_k copies it with a one-period delay.
    n_pair, n_per = 60, 300
    dpm = RNG.normal(0, 0.02, size=(n_pair, n_per))
    p_pm = 0.5 + np.cumsum(dpm, axis=1)
    p_k = np.roll(p_pm, 1, axis=1)
    p_k[:, 0] = 0.5
    panel = _panel(p_k.ravel(), p_pm.ravel(), n_per)
    ll = lead_lag(panel)
    assert ll["pm_predicts_k"]["cross_lag"] > 0.9
    assert ll["pm_predicts_k"]["cross_t"] > 10
    assert abs(ll["k_predicts_pm"]["cross_lag"]) < 0.05
    # xcorr peaks at PM leading by one
    xc = ll["xcorr_dpm_leads_dk"]
    assert xc["1"] == max(xc.values())


def test_gap_stats_zero_gap():
    n_per = 50
    p = np.tile(np.linspace(0.3, 0.7, n_per), 4)
    panel = _panel(p, p, n_per)
    pairs = pd.DataFrame({"domain": "sim", "unit": "s", "quality": "exact"},
                         index=np.arange(4))
    g = gap_stats(panel, pairs)
    assert g["mean_abs_gap"] == pytest.approx(0.0)
    assert g["frac_gt_5c"] == 0.0
