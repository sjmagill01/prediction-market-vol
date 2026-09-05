"""Analysis tests: the simulation is the oracle.

Bands are deliberately loose (fixed seed, N=200) but tight enough to catch a
broken estimator: a wrong variance budget, a mis-specified candidate rate, or
swapped diagnostics would blow through them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import vol_check as vc  # noqa: E402

N, SEED = 200, 7


def _results(**kw):
    mkts = vc.simulate(N, seed=SEED, **kw)
    panel = vc.candidate_rates(vc.daily_panel(mkts))
    a = vc.test_a(mkts)
    b = vc.test_b(panel)
    return a, b, panel


def test_clean_budget_slope_near_one():
    a, b, _ = _results()
    assert 0.85 < a.attrs["slope"] < 1.20


def test_clean_gaussian_fits_binomial_fails():
    _, b, panel = _results()
    assert 0.85 < b["v_gauss"]["slope"] < 1.35
    assert b["v_binom"]["slope"] < 0.85  # binomial overstates local risk
    # Gaussian bucket ratios flat in the bulk (0.15 <= p < 0.85)
    tab = vc.test_b_buckets(panel, b).set_index("bucket")
    mid = tab.loc["[0.30,0.45)":"[0.70,0.85)", "ratio_gauss"]
    assert ((mid > 0.85) & (mid < 1.15)).all()


def test_noise_inflates_integrated_slope():
    a, b, panel = _results(noise=0.02)
    assert a.attrs["slope"] > 1.4  # excess movement detected
    acf = vc.lambda_acf(panel)
    assert acf["acf_z_signed"].iloc[0] < -0.15  # bounce: negative signed lag-1


def test_stale_leaves_integrated_slope_unbiased():
    a, _, _ = _results(stale=0.5)
    assert 0.85 < a.attrs["slope"] < 1.20


def test_stochvol_shows_in_lambda_acf_only():
    _, _, panel = _results(stochvol=True)
    acf = vc.lambda_acf(panel)
    assert acf["acf_lambda"].iloc[0] > 0.15       # persistent vol clustering
    assert abs(acf["acf_z_signed"].iloc[0]) < 0.05  # no signed reversal


def test_power_fit_recovers_clock_and_scale():
    _, _, panel = _results()
    pf = vc.power_fit(panel)
    # probit truth: effective power between arcsin (1) and logit (2), clock ~ 1
    assert 1.3 < pf["gamma"] < 2.0
    assert 0.85 < pf["alpha"] < 1.15


def test_power_fit_noise_flattens_state_dependence():
    _, _, panel = _results(noise=0.02)
    pf = vc.power_fit(panel)
    assert pf["gamma"] < 1.0   # state-independent bounce drags gamma down
    assert pf["alpha"] < 0.6


def test_clean_acfs_are_flat():
    _, _, panel = _results()
    acf = vc.lambda_acf(panel)
    assert acf["acf_lambda"].abs().max() < 0.05
    assert acf["acf_z_signed"].abs().max() < 0.05
