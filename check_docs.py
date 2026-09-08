"""v2 doc-number checker: every headline number in README/NOTES must trace
to a file in results_v2/.

The v1 build drifted because prose numbers came from runs at slightly
different settings. v2 rule: any number quoted in README.md or NOTES.md is
registered here as a Claim, and this script recomputes it from the results
files and fails loudly on mismatch. Run it in CI-style before any commit
that touches docs:

    python check_docs.py

Each phase that lands a headline number appends to CLAIMS. A Claim is
(label, compute, quoted, tol) where compute() reads results_v2/ and
returns a float, quoted is the number as it appears in the docs, and tol
is the acceptable absolute difference (respect the docs' rounding).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

from analysis.config import RESULTS  # noqa: F401 - used by claim closures


@dataclass
class Claim:
    label: str                    # where the number appears, e.g. "README budget PM"
    compute: Callable[[], float]  # recompute from results_v2/
    quoted: float                 # the number as written in the docs
    tol: float                    # allowed |computed - quoted|


# Populated phase by phase (V2 onward). Empty registry passes trivially,
# but --strict below refuses to bless an empty one so this can't rot.
CLAIMS: list[Claim] = []


def _scalar(label: str, path: tuple[str, ...]):
    """Reader for results_v2/empirics_scalars_{label}.json."""
    def compute() -> float:
        import json
        with open(RESULTS / f"empirics_scalars_{label}.json") as fh:
            d = json.load(fh)
        for k in path:
            d = d[k]
        return float(d)
    return compute


def _jscalar(fname: str, path: tuple):
    """Reader for an arbitrary JSON file in results_v2/ (keys or list indices)."""
    def compute() -> float:
        import json
        with open(RESULTS / fname) as fh:
            d = json.load(fh)
        for k in path:
            d = d[k]
        return float(d)
    return compute


# ---- V2 core empirics (analysis/empirics_v2.py, run 2026-09-07)
CLAIMS += [
    Claim("V2 PM budget slope", _scalar("polymarket", ("budget_slope",)),
          1.19, 0.005),
    Claim("V2 Kalshi budget slope", _scalar("kalshi", ("budget_slope",)),
          1.432, 0.0005),
    Claim("V2 PM bulk gamma", _scalar("polymarket", ("power_bulk", "gamma")),
          1.51, 0.005),
    Claim("V2 Kalshi bulk gamma", _scalar("kalshi", ("power_bulk", "gamma")),
          1.035, 0.0005),
    Claim("V2 PM bulk alpha", _scalar("polymarket", ("power_bulk", "alpha")),
          0.53, 0.005),
    Claim("V2 Kalshi z-ACF lag 1", _scalar("kalshi", ("acf_z_lag1",)),
          -0.280, 0.0005),
    Claim("V2 Kalshi lambda-ACF lag 1", _scalar("kalshi", ("acf_lambda_lag1",)),
          0.217, 0.0005),
    Claim("V2 PM market count", _scalar("polymarket", ("n_markets",)),
          3382, 0),
    Claim("V2 Kalshi market count", _scalar("kalshi", ("n_markets",)),
          27465, 0),
]

# ---- V3 regime map + per-regime models (analysis/regimes_v3.py, run 2026-09-07)
# SLV variants list order: [full, no-sv, no-noise].
CLAIMS += [
    Claim("V3 PM R1 budget share",
          _jscalar("regimes_scalars_polymarket.json",
                   ("budget_by_regime", "R1 diffusive")), 0.613, 0.0005),
    Claim("V3 Kalshi R1 budget share",
          _jscalar("regimes_scalars_kalshi.json",
                   ("budget_by_regime", "R1 diffusive")), 0.833, 0.0005),
    Claim("V3 PM SLV segments",
          _jscalar("slv_polymarket.json", ("n_segments",)), 1158, 0),
    Claim("V3 Kalshi SLV segments",
          _jscalar("slv_kalshi.json", ("n_segments",)), 1881, 0),
    Claim("V3 PM SLV full ll/obs",
          _jscalar("slv_polymarket.json", ("variants", 0, "ll_per_obs")),
          1.672, 0.0005),
    Claim("V3 Kalshi SLV full ll/obs",
          _jscalar("slv_kalshi.json", ("variants", 0, "ll_per_obs")),
          1.142, 0.0005),
    Claim("V3 PM SLV no-sv d_ll",
          _jscalar("slv_polymarket.json", ("variants", 1, "d_ll")),
          -0.649, 0.0005),
    Claim("V3 Kalshi SLV no-sv d_ll",
          _jscalar("slv_kalshi.json", ("variants", 1, "d_ll")),
          -0.624, 0.0005),
    # Boundary solution, documented as a finding (grid probe 2026-09-07:
    # not a quadrature artifact; s re-pins at any S_MAX).
    Claim("V3 PM SLV s pinned at S_MAX",
          _jscalar("slv_polymarket.json", ("variants", 0, "s")), 2.0, 1e-6),
    Claim("V3 Kalshi SLV s pinned at S_MAX",
          _jscalar("slv_kalshi.json", ("variants", 0, "s")), 2.0, 1e-6),
    Claim("V3 PM endgame total gamma",
          _jscalar("endgame_polymarket.json", ("total", "g")), 1.510, 0.0005),
    Claim("V3 Kalshi endgame total gamma",
          _jscalar("endgame_kalshi.json", ("total", "g")), 0.614, 0.0005),
]

# ---- V4 zero-inflated layer + horse race + budget (analysis/race_v4.py,
# run 2026-09-07). zi variants list order: [zi, zi-off]. race models list
# order: [composite-zi, composite, global-slv, const-lam, bucket, garch].
CLAIMS += [
    # s unpins on both venues once the observation lives on the tick grid;
    # PM via staleness mass, Kalshi via tick censoring alone.
    Claim("V4 PM zi s interior",
          _jscalar("zi_polymarket.json", ("variants", 0, "s")), 2.104, 0.0005),
    Claim("V4 PM zi pi0",
          _jscalar("zi_polymarket.json", ("variants", 0, "pi0")),
          0.284, 0.0005),
    Claim("V4 PM zi-off d_ll",
          _jscalar("zi_polymarket.json", ("variants", 1, "d_ll")),
          -0.275, 0.0005),
    # Corrected universe (V1.1): the Kalshi zi fit degenerates (pi0 -> 0,
    # s runs to S_MAX4); the tick-mass measure ALONE unpins s, so the
    # headline interior s is the zi-off variant's.
    Claim("V4 Kalshi zi-off s interior",
          _jscalar("zi_kalshi.json", ("variants", 1, "s")), 2.557, 0.001),
    Claim("V4 Kalshi zi pi0 negligible",
          _jscalar("zi_kalshi.json", ("variants", 0, "pi0")), 0.000, 0.0005),
    Claim("V4 Kalshi zi-off d_ll ~ 0",
          _jscalar("zi_kalshi.json", ("variants", 1, "d_ll")), 0.002, 0.0005),
    Claim("V4 PM race n_obs",
          _jscalar("race_summary_polymarket.json", ("n_obs",)), 117252, 0),
    Claim("V4 Kalshi race n_obs",
          _jscalar("race_summary_kalshi.json", ("n_obs",)), 65651, 0),
    Claim("V4 PM composite-zi delta",
          _jscalar("race_summary_polymarket.json",
                   ("models", 0, "delta_vs_composite")), 0.069, 0.0005),
    Claim("V4 PM global-slv delta",
          _jscalar("race_summary_polymarket.json",
                   ("models", 2, "delta_vs_composite")), -4.101, 0.001),
    Claim("V4 Kalshi composite-zi delta",
          _jscalar("race_summary_kalshi.json",
                   ("models", 0, "delta_vs_composite")), 0.076, 0.0005),
    Claim("V4 Kalshi global-slv delta",
          _jscalar("race_summary_kalshi.json",
                   ("models", 2, "delta_vs_composite")), 0.211, 0.0005),
    Claim("V4 PM budget 30d full ratio",
          _jscalar("budget_v4_polymarket.json", ("tau~30d", "full_ratio")),
          1.157, 0.0005),
    Claim("V4 PM budget 30d realized ratio",
          _jscalar("budget_v4_polymarket.json", ("tau~30d", "real_ratio")),
          1.174, 0.0005),
    Claim("V4 Kalshi budget 30d full ratio",
          _jscalar("budget_v4_kalshi.json", ("tau~30d", "full_ratio")),
          1.375, 0.0005),
    Claim("V4 Kalshi budget 30d realized ratio",
          _jscalar("budget_v4_kalshi.json", ("tau~30d", "real_ratio")),
          1.597, 0.0005),
]

# ---- V5 cross-venue bridge (analysis/bridge_v5.py, run 2026-09-07).
# Daily alignment is wall-clock (Kalshi labels shifted +1 day; Kalshi daily
# candles are start-stamped, PM daily bars are 00 UTC point samples).
CLAIMS += [
    Claim("V5 pair count",
          _jscalar("bridge_v5.json", ("pairs", "n")), 82, 0),
    Claim("V5 exact pair count",
          _jscalar("bridge_v5.json", ("pairs", "n_exact")), 64, 0),
    Claim("V5 mean abs gap",
          _jscalar("bridge_v5.json", ("gap", "mean_abs_gap")),
          0.024, 0.0005),
    Claim("V5 gap frac > 5c",
          _jscalar("bridge_v5.json", ("gap", "frac_gt_5c")),
          0.086, 0.0005),
    Claim("V5 gap half-life days",
          _jscalar("bridge_v5.json", ("gap", "gap_half_life_days")),
          4.58, 0.005),
    Claim("V5 budget ratio-of-ratios (K/PM)",
          _jscalar("bridge_v5.json",
                   ("paired_budget", "median_ratio_of_ratios")),
          1.457, 0.0005),
    Claim("V5 budget frac pairs K > PM",
          _jscalar("bridge_v5.json", ("paired_budget", "frac_k_gt_pm")),
          0.720, 0.0005),
    Claim("V5 matched-panel gamma Kalshi",
          _jscalar("bridge_v5.json", ("paired_power", "kalshi", "gamma")),
          1.344, 0.0005),
    Claim("V5 matched-panel gamma PM",
          _jscalar("bridge_v5.json", ("paired_power", "polymarket", "gamma")),
          1.531, 0.0005),
    # Hourly panel: PM leads Kalshi; the reverse coefficient is an order
    # of magnitude smaller. Daily same-label lead-lag is stamp mechanics.
    Claim("V5 hourly PM->K cross lag coef",
          _jscalar("bridge_v5.json",
                   ("lead_lag_hourly", "pm_predicts_k", "cross_lag")),
          0.239, 0.0005),
    Claim("V5 hourly PM->K cross t",
          _jscalar("bridge_v5.json",
                   ("lead_lag_hourly", "pm_predicts_k", "cross_t")),
          57.9, 0.05),
    Claim("V5 hourly K->PM cross lag coef",
          _jscalar("bridge_v5.json",
                   ("lead_lag_hourly", "k_predicts_pm", "cross_lag")),
          0.018, 0.0005),
    Claim("V5 hourly contemporaneous xcorr",
          _jscalar("bridge_v5.json",
                   ("lead_lag_hourly", "xcorr_dpm_leads_dk", "0")),
          0.244, 0.0005),
    Claim("V5 hourly n_obs",
          _jscalar("bridge_v5.json", ("lead_lag_hourly", "n_obs")),
          198224, 0),
]


def _csv(fname: str, where: dict[str, str], field: str):
    """Reader for a CSV in results_v2/: first row matching `where`, column
    `field`."""
    def compute() -> float:
        import csv
        with open(RESULTS / fname, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if all(row[k] == v for k, v in where.items()):
                    return float(row[field])
        raise KeyError(f"{fname}: no row matching {where}")
    return compute


# ---- V6 docs pass (README/NOTES rewrite, 2026-09-07): claims for every
# number newly quoted in the docs. Data-pull provenance counts (catalog
# rows, bar-file counts) are deliberately NOT registered: they live in
# data_v2/, and this checker must run from committed results_v2/ alone.
CLAIMS += [
    # core empirics
    Claim("V6 PM move count", _scalar("polymarket", ("n_moves",)), 219593, 0),
    Claim("V6 Kalshi move count", _scalar("kalshi", ("n_moves",)), 708078, 0),
    Claim("V6 PM z-ACF lag 1", _scalar("polymarket", ("acf_z_lag1",)),
          -0.115, 0.0005),
    Claim("V6 PM lambda-ACF lag 1", _scalar("polymarket", ("acf_lambda_lag1",)),
          0.188, 0.0005),
    Claim("V6 PM budget se", _scalar("polymarket", ("budget_se",)),
          0.021, 0.0005),
    Claim("V6 Kalshi budget se", _scalar("kalshi", ("budget_se",)),
          0.009, 0.0005),
    Claim("V6 Kalshi bulk alpha", _scalar("kalshi", ("power_bulk", "alpha")),
          0.51, 0.005),
    Claim("V6 sim clean budget slope", _scalar("sim_clean", ("budget_slope",)),
          0.986, 0.0005),
    Claim("V6 sim noise budget slope", _scalar("sim_noise", ("budget_slope",)),
          1.207, 0.0005),
    Claim("V6 sim noise z-ACF lag 1", _scalar("sim_noise", ("acf_z_lag1",)),
          -0.195, 0.0005),
    Claim("V6 sim stochvol lambda-ACF lag 1",
          _scalar("sim_stochvol", ("acf_lambda_lag1",)), 0.126, 0.0005),
    Claim("V6 sim clean bulk gamma",
          _scalar("sim_clean", ("power_bulk", "gamma")), 1.57, 0.005),
    Claim("V6 sim clean bulk alpha",
          _scalar("sim_clean", ("power_bulk", "alpha")), 1.00, 0.005),
    Claim("V6 sim noise bulk gamma",
          _scalar("sim_noise", ("power_bulk", "gamma")), 1.11, 0.005),
    Claim("V6 sim noise bulk alpha",
          _scalar("sim_noise", ("power_bulk", "alpha")), 0.82, 0.005),
    Claim("V6 Kalshi longshot bucket ratio",
          _csv("empirics_budget_kalshi.csv",
               {"bucket": "[0.00,0.05)"}, "ratio"), 7.37, 0.005),
    Claim("V6 Kalshi favorite bucket ratio",
          _csv("empirics_budget_kalshi.csv",
               {"bucket": "[0.95,1.00)"}, "ratio"), 12.91, 0.005),
    # regimes
    Claim("V6 PM R2 budget share",
          _jscalar("regimes_scalars_polymarket.json",
                   ("budget_by_regime", "R2 itm/hazard")), 0.064, 0.0005),
    Claim("V6 PM R3 budget share",
          _jscalar("regimes_scalars_polymarket.json",
                   ("budget_by_regime", "R3 jumpy")), 0.323, 0.0005),
    Claim("V6 Kalshi R2 budget share",
          _jscalar("regimes_scalars_kalshi.json",
                   ("budget_by_regime", "R2 itm/hazard")), 0.039, 0.0005),
    Claim("V6 Kalshi R3 budget share",
          _jscalar("regimes_scalars_kalshi.json",
                   ("budget_by_regime", "R3 jumpy")), 0.128, 0.0005),
    Claim("V6 sim clean R1 share",
          _jscalar("regimes_scalars_sim_clean.json",
                   ("budget_by_regime", "R1 diffusive")), 0.997, 0.0005),
    Claim("V6 sim stale R2 share",
          _jscalar("regimes_scalars_sim_stale.json",
                   ("budget_by_regime", "R2 itm/hazard")), 0.993, 0.0005),
    # SLV
    Claim("V6 PM SLV steps",
          _jscalar("slv_polymarket.json", ("n_steps",)), 35299, 0),
    Claim("V6 Kalshi SLV steps",
          _jscalar("slv_kalshi.json", ("n_steps",)), 40046, 0),
    Claim("V6 PM SLV phi",
          _jscalar("slv_polymarket.json", ("variants", 0, "phi")),
          0.78, 0.005),
    Claim("V6 Kalshi SLV phi",
          _jscalar("slv_kalshi.json", ("variants", 0, "phi")), 0.71, 0.005),
    # endgame arrival/size split
    Claim("V6 PM endgame arrival g",
          _jscalar("endgame_polymarket.json", ("arrival", "g")), 0.32, 0.005),
    Claim("V6 PM endgame size g",
          _jscalar("endgame_polymarket.json", ("size", "g")), 0.95, 0.005),
    Claim("V6 Kalshi endgame arrival g",
          _jscalar("endgame_kalshi.json", ("arrival", "g")), 0.58, 0.005),
    Claim("V6 Kalshi endgame size g",
          _jscalar("endgame_kalshi.json", ("size", "g")), 0.00, 0.005),
    # ITM hazard headline cells
    Claim("V6 Kalshi hazard cell pi0",
          _csv("hazard_kalshi.csv",
               {"dist_bucket": "[0.00,0.05)", "tau_band": "[1,3)d"}, "pi0"),
          0.45, 0.005),
    Claim("V6 Kalshi hazard cell h/day",
          _csv("hazard_kalshi.csv",
               {"dist_bucket": "[0.00,0.05)", "tau_band": "[1,3)d"}, "h_day"),
          0.21, 0.005),
    Claim("V6 Kalshi hazard cell gap ticks",
          _csv("hazard_kalshi.csv",
               {"dist_bucket": "[0.00,0.05)", "tau_band": "[1,3)d"},
               "gap_mean_tk"), 23.9, 0.05),
    Claim("V6 Kalshi hazard cell w_away",
          _csv("hazard_kalshi.csv",
               {"dist_bucket": "[0.00,0.05)", "tau_band": "[1,3)d"}, "w_away"),
          0.52, 0.005),
    Claim("V6 PM hazard cell gap ticks",
          _csv("hazard_polymarket.csv",
               {"dist_bucket": "[0.00,0.05)", "tau_band": "[3,7)d"},
               "gap_mean_tk"), 30.6, 0.05),
    Claim("V6 PM hazard cell w_away",
          _csv("hazard_polymarket.csv",
               {"dist_bucket": "[0.00,0.05)", "tau_band": "[3,7)d"}, "w_away"),
          0.23, 0.005),
    # race deltas not previously claimed
    Claim("V6 PM const-lam delta",
          _jscalar("race_summary_polymarket.json",
                   ("models", 3, "delta_vs_composite")), -1.572, 0.0005),
    Claim("V6 PM bucket delta",
          _jscalar("race_summary_polymarket.json",
                   ("models", 4, "delta_vs_composite")), -1.122, 0.0005),
    Claim("V6 PM garch delta",
          _jscalar("race_summary_polymarket.json",
                   ("models", 5, "delta_vs_composite")), -0.733, 0.0005),
    Claim("V6 Kalshi const-lam delta",
          _jscalar("race_summary_kalshi.json",
                   ("models", 3, "delta_vs_composite")), -0.316, 0.0005),
    Claim("V6 Kalshi bucket delta",
          _jscalar("race_summary_kalshi.json",
                   ("models", 4, "delta_vs_composite")), -0.665, 0.0005),
    Claim("V6 Kalshi garch delta",
          _jscalar("race_summary_kalshi.json",
                   ("models", 5, "delta_vs_composite")), -0.826, 0.0005),
    Claim("V6 PM 30d zi jump share",
          _jscalar("budget_v4_polymarket.json",
                   ("tau~30d", "zi_jump_share")), 0.995, 0.0005),
    # bridge
    Claim("V6 gap n_obs",
          _jscalar("bridge_v5.json", ("gap", "n_obs")), 9341, 0),
    Claim("V6 median abs gap",
          _jscalar("bridge_v5.json", ("gap", "median_abs_gap")),
          0.0105, 0.0001),
    Claim("V6 mean gap exact pairs",
          _jscalar("bridge_v5.json",
                   ("gap", "mean_abs_gap_by_quality", "exact")),
          0.019, 0.0005),
    Claim("V6 mean gap approx pairs",
          _jscalar("bridge_v5.json",
                   ("gap", "mean_abs_gap_by_quality", "approx")),
          0.036, 0.0005),
    Claim("V6 mean gap election domain",
          _jscalar("bridge_v5.json",
                   ("gap", "mean_abs_gap_by_domain", "election")),
          0.097, 0.0005),
    Claim("V6 gap AR1 rho",
          _jscalar("bridge_v5.json", ("gap", "gap_ar1_rho")), 0.860, 0.0005),
    Claim("V6 daily K->PM cross lag (stamp mechanics)",
          _jscalar("bridge_v5.json",
                   ("lead_lag_daily", "k_predicts_pm", "cross_lag")),
          0.204, 0.0005),
    Claim("V6 daily K->PM cross t (stamp mechanics)",
          _jscalar("bridge_v5.json",
                   ("lead_lag_daily", "k_predicts_pm", "cross_t")),
          20.6, 0.05),
]

# pair counts by domain
for _dom, _n in [("fed_decision", 47), ("fed_chair_nom", 14),
                 ("btc_monthly", 8), ("btc_yearly", 6), ("election", 7)]:
    CLAIMS.append(Claim(
        f"V6 pair count {_dom}",
        _jscalar("bridge_v5.json", ("pairs", "by_domain", _dom)), _n, 0))

# endgame lifetime-spend shares (NOTES table, both venues, all bands)
for _venue, _shares in [
    ("polymarket", {"tau<=1d": 0.112, "tau(1,3]d": 0.050, "tau(3,7]d": 0.101,
                    "tau(7,30]d": 0.140, "tau>30d": 0.207,
                    "terminal jump": 0.390}),
    ("kalshi", {"tau<=1d": 0.535, "tau(1,3]d": 0.092, "tau(3,7]d": 0.070,
                "tau(7,30]d": 0.114, "tau>30d": 0.150,
                "terminal jump": 0.039}),
]:
    for _band, _q in _shares.items():
        CLAIMS.append(Claim(
            f"V6 {_venue} spend share {_band}",
            _csv(f"endgame_spend_{_venue}.csv", {"band": _band}, "share"),
            _q, 0.0005))

# budget-integration table (NOTES, both venues, all bands x engines)
for _venue, _rows in [
    ("polymarket", {"tau~30d": (1.157, 1.174, 1.163, 1.174),
                    "tau~14d": (1.077, 1.085, 1.077, 1.158),
                    "tau~7d": (1.041, 1.044, 1.041, 1.128),
                    "tau~3d": (1.021, 1.023, 1.021, 1.060)}),
    ("kalshi", {"tau~30d": (1.375, 1.537, 1.335, 1.597),
                "tau~14d": (1.112, 1.205, 1.096, 1.329),
                "tau~7d": (1.073, 1.117, 1.064, 1.204),
                "tau~3d": (1.032, 1.056, 1.029, 1.124)}),
]:
    for _band, (_full, _zi, _clam, _real) in _rows.items():
        for _field, _q in [("full_ratio", _full), ("zi_ratio", _zi),
                           ("clam_ratio", _clam), ("real_ratio", _real)]:
            CLAIMS.append(Claim(
                f"V6 {_venue} {_band} {_field}",
                _jscalar(f"budget_v4_{_venue}.json", (_band, _field)),
                _q, 0.0005))


# ---- V6.5 structure-first clustering (analysis/cluster_v65.py, run
# 2026-09-08). Cards list index == cluster id (cards are sorted); power list
# likewise. Fit-vs-fallback cell counts (3/8 PM, 4/8 Kalshi etc.) come from
# the run logs, not results_v2/, and are deliberately not registered, like
# the data-pull provenance counts.


def _ari_sim2pop(collapsed: bool):
    """ARI of the stored 2-pop sim clustering vs truth (key prefix n_)."""
    def compute() -> float:
        import csv
        from collections import Counter, defaultdict
        rows = []
        with open(RESULTS / "cluster65_features_sim_2pop.csv",
                  encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append((int(float(row["cluster"])),
                             int(row["key"].startswith("n_"))))
        lab = [c for c, _ in rows]
        if collapsed:
            votes = defaultdict(list)
            for c, t in rows:
                votes[c].append(t)
            maj = {c: int(sum(v) * 2 > len(v)) for c, v in votes.items()}
            lab = [maj[c] for c in lab]
        truth = [t for _, t in rows]
        ct = Counter(zip(lab, truth))
        comb = lambda x: x * (x - 1) / 2                      # noqa: E731
        s_ij = sum(comb(v) for v in ct.values())
        ra = Counter(lab)
        rb = Counter(truth)
        s_a = sum(comb(v) for v in ra.values())
        s_b = sum(comb(v) for v in rb.values())
        exp = s_a * s_b / comb(len(rows))
        mx = 0.5 * (s_a + s_b)
        return float((s_ij - exp) / (mx - exp))
    return compute


CLAIMS += [
    Claim("V6.5 sim 2-pop raw ARI", _ari_sim2pop(False), 0.411, 0.0005),
    Claim("V6.5 sim 2-pop majority-collapsed ARI", _ari_sim2pop(True),
          0.955, 0.0005),
    Claim("V6.5 sim 2-pop chosen k",
          _jscalar("cluster65_scalars_sim_2pop.json", ("k",)), 8, 0),
    Claim("V6.5 sim 1-pop chosen k",
          _jscalar("cluster65_scalars_sim_1pop.json", ("k",)), 7, 0),
]

for _venue, _short, _k, _ari in [("polymarket", "PM", 8, 0.007),
                                 ("kalshi", "Kalshi", 8, 0.016)]:
    _f = f"cluster65_scalars_{_venue}.json"
    CLAIMS += [
        Claim(f"V6.5 {_short} chosen k", _jscalar(_f, ("k",)), _k, 0),
        Claim(f"V6.5 {_short} ARI vs category",
              _jscalar(_f, ("ari_vs_category",)), _ari, 0.0005),
    ]

# per-cluster cards quoted in README/NOTES
for _lbl, _path, _q, _tol in [
    # Polymarket: excess-movement vs dead-until-resolution vs terminal type
    ("V6.5 PM cl1 share",
     ("cluster65_scalars_polymarket.json", ("cards", 1, "share")),
     0.152, 0.0005),
    ("V6.5 PM cl1 budget slope",
     ("cluster65_scalars_polymarket.json", ("cards", 1, "budget_slope")),
     1.80, 0.005),
    ("V6.5 PM cl4 share",
     ("cluster65_scalars_polymarket.json", ("cards", 4, "share")),
     0.118, 0.0005),
    ("V6.5 PM cl4 still frac",
     ("cluster65_scalars_polymarket.json", ("cards", 4, "still")),
     0.963, 0.0005),
    ("V6.5 PM cl4 budget slope",
     ("cluster65_scalars_polymarket.json", ("cards", 4, "budget_slope")),
     0.71, 0.005),
    ("V6.5 PM cl3 share",
     ("cluster65_scalars_polymarket.json", ("cards", 3, "share")),
     0.008, 0.0005),
    ("V6.5 PM cl3 terminal share",
     ("cluster65_scalars_polymarket.json", ("cards", 3, "term_share")),
     0.995, 0.0005),
    # Kalshi: the two excess carriers and the modal cluster
    ("V6.5 Kalshi cl2 share",
     ("cluster65_scalars_kalshi.json", ("cards", 2, "share")),
     0.110, 0.0005),
    ("V6.5 Kalshi cl2 budget slope",
     ("cluster65_scalars_kalshi.json", ("cards", 2, "budget_slope")),
     3.50, 0.005),
    ("V6.5 Kalshi cl4 share",
     ("cluster65_scalars_kalshi.json", ("cards", 4, "share")),
     0.049, 0.0005),
    ("V6.5 Kalshi cl4 budget slope",
     ("cluster65_scalars_kalshi.json", ("cards", 4, "budget_slope")),
     2.62, 0.005),
    ("V6.5 Kalshi cl5 share",
     ("cluster65_scalars_kalshi.json", ("cards", 5, "share")),
     0.330, 0.0005),
    ("V6.5 Kalshi cl5 budget slope",
     ("cluster65_scalars_kalshi.json", ("cards", 5, "budget_slope")),
     1.19, 0.005),
]:
    CLAIMS.append(Claim(_lbl, _jscalar(*_path), _q, _tol))

# per-cluster power gammas quoted in NOTES (all fittable Kalshi clusters,
# the two large fittable PM clusters, and the degenerate PM zero-stillness
# fit, reported as-is)
for _venue, _cl, _q in [("kalshi", 0, 1.85), ("kalshi", 2, 1.53),
                        ("kalshi", 3, 0.58), ("kalshi", 4, 2.30),
                        ("kalshi", 5, 1.53), ("kalshi", 6, 1.46),
                        ("kalshi", 7, 1.25),
                        ("polymarket", 1, 1.38), ("polymarket", 7, 1.87),
                        ("polymarket", 6, -1.86)]:
    CLAIMS.append(Claim(
        f"V6.5 {_venue} cl{_cl} power gamma",
        _jscalar(f"cluster65_scalars_{_venue}.json",
                 ("power", _cl, "gamma")), _q, 0.005))

# partitioned SLV gate (held-out ll/obs table in README/NOTES)
for _venue, _part, _q in [("polymarket", "global", 1.616),
                          ("polymarket", "category", 1.634),
                          ("polymarket", "cluster", 1.620),
                          ("kalshi", "global", 1.156),
                          ("kalshi", "category", 1.166),
                          ("kalshi", "cluster", 1.175)]:
    CLAIMS.append(Claim(
        f"V6.5 {_venue} SLV gate {_part}",
        _jscalar(f"cluster65_scalars_{_venue}.json",
                 ("slv_gate", _part, "heldout_ll_per_obs")), _q, 0.0005))


def main() -> int:
    strict = "--strict" in sys.argv
    if not CLAIMS:
        print("check_docs: no claims registered yet (V0 scaffold).")
        return 1 if strict else 0
    bad = 0
    for c in CLAIMS:
        got = c.compute()
        ok = abs(got - c.quoted) <= c.tol
        flag = "ok " if ok else "FAIL"
        print(f"[{flag}] {c.label}: quoted {c.quoted} computed {got:.6g} "
              f"(tol {c.tol})")
        bad += not ok
    print(f"check_docs: {len(CLAIMS) - bad}/{len(CLAIMS)} claims verified.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
