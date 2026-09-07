r"""V2 core empirics: variance budget, local rates, power family, diagnostics.

Reruns the v1 headline empirics on the frozen v2 vintage (4,000 PM +
50,000 Kalshi resolved markets, ~10x the v1 PM sample), written from
scratch against analysis/core.py. Four blocks per venue:

(A) Integrated budget. For a martingale probability the lifetime squared
    movement is fully budgeted: E[sum dp^2 + (final - p_last)^2] =
    p0(1-p0). We report the through-origin slope of RV on IV = p0(1-p0)
    and the per-p0-bucket ratio of sums with market-level bootstrap CIs.
    Slope > 1 = excess movement (bounce / overreaction); staleness does
    not bias it.

(B) Local candidate rates. Which (p, tau) rate spends the budget?
      binom : p(1-p)/tau        each day a mini-resolution
      gauss : phi(Phi^-1(p))^2/tau   latent-Brownian beliefs, tau-clock
      wf    : p(1-p)            calendar-time constant intensity (free
                                scale; only bucket flatness is tested)
    binom/gauss are parameter-free so their through-origin slope should
    be ~1 under the matching model; bucket ratios test shape.

(C) Power family. E[dp^2/dt] = c [p(1-p)]^gamma / tau^alpha via
    weighted log-log LS on (pq, tau) cell means (cell means kill the
    E[log chi^2] != log E[chi^2] bias). gamma = 1 is the arcsin/binomial
    scale, gamma = 2 the logit-Brownian scale; alpha = 1 is the
    resolution clock, 0 calendar time. Reported for all p and for the
    bulk p in [0.15, 0.85] (v1: PM bulk 1.95/0.84, Kalshi bulk ~0.9/0.4).

(D) Diagnostics. Excess kurtosis of z = dp/sqrt(gauss rate) by bucket,
    and pooled within-market ACFs of z (signed; negative lag-1 = bounce)
    and lambda = z^2 (persistence = stochvol/ARCH), winsorised.

Every estimator is oracle-gated in tests/test_empirics_v2.py against the
exact-martingale simulator in core.py. Outputs land in results_v2/:
  empirics_budget_{label}.csv     per-bucket budget ratios + CIs
  empirics_local_{label}.csv      per-bucket candidate ratios
  empirics_scalars_{label}.json   slopes, power fits, ACF summary
  empirics_kurt_{label}.csv, empirics_acf_{label}.csv

Usage:
  python analysis/empirics_v2.py                 # sim gates + both venues
  python analysis/empirics_v2.py --venue kalshi  # one venue
  python analysis/empirics_v2.py --sim-only      # oracle blocks only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.config import RESULTS, SEED                      # noqa: E402
from analysis.core import (Market, P_CLIP, bucket, bucket_label,  # noqa: E402
                           load_markets, simulate)

BULK = (0.15, 0.85)          # p range for the bulk power fit
N_BOOT = 1000                # market-level bootstrap draws for budget CIs


# ------------------------------------------------------------------ helpers
def origin_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Through-origin OLS slope with heteroskedasticity-robust s.e."""
    sxx = float(x @ x)
    if sxx <= 0:
        return np.nan, np.nan
    b = float(x @ y) / sxx
    return b, float(np.sqrt(np.sum((x * (y - b * x)) ** 2))) / sxx


def move_panel(mkts: list[Market]) -> pd.DataFrame:
    """One row per daily move with candidate variance rates attached."""
    rows = []
    for m in mkts:
        p = m.closes[:-1]
        dp = np.diff(m.closes)
        dt = np.diff(m.ts_days)
        tau = np.maximum(m.t_res - m.ts_days[:-1], 1.0)
        keep = (p > P_CLIP[0]) & (p < P_CLIP[1]) & (dt > 0)
        if keep.any():
            rows.append(pd.DataFrame({
                "key": m.key, "p": p[keep], "dp": dp[keep],
                "dp2": dp[keep] ** 2, "dt": dt[keep], "tau": tau[keep]}))
    pan = pd.concat(rows, ignore_index=True)
    z = norm.ppf(pan["p"].to_numpy())
    pan["v_binom"] = pan.p * (1 - pan.p) / pan.tau * pan.dt
    pan["v_gauss"] = norm.pdf(z) ** 2 / pan.tau * pan.dt
    pan["v_wf"] = pan.p * (1 - pan.p) * pan.dt
    return pan


# ------------------------------------------------------------------ block A
def budget(mkts: list[Market], seed: int = SEED) -> tuple[dict, pd.DataFrame]:
    p0 = np.array([m.closes[0] for m in mkts])
    iv = p0 * (1 - p0)
    rv = np.array([np.sum(np.diff(m.closes) ** 2)
                   + (m.final - m.closes[-1]) ** 2 for m in mkts])
    slope, se = origin_slope(iv, rv)
    rng = np.random.default_rng(seed)
    rows = []
    for b in np.unique(bucket(p0)):
        sel = bucket(p0) == b
        r, i = rv[sel], iv[sel]
        ratio = r.sum() / i.sum()
        idx = rng.integers(0, len(r), (N_BOOT, len(r)))
        boots = r[idx].sum(1) / i[idx].sum(1)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        rows.append({"bucket": bucket_label(int(b)), "n": int(sel.sum()),
                     "ratio": ratio, "ci_lo": lo, "ci_hi": hi})
    return {"slope": slope, "se": se}, pd.DataFrame(rows)


# ------------------------------------------------------------------ block B
def local_rates(pan: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y = pan["dp2"].to_numpy()
    slopes = {}
    for name in ("v_binom", "v_gauss", "v_wf"):
        b, se = origin_slope(pan[name].to_numpy(), y)
        slopes[name] = {"slope": b, "se": se}
    rows = []
    for b, g in pan.groupby(bucket(pan["p"].to_numpy())):
        row = {"bucket": bucket_label(int(b)), "n": len(g)}
        for name in ("v_binom", "v_gauss", "v_wf"):
            scale = slopes[name]["slope"] if name == "v_wf" else 1.0
            denom = g[name].mean() * scale
            row[f"ratio_{name[2:]}"] = g["dp2"].mean() / denom \
                if denom > 0 else np.nan
        rows.append(row)
    return slopes, pd.DataFrame(rows)


# ------------------------------------------------------------------ block C
def power_fit(pan: pd.DataFrame, bulk: bool = False) -> dict:
    d = pan
    if bulk:
        d = d[(d.p >= BULK[0]) & (d.p < BULK[1])]
    pq = (d.p * (1 - d.p)).to_numpy()
    tau = d["tau"].to_numpy()
    y = (d.dp2 / d.dt).to_numpy()
    pq_bins = np.unique(np.quantile(pq, np.linspace(0, 1, 15)))
    tau_bins = np.exp(np.linspace(np.log(tau.min()),
                                  np.log(tau.max() + 1), 7))
    cells = pd.DataFrame({
        "ci": np.searchsorted(pq_bins, pq, side="right") - 1,
        "cj": np.searchsorted(tau_bins, tau, side="right") - 1,
        "y": y, "pq": pq, "tau": tau})
    g = cells.groupby(["ci", "cj"]).agg(
        n=("y", "size"), my=("y", "mean"),
        mpq=("pq", "mean"), mtau=("tau", "mean"))
    g = g[(g.n >= 50) & (g.my > 0)]
    if len(g) < 8:
        return {"gamma": np.nan, "alpha": np.nan, "n_cells": len(g)}
    X = np.column_stack([np.ones(len(g)), np.log(g.mpq), np.log(g.mtau)])
    w = np.sqrt(g.n.to_numpy())
    beta, *_ = np.linalg.lstsq(X * w[:, None], np.log(g.my) * w, rcond=None)
    return {"gamma": float(beta[1]), "alpha": float(-beta[2]),
            "c": float(np.exp(beta[0])), "n_cells": int(len(g))}


# ------------------------------------------------------------------ block D
def _pooled_acf(d: pd.DataFrame, col: str, max_lag: int) -> list[float]:
    v = d[col] - d.groupby("key")[col].transform("mean")
    denom = float(v @ v)
    grp = d["key"].to_numpy()
    same = [grp[lag:] == grp[:-lag] for lag in range(1, max_lag + 1)]
    a = v.to_numpy()
    return [float(np.sum(a[lag:][s] * a[:-lag][s])) / denom
            for lag, s in zip(range(1, max_lag + 1), same)]


def diagnostics(pan: pd.DataFrame,
                max_lag: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = pan["dp"] / np.sqrt(pan["v_gauss"].clip(lower=1e-12))
    kurt_rows = []
    for b, g in z.groupby(bucket(pan["p"].to_numpy())):
        a = g.to_numpy()
        if len(a) < 30:
            continue
        m2 = np.mean((a - a.mean()) ** 2)
        m4 = np.mean((a - a.mean()) ** 4)
        kurt_rows.append({"bucket": bucket_label(int(b)), "n": len(a),
                          "excess_kurt": m4 / m2 ** 2 - 3})
    d = pd.DataFrame({"key": pan["key"], "z": z, "lam": z ** 2})
    d["lam"] = d["lam"].clip(upper=d["lam"].quantile(0.99))
    d["z"] = d["z"].clip(d["z"].quantile(0.005), d["z"].quantile(0.995))
    acf = pd.DataFrame({"lag": range(1, max_lag + 1),
                        "acf_z": _pooled_acf(d, "z", max_lag),
                        "acf_lambda": _pooled_acf(d, "lam", max_lag)})
    return pd.DataFrame(kurt_rows), acf


# ------------------------------------------------------------------- driver
def run(mkts: list[Market], label: str, write: bool = True) -> dict:
    print(f"\n========== {label}: {len(mkts)} resolved markets ==========")
    a, abuckets = budget(mkts)
    print(f"budget slope = {a['slope']:.3f} (se {a['se']:.3f})"
          "   [martingale => 1]")
    print(abuckets.to_string(index=False,
                             float_format=lambda v: f"{v:.3f}"))
    pan = move_panel(mkts)
    slopes, bbuckets = local_rates(pan)
    for name in ("v_binom", "v_gauss", "v_wf"):
        s = slopes[name]
        print(f"local {name[2:]:6s} slope = {s['slope']:.3f}"
              f" (se {s['se']:.3f})")
    print(bbuckets.to_string(index=False,
                             float_format=lambda v: f"{v:.3f}"))
    pf_all = power_fit(pan)
    pf_bulk = power_fit(pan, bulk=True)
    print(f"power all : gamma {pf_all['gamma']:.2f}"
          f" alpha {pf_all['alpha']:.2f} ({pf_all['n_cells']} cells)")
    print(f"power bulk: gamma {pf_bulk['gamma']:.2f}"
          f" alpha {pf_bulk['alpha']:.2f} ({pf_bulk['n_cells']} cells)")
    kurt, acf = diagnostics(pan)
    print(kurt.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(acf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    scalars = {"n_markets": len(mkts), "n_moves": int(len(pan)),
               "budget_slope": a["slope"], "budget_se": a["se"],
               "local": slopes, "power_all": pf_all, "power_bulk": pf_bulk,
               "acf_z_lag1": float(acf.acf_z.iloc[0]),
               "acf_lambda_lag1": float(acf.acf_lambda.iloc[0])}
    if write:
        RESULTS.mkdir(exist_ok=True)
        abuckets.to_csv(RESULTS / f"empirics_budget_{label}.csv", index=False)
        bbuckets.to_csv(RESULTS / f"empirics_local_{label}.csv", index=False)
        kurt.to_csv(RESULTS / f"empirics_kurt_{label}.csv", index=False)
        acf.to_csv(RESULTS / f"empirics_acf_{label}.csv", index=False)
        with open(RESULTS / f"empirics_scalars_{label}.json", "w") as fh:
            json.dump(scalars, fh, indent=1)
    return scalars


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=["polymarket", "kalshi"])
    ap.add_argument("--sim-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.venue:
        for tag, kw in (("sim_clean", {}), ("sim_noise", {"noise": 0.02}),
                        ("sim_stochvol", {"stochvol": True})):
            run(simulate(2000, **kw), tag)
        if args.sim_only:
            return 0
    venues = [args.venue] if args.venue else ["polymarket", "kalshi"]
    for v in venues:
        run(load_markets(v), v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
