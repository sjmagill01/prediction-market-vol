"""ITM-regime model: frozen prices + hazard-driven gaps. P2 of the modeling phase.

The regime map (analysis/regime_map.py) shows R2 cells -- p near a boundary
with plenty of time left, plus Kalshi's dormant long-dated coin flips -- where
the price is still far beyond what diffusion predicts and moves, when they
come, are gaps. The credit analogy: a decided-but-not-resolved market is a
bond trading near par; nothing happens until a reassessment event. So model
the daily increment on the venue tick grid directly (ticks k = dp / unit),
folded so positive k = away from the nearest boundary:

  P(k) = pi0 * 1{k=0}                          frozen day (no news, no trade)
       + (1 - pi0 - ph) * discN(k; sigma)      tick-censored micro-diffusion
       + ph * gap(k; rho_a, rho_t, w_a)        reassessment gap

with gap() a two-sided geometric (discrete Laplace) supported on |k| >= 2
ticks, with separate decay for away-moves (rho_a, weight w_a: the reversal
direction) and toward-moves (rho_t): heavier away-tails = jump-back-to-life
risk, the analogue of jump-to-default. The |k| >= 2 support is an
identification choice: 1-tick moves belong to the diffusion component
(without it the oracle confounds small gaps with diffusion). ph is a per-day hazard (observations are filtered to
dt <= 1.5 days, so probability ~ hazard). Fitted per (boundary-distance
bucket, tau band) cell by discrete MLE; every parameter is interpretable and
the model's implied variance is checked against the cell's empirical variance
(the budget check: nothing is smuggled).

Approximations, stated: prices are rounded to the venue tick (Kalshi 1c,
Polymarket 0.1c since sub-cent prints are common near the boundary); the
toward-boundary geometric ignores truncation at the boundary itself; high-p
cells are folded onto low-p by symmetry.

Oracle gate (--oracle): simulate cells from this exact pmf with known
parameters and check the MLE recovers them before fitting real data.

Usage:
  python analysis/itm_hazard.py --oracle [--seed 7]
  python analysis/itm_hazard.py --venue polymarket|kalshi|both
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import regime_map as rm  # noqa: E402
from analysis import vol_check as vc   # noqa: E402

UNIT = {"polymarket": 0.001, "kalshi": 0.01}
DIST_EDGES = np.array([0.0, 0.05, 0.15, 0.30, 0.45, 0.501])  # dist to boundary
MAX_DT = 1.5         # keep ~daily observations so ph reads as a daily hazard
MIN_CELL = 500
PARAMS = ("pi0", "ph", "sigma", "rho_a", "rho_t", "w_a")


# ------------------------------------------------------------------ pmf / MLE

def pmf(k: np.ndarray, th: dict) -> np.ndarray:
    """P(tick move = k) under frozen + diffusion + gap mixture."""
    pi0, ph, sig = th["pi0"], th["ph"], th["sigma"]
    diff_w = max(1 - pi0 - ph, 0.0)
    p = diff_w * (norm.cdf((k + 0.5) / sig) - norm.cdf((k - 0.5) / sig))
    p = p + pi0 * (k == 0)
    ka = np.abs(k)
    gap = np.where(k >= 2, th["w_a"] * (1 - th["rho_a"]) * th["rho_a"] ** (ka - 2),
                   np.where(k <= -2,
                            (1 - th["w_a"]) * (1 - th["rho_t"]) * th["rho_t"] ** (ka - 2),
                            0.0))
    return p + ph * gap


def _unpack(u: np.ndarray) -> dict:
    e0, e1 = np.exp(u[0]), np.exp(u[1])
    den = 1 + e0 + e1
    return {"pi0": e0 / den, "ph": e1 / den, "sigma": np.exp(u[2]),
            "rho_a": 1 / (1 + np.exp(-u[3])), "rho_t": 1 / (1 + np.exp(-u[4])),
            "w_a": 1 / (1 + np.exp(-u[5]))}


def _pack(th: dict) -> np.ndarray:
    diff_w = max(1 - th["pi0"] - th["ph"], 1e-6)
    lg = lambda q: np.log(q / (1 - q))
    return np.array([np.log(th["pi0"] / diff_w), np.log(th["ph"] / diff_w),
                     np.log(th["sigma"]), lg(th["rho_a"]), lg(th["rho_t"]),
                     lg(th["w_a"])])


def fit_cell(k: np.ndarray) -> dict:
    """Discrete MLE on tick moves; likelihood evaluated on unique values."""
    vals, counts = np.unique(k, return_counts=True)

    def nll(u):
        p = pmf(vals, _unpack(u))
        return -float(counts @ np.log(p + 1e-300)) / len(k)

    start = _pack({"pi0": 0.5, "ph": 0.05, "sigma": 1.0,
                   "rho_a": 0.7, "rho_t": 0.7, "w_a": 0.5})
    res = minimize(nll, start, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-7})
    th = _unpack(res.x)
    th["ll_per_obs"] = -res.fun
    th["converged"] = bool(res.success)
    return th


def summarise(th: dict, k: np.ndarray) -> dict:
    """Interpretable per-cell readout + the budget check."""
    ks = np.arange(2, 2001, dtype=float)  # gap support |k| >= 2, numeric moments
    pa = (1 - th["rho_a"]) * th["rho_a"] ** (ks - 2)
    pt = (1 - th["rho_t"]) * th["rho_t"] ** (ks - 2)
    e_abs = th["w_a"] * (pa @ ks) + (1 - th["w_a"]) * (pt @ ks)
    e_sq = th["w_a"] * (pa @ ks ** 2) + (1 - th["w_a"]) * (pt @ ks ** 2)
    diff_w = max(1 - th["pi0"] - th["ph"], 0.0)
    model_var = diff_w * th["sigma"] ** 2 + th["ph"] * e_sq
    emp_var = float(np.mean(k.astype(float) ** 2))
    return {"pi0": th["pi0"], "h_day": th["ph"], "sigma_tk": th["sigma"],
            "gap_mean_tk": e_abs, "w_away": th["w_a"],
            "gap_var_share": th["ph"] * e_sq / model_var if model_var > 0 else np.nan,
            "var_model_emp": model_var / emp_var if emp_var > 0 else np.nan,
            "ll_per_obs": th["ll_per_obs"]}


# ------------------------------------------------------------------ oracle

def run_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    cells = [
        {"pi0": 0.70, "ph": 0.04, "sigma": 0.8, "rho_a": 0.85, "rho_t": 0.60, "w_a": 0.65},
        {"pi0": 0.30, "ph": 0.10, "sigma": 1.5, "rho_a": 0.70, "rho_t": 0.75, "w_a": 0.40},
    ]
    n = 30_000
    kmax = 400
    grid = np.arange(-kmax, kmax + 1)
    for ci, truth in enumerate(cells):
        p = pmf(grid, truth)
        p = p / p.sum()
        k = rng.choice(grid, size=n, p=p)
        th = fit_cell(k)
        rows = [{"param": name, "truth": truth[name], "estimate": th[name]}
                for name in PARAMS]
        print(f"\noracle cell {ci + 1} (n={n}, converged={th['converged']}):")
        print(pd.DataFrame(rows).to_string(index=False,
                                           float_format=lambda v: f"{v:.4f}"))


# ------------------------------------------------------------------ venue run

def r2_ticks(venue: str) -> pd.DataFrame:
    """Folded tick moves for every observation that falls in an R2 cell."""
    mkts = vc.load_markets(venue, None)
    panel = vc.candidate_rates(vc.daily_panel(mkts))
    cells = rm.classify(rm.build_cells(panel))
    r2 = set(map(tuple, cells.loc[cells["regime"] == 2, ["pi", "ti"]].values))
    df = panel.copy()
    df["pi"] = vc._bucket(df["p"].to_numpy())
    df["ti"] = rm._tau_band(df["tau"].to_numpy())
    df = df[[t in r2 for t in zip(df["pi"], df["ti"])]]
    df = df[df["dt"] <= MAX_DT]
    # fold: positive = away from the nearest boundary
    away = np.where(df["p"] <= 0.5, df["dp"], -df["dp"])
    df["k"] = np.round(away / UNIT[venue]).astype(int)
    df["dist"] = np.minimum(df["p"], 1 - df["p"])
    df["di"] = np.clip(np.searchsorted(DIST_EDGES, df["dist"], side="right") - 1,
                       0, len(DIST_EDGES) - 2)
    return df


def run_venue(venue: str) -> None:
    df = r2_ticks(venue)
    print(f"\n=== {venue} ===  {len(df)} R2 observations "
          f"({df['key'].nunique()} markets), tick = {UNIT[venue]}")
    if df.empty:
        print("no R2 cells -- nothing to fit")
        return
    rows = []
    for (di, ti), g in df.groupby(["di", "ti"]):
        if len(g) < MIN_CELL:
            continue
        th = fit_cell(g["k"].to_numpy())
        d_lo, d_hi = DIST_EDGES[di], DIST_EDGES[di + 1]
        rows.append({"dist_bucket": f"[{d_lo:.2f},{min(d_hi, 0.5):.2f})",
                     "tau_band": rm._tau_label(int(ti)), "n": len(g),
                     **summarise(th, g["k"].to_numpy())})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("(pi0 frozen days; h_day gap hazard/day; sigma_tk diffusion sd and")
    print(" gap_mean_tk mean gap, both in ticks; w_away = gap mass moving away")
    print(" from the boundary; gap_var_share = variance carried by gaps;")
    print(" var_model_emp = model/empirical variance, the budget check)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--venue", choices=["polymarket", "kalshi", "both"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    if args.oracle:
        return run_oracle(args.seed)
    if not args.venue:
        ap.error("need --oracle or --venue")
    for v in (["polymarket", "kalshi"] if args.venue == "both" else [args.venue]):
        run_venue(v)


if __name__ == "__main__":
    main()
