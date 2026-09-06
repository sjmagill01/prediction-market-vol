"""Endgame regime: forced collapse into resolution. P3 of the modeling phase.

As tau -> 0 a market must spend its remaining budget p(1-p): the regime map
shows late/extreme cells turning lumpy (R3), with jump shares peaking near
expiry. This phase decomposes the late-life variance rate into its two
channels on the tick grid:

  arrival   nu(p, tau)  = P(move day),   move = |dp| >= 2 ticks (P2's gap
                                         convention: quiet = 0 or 1 tick)
  size      S(p, tau)   = E[dp^2 | move]

each fitted as a power law  a * [p(1-p)]^g / tau^b  on (pq, tau) cell means
(weighted log-log WLS, the power_fit recipe) within the endgame window
tau <= 30 days. Tick censoring is corrected before fitting: the visible-move
rate under-counts arrivals whose size rounded below 2 ticks, which leaks size
exponents into the arrival fit (the uncorrected oracle shows exactly this).
Per cell, the latent size s^2 is estimated by MLE of a discrete normal
truncated at |k| >= 2, and the arrival rate is de-censored by dividing by
the implied visibility P(|move| >= 2 ticks | s). Since  E[dp^2] = nu * S,  the venue's local variance
exponents decompose as gamma ~ g_arrival + g_size and alpha ~ b_arrival +
b_size: the question "does the endgame acceleration come from more frequent
moves or from bigger moves?" is answered by which channel owns the tau
exponent. A direct power fit of E[dp^2] on the same cells is printed as the
consistency check (sums of exponents should approximately match).

Also reported: the budget spend profile -- what share of each market's
lifetime variance (traversal + terminal resolution jump) is spent with
tau <= 1, 3, 7, 30 days -- the model-free statement of how much of the
action is endgame.

This is moment calibration, not likelihood: arrival/size power laws are
identified from cell means alone, and the P4 composite uses them only
through nu and S. Oracle gate (--oracle): simulate a compound-arrival
martingale with known exponents and check recovery.

Usage:
  python analysis/endgame.py --oracle [--seed 7]
  python analysis/endgame.py --venue polymarket|kalshi|both
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import vol_check as vc  # noqa: E402

UNIT = {"polymarket": 0.001, "kalshi": 0.01, "sim": 0.01}
TAU_MAX = 30.0
TAU_BINS = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 30.0])
SPEND_BANDS = [1.0, 3.0, 7.0, 30.0]
MAX_DT = 1.5
MIN_BIN = 100
MIN_MOVERS = 50


# ------------------------------------------------------------------ estimator

def _wls_loglog(cells: pd.DataFrame, ycol: str) -> dict:
    """log y = const + g*log pq - b*log tau, weights sqrt(n)."""
    c = cells[np.isfinite(cells[ycol]) & (cells[ycol] > 0)]
    if len(c) < 6:
        return {"a": np.nan, "g": np.nan, "b": np.nan, "n_cells": len(c)}
    X = np.column_stack([np.ones(len(c)), np.log(c["mpq"]), np.log(c["mtau"])])
    w = np.sqrt(c["n"].to_numpy())
    beta, *_ = np.linalg.lstsq(X * w[:, None], np.log(c[ycol]) * w, rcond=None)
    return {"a": float(np.exp(beta[0])), "g": float(beta[1]),
            "b": float(-beta[2]), "n_cells": len(c)}


def _latent_size(k: np.ndarray) -> float:
    """MLE sd (ticks) of a discrete normal observed truncated at |k| >= 2."""
    vals, counts = np.unique(np.abs(k), return_counts=True)

    def nll(s):
        vis = 2 * norm.sf(1.5 / s)
        p = (norm.cdf((vals + 0.5) / s) - norm.cdf((vals - 0.5) / s)) * 2 / vis
        return -float(counts @ np.log(p + 1e-300))

    return float(minimize_scalar(nll, bounds=(0.3, 200), method="bounded").x)


def decompose(panel: pd.DataFrame, unit: float) -> dict:
    """Arrival/size/total power laws on the endgame window, de-censored."""
    df = panel[(panel["tau"] <= TAU_MAX) & (panel["dt"] <= MAX_DT)].copy()
    df["k"] = np.round(df["dp"] / unit).astype(int)
    df["move"] = df["k"].abs() >= 2
    df["pq"] = df["p"] * (1 - df["p"])
    pq_bins = np.unique(np.quantile(df["pq"], np.linspace(0, 1, 11)))
    df["ci"] = np.searchsorted(pq_bins, df["pq"], side="right") - 1
    df["tj"] = np.searchsorted(TAU_BINS, df["tau"], side="right") - 1
    rows = []
    for (ci, tj), g in df.groupby(["ci", "tj"]):
        if len(g) < MIN_BIN:
            continue
        row = {"ci": ci, "tj": tj, "n": len(g), "mpq": g["pq"].mean(),
               "mtau": g["tau"].mean(), "tot": g["dp2"].mean(),
               "nu": np.nan, "size": np.nan}
        mov = g.loc[g["move"], "k"].to_numpy()
        if len(mov) >= MIN_MOVERS:
            s_tk = _latent_size(mov)
            vis = 2 * norm.sf(1.5 / s_tk)
            row["size"] = (s_tk * unit) ** 2          # latent E[dp^2 | fired]
            row["nu"] = min(g["move"].mean() / vis, 1.0)
        rows.append(row)
    g = pd.DataFrame(rows)
    return {"arrival": _wls_loglog(g, "nu"), "size": _wls_loglog(g, "size"),
            "total": _wls_loglog(g, "tot"), "n_obs": len(df),
            "move_frac": float(df["move"].mean())}


def spend_profile(mkts) -> pd.DataFrame:
    """Share of lifetime variance (traversal + terminal jump) by tau band."""
    bands = [0.0] + SPEND_BANDS + [np.inf]
    acc = np.zeros(len(bands))  # last slot = terminal resolution jump
    for m in mkts:
        dp2 = np.diff(m.closes) ** 2
        tau = np.maximum(m.t_res - m.ts_days[:-1], 0.0)
        idx = np.clip(np.searchsorted(bands, tau, side="left") - 1,
                      0, len(bands) - 2)
        np.add.at(acc, idx, dp2)
        acc[-1] += (m.final - m.closes[-1]) ** 2
    tot = acc.sum()
    labels = [f"tau<= {bands[i + 1]:.0f}d" if i == 0 else
              f"tau ({bands[i]:.0f},{bands[i + 1]:.0f}]d" if np.isfinite(bands[i + 1])
              else f"tau > {bands[i]:.0f}d"
              for i in range(len(bands) - 1)] + ["terminal jump"]
    return pd.DataFrame({"band": labels, "share": acc / tot})


def report(label: str, dec: dict) -> None:
    print(f"\n=== {label} ===  {dec['n_obs']} endgame obs (tau <= {TAU_MAX:.0f}d), "
          f"move frac {dec['move_frac']:.3f}")
    rows = []
    for name in ("arrival", "size", "total"):
        r = dec[name]
        rows.append({"channel": name, "a": r["a"], "g_pq": r["g"],
                     "b_tau": r["b"], "n_cells": r["n_cells"]})
    s = pd.DataFrame(rows)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ga, gs = dec["arrival"]["g"], dec["size"]["g"]
    ba, bs = dec["arrival"]["b"], dec["size"]["b"]
    print(f"decomposition check: g_arr + g_size = {ga + gs:.2f} vs total "
          f"{dec['total']['g']:.2f};  b_arr + b_size = {ba + bs:.2f} vs total "
          f"{dec['total']['b']:.2f}")


# ------------------------------------------------------------------ oracle

def simulate_compound(n: int, a: float, b: float, g1: float,
                      m: float, d: float, g2: float, seed: int) -> list:
    """Martingale-signed compound-arrival markets on a cent grid."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        L = int(rng.integers(10, 60))
        p = float(rng.uniform(0.1, 0.9))
        path = [p]
        for t in range(L):
            tau = L - t
            pq = p * (1 - p)
            nu = min(a * pq ** g1 / tau ** b, 1.0)
            if rng.uniform() < nu:
                s2 = m * pq ** g2 / tau ** d
                dp = rng.choice([-1, 1]) * abs(rng.normal(0, np.sqrt(s2)))
                p = float(np.clip(p + np.round(dp, 2), 0.01, 0.99))
            path.append(p)
        f = 1.0 if rng.uniform() < p else 0.0
        out.append(vc.Market(f"sim{i}", np.arange(L + 1, dtype=float),
                             np.array(path), f, float(L)))
    return out


def run_oracle(seed: int) -> None:
    truth = {"a": 0.9, "b": 0.5, "g1": 0.3, "m": 0.03, "d": 0.5, "g2": 1.0}
    mkts = simulate_compound(4000, truth["a"], truth["b"], truth["g1"],
                             truth["m"], truth["d"], truth["g2"], seed)
    dec = decompose(vc.daily_panel(mkts), UNIT["sim"])
    report("oracle", dec)
    print("\ntruth: arrival g_pq={g1:.2f} b_tau={b:.2f}; "
          "size g_pq={g2:.2f} b_tau={d:.2f}".format(**truth))
    print("(size 'a'/exponents blend in grid rounding + boundary clipping; "
          "judge on exponents)")


# ------------------------------------------------------------------ venue run

def run_venue(venue: str) -> None:
    mkts = vc.load_markets(venue, None)
    panel = vc.daily_panel(mkts)
    report(venue, decompose(panel, UNIT[venue]))
    print("\nbudget spend profile (share of lifetime variance):")
    print(spend_profile(mkts).to_string(index=False,
                                        float_format=lambda v: f"{v:.3f}"))


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
