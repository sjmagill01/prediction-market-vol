"""Budget consistency: does the composite spend p(1-p)? P5 of the modeling phase.

Every market quotes its own implied remaining variance: for a martingale
absorbed at 0/1, E[sum dp^2 + terminal jump^2 | p, tau] = p(1-p), the
variance-swap strike quoted by the price itself. The P4 composite was fitted
and scored one day ahead; this check integrates it: roll the *generative*
composite forward from mid-life states (Monte Carlo on the venue tick grid,
R1 cells stepping the SLV probit walk with log-lambda AR(1) truncated to the
same grid range as the filter, R2/R3 cells sampling the fitted tick mixtures,
thin cells the bucket Gaussian, terminal resolution contributing its exact
expectation p_T(1-p_T)) and compare the simulated remaining variance to the
budget, next to the realized continuation of the same markets.

The decomposition: each simulated/realized daily move is classified jump
(|dp| >= 0.02, two cents) or diffusive, and the terminal resolution jump
counts as jump. So the table answers, per starting tau band: does the model
spend the right total, and does it spend it through the right channel?

Aggregation is ratio-of-sums (sum of remaining variance over starts divided
by sum of budgets), the same stabilisation as the README budget figure;
single-market ratios are Bernoulli-noisy by construction.

Stated caveats: fits are in-sample (this is a consistency audit of the fitted
object, not a forecast contest; that was P4); the probit walk is a martingale
in z, only approximately in p; starts are restricted to p in [0.05, 0.95]
so budgets are bounded away from 0.

Sanity gate (--simulate): on simulator markets the realized ratio must be
~1 (the budget identity is a theorem); the model ratio should be close.

Usage:
  python analysis/budget_forecast.py --simulate 500 [--seed 7]
  python analysis/budget_forecast.py --venue polymarket|kalshi|both
                                     [--n-paths 200] [--max-markets 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import core_slv as cs     # noqa: E402
from analysis import horse_race as hr   # noqa: E402
from analysis import itm_hazard as ih   # noqa: E402
from analysis import regime_map as rm   # noqa: E402
from analysis import vol_check as vc    # noqa: E402

JUMP_THRESH = 0.02          # |dp| >= 2 cents counts as a jump, both venues
START_BANDS = (30.0, 14.0, 7.0, 3.0)
K_MAX = 400                 # tick-mixture sampling support


# ------------------------------------------------------------------ model MC

def _tables(fitted: dict):
    """Dense lookup tables from the fitted dict: regime array, per-cell
    mixture CDFs (searchsorted sampling), bucket variances."""
    n_pi = len(vc.P_BUCKETS) - 1
    n_ti = len(rm.TAU_EDGES) - 1
    reg = np.ones((n_pi, n_ti), dtype=int)
    for (pi, ti), r in fitted["regime"].items():
        reg[pi, ti] = r
    kg = np.arange(-K_MAX, K_MAX + 1)
    cdf = {}
    for cell, th in fitted["mix"].items():
        pr = np.maximum(ih.pmf(kg, th), 0.0)
        cdf[cell] = np.cumsum(pr / pr.sum())
    bvar = np.full((n_pi, n_ti), fitted["bucket"]["global"])
    for cell, v in fitted["bucket"].items():
        if cell != "global":
            bvar[cell] = v
    return reg, kg, cdf, bvar, n_ti


def mc_all(p0: np.ndarray, tau0: np.ndarray, fitted: dict, th: dict,
           unit: float, n_paths: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Per-start mean (diffusive, jump) remaining variance under the
    composite, with `th` as the R1 engine (th_r1 = full SLV, th_clam =
    constant-lambda control: if the full model overshoots and the control
    does not, the excess is the intensity tail, not the local factor).

    Fully batched: one flat (start x path) state vector stepped down the
    resolution clock; per-step work is vectorised numpy on the active set,
    tick mixtures sampled by searchsorted on cached CDFs."""
    c, alpha, phi, s = th["c"], th["alpha"], th["phi"], th["s"]
    g, _, _ = cs._grid(phi, s)
    lo_l, hi_l = float(g.min()), float(g.max())
    reg_arr, kg, cdf, bvar, n_ti = _tables(fitted)

    n_s = len(p0)
    N = n_s * n_paths
    p = np.repeat(p0, n_paths)
    tleft = np.repeat(np.ceil(tau0), n_paths)
    sd_st = s / np.sqrt(max(1 - phi ** 2, 1e-9))
    logl = np.clip(rng.normal(0.0, sd_st, N), lo_l, hi_l)
    var_d = np.zeros(N)
    var_j = np.zeros(N)

    for _ in range(int(tleft.max())):
        idx = np.flatnonzero(tleft > 0)
        if not len(idx):
            break
        pa, ta = p[idx], tleft[idx]
        ti = rm._tau_band(ta).astype(int)
        pi = vc._bucket(np.clip(pa, 1e-6, 1 - 1e-6)).astype(int)
        reg = reg_arr[pi, ti]
        dp = np.zeros(len(idx))

        r1 = reg == 1
        if r1.any():
            i1 = idx[r1]
            lam = np.exp(logl[i1])
            v = c * lam / np.maximum(ta[r1], 1.0) ** alpha
            z = norm.ppf(np.clip(pa[r1], *vc.P_CLIP))
            pn = norm.cdf(z + np.sqrt(v) * rng.standard_normal(len(i1)))
            dp[r1] = pn - pa[r1]
        nr = np.flatnonzero(~r1)
        if len(nr):
            code = pi[nr] * n_ti + ti[nr]
            for cval in np.unique(code):
                sel = nr[code == cval]           # positions within idx
                cell = (int(cval) // n_ti, int(cval) % n_ti)
                if cell in cdf:
                    u = rng.random(len(sel))
                    k = kg[np.searchsorted(cdf[cell], u)]
                    dp[sel] = np.where(pa[sel] <= 0.5, k, -k) * unit
                else:
                    dp[sel] = np.sqrt(bvar[cell]) * rng.standard_normal(len(sel))

        pn = np.clip(pa + dp, *vc.P_CLIP)
        d = pn - pa
        jump = np.abs(d) >= JUMP_THRESH
        var_j[idx] += np.where(jump, d * d, 0.0)
        var_d[idx] += np.where(jump, 0.0, d * d)
        p[idx] = pn
        logl[idx] = np.clip(phi * logl[idx] + s * rng.standard_normal(len(idx)),
                            lo_l, hi_l)
        tleft[idx] -= 1.0

    var_j += p * (1 - p)    # exact expectation of the terminal resolution jump
    return (var_d.reshape(n_s, n_paths).mean(1),
            var_j.reshape(n_s, n_paths).mean(1))


# ------------------------------------------------------------------ empirical

def pick_starts(m):
    """(band, index, p0) at the observation nearest each start band."""
    tau = m.t_res - m.ts_days
    for b in START_BANDS:
        i = int(np.argmin(np.abs(tau - b)))
        if abs(tau[i] - b) <= 0.4 * b and i <= len(m.closes) - 2:
            p0 = float(m.closes[i])
            if 0.05 <= p0 <= 0.95:
                yield b, i, p0, float(tau[i])


def realized_remaining(m, i: int) -> tuple[float, float]:
    dp = np.diff(m.closes[i:])
    jump = np.abs(dp) >= JUMP_THRESH
    vd = float((dp[~jump] ** 2).sum())
    vj = float((dp[jump] ** 2).sum()) + (m.final - m.closes[-1]) ** 2
    return vd, vj


# ------------------------------------------------------------------ run

def run_label(label: str, mkts: list, unit: float, args) -> None:
    fitted = hr.fit_all(mkts, label, unit, args.max_obs, args.seed, lean=True)
    rng = np.random.default_rng(args.seed)
    if len(mkts) > args.max_markets:
        mkts = [mkts[i] for i in
                rng.choice(len(mkts), args.max_markets, replace=False)]
    rows = []
    for m in mkts:
        for band, i, p0, tau0 in pick_starts(m):
            evd, evj = realized_remaining(m, i)
            rows.append({"key": m.key, "band": band, "budget": p0 * (1 - p0),
                         "p0": p0, "tau0": tau0, "evd": evd, "evj": evj})
    df = pd.DataFrame(rows)
    p0v, t0v = df["p0"].to_numpy(), df["tau0"].to_numpy()
    df["mvd"], df["mvj"] = mc_all(p0v, t0v, fitted, fitted["th_r1"], unit,
                                  args.n_paths, rng)
    df["cvd"], df["cvj"] = mc_all(p0v, t0v, fitted, fitted["th_clam"], unit,
                                  args.n_paths, rng)
    print(f"\n=== {label} ===  {len(df)} starts from {df['key'].nunique()} "
          f"markets, {args.n_paths} paths each")
    agg = df.groupby("band").agg(n=("budget", "size"), bud=("budget", "sum"),
                                 mvd=("mvd", "sum"), mvj=("mvj", "sum"),
                                 cvd=("cvd", "sum"), cvj=("cvj", "sum"),
                                 evd=("evd", "sum"), evj=("evj", "sum"))
    out = pd.DataFrame({
        "n": agg["n"],
        "model_ratio": (agg["mvd"] + agg["mvj"]) / agg["bud"],
        "clam_ratio": (agg["cvd"] + agg["cvj"]) / agg["bud"],
        "real_ratio": (agg["evd"] + agg["evj"]) / agg["bud"],
        "model_jump_share": agg["mvj"] / (agg["mvd"] + agg["mvj"]),
        "real_jump_share": agg["evj"] / (agg["evd"] + agg["evj"]),
    }).sort_index(ascending=False)
    out.index = [f"tau~{b:.0f}d" for b in out.index]
    print(out.to_string(float_format=lambda v: f"{v:.3f}"))
    print("(ratio = remaining variance / p0(1-p0), ratio-of-sums over starts;")
    print(" clam_ratio = same MC with the constant-lambda R1 engine;")
    print(f" jump = |dp| >= {JUMP_THRESH} or terminal resolution; realized")
    print(" ratio ~ 1 is the budget theorem, model ratio tests the fits)")
    res = Path(__file__).resolve().parent.parent / "results"
    res.mkdir(exist_ok=True)
    df.to_csv(res / f"budget_forecast_{label}.csv", index=False)
    print(f"rows saved: results/budget_forecast_{label}.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=["polymarket", "kalshi", "both"])
    ap.add_argument("--simulate", type=int, default=0)
    ap.add_argument("--n-paths", type=int, default=200)
    ap.add_argument("--max-markets", type=int, default=2_000)
    ap.add_argument("--max-obs", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    if args.simulate:
        mkts = vc.simulate(args.simulate, noise=0.01, stochvol=True,
                           seed=args.seed)
        run_label("sim", mkts, hr.UNIT["sim"], args)
        return
    if not args.venue:
        ap.error("need --venue or --simulate")
    for v in (["polymarket", "kalshi"] if args.venue == "both" else [args.venue]):
        mkts = vc.load_markets(v, None)
        run_label(v, mkts, hr.UNIT[v], args)


if __name__ == "__main__":
    main()
