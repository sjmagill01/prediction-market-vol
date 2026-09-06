"""Composite regime model vs baselines: out-of-sample horse race. P4.

The payoff question of the modeling phase: does the regime decomposition
(P0-P3) actually predict better, out of sample, than treating the walk as
one object? Protocol:

  Split      walk-forward by resolution date: two folds, train on markets
             resolving in the first 50% (then 75%) of the resolution-date
             order, test on the next 25% block. No test market's life
             overlaps the training criterion (resolution date), and all
             fitted objects (regime map, thetas, cell mixtures) come from
             the training markets only.
  Target     one-day-ahead distribution of the next tick move k = dp/unit,
             scored as log P(k) on the venue tick grid. Continuous models
             are integrated over the tick bin, so every model is scored in
             the same (discrete) measure.
  Models
    composite   R1 cells: the core SLV filter (P1 theta fit on the train R1
                rectangle; the filter tracks the whole market, scores only
                where the train regime map says R1). R2/R3 cells: the P2
                tick mixture (frozen + diffusion + gap) fitted per train
                cell on folded ticks. Cells with thin training data fall
                back to the bucket Gaussian.
    global-slv  the same SLV machinery fit on ALL training observations,
                no regime distinction: is the win from regimes or from SLV?
    const-lam   SLV with lambda == 1 (no stochastic clock).
    bucket      discretised Gaussian with per-(p,tau)-cell training
                variance: the nonparametric "just memorise the rates" bar.
    garch       pooled Gaussian GARCH(1,1) on dp, no state dependence: the
                off-the-shelf time-series bar.
  Metrics     mean log score per observation (pooled and per regime),
              Delta vs composite with a block bootstrap over test markets,
              and a randomised PIT for the composite (discrete scores are
              PIT-able via u = F(k-1) + V*P(k)).

Sanity gate (--simulate): on stochvol+noise simulator markets the SLV models
should beat bucket/garch and the composite should not lose to global-slv
(the simulator has a single regime, so they should roughly tie).

Heavy. Results land in results/horse_race_<label>.csv for the P5 writeup.

Usage:
  python analysis/horse_race.py --simulate 600 [--seed 7]
  python analysis/horse_race.py --venue polymarket|kalshi|both
                                [--max-obs 40000] [--max-test 2500]
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

from analysis import core_slv as cs    # noqa: E402
from analysis import itm_hazard as ih  # noqa: E402
from analysis import regime_map as rm  # noqa: E402
from analysis import vol_check as vc   # noqa: E402

UNIT = {"polymarket": 0.001, "kalshi": 0.01, "sim": 0.01}
FOLDS = [(0.50, 0.75), (0.75, 1.00)]
MIN_CELL_FIT = 500
N_BOOT = 1000
LL_FLOOR = np.log(1e-12)
MODELS = ("composite", "global-slv", "const-lam", "bucket", "garch")


# ------------------------------------------------------------------ data prep

def resolution_order(venue: str, mkts) -> list:
    """Markets sorted by absolute resolution date (last 1d bar timestamp)."""
    from pmdata import store
    q = """SELECT key, max(ts) AS res FROM bars
           WHERE venue = ? AND freq = '1440m' GROUP BY key"""
    d = store.con().execute(q, [venue]).fetchdf().set_index("key")["res"]
    return sorted(mkts, key=lambda m: d.get(m.key, pd.Timestamp.min))


def market_arrays(m) -> dict:
    p = np.clip(m.closes, *vc.P_CLIP)
    tau = np.maximum(m.t_res - m.ts_days, 1.0)
    return {"p": m.closes, "z": norm.ppf(p), "tau": tau,
            "dt": np.concatenate([[np.nan], np.diff(m.ts_days)])}


# ------------------------------------------------------------------ training

def fit_garch(train, max_obs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    dps, tot = [], 0
    for i in idx:
        dp = np.diff(train[i].closes)
        dps.append(dp)
        tot += len(dp)
        if tot >= max_obs:
            break

    def nll(u):
        w, a, b = np.exp(u[0]), 1 / (1 + np.exp(-u[1])), 1 / (1 + np.exp(-u[2]))
        a, b = a * 0.999, b * (1 - a) * 0.999   # stationarity a + b < 1
        tot_ll, n = 0.0, 0
        for dp in dps:
            s2 = w / max(1 - a - b, 1e-6)
            for x in dp:
                tot_ll += -0.5 * (np.log(2 * np.pi * s2) + x * x / s2)
                s2 = w + a * x * x + b * s2
            n += len(dp)
        return -tot_ll / n

    res = minimize(nll, np.array([np.log(1e-4), 0.0, 1.0]),
                   method="Nelder-Mead", options={"maxiter": 250})
    w, a, b = np.exp(res.x[0]), 1 / (1 + np.exp(-res.x[1])), 1 / (1 + np.exp(-res.x[2]))
    a, b = a * 0.999, b * (1 - a) * 0.999
    return {"w": float(w), "a": float(a), "b": float(b)}


def fit_all(train, venue: str, unit: float, max_obs: int, seed: int,
            lean: bool = False) -> dict:
    """All training-side objects: regime map, SLV thetas, mixtures, buckets.

    lean=True skips the global-SLV and GARCH fits (used by consumers like
    budget_forecast.py that only need th_r1/th_clam + mixtures + buckets)."""
    panel = vc.candidate_rates(vc.daily_panel(train))
    cells = rm.classify(rm.build_cells(panel))
    rect = rm.r1_rectangle(cells)
    regime = {(int(r["pi"]), int(r["ti"])): int(r["regime"])
              for _, r in cells.iterrows()}
    print(f"  train: {len(train)} markets, {len(panel)} obs, R1 rect "
          f"p [{rect['p_lo']:.2f},{rect['p_hi']:.2f}) tau >= {rect['tau_lo']:.0f}d")

    def slv(rect_, variant):
        segs = cs.build_segments(train, rect_)
        rng = np.random.default_rng(seed)
        rng.shuffle(segs)
        keep, tot = [], 0
        for s in segs:
            keep.append(s)
            tot += len(s[0]) - 1
            if tot >= max_obs:
                break
        b = cs.make_batches(keep)
        return cs.fit(b, variant, cs.start_values(b), maxiter=300)["theta"]

    all_rect = {"p_lo": 0.0, "p_hi": 1.0, "tau_lo": 0.0}
    th_r1 = slv(rect, "full")
    th_glob = None if lean else slv(all_rect, "full")
    th_clam = slv(all_rect, "no-sv")
    print(f"  theta R1={_fmt(th_r1)}"
          + ("" if lean else f"\n  theta glob={_fmt(th_glob)}")
          + f"\n  theta clam={_fmt(th_clam)}")

    # per-cell tick mixtures (folded) + Gaussian bucket variances
    df = panel.copy()
    df["pi"] = vc._bucket(df["p"].to_numpy())
    df["ti"] = rm._tau_band(df["tau"].to_numpy())
    away = np.where(df["p"] <= 0.5, df["dp"], -df["dp"])
    df["k"] = np.round(away / unit).astype(int)
    mix, bucket = {}, {}
    for (pi, ti), g in df.groupby(["pi", "ti"]):
        bucket[(int(pi), int(ti))] = float(g["dp"].var()) or unit ** 2
        if len(g) >= MIN_CELL_FIT and regime.get((int(pi), int(ti)), 1) != 1:
            mix[(int(pi), int(ti))] = ih.fit_cell(g["k"].to_numpy())
    bucket["global"] = float(df["dp"].var())
    garch = None if lean else fit_garch(train, max_obs, seed)
    print(f"  {len(mix)} cell mixtures"
          + ("" if lean else f", garch w={garch['w']:.2e} "
             f"a={garch['a']:.3f} b={garch['b']:.3f}"))
    return {"regime": regime, "th_r1": th_r1, "th_glob": th_glob,
            "th_clam": th_clam, "mix": mix, "bucket": bucket, "garch": garch}


def _fmt(th):
    return "(" + ", ".join(f"{k}={th[k]:.3f}" for k in cs.PARAM_NAMES) + ")"


# ------------------------------------------------------------------ predictives

def slv_forecast(theta: dict, z, dt, taup):
    """One-step-ahead predictive mixture (wp, mp, S) at each step k >= 1."""
    c, alpha, eta = theta["c"], theta["alpha"], theta["eta"]
    g, w0, T = cs._grid(theta["phi"], theta["s"])
    lam = np.exp(g)
    K = len(g)
    m = np.full(K, z[0])
    P = np.full(K, eta ** 2)
    w = w0.copy()
    out = []
    for k in range(1, len(z)):
        wp = w @ T
        Mix = (w[:, None] * T) / (wp[None, :] + 1e-300)
        mp = Mix.T @ m
        Pp = np.einsum("ij,ij->j", Mix, P[:, None] + (m[:, None] - mp[None, :]) ** 2)
        Pp = Pp + c * lam * dt[k] / taup[k] ** alpha
        S = Pp + eta ** 2
        out.append((wp.copy(), mp.copy(), S.copy()))
        ll = -0.5 * (np.log(2 * np.pi * S) + (z[k] - mp) ** 2 / S)
        logw = np.log(wp + 1e-300) + ll
        logw -= logw.max()
        wn = np.exp(logw)
        w = wn / wn.sum()
        G = Pp / S
        m = mp + G * (z[k] - mp)
        P = Pp * (1 - G)
    return out


def slv_bin_prob(fore, p_prev: float, k: int, unit: float) -> float:
    """P(next move lands in tick bin k) under the predictive z mixture."""
    wp, mp, S = fore
    lo = np.clip(p_prev + (k - 0.5) * unit, 1e-6, 1 - 1e-6)
    hi = np.clip(p_prev + (k + 0.5) * unit, 1e-6, 1 - 1e-6)
    zlo, zhi = norm.ppf(lo), norm.ppf(hi)
    sd = np.sqrt(S)
    pr = float((wp * (norm.cdf((zhi - mp) / sd) - norm.cdf((zlo - mp) / sd))).sum())
    return max(pr, 1e-12)


def gauss_bin_prob(var: float, k: int, unit: float) -> float:
    sd = max(np.sqrt(var), 1e-9)
    return max(float(norm.cdf((k + 0.5) * unit / sd) - norm.cdf((k - 0.5) * unit / sd)),
               1e-12)


# ------------------------------------------------------------------ scoring

def score_market(m, fitted: dict, unit: float, rng) -> list[dict]:
    arr = market_arrays(m)
    p, z, tau, dt = arr["p"], arr["z"], arr["tau"], arr["dt"]
    f_r1 = slv_forecast(fitted["th_r1"], z, dt, tau)
    f_gl = slv_forecast(fitted["th_glob"], z, dt, tau)
    f_cl = slv_forecast(fitted["th_clam"], z, dt, tau)
    garch = fitted["garch"]
    s2 = garch["w"] / max(1 - garch["a"] - garch["b"], 1e-6)
    rows = []
    for k_step in range(1, len(p)):
        p_prev = p[k_step - 1]
        dpv = p[k_step] - p[k_step - 1]
        kk = int(round(dpv / unit))
        pi = int(vc._bucket(np.array([np.clip(p_prev, 1e-6, 1 - 1e-6)]))[0])
        ti = int(rm._tau_band(np.array([tau[k_step - 1]]))[0])
        reg = fitted["regime"].get((pi, ti), 1)
        bvar = fitted["bucket"].get((pi, ti), fitted["bucket"]["global"])
        kf = kk if p_prev <= 0.5 else -kk   # folded: + = away from boundary

        pr_gl = slv_bin_prob(f_gl[k_step - 1], p_prev, kk, unit)
        pr_cl = slv_bin_prob(f_cl[k_step - 1], p_prev, kk, unit)
        pr_bk = gauss_bin_prob(bvar, kk, unit)
        pr_ga = gauss_bin_prob(s2, kk, unit)
        if reg == 1:
            pr_co = slv_bin_prob(f_r1[k_step - 1], p_prev, kk, unit)
            pit_cdf = None
        else:
            th = fitted["mix"].get((pi, ti))
            pr_co = float(ih.pmf(np.array([kf]), th)[0]) if th else pr_bk
            pit_cdf = th
        # randomised PIT for the composite on the tick grid
        if pit_cdf is not None:
            ks = np.arange(-max(abs(kf), 2) - 60, kf)
            f_lo = float(ih.pmf(ks, pit_cdf).sum())
        else:
            wp, mp, S = f_r1[k_step - 1]
            lo = np.clip(p_prev + (kf if p_prev <= 0.5 else -kf) * unit - 0.5 * unit,
                         1e-6, 1 - 1e-6)  # lower edge of observed bin (unfolded)
            lo = np.clip(p_prev + (kk - 0.5) * unit, 1e-6, 1 - 1e-6)
            f_lo = float((wp * norm.cdf((norm.ppf(lo) - mp) / np.sqrt(S))).sum())
        pit = f_lo + rng.uniform() * max(pr_co, 1e-12)

        rows.append({"key": m.key, "tau": tau[k_step - 1], "p": p_prev,
                     "regime": reg,
                     "composite": max(np.log(pr_co), LL_FLOOR),
                     "global-slv": max(np.log(pr_gl), LL_FLOOR),
                     "const-lam": max(np.log(pr_cl), LL_FLOOR),
                     "bucket": max(np.log(pr_bk), LL_FLOOR),
                     "garch": max(np.log(pr_ga), LL_FLOOR),
                     "pit": min(max(pit, 0.0), 1.0)})
        x = dpv
        s2 = garch["w"] + garch["a"] * x * x + garch["b"] * s2
    return rows


# ------------------------------------------------------------------ evaluation

def bootstrap_delta(df: pd.DataFrame, base: str, other: str, rng) -> tuple:
    per = df.groupby("key").agg(d=(base, "mean"), n=(base, "size"),
                                o=(other, "mean"))
    d = (per["d"] - per["o"]).to_numpy()
    n = per["n"].to_numpy()
    idx = rng.integers(0, len(per), (N_BOOT, len(per)))
    boots = (d[idx] * n[idx]).sum(1) / n[idx].sum(1)
    return tuple(np.quantile(boots, [0.025, 0.975]))


def evaluate(label: str, test_rows: list[dict], seed: int) -> pd.DataFrame:
    df = pd.DataFrame(test_rows)
    rng = np.random.default_rng(seed)
    print(f"\n--- {label}: {len(df)} test obs, {df['key'].nunique()} markets")
    means = {mdl: df[mdl].mean() for mdl in MODELS}
    rows = []
    for mdl in MODELS:
        row = {"model": mdl, "ll_per_obs": means[mdl],
               "delta_vs_composite": means[mdl] - means["composite"]}
        if mdl != "composite":
            lo, hi = bootstrap_delta(df, "composite", mdl, rng)
            row["d_ci_lo"], row["d_ci_hi"] = lo, hi
        rows.append(row)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("(delta_vs_composite < 0 = composite wins; CI = 95% block bootstrap")
    print(" over test markets of composite - model)")
    reg = df.groupby("regime")[list(MODELS)].mean()
    reg.index = [rm.REGIME_NAMES[i] for i in reg.index]
    print("mean log score by train-map regime:")
    print(reg.to_string(float_format=lambda v: f"{v:.3f}"))
    dec = np.histogram(df["pit"], bins=10, range=(0, 1))[0] / len(df)
    print("composite randomised PIT deciles (calibrated = 0.100):")
    print("  " + " ".join(f"{d:.3f}" for d in dec))
    df["label"] = label
    return df


def run(label: str, mkts: list, unit: float, max_obs: int, max_test: int,
        seed: int, ordered: bool = True) -> None:
    rng = np.random.default_rng(seed)
    all_rows = []
    for lo, hi in FOLDS:
        n = len(mkts)
        train = mkts[:int(lo * n)]
        test = mkts[int(lo * n):int(hi * n)]
        if len(test) > max_test:
            test = [test[i] for i in rng.choice(len(test), max_test, False)]
        print(f"\nfold train<{lo:.0%} test [{lo:.0%},{hi:.0%}): "
              f"{len(train)} train / {len(test)} test markets")
        fitted = fit_all(train, label, unit, max_obs, seed)
        for m in test:
            all_rows.extend(score_market(m, fitted, unit, rng))
    df = evaluate(label, all_rows, seed)
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"horse_race_{label}.csv", index=False)
    print(f"rows saved: results/horse_race_{label}.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=["polymarket", "kalshi", "both"])
    ap.add_argument("--simulate", type=int, default=0)
    ap.add_argument("--max-obs", type=int, default=40_000)
    ap.add_argument("--max-test", type=int, default=2_500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    if args.simulate:
        mkts = vc.simulate(args.simulate, noise=0.01, stochvol=True, seed=args.seed)
        # simulator markets are exchangeable; index order stands in for time
        run("sim", mkts, UNIT["sim"], args.max_obs, args.max_test, args.seed)
        return
    if not args.venue:
        ap.error("need --venue or --simulate")
    for v in (["polymarket", "kalshi"] if args.venue == "both" else [args.venue]):
        mkts = resolution_order(v, vc.load_markets(v, None))
        run(v, mkts, UNIT[v], args.max_obs, args.max_test, args.seed)


if __name__ == "__main__":
    main()
