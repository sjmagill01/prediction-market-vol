r"""V3 regime structure: the (p, tau) map and one model per regime.

The V2 empirics treat each venue's panel as one object, but the dynamics
are not one object: mid-p far-from-expiry markets diffuse, decided-but-
unresolved markets freeze and gap, and near-expiry markets are forced to
spend their remaining budget in lumps. Four blocks, all on the frozen v2
vintage via analysis/core.py:

(A) Regime map. Per (price bucket, tau band) cell:
      still_frac   fraction of daily moves with |dp| < half a cent
      excess_still still_frac minus what tick-censored gauss diffusion
                   itself predicts, 2 Phi(halftick/sigma) - 1 averaged
                   over the cell (raw stillness mislabels clean longshots)
      kurt_z       excess kurtosis of z = dp/sqrt(gauss rate)
      jump_share   1 - (pi/2) E|dp_t||dp_{t+1}| / E dp^2 (bipower)
      ratio_fit    cell mean dp^2/dt over the venue bulk power-family fit
      budget_share cell share of total traversal variance
    Classification: R2 "itm/hazard" if excess_still >= 0.3 (wins), else
    R3 "jumpy" if kurt_z >= 30, else R1 "diffusive". The R1 rectangle
    (largest majority-R1 p-range x tau-range) scopes the SLV fit.

(B) R1 SLV. THEORY.md forces any admissible stochvol model on [0,1] into
    Var(dp) = phi(Phi^-1(p))^2 lambda_t dt; in probit coordinates that is
    a random walk with a stochastic clock:
      z_k = x_k + e_k, e_k ~ N(0, eta^2)
      x_{k+1} - x_k ~ N(0, c lambda_k dt_k / tau_k^alpha)
      log lambda ~ AR(1)(phi, stationary sd s) on a K-point grid
    Pooled theta = (c, alpha, eta, phi, s) per venue by a collapsed
    mixture (IMM) Kalman filter + Nelder-Mead. Bounds s <= S_MAX,
    eta >= ETA_MIN block the Gaussian-mixture degeneracy on exact price
    repeats; a pinned bound means the model is straining to imitate the
    zero-move point mass. Variants no-sv (lambda == 1) and no-noise
    (eta ~= 0) answer which layers earn their keep; PIT deciles of the
    one-step predictive report calibration. Approximations, stated: the
    lambda transition uses a 1-day step for every gap (gaps inside R1
    are overwhelmingly 1 day; the x-step uses the actual dt and tau);
    Gaussian noise stands in for the tick grid (tick censoring is R2
    business, block C).

(C) R2 itm-hazard. Daily tick moves k = dp/unit, folded so positive =
    away from the nearest boundary, modelled directly:
      P(k) = pi0 1{k=0} + (1 - pi0 - h) discN(k; sigma)
             + h gap(k; rho_a, rho_t, w_a)
    with gap a two-sided geometric supported on |k| >= 2 (1-tick moves
    belong to the diffusion; without the support cut the oracle
    confounds small gaps with diffusion). Per (boundary-distance, tau
    band) cell by discrete MLE; model variance vs empirical variance is
    the per-cell budget check.

(D) R3 endgame. Late-life (tau <= 30d) variance rate decomposed into
      arrival nu(p, tau) = P(move day), move = |dp| >= 2 ticks
      size    S(p, tau)  = E[dp^2 | move]
    each a power law a [p(1-p)]^g / tau^b on (pq, tau) cell means (the
    power_fit recipe). Tick censoring is corrected first: latent size by
    MLE of a discrete normal truncated at |k| >= 2, arrival de-censored
    by the implied visibility (the uncorrected oracle leaks size
    exponents into the arrival fit). Since E[dp^2] = nu S the exponents
    should add up to the direct total fit, printed as the check. Plus
    the model-free spend profile: share of lifetime variance (traversal
    + terminal jump) with tau <= 1/3/7/30 days.

Blocks A and D read the tick grid so they use the unclipped core panel;
the map's standardised metrics and the SLV use the probit-safe clip.
Oracle gates live in tests/test_regimes_v3.py; main() reruns them before
the venues. Outputs in results_v2/:
  regimes_cells_{label}.csv       per-cell metrics + regime
  regimes_scalars_{label}.json    rectangle, budget by regime
  slv_{label}.json                variant table + PIT deciles
  hazard_{label}.csv              per-cell hazard fits
  endgame_{label}.json, endgame_spend_{label}.csv

Usage:
  python analysis/regimes_v3.py                    # oracles + both venues
  python analysis/regimes_v3.py --venue kalshi
  python analysis/regimes_v3.py --sim-only         # oracle blocks only
  python analysis/regimes_v3.py --skip-slv         # cheap blocks only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.config import RESULTS, SEED, UNIT                   # noqa: E402
from analysis.core import (Market, P_BUCKETS, P_CLIP, TAU_EDGES,  # noqa: E402
                           bucket, bucket_label, load_markets, panel,
                           simulate, tau_band, tau_label)
from analysis.empirics_v2 import move_panel, power_fit            # noqa: E402

REGIME_NAMES = {1: "R1 diffusive", 2: "R2 itm/hazard", 3: "R3 jumpy"}


# ================================================================ (A) map
STILL_TICK = 0.005      # half a cent: "nothing happened today"
STILL_THRESH = 0.3      # R2 if stillness beyond the diffusive prediction
KURT_THRESH = 30.0      # R3 if active but lumpy
MIN_CELL = 200


def map_cells(pan: pd.DataFrame) -> pd.DataFrame:
    """Per-(p bucket, tau band) regime metrics from the move panel."""
    d = pan.copy()
    d["pi"] = bucket(d["p"].to_numpy())
    d["ti"] = tau_band(d["tau"].to_numpy())
    sig = np.sqrt(d["v_gauss"].clip(lower=1e-12))
    d["z"] = d["dp"] / sig
    d["still"] = d["dp"].abs() < STILL_TICK
    d["pred_still"] = 2 * norm.cdf(STILL_TICK / sig) - 1
    # bipower needs consecutive ~daily pairs within one market
    nxt = (d["key"] == d["key"].shift(-1)) & d["dt"].between(0.9, 1.1) \
        & d["dt"].shift(-1).between(0.9, 1.1)
    d["bp"] = np.where(nxt, d["dp"].abs() * d["dp"].shift(-1).abs(), np.nan)
    d["bp_dp2"] = np.where(nxt, d["dp2"], np.nan)

    pf = power_fit(pan, bulk=True)
    rows = []
    for (pi, ti), g in d.groupby(["pi", "ti"]):
        if len(g) < MIN_CELL:
            continue
        z = g["z"].to_numpy()
        z = z[np.isfinite(z)]
        m2 = np.mean((z - z.mean()) ** 2)
        m4 = np.mean((z - z.mean()) ** 4)
        bp, b2 = g["bp"].dropna(), g["bp_dp2"].dropna().sum()
        jump = (1 - (np.pi / 2) * bp.sum() / b2
                if len(bp) >= 50 and b2 > 0 else np.nan)
        pred = (pf["c"] * (g["p"] * (1 - g["p"])).mean() ** pf["gamma"]
                / g["tau"].mean() ** pf["alpha"])
        rows.append({
            "pi": int(pi), "ti": int(ti),
            "p_bucket": bucket_label(int(pi)), "tau_band": tau_label(int(ti)),
            "n": len(g), "n_mkts": g["key"].nunique(),
            "still_frac": float(g["still"].mean()),
            "excess_still": float(g["still"].mean() - g["pred_still"].mean()),
            "kurt_z": float(m4 / m2 ** 2 - 3) if m2 > 0 else np.nan,
            "jump_share": float(max(jump, 0.0)) if np.isfinite(jump) else np.nan,
            "ratio_fit": float((g["dp2"] / g["dt"]).mean() / pred)
            if pred > 0 else np.nan,
            "budget_share": float(g["dp2"].sum())})
    cells = pd.DataFrame(rows)
    cells["budget_share"] /= cells["budget_share"].sum()
    r = np.ones(len(cells), int)
    r[cells["kurt_z"] >= KURT_THRESH] = 3
    r[cells["excess_still"] >= STILL_THRESH] = 2      # R2 wins over R3
    cells["regime"] = r
    cells.attrs["power_fit"] = pf
    return cells


def r1_rect(cells: pd.DataFrame) -> dict:
    """Largest p-range / tau-range whose cells are majority R1.

    p-scan on tau >= 7d cells (stable base), then tau-scan on retained p.
    """
    base = cells[cells["ti"] >= 2]
    p_ok = [pi for pi, g in base.groupby("pi") if (g["regime"] == 1).mean() > 0.5]
    t_ok = [ti for ti, g in cells[cells["pi"].isin(p_ok)].groupby("ti")
            if (g["regime"] == 1).mean() > 0.5]
    return {"p_lo": float(P_BUCKETS[min(p_ok)]) if p_ok else np.nan,
            "p_hi": float(P_BUCKETS[max(p_ok) + 1]) if p_ok else np.nan,
            "tau_lo": float(TAU_EDGES[min(t_ok)]) if t_ok else np.nan,
            "p_buckets": p_ok, "tau_bands": t_ok}


def budget_by_regime(cells: pd.DataFrame) -> dict:
    return {REGIME_NAMES[int(k)]: float(v) for k, v in
            cells.groupby("regime")["budget_share"].sum().items()}


# ================================================================ (B) SLV
K_GRID = 13         # log-lambda grid nodes
S_MAX = 2.0         # vol-of-vol bound (degeneracy guard on price repeats)
ETA_MIN = 1e-3      # noise floor (same guard)
MAX_GAP = 3.0       # break segments at gaps > 3 days
MIN_SEG = 8
BATCH = 1024
MAX_OBS = 40_000    # filtered steps per venue fit (subsample above this)
SLV_PARAMS = ("c", "alpha", "eta", "phi", "s")
SLV_VARIANTS = {"full": {}, "no-sv": {"phi": 0.0, "s": 1e-8},
                "no-noise": {"eta": 1e-6}}


def slv_segments(mkts: list[Market], rect: dict) -> list[tuple]:
    """Contiguous in-rectangle runs of closes as (z, dt, tau_prev)."""
    segs = []
    for m in mkts:
        p = np.clip(m.closes, *P_CLIP)
        tau = np.maximum(m.t_res - m.ts_days, 1.0)
        ok = (m.closes >= rect["p_lo"]) & (m.closes < rect["p_hi"]) \
            & (tau >= rect["tau_lo"])
        gap = np.concatenate([[1.0], np.diff(m.ts_days)])
        run = np.cumsum(~ok | (gap > MAX_GAP))
        for _, idx in pd.Series(np.arange(len(p))).groupby(run):
            i = idx.to_numpy()
            i = i[ok[i]]
            if len(i) < MIN_SEG:
                continue
            segs.append((norm.ppf(p[i]),
                         np.concatenate([[np.nan], np.diff(m.ts_days[i])]),
                         np.concatenate([[np.nan], tau[i][:-1]])))
    return segs


def slv_batches(segs: list[tuple]) -> list[tuple]:
    """Pad length-sorted segments into (Z, DT, TAUP, lengths) arrays."""
    segs = sorted(segs, key=lambda s: -len(s[0]))
    out = []
    for b0 in range(0, len(segs), BATCH):
        chunk = segs[b0:b0 + BATCH]
        M, T = len(chunk), max(len(s[0]) for s in chunk)
        Z, DT, TP = (np.full((M, T), np.nan) for _ in range(3))
        L = np.empty(M, int)
        for i, (z, dt, tp) in enumerate(chunk):
            L[i] = len(z)
            Z[i, :L[i]], DT[i, :L[i]], TP[i, :L[i]] = z, dt, tp
        out.append((Z, DT, TP, L))
    return out


def _lam_grid(phi: float, s: float):
    """Log-lambda nodes, stationary weights, 1-day transition matrix."""
    if s <= 1e-6:
        return np.zeros(1), np.ones(1), np.ones((1, 1))
    g = np.linspace(-3 * s, 3 * s, K_GRID)
    w0 = norm.pdf(g, 0, s)
    w0 /= w0.sum()
    innov = s * np.sqrt(max(1 - phi ** 2, 1e-10))
    T = norm.pdf(g[None, :], phi * g[:, None], innov) + 1e-300
    return g, w0, T / T.sum(1, keepdims=True)


def slv_loglik(theta: dict, batches: list, collect: bool = False):
    """Collapsed-mixture (IMM) filtered log-likelihood of steps k >= 1."""
    c, alpha, eta = theta["c"], theta["alpha"], theta["eta"]
    g, w0, T = _lam_grid(theta["phi"], theta["s"])
    lam = np.exp(g)
    total, n_obs, pits = 0.0, 0, []
    for Z, DT, TP, L in batches:
        M, Tn = Z.shape
        K = len(g)
        m = np.repeat(Z[:, :1], K, axis=1)      # x posterior mean per node
        P = np.full((M, K), eta ** 2)
        w = np.tile(w0, (M, 1))
        for k in range(1, Tn):
            act = k < L
            if not act.any():
                break
            # HMM node mixing, moment-matched collapse
            W = w[:, :, None] * T[None, :, :]
            wp = W.sum(1)
            mix = W / (wp[:, None, :] + 1e-300)
            mp = np.einsum("mij,mi->mj", mix, m)
            dm = m[:, :, None] - mp[:, None, :]
            Pp = np.einsum("mij,mij->mj", mix, P[:, :, None] + dm ** 2)
            # state predict with the actual (dt, tau), then Kalman update
            step = np.where(act, DT[:, k] / TP[:, k] ** alpha, 1.0)
            Pp = Pp + c * lam[None, :] * step[:, None]
            S = Pp + eta ** 2
            resid = np.where(act, Z[:, k], 0.0)[:, None] - mp
            ll = -0.5 * (np.log(2 * np.pi * S) + resid ** 2 / S)
            logw = np.log(wp + 1e-300) + ll
            mx = logw.max(1, keepdims=True)
            step_ll = mx[:, 0] + np.log(np.exp(logw - mx).sum(1))
            total += step_ll[act].sum()
            n_obs += int(act.sum())
            if collect:
                pit = (wp * norm.cdf(resid / np.sqrt(S))).sum(1)
                pits.append(pit[act])
            gain = Pp / S
            a = act[:, None]
            m = np.where(a, mp + gain * resid, m)
            P = np.where(a, Pp * (1 - gain), P)
            w = np.where(a, np.exp(logw - step_ll[:, None]), w)
    if collect:
        return total, n_obs, np.concatenate(pits) if pits else np.array([])
    return total, n_obs


def _slv_pack(theta: dict, free: list[str]) -> np.ndarray:
    u = []
    for name in free:
        v = theta[name]
        if name == "phi":
            u.append(np.arctanh(np.clip(v, -0.999, 0.999)))
        elif name == "alpha":
            u.append(v)
        elif name == "s":
            q = np.clip(v / S_MAX, 1e-6, 1 - 1e-6)
            u.append(np.log(q / (1 - q)))
        elif name == "eta":
            u.append(np.log(max(v - ETA_MIN, 1e-8)))
        else:
            u.append(np.log(v))
    return np.array(u)


def _slv_unpack(u: np.ndarray, free: list[str], fixed: dict) -> dict:
    theta = dict(fixed)
    for name, v in zip(free, u):
        if name == "phi":
            theta[name] = float(np.tanh(v))
        elif name == "alpha":
            theta[name] = float(v)
        elif name == "s":
            theta[name] = float(S_MAX / (1 + np.exp(-v)))
        elif name == "eta":
            theta[name] = float(ETA_MIN + np.exp(v))
        else:
            theta[name] = float(np.exp(v))
    return theta


def slv_fit(batches: list, variant: str, start: dict,
            maxiter: int = 400) -> dict:
    fixed = SLV_VARIANTS[variant]
    free = [n for n in SLV_PARAMS if n not in fixed]

    def nll(u):
        ll, n = slv_loglik(_slv_unpack(u, free, fixed), batches)
        return -ll / n if n else np.inf

    t0 = time.time()
    res = minimize(nll, _slv_pack(start, free), method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5})
    theta = _slv_unpack(res.x, free, fixed)
    ll, n = slv_loglik(theta, batches)
    return {"variant": variant, "theta": theta, "ll": ll, "n": n,
            "ll_per_obs": ll / n, "n_free": len(free),
            "seconds": time.time() - t0, "converged": bool(res.success)}


def slv_start(batches: list) -> dict:
    """Moment start: c from mean dz^2, dt/tau-adjusted at alpha = 0.5."""
    Z, DT, TP, _ = batches[0]
    y = np.diff(Z, axis=1) ** 2 * TP[:, 1:] ** 0.5 / DT[:, 1:]
    return {"c": max(float(np.nanmean(y)), 1e-4), "alpha": 0.5,
            "eta": 0.02, "phi": 0.9, "s": 0.7}


def simulate_slv(n: int, theta: dict, seed: int = SEED) -> list[tuple]:
    """Segments from the exact SLV model (dt = 1, tau counting down)."""
    rng = np.random.default_rng(seed)
    segs = []
    for _ in range(n):
        L = int(rng.integers(40, 301))
        tau = np.arange(L, 0, -1.0)
        lg = rng.normal(0, theta["s"])
        x = rng.normal(0, 0.7)
        z = np.empty(L)
        z[0] = x + rng.normal(0, theta["eta"])
        for t in range(1, L):
            lg = theta["phi"] * lg + rng.normal(
                0, theta["s"] * np.sqrt(max(1 - theta["phi"] ** 2, 1e-10)))
            x += rng.normal(0, np.sqrt(
                theta["c"] * np.exp(lg) / tau[t - 1] ** theta["alpha"]))
            z[t] = x + rng.normal(0, theta["eta"])
        segs.append((z, np.concatenate([[np.nan], np.ones(L - 1)]),
                     np.concatenate([[np.nan], tau[:-1]])))
    return segs


# ============================================================ (C) hazard
DIST_EDGES = np.array([0.0, 0.05, 0.15, 0.30, 0.45, 0.501])
HAZ_MAX_DT = 1.5    # ~daily rows so the hazard reads per day
HAZ_MIN_CELL = 500
HAZ_PARAMS = ("pi0", "h", "sigma", "rho_a", "rho_t", "w_a")


def haz_pmf(k: np.ndarray, th: dict) -> np.ndarray:
    """P(tick move = k): frozen + tick-censored discN + geometric gaps."""
    diff_w = max(1 - th["pi0"] - th["h"], 0.0)
    p = diff_w * (norm.cdf((k + 0.5) / th["sigma"])
                  - norm.cdf((k - 0.5) / th["sigma"]))
    p = p + th["pi0"] * (k == 0)
    ka = np.abs(k)
    gap = np.where(
        k >= 2, th["w_a"] * (1 - th["rho_a"]) * th["rho_a"] ** (ka - 2),
        np.where(k <= -2,
                 (1 - th["w_a"]) * (1 - th["rho_t"]) * th["rho_t"] ** (ka - 2),
                 0.0))
    return p + th["h"] * gap


def _haz_unpack(u: np.ndarray) -> dict:
    e0, e1 = np.exp(u[0]), np.exp(u[1])
    den = 1 + e0 + e1
    sg = lambda v: 1 / (1 + np.exp(-v))                       # noqa: E731
    return {"pi0": e0 / den, "h": e1 / den, "sigma": np.exp(u[2]),
            "rho_a": sg(u[3]), "rho_t": sg(u[4]), "w_a": sg(u[5])}


def haz_fit(k: np.ndarray) -> dict:
    """Discrete MLE of the mixture on tick moves (unique-value weighted)."""
    vals, counts = np.unique(k, return_counts=True)

    def nll(u):
        return -float(counts @ np.log(haz_pmf(vals, _haz_unpack(u)) + 1e-300)) \
            / len(k)

    lg = lambda q: np.log(q / (1 - q))                        # noqa: E731
    start = np.array([np.log(0.5 / 0.45), np.log(0.05 / 0.45), 0.0,
                      lg(0.7), lg(0.7), lg(0.5)])
    res = minimize(nll, start, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-7})
    th = _haz_unpack(res.x)
    th["ll_per_obs"] = -float(res.fun)
    th["converged"] = bool(res.success)
    return th


def haz_summary(th: dict, k: np.ndarray) -> dict:
    """Interpretable readout + model/empirical variance (budget check)."""
    ks = np.arange(2, 2001, dtype=float)
    pa = (1 - th["rho_a"]) * th["rho_a"] ** (ks - 2)
    pt = (1 - th["rho_t"]) * th["rho_t"] ** (ks - 2)
    e_abs = th["w_a"] * (pa @ ks) + (1 - th["w_a"]) * (pt @ ks)
    e_sq = th["w_a"] * (pa @ ks ** 2) + (1 - th["w_a"]) * (pt @ ks ** 2)
    diff_w = max(1 - th["pi0"] - th["h"], 0.0)
    mvar = diff_w * th["sigma"] ** 2 + th["h"] * e_sq
    evar = float(np.mean(k.astype(float) ** 2))
    return {"pi0": th["pi0"], "h_day": th["h"], "sigma_tk": th["sigma"],
            "gap_mean_tk": float(e_abs), "w_away": th["w_a"],
            "gap_var_share": th["h"] * e_sq / mvar if mvar > 0 else np.nan,
            "var_model_emp": mvar / evar if evar > 0 else np.nan,
            "ll_per_obs": th["ll_per_obs"]}


def r2_ticks(raw: pd.DataFrame, cells: pd.DataFrame, unit: float) -> pd.DataFrame:
    """Folded tick moves for rows falling in R2 cells (unclipped panel)."""
    r2 = set(map(tuple, cells.loc[cells["regime"] == 2, ["pi", "ti"]].values))
    d = raw[raw["dt"] <= HAZ_MAX_DT].copy()
    d["pi"] = bucket(d["p"].to_numpy())
    d["ti"] = tau_band(d["tau"].to_numpy())
    d = d[[t in r2 for t in zip(d["pi"], d["ti"])]]
    away = np.where(d["p"] <= 0.5, d["dp"], -d["dp"])
    d["k"] = np.round(away / unit).astype(int)
    dist = np.minimum(d["p"], 1 - d["p"])
    d["di"] = np.clip(np.searchsorted(DIST_EDGES, dist, side="right") - 1,
                      0, len(DIST_EDGES) - 2)
    return d


# =========================================================== (D) endgame
TAU_MAX = 30.0
TAU_BINS = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 30.0])
SPEND_BANDS = [1.0, 3.0, 7.0, 30.0]
END_MIN_BIN = 100
END_MIN_MOVERS = 50


def latent_size(k: np.ndarray) -> float:
    """MLE sd (ticks) of a discrete normal observed only at |k| >= 2."""
    vals, counts = np.unique(np.abs(k), return_counts=True)

    def nll(s):
        vis = 2 * norm.sf(1.5 / s)
        p = (norm.cdf((vals + 0.5) / s) - norm.cdf((vals - 0.5) / s)) * 2 / vis
        return -float(counts @ np.log(p + 1e-300))

    return float(minimize_scalar(nll, bounds=(0.3, 200), method="bounded").x)


def _wls_loglog(cells: pd.DataFrame, ycol: str) -> dict:
    c = cells[np.isfinite(cells[ycol]) & (cells[ycol] > 0)]
    if len(c) < 6:
        return {"a": np.nan, "g": np.nan, "b": np.nan, "n_cells": len(c)}
    X = np.column_stack([np.ones(len(c)), np.log(c["mpq"]), np.log(c["mtau"])])
    w = np.sqrt(c["n"].to_numpy())
    beta, *_ = np.linalg.lstsq(X * w[:, None], np.log(c[ycol]) * w, rcond=None)
    return {"a": float(np.exp(beta[0])), "g": float(beta[1]),
            "b": float(-beta[2]), "n_cells": int(len(c))}


def endgame_decompose(raw: pd.DataFrame, unit: float) -> dict:
    """Arrival/size/total power laws on tau <= 30d, tick-de-censored."""
    d = raw[(raw["tau"] <= TAU_MAX) & (raw["dt"] <= HAZ_MAX_DT)].copy()
    d["k"] = np.round(d["dp"] / unit).astype(int)
    d["move"] = d["k"].abs() >= 2
    d["pq"] = d["p"] * (1 - d["p"])
    pq_bins = np.unique(np.quantile(d["pq"], np.linspace(0, 1, 11)))
    d["ci"] = np.searchsorted(pq_bins, d["pq"], side="right") - 1
    d["tj"] = np.searchsorted(TAU_BINS, d["tau"], side="right") - 1
    rows = []
    for (ci, tj), g in d.groupby(["ci", "tj"]):
        if len(g) < END_MIN_BIN:
            continue
        row = {"ci": ci, "tj": tj, "n": len(g), "mpq": g["pq"].mean(),
               "mtau": g["tau"].mean(), "tot": (g["dp2"] / g["dt"]).mean(),
               "nu": np.nan, "size": np.nan}
        mov = g.loc[g["move"], "k"].to_numpy()
        if len(mov) >= END_MIN_MOVERS:
            s_tk = latent_size(mov)
            vis = 2 * norm.sf(1.5 / s_tk)
            row["size"] = (s_tk * unit) ** 2
            row["nu"] = min(float(g["move"].mean()) / vis, 1.0)
        rows.append(row)
    g = pd.DataFrame(rows)
    return {"arrival": _wls_loglog(g, "nu"), "size": _wls_loglog(g, "size"),
            "total": _wls_loglog(g, "tot"), "n_obs": int(len(d)),
            "move_frac": float(d["move"].mean())}


def spend_profile(mkts: list[Market]) -> pd.DataFrame:
    """Lifetime-variance share (traversal + terminal jump) by tau band."""
    bands = [0.0] + SPEND_BANDS + [np.inf]
    acc = np.zeros(len(bands))          # last slot = terminal jump
    for m in mkts:
        dp2 = np.diff(m.closes) ** 2
        tau = np.maximum(m.t_res - m.ts_days[:-1], 0.0)
        idx = np.clip(np.searchsorted(bands, tau, side="left") - 1,
                      0, len(bands) - 2)
        np.add.at(acc, idx, dp2)
        acc[-1] += (m.final - m.closes[-1]) ** 2
    labels = [f"tau<={bands[i + 1]:.0f}d" if i == 0 else
              f"tau({bands[i]:.0f},{bands[i + 1]:.0f}]d"
              if np.isfinite(bands[i + 1]) else f"tau>{bands[i]:.0f}d"
              for i in range(len(bands) - 1)] + ["terminal jump"]
    return pd.DataFrame({"band": labels, "share": acc / acc.sum()})


def simulate_compound(n: int, a: float, b: float, g1: float, m: float,
                      d: float, g2: float, seed: int = SEED) -> list[Market]:
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
            if rng.uniform() < min(a * pq ** g1 / tau ** b, 1.0):
                s2 = m * pq ** g2 / tau ** d
                dp = rng.choice([-1, 1]) * abs(rng.normal(0, np.sqrt(s2)))
                p = float(np.clip(p + np.round(dp, 2), 0.01, 0.99))
            path.append(p)
        out.append(Market(f"sim{i:05d}", np.arange(L + 1, dtype=float),
                          np.array(path), float(rng.uniform() < p), float(L)))
    return out


# ================================================================ drivers
def run_map(mkts: list[Market], label: str, write: bool = True):
    cells = map_cells(move_panel(mkts))
    rect = r1_rect(cells)
    shares = budget_by_regime(cells)
    print(f"\n===== {label}: regime map ({len(cells)} cells) =====")
    show = cells.drop(columns=["pi", "ti"]).copy()
    show["regime"] = show["regime"].map(REGIME_NAMES)
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("budget share by regime: "
          + "  ".join(f"{k} {v:.3f}" for k, v in shares.items()))
    print(f"R1 rectangle: p in [{rect['p_lo']:.2f}, {rect['p_hi']:.2f}), "
          f"tau >= {rect['tau_lo']:.0f}d")
    if write:
        RESULTS.mkdir(exist_ok=True)
        cells.to_csv(RESULTS / f"regimes_cells_{label}.csv", index=False)
        with open(RESULTS / f"regimes_scalars_{label}.json", "w") as fh:
            json.dump({"n_cells": len(cells), "budget_by_regime": shares,
                       "rect": {k: rect[k] for k in ("p_lo", "p_hi", "tau_lo")},
                       "thresholds": {"still": STILL_THRESH,
                                      "kurt": KURT_THRESH}}, fh, indent=1)
    return cells, rect


def run_slv(mkts: list[Market], rect: dict, label: str,
            max_obs: int = MAX_OBS, write: bool = True) -> dict:
    segs = slv_segments(mkts, rect)
    rng = np.random.default_rng(SEED)
    rng.shuffle(segs)
    keep, tot = [], 0
    for s in segs:
        keep.append(s)
        tot += len(s[0]) - 1
        if tot >= max_obs:
            break
    print(f"\n===== {label}: R1 SLV ===== {len(keep)}/{len(segs)} segments, "
          f"{tot} filtered steps")
    batches = slv_batches(keep)
    start = slv_start(batches)
    fits = [slv_fit(batches, v, start) for v in SLV_VARIANTS]
    full = fits[0]
    rows = []
    for r in fits:
        row = {"variant": r["variant"],
               **{k: r["theta"][k] for k in SLV_PARAMS},
               "ll_per_obs": r["ll_per_obs"],
               "d_ll": r["ll_per_obs"] - full["ll_per_obs"]}
        if r is not full:
            row["LR_p"] = float(chi2.sf(2 * (full["ll"] - r["ll"]),
                                        full["n_free"] - r["n_free"]))
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))
    _, _, pits = slv_loglik(full["theta"], batches, collect=True)
    dec = (np.histogram(pits, bins=10, range=(0, 1))[0] / len(pits)).tolist()
    print("PIT deciles: " + " ".join(f"{v:.3f}" for v in dec))
    out = {"n_segments": len(keep), "n_steps": tot,
           "variants": rows, "pit_deciles": dec}
    if write:
        with open(RESULTS / f"slv_{label}.json", "w") as fh:
            json.dump(out, fh, indent=1)
    return out


def run_hazard(mkts: list[Market], cells: pd.DataFrame, label: str,
               unit: float, write: bool = True) -> pd.DataFrame:
    d = r2_ticks(panel(mkts), cells, unit)
    print(f"\n===== {label}: R2 itm-hazard ===== {len(d)} obs "
          f"({d['key'].nunique() if len(d) else 0} markets), tick {unit}")
    rows = []
    for (di, ti), g in d.groupby(["di", "ti"]):
        if len(g) < HAZ_MIN_CELL:
            continue
        th = haz_fit(g["k"].to_numpy())
        rows.append({"dist_bucket": f"[{DIST_EDGES[di]:.2f},"
                     f"{min(DIST_EDGES[di + 1], 0.5):.2f})",
                     "tau_band": tau_label(int(ti)), "n": len(g),
                     **haz_summary(th, g["k"].to_numpy())})
    out = pd.DataFrame(rows)
    if len(out):
        print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        print("no R2 cells above the fit floor")
    if write:
        out.to_csv(RESULTS / f"hazard_{label}.csv", index=False)
    return out


def run_endgame(mkts: list[Market], label: str, unit: float,
                write: bool = True) -> dict:
    dec = endgame_decompose(panel(mkts), unit)
    print(f"\n===== {label}: R3 endgame ===== {dec['n_obs']} obs "
          f"(tau <= {TAU_MAX:.0f}d), move frac {dec['move_frac']:.3f}")
    for name in ("arrival", "size", "total"):
        r = dec[name]
        print(f"  {name:8s} a {r['a']:.3f}  g_pq {r['g']:.3f}  "
              f"b_tau {r['b']:.3f}  ({r['n_cells']} cells)")
    print(f"  check: g_arr+g_size {dec['arrival']['g'] + dec['size']['g']:.2f}"
          f" vs {dec['total']['g']:.2f};  b_arr+b_size "
          f"{dec['arrival']['b'] + dec['size']['b']:.2f}"
          f" vs {dec['total']['b']:.2f}")
    spend = spend_profile(mkts)
    print(spend.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if write:
        with open(RESULTS / f"endgame_{label}.json", "w") as fh:
            json.dump(dec, fh, indent=1)
        spend.to_csv(RESULTS / f"endgame_spend_{label}.csv", index=False)
    return dec


def run_venue(venue: str, skip_slv: bool = False) -> None:
    mkts = load_markets(venue)
    cells, rect = run_map(mkts, venue)
    run_hazard(mkts, cells, venue, UNIT[venue])
    run_endgame(mkts, venue, UNIT[venue])
    if not skip_slv:
        run_slv(mkts, rect, venue)


def run_oracles() -> None:
    """Recovery checks on known truths; the pytest gates mirror these."""
    print("===== oracle: map on clean simulator (expect ~all R1) =====")
    cells, _ = run_map(simulate(2000), "sim_clean")
    print("===== oracle: map on stale simulator (expect R2 cells) =====")
    run_map(simulate(2000, stale=0.5), "sim_stale")
    print("\n===== oracle: SLV recovery =====")
    truth = {"c": 0.02, "alpha": 0.8, "eta": 0.05, "phi": 0.95, "s": 0.7}
    batches = slv_batches(simulate_slv(400, truth))
    r = slv_fit(batches, "full", slv_start(batches))
    print(pd.DataFrame([{"param": k, "truth": truth[k],
                         "estimate": r["theta"][k]} for k in SLV_PARAMS])
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"({r['seconds']:.0f}s, converged={r['converged']})")
    print("\n===== oracle: hazard MLE recovery =====")
    rng = np.random.default_rng(SEED)
    grid = np.arange(-400, 401)
    for truth in ({"pi0": 0.70, "h": 0.04, "sigma": 0.8, "rho_a": 0.85,
                   "rho_t": 0.60, "w_a": 0.65},
                  {"pi0": 0.30, "h": 0.10, "sigma": 1.5, "rho_a": 0.70,
                   "rho_t": 0.75, "w_a": 0.40}):
        p = haz_pmf(grid, truth)
        th = haz_fit(rng.choice(grid, size=30_000, p=p / p.sum()))
        print(pd.DataFrame([{"param": k, "truth": truth[k], "estimate": th[k]}
                            for k in HAZ_PARAMS])
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n===== oracle: endgame exponent recovery =====")
    truth = {"a": 0.9, "b": 0.5, "g1": 0.3, "m": 0.03, "d": 0.5, "g2": 1.0}
    mkts = simulate_compound(4000, **truth)
    dec = endgame_decompose(panel(mkts), UNIT["sim"])
    print(f"arrival g {dec['arrival']['g']:.2f} (truth {truth['g1']}), "
          f"b {dec['arrival']['b']:.2f} (truth {truth['b']}); "
          f"size g {dec['size']['g']:.2f} (truth {truth['g2']}), "
          f"b {dec['size']['b']:.2f} (truth {truth['d']})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=["polymarket", "kalshi"])
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--skip-slv", action="store_true")
    args = ap.parse_args(argv)
    if not args.venue:
        run_oracles()
        if args.sim_only:
            return 0
    for v in ([args.venue] if args.venue else ["polymarket", "kalshi"]):
        run_venue(v, skip_slv=args.skip_slv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
