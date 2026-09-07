r"""V4: zero-inflated observation layer, horse race, budget integration.

V3 ended with a diagnosis: the SLV pins its vol-of-vol s at the bound on
both venues, and the grid probe showed the pin is structural, not
numerical -- the likelihood wants an ever-wider vol mixture because the
Gaussian observation layer cannot express the exact-zero point mass of
stale prints. V4 builds the layer the diagnosis asks for and then asks
the two payoff questions: does anything here predict better out of
sample, and does the fitted object integrate to budget-consistent
dynamics? Three blocks:

(A) ZI-SLV. The V3 latent structure (probit random walk x with an AR(1)
    log-intensity clock) with the observation moved onto the venue tick
    grid and zero-inflated:
      P(tick k | filter) = pi0 1{k = 0} + (1 - pi0) Integral_bin N(z; m, S)
    This is a proper likelihood in one shared (discrete) measure, so the
    nested pi0 = 0 fit ("zi-off") is directly comparable by LR test.
    Headline check: with the zero mass carried by pi0, does s come off
    the bound? S_MAX is deliberately raised to 3.0 here so an interior
    optimum is observable (V3 kept 2.0 as a guard; the probe showed the
    plain model pins at any bound you set). Filter approximations,
    stated: the Kalman update uses the bin-midpoint innovation (interval
    update replaced by a point update), and stale steps update the
    intensity weights only through the mixture responsibilities.

(B) Horse race. Walk-forward by resolution date (catalog end field), two
    folds (train first 50% / test next 25%, then 75/25). Target: one-day-
    ahead log P(k) on the venue tick grid; continuous models integrated
    over the bin so every model is scored in the same measure. Models:
      composite-zi  regime split; R1 cells scored by the ZI-SLV filter,
                    R2/R3 cells by per-cell frozen+gap tick mixtures
      composite     same but R1 scored by the plain SLV filter
      global-slv    one SLV fit on everything (no regimes)
      const-lam     SLV with lambda == 1
      bucket        per-cell Gaussian variance (memorise the rates)
      garch         pooled GARCH(1,1) on dp
    Mean log score per obs, pooled and per regime, block bootstrap over
    test markets for deltas vs the plain composite.

(C) Budget integration. Roll the generative composite forward from
    starts nearest tau = 30/14/7/3d (p0 in [0.05, 0.95]), 200 paths on
    the tick state grid, and compare remaining variance to the p0(1-p0)
    budget next to the realized continuation. Three R1 engines: the
    stored V3 full SLV, the block-A ZI-SLV (a stale day spends nothing),
    and the const-lambda control. V3's diagnosis predicts the full
    engine overshoots through the intensity tail; the question is
    whether the ZI engine closes it. In-sample by design (a consistency
    audit of the fitted object; the forecast contest is block B).
    Thetas are read from the stored V3/block-A results, so block C needs
    no new fits.

Oracle gates live in tests/test_race_v4.py: ZI recovery of pi0 (and its
null on clean data), batched-vs-single filter equivalence, tick masses
summing to one, race ranking on the simulator, and the realized budget
ratio ~ 1 (a theorem) on the simulator.

Outputs in results_v2/:
  zi_{label}.json          block A variant table + PIT deciles
  race_{label}.csv         block B per-observation scores
  race_summary_{label}.json
  budget_v4_{label}.csv    block C per-start rows
  budget_v4_{label}.json   per-band ratio table

Usage:
  python analysis/race_v4.py                      # oracles + both venues
  python analysis/race_v4.py --venue kalshi --block a
  python analysis/race_v4.py --sim-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import regimes_v3 as rv                             # noqa: E402
from analysis.config import DATA_V2, RESULTS, SEED, UNIT          # noqa: E402
from analysis.core import (Market, P_CLIP, bucket, load_markets,  # noqa: E402
                           panel, simulate, tau_band)
from analysis.empirics_v2 import move_panel                       # noqa: E402

# ============================================================== (A) ZI-SLV
S_MAX4 = 3.0        # wider than V3's 2.0 on purpose: interior s observable
PI0_MAX = 0.95
EDGE_CLIP = (1e-4, 1 - 1e-4)
ZI_PARAMS = ("c", "alpha", "eta", "phi", "s", "pi0")
ZI_VARIANTS = {"zi": {}, "zi-off": {"pi0": 0.0}}
MAX_OBS = 40_000    # match V3 so block A is read against the V3 tables


def make_seg(p: np.ndarray, dt: np.ndarray, tp: np.ndarray,
             unit: float) -> tuple:
    """(z, zlo, zhi, isz, dt, taup) for one contiguous run of closes."""
    pc = np.clip(p, *P_CLIP)
    z = norm.ppf(pc)
    zlo = norm.ppf(np.clip(pc - unit / 2, *EDGE_CLIP))
    zhi = norm.ppf(np.clip(pc + unit / 2, *EDGE_CLIP))
    k = np.round(np.diff(p) / unit)
    isz = np.concatenate([[False], k == 0])
    return (z, zlo, zhi, isz, dt, tp)


def zi_segments(mkts: list[Market], rect: dict, unit: float) -> list[tuple]:
    """In-rectangle runs, same segmentation rules as the V3 SLV."""
    segs = []
    for m in mkts:
        tau = np.maximum(m.t_res - m.ts_days, 1.0)
        ok = (m.closes >= rect["p_lo"]) & (m.closes < rect["p_hi"]) \
            & (tau >= rect["tau_lo"])
        gap = np.concatenate([[1.0], np.diff(m.ts_days)])
        run = np.cumsum(~ok | (gap > rv.MAX_GAP))
        for _, idx in pd.Series(np.arange(len(m.closes))).groupby(run):
            i = idx.to_numpy()
            i = i[ok[i]]
            if len(i) < rv.MIN_SEG:
                continue
            segs.append(make_seg(
                m.closes[i],
                np.concatenate([[np.nan], np.diff(m.ts_days[i])]),
                np.concatenate([[np.nan], tau[i][:-1]]), unit))
    return segs


def zi_batches(segs: list[tuple]) -> list[tuple]:
    segs = sorted(segs, key=lambda s: -len(s[0]))
    out = []
    for b0 in range(0, len(segs), rv.BATCH):
        chunk = segs[b0:b0 + rv.BATCH]
        M, T = len(chunk), max(len(s[0]) for s in chunk)
        Z, ZL, ZH, DT, TP = (np.full((M, T), np.nan) for _ in range(5))
        IZ = np.zeros((M, T), bool)
        L = np.empty(M, int)
        for i, (z, zl, zh, iz, dt, tp) in enumerate(chunk):
            L[i] = len(z)
            Z[i, :L[i]], ZL[i, :L[i]], ZH[i, :L[i]] = z, zl, zh
            IZ[i, :L[i]], DT[i, :L[i]], TP[i, :L[i]] = iz, dt, tp
        out.append((Z, ZL, ZH, IZ, DT, TP, L))
    return out


def zi_loglik(theta: dict, batches: list, collect: bool = False):
    """Filtered tick-mass log-likelihood of steps k >= 1 (proper measure)."""
    c, alpha, eta, pi0 = theta["c"], theta["alpha"], theta["eta"], theta["pi0"]
    g, w0, T = rv._lam_grid(theta["phi"], theta["s"])
    lam = np.exp(g)
    rng = np.random.default_rng(SEED)
    total, n_obs, pits = 0.0, 0, []
    for Z, ZL, ZH, IZ, DT, TP, L in batches:
        M, Tn = Z.shape
        K = len(g)
        m = np.repeat(Z[:, :1], K, axis=1)
        P = np.full((M, K), eta ** 2)
        w = np.tile(w0, (M, 1))
        for k in range(1, Tn):
            act = k < L
            if not act.any():
                break
            # collapsed moments as matmuls (see regimes_v3.slv_loglik)
            wp = w @ T
            inv = 1.0 / (wp + 1e-300)
            mp = ((w * m) @ T) * inv
            Pp = np.maximum(((w * (P + m * m)) @ T) * inv - mp ** 2, 0.0)
            step = np.where(act, DT[:, k] / TP[:, k] ** alpha, 1.0)
            Pp = Pp + c * lam[None, :] * step[:, None]
            S = Pp + eta ** 2
            sd = np.sqrt(S)
            zl = np.where(act, ZL[:, k], -1.0)[:, None]
            zh = np.where(act, ZH[:, k], 1.0)[:, None]
            z = np.where(act, Z[:, k], 0.0)[:, None]
            iz = (IZ[:, k] & act)[:, None]
            cdf_lo = ndtr((zl - mp) / sd)
            mass = np.maximum(ndtr((zh - mp) / sd) - cdf_lo, 1e-300)
            node_l = (1 - pi0) * mass + pi0 * iz
            logw = np.log(wp + 1e-300) + np.log(node_l + 1e-300)
            mx = logw.max(1, keepdims=True)
            step_ll = mx[:, 0] + np.log(np.exp(logw - mx).sum(1))
            total += step_ll[act].sum()
            n_obs += int(act.sum())
            if collect:
                prev = np.where(act, Z[:, k - 1] < ZL[:, k], False)
                f_lo = (1 - pi0) * (wp * cdf_lo).sum(1) + pi0 * prev
                pit = f_lo + rng.random(M) * np.exp(step_ll)
                pits.append(np.clip(pit[act], 0.0, 1.0))
            # within-node split: genuine (updated) vs stale (predicted)
            q = (1 - pi0) * mass / (node_l + 1e-300)
            gain = Pp / S
            resid = z - mp
            m_u = mp + gain * resid
            P_u = Pp * (1 - gain)
            m_n = q * m_u + (1 - q) * mp
            P_n = q * (P_u + (m_u - m_n) ** 2) + (1 - q) * (Pp + (mp - m_n) ** 2)
            a = act[:, None]
            m = np.where(a, m_n, m)
            P = np.where(a, P_n, P)
            w = np.where(a, np.exp(logw - step_ll[:, None]), w)
    if collect:
        return total, n_obs, np.concatenate(pits) if pits else np.array([])
    return total, n_obs


def _zi_pack(theta: dict, free: list[str]) -> np.ndarray:
    u = []
    for name in free:
        v = theta[name]
        if name == "phi":
            u.append(np.arctanh(np.clip(v, -0.999, 0.999)))
        elif name == "alpha":
            u.append(v)
        elif name == "s":
            q = np.clip(v / S_MAX4, 1e-6, 1 - 1e-6)
            u.append(np.log(q / (1 - q)))
        elif name == "eta":
            u.append(np.log(max(v - rv.ETA_MIN, 1e-8)))
        elif name == "pi0":
            q = np.clip(v / PI0_MAX, 1e-6, 1 - 1e-6)
            u.append(np.log(q / (1 - q)))
        else:
            u.append(np.log(v))
    return np.array(u)


def _zi_unpack(u: np.ndarray, free: list[str], fixed: dict) -> dict:
    theta = dict(fixed)
    for name, v in zip(free, u):
        if name == "phi":
            theta[name] = float(np.tanh(v))
        elif name == "alpha":
            theta[name] = float(v)
        elif name == "s":
            theta[name] = float(S_MAX4 / (1 + np.exp(-v)))
        elif name == "eta":
            theta[name] = float(rv.ETA_MIN + np.exp(v))
        elif name == "pi0":
            theta[name] = float(PI0_MAX / (1 + np.exp(-v)))
        else:
            theta[name] = float(np.exp(v))
    return theta


def zi_fit(batches: list, variant: str, start: dict,
           maxiter: int = 400) -> dict:
    fixed = ZI_VARIANTS[variant]
    free = [n for n in ZI_PARAMS if n not in fixed]

    def nll(u):
        ll, n = zi_loglik(_zi_unpack(u, free, fixed), batches)
        return -ll / n if n else np.inf

    t0 = time.time()
    res = minimize(nll, _zi_pack(start, free), method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5})
    theta = _zi_unpack(res.x, free, fixed)
    ll, n = zi_loglik(theta, batches)
    return {"variant": variant, "theta": theta, "ll": ll, "n": n,
            "ll_per_obs": ll / n, "n_free": len(free),
            "seconds": time.time() - t0, "converged": bool(res.success)}


def zi_start(batches: list) -> dict:
    Z, _, _, IZ, DT, TP, L = batches[0]
    y = np.diff(Z, axis=1) ** 2 * TP[:, 1:] ** 0.5 / DT[:, 1:]
    act = np.arange(1, Z.shape[1])[None, :] < L[:, None]
    zfrac = float(IZ[:, 1:][act].mean()) if act.any() else 0.2
    return {"c": max(float(np.nanmean(y)), 1e-4), "alpha": 0.5,
            "eta": 0.02, "phi": 0.9, "s": 0.7,
            "pi0": float(np.clip(0.5 * zfrac, 0.02, 0.6))}


def simulate_zi(n: int, theta: dict, unit: float = 0.01,
                seed: int = SEED) -> list[tuple]:
    """SLV latent path, prints rounded to the tick grid, stale w.p. pi0."""
    rng = np.random.default_rng(seed)
    segs = []
    for _ in range(n):
        L = int(rng.integers(40, 301))
        tau = np.arange(L, 0, -1.0)
        lg = rng.normal(0, theta["s"])
        x = rng.normal(0, 0.7)
        p = np.empty(L)
        p[0] = norm.cdf(x + rng.normal(0, theta["eta"]))
        for t in range(1, L):
            lg = theta["phi"] * lg + rng.normal(
                0, theta["s"] * np.sqrt(max(1 - theta["phi"] ** 2, 1e-10)))
            x += rng.normal(0, np.sqrt(
                theta["c"] * np.exp(lg) / tau[t - 1] ** theta["alpha"]))
            if rng.random() < theta["pi0"]:
                p[t] = p[t - 1]
            else:
                p[t] = norm.cdf(x + rng.normal(0, theta["eta"]))
        obs = np.clip(np.round(p / unit) * unit, unit, 1 - unit)
        segs.append(make_seg(obs, np.concatenate([[np.nan], np.ones(L - 1)]),
                             np.concatenate([[np.nan], tau[:-1]]), unit))
    return segs


def run_zi(mkts: list[Market], rect: dict, label: str, unit: float,
           max_obs: int = MAX_OBS, write: bool = True) -> dict:
    segs = zi_segments(mkts, rect, unit)
    rng = np.random.default_rng(SEED)
    rng.shuffle(segs)
    keep, tot = [], 0
    for s in segs:
        keep.append(s)
        tot += len(s[0]) - 1
        if tot >= max_obs:
            break
    print(f"\n===== {label}: ZI-SLV ===== {len(keep)}/{len(segs)} segments, "
          f"{tot} steps (tick-mass measure, S_MAX={S_MAX4})")
    batches = zi_batches(keep)
    start = zi_start(batches)
    fits = [zi_fit(batches, v, start) for v in ZI_VARIANTS]
    full = fits[0]
    rows = []
    for r in fits:
        row = {"variant": r["variant"],
               **{k: r["theta"][k] for k in ZI_PARAMS},
               "ll_per_obs": r["ll_per_obs"],
               "d_ll": r["ll_per_obs"] - full["ll_per_obs"]}
        if r is not full:
            row["LR_p"] = float(chi2.sf(2 * (full["ll"] - r["ll"]),
                                        full["n_free"] - r["n_free"]))
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))
    s_hat = full["theta"]["s"]
    print(f"s = {s_hat:.3f} -> " + ("INTERIOR (ZI unpins the vol mixture)"
          if s_hat < 0.95 * S_MAX4 else "still pinned at S_MAX"))
    _, _, pits = zi_loglik(full["theta"], batches, collect=True)
    dec = (np.histogram(pits, bins=10, range=(0, 1))[0] / len(pits)).tolist()
    print("PIT deciles: " + " ".join(f"{v:.3f}" for v in dec))
    out = {"n_segments": len(keep), "n_steps": tot, "s_max": S_MAX4,
           "variants": rows, "pit_deciles": dec}
    if write:
        RESULTS.mkdir(exist_ok=True)
        with open(RESULTS / f"zi_{label}.json", "w") as fh:
            json.dump(out, fh, indent=1)
    return out


# ========================================================== (B) horse race
FOLDS = [(0.50, 0.75), (0.75, 1.00)]
RACE_MAX_OBS = 20_000
MAX_TEST = 1_500
MIN_CELL_FIT = 500
N_BOOT = 1000
LL_FLOOR = float(np.log(1e-12))
MODELS = ("composite-zi", "composite", "global-slv", "const-lam",
          "bucket", "garch")
ALL_RECT = {"p_lo": 0.0, "p_hi": 1.0, "tau_lo": 0.0}


def res_order(venue: str, mkts: list[Market]) -> list[Market]:
    """Markets sorted by catalog resolution (end) date."""
    cat = pd.read_parquet(DATA_V2 / "catalog" / f"{venue}.parquet",
                          columns=["key", "end"])
    end = dict(zip(cat.key.astype(str),
                   pd.to_datetime(cat["end"], utc=True, errors="coerce")))
    tmin = pd.Timestamp.min.tz_localize("UTC")

    def when(m):
        v = end.get(m.key)
        return v if pd.notna(v) else tmin
    return sorted(mkts, key=when)


def fit_garch(train: list[Market], max_obs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    dps, tot = [], 0
    for i in rng.permutation(len(train)):
        dp = np.diff(train[i].closes)
        if not len(dp):
            continue
        dps.append(dp)
        tot += len(dp)
        if tot >= max_obs:
            break
    # pad to a matrix: the recursion is sequential in t but vectorised
    # across series (padded cells recurse harmlessly and are never scored)
    M, Tn = len(dps), max(len(d) for d in dps)
    X = np.zeros((M, Tn))
    ACT = np.zeros((M, Tn), bool)
    for i, d in enumerate(dps):
        X[i, :len(d)], ACT[i, :len(d)] = d, True
    X2 = X * X
    n_tot = int(ACT.sum())

    def nll(u):
        w = np.exp(u[0])
        a = 0.999 / (1 + np.exp(-u[1]))
        b = (1 - a) * 0.999 / (1 + np.exp(-u[2]))
        s2 = np.full(M, w / max(1 - a - b, 1e-6))
        ll = 0.0
        for t in range(Tn):
            step = np.log(s2) + X2[:, t] / s2
            ll += step[ACT[:, t]].sum()
            s2 = w + a * X2[:, t] + b * s2
        return 0.5 * (ll + n_tot * np.log(2 * np.pi)) / n_tot

    res = minimize(nll, np.array([np.log(1e-4), 0.0, 1.0]),
                   method="Nelder-Mead", options={"maxiter": 250})
    u = res.x
    a = 0.999 / (1 + np.exp(-u[1]))
    return {"w": float(np.exp(u[0])), "a": float(a),
            "b": float((1 - a) * 0.999 / (1 + np.exp(-u[2])))}


def _sub_fit(fitter, seg_fn, batch_fn, start_fn, train, rect, variant,
             max_obs, maxiter, seed, **kw):
    segs = seg_fn(train, rect, **kw)
    rng = np.random.default_rng(seed)
    rng.shuffle(segs)
    keep, tot = [], 0
    for s in segs:
        keep.append(s)
        tot += len(s[0]) - 1
        if tot >= max_obs:
            break
    b = batch_fn(keep)
    return fitter(b, variant, start_fn(b), maxiter=maxiter)["theta"]


def fit_fold(train: list[Market], unit: float, max_obs: int, seed: int,
             maxiter: int = 300) -> dict:
    """All training-side objects for one fold."""
    pan = move_panel(train)
    cells = rv.map_cells(pan)
    rect = rv.r1_rect(cells)
    regime = {(int(r["pi"]), int(r["ti"])): int(r["regime"])
              for _, r in cells.iterrows()}
    print(f"  train {len(train)} mkts, {len(pan)} obs; R1 rect "
          f"p [{rect['p_lo']:.2f},{rect['p_hi']:.2f}) "
          f"tau >= {rect['tau_lo']:.0f}d")

    def slv(rect_, variant):
        return _sub_fit(rv.slv_fit, lambda t, r: rv.slv_segments(t, r),
                        rv.slv_batches, rv.slv_start, train, rect_, variant,
                        max_obs, maxiter, seed)

    th_r1 = slv(rect, "full")
    th_zi = _sub_fit(zi_fit, zi_segments, zi_batches, zi_start, train, rect,
                     "zi", max_obs, maxiter, seed, unit=unit)
    th_gl = slv(ALL_RECT, "full")
    th_cl = slv(ALL_RECT, "no-sv")
    for nm, th in (("r1", th_r1), ("zi", th_zi), ("glob", th_gl),
                   ("clam", th_cl)):
        print(f"  theta {nm:4s} " + " ".join(
            f"{k}={th[k]:.3f}" for k in th if k in ZI_PARAMS))

    raw = panel(train)
    raw["pi"] = bucket(raw["p"].to_numpy())
    raw["ti"] = tau_band(raw["tau"].to_numpy())
    away = np.where(raw["p"] <= 0.5, raw["dp"], -raw["dp"])
    raw["k"] = np.round(away / unit).astype(int)
    mix, bvar = {}, {}
    for (pi, ti), grp in raw.groupby(["pi", "ti"]):
        bvar[(int(pi), int(ti))] = float(grp["dp"].var()) or unit ** 2
        if len(grp) >= MIN_CELL_FIT and regime.get((int(pi), int(ti)), 1) != 1:
            mix[(int(pi), int(ti))] = rv.haz_fit(grp["k"].to_numpy())
    bvar["global"] = float(raw["dp"].var())
    garch = fit_garch(train, max_obs, seed)
    print(f"  {len(mix)} cell mixtures; garch w={garch['w']:.2e} "
          f"a={garch['a']:.3f} b={garch['b']:.3f}")
    return {"regime": regime, "th_r1": th_r1, "th_zi": th_zi,
            "th_glob": th_gl, "th_clam": th_cl, "mix": mix, "bvar": bvar,
            "garch": garch}


def slv_forecast(theta: dict, z, dt, taup) -> list[tuple]:
    """One-step predictive mixture (wp, mp, S) at each step k >= 1."""
    c, alpha, eta = theta["c"], theta["alpha"], theta["eta"]
    g, w0, T = rv._lam_grid(theta["phi"], theta["s"])
    lam = np.exp(g)
    K = len(g)
    m = np.full(K, z[0])
    P = np.full(K, eta ** 2)
    w = w0.copy()
    out = []
    for k in range(1, len(z)):
        wp = w @ T
        inv = 1.0 / (wp + 1e-300)
        mp = ((w * m) @ T) * inv
        Pp = np.maximum(((w * (P + m * m)) @ T) * inv - mp ** 2, 0.0)
        Pp = Pp + c * lam * dt[k] / taup[k] ** alpha
        S = Pp + eta ** 2
        out.append((wp.copy(), mp.copy(), S.copy()))
        ll = -0.5 * (np.log(2 * np.pi * S) + (z[k] - mp) ** 2 / S)
        logw = np.log(wp + 1e-300) + ll
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        gain = Pp / S
        m = mp + gain * (z[k] - mp)
        P = Pp * (1 - gain)
    return out


def zi_forecast(theta: dict, seg: tuple):
    """Predictives (wp, mp, S) + per-step tick-mass ll for one path."""
    z, zl, zh, iz, dt, tp = seg
    c, alpha, eta, pi0 = theta["c"], theta["alpha"], theta["eta"], theta["pi0"]
    g, w0, T = rv._lam_grid(theta["phi"], theta["s"])
    lam = np.exp(g)
    K = len(g)
    m = np.full(K, z[0])
    P = np.full(K, eta ** 2)
    w = w0.copy()
    out, lls = [], []
    for k in range(1, len(z)):
        wp = w @ T
        inv = 1.0 / (wp + 1e-300)
        mp = ((w * m) @ T) * inv
        Pp = np.maximum(((w * (P + m * m)) @ T) * inv - mp ** 2, 0.0)
        Pp = Pp + c * lam * dt[k] / tp[k] ** alpha
        S = Pp + eta ** 2
        sd = np.sqrt(S)
        out.append((wp.copy(), mp.copy(), S.copy()))
        mass = np.maximum(ndtr((zh[k] - mp) / sd)
                          - ndtr((zl[k] - mp) / sd), 1e-300)
        node_l = (1 - pi0) * mass + pi0 * bool(iz[k])
        obs_l = float(wp @ node_l)
        lls.append(np.log(obs_l + 1e-300))
        q = (1 - pi0) * mass / (node_l + 1e-300)
        gain = Pp / S
        m_u = mp + gain * (z[k] - mp)
        P_u = Pp * (1 - gain)
        m_n = q * m_u + (1 - q) * mp
        P = q * (P_u + (m_u - m_n) ** 2) + (1 - q) * (Pp + (mp - m_n) ** 2)
        m = m_n
        w = wp * node_l / (obs_l + 1e-300)
        w /= w.sum()
    return out, np.array(lls)


def slv_bin_prob(fore: tuple, p_prev: float, k: int, unit: float) -> float:
    wp, mp, S = fore
    lo = np.clip(p_prev + (k - 0.5) * unit, *EDGE_CLIP)
    hi = np.clip(p_prev + (k + 0.5) * unit, *EDGE_CLIP)
    sd = np.sqrt(S)
    pr = float((wp * (ndtr((ndtri(hi) - mp) / sd)
                      - ndtr((ndtri(lo) - mp) / sd))).sum())
    return max(pr, 1e-12)


def zi_bin_prob(fore: tuple, pi0: float, p_prev: float, k: int,
                unit: float) -> float:
    pr = (1 - pi0) * slv_bin_prob(fore, p_prev, k, unit)
    if k == 0:
        pr += pi0
    return max(pr, 1e-12)


def gauss_bin_prob(var: float, k: int, unit: float) -> float:
    sd = max(np.sqrt(var), 1e-9)
    return max(float(ndtr((k + 0.5) * unit / sd)
                     - ndtr((k - 0.5) * unit / sd)), 1e-12)


def score_market(m: Market, fitted: dict, unit: float, rng) -> list[dict]:
    p = m.closes
    pc = np.clip(p, *P_CLIP)
    z = norm.ppf(pc)
    tau = np.maximum(m.t_res - m.ts_days, 1.0)
    dt = np.concatenate([[np.nan], np.diff(m.ts_days)])
    taup = np.concatenate([[np.nan], tau[:-1]])
    seg = make_seg(p, dt, taup, unit)
    f_r1 = slv_forecast(fitted["th_r1"], z, dt, taup)
    f_gl = slv_forecast(fitted["th_glob"], z, dt, taup)
    f_cl = slv_forecast(fitted["th_clam"], z, dt, taup)
    f_zi, _ = zi_forecast(fitted["th_zi"], seg)
    pi0 = fitted["th_zi"]["pi0"]
    garch = fitted["garch"]
    s2 = garch["w"] / max(1 - garch["a"] - garch["b"], 1e-6)
    rows = []
    for j in range(1, len(p)):
        p_prev = float(p[j - 1])
        kk = int(round((p[j] - p[j - 1]) / unit))
        pi = int(bucket(np.array([np.clip(p_prev, 1e-6, 1 - 1e-6)]))[0])
        ti = int(tau_band(np.array([tau[j - 1]]))[0])
        reg = fitted["regime"].get((pi, ti), 1)
        bv = fitted["bvar"].get((pi, ti), fitted["bvar"]["global"])
        kf = kk if p_prev <= 0.5 else -kk

        pr_gl = slv_bin_prob(f_gl[j - 1], p_prev, kk, unit)
        pr_cl = slv_bin_prob(f_cl[j - 1], p_prev, kk, unit)
        pr_bk = gauss_bin_prob(bv, kk, unit)
        pr_ga = gauss_bin_prob(s2, kk, unit)
        th_mix = fitted["mix"].get((pi, ti)) if reg != 1 else None
        if reg == 1:
            pr_co = slv_bin_prob(f_r1[j - 1], p_prev, kk, unit)
            pr_cz = zi_bin_prob(f_zi[j - 1], pi0, p_prev, kk, unit)
        elif th_mix is not None:
            pr_co = pr_cz = max(float(rv.haz_pmf(np.array([kf]),
                                                 th_mix)[0]), 1e-12)
        else:
            pr_co = pr_cz = pr_bk
        # randomised tick PIT for composite-zi
        if reg == 1:
            wp, mp, S = f_zi[j - 1]
            sd = np.sqrt(S)
            zlo = seg[1][j]
            f_lo = float((1 - pi0) * (wp * ndtr((zlo - mp) / sd)).sum()
                         + pi0 * (z[j - 1] < zlo))
        elif th_mix is not None:
            ks = np.arange(-max(abs(kf), 2) - 60, kf)
            f_lo = float(rv.haz_pmf(ks, th_mix).sum())
        else:
            f_lo = float(ndtr((kk - 0.5) * unit / max(np.sqrt(bv), 1e-9)))
        pit = f_lo + rng.uniform() * pr_cz
        rows.append({"key": m.key, "tau": float(tau[j - 1]), "p": p_prev,
                     "regime": reg,
                     "composite-zi": max(float(np.log(pr_cz)), LL_FLOOR),
                     "composite": max(float(np.log(pr_co)), LL_FLOOR),
                     "global-slv": max(float(np.log(pr_gl)), LL_FLOOR),
                     "const-lam": max(float(np.log(pr_cl)), LL_FLOOR),
                     "bucket": max(float(np.log(pr_bk)), LL_FLOOR),
                     "garch": max(float(np.log(pr_ga)), LL_FLOOR),
                     "pit": float(min(max(pit, 0.0), 1.0))})
        x = float(p[j] - p[j - 1])
        s2 = garch["w"] + garch["a"] * x * x + garch["b"] * s2
    return rows


def bootstrap_delta(df: pd.DataFrame, base: str, other: str, rng) -> tuple:
    per = df.groupby("key").agg(d=(base, "mean"), n=(base, "size"),
                                o=(other, "mean"))
    d = (per["d"] - per["o"]).to_numpy()
    n = per["n"].to_numpy()
    idx = rng.integers(0, len(per), (N_BOOT, len(per)))
    boots = (d[idx] * n[idx]).sum(1) / n[idx].sum(1)
    return tuple(float(v) for v in np.quantile(boots, [0.025, 0.975]))


def evaluate(label: str, rows: list[dict], seed: int,
             write: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    print(f"\n--- {label}: {len(df)} test obs, {df['key'].nunique()} markets")
    summary = []
    for mdl in MODELS:
        row = {"model": mdl, "ll_per_obs": float(df[mdl].mean()),
               "delta_vs_composite": float(df[mdl].mean()
                                           - df["composite"].mean())}
        if mdl != "composite":
            row["d_ci_lo"], row["d_ci_hi"] = bootstrap_delta(
                df, "composite", mdl, rng)
        summary.append(row)
    out = pd.DataFrame(summary)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("(delta > 0 beats plain composite; CI = 95% block bootstrap of")
    print(" composite - model over test markets, so CI < 0 means model wins)")
    reg = df.groupby("regime")[list(MODELS)].mean()
    reg.index = [rv.REGIME_NAMES[i] for i in reg.index]
    print("mean log score by train-map regime:")
    print(reg.to_string(float_format=lambda v: f"{v:.3f}"))
    dec = (np.histogram(df["pit"], bins=10, range=(0, 1))[0]
           / len(df)).tolist()
    print("composite-zi randomised PIT deciles:")
    print("  " + " ".join(f"{v:.3f}" for v in dec))
    if write:
        df.to_csv(RESULTS / f"race_{label}.csv", index=False)
        with open(RESULTS / f"race_summary_{label}.json", "w") as fh:
            json.dump({"n_obs": len(df), "models": summary,
                       "by_regime": {str(i): {mdl: float(reg.loc[i, mdl])
                                              for mdl in MODELS}
                                     for i in reg.index},
                       "pit_deciles": dec}, fh, indent=1)
    return df


def run_race(label: str, ordered: list[Market], unit: float,
             max_obs: int = RACE_MAX_OBS, max_test: int = MAX_TEST,
             maxiter: int = 300, seed: int = SEED,
             write: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for lo, hi in FOLDS:
        n = len(ordered)
        train = ordered[:int(lo * n)]
        test = ordered[int(lo * n):int(hi * n)]
        if len(test) > max_test:
            test = [test[i] for i in rng.choice(len(test), max_test, False)]
        print(f"\nfold train<{lo:.0%} test [{lo:.0%},{hi:.0%}): "
              f"{len(train)} train / {len(test)} test markets")
        fitted = fit_fold(train, unit, max_obs, seed, maxiter=maxiter)
        for m in test:
            rows.extend(score_market(m, fitted, unit, rng))
    return evaluate(label, rows, seed, write=write)


# ==================================================== (C) budget integration
JUMP_THRESH = 0.02
START_BANDS = (30.0, 14.0, 7.0, 3.0)
K_MAX = 400
N_PATHS = 200
MAX_START_MKTS = 2_000
ENGINES = ("full", "zi", "clam")


def stored_thetas(label: str) -> dict:
    """R1 engines from the stored V3 + block-A results (no refits)."""
    with open(RESULTS / f"slv_{label}.json") as fh:
        slv = {r["variant"]: r for r in json.load(fh)["variants"]}
    with open(RESULTS / f"zi_{label}.json") as fh:
        zi = {r["variant"]: r for r in json.load(fh)["variants"]}
    pick = lambda r, ks: {k: float(r[k]) for k in ks}       # noqa: E731
    return {"full": pick(slv["full"], rv.SLV_PARAMS),
            "clam": pick(slv["no-sv"], rv.SLV_PARAMS),
            "zi": pick(zi["zi"], ZI_PARAMS)}


def mc_tables(regime: dict, mix: dict, bvar: dict):
    from analysis.core import P_BUCKETS, TAU_EDGES
    n_pi, n_ti = len(P_BUCKETS) - 1, len(TAU_EDGES) - 1
    reg = np.ones((n_pi, n_ti), int)
    for (pi, ti), r in regime.items():
        reg[pi, ti] = r
    kg = np.arange(-K_MAX, K_MAX + 1)
    cdf = {}
    for cell, th in mix.items():
        pr = np.maximum(rv.haz_pmf(kg, th), 0.0)
        cdf[cell] = np.cumsum(pr / pr.sum())
    bv = np.full((n_pi, n_ti), bvar["global"])
    for cell, v in bvar.items():
        if cell != "global":
            bv[cell] = v
    return reg, kg, cdf, bv, n_ti


def mc_engine(p0: np.ndarray, tau0: np.ndarray, tables, th: dict,
              unit: float, n_paths: int, rng,
              zi: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per-start mean (diffusive, jump) remaining variance, batched MC."""
    reg_arr, kg, cdf, bv, n_ti = tables
    c, alpha, phi, s = th["c"], th["alpha"], th["phi"], th["s"]
    pi0 = th.get("pi0", 0.0) if zi else 0.0
    g, _, _ = rv._lam_grid(phi, s)
    lo_l, hi_l = float(g.min()), float(g.max())
    n_s = len(p0)
    N = n_s * n_paths
    p = np.repeat(p0, n_paths)
    tleft = np.repeat(np.ceil(tau0), n_paths)
    sd0 = s if s > 1e-6 else 1e-6
    logl = np.clip(rng.normal(0.0, sd0, N), lo_l, hi_l)
    var_d, var_j = np.zeros(N), np.zeros(N)
    for _ in range(int(tleft.max())):
        idx = np.flatnonzero(tleft > 0)
        if not len(idx):
            break
        pa, ta = p[idx], tleft[idx]
        ti = tau_band(ta).astype(int)
        pi = bucket(np.clip(pa, 1e-6, 1 - 1e-6)).astype(int)
        reg = reg_arr[pi, ti]
        dp = np.zeros(len(idx))
        r1 = reg == 1
        if r1.any():
            i1 = idx[r1]
            lam = np.exp(logl[i1])
            v = c * lam / np.maximum(ta[r1], 1.0) ** alpha
            zz = ndtri(np.clip(pa[r1], *P_CLIP))
            pn = ndtr(zz + np.sqrt(v) * rng.standard_normal(len(i1)))
            move = pn - pa[r1]
            if pi0 > 0:
                move = np.where(rng.random(len(i1)) < pi0, 0.0, move)
            dp[r1] = move
        nr = np.flatnonzero(~r1)
        if len(nr):
            code = pi[nr] * n_ti + ti[nr]
            for cval in np.unique(code):
                sel = nr[code == cval]
                cell = (int(cval) // n_ti, int(cval) % n_ti)
                if cell in cdf:
                    u = rng.random(len(sel))
                    k = kg[np.searchsorted(cdf[cell], u)]
                    dp[sel] = np.where(pa[sel] <= 0.5, k, -k) * unit
                else:
                    dp[sel] = np.sqrt(bv[cell]) \
                        * rng.standard_normal(len(sel))
        pn = np.clip(pa + dp, *P_CLIP)
        d = pn - pa
        jump = np.abs(d) >= JUMP_THRESH
        var_j[idx] += np.where(jump, d * d, 0.0)
        var_d[idx] += np.where(jump, 0.0, d * d)
        p[idx] = pn
        logl[idx] = np.clip(
            phi * logl[idx]
            + sd0 * np.sqrt(max(1 - phi ** 2, 1e-10))
            * rng.standard_normal(len(idx)), lo_l, hi_l)
        tleft[idx] -= 1.0
    var_j += p * (1 - p)    # exact expectation of the terminal jump
    return (var_d.reshape(n_s, n_paths).mean(1),
            var_j.reshape(n_s, n_paths).mean(1))


def pick_starts(m: Market):
    tau = m.t_res - m.ts_days
    for b in START_BANDS:
        i = int(np.argmin(np.abs(tau - b)))
        if abs(tau[i] - b) <= 0.4 * b and i <= len(m.closes) - 2:
            p0 = float(m.closes[i])
            if 0.05 <= p0 <= 0.95:
                yield b, i, p0, float(tau[i])


def realized_remaining(m: Market, i: int) -> tuple[float, float]:
    dp = np.diff(m.closes[i:])
    jump = np.abs(dp) >= JUMP_THRESH
    return (float((dp[~jump] ** 2).sum()),
            float((dp[jump] ** 2).sum()) + (m.final - m.closes[-1]) ** 2)


def budget_fit_cheap(mkts: list[Market], unit: float) -> tuple:
    """Regime map + mixtures + buckets on all markets (no SLV fits)."""
    cells = rv.map_cells(move_panel(mkts))
    regime = {(int(r["pi"]), int(r["ti"])): int(r["regime"])
              for _, r in cells.iterrows()}
    raw = panel(mkts)
    raw["pi"] = bucket(raw["p"].to_numpy())
    raw["ti"] = tau_band(raw["tau"].to_numpy())
    away = np.where(raw["p"] <= 0.5, raw["dp"], -raw["dp"])
    raw["k"] = np.round(away / unit).astype(int)
    mix, bvar = {}, {}
    for (pi, ti), grp in raw.groupby(["pi", "ti"]):
        bvar[(int(pi), int(ti))] = float(grp["dp"].var()) or unit ** 2
        if len(grp) >= MIN_CELL_FIT and regime.get((int(pi), int(ti)), 1) != 1:
            mix[(int(pi), int(ti))] = rv.haz_fit(grp["k"].to_numpy())
    bvar["global"] = float(raw["dp"].var())
    return regime, mix, bvar


def run_budget(mkts: list[Market], label: str, unit: float,
               thetas: dict | None = None, n_paths: int = N_PATHS,
               max_mkts: int = MAX_START_MKTS, seed: int = SEED,
               write: bool = True) -> pd.DataFrame:
    thetas = thetas or stored_thetas(label)
    regime, mix, bvar = budget_fit_cheap(mkts, unit)
    tables = mc_tables(regime, mix, bvar)
    rng = np.random.default_rng(seed)
    if len(mkts) > max_mkts:
        mkts = [mkts[i] for i in rng.choice(len(mkts), max_mkts, False)]
    rows = []
    for m in mkts:
        for band, i, p0, tau0 in pick_starts(m):
            evd, evj = realized_remaining(m, i)
            rows.append({"key": m.key, "band": band, "p0": p0, "tau0": tau0,
                         "budget": p0 * (1 - p0), "evd": evd, "evj": evj})
    df = pd.DataFrame(rows)
    p0v, t0v = df["p0"].to_numpy(), df["tau0"].to_numpy()
    for eng in ENGINES:
        vd, vj = mc_engine(p0v, t0v, tables, thetas[eng], unit, n_paths,
                           rng, zi=(eng == "zi"))
        df[f"{eng}_vd"], df[f"{eng}_vj"] = vd, vj
    print(f"\n=== {label}: budget integration ===  {len(df)} starts from "
          f"{df['key'].nunique()} markets, {n_paths} paths")
    agg = df.groupby("band").sum(numeric_only=True)
    tab = pd.DataFrame({"n": df.groupby("band").size()})
    for eng in ENGINES:
        tab[f"{eng}_ratio"] = (agg[f"{eng}_vd"] + agg[f"{eng}_vj"]) \
            / agg["budget"]
    tab["real_ratio"] = (agg["evd"] + agg["evj"]) / agg["budget"]
    tab["zi_jump_share"] = agg["zi_vj"] / (agg["zi_vd"] + agg["zi_vj"])
    tab["real_jump_share"] = agg["evj"] / (agg["evd"] + agg["evj"])
    tab = tab.sort_index(ascending=False)
    tab.index = [f"tau~{b:.0f}d" for b in tab.index]
    print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    print("(ratio = remaining variance / p0(1-p0), ratio-of-sums; engines:")
    print(" full = V3 SLV, zi = block-A ZI-SLV, clam = constant lambda;")
    print(" realized ~ 1 is the budget theorem)")
    if write:
        df.to_csv(RESULTS / f"budget_v4_{label}.csv", index=False)
        with open(RESULTS / f"budget_v4_{label}.json", "w") as fh:
            json.dump({r: {c: float(tab.loc[r, c]) for c in tab.columns}
                       for r in tab.index}, fh, indent=1)
    return df


# ================================================================ drivers
def run_venue(venue: str, blocks: str, args) -> None:
    mkts = load_markets(venue)
    unit = UNIT[venue]
    if "a" in blocks:
        with open(RESULTS / f"regimes_scalars_{venue}.json") as fh:
            rect = json.load(fh)["rect"]
        run_zi(mkts, rect, venue, unit, max_obs=args.max_obs)
    if "b" in blocks:
        run_race(venue, res_order(venue, mkts), unit,
                 max_obs=args.race_max_obs, max_test=args.max_test)
    if "c" in blocks:
        run_budget(mkts, venue, unit, n_paths=args.n_paths)


def run_oracles() -> None:
    """Cheap recovery prints; the pytest gates are the real enforcement."""
    truth = {"c": 0.02, "alpha": 0.5, "eta": 0.02, "phi": 0.9, "s": 0.6,
             "pi0": 0.25}
    segs = simulate_zi(200, truth)
    b = zi_batches(segs)
    r = zi_fit(b, "zi", zi_start(b), maxiter=250)
    print("===== oracle: ZI recovery =====")
    print(pd.DataFrame([{"param": k, "truth": truth[k],
                         "estimate": r["theta"][k]} for k in ZI_PARAMS])
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"({r['seconds']:.0f}s, converged={r['converged']})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=["polymarket", "kalshi"])
    ap.add_argument("--block", default="abc")
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--max-obs", type=int, default=MAX_OBS)
    ap.add_argument("--race-max-obs", type=int, default=RACE_MAX_OBS)
    ap.add_argument("--max-test", type=int, default=MAX_TEST)
    ap.add_argument("--n-paths", type=int, default=N_PATHS)
    args = ap.parse_args(argv)
    if not args.venue:
        run_oracles()
        if args.sim_only:
            return 0
    for v in ([args.venue] if args.venue else ["polymarket", "kalshi"]):
        run_venue(v, args.block, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
