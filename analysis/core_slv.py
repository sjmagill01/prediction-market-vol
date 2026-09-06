"""Core-regime SLV fit. P1 of the modeling phase.

THEORY.md shows any admissible stochastic-vol model on [0,1] is forced into
SLV form Var(dp) = phi(Phi^-1(p))^2 lambda_t dt. In probit coordinates
z = Phi^-1(p) that is simply a random walk with stochastic clock speed, so we
fit, on the R1 (diffusive) rectangle from analysis/regime_map.py only:

  observation   z_k = x_k + e_k,          e_k ~ N(0, eta^2)  iid
  state         x_{k+1} - x_k ~ N(0, c * lambda_k * dt_k / tau_k^alpha)
  intensity     log lambda: AR(1), persistence phi, stationary sd s

Hyperparameters theta = (c, alpha, eta, phi, s) are pooled across all markets
of a venue (a single market never identifies a vol-of-vol; pooling thousands
does -- the anti-1V1J design). Estimation is exact conditional on a
discretised intensity: log lambda lives on a K-point grid, giving an HMM whose
emissions are linear-Gaussian in x; we run a collapsed mixture Kalman filter
(IMM: per grid node keep a Gaussian for x, mix with the HMM transition each
step) and maximise the filtered likelihood with Nelder-Mead.

Approximations, stated: the lambda transition matrix uses a 1-day step for
every observation gap (gaps are overwhelmingly 1 day inside R1; the x-step
variance does use the actual dt and tau); Gaussian observation noise stands
in for Kalshi's 1-cent tick grid (tick censoring is R2 business, phase P2).
Bounds s <= S_MAX and eta >= ETA_MIN are imposed because exact price repeats
otherwise let a near-zero-lambda node spike the likelihood without limit
(the standard Gaussian-mixture degeneracy); an estimate pinned at a bound
means the Gaussian layer is straining to imitate the zero-move point mass.

Nested variants answer "which layers earn their keep":
  full      (c, alpha, eta, phi, s)
  no-sv     lambda == 1                (is the stochastic clock real?)
  no-noise  eta ~= 0                   (is observation noise real?)
compared by per-observation log-likelihood, plus a PIT table for the full
model's one-step-ahead predictive (uniform = calibrated; a spike in the
middle deciles = the point mass of zero moves the Gaussian cannot express).

Oracle gate (--oracle): simulate from this exact model with known theta and
check the fit recovers it before believing anything on real data.

Usage:
  python analysis/core_slv.py --oracle [--n 600] [--seed 7]
  python analysis/core_slv.py --venue polymarket|kalshi|both
                              [--max-obs 40000] [--seed 7]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import regime_map as rm  # noqa: E402
from analysis import vol_check as vc   # noqa: E402

K_GRID = 13          # log-lambda grid points
S_MAX = 2.0          # bound on vol-of-vol: without it a near-zero-lambda grid
ETA_MIN = 1e-3       # node + tiny eta spike the likelihood on exact price
                     # repeats (unbounded Gaussian-mixture degeneracy)
MAX_GAP = 3.0        # break a segment at observation gaps > 3 days
MIN_SEG = 8          # minimum observations per segment
BATCH = 1024         # segments per padded batch (length-sorted to cut padding)

PARAM_NAMES = ("c", "alpha", "eta", "phi", "s")

VARIANTS = {
    "full":     {},
    "no-sv":    {"phi": 0.0, "s": 1e-8},
    "no-noise": {"eta": 1e-6},
}


# ------------------------------------------------------------------ segments

def build_segments(mkts, rect) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split each market into contiguous in-rectangle runs of daily closes.

    Returns (z, dt, tau_prev) per segment: z_k = Phi^-1(p_k); dt_k and
    tau_prev_k describe the interval (k-1, k] and are unused at k = 0.
    """
    segs = []
    for m in mkts:
        p = np.clip(m.closes, *vc.P_CLIP)
        tau = np.maximum(m.t_res - m.ts_days, 1.0)
        ok = (m.closes >= rect["p_lo"]) & (m.closes < rect["p_hi"]) & \
             (tau >= rect["tau_lo"])
        gap = np.concatenate([[1.0], np.diff(m.ts_days)])
        brk = ~ok | (gap > MAX_GAP)
        run_id = np.cumsum(brk)
        for _, idx in pd.Series(np.arange(len(p))).groupby(run_id):
            i = idx.to_numpy()
            i = i[ok[i]]
            if len(i) < MIN_SEG:
                continue
            segs.append((norm.ppf(p[i]),
                         np.concatenate([[np.nan], np.diff(m.ts_days[i])]),
                         np.concatenate([[np.nan], tau[i][:-1]])))
    return segs


def make_batches(segs) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pad length-sorted segments into (Z, DT, TAUP, lengths) batches."""
    segs = sorted(segs, key=lambda s: -len(s[0]))
    batches = []
    for b0 in range(0, len(segs), BATCH):
        chunk = segs[b0:b0 + BATCH]
        T = max(len(s[0]) for s in chunk)
        M = len(chunk)
        Z = np.full((M, T), np.nan)
        DT = np.full((M, T), np.nan)
        TP = np.full((M, T), np.nan)
        L = np.empty(M, int)
        for i, (z, dt, tp) in enumerate(chunk):
            L[i] = len(z)
            Z[i, :L[i]], DT[i, :L[i]], TP[i, :L[i]] = z, dt, tp
        batches.append((Z, DT, TP, L))
    return batches


# ------------------------------------------------------------------ filter

def _grid(phi: float, s: float):
    """Log-lambda grid, stationary weights, 1-day HMM transition matrix."""
    if s <= 1e-6:
        return np.zeros(1), np.ones(1), np.ones((1, 1))
    g = np.linspace(-3 * s, 3 * s, K_GRID)
    w0 = norm.pdf(g, 0, s)
    w0 /= w0.sum()
    innov = s * np.sqrt(max(1 - phi ** 2, 1e-10))
    T = norm.pdf(g[None, :], phi * g[:, None], innov) + 1e-300
    return g, w0, T / T.sum(1, keepdims=True)


def loglik(theta: dict, batches, collect: bool = False):
    """Filtered log-likelihood of steps k>=1; optionally collect PIT values."""
    c, alpha, eta = theta["c"], theta["alpha"], theta["eta"]
    g, w0, T = _grid(theta["phi"], theta["s"])
    lam = np.exp(g)
    total, n_obs, pits = 0.0, 0, []
    for Z, DT, TP, L in batches:
        M, Tn = Z.shape
        K = len(g)
        m = np.repeat(Z[:, :1], K, axis=1)          # x | first obs
        P = np.full((M, K), eta ** 2)
        w = np.tile(w0, (M, 1))
        for k in range(1, Tn):
            act = k < L
            if not act.any():
                break
            # HMM mixing + IMM moment-matched collapse
            W = w[:, :, None] * T[None, :, :]
            wp = W.sum(1)
            Mix = W / (wp[:, None, :] + 1e-300)
            mp = np.einsum("mij,mi->mj", Mix, m)
            dm = m[:, :, None] - mp[:, None, :]
            Pp = np.einsum("mij,mij->mj", Mix, P[:, :, None] + dm ** 2)
            # state predict (actual dt, tau) + Kalman update
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
            wn = np.exp(logw - step_ll[:, None])
            G = Pp / S
            a = act[:, None]
            m = np.where(a, mp + G * resid, m)
            P = np.where(a, Pp * (1 - G), P)
            w = np.where(a, wn, w)
    if collect:
        return total, n_obs, np.concatenate(pits) if pits else np.array([])
    return total, n_obs


# ------------------------------------------------------------------ fitting

def _pack(theta: dict, free: list[str]) -> np.ndarray:
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


def _unpack(u: np.ndarray, free: list[str], fixed: dict) -> dict:
    theta = dict(fixed)
    for name, v in zip(free, u):
        if name == "phi":
            theta[name] = np.tanh(v)
        elif name == "alpha":
            theta[name] = v
        elif name == "s":
            theta[name] = S_MAX / (1 + np.exp(-v))
        elif name == "eta":
            theta[name] = ETA_MIN + np.exp(v)
        else:
            theta[name] = np.exp(v)
    return theta


def fit(batches, variant: str, start: dict, maxiter: int = 400) -> dict:
    fixed = VARIANTS[variant]
    free = [n for n in PARAM_NAMES if n not in fixed]

    def nll(u):
        ll, n = loglik(_unpack(u, free, fixed), batches)
        return -ll / n if n else np.inf

    t0 = time.time()
    res = minimize(nll, _pack(start, free), method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5})
    theta = _unpack(res.x, free, fixed)
    ll, n = loglik(theta, batches)
    return {"variant": variant, "theta": theta, "ll": ll, "n": n,
            "ll_per_obs": ll / n, "n_free": len(free),
            "seconds": time.time() - t0, "converged": res.success}


def start_values(batches) -> dict:
    """Crude moment starts: c from mean dz^2 dt-tau-adjusted at alpha=0.5."""
    Z, DT, TP, L = batches[0]
    dz2 = np.diff(Z, axis=1) ** 2
    y = dz2 * TP[:, 1:] ** 0.5 / DT[:, 1:]
    c0 = float(np.nanmean(y))
    return {"c": max(c0, 1e-4), "alpha": 0.5, "eta": 0.02, "phi": 0.9, "s": 0.7}


# ------------------------------------------------------------------ oracle

def simulate_model(n: int, theta: dict, seed: int = 7) -> list:
    """Generate segments from the exact model (dt = 1, tau declining)."""
    rng = np.random.default_rng(seed)
    g, w0, T = _grid(theta["phi"], theta["s"])
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
        dt = np.concatenate([[np.nan], np.ones(L - 1)])
        tp = np.concatenate([[np.nan], tau[:-1]])
        segs.append((z, dt, tp))
    return segs


def run_oracle(n: int, seed: int) -> None:
    truth = {"c": 0.02, "alpha": 0.8, "eta": 0.05, "phi": 0.95, "s": 0.7}
    batches = make_batches(simulate_model(n, truth, seed))
    n_obs = sum(int(b[3].sum()) - len(b[3]) for b in batches)
    print(f"oracle: {n} segments, {n_obs} filtered steps, truth {truth}")
    r = fit(batches, "full", start_values(batches))
    rows = [{"param": k, "truth": truth[k], "estimate": r["theta"][k]}
            for k in PARAM_NAMES]
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))
    print(f"ll/obs {r['ll_per_obs']:.4f}  ({r['seconds']:.0f}s, "
          f"converged={r['converged']})")


# ------------------------------------------------------------------ venue run

def run_venue(venue: str, max_obs: int, seed: int) -> None:
    mkts = vc.load_markets(venue, None)
    panel = vc.candidate_rates(vc.daily_panel(mkts))
    rect = rm.r1_rectangle(rm.classify(rm.build_cells(panel)))
    print(f"\n=== {venue} ===  R1 rectangle: p in [{rect['p_lo']:.2f}, "
          f"{rect['p_hi']:.2f}), tau >= {rect['tau_lo']:.0f}d")
    segs = build_segments(mkts, rect)
    rng = np.random.default_rng(seed)
    rng.shuffle(segs)
    tot, keep = 0, []
    for s in segs:
        keep.append(s)
        tot += len(s[0]) - 1
        if tot >= max_obs:
            break
    print(f"{len(keep)} segments of {len(segs)} ({tot} filtered steps"
          f"{', subsampled' if len(keep) < len(segs) else ''})")
    batches = make_batches(keep)
    start = start_values(batches)
    results = [fit(batches, v, start) for v in VARIANTS]
    full = results[0]
    rows = []
    for r in results:
        row = {"variant": r["variant"], **{k: r["theta"][k] for k in PARAM_NAMES},
               "ll_per_obs": r["ll_per_obs"],
               "d_ll_per_obs": r["ll_per_obs"] - full["ll_per_obs"]}
        if r is not full:
            lr = 2 * (full["ll"] - r["ll"])
            row["LR_p"] = chi2.sf(lr, full["n_free"] - r["n_free"])
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))
    _, _, pits = loglik(full["theta"], batches, collect=True)
    dec = np.histogram(pits, bins=10, range=(0, 1))[0] / len(pits)
    print("PIT deciles (calibrated = 0.100 each):")
    print("  " + " ".join(f"{d:.3f}" for d in dec))
    lam_mean = np.exp(full["theta"]["s"] ** 2 / 2)
    print(f"mean intensity E[lambda] = {lam_mean:.2f}; mean variance rate "
          f"c*E[lambda] = {full['theta']['c'] * lam_mean:.4f} (probit units/day "
          f"at tau=1)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--venue", choices=["polymarket", "kalshi", "both"])
    ap.add_argument("--max-obs", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    if args.oracle:
        return run_oracle(args.n, args.seed)
    if not args.venue:
        ap.error("need --oracle or --venue")
    for v in (["polymarket", "kalshi"] if args.venue == "both" else [args.venue]):
        run_venue(v, args.max_obs, args.seed)


if __name__ == "__main__":
    main()
