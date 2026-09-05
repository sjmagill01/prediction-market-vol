r"""Realised variance vs p(1-p): model-free and model-dependent volatility tests.

A prediction-market price p_t is a probability: bounded in [0,1] and absorbed
at 0/1 at resolution. We work in price differences dp (log returns are
meaningless near the boundary) and run two distinct tests plus diagnostics.

(A) Integrated, model-free ("variance budget" / realised-vs-implied).
    If p is a martingale, the total squared movement over a market's life is
    exactly budgeted:  E[ sum dp^2 + (final - last_close)^2 ] = p0(1-p0),
    because the sum telescopes to Var(final | p0) and final is Bernoulli(p0).
    p0(1-p0) is therefore the market's own model-free implied variance (the
    variance-swap strike quoted by the price itself). We regress
    RV_life on p0(1-p0) through the origin. Slope > 1 = excess movement
    (bid-ask bounce, overreaction / non-Bayesian updating); slope < 1 hints at
    risk premia contaminating price-as-probability. Staleness does NOT bias
    this test: a martingale sampled at irregular times keeps the identity.

(B) Local, model-dependent: which variance-rate function of (p, tau) does the
    budget get spent through? With tau = days to resolution, daily dp^2 is
    compared to three candidates (per-day rates, scaled by the actual day gap):
      1. p(1-p)/tau           "binomial" intuition: each day a mini-resolution
      2. phi(Phi^-1(p))^2/tau latent-Brownian beliefs: p = Phi(X_t/sqrt(v_t)),
                              uniform information flow v_t = tau/T
      3. p(1-p)               tau-free (Wright-Fisher-style constant intensity;
                              free scale, so its slope is an estimated
                              intensity and only bucket *flatness* is tested)
    Candidates 1-2 are fully specified (no free parameter): slope ~ 1 means
    the model prices local risk correctly. We also tabulate mean dp^2 by
    p bucket against each candidate; a good model is flat across buckets.
    Expectation: the Gaussian form fits and p(1-p)/tau fails in the tails
    (longshots move far less per day than p(1-p)/tau predicts, unless news
    arrives as jumps).

Diagnostics for a possible third (stochastic-vol) layer. Any admissible
stochastic-vol model here is forced into SLV form
    Var(dp) = phi(Phi^-1(p))^2 * lambda_t dt
(the local factor is pinned by the boundary; Heston-ness lives in lambda_t).
We therefore compute lambda_hat = dp^2 / candidate-2-rate and report:
  - excess kurtosis of the standardised move by p bucket (jumps OR stochvol),
  - pooled ACF of lambda_hat, lags 1-5 (persistence = stochastic vol /ARCH;
    kurtosis without persistence = jumps; negative lag-1 = bid-ask bounce).

--simulate N generates N latent-Brownian martingale markets (lifetimes
20-400 days) and runs the same tests; --noise adds additive microstructure
noise, --stale freezes a day's price with given probability, --stochvol makes
the information-flow intensity a persistent lognormal AR(1) (budget kept).
Clean benchmark: slope (A) ~ 1.0-1.1, Gaussian local ratios ~ 0.9-1.05;
--noise 0.02 pushes slope (A) to ~1.7+; --stale 0.5 leaves it ~1.0.

Usage:
  python analysis/vol_check.py --simulate 600 [--noise 0.02] [--stale 0.5]
                               [--stochvol] [--seed 7] [--plot]
  python analysis/vol_check.py --venue polymarket [--min-volume N] [--plot]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

P_BUCKETS = np.array([0.0, 0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95, 1.0])
P_CLIP = (0.005, 0.995)   # for Phi^-1; also drop test-B obs outside this
MIN_OBS = 5               # minimum daily closes per market


@dataclass
class Market:
    key: str
    ts_days: np.ndarray   # observation times in days (float, increasing)
    closes: np.ndarray    # daily closes in [0,1]
    final: float          # 0.0 or 1.0
    t_res: float          # resolution time in days (same clock as ts_days)


# ------------------------------------------------------------------ loading

def load_markets(venue: str, min_volume: float | None) -> list[Market]:
    from pmdata import store
    c = store.con()
    q = """
        SELECT b.key, b.ts, b.close, c.final_price
        FROM bars b
        JOIN catalog c ON c.key = b.key AND c.venue = b.venue
        WHERE b.venue = ? AND b.freq = '1440m'
          AND c.closed AND c.final_price IN (0.0, 1.0)
    """
    params = [venue]
    if min_volume is not None:
        q += " AND c.volume >= ?"
        params.append(min_volume)
    q += " ORDER BY b.key, b.ts"
    df = c.execute(q, params).fetchdf()
    if df.empty:
        return []
    out = []
    for key, g in df.groupby("key", sort=False):
        closes = g["close"].to_numpy(float)
        ts = pd.to_datetime(g["ts"], utc=True)
        days = (ts - ts.iloc[0]).dt.total_seconds().to_numpy() / 86400.0
        if len(closes) < MIN_OBS:
            continue
        if np.nanmin(closes) < -1e-9 or np.nanmax(closes) > 1 + 1e-9:
            continue  # defensive: prices must be probabilities
        # resolution proxy: last observed bar (trading halts at resolution)
        out.append(Market(str(key), days, np.clip(closes, 0.0, 1.0),
                          float(g["final_price"].iloc[0]), days[-1]))
    return out


# ------------------------------------------------------------------ simulation

def simulate(n: int, noise: float = 0.0, stale: float = 0.0,
             stochvol: bool = False, seed: int = 7) -> list[Market]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        L = int(rng.integers(20, 401))
        # information-arrival weights (sum to 1 = total signal variance)
        if stochvol:
            e = rng.normal(0, 0.5, L)
            lg = np.empty(L)
            lg[0] = e[0]
            for t in range(1, L):
                lg[t] = 0.95 * lg[t - 1] + e[t]
            w = np.exp(lg)
            w = w / w.sum()
        else:
            w = np.full(L, 1.0 / L)
        p0 = rng.uniform(0.02, 0.98)
        x = norm.ppf(p0)                       # latent state, total var 1
        v = 1.0                                # remaining signal variance
        p_path = [p0]
        for t in range(L):
            x += rng.normal(0.0, np.sqrt(w[t]))
            v -= w[t]
            if v > 1e-12:
                p_path.append(norm.cdf(x / np.sqrt(v)))
        p = np.asarray(p_path)
        final = 1.0 if x > 0 else 0.0
        # microstructure noise on observed prices (not on the outcome)
        if noise > 0:
            q = np.clip(p + rng.normal(0, noise, len(p)), 0.001, 0.999)
        else:
            q = p.copy()
        q[0] = p[0]
        # staleness: a day's print repeats the previous one
        if stale > 0:
            for t in range(1, len(q)):
                if rng.uniform() < stale:
                    q[t] = q[t - 1]
        ts = np.arange(len(q), dtype=float)
        out.append(Market(f"sim{i}", ts, q, final, float(L)))
    return out


# ------------------------------------------------------------------ helpers

def _origin_reg(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Through-origin OLS slope with heteroskedasticity-robust s.e."""
    sxx = float(np.sum(x * x))
    if sxx <= 0:
        return np.nan, np.nan
    b = float(np.sum(x * y)) / sxx
    e = y - b * x
    se = float(np.sqrt(np.sum((x * e) ** 2))) / sxx
    return b, se


def _bucket(p: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(P_BUCKETS, p, side="right") - 1,
                   0, len(P_BUCKETS) - 2)


def _bucket_label(i: int) -> str:
    return f"[{P_BUCKETS[i]:.2f},{P_BUCKETS[i+1]:.2f})"


# ------------------------------------------------------------------ test A

def test_a(mkts: list[Market]) -> pd.DataFrame:
    rows = []
    for m in mkts:
        dp = np.diff(m.closes)
        rv = float(np.sum(dp * dp) + (m.final - m.closes[-1]) ** 2)
        p0 = m.closes[0]
        rows.append({"key": m.key, "p0": p0, "iv": p0 * (1 - p0), "rv": rv})
    df = pd.DataFrame(rows)
    slope, se = _origin_reg(df["iv"].to_numpy(), df["rv"].to_numpy())
    df.attrs["slope"], df.attrs["se"] = slope, se
    return df


def test_a_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = _bucket(df["p0"].to_numpy())
    g = df.groupby("bucket").agg(n=("rv", "size"), sum_rv=("rv", "sum"),
                                 sum_iv=("iv", "sum"))
    g["ratio"] = g["sum_rv"] / g["sum_iv"]
    g.index = [_bucket_label(i) for i in g.index]
    return g


# ------------------------------------------------------------------ test B

def daily_panel(mkts: list[Market]) -> pd.DataFrame:
    """One row per daily move: p_prev, dp2, tau (days to resolution), dt."""
    frames = []
    for m in mkts:
        p = m.closes
        t = m.ts_days
        dp = np.diff(p)
        p_prev = p[:-1]
        dt = np.diff(t)
        tau = np.maximum(m.t_res - t[:-1], 1.0)
        keep = (p_prev > P_CLIP[0]) & (p_prev < P_CLIP[1]) & (dt > 0)
        if not keep.any():
            continue
        frames.append(pd.DataFrame({
            "key": m.key, "p": p_prev[keep], "dp2": dp[keep] ** 2,
            "dp": dp[keep], "tau": tau[keep], "dt": dt[keep],
        }))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def candidate_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-observation expected dp^2 under each candidate (scaled by day gap)."""
    df = df.copy()
    p, tau, dt = df["p"].to_numpy(), df["tau"].to_numpy(), df["dt"].to_numpy()
    z = norm.ppf(np.clip(p, *P_CLIP))
    df["v_binom"] = p * (1 - p) / tau * dt
    df["v_gauss"] = norm.pdf(z) ** 2 / tau * dt
    df["v_wf"] = p * (1 - p) * dt  # free scale
    return df


def test_b(df: pd.DataFrame) -> dict:
    res = {}
    y = df["dp2"].to_numpy()
    for name in ("v_binom", "v_gauss", "v_wf"):
        b, se = _origin_reg(df[name].to_numpy(), y)
        res[name] = {"slope": b, "se": se}
    return res


def test_b_buckets(df: pd.DataFrame, slopes: dict) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = _bucket(df["p"].to_numpy())
    rows = []
    for i, g in df.groupby("bucket"):
        row = {"bucket": _bucket_label(int(i)), "n": len(g),
               "mean_dp2": g["dp2"].mean()}
        for name in ("v_binom", "v_gauss", "v_wf"):
            denom = g[name].mean()
            # v_wf has a free scale: normalise by its fitted slope so that
            # flatness across buckets is the comparable criterion
            scale = slopes[name]["slope"] if name == "v_wf" else 1.0
            row[f"ratio_{name[2:]}"] = g["dp2"].mean() / (denom * scale) \
                if denom > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ diagnostics

def kurtosis_table(df: pd.DataFrame) -> pd.DataFrame:
    """Excess kurtosis of the Gaussian-standardised move z = dp/sqrt(v_gauss)."""
    df = df.copy()
    df["z"] = df["dp"] / np.sqrt(df["v_gauss"].clip(lower=1e-12))
    df["bucket"] = _bucket(df["p"].to_numpy())
    rows = []
    for i, g in df.groupby("bucket"):
        z = g["z"].to_numpy()
        z = z[np.isfinite(z)]
        if len(z) < 30:
            continue
        m2 = np.mean((z - z.mean()) ** 2)
        m4 = np.mean((z - z.mean()) ** 4)
        rows.append({"bucket": _bucket_label(int(i)), "n": len(z),
                     "excess_kurtosis": m4 / m2 ** 2 - 3 if m2 > 0 else np.nan})
    return pd.DataFrame(rows)


def _pooled_acf(d: pd.DataFrame, col: str, max_lag: int) -> list[float]:
    d = d[["key", col]].copy()
    d[col] = d[col] - d.groupby("key")[col].transform("mean")
    denom = float(np.sum(d[col] ** 2))
    out = []
    for lag in range(1, max_lag + 1):
        num = 0.0
        for _, g in d.groupby("key", sort=False):
            v = g[col].to_numpy()
            if len(v) > lag:
                num += float(np.sum(v[lag:] * v[:-lag]))
        out.append(num / denom if denom > 0 else np.nan)
    return out


def lambda_acf(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """Pooled within-market ACFs of the Gaussian-standardised move.

    z   = dp / sqrt(v_gauss)   signed: negative lag-1 = bid-ask bounce /
                               microstructure reversal (MA(1) in dp)
    lam = z^2                  squares: persistence = stochvol / ARCH.
    NB additive noise makes lam ACF *positive* too (squares of an MA(1)
    cluster), so read lam persistence together with the signed row: bounce =
    negative z ACF + positive lam ACF; true stochvol = ~0 z ACF + positive
    lam ACF.
    """
    d = pd.DataFrame({"key": df["key"],
                      "z": df["dp"] / np.sqrt(df["v_gauss"].clip(lower=1e-12))})
    d["lam"] = d["z"] ** 2
    # winsorise: a single explosive ratio (tiny denominator) would own the ACF
    d["lam"] = d["lam"].clip(upper=d["lam"].quantile(0.99))
    d["z"] = d["z"].clip(lower=d["z"].quantile(0.005), upper=d["z"].quantile(0.995))
    return pd.DataFrame({
        "lag": range(1, max_lag + 1),
        "acf_z_signed": _pooled_acf(d, "z", max_lag),
        "acf_lambda": _pooled_acf(d, "lam", max_lag),
    })


# ------------------------------------------------------------------ driver

def run(mkts: list[Market], label: str, plot: bool = False) -> dict:
    print(f"\n================ {label}: {len(mkts)} resolved markets ================")
    if not mkts:
        print("no markets -- nothing to do")
        return {}
    a = test_a(mkts)
    slope, se = a.attrs["slope"], a.attrs["se"]
    print(f"\n--- Test A (integrated, model-free): RV_life vs p0(1-p0), through origin")
    print(f"slope = {slope:.3f}  (robust se {se:.3f})   [martingale => 1; >1 = excess movement]")
    print("\nratio of sums (RV/IV) by p0 bucket:")
    print(test_a_buckets(a).to_string())

    panel = candidate_rates(daily_panel(mkts))
    b = test_b(panel)
    print(f"\n--- Test B (local): daily dp^2 vs candidate variance rates, through origin")
    for name, lab in (("v_binom", "p(1-p)/tau        (binomial)"),
                      ("v_gauss", "phi(Phi^-1(p))^2/tau (Gaussian)"),
                      ("v_wf", "p(1-p)            (tau-free, free scale)")):
        r = b[name]
        print(f"  {lab:44s} slope = {r['slope']:.3f} (se {r['se']:.3f})")
    print("\nmean dp^2 / mean candidate by p bucket (v_wf normalised by its slope; flat = good):")
    print(test_b_buckets(panel, b).to_string(index=False,
                                             float_format=lambda v: f"{v:.3f}"))

    print("\n--- Diagnostics for a stochastic-vol third layer")
    print("excess kurtosis of dp/sqrt(v_gauss) by p bucket (jumps OR stochvol):")
    print(kurtosis_table(panel).to_string(index=False,
                                          float_format=lambda v: f"{v:.2f}"))
    print("\npooled ACFs of standardised move z=dp/sqrt(v_gauss) and lambda=z^2:")
    print("(neg z lag-1 = bounce/reversal; lam persistence w/ ~0 z = stochvol;")
    print(" fat tails w/ ~0 both = jumps)")
    print(lambda_acf(panel).to_string(index=False,
                                      float_format=lambda v: f"{v:.3f}"))

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(a["iv"], a["rv"], s=8, alpha=0.4)
        lim = max(0.26, float(a["rv"].quantile(0.99)))
        ax.plot([0, 0.25], [0, 0.25], "r-", lw=1, label="45$^\\circ$ (martingale)")
        ax.plot([0, 0.25], [0, 0.25 * slope], "b--", lw=1,
                label=f"fit slope {slope:.2f}")
        ax.set_xlim(0, 0.26)
        ax.set_ylim(0, lim)
        ax.set_xlabel("$p_0(1-p_0)$  (model-free implied variance)")
        ax.set_ylabel("realised lifetime variance")
        ax.set_title(f"{label}: realised vs implied variance")
        ax.legend()
        out = Path(__file__).parent / f"rv_vs_iv_{label.replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\nplot saved: {out}")

    return {"a_slope": slope, "a_se": se, "b": b,
            "b_buckets": test_b_buckets(panel, b),
            "kurt": kurtosis_table(panel), "acf": lambda_acf(panel)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=["polymarket", "kalshi"])
    ap.add_argument("--min-volume", type=float)
    ap.add_argument("--simulate", type=int)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--stale", type=float, default=0.0)
    ap.add_argument("--stochvol", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)
    if args.simulate:
        tags = []
        if args.noise:
            tags.append(f"noise{args.noise}")
        if args.stale:
            tags.append(f"stale{args.stale}")
        if args.stochvol:
            tags.append("stochvol")
        label = "sim " + (" ".join(tags) if tags else "clean")
        mkts = simulate(args.simulate, args.noise, args.stale,
                        args.stochvol, args.seed)
        return run(mkts, label, args.plot)
    if not args.venue:
        ap.error("need --venue or --simulate")
    mkts = load_markets(args.venue, args.min_volume)
    return run(mkts, args.venue, args.plot)


if __name__ == "__main__":
    main()
