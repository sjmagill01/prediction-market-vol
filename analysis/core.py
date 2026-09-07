"""v2 shared core: the state space, the panel, and the simulator.

Everything the phase scripts have in common lives here: the (p, tau)
discretisation, the Market container, the daily increment panel, and the
exact-martingale simulator used for every oracle gate. v1 spread these
across five modules with cross-imports; v2 has one owner.

The simulator is the reference object for the whole project. It builds an
EXACT bounded martingale by the latent-Gaussian construction: a hidden
signal z accumulates information increments u_t, the remaining information
R_t = sum_{s>=t} u_s runs to 0 at the deadline, and

    p_t = Phi(z_t / sqrt(R_t)),   final = 1{z_T > 0}

so E[p_{t+1} | F_t] = p_t holds exactly, including the terminal step
(E[1{z_T>0} | z_{T-1}] = Phi(z_{T-1}/sqrt(u_{T-1})) = p_{T-1}). On top of
the truth sit the observation knobs: additive bid-ask noise (creates the
MA(1) bounce signature), staleness (repeats yesterday's print), and
stochastic information flow (log-AR(1) u_t, creates lambda-clustering).
The budget identity E[sum dp^2 + (final - p_T)^2] = p0(1-p0) is a theorem
for the truth; tests/test_core.py enforces it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.config import DATA_V2, SEED

# ---------------------------------------------------------------- state grid
P_BUCKETS = np.array([0.0, 0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85,
                      0.95, 1.0])
TAU_EDGES = np.array([1.0, 3.0, 7.0, 14.0, 30.0, 90.0, np.inf])
P_CLIP = (0.005, 0.995)     # probit-safe interior


def bucket(p: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(P_BUCKETS, p, side="right") - 1,
                   0, len(P_BUCKETS) - 2)


def bucket_label(i: int) -> str:
    return f"[{P_BUCKETS[i]:.2f},{P_BUCKETS[i + 1]:.2f})"


def tau_band(tau: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(TAU_EDGES, tau, side="right") - 1,
                   0, len(TAU_EDGES) - 2)


def tau_label(j: int) -> str:
    hi = TAU_EDGES[j + 1]
    return f"[{TAU_EDGES[j]:.0f},{'inf' if np.isinf(hi) else f'{hi:.0f}'})d"


# ------------------------------------------------------------------- market
@dataclass
class Market:
    """One resolved market: daily closes on a day grid, terminal outcome."""
    key: str
    ts_days: np.ndarray      # observation times in days from listing
    closes: np.ndarray       # prices in [0, 1] at ts_days
    final: float             # resolved outcome, 0 or 1
    t_res: float             # resolution time in days from listing


def panel(mkts: list[Market]) -> pd.DataFrame:
    """Stacked daily increments: one row per consecutive close pair."""
    rows = []
    for m in mkts:
        if len(m.closes) < 2:
            continue
        p = m.closes[:-1]
        dp = np.diff(m.closes)
        dt = np.diff(m.ts_days)
        rows.append(pd.DataFrame({
            "key": m.key, "p": p, "dp": dp, "dp2": dp * dp, "dt": dt,
            "tau": m.t_res - m.ts_days[:-1]}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ loading
MIN_OBS = 5     # minimum daily closes for a market to enter the panel


def load_markets(venue: str) -> list[Market]:
    """All resolved binary markets of the v2 vintage with daily bars.

    Reads data_v2/catalog/{venue}.parquet + data_v2/bars/{venue}/1440m/.
    Outcome must be exactly 0 or 1 (Kalshi scalar settlements excluded).
    Closes are clipped to [0, 1]: the 2026-09-07 full-vintage audit found
    exactly one out-of-range close (a 1.0025 Polymarket settlement-day
    print), so the raw files stay untouched and the clip lives here.
    Resolution time is proxied by the last observed bar (trading halts at
    resolution); the terminal budget term (final - last close)^2 makes the
    identity exact regardless of when bars stop.
    """
    cat = pd.read_parquet(DATA_V2 / "catalog" / f"{venue}.parquet",
                          columns=["key", "closed", "final_price"])
    cat = cat[cat.closed & cat.final_price.isin([0.0, 1.0])]
    final = dict(zip(cat.key.astype(str), cat.final_price))
    out = []
    for f in sorted((DATA_V2 / "bars" / venue / "1440m").glob("*.parquet")):
        key = f.stem
        if key not in final:
            continue
        df = pd.read_parquet(f, columns=["ts", "close"])
        if len(df) < MIN_OBS:
            continue
        ts = pd.to_datetime(df["ts"], utc=True)
        days = (ts - ts.iloc[0]).dt.total_seconds().to_numpy() / 86400.0
        closes = np.clip(df["close"].to_numpy(float), 0.0, 1.0)
        out.append(Market(key=key, ts_days=days, closes=closes,
                          final=float(final[key]), t_res=float(days[-1])))
    return out


# ---------------------------------------------------------------- simulator
def simulate(n: int, noise: float = 0.0, stale: float = 0.0,
             stochvol: bool = False, seed: int = SEED,
             t_min: int = 10, t_max: int = 120,
             sv_phi: float = 0.95, sv_s: float = 0.4) -> list[Market]:
    """n exact-martingale markets with observation-layer knobs.

    noise    : sd of additive observation noise (bid-ask bounce).
    stale    : probability a day repeats the previous print.
    stochvol : information increments u_t = exp(l_t), l log-AR(1)
               (phi=sv_phi, innovation sd=sv_s); else u_t = 1.
    The latent construction is exact regardless of the knobs; the knobs
    corrupt only the observed closes, never the truth or the outcome.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        T = int(rng.integers(t_min, t_max + 1))
        if stochvol:
            l = np.empty(T)
            l[0] = rng.normal(0.0, sv_s / np.sqrt(1 - sv_phi ** 2))
            eps = rng.normal(0.0, sv_s, T - 1)
            for t in range(1, T):
                l[t] = sv_phi * l[t - 1] + eps[t - 1]
            u = np.exp(l)
        else:
            u = np.ones(T)
        # remaining information R_t = sum_{s>=t} u_s; R[T] = 0
        R = np.concatenate([np.cumsum(u[::-1])[::-1], [0.0]])
        # start the signal with its own prior spread so p0 varies
        z = np.empty(T + 1)
        z[0] = rng.normal(0.0, np.sqrt(0.5 * R[0]))
        z[1:] = z[0] + np.cumsum(np.sqrt(u) * rng.standard_normal(T))
        p = norm.cdf(z[:-1] / np.sqrt(R[:-1]))
        final = float(z[T] > 0)
        obs = p.copy()
        if noise > 0:
            obs = obs + rng.normal(0.0, noise, T)
        if stale > 0:
            rep = rng.random(T) < stale
            for t in range(1, T):
                if rep[t]:
                    obs[t] = obs[t - 1]
        obs = np.clip(obs, 1e-4, 1 - 1e-4)
        out.append(Market(key=f"sim{i:05d}", ts_days=np.arange(T, dtype=float),
                          closes=obs, final=final, t_res=float(T)))
    return out
