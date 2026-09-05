"""Bounce-vs-jump decomposition of hourly prediction-market variance.

The daily tests (vol_check.py) find excess movement on both venues but cannot
say how much is microstructure (bid-ask bounce) versus genuine jumps. Hourly
bars can, via three independent instruments, each pooled by p bucket:

1. Trade vs quote RV. Kalshi hourly bars carry bid/ask closes. Bounce lives
   in trade prints, not in the quote midpoint, so
       bounce_share_quote = 1 - sum(dm^2) / sum(dc^2)
   (c = trade close, m = midpoint) is a direct, model-free bounce estimate.

2. Roll autocovariance. Under observed = efficient + iid noise u,
   dp_t = de_t + u_t - u_{t-1} is MA(1) with gamma1 = Cov(dp_t, dp_{t+1})
   = -Var(u); the bounce contribution to RV is 2 Var(u) per increment, so
       bounce_share_roll = -2 * gamma1_pooled / mean(dp^2).

3. Bipower variation on midpoints. BV = (pi/2) * mean(|dm_t||dm_{t+1}|)
   converges to the continuous (diffusive) variance only; RV - BV is jump
   variance. Computed on midpoints so bounce does not inflate it:
       jump_share = 1 - (pi/2) * sum(|dm_t||dm_{t+1}|) / sum(dm^2).

Also reported: the signature ratio RV(1h)/RV(1d) per market over the common
span (> 1 = variance inflates at higher sampling frequency = bounce).

Caveats: the 1-cent grid makes hourly dp mostly 0 or +/-1 tick in quiet
markets, which *creates* negative autocovariance (tick rounding is itself a
noise process); pairs are restricted to consecutive hours (gap = 1h) so
overnight closures never enter the autocovariance or BV products. Polymarket
hourly bars have no bid/ask (single price series), so only instruments 2-3
apply there, and only 24 markets have hourly history (CLOB drops intraday
history for old closed markets).

Usage:
  python analysis/intraday_decomp.py --venue kalshi [--min-volume N]
  python analysis/intraday_decomp.py --venue polymarket
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.vol_check import P_BUCKETS, _bucket, _bucket_label  # noqa: E402

MIN_HOURS = 48          # minimum hourly closes per market
HOUR = pd.Timedelta(hours=1)


def load_hourly(venue: str, min_volume: float | None) -> pd.DataFrame:
    from pmdata import store
    c = store.con()
    q = """
        SELECT b.key, b.ts, b.close, b.bid, b.ask
        FROM bars b
        JOIN catalog c ON c.key = b.key AND c.venue = b.venue
        WHERE b.venue = ? AND b.freq = '60m'
          AND c.closed AND c.final_price IN (0.0, 1.0)
    """
    params: list = [venue]
    if min_volume is not None:
        q += " AND c.volume >= ?"
        params.append(min_volume)
    q += " ORDER BY b.key, b.ts"
    return c.execute(q, params).fetchdf()


def load_daily_rv(venue: str, keys: pd.Series) -> pd.Series:
    """Per-market realised variance from 1d bars (for the signature ratio)."""
    from pmdata import store
    c = store.con()
    df = c.execute(
        "SELECT key, ts, close FROM bars WHERE venue = ? AND freq = '1440m' "
        "ORDER BY key, ts", [venue]).fetchdf()
    df = df[df["key"].isin(set(keys))]
    return df.groupby("key")["close"].apply(lambda s: float(np.sum(np.diff(s) ** 2)))


def pair_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-hour increment pairs: dp, dp_next, dm, dm_next, p bucket."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    g = df.groupby("key", sort=False)
    df["mid"] = ((df["bid"] + df["ask"]) / 2.0).fillna(df["close"])
    df["dt"] = g["ts"].diff()
    df["dp"] = g["close"].diff()
    df["dm"] = g["mid"].diff()
    df["p_prev"] = g["close"].shift(1)
    # increment is valid only when it spans exactly one hour
    df["ok"] = df["dt"] == HOUR
    df["dp_next"] = g["dp"].shift(-1)
    df["dm_next"] = g["dm"].shift(-1)
    df["ok_next"] = g["ok"].shift(-1, fill_value=False)
    df = df[df["ok"]]
    df["bucket"] = _bucket(df["p_prev"].to_numpy())
    return df


def decompose(pairs: pd.DataFrame, has_quotes: bool = True) -> pd.DataFrame:
    rows = []
    groups = [("all", pairs)] + [(_bucket_label(int(i)), g)
                                 for i, g in pairs.groupby("bucket")]
    for label, g in groups:
        dp, dm = g["dp"].to_numpy(), g["dm"].to_numpy()
        n = len(g)
        rv_trade = float(np.sum(dp ** 2))
        rv_mid = float(np.nansum(dm ** 2))
        both = g["ok_next"].to_numpy()
        d1, d2 = dp[both], g["dp_next"].to_numpy()[both]
        m1, m2 = dm[both], g["dm_next"].to_numpy()[both]
        gamma1 = float(np.mean(d1 * d2)) if both.any() else np.nan
        var_dp = float(np.mean(dp ** 2))
        bv = float(np.pi / 2 * np.nansum(np.abs(m1) * np.abs(m2)))
        rv_mid_b = float(np.nansum(m1 ** 2))
        rows.append({
            "bucket": label, "n": n,
            "bounce_quote": 1 - rv_mid / rv_trade
                            if has_quotes and rv_trade > 0 else np.nan,
            "bounce_roll": -2 * gamma1 / var_dp if var_dp > 0 else np.nan,
            "jump_share_mid": 1 - bv / rv_mid_b if rv_mid_b > 0 else np.nan,
            "frac_zero_dp": float(np.mean(dp == 0)),
        })
    return pd.DataFrame(rows)


def signature_ratio(df: pd.DataFrame, venue: str) -> pd.Series:
    """Per-market RV(1h)/RV(1d) over each market's hourly span."""
    hr = df.groupby("key").agg(
        rv1h=("dp", lambda s: float(np.sum(s.dropna() ** 2))),
        t0=("ts", "min"), t1=("ts", "max"))
    from pmdata import store
    c = store.con()
    daily = c.execute(
        "SELECT key, ts, close FROM bars WHERE venue = ? AND freq = '1440m' "
        "ORDER BY key, ts", [venue]).fetchdf()
    daily = daily[daily["key"].isin(hr.index)]
    daily["ts"] = pd.to_datetime(daily["ts"], utc=True)
    out = {}
    for key, g in daily.groupby("key"):
        span = hr.loc[key]
        g = g[(g["ts"] >= span["t0"] - HOUR) & (g["ts"] <= span["t1"] + HOUR)]
        rv1d = float(np.sum(np.diff(g["close"]) ** 2))
        if rv1d > 0:
            out[key] = hr.loc[key, "rv1h"] / rv1d
    return pd.Series(out, name="rv1h_over_rv1d")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True, choices=["polymarket", "kalshi"])
    ap.add_argument("--min-volume", type=float)
    args = ap.parse_args(argv)

    raw = load_hourly(args.venue, args.min_volume)
    counts = raw.groupby("key")["close"].size()
    keep = counts[counts >= MIN_HOURS].index
    raw = raw[raw["key"].isin(keep)]
    print(f"\n========== {args.venue}: {raw['key'].nunique()} markets, "
          f"{len(raw)} hourly bars ==========")

    pairs = pair_panel(raw)
    print(f"{len(pairs)} consecutive-hour increments\n")
    tab = decompose(pairs, has_quotes=bool(raw["bid"].notna().any()))
    print("variance decomposition by p bucket (shares of hourly RV):")
    print("(bounce_quote = 1 - RV_mid/RV_trade; bounce_roll = -2*gamma1/var;")
    print(" jump_share_mid = 1 - BV/RV on midpoints; shares can overlap)")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    sig = signature_ratio(pairs, args.venue)
    q = sig.quantile([0.25, 0.5, 0.75])
    print(f"\nsignature ratio RV(1h)/RV(1d), {len(sig)} markets:")
    print(f"  median {q[0.5]:.2f}   IQR [{q[0.25]:.2f}, {q[0.75]:.2f}]"
          f"   (>1 = variance inflates at higher frequency = bounce)")
    return tab, sig


if __name__ == "__main__":
    main()
