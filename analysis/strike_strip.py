"""Strike-strip implied volatility from Kalshi bracketed markets.

Bracketed events are digital-option strips on a common underlying: at one
instant, the prices of the member markets are risk-neutral probabilities of
disjoint bins ('between' floor/cap) plus tail digitals ('greater'/'less').
That is a full risk-neutral distribution, from which a genuinely
forward-looking implied vol follows (construction 3 in THEORY.md sec 4) --
unlike lambda_imp, nothing here is computed from realised movement.

Per event-day:
  bins  : 'between' markets -> [floor, cap) with prob = price; extreme
          'greater'/'less' markets -> open tails, width = median bin width.
          Ladder events (all-'greater' chains) are differenced into bins.
  checks: total probability in [0.7, 1.3] (else the strip is stale or
          incomplete and the day is dropped); probs renormalised to 1.
  moments: implied mean and std of the underlying.
  IV    : tau = days to event close.  For level underlyings (index, oil,
          gas) sigma_prop = (std/mean)/sqrt(tau_yr), a Black-Scholes-style
          annualised proportional vol.  For rate-like underlyings (CPI %,
          yields, unemployment) sigma_norm = std/sqrt(tau_yr) in absolute
          points per sqrt-year (normal vol, the swaption convention).

Built-in validation: S&P/Nasdaq annual strips should land near equity IV
(~10-25%; observed 17%/13%), WTI near oil vol (observed 46%, rich vs ~35%
OVX but the right order of magnitude), 10Y Treasury near swaption normal
vol (~100bp/sqrt-yr; observed 100bp). Numbers far outside those ranges
would indicate a broken construction.

Excluded: running-extremum series (BTC max/min etc.) -- a max over a period
is not a terminal value, so its distribution has no IV interpretation.

Usage:
  python analysis/strike_strip.py [--series KXINXY] [--plot]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# terminal-value series whitelist: ticker -> ("prop"|"norm", description)
SERIES = {
    "KXINXY":       ("prop", "S&P 500 end-of-year level"),
    "KXNASDAQ100Y": ("prop", "Nasdaq-100 end-of-year level"),
    "KXWTIW":       ("prop", "WTI crude Friday close"),
    "KXAAAGASM":    ("prop", "AAA gas price, month-end"),
    "KXAAAGASW":    ("prop", "AAA gas price, week-end"),
    "KXCPIYOY":     ("norm", "CPI YoY %"),
    "KXCPI":        ("norm", "CPI MoM %"),
    "KXTNOTEW":     ("norm", "10Y Treasury yield, Friday"),
    "KXPAYROLLS":   ("norm", "Nonfarm payrolls (jobs, monthly change)"),
    "KXU3":         ("norm", "Unemployment rate %"),
}
MIN_STRIKES = 5
PROB_TOTAL_RANGE = (0.7, 1.3)
MIN_TAU_DAYS = 2.0


def load_strips() -> pd.DataFrame:
    """Daily closes for all markets in whitelisted numeric-strike events."""
    from pmdata import store
    c = store.con()
    cat = c.execute("SELECT * FROM catalog WHERE venue='kalshi'").fetchdf()
    cat = cat[cat["series_ticker"].isin(SERIES)]
    cat["fs"] = pd.to_numeric(cat["floor_strike"], errors="coerce")
    cat["cs"] = pd.to_numeric(cat["cap_strike"], errors="coerce")
    cat = cat[cat["fs"].notna() | cat["cs"].notna()]
    sizes = cat.groupby("event_ticker")["key"].transform("size")
    cat = cat[sizes >= MIN_STRIKES]
    bars = c.execute(
        "SELECT key, ts, close FROM bars WHERE venue='kalshi' AND freq='1440m'"
    ).fetchdf()
    df = bars.merge(cat[["key", "event_ticker", "series_ticker", "strike_type",
                         "fs", "cs", "end"]], on="key")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True, format="mixed")
    return df


def _bins(g: pd.DataFrame) -> pd.DataFrame | None:
    """Turn one event-day of strike markets into disjoint (lo, hi, prob) bins."""
    bet = g[g["strike_type"] == "between"]
    if len(bet) >= 3:
        bins = bet[["fs", "cs", "close"]].rename(
            columns={"fs": "lo", "cs": "hi", "close": "prob"})
        w = float(np.median(bins["hi"] - bins["lo"]))
        gt = g[g["strike_type"].isin(["greater", "greater_or_equal"])]
        lt = g[g["strike_type"].isin(["less", "less_or_equal"])]
        rows = [bins]
        if len(gt):  # highest 'greater' strike = upper tail
            r = gt.loc[gt["fs"].idxmax()]
            rows.append(pd.DataFrame({"lo": [r["fs"]], "hi": [r["fs"] + w],
                                      "prob": [r["close"]]}))
        if len(lt):
            r = lt.loc[lt["cs"].idxmin()]
            rows.append(pd.DataFrame({"lo": [r["cs"] - w], "hi": [r["cs"]],
                                      "prob": [r["close"]]}))
        return pd.concat(rows, ignore_index=True)
    # ladder mode: a chain of 'greater' digitals is a survival function
    gt = g[g["strike_type"].isin(["greater", "greater_or_equal"])]
    gt = gt.dropna(subset=["fs"]).sort_values("fs")
    if len(gt) < MIN_STRIKES:
        return None
    k = gt["fs"].to_numpy()
    s = gt["close"].to_numpy()          # P(S > k), should be decreasing
    s = np.minimum.accumulate(s)        # enforce monotone survival
    w = float(np.median(np.diff(k)))
    lo = np.concatenate([[k[0] - w], k])
    hi = np.concatenate([k, [k[-1] + w]])
    prob = np.concatenate([[1 - s[0]], -np.diff(s), [s[-1]]])
    return pd.DataFrame({"lo": lo, "hi": hi, "prob": prob})


def event_day_iv(g: pd.DataFrame) -> dict | None:
    bins = _bins(g)
    if bins is None:
        return None
    total = float(bins["prob"].sum())
    if not (PROB_TOTAL_RANGE[0] <= total <= PROB_TOTAL_RANGE[1]):
        return None
    p = bins["prob"].to_numpy() / total
    mid = ((bins["lo"] + bins["hi"]) / 2).to_numpy()
    mean = float(np.sum(p * mid))
    var = float(np.sum(p * (mid - mean) ** 2))
    if var <= 0:
        return None
    tau_d = (g["end"].iloc[0] - g["ts"].iloc[0]).total_seconds() / 86400.0
    if tau_d < MIN_TAU_DAYS:
        return None
    tau_y = tau_d / 365.25
    std = np.sqrt(var)
    return {"mean": mean, "std": std, "tau_days": tau_d, "total_prob": total,
            "n_strikes": len(g),
            "iv_prop": std / abs(mean) / np.sqrt(tau_y) if mean != 0 else np.nan,
            "iv_norm": std / np.sqrt(tau_y)}


def build_iv_panel(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ev, ts), g in df.groupby(["event_ticker", "ts"]):
        r = event_day_iv(g)
        if r:
            r.update({"event": ev, "ts": ts,
                      "series": g["series_ticker"].iloc[0]})
            rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", help="restrict to one series ticker")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    df = load_strips()
    if args.series:
        df = df[df["series_ticker"] == args.series]
    panel = build_iv_panel(df)
    if panel.empty:
        print("no valid strip-days")
        return panel
    print(f"\n{len(panel)} valid event-days across "
          f"{panel['event'].nunique()} events, {panel['series'].nunique()} series\n")

    kind = {s: SERIES[s][0] for s in SERIES}
    panel["iv"] = np.where(panel["series"].map(kind) == "prop",
                           panel["iv_prop"], panel["iv_norm"])
    summ = panel.groupby("series").agg(
        kind=("series", lambda s: kind[s.iloc[0]]),
        events=("event", "nunique"), days=("ts", "size"),
        med_iv=("iv", "median"), q25=("iv", lambda s: s.quantile(0.25)),
        q75=("iv", lambda s: s.quantile(0.75)),
        med_tau=("tau_days", "median"),
        med_total_prob=("total_prob", "median"))
    summ["desc"] = [SERIES[s][1] for s in summ.index]
    print("annualised implied vol by series")
    print("(kind=prop: (std/mean)/sqrt(tau_yr), BS-style; kind=norm: std/sqrt(tau_yr),")
    print(" absolute points per sqrt-year, swaption convention)")
    print(summ.to_string(float_format=lambda v: f"{v:.3f}"))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = panel[panel["series"].isin(["KXINXY", "KXWTIW", "KXCPIYOY"])]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, s in zip(axes, ["KXINXY", "KXWTIW", "KXCPIYOY"]):
            sub = show[show["series"] == s]
            for ev, g in sub.groupby("event"):
                g = g.sort_values("tau_days")
                ax.plot(g["tau_days"], g["iv"], lw=0.8, alpha=0.6)
            ax.set_title(f"{s} ({SERIES[s][0]})")
            ax.set_xlabel("days to resolution")
            ax.set_ylabel("annualised IV")
            ax.invert_xaxis()
        fig.tight_layout()
        out = Path(__file__).parent / "strike_strip_iv.png"
        fig.savefig(out, dpi=120)
        print(f"\nplot saved: {out}")
    return panel


if __name__ == "__main__":
    main()
