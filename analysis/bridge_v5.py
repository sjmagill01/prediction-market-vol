"""V5: cross-venue bridge on matched Polymarket/Kalshi pairs.

Match table (82 pairs, user-signed-off 2026-09-07, approx tiers kept and
flagged) is rebuilt deterministically from the frozen catalogs:

  - fed_decision: meeting x bucket. Exact for H0/C25/C26; PM's hike bucket
    is "25+ bps" (= Kalshi H25 u H26) so H25 matches are flagged approx.
  - fed_chair_nom: candidate name, exact.
  - btc_monthly: month x strike vs KXBTCMAXMON, exact.
  - btc_yearly: strike vs KXBTCMAXY-25-DEC31; approx (listing dates differ,
    both are running-max markets). KXBTCMAX150-style "hit by date" series
    excluded: cumulative payoff, different contract.
  - election: PRES-2024 + POPVOTE-24 exact; NYC mayor approx (Kalshi prices
    the party of the winner, PM the candidate; equal conditional on the
    nomination).

Analyses (daily bars; hourly lead-lag is best-effort on 60m bars if
pull_v5_hourly.py has been run):
  1. gap: g = p_pm - p_k on aligned days; level stats, tail, AR(1)
     persistence (within-pair demeaned) and implied half-life.
  2. paired budget: per pair per venue, sum dp^2 over the common window vs
     p0(1-p0) at the common start; venue ratio-of-ratios on identical events.
  3. paired power: pooled log E[dp^2] vs log p(1-p) slope per venue on the
     matched panel only.
  4. lead-lag: pooled cross-correlation of daily changes at lags -3..+3 and
     the two cross-venue predictive regressions; same at 60m if available.

Outputs: results_v2/bridge_pairs.csv, bridge_pair_metrics.csv,
bridge_v5.json. Reads only data_v2/ + analysis.config.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

from analysis.config import DATA_V2, RESULTS

BARS = DATA_V2 / "bars"

MONTHS = {"january": "JAN", "february": "FEB", "march": "MAR", "april": "APR",
          "may": "MAY", "june": "JUN", "july": "JUL", "august": "AUG",
          "september": "SEP", "october": "OCT", "november": "NOV",
          "december": "DEC"}

# Hand-matched election pairs: (unit, kalshi_key, pm_question, quality)
ELECTION_PAIRS = [
    ("PRES24-Harris", "PRES-2024-KH",
     "Will Kamala Harris win the 2024 US Presidential Election?", "exact"),
    ("PRES24-Trump", "PRES-2024-DJT",
     "Will Donald Trump win the 2024 US Presidential Election?", "exact"),
    ("POPVOTE24-D", "POPVOTE-24-D",
     "Kamala Harris wins the popular vote?", "exact"),
    ("POPVOTE24-R", "POPVOTE-24-R",
     "Will Donald Trump win the popular vote in the 2024 Presidential "
     "Election?", "exact"),
    ("NYC25-Mamdani/D", "KXMAYORNYCPARTY-25-D",
     "Will Zohran Mamdani win the 2025 NYC mayoral election?", "approx"),
    ("NYC25-Cuomo", "KXMAYORNYCPARTY-25-AC",
     "Will Andrew Cuomo win the 2025 NYC mayoral election?", "approx"),
    ("NYC25-Sliwa/R", "KXMAYORNYCPARTY-25-R",
     "Will Curtis Sliwa win the 2025 NYC mayoral election?", "approx"),
]


def _bar_keys(venue: str, freq: str = "1440m") -> set[str]:
    return {f.stem for f in (BARS / venue / freq).glob("*.parquet")}


def _catalogs() -> tuple[pd.DataFrame, pd.DataFrame]:
    k = pd.read_parquet(DATA_V2 / "catalog" / "kalshi.parquet")
    p = pd.read_parquet(DATA_V2 / "catalog" / "polymarket.parquet")
    k = k[k.key.isin(_bar_keys("kalshi"))].reset_index(drop=True)
    p = p[p.key.isin(_bar_keys("polymarket"))].reset_index(drop=True)
    return k, p


def _pm_fed_bucket(q: str) -> str | None:
    s = q.lower()
    if "no change" in s:
        return "H0"
    if "decrease" in s and "25 bps" in s and "25+" not in s:
        return "C25"
    if "decrease" in s and "50+" in s:
        return "C26"
    if "increase" in s and "25+" in s:
        return "H25PLUS"
    return None


def _pm_fed_meeting(row) -> tuple[str | None, str | None]:
    s = (row.question or "").lower() + " " + (row.event_slug or "")
    m = re.search(r"(january|february|march|april|may|june|july|august|"
                  r"september|october|november|december)\s*(20\d\d)?", s)
    if not m:
        return None, None
    yr = m.group(2)
    if yr is None:
        e = pd.to_datetime(row["end"], utc=True, format="ISO8601",
                           errors="coerce")
        yr = str(e.year) if pd.notna(e) else None
    return (yr[-2:] if yr else None), MONTHS[m.group(1)]


def build_pairs() -> pd.DataFrame:
    """Rebuild the signed-off 82-pair match table from the frozen catalogs."""
    k, p = _catalogs()
    rows: list[tuple] = []

    # fed_decision
    fk = k[k.series_ticker == "KXFEDDECISION"].copy()
    m = fk.key.str.extract(r"KXFEDDECISION-(\d{2})([A-Z]{3})-([A-Z]\d+)")
    fk["yy"], fk["mon"], fk["bucket"] = m[0], m[1], m[2]
    fk = fk.set_index(["yy", "mon", "bucket"]).sort_index()
    fp = p[p.event_slug.str.contains("fed-decision|fed-interest-rates",
                                     na=False)].copy()
    fp["bucket"] = fp.question.map(_pm_fed_bucket)
    fp[["yy", "mon"]] = fp.apply(
        lambda r: pd.Series(_pm_fed_meeting(r)), axis=1)
    fp = fp.dropna(subset=["bucket", "yy"])
    for _, r in fp.iterrows():
        bkt = "H25" if r.bucket == "H25PLUS" else r.bucket
        key = (r.yy, r.mon, bkt)
        if key in fk.index:
            kr = fk.loc[key]
            kr = kr.iloc[0] if isinstance(kr, pd.DataFrame) else kr
            rows.append(("fed_decision", f"{r.mon}{r.yy}-{r.bucket}",
                         kr.key, r.key, kr.volume, r.volume,
                         "approx" if r.bucket == "H25PLUS" else "exact"))

    # fed_chair_nom
    ck = k[k.series_ticker == "KXFEDCHAIRNOM"].copy()
    cp = p[p.event_slug == "who-will-trump-nominate-as-fed-chair"].copy()
    ck["name"] = ck.question.str.extract(r"nominate (.+?) as Fed Chair")[0]
    cp["name"] = cp.question.str.extract(
        r"nominate (.+?) as the next Fed chair")[0]
    mm = cp.dropna(subset=["name"]).merge(
        ck.dropna(subset=["name"]), on="name", suffixes=("_pm", "_k"))
    for _, r in mm.iterrows():
        rows.append(("fed_chair_nom", r["name"], r.key_k, r.key_pm,
                     r.volume_k, r.volume_pm, "exact"))

    # btc_monthly
    pat = r"Will Bitcoin reach \$([\d,]+)(k?) in (\w+)\?"
    bp = p[p.question.str.match(pat, na=False)].copy()
    ext = bp.question.str.extract(pat)
    bp["strike"] = (ext[0].str.replace(",", "").astype(float)
                    * ext[1].map({"k": 1000.0, "": 1.0}))
    bp["mon"] = ext[2].str.lower().map(MONTHS)
    bp["yy"] = pd.to_datetime(
        bp["end"], utc=True, format="ISO8601", errors="coerce"
    ).dt.year.astype("Int64").astype(str).str[-2:]
    bk = k[k.series_ticker == "KXBTCMAXMON"].copy()
    ek = bk.key.str.extract(r"KXBTCMAXMON-BTC-(\d{2})([A-Z]{3})\d{2}-\d+")
    bk["yy"], bk["mon"] = ek[0], ek[1]
    bk["strike_round"] = (bk.floor_strike + 0.01).round(0)
    mm = bp.merge(bk, left_on=["yy", "mon", "strike"],
                  right_on=["yy", "mon", "strike_round"],
                  suffixes=("_pm", "_k"))
    for _, r in mm.iterrows():
        rows.append(("btc_monthly", f"{r.mon}{r.yy}-{int(r.strike)}",
                     r.key_k, r.key_pm, r.volume_k, r.volume_pm, "exact"))

    # btc_yearly (2025)
    paty = r"Will Bitcoin reach \$([\d,]+) by December 31, 2025\?"
    yp = p[p.question.str.match(paty, na=False)].copy()
    yp["strike"] = yp.question.str.extract(paty)[0].str.replace(
        ",", "").astype(float)
    yk = k[(k.series_ticker == "KXBTCMAXY")
           & k.key.str.contains("-25-DEC31")].copy()
    yk["strike_round"] = (yk.floor_strike + 0.01).round(0)
    mm = yp.merge(yk, left_on="strike", right_on="strike_round",
                  suffixes=("_pm", "_k"))
    for _, r in mm.iterrows():
        rows.append(("btc_yearly", f"Y25-{int(r.strike)}", r.key_k, r.key_pm,
                     r.volume_k, r.volume_pm, "approx"))

    # election (hand-matched)
    for unit, kkey, pq, qual in ELECTION_PAIRS:
        kr = k[k.key == kkey]
        pr = p[p.question == pq].sort_values("volume", ascending=False)
        if len(kr) and len(pr):
            rows.append(("election", unit, kkey, pr.key.iloc[0],
                         kr.volume.iloc[0], pr.volume.iloc[0], qual))

    df = pd.DataFrame(rows, columns=["domain", "unit", "k_key", "pm_key",
                                     "k_vol", "pm_vol", "quality"])
    return df.sort_values(["domain", "unit"]).reset_index(drop=True)


def _load_aligned(pairs: pd.DataFrame, freq: str = "1440m") -> pd.DataFrame:
    """Long panel: (pair_id, ts, p_k, p_pm) on the intersection of bars.

    Wall-clock alignment for daily bars: Kalshi daily candles are stamped
    at window START (midnight US/Eastern; verified with the PRES-2024
    election-night jump, which sits in the bar stamped Nov 6 05:00 UTC),
    so the label-t close occurs at wall clock t+1 ~04-05 UTC. Polymarket
    daily "bars" are point samples AT the label (00:00 UTC). Shifting
    Kalshi labels +1 day therefore aligns closes to within 4-5 hours;
    unshifted, same-label closes are ~28h apart and a spurious one-day
    "Kalshi leads" appears. 60m bars align on the hour (Kalshi hourly
    stamps are window starts too, so shift +1h).
    """
    out = []
    for pid, r in pairs.iterrows():
        fk = BARS / "kalshi" / freq / f"{r.k_key}.parquet"
        fp = BARS / "polymarket" / freq / f"{r.pm_key}.parquet"
        if not (fk.exists() and fp.exists()):
            continue
        bk = pd.read_parquet(fk, columns=["ts", "close"])
        bp = pd.read_parquet(fp, columns=["ts", "close"])
        if freq == "1440m":
            bk["ts"] = bk.ts.dt.floor("D") + pd.Timedelta(days=1)
            bp["ts"] = bp.ts.dt.floor("D")
        else:
            bk["ts"] = bk.ts.dt.floor("h") + pd.Timedelta(hours=1)
            bp["ts"] = bp.ts.dt.floor("h")
        m = bk.drop_duplicates("ts", keep="last").merge(
            bp.drop_duplicates("ts", keep="last"), on="ts",
            suffixes=("_k", "_pm"))
        if len(m) < 3:
            continue
        m = m.rename(columns={"close_k": "p_k", "close_pm": "p_pm"})
        m["pair_id"] = pid
        out.append(m[["pair_id", "ts", "p_k", "p_pm"]])
    if not out:
        return pd.DataFrame(columns=["pair_id", "ts", "p_k", "p_pm"])
    return pd.concat(out, ignore_index=True).sort_values(
        ["pair_id", "ts"]).reset_index(drop=True)


def _within_ar1(panel: pd.DataFrame, col: str) -> float:
    """Pooled AR(1) coefficient of `col`, demeaned within pair."""
    g = panel[col] - panel.groupby("pair_id")[col].transform("mean")
    lag = g.groupby(panel.pair_id).shift(1)
    ok = lag.notna()
    x, y = lag[ok].to_numpy(), g[ok].to_numpy()
    return float(x @ y / (x @ x))


def gap_stats(panel: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    gap = panel.p_pm - panel.p_k
    dom = pairs.domain.reindex(panel.pair_id).to_numpy()
    qual = pairs.quality.reindex(panel.pair_id).to_numpy()
    by_dom = {d: float(np.mean(np.abs(gap[dom == d])))
              for d in pairs.domain.unique()}
    by_qual = {q: float(np.mean(np.abs(gap[qual == q])))
               for q in ("exact", "approx")}
    rho = _within_ar1(panel.assign(gap=gap), "gap")
    return {
        "n_obs": int(len(panel)),
        "n_pairs": int(panel.pair_id.nunique()),
        "mean_abs_gap": float(np.abs(gap).mean()),
        "median_abs_gap": float(np.abs(gap).median()),
        "rms_gap": float(np.sqrt((gap ** 2).mean())),
        "frac_gt_5c": float((np.abs(gap) > 0.05).mean()),
        "mean_abs_gap_by_domain": by_dom,
        "mean_abs_gap_by_quality": by_qual,
        "gap_ar1_rho": rho,
        "gap_half_life_days": float(np.log(0.5) / np.log(abs(rho)))
        if 0 < abs(rho) < 1 else float("nan"),
    }


def paired_budget(panel: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Per pair per venue: sum dp^2 over the common window vs p0(1-p0)."""
    g = panel.groupby("pair_id")
    d_k = g.p_k.diff()
    d_pm = g.p_pm.diff()
    agg = pd.DataFrame({
        "ss_k": (d_k ** 2).groupby(panel.pair_id).sum(),
        "ss_pm": (d_pm ** 2).groupby(panel.pair_id).sum(),
        "p0_k": g.p_k.first(),
        "p0_pm": g.p_pm.first(),
        "n": g.size(),
    })
    agg["bud_k"] = agg.p0_k * (1 - agg.p0_k)
    agg["bud_pm"] = agg.p0_pm * (1 - agg.p0_pm)
    ok = (agg.bud_k > 1e-4) & (agg.bud_pm > 1e-4)
    agg = agg[ok]
    agg["ratio_k"] = agg.ss_k / agg.bud_k
    agg["ratio_pm"] = agg.ss_pm / agg.bud_pm
    agg["ratio_of_ratios"] = agg.ratio_k / agg.ratio_pm
    return {
        "n_pairs": int(len(agg)),
        "median_ratio_k": float(agg.ratio_k.median()),
        "median_ratio_pm": float(agg.ratio_pm.median()),
        "median_ratio_of_ratios": float(agg.ratio_of_ratios.median()),
        "frac_k_gt_pm": float((agg.ratio_of_ratios > 1).mean()),
    }, agg


def paired_power(panel: pd.DataFrame) -> dict:
    """Pooled slope of log dp^2 on log p(1-p), per venue, matched panel only.

    Bucketed like empirics_v2: 14 quantile buckets of p(1-p), regress
    log mean(dp^2) on log mean p(1-p) across buckets.
    """
    out = {}
    for venue, pcol in (("kalshi", "p_k"), ("polymarket", "p_pm")):
        p = panel[pcol]
        dp = panel.groupby("pair_id")[pcol].diff()
        lvl = (p * (1 - p)).groupby(panel.pair_id).shift(1)
        ok = dp.notna() & lvl.notna() & (lvl > 1e-6)
        x, y = lvl[ok].to_numpy(), (dp[ok] ** 2).to_numpy()
        q = np.unique(np.quantile(x, np.linspace(0, 1, 15)))
        nb = len(q) - 1
        b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, nb - 1)
        cnt = np.bincount(b, minlength=nb)
        mx = np.array([x[b == i].mean() if cnt[i] else np.nan
                       for i in range(nb)])
        my = np.array([y[b == i].mean() if cnt[i] else np.nan
                       for i in range(nb)])
        keep = np.nan_to_num(my) > 0
        lx, ly = np.log(mx[keep]), np.log(my[keep])
        gamma = float(np.polyfit(lx, ly, 1)[0])
        out[venue] = {"gamma": gamma, "n_obs": int(ok.sum())}
    return out


def lead_lag(panel: pd.DataFrame, max_lag: int = 3) -> dict:
    """Pooled cross-correlation of changes + predictive regressions."""
    g = panel.groupby("pair_id")
    dk = g.p_k.diff()
    dpm = g.p_pm.diff()
    pid = panel.pair_id
    xcorr = {}
    for lag in range(-max_lag, max_lag + 1):
        a = dpm.groupby(pid).shift(lag) if lag > 0 else dpm
        b = dk if lag > 0 else dk.groupby(pid).shift(-lag)
        ok = a.notna() & b.notna()
        xcorr[str(lag)] = float(np.corrcoef(a[ok], b[ok])[0, 1])

    def _predict(dy: pd.Series, dx: pd.Series) -> dict:
        """OLS dy_t ~ dy_{t-1} + dx_{t-1}, pooled."""
        y = dy
        x1 = dy.groupby(pid).shift(1)
        x2 = dx.groupby(pid).shift(1)
        ok = y.notna() & x1.notna() & x2.notna()
        X = np.column_stack([np.ones(ok.sum()), x1[ok], x2[ok]])
        yv = y[ok].to_numpy()
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        dof = len(yv) - 3
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X))
                     * (resid @ resid / dof))
        return {"own_lag": float(beta[1]), "cross_lag": float(beta[2]),
                "cross_t": float(beta[2] / se[2]), "n": int(ok.sum())}

    return {"xcorr_dpm_leads_dk": xcorr,
            "pm_predicts_k": _predict(dk, dpm),
            "k_predicts_pm": _predict(dpm, dk)}


def main(argv: list[str]) -> int:
    pairs = build_pairs()
    pairs.to_csv(RESULTS / "bridge_pairs.csv", index=False)
    print(f"bridge: {len(pairs)} pairs "
          f"({(pairs.quality == 'exact').sum()} exact)")

    panel = _load_aligned(pairs, "1440m")
    gaps = gap_stats(panel, pairs)
    bud, agg = paired_budget(panel, pairs)
    power = paired_power(panel)
    ll = lead_lag(panel)

    agg = agg.join(pairs[["domain", "unit", "quality"]])
    agg.to_csv(RESULTS / "bridge_pair_metrics.csv", index_label="pair_id")

    out = {"pairs": {"n": int(len(pairs)),
                     "n_exact": int((pairs.quality == "exact").sum()),
                     "by_domain": pairs.domain.value_counts().to_dict()},
           "gap": gaps, "paired_budget": bud, "paired_power": power,
           "lead_lag_daily": ll}

    # best-effort hourly lead-lag
    if (BARS / "kalshi" / "60m").exists() and (
            BARS / "polymarket" / "60m").exists():
        panel_h = _load_aligned(pairs, "60m")
        if len(panel_h):
            out["lead_lag_hourly"] = lead_lag(panel_h, max_lag=6)
            out["lead_lag_hourly"]["n_obs"] = int(len(panel_h))
            print(f"bridge: hourly panel {len(panel_h)} obs, "
                  f"{panel_h.pair_id.nunique()} pairs")
    else:
        print("bridge: no 60m bars, skipping hourly lead-lag "
              "(run pull_v5_hourly.py)")

    with open(RESULTS / "bridge_v5.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("gap", "paired_budget")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
