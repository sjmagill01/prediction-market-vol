"""V1 vintage pull: fresh scans + daily bars into data_v2/, one frozen run.

Reuses the tested pmdata venue adapters (parsers carry the dollar-string
fix and its regression tests) but owns its storage: everything lands under
data_v2/ with fresh state, so the v1 data/ tree is never touched.

Phases, in order (each is resumable; scans skip if their catalog exists,
bar phases skip keys in data_v2/state/):

    pm_scan      Gamma closed markets, vol >= PM_MIN_VOLUME (~100k rows)
    kalshi_scan  per-series two-tier scan (keeps category/frequency) +
                 global historical sweep for pre-cutoff markets
    pm_bars      daily closes for the top PM_TOP_N resolved PM markets
    pm_tags      event tags for the backfilled PM set (categorization)
    kalshi_bars  daily candles for closed Kalshi markets over the floors
    verify       scale/range gates + diff vs the repaired v1 Kalshi files

Progress: data_v2/progress.json is rewritten every ~10s with counts, rate
and ETA for the current phase; full log in data_v2/pull.log. Check with
`python pull_status.py`.

Usage: python pull_v2.py [phase ...]     (default: all phases in order)
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time

import numpy as np
import pandas as pd

from analysis.config import (DATA_V2, KALSHI_MIN_LIFETIME_DAYS,
                             KALSHI_MIN_VOLUME, PM_MIN_VOLUME, PM_TOP_N,
                             SEED)
from pmdata import kalshi, polymarket
from pmdata.http import get_json

log = logging.getLogger("pull_v2")

CATALOG = DATA_V2 / "catalog"
BARS = DATA_V2 / "bars"
STATE = DATA_V2 / "state"
PROGRESS = DATA_V2 / "progress.json"
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe(key: str) -> str:
    return _SAFE.sub("_", str(key))


# ---------------------------------------------------------------- progress
class Progress:
    """Phase progress with rate/ETA, checkpointed to progress.json."""

    def __init__(self, phase: str, total: int):
        self.phase, self.total = phase, total
        self.done = self.bars = 0
        self.t0 = time.time()
        self._last = 0.0
        self.write()

    def step(self, bars: int = 0) -> None:
        self.done += 1
        self.bars += bars
        if time.time() - self._last > 10:
            self.write()

    def write(self, status: str = "running") -> None:
        el = max(time.time() - self.t0, 1e-9)
        rate = self.done / el
        eta_s = (self.total - self.done) / rate if rate > 0 else None
        PROGRESS.write_text(json.dumps({
            "phase": self.phase, "status": status,
            "done": self.done, "total": self.total, "bar_rows": self.bars,
            "elapsed_min": round(el / 60, 1),
            "rate_per_min": round(rate * 60, 1),
            "eta_min": None if eta_s is None else round(eta_s / 60, 1),
            "eta_at": None if eta_s is None else time.strftime(
                "%H:%M", time.localtime(time.time() + eta_s)),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))
        self._last = time.time()

    def finish(self) -> None:
        self.write("done")


def _load_done(name: str) -> set[str]:
    p = STATE / f"{name}.json"
    if not p.exists():
        return set()
    return set(json.loads(p.read_text()).get("done", []))


def _save_done(name: str, done: set[str]) -> None:
    (STATE / f"{name}.json").write_text(json.dumps({"done": sorted(done)}))


# ------------------------------------------------------------------- scans
def pm_scan() -> None:
    out = CATALOG / "polymarket.parquet"
    if out.exists():
        log.info("pm_scan: %s exists, skipping", out)
        return
    pr = Progress("pm_scan", 1)
    cat = polymarket.fetch_universe(closed=True, min_volume=PM_MIN_VOLUME)
    cat.to_parquet(out, index=False)
    log.info("pm_scan: %d markets -> %s", len(cat), out)
    pr.step()
    pr.finish()


def kalshi_scan() -> None:
    out = CATALOG / "kalshi.parquet"
    if out.exists():
        log.info("kalshi_scan: %s exists, skipping", out)
        return
    series = kalshi.fetch_series()
    cat_map = {s["ticker"]: s.get("category") for s in series}
    freq_map = {s["ticker"]: s.get("frequency") for s in series}
    pr = Progress("kalshi_scan", len(series))
    rows: list[dict] = []
    for s in series:
        st = s["ticker"]
        for ep in ("/markets", "/historical/markets"):
            cursor = None
            try:
                while True:
                    params = {"series_ticker": st, "limit": 1000}
                    if ep == "/markets":
                        params["status"] = "settled"
                    if cursor:
                        params["cursor"] = cursor
                    j = get_json(kalshi.base() + ep, params=params)
                    if not j:
                        break
                    for m in j.get("markets", []):
                        r = kalshi._market_row(m, st, s.get("category"))
                        r["frequency"] = s.get("frequency")
                        rows.append(r)
                    cursor = j.get("cursor")
                    if not cursor:
                        break
            except Exception as e:  # noqa: BLE001 - one series never kills the scan
                log.warning("kalshi series %s (%s) failed: %s", st, ep, e)
        pr.step()
    log.info("kalshi_scan: per-series done, %d rows; global historical sweep",
             len(rows))
    hist = kalshi.fetch_universe_historical(
        min_volume=KALSHI_MIN_VOLUME,
        min_lifetime_days=KALSHI_MIN_LIFETIME_DAYS)
    if not hist.empty:
        hist["category"] = hist["series_ticker"].map(cat_map)
        hist["frequency"] = hist["series_ticker"].map(freq_map)
    cat = pd.concat([pd.DataFrame(rows), hist], ignore_index=True)
    cat = cat.drop_duplicates(subset=["key"], keep="first").reset_index(drop=True)
    cat.to_parquet(out, index=False)
    log.info("kalshi_scan: %d markets (%d from global sweep) -> %s",
             len(cat), len(hist), out)
    pr.finish()


# -------------------------------------------------------------------- bars
def _pm_selection() -> pd.DataFrame:
    cat = pd.read_parquet(CATALOG / "polymarket.parquet")
    sel = cat[cat["closed"].astype(bool) & cat["final_price"].notna()].copy()
    sel["volume"] = pd.to_numeric(sel["volume"], errors="coerce")
    return sel.sort_values("volume", ascending=False).head(PM_TOP_N)


def pm_bars() -> None:
    sel = _pm_selection()
    outdir = BARS / "polymarket" / "1440m"
    outdir.mkdir(parents=True, exist_ok=True)
    done = _load_done("polymarket_1440m")
    todo = sel[~sel["key"].isin(done)]
    log.info("pm_bars: %d markets (%d already done)", len(todo), len(done))
    pr = Progress("pm_bars", len(todo))
    for i, row in enumerate(todo.itertuples(index=False), 1):
        n = 0
        try:
            bars = polymarket.fetch_bars(row.key, 1440)
            if len(bars):
                bars.to_parquet(outdir / f"{safe(row.key)}.parquet", index=False)
                n = len(bars)
            done.add(row.key)
        except Exception as e:  # noqa: BLE001 - one market never kills the run
            log.warning("pm_bars %s failed: %s", row.key, e)
        pr.step(n)
        if i % 25 == 0:
            _save_done("polymarket_1440m", done)
    _save_done("polymarket_1440m", done)
    log.info("pm_bars: done, %d bar-rows", pr.bars)
    pr.finish()


def pm_tags() -> None:
    """Event tags for the backfilled PM set: the categorization layer.

    Gamma market objects carry no category; /events/{id} carries a tags
    list (labels like Politics, Crypto, Sports). One request per unique
    event of the top-N selection.
    """
    out = CATALOG / "pm_event_tags.parquet"
    have: dict[str, list[str]] = {}
    if out.exists():
        old = pd.read_parquet(out)
        have = {r.event_id: json.loads(r.tags) for r in old.itertuples()}
    ids = _pm_selection()["event_id"].dropna().astype(str).unique()
    todo = [i for i in ids if i not in have]
    log.info("pm_tags: %d events (%d already done)", len(todo), len(have))
    pr = Progress("pm_tags", len(todo))
    for i, ev_id in enumerate(todo, 1):
        try:
            e = get_json(f"{GAMMA_EVENTS}/{ev_id}")
            tags = [t.get("label") for t in (e or {}).get("tags") or []
                    if t.get("label")]
            have[ev_id] = tags
        except Exception as e:  # noqa: BLE001
            log.warning("pm_tags %s failed: %s", ev_id, e)
        pr.step()
        if i % 200 == 0:
            _write_tags(out, have)
    _write_tags(out, have)
    log.info("pm_tags: done, %d events tagged", len(have))
    pr.finish()


def _write_tags(out, have: dict[str, list[str]]) -> None:
    pd.DataFrame({"event_id": list(have),
                  "tags": [json.dumps(v) for v in have.values()]}
                 ).to_parquet(out, index=False)


def _kalshi_selection() -> pd.DataFrame:
    cat = pd.read_parquet(CATALOG / "kalshi.parquet")
    cat["volume"] = pd.to_numeric(cat["volume"], errors="coerce")
    start = pd.to_datetime(cat["start"], utc=True, errors="coerce")
    end = pd.to_datetime(cat["end"], utc=True, errors="coerce")
    life = (end - start).dt.total_seconds() / 86400
    sel = cat[cat["closed"].astype(bool)
              & (cat["volume"] >= KALSHI_MIN_VOLUME)
              & (life >= KALSHI_MIN_LIFETIME_DAYS)].copy()
    # note: itertuples mangles leading-underscore names, so no _-prefix here
    sel["start_ts"], sel["end_ts"] = start[sel.index], end[sel.index]
    return sel.sort_values("volume", ascending=False)


def kalshi_bars() -> None:
    sel = _kalshi_selection()
    outdir = BARS / "kalshi" / "1440m"
    outdir.mkdir(parents=True, exist_ok=True)
    done = _load_done("kalshi_1440m")
    todo = sel[~sel["key"].isin(done)]
    log.info("kalshi_bars: %d markets (%d already done)", len(todo), len(done))
    pr = Progress("kalshi_bars", len(todo))
    for i, row in enumerate(todo.itertuples(index=False), 1):
        n = 0
        try:
            end = row.end_ts
            st_ts = pd.to_datetime(row.settlement_ts, utc=True, errors="coerce") \
                if hasattr(row, "settlement_ts") else pd.NaT
            if pd.notna(st_ts):
                end = max(end, st_ts) if pd.notna(end) else st_ts
            bars = kalshi.fetch_bars(row.key, 1440, row.series_ticker,
                                     start=row.start_ts,
                                     end=None if pd.isna(end)
                                     else end + pd.Timedelta(days=1),
                                     closed=True)
            if len(bars):
                bars.to_parquet(outdir / f"{safe(row.key)}.parquet", index=False)
                n = len(bars)
            done.add(row.key)
        except Exception as e:  # noqa: BLE001
            log.warning("kalshi_bars %s failed: %s", row.key, e)
        pr.step(n)
        if i % 25 == 0:
            _save_done("kalshi_1440m", done)
    _save_done("kalshi_1440m", done)
    log.info("kalshi_bars: done, %d bar-rows", pr.bars)
    pr.finish()


# ------------------------------------------------------------------ verify
def verify() -> None:
    """Scale/range gates on the fresh pull + diff vs repaired v1 Kalshi."""
    rng = np.random.default_rng(SEED)
    pr = Progress("verify", 3)
    for venue in ("polymarket", "kalshi"):
        files = sorted((BARS / venue / "1440m").glob("*.parquet"))
        take = rng.choice(len(files), size=min(300, len(files)), replace=False)
        mx = []
        bad = 0
        for k in take:
            df = pd.read_parquet(files[k])
            c = df["close"].to_numpy()
            if len(c) == 0 or np.nanmin(c) < -1e-9 or np.nanmax(c) > 1 + 1e-9:
                bad += 1
            mx.append(np.nanmax(c) if len(c) else np.nan)
        med = float(np.nanmedian(mx))
        log.info("verify %s: %d files, sample med(max close)=%.3f, "
                 "%d out-of-range", venue, len(files), med, bad)
        if med < 0.2:
            log.error("verify %s: SCALE ALARM med(max close)=%.4f "
                      "(100x-style bug?)", venue, med)
        pr.step()
    # diff fresh Kalshi closes vs the repaired v1 files on overlapping keys
    v1dir = DATA_V2.parent / "data" / "bars" / "kalshi" / "1440m"
    v2dir = BARS / "kalshi" / "1440m"
    common = sorted({p.name for p in v2dir.glob("*.parquet")}
                    & {p.name for p in v1dir.glob("*.parquet")})
    if common:
        pick = rng.choice(len(common), size=min(300, len(common)),
                          replace=False)
        diffs = []
        for k in pick:
            a = pd.read_parquet(v1dir / common[k])[["ts", "close"]]
            b = pd.read_parquet(v2dir / common[k])[["ts", "close"]]
            m = a.merge(b, on="ts", suffixes=("_v1", "_v2"))
            if len(m):
                diffs.append(float((m["close_v1"] - m["close_v2"]).abs().max()))
        if diffs:
            d = np.array(diffs)
            log.info("verify kalshi-vs-v1: %d keys compared, max|dclose| "
                     "median=%.4g p95=%.4g max=%.4g", len(d),
                     np.median(d), np.quantile(d, 0.95), d.max())
    else:
        log.info("verify kalshi-vs-v1: no overlapping keys")
    pr.step()
    pr.finish()


# -------------------------------------------------------------------- main
PHASES = {"pm_scan": pm_scan, "kalshi_scan": kalshi_scan,
          "pm_bars": pm_bars, "pm_tags": pm_tags,
          "kalshi_bars": kalshi_bars, "verify": verify}


def main(argv: list[str]) -> int:
    for d in (CATALOG, BARS, STATE):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(DATA_V2 / "pull.log", encoding="utf-8"),
                  logging.StreamHandler()])
    names = argv or list(PHASES)
    for name in names:
        if name not in PHASES:
            print(f"unknown phase {name}; choose from {list(PHASES)}")
            return 2
    t0 = time.time()
    for name in names:
        log.info("=== phase %s", name)
        PHASES[name]()
    log.info("=== all phases done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
