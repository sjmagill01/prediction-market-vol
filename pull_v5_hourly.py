"""V5 top-up: 60m bars for the 82 signed-off bridge pairs (both venues).

Daily bars cannot settle cross-venue lead-lag: Kalshi daily candles are
stamped at window start (midnight US/Eastern, close = end of window) while
Polymarket daily "bars" are point samples at 00:00 UTC, so same-label
closes are ~28h apart. Hourly bars give wall-clock alignment.

Reads results_v2/bridge_pairs.csv (written by analysis.bridge_v5) plus the
frozen catalogs; writes data_v2/bars/{venue}/60m/{key}.parquet. Resumable:
existing files are skipped.

Usage: python pull_v5_hourly.py
"""
from __future__ import annotations

import logging
import sys

import pandas as pd

from analysis.config import DATA_V2, RESULTS
from pmdata import kalshi, polymarket

log = logging.getLogger("pull_v5_hourly")
BARS = DATA_V2 / "bars"


def _ts(x) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, format="ISO8601", errors="coerce")


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    # dtype=str: PM keys are ~77-digit token ids; default parsing turns them
    # into Python big ints that overflow downstream arrow ops.
    pairs = pd.read_csv(RESULTS / "bridge_pairs.csv",
                        dtype={"k_key": str, "pm_key": str})
    kcat = pd.read_parquet(DATA_V2 / "catalog" / "kalshi.parquet")
    pcat = pd.read_parquet(DATA_V2 / "catalog" / "polymarket.parquet")
    kcat = kcat[kcat.key.isin(pairs.k_key)].set_index("key")
    pcat = pcat[pcat.key.isin(pairs.pm_key)].set_index("key")

    kdir = BARS / "kalshi" / "60m"
    pdir = BARS / "polymarket" / "60m"
    kdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_fail = 0
    for key in pairs.k_key.unique():
        out = kdir / f"{key}.parquet"
        if out.exists():
            n_skip += 1
            continue
        r = kcat.loc[key]
        end = _ts(r["end"])
        st = _ts(r.get("settlement_ts"))
        if pd.notna(st):
            end = max(end, st) if pd.notna(end) else st
        try:
            bars = kalshi.fetch_bars(
                key, 60, r.series_ticker, start=_ts(r["start"]),
                end=None if pd.isna(end) else end + pd.Timedelta(days=1),
                closed=True)
            if len(bars):
                bars.to_parquet(out, index=False)
                n_ok += 1
                log.info("kalshi %s: %d bars", key, len(bars))
            else:
                n_fail += 1
                log.warning("kalshi %s: empty", key)
        except Exception as e:  # noqa: BLE001 - one market never kills the run
            n_fail += 1
            log.warning("kalshi %s failed: %s", key, e)

    for key in pairs.pm_key.unique():
        out = pdir / f"{key}.parquet"
        if out.exists():
            n_skip += 1
            continue
        r = pcat.loc[key]
        start, end = _ts(r["start"]), _ts(r["end"])
        if pd.isna(start):
            log.warning("pm %s: no start ts, skipping", key)
            n_fail += 1
            continue
        try:
            # +2d past end: PM markets can trade past end_date to resolution.
            bars = polymarket.fetch_bars(
                key, 60, start=start,
                end=None if pd.isna(end) else end + pd.Timedelta(days=2))
            if len(bars):
                bars.to_parquet(out, index=False)
                n_ok += 1
                log.info("pm %s...: %d bars", key[:16], len(bars))
            else:
                n_fail += 1
                log.warning("pm %s...: empty", key[:16])
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            log.warning("pm %s... failed: %s", key[:16], e)

    log.info("done: %d pulled, %d skipped, %d failed", n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
