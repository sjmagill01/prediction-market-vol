"""CLI: python -m pmdata.cli {universe,backfill,update,sql} ...

universe  --venue {polymarket,kalshi} [--closed|--open] [--status ...]
          [--min-volume N] [--max-series N]
backfill  --venue ... --freq {1m,1h,1d} [--only-closed] [--min-volume N] [--limit N]
update    re-pull all live (not closed) markets at every freq -- cron target
sql "<query>"  run against the DuckDB views and print

Per-market try/except so one bad market never kills a run; progress every 50.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import kalshi, polymarket, store
from .config import FREQS

log = logging.getLogger("pmdata")


def cmd_universe(args) -> None:
    if args.venue == "polymarket":
        closed = True if args.closed else False if args.open else None
        df = polymarket.fetch_universe(closed=closed, min_volume=args.min_volume)
    elif args.historical:
        df = kalshi.fetch_universe_historical(min_volume=args.min_volume,
                                              min_lifetime_days=args.min_lifetime)
    else:
        status = args.status or ("settled" if args.closed else "open" if args.open else None)
        df = kalshi.fetch_universe(status=status, min_volume=args.min_volume,
                                   max_series=args.max_series)
    if df.empty:
        log.warning("universe: no markets returned")
        return
    path = store.write_catalog(args.venue, df)
    log.info("universe: %d rows -> %s", len(df), path)


def _select_markets(venue: str, only_closed: bool, min_volume: float | None,
                    limit: int | None) -> pd.DataFrame:
    cat = store.read_catalog(venue)
    if cat.empty:
        log.error("catalog for %s is empty -- run `universe` first", venue)
        sys.exit(1)
    if only_closed:
        cat = cat[cat["closed"].fillna(False)]
    if min_volume is not None:
        cat = cat[pd.to_numeric(cat["volume"], errors="coerce") >= min_volume]
    cat = cat.sort_values("volume", ascending=False)
    if limit:
        cat = cat.head(limit)
    return cat.reset_index(drop=True)


def _backfill(venue: str, freq_name: str, cat: pd.DataFrame) -> None:
    fmin = FREQS[freq_name]
    done = store.load_done(venue, fmin)
    todo = cat[~cat["key"].isin(done)]
    log.info("backfill %s %s: %d markets (%d already done)",
             venue, freq_name, len(todo), len(cat) - len(todo))
    newly_done: set[str] = set()
    n_bars = 0
    for i, row in enumerate(todo.itertuples(index=False), 1):
        try:
            if venue == "polymarket":
                bars = polymarket.fetch_bars(row.key, fmin,
                                             start=row.start, end=row.end)
            else:
                is_closed = bool(row.closed) if pd.notna(row.closed) else False
                bars = kalshi.fetch_bars(row.key, fmin, row.series_ticker,
                                         start=row.start, end=row.end,
                                         closed=is_closed)
            n = store.write_bars(venue, fmin, row.key, bars)
            n_bars += n
            if pd.notna(row.closed) and bool(row.closed) and n > 0:
                newly_done.add(row.key)
        except Exception as e:  # noqa: BLE001 - one bad market never kills a run
            log.warning("market %s failed: %s", row.key, e)
        if i % 50 == 0:
            log.info("backfill %s %s: %d/%d markets, %d bar-rows so far",
                     venue, freq_name, i, len(todo), n_bars)
    if newly_done:
        store.mark_done(venue, fmin, newly_done)
    log.info("backfill %s %s done: %d bar-rows, %d markets checkpointed",
             venue, freq_name, n_bars, len(newly_done))


def cmd_backfill(args) -> None:
    cat = _select_markets(args.venue, args.only_closed, args.min_volume, args.limit)
    _backfill(args.venue, args.freq, cat)


def cmd_update(args) -> None:
    """Re-pull all live (not closed) markets at every freq."""
    for venue in ("polymarket", "kalshi"):
        cat = store.read_catalog(venue)
        if cat.empty:
            continue
        live = cat[~cat["closed"].fillna(False)].reset_index(drop=True)
        for freq_name in FREQS:
            _backfill(venue, freq_name, live)


def cmd_sql(args) -> None:
    c = store.con()
    df = c.execute(args.query).fetchdf()
    with pd.option_context("display.width", 200, "display.max_rows", 200,
                           "display.max_columns", 50):
        print(df.to_string(index=False))


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="pmdata")
    sub = ap.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("universe")
    u.add_argument("--venue", required=True, choices=["polymarket", "kalshi"])
    g = u.add_mutually_exclusive_group()
    g.add_argument("--closed", action="store_true")
    g.add_argument("--open", action="store_true")
    u.add_argument("--status", help="kalshi status filter, e.g. settled/open")
    u.add_argument("--min-volume", type=float)
    u.add_argument("--max-series", type=int, help="kalshi: cap series scanned")
    u.add_argument("--historical", action="store_true",
                   help="kalshi: page the global /historical/markets endpoint")
    u.add_argument("--min-lifetime", type=float, default=5.0,
                   help="kalshi --historical: min market lifetime in days")
    u.set_defaults(func=cmd_universe)

    b = sub.add_parser("backfill")
    b.add_argument("--venue", required=True, choices=["polymarket", "kalshi"])
    b.add_argument("--freq", required=True, choices=list(FREQS))
    b.add_argument("--only-closed", action="store_true")
    b.add_argument("--min-volume", type=float)
    b.add_argument("--limit", type=int)
    b.set_defaults(func=cmd_backfill)

    up = sub.add_parser("update")
    up.set_defaults(func=cmd_update)

    q = sub.add_parser("sql")
    q.add_argument("query")
    q.set_defaults(func=cmd_sql)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
