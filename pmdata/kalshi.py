"""Kalshi adapter: series/markets universe + candlesticks bars. Public reads.

Base URL: both candidates in config are probed via GET /exchange/status; the
first that answers is cached for the process.

Universe: GET /series (the API returns ALL series in one page and ignores
`limit` -- observed live, see NOTES.md), then GET /markets?series_ticker=...
per series with cursor pagination. key = market ticker. The market object no
longer carries series_ticker, so we record it from the query context.

Numbers: Kalshi replaced integer-cent fields with *_dollars / *_fp string
fields (Aug 2026). _num() reads whichever generation is present.

Bars: GET /series/{series}/markets/{ticker}/candlesticks with
period_interval in {1,60,1440}, chunked under the per-request candle cap.
Markets settled before the /historical/cutoff timestamps must be read from
GET /historical/markets/{ticker}/candlesticks; we also fall back to it
whenever the live endpoint returns nothing for a closed market.
"""
from __future__ import annotations

import datetime as dt
import logging
import math

import pandas as pd

from .config import KALSHI_BASES, KALSHI_MAX_CANDLES
from .http import get_json

log = logging.getLogger(__name__)

VENUE = "kalshi"
_base_cache: str | None = None


def base() -> str:
    global _base_cache
    if _base_cache:
        return _base_cache
    for b in KALSHI_BASES:
        try:
            st = get_json(b + "/exchange/status")
        except Exception as e:  # noqa: BLE001 - probing alternates
            log.warning("kalshi base %s failed: %s", b, e)
            st = None
        if st is not None:
            _base_cache = b
            log.info("kalshi base: %s", b)
            return b
    raise RuntimeError("no Kalshi base URL reachable")


# ---------------------------------------------------------------- numbers

def _num(d: dict | None, name: str, cents: bool = False) -> float:
    """Read a numeric field across API generations.

    New style: {name}_dollars ("0.9700") or {name}_fp ("249280.80") strings.
    Old style: integer field {name} (prices in cents when cents=True).
    """
    if not d:
        return math.nan
    v = d.get(f"{name}_dollars")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return math.nan
    v = d.get(f"{name}_fp")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return math.nan
    v = d.get(name)
    if v is None:
        return math.nan
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x / 100.0 if cents else x


def _ts(iso: str | None) -> pd.Timestamp | None:
    if not iso:
        return None
    return pd.Timestamp(iso).tz_convert("UTC") if pd.Timestamp(iso).tzinfo \
        else pd.Timestamp(iso, tz="UTC")


# ---------------------------------------------------------------- universe

def _desc(s: str) -> str:
    """Invert an ASCII timestamp string so ascending sort = newest first."""
    return "".join(chr(255 - ord(ch)) for ch in s)


def fetch_series(max_series: int | None = None) -> list[dict]:
    j = get_json(base() + "/series", params={"limit": 1000}) or {}
    series = j.get("series", [])
    # The API ships ALL series in one page (limit ignored). Old series have no
    # markets in the live tier at all (historical tier only). Scan order:
    # long-horizon frequencies first (their markets have multi-day price
    # paths; hourly/sports series settle within minutes and produce no daily
    # bars), then most recently updated; ticker as deterministic tiebreak.
    freq_rank = {"annual": 0, "quarterly": 1, "monthly": 2, "weekly": 3,
                 "one_off": 4, "custom": 5, "daily": 6, "hourly": 7,
                 "fifteen_min": 8}

    def _key(s):
        rank = freq_rank.get(s.get("frequency"), 4)
        recency = s.get("last_updated_ts") or ""
        return (rank, _desc(recency), s.get("ticker", ""))

    series.sort(key=_key)
    if max_series:
        series = series[:max_series]
    return series


def _market_row(m: dict, series_ticker: str, category: str | None) -> dict:
    result = (m.get("result") or "").lower()
    final = 1.0 if result == "yes" else 0.0 if result == "no" else None
    status = (m.get("status") or "").lower()
    closed = status in ("settled", "finalized", "closed") or final is not None
    return {
        "venue": VENUE,
        "key": m["ticker"],
        "question": m.get("title"),
        "start": m.get("open_time"),
        "end": m.get("close_time"),
        "closed": closed,
        "final_price": final,
        "volume": _num(m, "volume"),
        # venue-specific extras (strike fields kept for implied-vol work)
        "series_ticker": series_ticker,
        "event_ticker": m.get("event_ticker"),
        "status": status,
        "result": result or None,
        "floor_strike": m.get("floor_strike"),
        "cap_strike": m.get("cap_strike"),
        "strike_type": m.get("strike_type"),
        "category": category,
        "open_interest": _num(m, "open_interest"),
        "settlement_ts": m.get("settlement_ts"),
    }


def fetch_universe(status: str | None = "settled", min_volume: float | None = None,
                   max_series: int | None = None) -> pd.DataFrame:
    rows = []
    series = fetch_series(max_series)
    log.info("kalshi: %d series to scan", len(series))
    # Markets settled before the historical cutoff are invisible to the live
    # /markets endpoint, so each series is scanned on BOTH tiers; the
    # historical tier accepts series_ticker + cursor just like the live one
    # (all its markets are status=finalized, so the status filter only applies
    # to the live tier).
    want_settled = status in (None, "settled", "finalized", "closed")
    for i, s in enumerate(series):
        st = s["ticker"]
        endpoints = ["/markets"] + (["/historical/markets"] if want_settled else [])
        for ep in endpoints:
            cursor = None
            try:
                while True:
                    params = {"series_ticker": st, "limit": 1000}
                    if status and ep == "/markets":
                        params["status"] = status
                    if cursor:
                        params["cursor"] = cursor
                    j = get_json(base() + ep, params=params)
                    if not j:
                        break
                    for m in j.get("markets", []):
                        row = _market_row(m, st, s.get("category"))
                        if min_volume is not None and not (row["volume"] >= min_volume):
                            continue
                        rows.append(row)
                    cursor = j.get("cursor")
                    if not cursor:
                        break
            except Exception as e:  # noqa: BLE001 - one bad series never kills a run
                log.warning("kalshi series %s (%s) failed: %s", st, ep, e)
        if (i + 1) % 50 == 0:
            log.info("kalshi universe: %d/%d series, %d markets", i + 1, len(series), len(rows))
    return pd.DataFrame(rows)


def fetch_universe_historical(min_volume: float | None = None,
                              min_lifetime_days: float = 5.0,
                              max_pages: int | None = None) -> pd.DataFrame:
    """Page the GLOBAL /historical/markets endpoint (no series filter).

    Recovers the complete pre-cutoff settled universe, which the per-series
    scan only samples. The stream is dominated by minutes-lived sports/event
    shards, so rows are filtered client-side on lifetime (open->close) and
    volume. series_ticker is derived from the ticker prefix (SERIES-EVENT-MKT)
    since the market object does not carry it.
    """
    rows = []
    cursor = None
    pages = 0
    while True:
        params = {"limit": 1000}
        if cursor:
            params["cursor"] = cursor
        j = get_json(base() + "/historical/markets", params=params)
        if not j:
            break
        for m in j.get("markets", []):
            o, c = _ts(m.get("open_time")), _ts(m.get("close_time"))
            if o is None or c is None or (c - o) < pd.Timedelta(days=min_lifetime_days):
                continue
            row = _market_row(m, m.get("ticker", "").split("-")[0], None)
            if min_volume is not None and not (row["volume"] >= min_volume):
                continue
            rows.append(row)
        cursor = j.get("cursor")
        pages += 1
        if pages % 100 == 0:
            log.info("kalshi historical: %d pages, %d kept", pages, len(rows))
        if not cursor or (max_pages and pages >= max_pages):
            break
    log.info("kalshi historical: done, %d pages, %d markets kept", pages, len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- candles

_cutoff_cache: list = []  # [Timestamp|None] once fetched


def historical_cutoff() -> pd.Timestamp | None:
    if _cutoff_cache:
        return _cutoff_cache[0]
    j = get_json(base() + "/historical/cutoff")
    # observed live shape: flat dict of ISO timestamps, e.g.
    # {"market_settled_ts": "2026-07-06T00:00:00Z", ...}
    val = _ts(j.get("market_settled_ts")) if j else None
    _cutoff_cache.append(val)
    return val


def _parse_candles(candles: list[dict]) -> pd.DataFrame:
    rows = []
    for c in candles:
        price, ybid, yask = c.get("price"), c.get("yes_bid"), c.get("yes_ask")
        o = _num(price, "open", cents=True)
        h = _num(price, "high", cents=True)
        l = _num(price, "low", cents=True)  # noqa: E741
        cl = _num(price, "close", cents=True)
        bid = _num(ybid, "close", cents=True)
        ask = _num(yask, "close", cents=True)
        if math.isnan(cl):
            # no trades in the bar: fall back to bid/ask midpoint for continuity
            mid = (bid + ask) / 2.0
            o = h = l = cl = mid
        rows.append({
            "ts": pd.Timestamp(int(c["end_period_ts"]), unit="s", tz="UTC"),
            "open": o, "high": h, "low": l, "close": cl,
            "bid": bid, "ask": ask,
            "volume": _num(c, "volume"),
            "open_interest": _num(c, "open_interest"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["close"])
    return df


def fetch_bars(key: str, freq_minutes: int, series_ticker: str,
               start: pd.Timestamp | None = None,
               end: pd.Timestamp | None = None,
               closed: bool = False) -> pd.DataFrame:
    if freq_minutes not in (1, 60, 1440):
        raise ValueError("kalshi supports period_interval in {1,60,1440} only")
    start = None if start is None or pd.isna(start) else start
    end = None if end is None or pd.isna(end) else end
    now = pd.Timestamp.now(tz="UTC")
    t1 = int(min(end or now, now).timestamp())
    t0 = int((start or (now - pd.Timedelta(days=90))).timestamp())
    if t0 >= t1:
        return pd.DataFrame()
    cutoff = historical_cutoff()
    use_historical = bool(closed and cutoff is not None
                          and pd.Timestamp(t1, unit="s", tz="UTC") < cutoff)
    step = KALSHI_MAX_CANDLES * freq_minutes * 60
    frames = []
    for url in _endpoints(key, series_ticker, use_historical):
        frames = []
        lo = t0
        ok = True
        while lo < t1:
            hi = min(lo + step, t1)
            j = get_json(url, params={"start_ts": lo, "end_ts": hi,
                                      "period_interval": freq_minutes})
            if j is None:
                ok = False
                break
            frames.append(_parse_candles(j.get("candlesticks", [])))
            lo = hi
        got = sum(len(f) for f in frames)
        if ok and got:
            break
        # fall through to the alternate endpoint when empty/404 (closed markets
        # may have migrated to the historical tier since the cutoff pull)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def _endpoints(key: str, series_ticker: str, historical_first: bool) -> list[str]:
    live = f"{base()}/series/{series_ticker}/markets/{key}/candlesticks"
    hist = f"{base()}/historical/markets/{key}/candlesticks"
    return [hist, live] if historical_first else [live, hist]
