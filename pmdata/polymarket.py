"""Polymarket adapter: Gamma (universe) + CLOB prices-history (bars).

Universe: GET {GAMMA_BASE}/markets ordered by volumeNum desc with
volume-descending cursor pagination (volume_num_max = smallest volume seen;
plain offsets 422 beyond ~2000), closed filter, volume_num_min filter.
Gamma encodes clobTokenIds,
outcomes and outcomePrices as JSON strings; parse defensively. We track the
Yes outcome token only; key = that token id.

Bars: GET {CLOB_BASE}/prices-history?market=<token>. Single price series
{"history":[{"t":unix,"p":price}]} with no OHLC; OHLC is built by resampling
(label=right, closed=right). bid/ask/volume/open_interest are NaN here.
Absolute startTs/endTs windows longer than ~15-20 days are rejected (HTTP 400),
so intraday pulls are chunked per PM_CHUNK_DAYS.
"""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from .config import CLOB_BASE, GAMMA_BASE, PM_CHUNK_DAYS
from .http import get_json

log = logging.getLogger(__name__)

VENUE = "polymarket"
PAGE = 100  # Gamma silently caps limit at 100 (limit=500 returns 100 rows)


def _jloads(v, default=None):
    """Gamma returns list-valued fields as JSON-encoded strings; be defensive."""
    if v is None:
        return default
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return default


def parse_market(m: dict) -> dict | None:
    """One Gamma market dict -> catalog row (Yes outcome only), or None."""
    outcomes = _jloads(m.get("outcomes"), [])
    tokens = _jloads(m.get("clobTokenIds"), [])
    prices = _jloads(m.get("outcomePrices"), [])
    if not outcomes or not tokens or len(tokens) != len(outcomes):
        return None
    try:
        yes_idx = [o.lower() for o in outcomes].index("yes")
    except ValueError:
        yes_idx = 0  # scalar/categorical two-outcome markets: track first outcome
    key = str(tokens[yes_idx])
    if not key:
        return None
    closed = bool(m.get("closed"))
    final = None
    if closed and prices and len(prices) > yes_idx:
        try:
            fp = float(prices[yes_idx])
            # only accept fully resolved 0/1 prints as final_price
            if fp <= 0.005 or fp >= 0.995:
                final = round(fp)
        except (TypeError, ValueError):
            pass
    events = m.get("events") or []
    ev = events[0] if events else {}
    return {
        "venue": VENUE,
        "key": key,
        "question": m.get("question"),
        "start": m.get("startDate"),
        "end": m.get("endDate"),
        "closed": closed,
        "final_price": final,
        "volume": m.get("volumeNum") or m.get("volume"),
        # venue-specific extras
        "condition_id": m.get("conditionId"),
        "slug": m.get("slug"),
        "outcome": outcomes[yes_idx],
        "event_id": ev.get("id"),
        "event_slug": ev.get("slug"),
        "neg_risk": bool(m.get("negRisk")),
        "gamma_id": m.get("id"),
    }


def fetch_universe(closed: bool | None = None, min_volume: float | None = None,
                   max_markets: int | None = None) -> pd.DataFrame:
    """Paginate Gamma /markets ordered by volume desc; return catalog rows."""
    # Gamma caps `limit` at 100 and rejects offsets beyond ~2000 (HTTP 422),
    # so we page by descending volume instead: after each page, constrain
    # volume_num_max to the smallest volume seen. Dedupe on key downstream.
    rows: list[dict] = []
    vol_max: float | None = None
    seen: set[str] = set()
    while True:
        params = {"limit": PAGE, "offset": 0,
                  "order": "volumeNum", "ascending": "false"}
        if closed is not None:
            params["closed"] = "true" if closed else "false"
        if min_volume is not None:
            params["volume_num_min"] = min_volume
        if vol_max is not None:
            params["volume_num_max"] = vol_max
        page = get_json(f"{GAMMA_BASE}/markets", params=params)
        if not page:
            break
        vols = []
        for m in page:
            row = parse_market(m)
            if row is not None and row["key"] not in seen:
                seen.add(row["key"])
                rows.append(row)
            try:
                vols.append(float(m.get("volumeNum") or m.get("volume") or 0))
            except (TypeError, ValueError):
                pass
        log.info("gamma universe: vol_max=%s total_rows=%d", vol_max, len(rows))
        if max_markets and len(rows) >= max_markets:
            rows = rows[:max_markets]
            break
        if len(page) < PAGE or not vols:
            break
        new_max = min(vols)
        if vol_max is not None and new_max >= vol_max:
            break  # no progress (massive tie); avoid infinite loop
        vol_max = new_max
    return pd.DataFrame(rows)


def _history(token: str, params: dict) -> list[dict]:
    j = get_json(f"{CLOB_BASE}/prices-history", params={"market": token, **params})
    if not j:
        return []
    return j.get("history", [])


def fetch_bars(key: str, freq_minutes: int,
               start: pd.Timestamp | None = None,
               end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Pull the price series and resample to OHLC bars of freq_minutes."""
    pts: list[dict] = []
    if freq_minutes >= 1440:
        pts = _history(key, {"interval": "max", "fidelity": 1440})
    else:
        chunk_days = PM_CHUNK_DAYS.get(freq_minutes, 2)
        now = int(time.time())
        t1 = int(end.timestamp()) if end is not None else now
        t1 = min(t1, now)
        t0 = int(start.timestamp()) if start is not None else t1 - 30 * 86400
        step = chunk_days * 86400
        lo = t0
        while lo < t1:
            hi = min(lo + step, t1)
            pts.extend(_history(key, {"startTs": lo, "endTs": hi,
                                      "fidelity": freq_minutes}))
            lo = hi
    if not pts:
        return pd.DataFrame()
    s = pd.DataFrame(pts)
    s["ts"] = pd.to_datetime(s["t"], unit="s", utc=True)
    s["p"] = pd.to_numeric(s["p"], errors="coerce")
    s = s.dropna(subset=["p"]).drop_duplicates(subset=["ts"]).set_index("ts").sort_index()
    if s.empty:
        return pd.DataFrame()
    o = s["p"].resample(f"{freq_minutes}min", label="right", closed="right").ohlc()
    o = o.dropna(subset=["close"]).reset_index()
    return o  # store.write_bars fills bid/ask/volume/open_interest with NaN
