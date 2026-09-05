"""Parquet storage, DuckDB views, and per-(venue,freq) checkpoints.

Layout (do not deviate):
  data/catalog/{venue}.parquet          one row per tracked outcome
  data/bars/{venue}/{freq}/{key}.parquet  freq like "1440m"; columns exactly:
      ts (UTC), open, high, low, close, bid, ask, volume, open_interest
  data/state/{venue}_{freq}.json        {"done": [key, ...]} closed+fully-pulled
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import duckdb
import pandas as pd

from .config import BARS_DIR, CATALOG_DIR, DATA_DIR, STATE_DIR

log = logging.getLogger(__name__)

BAR_COLS = ["ts", "open", "high", "low", "close", "bid", "ask", "volume", "open_interest"]

CATALOG_BASE_COLS = [
    "venue", "key", "question", "start", "end", "closed", "final_price", "volume",
]

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(key: str) -> str:
    """Filesystem-safe filename for a market key."""
    return _SAFE.sub("_", str(key))


# ---------------------------------------------------------------- catalog

def write_catalog(venue: str, df: pd.DataFrame) -> Path:
    """Merge new catalog rows with any existing file, dedupe on key (keep new)."""
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    path = CATALOG_DIR / f"{venue}.parquet"
    for c in CATALOG_BASE_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df["closed"] = df["closed"].astype("boolean")
    df["final_price"] = pd.to_numeric(df["final_price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["key"] = df["key"].astype(str)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=["key"], keep="last").reset_index(drop=True)
    df.to_parquet(path, index=False)
    return path


def read_catalog(venue: str) -> pd.DataFrame:
    path = CATALOG_DIR / f"{venue}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=CATALOG_BASE_COLS)
    return pd.read_parquet(path)


# ---------------------------------------------------------------- bars

def bars_path(venue: str, freq_minutes: int, key: str) -> Path:
    return BARS_DIR / venue / f"{freq_minutes}m" / f"{safe_name(key)}.parquet"


def write_bars(venue: str, freq_minutes: int, key: str, df: pd.DataFrame) -> int:
    """Merge with existing file, dedupe on ts (keep last). Returns rows written.

    Incoming df must have a 'ts' column (tz-aware UTC) and close; missing bar
    columns are filled with NaN so the schema is always exactly BAR_COLS.
    """
    if df is None or df.empty:
        return 0
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for c in BAR_COLS:
        if c not in df.columns:
            df[c] = float("nan")
    df = df[BAR_COLS]
    for c in BAR_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    path = bars_path(venue, freq_minutes, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        old["ts"] = pd.to_datetime(old["ts"], utc=True)
        df = pd.concat([old, df], ignore_index=True)
    df = (
        df.drop_duplicates(subset=["ts"], keep="last")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    df.to_parquet(path, index=False)
    return len(df)


# ---------------------------------------------------------------- checkpoint

def _state_path(venue: str, freq_minutes: int) -> Path:
    return STATE_DIR / f"{venue}_{freq_minutes}m.json"


def load_done(venue: str, freq_minutes: int) -> set[str]:
    path = _state_path(venue, freq_minutes)
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()).get("done", []))


def mark_done(venue: str, freq_minutes: int, keys: set[str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _state_path(venue, freq_minutes)
    done = load_done(venue, freq_minutes) | set(keys)
    path.write_text(json.dumps({"done": sorted(done)}, indent=0))


# ---------------------------------------------------------------- duckdb

def con() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with `bars` and `catalog` views.

    bars: venue, freq, key parsed from the filename; all bar columns.
    catalog: union of data/catalog/*.parquet.
    """
    c = duckdb.connect()
    bars_glob = str(BARS_DIR / "*" / "*" / "*.parquet").replace("\\", "/")
    cat_glob = str(CATALOG_DIR / "*.parquet").replace("\\", "/")
    if any(BARS_DIR.rglob("*.parquet")):
        c.execute(
            f"""
            CREATE VIEW bars AS
            SELECT
              regexp_extract(replace(filename, '\\', '/'), '/bars/([^/]+)/', 1) AS venue,
              regexp_extract(replace(filename, '\\', '/'), '/bars/[^/]+/([^/]+)/', 1) AS freq,
              regexp_extract(replace(filename, '\\', '/'), '/([^/]+)\\.parquet$', 1) AS key,
              * EXCLUDE (filename)
            FROM read_parquet('{bars_glob}', filename=true, union_by_name=true)
            """
        )
    else:
        c.execute(
            "CREATE VIEW bars AS SELECT NULL::VARCHAR venue, NULL::VARCHAR freq, "
            "NULL::VARCHAR key, NULL::TIMESTAMPTZ ts, NULL::DOUBLE open, NULL::DOUBLE high, "
            "NULL::DOUBLE low, NULL::DOUBLE close, NULL::DOUBLE bid, NULL::DOUBLE ask, "
            "NULL::DOUBLE volume, NULL::DOUBLE open_interest WHERE 1=0"
        )
    if any(CATALOG_DIR.glob("*.parquet")):
        c.execute(
            f"CREATE VIEW catalog AS SELECT * FROM read_parquet('{cat_glob}', union_by_name=true)"
        )
    else:
        c.execute(
            "CREATE VIEW catalog AS SELECT NULL::VARCHAR venue, NULL::VARCHAR key, "
            "NULL::VARCHAR question, NULL::TIMESTAMPTZ \"start\", NULL::TIMESTAMPTZ \"end\", "
            "NULL::BOOLEAN closed, NULL::DOUBLE final_price, NULL::DOUBLE volume WHERE 1=0"
        )
    return c
