"""Parser unit tests against fixture JSON captured from the live APIs."""
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmdata import kalshi, polymarket  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text())


# ---------------------------------------------------------------- polymarket

def test_gamma_parse_market():
    markets = load("gamma_markets.json")
    row = polymarket.parse_market(markets[0])
    assert row is not None
    assert row["venue"] == "polymarket"
    assert row["key"].isdigit() and len(row["key"]) > 30  # clob token id
    assert row["closed"] is True
    assert row["final_price"] in (0, 1)
    assert float(row["volume"]) > 100_000
    assert row["outcome"].lower() == "yes"
    assert row["event_slug"]


def test_gamma_parse_market_defensive():
    # JSON-string fields malformed -> None, no raise
    assert polymarket.parse_market({"outcomes": "not json", "clobTokenIds": "["}) is None
    assert polymarket.parse_market({}) is None


def test_clob_history_resample(monkeypatch):
    hist = load("clob_prices_history_daily.json")["history"]
    monkeypatch.setattr(polymarket, "_history", lambda key, params: hist)
    bars = polymarket.fetch_bars("tok", 1440)
    assert not bars.empty
    assert str(bars["ts"].dt.tz) == "UTC"
    assert bars["close"].between(0, 1).all()
    assert (bars["high"] >= bars["low"]).all()
    # label=right, closed=right: bar timestamps are period ends
    assert bars["ts"].is_monotonic_increasing


# ---------------------------------------------------------------- kalshi

def test_num_new_generation():
    assert kalshi._num({"volume_fp": "249280.80"}, "volume") == pytest.approx(249280.8)
    assert kalshi._num({"close_dollars": "0.9700"}, "close", cents=True) == pytest.approx(0.97)


def test_num_old_generation_cents():
    # pre-Aug-2026 integer-cent generation
    assert kalshi._num({"close": 97}, "close", cents=True) == pytest.approx(0.97)
    assert kalshi._num({"volume": 1234}, "volume") == pytest.approx(1234.0)


def test_num_historical_tier_dollar_strings():
    # /historical/.../candlesticks: legacy field NAME, dollar STRING value.
    # Dividing these by 100 scaled every historical bar down 100x (found
    # 2026-09-05 when index strike strips summed to 0.01 instead of 1).
    assert kalshi._num({"close": "0.7200"}, "close", cents=True) == pytest.approx(0.72)
    assert kalshi._num({"close": "0.0050"}, "close", cents=True) == pytest.approx(0.005)


def test_num_missing_is_nan():
    assert math.isnan(kalshi._num({}, "close"))
    assert math.isnan(kalshi._num(None, "close"))


def test_parse_candles_fixture():
    candles = load("kalshi_candlesticks_1d.json")["candlesticks"]
    df = kalshi._parse_candles(candles)
    assert not df.empty
    for c in ("open", "high", "low", "close", "bid", "ask"):
        assert df[c].dropna().between(0, 1).all(), c
    assert str(df["ts"].dt.tz) == "UTC"
    assert (df["volume"].dropna() >= 0).all()


def test_parse_candles_midpoint_fallback():
    c = {"end_period_ts": 1700000000,
         "price": {},  # no trades in bar
         "yes_bid": {"close_dollars": "0.4000"},
         "yes_ask": {"close_dollars": "0.5000"},
         "volume_fp": "0.00", "open_interest_fp": "10.00"}
    df = kalshi._parse_candles([c])
    assert df["close"].iloc[0] == pytest.approx(0.45)


def test_market_row_fixture():
    m = load("kalshi_markets_settled_rich.json")["markets"][0]
    row = kalshi._market_row(m, "KXHIGHNY", "Climate and Weather")
    assert row["key"] == m["ticker"]
    assert row["final_price"] in (0.0, 1.0)
    assert row["closed"] is True
    assert row["series_ticker"] == "KXHIGHNY"
    assert row["volume"] > 0
    # strike metadata kept for later implied-vol work
    assert "cap_strike" in row and "strike_type" in row


def test_cutoff_fixture_shape():
    j = load("kalshi_historical_cutoff.json")
    ts = kalshi._ts(j["market_settled_ts"])
    assert isinstance(ts, pd.Timestamp) and ts.tzinfo is not None
