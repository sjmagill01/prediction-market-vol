"""One-off probe: hit every endpoint we plan to use, save fixtures, print shapes.

Run:  .venv/Scripts/python.exe tests/probe_apis.py
Not a pytest file; kept for reproducibility of the fixtures.
"""
import json
import sys
import time
from pathlib import Path

import requests

FIX = Path(__file__).parent / "fixtures"
FIX.mkdir(exist_ok=True)

S = requests.Session()
S.headers["User-Agent"] = "pm-vol-probe/0.1"


def save(name, obj):
    (FIX / name).write_text(json.dumps(obj, indent=1)[:2_000_000])
    print(f"--- saved {name}")


def jget(url, **params):
    r = S.get(url, params=params or None, timeout=30)
    print(f"GET {r.url} -> {r.status_code}")
    if r.status_code != 200:
        print(r.text[:400])
        return None
    return r.json()


print("=========== POLYMARKET GAMMA /markets ===========")
g = jget("https://gamma-api.polymarket.com/markets", limit=2, offset=0,
         closed="true", order="volumeNum", ascending="false", volume_num_min=100000)
if g:
    save("gamma_markets.json", g)
    m = g[0] if isinstance(g, list) else g
    print("type:", type(g).__name__, "n:", len(g) if isinstance(g, list) else "?")
    print("keys:", sorted(m.keys()))
    for k in ("clobTokenIds", "outcomes", "outcomePrices", "volumeNum", "volume",
              "closed", "startDate", "endDate", "events"):
        v = m.get(k)
        print(f"  {k}: {type(v).__name__} = {str(v)[:160]}")
    tok = None
    try:
        tok = json.loads(m["clobTokenIds"])[0]
    except Exception as e:
        print("clobTokenIds parse issue:", e, repr(m.get("clobTokenIds"))[:200])
    print("first token:", tok)

    print("=========== POLYMARKET CLOB /prices-history (daily, interval=max) ===========")
    if tok:
        h = jget("https://clob.polymarket.com/prices-history", market=tok,
                 interval="max", fidelity=1440)
        if h:
            save("clob_prices_history_daily.json", h)
            hist = h.get("history", [])
            print("daily points:", len(hist), "first:", hist[:2], "last:", hist[-2:])

print()
print("=========== POLYMARKET: high-volume OPEN market for thinning test ===========")
g2 = jget("https://gamma-api.polymarket.com/markets", limit=1, closed="false",
          order="volumeNum", ascending="false", volume_num_min=1000000)
if g2:
    m2 = g2[0]
    print("open mkt:", m2.get("question", "")[:80], "vol:", m2.get("volumeNum"))
    try:
        tok2 = json.loads(m2["clobTokenIds"])[0]
    except Exception:
        tok2 = None
    if tok2:
        now = int(time.time())
        for days in (1, 2, 5, 10):
            h = jget("https://clob.polymarket.com/prices-history", market=tok2,
                     startTs=now - days * 86400, endTs=now, fidelity=1)
            n = len(h.get("history", [])) if h else 0
            exp = days * 1440
            print(f"  1m fidelity, {days}d window: {n} pts (uniform grid would be {exp})")
            if days == 2 and h:
                save("clob_prices_history_1m_2d.json", h)
        for days in (10, 30, 60):
            h = jget("https://clob.polymarket.com/prices-history", market=tok2,
                     startTs=now - days * 86400, endTs=now, fidelity=60)
            n = len(h.get("history", [])) if h else 0
            print(f"  1h fidelity, {days}d window: {n} pts (grid {days*24})")

print()
print("=========== KALSHI base URL probe ===========")
base = None
for b in ["https://api.elections.kalshi.com/trade-api/v2",
          "https://external-api.kalshi.com/trade-api/v2"]:
    try:
        st = jget(b + "/exchange/status")
    except requests.RequestException as e:
        print(b, "->", type(e).__name__, str(e)[:120])
        st = None
    if st is not None:
        print(b, "OK:", st)
        if base is None:
            base = b
            save("kalshi_exchange_status.json", st)
if not base:
    print("NO KALSHI BASE WORKS"); sys.exit(1)
print("using base:", base)

print("=========== KALSHI /series ===========")
sr = jget(base + "/series", limit=5)
if sr:
    save("kalshi_series.json", sr)
    print("top keys:", list(sr.keys()))
    ser = sr.get("series", [])
    print("n series page:", len(ser), "cursor:", str(sr.get("cursor"))[:40])
    if ser:
        print("series[0] keys:", sorted(ser[0].keys()))
        print("series[0]:", {k: str(v)[:60] for k, v in ser[0].items()})

print("=========== KALSHI /markets (settled) ===========")
mk = jget(base + "/markets", limit=5, status="settled")
if mk:
    save("kalshi_markets_settled.json", mk)
    mm = mk.get("markets", [])
    print("n:", len(mm), "cursor:", str(mk.get("cursor"))[:40])
    if mm:
        raw = mm[0]
        print("RAW MARKET:")
        print(json.dumps(raw, indent=1)[:2500])

print("=========== KALSHI candlesticks on one settled market ===========")
if mk and mm:
    # pick one with volume
    cand = max(mm, key=lambda r: float(r.get("volume", 0) or r.get("volume_fp", 0) or 0))
    st_, tk_ = cand.get("series_ticker"), cand["ticker"]
    if not st_:
        st_ = cand.get("event_ticker", "").split("-")[0]
    print("market:", tk_, "series:", st_, "open:", cand.get("open_time"),
          "close:", cand.get("close_time"), "result:", cand.get("result"))
    import datetime as dt
    def ts(s):
        return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    t0, t1 = ts(cand["open_time"]), ts(cand["close_time"])
    cs = jget(f"{base}/series/{st_}/markets/{tk_}/candlesticks",
              start_ts=t0, end_ts=t1, period_interval=1440)
    if cs:
        save("kalshi_candlesticks_1d.json", cs)
        cc = cs.get("candlesticks", [])
        print("n candles:", len(cc))
        if cc:
            print("RAW CANDLE:")
            print(json.dumps(cc[-1], indent=1)[:2000])
    else:
        print("live candlesticks empty/404 -> try historical")
        cs = jget(f"{base}/historical/markets/{tk_}/candlesticks",
                  start_ts=t0, end_ts=t1, period_interval=1440)
        if cs:
            save("kalshi_candlesticks_1d.json", cs)
            print(json.dumps(cs.get("candlesticks", [])[-1:], indent=1)[:2000])

print("=========== KALSHI /historical/cutoff ===========")
co = jget(base + "/historical/cutoff")
if co is not None:
    save("kalshi_historical_cutoff.json", co)
    print("RAW CUTOFF JSON:")
    print(json.dumps(co, indent=1)[:1500])
