# pm-vol: prediction-market volatility pipeline

Pulls daily and intraday prediction-market prices from Polymarket and Kalshi
into local parquet, exposes them through DuckDB views, and runs a first
analysis comparing realised variance to the martingale budget p(1-p).

- **THEORY.md**: the modeling framework (variance budget, admissible local-vol
  models, why verbatim Heston is inadmissible, implied-vol constructions).
- **NOTES.md**: empirical findings and every place the live APIs differed from
  their documentation.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
```

Python 3.11+. No API keys required; both venues are read publicly.

## Usage

```bash
# 1. build the universe (catalog of markets)
python -m pmdata.cli universe --venue polymarket --closed --min-volume 100000
python -m pmdata.cli universe --venue kalshi --status settled --max-series 60

# 2. backfill bars (1d first; 1m/1h are much heavier)
python -m pmdata.cli backfill --venue polymarket --freq 1d --limit 300
python -m pmdata.cli backfill --venue kalshi --freq 1d

# 3. inspect via DuckDB
python -m pmdata.cli sql "select venue, freq, count(*) bars, count(distinct key) mkts from bars group by 1,2"

# 4. re-pull all live markets at every freq (cron target)
python -m pmdata.cli update

# 5. analysis (see THEORY.md for what the tests mean)
python analysis/vol_check.py --simulate 600 --plot     # calibrated benchmark
python analysis/vol_check.py --venue polymarket --plot
python analysis/vol_check.py --venue kalshi --plot

# tests
python -m pytest tests -q
```

## Layout

```
pmdata/config.py      endpoints, rate limits, FREQS, chunk sizes, paths
pmdata/http.py        shared Session: retries w/ backoff, Retry-After, ~8 rps
                      token bucket per host, 404 -> None
pmdata/store.py       parquet writers (incremental, dedupe on ts), checkpoints,
                      DuckDB `bars` + `catalog` views
pmdata/polymarket.py  Gamma universe + CLOB prices-history -> resampled OHLC
pmdata/kalshi.py      series/markets universe (live + historical tiers),
                      candlesticks with dual-generation _num() parsing
pmdata/cli.py         universe / backfill / update / sql
analysis/vol_check.py tests A/B + stochvol diagnostics + simulator
tests/                parser tests on captured fixtures; sim-oracle tests

data/catalog/{venue}.parquet          one row per tracked outcome
data/bars/{venue}/{freq}/{key}.parquet  ts,open,high,low,close,bid,ask,
                                        volume,open_interest (probs in [0,1])
data/state/{venue}_{freq}.json        finished closed markets (skipped on rerun)
```

## Caveats

- Polymarket bars are resampled from a single price series: no true OHLC
  inside a bar; bid/ask/volume/open_interest are NaN for that venue.
- Kalshi bars with no trades fall back to the bid/ask midpoint close.
- The Kalshi smoke catalog scans a subset of series (long-horizon frequencies
  first); it is not the full exchange history. The complete settled universe
  requires paging the global `/historical/markets` endpoint.
- 1-cent ticks floor measurable variance; extreme-p variance estimates are
  contaminated by tick bounce (see THEORY.md, "Honest limitations").
- Resolution time is proxied by the last observed bar.
