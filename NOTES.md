# NOTES: empirical findings + API discrepancies

Companion to THEORY.md (the modeling framework). Everything here was observed
live during the 2026-09-04 build; fixtures in `tests/fixtures/` are captures
from the same session.

## API fixes (every place the live APIs differed from their docs)

### Polymarket

1. **Gamma `limit` silently capped at 100.** Requesting `limit=500` returns
   100 rows with no error. Page size is fixed at 100 (`PAGE=100`).
2. **Gamma offset pagination breaks at ~2000.** `offset=2100` returns HTTP
   422. Replaced with volume-descending cursor pagination: order by
   `volumeNum` desc and pass `volume_num_max` = smallest volume seen on the
   previous page; dedupe on key; break if the max fails to decrease
   (tie guard). This walks the whole universe.
3. **CLOB `prices-history` window cap is absolute, not per-fidelity.** The
   directive assumed ~30-day chunks at fidelity=60; in fact any window over
   ~15–20 days returns HTTP 400 "interval is too long" at *every* fidelity.
   15d confirmed OK, 20d fails. Additionally 1m data thins slightly (<1%)
   for windows ≥5d. Chunk sizes set in `PM_CHUNK_DAYS`: 1m → 2d, 1h → 14d,
   1d → uninterrupted via `interval=max&fidelity=1440`.
4. **Gamma JSON-string-encoded fields.** `outcomes`, `outcomePrices`,
   `clobTokenIds` arrive as JSON *strings inside JSON*; parsed defensively
   (malformed → row dropped, no raise).
5. **`final_price` inferred, not provided.** Accepted only when the last
   outcome price is ≥0.995 or ≤0.005, then rounded to {0,1}. Anything else
   is left unresolved rather than guessed.

### Kalshi

6. **Both base URLs work.** elections API and the external trade API both
   answer `/exchange/status`; first reachable is cached.
7. **`GET /series` ignores `limit` and has no cursor.** It ships ALL series
   (13,836) in a single page.
8. **Market objects no longer carry `series_ticker`.** Recorded from the
   query context instead.
9. **Numeric fields migrated to string `*_dollars` / `*_fp` (Aug 2026).**
   No integer-cent fields observed live at all. `_num()` reads
   `{name}_dollars` → `{name}_fp` → legacy integer `{name}` (÷100 when
   `cents=True`) so old fixtures/tiers still parse.
10. **Live `/markets` hides markets settled before the historical cutoff.**
    `GET /historical/cutoff` returns a flat dict of ISO timestamps
    (`market_settled_ts` = 2026-07-06). Pre-cutoff settled markets exist
    only under `/historical/markets`, which (undocumented) accepts
    `series_ticker` + `cursor` exactly like the live endpoint, with all
    markets `status=finalized`. The universe scan therefore hits BOTH tiers
    per series; bar fetches fall back to the historical candlesticks
    endpoint whenever the live one is empty/404 for a closed market.
11. **Recency-sorted series are useless for daily bars.** The most recently
    updated series are minutes-lived sports/hourly event shards (a
    recency-only scan produced 653 markets, all < 5 days of life). Series
    are scanned by frequency rank (annual → monthly → weekly → one_off …)
    then recency, which yields multi-day price paths.
12. **Candles with no trades have null `price.*`.** Fallback to the
    bid/ask midpoint close for continuity (tested against fixture).

## Data pulled (smoke scope: 1d bars only)

| venue | catalog | bars (1440m) | markets with bars |
|---|---|---|---|
| polymarket | 100,509 closed, vol ≥ $100k | 49,218 | 298 |
| kalshi | 1,009 (incl. 334 resolved multi-day) | 43,195 | 431 |

Sanity: all closes in [0,1] (0 violations), timestamps tz-aware UTC,
resolved markets have final_price ∈ {0,1}.

## Findings

### Simulation oracle (600 markets, latent-Gaussian truth)

- Clean: test A slope **0.991** (budget holds); Gaussian local slope 1.062
  vs binomial 0.600 (binomial overstates tail risk as designed); bucket
  ratios flat ~0.85–1.15 in the bulk; both ACFs ≈ 0.
- `--noise 0.02` (bid-ask bounce): slope A → **1.803**; signed z-ACF lag-1
  **−0.276** (MA(1) reversal); λ-ACF mildly *positive* (+0.10 — squares of
  an MA(1) cluster; this corrected an earlier wrong sign expectation and is
  why the signed z-ACF exists).
- `--stale 0.5`: slope A → **1.017** — staleness does not bias the budget
  test, as the theory predicts.
- `--stochvol`: λ-ACF **+0.278**, slowly decaying, with z-ACF ≈ 0 — the
  clean signature of genuine vol clustering.

### Polymarket (298 resolved markets)

- **Test A slope = 1.332 (robust se 0.096)**: ~33% more lifetime variance
  than the martingale budget — significant excess movement
  (Augenblick-Rabin direction), part of it bounce (see ACF).
- **Test B: latent-Gaussian ≫ binomial** (slope 0.777 vs 0.452). The
  Gaussian rate is the better local model, mildly *over*-predicting
  movement in the bulk.
- **Tails are jump-dominated.** Kurtosis of z is enormous in extreme-p
  buckets (≈589 in [0,0.05), 30+ broadly): longshots sit still and then
  gap, exactly the predicted diffusion-failure pattern.
- **Diagnostics**: z-ACF lag-1 −0.085 (mild bounce), λ-ACF +0.223 and
  decaying — Polymarket shows *real* information-flow clustering on top of
  the local model, i.e. a layer-3 (SLV) effect worth fitting.

### Kalshi (334 resolved multi-day markets)

- **Test A slope = 3.627 (se 0.690)** — but this is a composition artefact:
  310 of 334 markets are longshots (p0 < 0.05), where the budget is tiny
  and the 1-cent tick floors realised variance far above it (bucket ratio
  ≈ 103 in [0,0.05)). Outside the longshot bucket the sample is thin
  (n = 1–9 per bucket), so the aggregate slope should not be read as a
  venue-wide risk premium.
- **Test B: Gaussian again wins** (slope 0.893 vs binomial 0.510), with
  upward deviation in extreme buckets (ratio_gauss ≈ 12.8 at [0.95,1)) —
  tick censoring + jumps, as flagged in THEORY.md §5.
- **Diagnostics**: z-ACF lag-1 **−0.260** (strong bid-ask bounce — matches
  the `--noise` sim signature almost exactly) and λ-ACF +0.315. Kalshi's
  excess movement is substantially microstructure, not belief updating.

### Cross-venue read

Both venues reject the binomial p(1-p)/τ rate and are broadly consistent
with the zero-parameter latent-Gaussian rate in the bulk, with more
movement than it allows in the tails (jumps + tick bounce). Polymarket's
excess movement looks partly informational (mild bounce, real λ
clustering); Kalshi's is mostly microstructural (strong bounce, longshot
tick censoring). Next steps that would sharpen this: intraday (1m/1h)
backfill for bounce-vs-jump separation, and the strike-strip IV
construction on Kalshi bracketed series (metadata already in the catalog).

## Unresolved

- Kalshi complete settled universe: the smoke catalog scans a subset of
  series; paging the global `/historical/markets` endpoint (no
  series_ticker filter) would recover the full history.
- 1m/1h backfills not run (deliberately out of smoke scope; heavy).
- Kalshi non-longshot sample too thin for bucket-level inference; needs
  the full historical universe above.
- Repo is git-initialized with nothing committed (awaiting your call;
  parent rsumplay repo is local-only).
