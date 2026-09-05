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

## Data pulled

| venue | catalog | 1d bars | 1d mkts | 1h bars | 1h mkts |
|---|---|---|---|---|---|
| polymarket | 100,509 closed, vol ≥ $100k | 49,218 | 298 | 162,696 | 24 |
| kalshi | 103,797 (69,137 resolved multi-day) | 869,099 | 14,048 | 8,106,493 | 5,302 |

The Kalshi catalog is the wide per-series scan of all 2,129
annual/quarterly/monthly/weekly series over both tiers (2026-09-05); 1d bars
cover all closed markets with volume ≥ $10k, 1h bars volume ≥ $50k.

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

### Kalshi (12,731 resolved markets, wide universe)

An earlier 334-market smoke sample gave slope 3.627 dominated entirely by
longshots; the wide universe supersedes it and separates the effects.

- **Test A slope = 2.652 (se 0.087).** The longshot bucket is still
  extreme (ratio ≈ 71 on n = 11,451: the 1-cent tick floors realised
  variance far above a tiny budget), but the mid-p buckets are now well
  populated (n = 99–338) and show ratios **1.75–2.68**: Kalshi carries
  roughly 2x excess movement even away from the boundary. Against the sim
  benchmarks (noise 0.02 → slope 1.80, z-ACF −0.28), a large share of this
  is bid-ask bounce, but not obviously all of it.
- **Test B: Gaussian again wins** (slope 0.866 vs binomial 0.489). The
  striking pattern is the high-p tail: ratio_gauss 4.9 at [0.85,0.95) and
  **21.5 at [0.95,1)** — near-favorites move far more than any diffusion
  allows. Combined with excess kurtosis of 87 (and 5,911 in the longshot
  bucket), resolution risk arrives as jumps.
- **Diagnostics**: z-ACF lag-1 **−0.265** (strong bounce, matching the
  `--noise` sim signature) and λ-ACF +0.230 decaying over ~5 lags (real
  clustering). Kalshi's excess = microstructure bounce + jump arrivals on
  top of a roughly Gaussian bulk.

### Cross-venue read

Both venues reject the binomial p(1-p)/τ rate and are broadly consistent
with the zero-parameter latent-Gaussian rate in the bulk, with more
movement than it allows in the tails (jumps + tick bounce). Polymarket's
excess movement (slope 1.33) looks partly informational (mild bounce, real
λ clustering); Kalshi's (2.65, ~2x even mid-range) mixes strong bid-ask
bounce with jump-dominated resolution, most visible in near-favorites.
1h bars are now on disk for both venues (8.1M Kalshi rows) for a proper
bounce-vs-jump decomposition at the intraday scale; the other sharpening
step is the strike-strip IV construction on Kalshi bracketed series
(metadata already in the catalog).

## Unresolved

- **Global `/historical/markets` is a filterless firehose.** It ignores
  every parameter except limit/cursor (min/max_close_ts, status, order all
  silently no-ops), streams close_time-descending, and held ~5M+ rows at
  the point we abandoned a full walk (4,900 pages ≈ 4.9M markets, mostly
  minutes-lived sports shards). The wide per-series scan replaced it; the
  daily/hourly/custom/one_off series remain unscanned (deliberately —
  they are dominated by sub-day markets that produce no daily bars).
- **Polymarket CLOB drops intraday history for old closed markets.** The
  1h backfill returned data for only 24 of the top 300 markets; daily via
  `interval=max` still works for all. Intraday analysis on Polymarket is
  limited to recently-closed/live markets.
- 1m backfills not run (very heavy; only worth it for case studies).
- Analysis still runs on 1d bars only; the 1h panel (8.1M Kalshi rows) is
  unexploited — intraday bounce-vs-jump decomposition is the next analysis.
