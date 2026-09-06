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
13. **Historical candlesticks return dollar STRINGS under legacy field
    names.** `/historical/markets/{t}/candlesticks` ships
    `price.close = "0.7200"` (no `_dollars` suffix), while the legacy
    integer generation used the same names for cents. `_num(..., cents=True)`
    divided the strings by 100, silently storing every historical-tier bar
    100x too small -- 12,244 of 14,048 daily keys and 4,878 of 5,302 hourly
    keys. Found 2026-09-05 when the KXINXY strike strip summed to 0.011
    instead of 1. Fix: strings are always dollars; only integer legacy
    fields are cents. Repair: in-place x100 on keys with max(close) <= 0.011
    and sub-half-cent grid values (the two rules agree exactly with the
    pre/post-cutoff split; 21 pre-cutoff keys with live-scale bars and 104
    post-cutoff true dead longshots correctly left alone). Verified by
    re-fetching 20 random keys with the fixed parser: all match the
    repaired files bar-for-bar. Sub-penny prices are real on Kalshi
    (the `_fp` generation exists for this), so grid fineness alone does
    not imply mis-scaling.

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

### Polymarket (290 resolved markets; 298 have 1d bars, 8 lack a final price)

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

### Kalshi (12,731 resolved markets, wide universe, post scale-repair)

Two earlier vintages are superseded: the 334-market smoke sample (slope
3.627, all longshots) and the first wide-universe run (slope 2.652,
longshot ratio 71), which was contaminated by the 100x historical-bar
scaling bug (API fix #13) — most "longshots" were mis-scaled ordinary
markets. Corrected numbers:

- **Test A slope = 2.405 (se 0.031).** The bucket profile is now roughly
  symmetric: mid-p buckets (n = 756–2,971) sit at ratios **1.99–2.57**,
  with both extremes elevated (12.4 at [0,0.05), 11.7 at [0.95,1) —
  tick-floored budgets). Kalshi carries ~2x excess movement across the
  whole range. Against the sim benchmarks (noise 0.02 → slope 1.80,
  z-ACF −0.28), a large share of this is bid-ask bounce, but not all.
- **Test B: Gaussian again wins** (slope 0.710 vs binomial 0.403), flat-ish
  ratios 1.2–2.9 in the bulk, blowing up only at the extremes (ratio_gauss
  11.3 / 14.9) — boundary movement exceeds any diffusion. Excess kurtosis
  14–58 in the bulk, 300–460 at the extremes: resolution risk arrives as
  jumps.
- **Diagnostics**: z-ACF lag-1 **−0.271** (strong bounce, matching the
  `--noise` sim signature) and λ-ACF +0.180 decaying over ~5 lags (real
  clustering). Kalshi's excess = microstructure bounce + jump arrivals on
  top of a roughly Gaussian bulk.

### Cross-venue read

Both venues reject the binomial p(1-p)/τ rate and are broadly consistent
with the zero-parameter latent-Gaussian rate in the bulk, with more
movement than it allows in the tails (jumps + tick bounce). Polymarket's
excess movement (slope 1.33) looks partly informational (mild bounce, real
λ clustering); Kalshi's (2.41, ~2x even mid-range) mixes strong bid-ask
bounce with jump-dominated resolution, most visible in near-favorites.
The two sharpening steps, both done below: the intraday bounce-vs-jump
decomposition on the 1h bars, and the strike-strip IV construction on
Kalshi bracketed series.

### Power-family fit: the best [0,1] coordinate (see THEORY.md, transform view)

`E[dp^2/dt] = c [p(1-p)]^gamma / tau^alpha` fitted on (p, tau) cell means:

| sample | gamma | alpha | reading |
|---|---|---|---|
| sim clean (probit truth) | 1.64 | 1.02 | probit ~ effective power 1.6, clock recovered |
| sim noise 0.02 | 0.68 | 0.32 | bounce flattens both exponents |
| polymarket, all p | 1.84 | 0.61 | near-logit scale |
| polymarket, bulk [.05,.95] | **1.95** | **0.84** | **log-odds Brownian, tau-scaled clock** |
| kalshi, all p | 0.53 | 0.31 | bounce-flattened |
| kalshi, bulk [.05,.95] | 0.81 | 0.40 | bounce-flattened (matches noise sim) |

Polymarket's bulk is strikingly close to a logit-Brownian with a
resolution-scaled clock: the best "custom-to-[0,1]" transform there is the
log-odds. Kalshi's fit is flattened exactly the way simulated additive
noise flattens it (0.68/0.32) -- a third, independent confirmation that
Kalshi's excess is microstructural. (A pre-repair vintage showed
full-sample gamma = 3.0, a "longshot collapse" that was entirely the 100x
scaling bug parking ordinary markets at p ~ 0.005.)

### Intraday bounce-vs-jump decomposition (1h bars, analysis/intraday_decomp.py)

Three independent instruments, pooled by p bucket.

Kalshi (5,096 markets, 7.34M consecutive-hour increments, post
scale-repair):

- **~half of hourly trade RV is bid-ask bounce**, and the two instruments
  agree: quote-based (1 - RV_mid/RV_trade) 0.32-0.58 in the bulk,
  Roll (-2*gamma1/var) 0.28-0.66. The longshot bucket is the exception
  (quote bounce 0.07): sub-penny prices leave little room for spread.
- **Jumps carry 34-69% of quote-side variance** (bipower on midpoints,
  bounce-free), peaking at 0.69 in the 0.45-0.55 bucket: coin-flip
  markets move by discrete news arrivals.
- Signature ratio RV(1h)/RV(1d): median **3.51** (IQR 1.9-8.5). Hourly RV
  is ~3.5x daily RV, i.e. bounce dominates at high frequency; the daily
  bars used in vol_check are far less contaminated but not clean (daily
  z-ACF -0.27).
- 69% of hourly trade increments are exactly zero (tick grid + quiet hours).

Polymarket (24 markets with hourly history -- indicative only):

- No bid/ask on this venue, so only Roll + bipower apply. Roll bounce ~0 in
  the bulk (+/-0.1 sampling noise), 0.67 in the longshot bucket (tick
  rounding). Jump share is high (0.74 overall). Signature median 1.27.
- Consistent with the daily story: Polymarket's excess movement is mostly
  informational/jumpy, not bounce.

### Strike-strip implied vol (analysis/strike_strip.py)

Construction 3 from THEORY.md sec 4: bracketed Kalshi events are digital
strips, so one event-day of member-market closes is a full risk-neutral
distribution of the underlying; moments give a genuinely forward-looking
IV. 'between' markets are disjoint bins plus greater/less tails; ladder
events (all-'greater' chains) are survival functions differenced into
bins. Days whose raw probabilities sum outside [0.7, 1.3] are dropped as
stale/incomplete (this gate is what exposed the 100x bug); the rest are
renormalised. Running-extremum series (BTC max/min) are excluded: a
period max is not a terminal value.

6,806 valid event-days, 179 events, 10 series (med total prob ~1.00):

| series | kind | med IV | reading |
|---|---|---|---|
| KXINXY (S&P eoy) | prop | **0.167** | equity IV ~17%: lands on VIX-range |
| KXNASDAQ100Y | prop | 0.132 | same ballpark |
| KXWTIW (weekly) | prop | 0.464 | oil vol, slightly rich vs ~35% OVX |
| KXAAAGASM / W | prop | 0.156 / 0.100 | retail gas: sticky, low vol |
| KXTNOTEW (10Y) | norm | **1.003** | ~100bp/sqrt-yr: swaption normal-vol range |
| KXCPIYOY / KXCPI | norm | 0.72 / 0.60 | inflation uncertainty in %-pts |
| KXU3 | norm | 0.636 | unemployment %-pts |
| KXPAYROLLS | norm | ~119k | jobs per sqrt-yr (monthly-change strips) |

The two external anchors both validate: S&P annual strips imply ~17%
annualised vol (VIX-consistent) and 10Y Treasury weekly strips imply
~100bp normal vol (the swaption convention and level). This is the
forward-looking IV the lambda_imp construction cannot give, and it prices
off order books alone -- no realised movement enters.

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
- The intraday decomposition treats tick rounding as bounce (Roll picks up
  negative autocovariance from the 1-cent grid itself); separating grid
  effects from spread effects needs sub-tick quote data we don't have.
- Strike-strip IV uses daily closes across member markets that need not be
  simultaneous quotes; the total-prob gate catches gross staleness but a
  same-timestamp order-book snapshot would be cleaner.
