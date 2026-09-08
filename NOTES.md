# NOTES: results in full + API discrepancies

Companion to THEORY.md (the modeling framework) and README.md (the story).
Everything below the API section is the v2 vintage (markets resolved by
2026-09-06, pulled 2026-09-05/06, analyses run 2026-09-07); every number is
registered in `check_docs.py` and recomputed from `results_v2/` on demand.
The v1-vintage findings that were not redone on v2 are kept at the bottom,
clearly labeled.

## API fixes (every place the live APIs differed from their docs)

Fixes 1-13 were found during the v1 build (2026-09-04/05) and remain live
in the `pmdata/` adapters; 14-17 were found during the v2/v5 builds.

### Polymarket

1. **Gamma `limit` silently capped at 100.** Requesting `limit=500` returns
   100 rows with no error. Page size is fixed at 100 (`PAGE=100`).
2. **Gamma offset pagination breaks at ~2000.** `offset=2100` returns HTTP
   422. Replaced with volume-descending cursor pagination: order by
   `volumeNum` desc and pass `volume_num_max` = smallest volume seen on the
   previous page; dedupe on key; break if the max fails to decrease
   (tie guard). This walks the whole universe.
3. **CLOB `prices-history` window cap is absolute, not per-fidelity.** Any
   window over ~15-20 days returns HTTP 400 "interval is too long" at
   *every* fidelity; 15d confirmed OK, 20d fails. Additionally 1m data
   thins slightly (<1%) for windows >= 5d. Chunk sizes set in
   `PM_CHUNK_DAYS`: 1m -> 2d, 1h -> 14d, 1d -> uninterrupted via
   `interval=max&fidelity=1440`.
4. **Gamma JSON-string-encoded fields.** `outcomes`, `outcomePrices`,
   `clobTokenIds` arrive as JSON *strings inside JSON*; parsed defensively
   (malformed -> row dropped, no raise).
5. **`final_price` inferred, not provided.** Accepted only when the last
   outcome price is >= 0.995 or <= 0.005, then rounded to {0,1}. Anything
   else is left unresolved rather than guessed.

### Kalshi

6. **Both base URLs work.** elections API and the external trade API both
   answer `/exchange/status`; first reachable is cached.
7. **`GET /series` ignores `limit` and has no cursor.** It ships ALL series
   in a single page.
8. **Market objects no longer carry `series_ticker`.** Recorded from the
   query context instead.
9. **Numeric fields migrated to string `*_dollars` / `*_fp` (Aug 2026).**
   No integer-cent fields observed live at all. `_num()` reads
   `{name}_dollars` -> `{name}_fp` -> legacy integer `{name}` (/100 when
   `cents=True`) so old fixtures/tiers still parse.
10. **Live `/markets` hides markets settled before the historical cutoff.**
    `GET /historical/cutoff` returns a flat dict of ISO timestamps.
    Pre-cutoff settled markets exist only under `/historical/markets`,
    which (undocumented) accepts `series_ticker` + `cursor` exactly like
    the live endpoint, with all markets `status=finalized`. The universe
    scan therefore hits BOTH tiers per series; bar fetches fall back to the
    historical candlesticks endpoint whenever the live one is empty/404
    for a closed market.
11. **Recency-sorted series are useless for daily bars.** The most recently
    updated series are minutes-lived sports/hourly event shards. Series are
    scanned by frequency rank (annual -> monthly -> weekly -> one_off ...)
    then recency, which yields multi-day price paths.
12. **Candles with no trades have null `price.*`.** Fallback to the
    bid/ask midpoint close for continuity (tested against fixture).
13. **Historical candlesticks return dollar STRINGS under legacy field
    names.** `/historical/markets/{t}/candlesticks` ships
    `price.close = "0.7200"` (no `_dollars` suffix), while the legacy
    integer generation used the same names for cents. `_num(..., cents=True)`
    divided the strings by 100, silently storing every historical-tier bar
    100x too small. Found 2026-09-05 when a strike strip summed to 0.011
    instead of 1. Fix: strings are always dollars; only integer legacy
    fields are cents. Repaired in place and verified by re-fetching random
    keys with the fixed parser. Sub-penny prices are real on Kalshi (the
    `_fp` generation exists for this), so grid fineness alone does not
    imply mis-scaling.

### Found during the v2/v5 builds

14. **Kalshi timestamps mix fractional-second precision, and pandas format
    inference silently NaTs the minority format.** `pd.to_datetime(...,
    errors="coerce")` infers the format from the first rows; every
    timestamp in the other precision coerces to NaT with no warning. The
    v2 selection computed lifetime from those NaTs and silently dropped
    12,966 of the top-50,000 Kalshi markets ($23.3B of volume, including
    PRES-2024) from the bars pull. Found 2026-09-07; `format="ISO8601"` is
    load-bearing everywhere a Kalshi timestamp is parsed, and the missing
    markets were topped up before any analysis was rerun.
15. **Kalshi daily candles are stamped at window START.** A daily candle
    labeled midnight US/Eastern carries the close from the END of that
    window; Polymarket daily "bars" are point samples at 00:00 UTC. Two
    same-label daily closes are therefore ~28 hours apart in wall-clock
    time, and naive same-label alignment manufactures a spurious
    Kalshi-leads-Polymarket signal at the daily frequency (the daily
    cross-lag coefficient 0.204, t = 20.6, is stamp mechanics, not
    information flow). Cross-venue lead-lag is measured on hourly bars
    with explicit wall-clock alignment; daily Kalshi labels are shifted
    +1 day before any cross-venue join.
16. **Polymarket price history truncates at the market's scheduled end
    date.** The daily `interval=max` series stops there: prices set
    between the scheduled end and actual resolution (the 2024
    election-night jump, for instance) never appear. Explicit windows can
    reach past the end date, so the V5 hourly top-up requests end + 2
    days; for markets whose post-end history was never retained, that
    segment is simply absent. This inflates measured cross-venue gaps in
    the election domain and is flagged where it bites.
17. **Polymarket keys are ~77-digit token ids.** Default CSV/parquet
    round-trips parse them as Python big ints that overflow downstream
    arrow ops; every reader passes `dtype=str` for key columns.

## Data pulled (v2 vintage, frozen in analysis/config.py)

| venue | catalog rows | selection | daily-bar files | analysis panel |
|---|---|---|---|---|
| polymarket | 100,828 | top 4,000 closed by volume (>= $100k) | 3,963 | 3,382 resolved mkts / 219,593 moves |
| kalshi | 496,381 | top 50,000 closed (>= $10k vol, >= 2d life) | 50,000 | 27,465 resolved mkts / 708,078 moves |

The Kalshi catalog is the union of a per-series scan (annual -> weekly
frequencies, both live and historical tiers) and the global historical
sweep, deduped on key. Polymarket event tags (Politics, Crypto, ...) cover
2,139 events via one `/events/{id}` call each. Hourly bars exist for the
82 cross-venue bridge pairs on both venues (V5 top-up), 198,224 aligned
hourly observations. Sanity gates: closes in [0,1], tz-aware UTC
timestamps, resolved markets have final_price in {0,1}.

## Findings (v2 vintage)

### Simulation oracle (exact-martingale simulator, 2,000 markets per knob)

The latent-Gaussian simulator provides calibrated benchmarks; every
estimator in the build is oracle-gated on it (run with known parameters,
required to recover them) before touching venue data.

- Clean: budget slope **0.986** (the model-free identity holds); bulk
  power fit gamma 1.57, alpha 1.00 (recovers the probit truth); both ACFs
  ~ 0.
- 2c additive noise: slope **1.207**; z-ACF lag-1 **-0.195** (the MA(1)
  bounce signature); bulk power fit dragged to gamma 1.11, alpha 0.82.
- Staleness: slope unbiased, as the theory predicts.
- Stochvol: lambda-ACF lag-1 **+0.126**, slowly decaying, z-ACF ~ 0; the
  clean signature of genuine intensity clustering.

### Variance budget + diagnostics

| venue | slope (se) | z-ACF lag-1 | lambda-ACF lag-1 |
|---|---|---|---|
| polymarket | **1.190** (0.021) | -0.115 | +0.188 |
| kalshi | **1.432** (0.009) | -0.280 | +0.217 |

Polymarket carries ~19% excess lifetime movement, roughly uniform across
starting-price buckets, with mild bounce and real intensity clustering.
Kalshi's excess is larger and concentrated at the extremes: bucket ratios
run 1.2-1.6x in the bulk but **7.37** at [0, 0.05) and **12.91** at
[0.95, 1), where the tick grid floors the budget. Kalshi's z-ACF of -0.280
exceeds the 2c-noise sim benchmark (-0.195): strong bid-ask bounce on top
of jump-dominated boundary behavior.

### Power-family fit: which [0,1] coordinate

`E[dp^2/dt] = c [p(1-p)]^gamma / tau^alpha` on (p, tau) cell means, bulk =
p in [0.05, 0.95]:

| sample | gamma | alpha |
|---|---|---|
| sim clean (probit truth) | 1.57 | 1.00 |
| sim noise 2c | 1.11 | 0.82 |
| polymarket bulk | **1.51** | **0.53** |
| kalshi bulk | 1.03 | 0.51 |

Polymarket sits between the Wright-Fisher (gamma 1) and logit (gamma 2)
scales, close to the probit walk's effective exponent ~1.6; Kalshi is
dragged toward gamma 1 the way simulated state-independent noise drags the
fit. Both venues run the clock at ~tau^-0.5, slower than the 1/tau of a
constant-information-rate model: information arrives front-loaded relative
to the resolution clock. The v1 vintage (290 Polymarket markets) put PM at
gamma 1.95 / alpha 0.84 and read it as log-odds Brownian motion; the
full-universe estimate walks that back.

### Regime map

Per-cell classification on (price bucket, tau band) via excess stillness
beyond tick-censored diffusion, bipower jump share, and kurtosis (R3
trigger kurt >= 30). Variance-budget shares R1/R2/R3:

- polymarket: **0.613** / 0.064 / 0.323; largest all-R1 rectangle
  p in [0.30, 0.70), tau >= 1d
- kalshi: **0.833** / 0.039 / 0.128; largest all-R1 rectangle
  p in [0.30, 0.85), tau >= 1d

The oracle gates: the clean sim classifies 99.7% R1, the stale sim 99.3%
R2. Note the flip from the v1 vintage (which had PM more diffusive than
Kalshi): on the full universes Kalshi is the more diffusive venue and
Polymarket spends 32% of its budget in R3 jump cells.

### SLV core (collapsed-mixture Kalman filter, R1 segments)

Pooled theta = (c, alpha, eta, phi, s) on the probit-walk state space with
log-AR(1) intensity:

- polymarket: 1,158 segments / 35,299 steps; full ll/obs **1.672**;
  dropping stochastic vol costs **0.649** nats/obs (LR p ~ 0); phi = 0.78.
- kalshi: 1,881 segments / 40,046 steps; full ll/obs **1.142**; dropping
  stochastic vol costs **0.624** nats/obs; phi = 0.71.

Both venues pin the intensity mixing spread s at the 2.0 bound (grid probe
2026-09-07: re-pins at any bound, not a quadrature artifact) and the PIT
is center-humped: the continuous observation layer cannot express the
data's many exact-zero days. Diagnosis confirmed and repaired by the
zero-inflated tick layer below.

### Zero-inflated tick observation layer (V4)

Same state dynamics, observation moved onto the venue tick grid with an
explicit staleness point-mass pi0:

- polymarket: pi0 = **0.284**, s interior at **2.104**, and the zi layer
  is decisively preferred (dropping it costs **0.275** nats/obs, LR p ~ 0).
  Staleness was the missing mass.
- kalshi: the zi fit degenerates (pi0 ~ **0**, s runs to the wider bound);
  tick censoring alone unpins s at **2.557** interior (zi on/off d_ll =
  0.002, indistinguishable). One-cent rounding already absorbs the zeros.

Same symptom, two venue-specific mechanisms; the PIT deciles flatten to
~0.10 each on both venues.

### ITM hazard (R2 cells: frozen prices punctuated by gaps)

Mixture P(k) = pi0 + tick-censored diffusion + hazard x two-sided
geometric gap, per (distance bucket, tau band) cell; |k| >= 2 moves
identify the gap component. Near-boundary Kalshi cells at tau 1-3d show
pi0 = 0.45 with a **21%/day** hazard of a **~24-tick** mean gap, roughly
symmetric in direction (w_away = 0.52): jump-back-to-life risk, the credit
analogy. Polymarket's near-boundary cells gap larger (~31 ticks at tau
3-7d) with most gaps toward the boundary (w_away = 0.23) and lower
frequency. Full per-cell tables in `results_v2/hazard_*.csv`.

### Endgame (R3: arrival x size power laws in pq and tau)

Total variance-rate exponents in pq: polymarket **1.510**, kalshi
**0.614**. On Polymarket the pq dependence lives in move size (size g =
0.95, arrival g = 0.32); on Kalshi it lives in arrival frequency (arrival
g = 0.58, size g ~ 0). Lifetime variance-spend profiles differ sharply:

| band | polymarket | kalshi |
|---|---|---|
| tau <= 1d | 0.112 | 0.535 |
| tau (1,3]d | 0.050 | 0.092 |
| tau (3,7]d | 0.101 | 0.070 |
| tau (7,30]d | 0.140 | 0.114 |
| tau > 30d | 0.207 | 0.150 |
| terminal jump | **0.390** | **0.039** |

Polymarket resolves by leaping: 39% of lifetime variance is the terminal
resolution jump itself. Kalshi grinds: 53% of lifetime variance is spent
trading in the final day, and the terminal jump carries only 4%.

### Walk-forward horse race (V4)

Train on the first 50% then 75% of markets by resolution date, score
log P(next tick move) on the venue grid; block-bootstrap CIs on deltas.
117,252 scored observations (PM) / 65,651 (Kalshi). Delta vs the plain
composite, nats/obs (positive = better):

| model | polymarket | kalshi |
|---|---|---|
| composite-zi | **+0.069** | **+0.076** |
| composite | 0 | 0 |
| global-slv | -4.101 | **+0.211** |
| const-lam | -1.572 | -0.316 |
| bucket | -1.122 | -0.665 |
| garch | -0.733 | -0.826 |

The zi observation layer wins on both venues (CIs exclude zero). The
regime split is essential on Polymarket (one pooled SLV degenerates) but
not on Kalshi, where a single global SLV wins outright and leads in every
regime: with 8x more markets per fit, the latent-intensity mixture spans
the frozen and jumpy cells on its own. GARCH, the imported equity default,
loses by 0.73-0.83 nats/obs everywhere.

### Budget-consistency audit (V4, MC integration of the fitted composite)

Roll the fitted composite forward from starts nearest tau = 30/14/7/3d
(p0 in [0.05, 0.95]); ratio = remaining variance / p0(1-p0), ratio of
sums, vs the realized continuation of the same markets:

| start | PM full | PM zi | PM c-lam | PM real | K full | K zi | K c-lam | K real |
|---|---|---|---|---|---|---|---|---|
| tau ~ 30d | 1.157 | 1.174 | 1.163 | 1.174 | 1.375 | 1.537 | 1.335 | 1.597 |
| tau ~ 14d | 1.077 | 1.085 | 1.077 | 1.158 | 1.112 | 1.205 | 1.096 | 1.329 |
| tau ~ 7d | 1.041 | 1.044 | 1.041 | 1.128 | 1.073 | 1.117 | 1.064 | 1.204 |
| tau ~ 3d | 1.021 | 1.023 | 1.021 | 1.060 | 1.032 | 1.056 | 1.029 | 1.124 |

The model engines land on the budget and near the realized ratios at every
horizon; an earlier (pre-zi) vintage overshot 2-6x, and removing the heavy
intensity tail via the tick observation layer is what closed it. From
these mid-life states, 99%+ of remaining variance rides the jump/terminal
channel in both model and data (PM 30d jump share 0.995), so this audits
totals, not path texture. In-sample by design.

### Cross-venue bridge (V5, 82 matched pairs)

82 hand-matched contract pairs, 64 exact: 47 fed_decision, 14
fed_chair_nom, 8 btc_monthly, 6 btc_yearly, 7 election. 9,341 overlapping
pair-days.

- **Gaps**: mean |gap| 2.4c, median 1.1c; exact pairs 1.9c vs approximate
  3.6c; 8.6% of pair-days differ by > 5c. Within-pair AR(1) rho = 0.860,
  half-life **4.58 days**. Election-domain gaps (9.7c mean) are inflated
  by the Polymarket end-date truncation (API fix 16).
- **Paired budget**: Kalshi spends **1.457x** Polymarket's variance on the
  same events (median ratio of ratios; Kalshi higher on 72.0% of pairs).
  The venue effect survives full composition control.
- **Paired power fit**: on the matched panel Kalshi gamma rises to
  **1.344** (from 1.03 venue-wide) vs Polymarket **1.531**: same events on
  both venues walk far more alike, with residual bounce-flattening on
  Kalshi.
- **Lead-lag (hourly, 198,224 aligned obs)**: lagged PM changes predict
  Kalshi changes with coefficient **0.239** (t = 57.9); the reverse is
  0.018. Contemporaneous hourly xcorr 0.244. The daily-frequency reverse
  result (K predicts PM, 0.204, t = 20.6) is stamp mechanics (API fix 15),
  not information flow.

## v1-vintage findings (2026-09-04/05 build, not redone on v2)

Kept for the record; these numbers come from the smaller v1 universe
(298 PM / 14,048 Kalshi daily-bar markets) and are NOT checked by the v2
claims registry.

### Intraday bounce-vs-jump decomposition (1h bars)

Kalshi (5,096 markets, 7.34M consecutive-hour increments): ~half of hourly
trade RV is bid-ask bounce, with quote-based (1 - RV_mid/RV_trade,
0.32-0.58) and Roll (0.28-0.66) instruments agreeing in the bulk; jumps
carry 34-69% of quote-side (bounce-free) variance, peaking in coin-flip
buckets; signature ratio RV(1h)/RV(1d) median 3.51; 69% of hourly trade
increments are exactly zero. Polymarket (24 markets, indicative): Roll
bounce ~0 in the bulk, jump share high (0.74): the excess is
informational/jumpy, not bounce. Consistent with the v2 daily story.

### Strike-strip implied vol (bracketed events as digital strips)

One event-day of member-market closes is a full risk-neutral distribution;
moments give a genuinely forward-looking IV (THEORY.md construction 3).
6,806 valid event-days, 179 events, 10 series. Both external anchors
validate: S&P end-of-year strips imply ~17% annualized vol
(VIX-consistent) and 10Y Treasury weekly strips imply ~100bp/sqrt-yr
normal vol (the swaption convention and level). Oil, gas, CPI,
unemployment, payrolls strips all land in plausible ranges. This is the
forward-looking IV that no single-market filtered intensity can give; it
prices off order books alone. Total-probability gate [0.7, 1.3] on raw
strip sums (this gate is what exposed API fix 13); running-extremum series
excluded.

## Unresolved

- **Global `/historical/markets` is a filterless firehose.** It ignores
  every parameter except limit/cursor (min/max_close_ts, status, order all
  silently no-op) and streams close_time-descending; the library sweep
  walks it with volume/lifetime floors applied client-side.
- **Polymarket CLOB drops intraday history for old closed markets.**
  Hourly backfills only cover recently-closed markets; daily via
  `interval=max` works for all.
- 1m backfills not run (very heavy; only worth it for case studies).
- The intraday decomposition treats tick rounding as bounce (Roll picks up
  negative autocovariance from the 1-cent grid itself); separating grid
  effects from spread effects needs sub-tick quote data we don't have.
- Strike-strip IV uses daily closes across member markets that need not be
  simultaneous quotes; a same-timestamp order-book snapshot would be
  cleaner. Rebuilding strips and calendar pairs on the v2 vintage is the
  natural next layer (see README).
