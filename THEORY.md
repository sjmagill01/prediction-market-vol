# How to think about the volatility of prediction-market prices

This note records the modeling framework behind the analyses in this repo: what
"volatility" can even mean for an asset that is a probability, which classes of
models are admissible, why an equity-style Heston model cannot be transplanted
verbatim, and what "implied volatility" means here. It is the design rationale
for the tests in the code; the empirical results live in NOTES.md.

## 1. The object is a bounded, absorbed martingale

A prediction-market price is (up to fees and risk premia) a conditional
probability: `p_t = E[1_event | F_t]`. Three structural facts follow, and each
one breaks a default habit imported from equities.

**(i) Log returns are the wrong coordinate.** Near p = 0 log returns explode;
near p = 1 they compress. The natural increment is the price difference
`dp`, and every test in this project works in dp.

**(ii) Volatility must be state-dependent, by construction.** A price at 0.02
cannot have the same dp-volatility as one at 0.50: the boundary forbids it.
"What is the vol?" is not a well-posed question; the well-posed question is
*which function of (p, tau) is the local variance rate*, where tau is time to
resolution.

**(iii) The clock is time-to-resolution, not calendar time.** The price must
reach 0 or 1 at resolution. Any market still near 0.5 close to its deadline
must move violently; a market pinned at 0.98 must barely move. Stationary
calendar-time vol models (GARCH on returns, etc.) have no mechanism for either.

## 2. The variance budget: the market quotes its own implied variance

For any martingale absorbed at 0/1, squared increments telescope into the
variance of the terminal Bernoulli outcome:

```
E[ sum_t dp_t^2  +  (final - p_last)^2 ]  =  p_0 (1 - p_0)
```

This identity is *model-free*: it holds under any dynamics whatsoever
(stochastic vol, jumps, irregular or stale sampling) as long as p is a
martingale. Two consequences:

- `p(1-p)` is the market's own **model-free implied integrated variance** for
  the rest of its life: the exact analog of a variance-swap strike, quoted by
  the price itself with no model and no inversion.
- Regressing realised lifetime variance on `p0(1-p0)` (test A) is literally a
  **realised-vs-implied variance** comparison, and slope minus one is a
  variance risk premium for prediction markets. Slope > 1 means excess
  movement: bid-ask bounce, overreaction, non-Bayesian updating (this is the
  Augenblick-Rabin excess-movement idea). Slope < 1 cannot happen for a true
  martingale, so it flags risk premia contaminating the price-as-probability
  reading (favorite-longshot bias) or unresolved-sample artefacts.

Staleness does **not** bias this test: a martingale sampled at irregular times
keeps the identity. The simulation confirms it (`--stale 0.5` leaves the slope
at ~1.0 while halving the number of distinct prints).

## 3. The three-layer hierarchy of admissible models

Because the budget fixes the *total*, models can only differ in **where in
(p, tau) space the budget is spent** and in **how the spending rate
fluctuates**. That gives a clean hierarchy:

| Layer | Object | Test in this repo |
|---|---|---|
| 1. Budget (model-free) | `E[sum dp^2] = p0(1-p0)` | Test A |
| 2. Local leverage function | variance rate as f(p, tau) | Test B, three candidates |
| 3. Vol-of-vol dynamics | fluctuations of the information-flow intensity | kurtosis + ACF diagnostics |

### Layer 2: candidate local models

With tau = days to resolution:

1. **Binomial, `p(1-p)/tau`.** Treats each day as a miniature resolution.
   Spends variance heavily in the tails.
2. **Latent-Gaussian, `phi(Phi^-1(p))^2 / tau`.** Beliefs driven by a Brownian
   signal: `p_t = Phi(X_t / sqrt(v_t))` with remaining signal variance
   v_t proportional to tau. Ito gives `Var(dp) = phi(Phi^-1(p))^2 (dt/tau)`.
   Note this model has **no free volatility parameter**: martingality plus the
   one-factor Gaussian structure fully determine the rate. As p -> 0 the rate
   falls like `p^2 * Phi^-1(p)^2`, much faster than p(1-p) ~ p: longshots
   should barely move day to day.
3. **tau-free, `c * p(1-p)` (Wright-Fisher-style).** Constant information
   intensity, ignoring the resolution clock. It has a free scale c, so only
   the *flatness* of its bucket ratios is testable. Comparing it to
   candidate 2 asks whether tau matters at all.

### The transform view: which coordinate makes increments homoskedastic?

Black-Scholes does not model prices directly; it picks the coordinate (log)
in which increments are iid Gaussian. The same move is available on [0,1]:
choose g so that y = g(p) has state-independent variance. By Ito, a local
rate Var(dp) = f(p) dt is flattened by any g with g'(p) = f(p)^{-1/2}, so
*choosing a transform and choosing a local-vol function are the same choice*:

| local rate f(p) | flattening transform g |
|---|---|
| p(1-p) | arcsin(2p-1) (Wright-Fisher angular scale) |
| p^2(1-p)^2 | logit(p) = log(p/(1-p)) |
| phi(Phi^-1(p))^2 | probit(p) = Phi^-1(p) |

Two cautions. First, the transform must stretch the boundaries to infinity:
g' must blow up as p -> 0, 1, else the transformed process is still bounded
and still forced to be heteroskedastic. arctan(p - 1/2) fails this test --
its derivative is *finite* everywhere on [0,1] (arctan tames infinite
domains, but [0,1] needs the opposite), so it is essentially a linear
rescaling near the boundary and fixes nothing. logit and probit are the
correct analogues of log. Second, g(p) is not a martingale even when p is
(Jensen); like log-price under BS it acquires a state-dependent drift, so
the transform is a variance-stabilising coordinate, not a new price.

This suggests fitting the *power family*

```
E[dp^2/dt] = c * [p(1-p)]^gamma / tau^alpha
```

which nests the candidates (gamma=1, alpha=1: binomial; gamma=1, alpha=0:
Wright-Fisher) and brackets the probit rate, whose boundary behaviour
phi(Phi^-1(p))^2 ~ p^2 * 2ln(1/p) is "logit plus a log correction"
(effective gamma ~ 1.6 over the sampled range in simulation, exactly what
the power fit recovers on the clean sim: gamma 1.57, alpha 1.00). The
fitted gamma answers "which custom transform is best" directly:
gamma-hat = 2 says log-odds Brownian motion, gamma-hat = 1 says angular
scale. The fit is also a bounce detector: additive noise is
state-*independent*, so it drags both exponents down (simulated 2c noise:
gamma 1.11, alpha 0.82 in the bulk).

The predicted failure pattern is diagnostic: if the Gaussian form fits the
bulk but the extreme-p buckets show *more* movement than it allows, news is
arriving as **jumps** (court rulings, poll releases) rather than continuous
diffusion; longshots then move only when lumps land.

### Layer 3: where Heston lives, and why verbatim Heston is inadmissible

Heston for equities puts a CIR variance process on returns with **no
state-dependence on the price level** (log-price scale invariance). Here that
is structurally forbidden:

- a variance rate that does not vanish at p = 0, 1 leaks probability out of
  [0,1];
- a free long-run variance level violates the budget, which fixes total
  expected variance ex ante at `p0(1-p0)`;
- CIR variance can wander indefinitely, but here the information clock must
  **complete**: remaining signal variance v_t must hit 0 at resolution
  (integrability of the intensity into T), otherwise the price cannot be
  absorbed at 0/1.

So any admissible stochastic-vol model is forced into a
**stochastic-local-volatility (SLV) form**:

```
Var(dp_t) = phi(Phi^-1(p_t))^2 * lambda_t dt ,      lambda_t = -d log v_t
```

The local factor is pinned by the boundary and martingale structure; all the
Heston-ness (mean reversion, vol-of-vol, leverage correlation) lives in the
dynamics of the **information-flow intensity** lambda_t. In this precise sense
prediction markets are *more constrained* than equities, which is exactly why
the model-free tests are so clean.

### Detecting whether a third layer is needed

Define the model-normalised series `lambda_hat = dp^2 / [phi(Phi^-1(p))^2/tau]`
and the signed standardised move `z = dp / sqrt(...)`. Then, verified against
the simulator's knobs:

| Signature | z (signed) ACF lag-1 | lambda ACF | interpretation |
|---|---|---|---|
| clean martingale | ~0 | ~0 | layer 2 suffices |
| additive noise (`--noise`) | strongly negative | mildly positive | bid-ask bounce (MA(1) in dp; its squares cluster) |
| staleness (`--stale`) | ~0 | ~0 / slightly negative | thinning only; test A unbiased |
| stochastic vol (`--stochvol`) | ~0 | positive, slowly decaying | genuine vol clustering: fit lambda_t dynamics |
| jumps | ~0 | ~0 | fat tails in kurtosis table without persistence |

The correct order of operations is therefore: run the diagnostics first, and
fit a Heston-type lambda_t process only if the lambda ACF demands it.

## 4. Implied volatility: three constructions

**(1) Model-free integrated implied variance = `p(1-p)`.** Already discussed;
test A is the realised-vs-implied comparison and its slope is the variance
risk premium. This is the prediction market's built-in VIX, per market.

**(2) Filtered information intensity: NOT an implied vol.** Once information
flow is allowed to be non-uniform, `lambda_t` is the free object, and

```
lambda_hat = local variance of dp  /  phi(Phi^-1(p))^2
```

strips out the mechanical (p, tau) state-dependence the way Black-Scholes IV
strips out moneyness/maturity, leaving a "how fast is uncertainty resolving"
number comparable across markets, categories, and time. But it is a
**realised/filtered** quantity, not an implied one, and this repo never calls
it IV: Black-Scholes IV exists because *two* instruments (an option and its
underlying) are priced on one filtration and the option price can be
inverted; lambda_hat is extracted from a single instrument's own past moves
and is backward-looking. Genuinely forward-looking constructions need a
second instrument: strike strips (construction 3) or calendar pairs
(a by-T1 vs by-T2 contract on the same event implies a hazard rate over
[T1, T2]).

**(3) Strike-based IV from bracketed markets.** Kalshi ranged markets (CPI
above x%, S&P close in [a,b], Fed brackets) are digital-option strips on a
common underlying: prices across strikes at one instant are the risk-neutral
CDF of the underlying, from which an implied distribution and a genuine
forward-looking implied vol follow. The catalog deliberately stores
`floor_strike`, `cap_strike`, `strike_type`, and `event_ticker` (Kalshi) and
`event_id`/`event_slug` (Polymarket) so this construction is possible without
re-pulling. Built-in validation: S&P brackets against VIX-range equity IV and
Treasury brackets against swaption-style normal vol (see NOTES.md).
Running-extremum series (BTC max/min over a period) are excluded: a running
maximum is not a terminal value, so its distribution has no IV interpretation.

## 5. Honest limitations

- **Tick censoring.** The 1-cent grid floors measurable variance: for p near
  0.01 the Gaussian rate predicts daily moves far below one tick, so extreme-p
  buckets of test B are censored by discreteness, not only by jumps. Read the
  kurtosis table alongside.
- **Risk-neutral vs objective.** Prices embed fees and any favorite-longshot
  premium; test A's slope mixes excess movement with premium effects.
- **Resolution-time proxy.** tau uses the last observed bar as the resolution
  date; early-resolved markets make tau conservative (too small near the end).
- **Selection.** Volume filters select liquid markets; noise conclusions do
  not automatically extend to the illiquid tail.
