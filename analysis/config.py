"""v2 vintage configuration: every knob behind every number, in one place.

The v1 build accumulated numbers from runs at slightly different filter
settings. v2 freezes ONE vintage: this module is the only source of
universe definitions, filters, tick units, and seeds. Analysis modules
import from here and from analysis/core.py only; nothing else defines a
threshold.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- vintage
SNAPSHOT = "2026-09-06"     # universe frozen at markets resolved by this date
SEED = 7                    # global seed; per-script rngs derive from it

ROOT = Path(__file__).resolve().parent.parent
DATA_V2 = ROOT / "data_v2"          # fresh pull lands here (v1 data untouched)
RESULTS = ROOT / "results_v2"
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------- universe
# Polymarket: catalog scan keeps closed markets with volume >= the floor;
# the daily backfill takes the top PM_TOP_N by volume (the >=100k catalog
# holds ~100k markets, so top-N is the binding constraint, revisited at V1
# once the fresh scan reports counts).
PM_MIN_VOLUME = 100_000
PM_TOP_N = 4_000

# Kalshi: wide per-series scan (annual/quarterly/monthly/weekly, both API
# tiers), daily bars for closed markets with volume >= the floor and a
# lifetime long enough to produce at least two daily closes.
KALSHI_MIN_VOLUME = 10_000
KALSHI_MIN_LIFETIME_DAYS = 2.0

# ------------------------------------------------------------------- grids
# Venue tick units for the observation grids (PM quotes to 0.1 cent).
UNIT = {"polymarket": 0.001, "kalshi": 0.01, "sim": 0.01}
