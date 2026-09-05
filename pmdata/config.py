"""Central configuration: endpoints, rate limits, frequencies, paths."""
from pathlib import Path

# --- Polymarket ---
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# --- Kalshi ---
# Two public base-URL candidates; kalshi.py probes /exchange/status and picks
# the first one that answers. Public (unauthenticated) reads only.
KALSHI_BASES = [
    "https://api.elections.kalshi.com/trade-api/v2",
    "https://external-api.kalshi.com/trade-api/v2",
]

# --- Rate limiting ---
# Conservative default per host; both venues tolerate ~10 rps on public reads.
RATE_LIMIT_RPS = 8.0

# --- Frequencies ---
# name -> bar length in minutes. Kalshi candlesticks accept exactly {1,60,1440}.
FREQS = {"1m": 1, "1h": 60, "1d": 1440}

# --- Storage layout ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CATALOG_DIR = DATA_DIR / "catalog"
BARS_DIR = DATA_DIR / "bars"
STATE_DIR = DATA_DIR / "state"

# Polymarket prices-history chunking (days of wall time per request), tuned
# empirically (see NOTES.md "API fixes"). The binding constraint is a hard
# server-side window cap: startTs..endTs spans of ~20d+ return HTTP 400
# ("interval is too long") at ANY fidelity; 15d is accepted. Mild thinning
# (<1%) appears for 1m windows beyond ~5d, so 1m uses 2d chunks.
PM_CHUNK_DAYS = {1: 2, 60: 14, 1440: None}  # None => single interval=max call

# Kalshi candlesticks: hard cap of 5000 candles per request (docs); stay under.
KALSHI_MAX_CANDLES = 4900
