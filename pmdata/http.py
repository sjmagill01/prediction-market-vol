"""One shared requests.Session with retry/backoff and a per-host token bucket.

- urllib3 Retry handles 429/5xx with exponential backoff and honours
  Retry-After headers (respect_retry_after_header=True is the default).
- A simple token-bucket throttle (RATE_LIMIT_RPS per host) runs client-side so
  we rarely hit 429 in the first place.
- 404 is treated as "resource does not exist" and returns None rather than
  raising, because both venues use 404 for delisted/unknown markets.
"""
from __future__ import annotations

import logging
import threading
import time
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import RATE_LIMIT_RPS

log = logging.getLogger(__name__)


class _TokenBucket:
    def __init__(self, rps: float, burst: int = 8):
        self.rps = rps
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def take(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rps)
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rps
            time.sleep(wait)


_buckets: dict[str, _TokenBucket] = {}
_buckets_lock = threading.Lock()


def _bucket(host: str) -> _TokenBucket:
    with _buckets_lock:
        if host not in _buckets:
            _buckets[host] = _TokenBucket(RATE_LIMIT_RPS)
        return _buckets[host]


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "pm-vol/0.1 (research; contact: local)"})
    return s


SESSION = make_session()


def get_json(url: str, params: dict | None = None, timeout: float = 30.0):
    """GET url, return decoded JSON. 404 -> None. Raises on other errors."""
    host = urlsplit(url).netloc
    _bucket(host).take()
    r = SESSION.get(url, params=params, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()
