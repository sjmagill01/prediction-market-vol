"""v2 doc-number checker: every headline number in README/NOTES must trace
to a file in results_v2/.

The v1 build drifted because prose numbers came from runs at slightly
different settings. v2 rule: any number quoted in README.md or NOTES.md is
registered here as a Claim, and this script recomputes it from the results
files and fails loudly on mismatch. Run it in CI-style before any commit
that touches docs:

    python check_docs.py

Each phase that lands a headline number appends to CLAIMS. A Claim is
(label, compute, quoted, tol) where compute() reads results_v2/ and
returns a float, quoted is the number as it appears in the docs, and tol
is the acceptable absolute difference (respect the docs' rounding).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

from analysis.config import RESULTS  # noqa: F401 - used by claim closures


@dataclass
class Claim:
    label: str                    # where the number appears, e.g. "README budget PM"
    compute: Callable[[], float]  # recompute from results_v2/
    quoted: float                 # the number as written in the docs
    tol: float                    # allowed |computed - quoted|


# Populated phase by phase (V2 onward). Empty registry passes trivially,
# but --strict below refuses to bless an empty one so this can't rot.
CLAIMS: list[Claim] = []


def _scalar(label: str, path: tuple[str, ...]):
    """Reader for results_v2/empirics_scalars_{label}.json."""
    def compute() -> float:
        import json
        with open(RESULTS / f"empirics_scalars_{label}.json") as fh:
            d = json.load(fh)
        for k in path:
            d = d[k]
        return float(d)
    return compute


# ---- V2 core empirics (analysis/empirics_v2.py, run 2026-09-07)
CLAIMS += [
    Claim("V2 PM budget slope", _scalar("polymarket", ("budget_slope",)),
          1.19, 0.005),
    Claim("V2 Kalshi budget slope", _scalar("kalshi", ("budget_slope",)),
          1.46, 0.005),
    Claim("V2 PM bulk gamma", _scalar("polymarket", ("power_bulk", "gamma")),
          1.51, 0.005),
    Claim("V2 Kalshi bulk gamma", _scalar("kalshi", ("power_bulk", "gamma")),
          1.12, 0.005),
    Claim("V2 PM bulk alpha", _scalar("polymarket", ("power_bulk", "alpha")),
          0.53, 0.005),
    Claim("V2 Kalshi z-ACF lag 1", _scalar("kalshi", ("acf_z_lag1",)),
          -0.277, 0.0005),
    Claim("V2 Kalshi lambda-ACF lag 1", _scalar("kalshi", ("acf_lambda_lag1",)),
          0.205, 0.0005),
    Claim("V2 PM market count", _scalar("polymarket", ("n_markets",)),
          3382, 0),
    Claim("V2 Kalshi market count", _scalar("kalshi", ("n_markets",)),
          28076, 0),
]


def main() -> int:
    strict = "--strict" in sys.argv
    if not CLAIMS:
        print("check_docs: no claims registered yet (V0 scaffold).")
        return 1 if strict else 0
    bad = 0
    for c in CLAIMS:
        got = c.compute()
        ok = abs(got - c.quoted) <= c.tol
        flag = "ok " if ok else "FAIL"
        print(f"[{flag}] {c.label}: quoted {c.quoted} computed {got:.6g} "
              f"(tol {c.tol})")
        bad += not ok
    print(f"check_docs: {len(CLAIMS) - bad}/{len(CLAIMS)} claims verified.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
