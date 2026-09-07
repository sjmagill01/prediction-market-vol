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


def _jscalar(fname: str, path: tuple):
    """Reader for an arbitrary JSON file in results_v2/ (keys or list indices)."""
    def compute() -> float:
        import json
        with open(RESULTS / fname) as fh:
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

# ---- V3 regime map + per-regime models (analysis/regimes_v3.py, run 2026-09-07)
# SLV variants list order: [full, no-sv, no-noise].
CLAIMS += [
    Claim("V3 PM R1 budget share",
          _jscalar("regimes_scalars_polymarket.json",
                   ("budget_by_regime", "R1 diffusive")), 0.613, 0.0005),
    Claim("V3 Kalshi R1 budget share",
          _jscalar("regimes_scalars_kalshi.json",
                   ("budget_by_regime", "R1 diffusive")), 0.828, 0.0005),
    Claim("V3 PM SLV segments",
          _jscalar("slv_polymarket.json", ("n_segments",)), 1158, 0),
    Claim("V3 Kalshi SLV segments",
          _jscalar("slv_kalshi.json", ("n_segments",)), 1818, 0),
    Claim("V3 PM SLV full ll/obs",
          _jscalar("slv_polymarket.json", ("variants", 0, "ll_per_obs")),
          1.672, 0.0005),
    Claim("V3 Kalshi SLV full ll/obs",
          _jscalar("slv_kalshi.json", ("variants", 0, "ll_per_obs")),
          1.007, 0.0005),
    Claim("V3 PM SLV no-sv d_ll",
          _jscalar("slv_polymarket.json", ("variants", 1, "d_ll")),
          -0.649, 0.0005),
    Claim("V3 Kalshi SLV no-sv d_ll",
          _jscalar("slv_kalshi.json", ("variants", 1, "d_ll")),
          -0.524, 0.0005),
    # Boundary solution, documented as a finding (grid probe 2026-09-07:
    # not a quadrature artifact; s re-pins at any S_MAX).
    Claim("V3 PM SLV s pinned at S_MAX",
          _jscalar("slv_polymarket.json", ("variants", 0, "s")), 2.0, 1e-6),
    Claim("V3 Kalshi SLV s pinned at S_MAX",
          _jscalar("slv_kalshi.json", ("variants", 0, "s")), 2.0, 1e-6),
    Claim("V3 PM endgame total gamma",
          _jscalar("endgame_polymarket.json", ("total", "g")), 1.510, 0.0005),
    Claim("V3 Kalshi endgame total gamma",
          _jscalar("endgame_kalshi.json", ("total", "g")), 0.556, 0.0005),
]

# ---- V4 zero-inflated layer + horse race + budget (analysis/race_v4.py,
# run 2026-09-07). zi variants list order: [zi, zi-off]. race models list
# order: [composite-zi, composite, global-slv, const-lam, bucket, garch].
CLAIMS += [
    # s unpins on both venues once the observation lives on the tick grid;
    # PM via staleness mass, Kalshi via tick censoring alone.
    Claim("V4 PM zi s interior",
          _jscalar("zi_polymarket.json", ("variants", 0, "s")), 2.104, 0.0005),
    Claim("V4 PM zi pi0",
          _jscalar("zi_polymarket.json", ("variants", 0, "pi0")),
          0.284, 0.0005),
    Claim("V4 PM zi-off d_ll",
          _jscalar("zi_polymarket.json", ("variants", 1, "d_ll")),
          -0.275, 0.0005),
    Claim("V4 Kalshi zi s interior",
          _jscalar("zi_kalshi.json", ("variants", 0, "s")), 2.421, 0.001),
    Claim("V4 Kalshi zi-off s interior",
          _jscalar("zi_kalshi.json", ("variants", 1, "s")), 2.429, 0.001),
    Claim("V4 Kalshi zi pi0",
          _jscalar("zi_kalshi.json", ("variants", 0, "pi0")), 0.006, 0.0005),
    Claim("V4 PM race n_obs",
          _jscalar("race_summary_polymarket.json", ("n_obs",)), 117252, 0),
    Claim("V4 Kalshi race n_obs",
          _jscalar("race_summary_kalshi.json", ("n_obs",)), 49209, 0),
    Claim("V4 PM composite-zi delta",
          _jscalar("race_summary_polymarket.json",
                   ("models", 0, "delta_vs_composite")), 0.069, 0.0005),
    Claim("V4 PM global-slv delta",
          _jscalar("race_summary_polymarket.json",
                   ("models", 2, "delta_vs_composite")), -4.101, 0.001),
    Claim("V4 Kalshi composite-zi delta",
          _jscalar("race_summary_kalshi.json",
                   ("models", 0, "delta_vs_composite")), 0.016, 0.0005),
    Claim("V4 Kalshi global-slv delta",
          _jscalar("race_summary_kalshi.json",
                   ("models", 2, "delta_vs_composite")), 0.179, 0.0005),
    Claim("V4 PM budget 30d full ratio",
          _jscalar("budget_v4_polymarket.json", ("tau~30d", "full_ratio")),
          1.157, 0.0005),
    Claim("V4 PM budget 30d realized ratio",
          _jscalar("budget_v4_polymarket.json", ("tau~30d", "real_ratio")),
          1.174, 0.0005),
    Claim("V4 Kalshi budget 30d full ratio",
          _jscalar("budget_v4_kalshi.json", ("tau~30d", "full_ratio")),
          1.468, 0.001),
    Claim("V4 Kalshi budget 30d realized ratio",
          _jscalar("budget_v4_kalshi.json", ("tau~30d", "real_ratio")),
          1.626, 0.0005),
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
