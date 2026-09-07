"""v2 driver: one command reproduces the whole build, in order, deterministically.

Usage:
    python run_all.py                 # run every implemented phase
    python run_all.py --only v2      # run one phase
    python run_all.py --from v3      # run v3 and everything after it
    python run_all.py --list         # show phases and status

Each phase is a callable that reads only data_v2/ + analysis/config.py and
writes only results_v2/ + figures/. Phases not yet built raise
NotImplementedError so the driver is honest about what exists.
"""
from __future__ import annotations

import argparse
import sys
import time


def v1_data():
    """Fresh universe scans + daily backfills into data_v2/ (both venues)."""
    import pull_v2
    if pull_v2.main([]):
        raise RuntimeError("V1 pull failed")


def v2_empirics():
    """Budget test, power family, diagnostics on the v2 vintage."""
    from analysis import empirics_v2
    if empirics_v2.main([]):
        raise RuntimeError("V2 empirics failed")


def v3_regimes():
    """Regime map + per-regime models (SLV, itm-hazard, endgame)."""
    raise NotImplementedError("V3: regime modeling not built yet")


def v4_race():
    """Horse race + budget integration + zero-inflated observation layer."""
    raise NotImplementedError("V4: horse race not built yet")


def v5_bridge():
    """Cross-venue matched-pair bridge analyses."""
    raise NotImplementedError("V5: bridge not built yet")


def v6_docs():
    """Figures + doc-number check (delegates to check_docs.py)."""
    raise NotImplementedError("V6: docs/figures pass not built yet")


PHASES = [
    ("v1", v1_data),
    ("v2", v2_empirics),
    ("v3", v3_regimes),
    ("v4", v4_race),
    ("v5", v5_bridge),
    ("v6", v6_docs),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", metavar="PHASE")
    ap.add_argument("--from", dest="from_", metavar="PHASE")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    names = [n for n, _ in PHASES]
    if args.list:
        for name, fn in PHASES:
            print(f"{name}: {fn.__doc__.splitlines()[0]}")
        return 0

    if args.only:
        selected = [(n, f) for n, f in PHASES if n == args.only]
    elif args.from_:
        i = names.index(args.from_)
        selected = PHASES[i:]
    else:
        selected = PHASES

    for name, fn in selected:
        t0 = time.time()
        print(f"=== {name}: {fn.__doc__.splitlines()[0]}")
        fn()
        print(f"=== {name} done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
