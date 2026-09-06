"""Check on the V1 vintage pull: current phase, rate, ETA, file counts.

Usage: python pull_status.py [--tail N]
"""
from __future__ import annotations

import json
import sys

from analysis.config import DATA_V2


def main() -> int:
    prog = DATA_V2 / "progress.json"
    if prog.exists():
        p = json.loads(prog.read_text())
        pct = 100.0 * p["done"] / p["total"] if p["total"] else 100.0
        print(f"phase       : {p['phase']} [{p['status']}]")
        print(f"progress    : {p['done']}/{p['total']} ({pct:.1f}%)")
        print(f"bar rows    : {p['bar_rows']}")
        print(f"rate        : {p['rate_per_min']}/min "
              f"(elapsed {p['elapsed_min']} min)")
        print(f"phase ETA   : {p['eta_min']} min (~{p['eta_at']})")
        print(f"last update : {p['updated']}")
    else:
        print("no progress.json yet (pull not started)")
    for venue in ("polymarket", "kalshi"):
        d = DATA_V2 / "bars" / venue / "1440m"
        n = len(list(d.glob("*.parquet"))) if d.exists() else 0
        print(f"{venue:11s} : {n} bar files")
    for cat in sorted((DATA_V2 / "catalog").glob("*.parquet")) \
            if (DATA_V2 / "catalog").exists() else []:
        print(f"catalog     : {cat.name}")
    tail = 5
    if "--tail" in sys.argv:
        tail = int(sys.argv[sys.argv.index("--tail") + 1])
    logf = DATA_V2 / "pull.log"
    if logf.exists():
        lines = logf.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"--- last {tail} log lines ---")
        for ln in lines[-tail:]:
            print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
