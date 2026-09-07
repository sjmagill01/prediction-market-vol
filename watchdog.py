"""Supervise pull_v2.py: restart it if progress.json goes stale.

The 2026-09-06 pull froze for 30+ min inside one HTTP call. All phases
are resumable, so the cheap fix is: run the pull as a child process,
and if progress.json has not been touched for STALE_S seconds, kill and
restart it. Exits 0 when the pull completes cleanly.

Usage: python watchdog.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
PROGRESS = HERE / "data_v2" / "progress.json"
STALE_S = 600
MAX_RESTARTS = 30


def start() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(HERE / "pull_v2.py")],
                            cwd=HERE)


def main() -> int:
    restarts = 0
    proc = start()
    print(f"watchdog: started pull pid={proc.pid}", flush=True)
    while True:
        time.sleep(60)
        rc = proc.poll()
        if rc == 0:
            print("watchdog: pull completed cleanly", flush=True)
            return 0
        if rc is not None or (
                PROGRESS.exists()
                and time.time() - PROGRESS.stat().st_mtime > STALE_S):
            why = f"exit rc={rc}" if rc is not None else \
                f"progress stale >{STALE_S}s"
            restarts += 1
            if restarts > MAX_RESTARTS:
                print(f"watchdog: giving up after {MAX_RESTARTS} restarts",
                      flush=True)
                return 1
            if rc is None:
                proc.kill()
                proc.wait()
            proc = start()
            print(f"watchdog: restart #{restarts} ({why}) "
                  f"pid={proc.pid}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
