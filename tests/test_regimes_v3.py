"""Oracle gates for analysis/regimes_v3.py.

Each block must recover known truth on its simulator before venue numbers
are believed. Thresholds bracket the seed-7 reference run (map: 43/44
clean cells R1; SLV within 1% on all params; hazard within 0.05; endgame
exponents 0.31/0.50/1.01/0.50 vs truth 0.3/0.5/1.0/0.5) with margin for
seed drift. The SLV fixture is the expensive one (~2 min).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.config import SEED, UNIT
from analysis.core import panel, simulate
from analysis.empirics_v2 import move_panel
from analysis.regimes_v3 import (budget_by_regime, endgame_decompose,
                                 haz_fit, haz_pmf, map_cells, r1_rect,
                                 simulate_compound, simulate_slv,
                                 slv_batches, slv_fit, slv_start)

SLV_TRUTH = {"c": 0.02, "alpha": 0.8, "eta": 0.05, "phi": 0.95, "s": 0.7}


@pytest.fixture(scope="module")
def clean_cells():
    return map_cells(move_panel(simulate(2000)))


@pytest.fixture(scope="module")
def stale_cells():
    return map_cells(move_panel(simulate(2000, stale=0.5)))


@pytest.fixture(scope="module")
def slv_oracle():
    batches = slv_batches(simulate_slv(400, SLV_TRUTH))
    full = slv_fit(batches, "full", slv_start(batches))
    return full, batches


# ------------------------------------------------------------------ map
def test_map_clean_classifies_r1(clean_cells):
    assert (clean_cells["regime"] == 1).mean() >= 0.95
    assert budget_by_regime(clean_cells).get("R1 diffusive", 0.0) >= 0.99


def test_map_clean_rectangle_wide(clean_cells):
    rect = r1_rect(clean_cells)
    assert rect["p_lo"] <= 0.05 and rect["p_hi"] >= 0.95
    assert rect["tau_lo"] <= 7.0


def test_map_stale_triggers_r2(stale_cells):
    assert budget_by_regime(stale_cells).get("R2 itm/hazard", 0.0) >= 0.80
    assert (stale_cells["regime"] == 2).mean() >= 0.60


# ------------------------------------------------------------------ SLV
def test_slv_recovers_truth(slv_oracle):
    th = slv_oracle[0]["theta"]
    assert 0.015 <= th["c"] <= 0.027
    assert 0.70 <= th["alpha"] <= 0.90
    assert 0.035 <= th["eta"] <= 0.065
    assert th["phi"] >= 0.90
    assert 0.60 <= th["s"] <= 0.80


def test_slv_no_sv_is_worse(slv_oracle):
    # reference gap +0.0055/obs (LR ~ 750 on 68k steps); require the sign
    # with margin, not the magnitude, which shrinks as no-sv re-tunes c
    full, batches = slv_oracle
    nosv = slv_fit(batches, "no-sv", slv_start(batches))
    assert full["ll_per_obs"] - nosv["ll_per_obs"] > 0.002


# --------------------------------------------------------------- hazard
def test_hazard_recovers_truth():
    rng = np.random.default_rng(SEED)
    grid = np.arange(-400, 401)
    for truth in ({"pi0": 0.70, "h": 0.04, "sigma": 0.8, "rho_a": 0.85,
                   "rho_t": 0.60, "w_a": 0.65},
                  {"pi0": 0.30, "h": 0.10, "sigma": 1.5, "rho_a": 0.70,
                   "rho_t": 0.75, "w_a": 0.40}):
        p = haz_pmf(grid, truth)
        th = haz_fit(rng.choice(grid, size=30_000, p=p / p.sum()))
        for name, tv in truth.items():
            tol = 0.15 * tv if name == "sigma" else 0.06
            assert abs(th[name] - tv) <= tol, (name, th[name], tv)


# -------------------------------------------------------------- endgame
def test_endgame_recovers_exponents():
    mkts = simulate_compound(4000, a=0.9, b=0.5, g1=0.3,
                             m=0.03, d=0.5, g2=1.0)
    dec = endgame_decompose(panel(mkts), UNIT["sim"])
    assert 0.15 <= dec["arrival"]["g"] <= 0.45      # truth 0.3
    assert 0.35 <= dec["arrival"]["b"] <= 0.65      # truth 0.5
    assert 0.80 <= dec["size"]["g"] <= 1.20         # truth 1.0
    assert 0.35 <= dec["size"]["b"] <= 0.65         # truth 0.5
