"""Oracle gates for V6.5: fingerprints, GMM machinery, structure recovery.

Each gate checks a known truth the pipeline must recover, mirroring the
regimes_v3/race_v4 pattern: if the tool cannot pass these, its venue
readouts mean nothing.
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.cluster_v65 import (FEATURES, adjusted_rand,      # noqa: E402
                                  feature_matrix, fingerprints, gmm_fit,
                                  gmm_labels, select_k)
from analysis.config import SEED                                 # noqa: E402
from analysis.core import simulate                               # noqa: E402


def _two_pops():
    """fingerprints() sorts by key, so truth must be derived from the key."""
    a = simulate(500, seed=SEED)
    b = [replace(m, key="n_" + m.key)
         for m in simulate(500, stale=0.6, noise=0.02, seed=SEED + 1)]
    return a + b


# ------------------------------------------------------------- fingerprints
def test_fingerprints_outcome_free():
    """Flipping every resolved outcome must not change a single feature."""
    mkts = simulate(200, seed=SEED)
    flipped = [replace(m, final=1.0 - m.final) for m in mkts]
    a, b = fingerprints(mkts), fingerprints(flipped)
    pd.testing.assert_frame_equal(a, b)


def test_fingerprints_shapes_and_ranges():
    F = fingerprints(simulate(300, seed=SEED))
    assert set(FEATURES) <= set(F.columns)
    assert F[["f1", "f2", "f3", "f4"]].to_numpy().min() >= 0.0
    assert (F[["f1", "f2", "f3", "f4"]].sum(axis=1) <= 1.0 + 1e-9).all()
    assert F["ar1"].between(-1, 1).all()
    assert F["still"].between(0, 1).all()
    assert np.isfinite(F[FEATURES].to_numpy()).all()


def test_fingerprints_detect_knobs():
    """Pure staleness raises stillness; additive noise drives AR1 negative."""
    a = fingerprints(simulate(400, seed=SEED))
    s = fingerprints(simulate(400, stale=0.6, seed=SEED + 1))
    n = fingerprints(simulate(400, noise=0.02, seed=SEED + 2))
    assert s["still"].mean() > a["still"].mean() + 0.15
    assert n["ar1"].mean() < a["ar1"].mean() - 0.05   # bounce is negative AR1
    # the clustered (state-normalised) features must see the same knobs
    assert s["xstill"].mean() > a["xstill"].mean() + 0.2
    assert n["ar1z"].mean() < a["ar1z"].mean() - 0.1
    assert n["lam"].mean() > a["lam"].mean() + 0.5    # added noise raises z^2


# --------------------------------------------------------------------- GMM
def test_gmm_two_gaussians():
    """Well-separated 2-component truth: k=2 selected, near-perfect labels."""
    rng = np.random.default_rng(SEED)
    X = np.vstack([rng.normal(0, 1, (600, 4)), rng.normal(5, 1, (400, 4))])
    truth = np.array([0] * 600 + [1] * 400)
    model, tab = select_k(X, np.random.default_rng(SEED), k_range=range(1, 5))
    assert model["k"] == 2, tab
    assert adjusted_rand(gmm_labels(model, X), truth) > 0.95


def test_gmm_one_gaussian_selects_k1():
    """A single Gaussian population must not be split (k=1 reachable)."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(0, 1, (1200, 5))
    model, _ = select_k(X, np.random.default_rng(SEED), k_range=range(1, 5))
    assert model["k"] == 1


def test_gmm_loglik_improves_with_truth_k():
    rng = np.random.default_rng(SEED)
    X = np.vstack([rng.normal(0, 1, (500, 3)), rng.normal(4, 1, (500, 3))])
    r1 = gmm_fit(X, 1, np.random.default_rng(SEED))
    r2 = gmm_fit(X, 2, np.random.default_rng(SEED))
    assert r2["ll"] > r1["ll"] + 500          # decisive, not epsilon
    assert r2["bic"] < r1["bic"]


def test_adjusted_rand_bounds():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert abs(adjusted_rand(a, a) - 1.0) < 1e-12
    perm = np.array([1, 1, 2, 2, 0, 0])       # relabeled, same partition
    assert abs(adjusted_rand(a, perm) - 1.0) < 1e-12
    rng = np.random.default_rng(SEED)
    r = adjusted_rand(rng.integers(0, 3, 3000), rng.integers(0, 3, 3000))
    assert abs(r) < 0.05


# ---------------------------------------------------------------- recovery
def test_two_pop_sim_recovered():
    """End-to-end: clean vs stale+noise sims separate into clusters."""
    F = fingerprints(_two_pops())
    truth = F["key"].str.startswith("n_").astype(int).to_numpy()
    X = feature_matrix(F)
    model, tab = select_k(X, np.random.default_rng(SEED))
    assert model["k"] >= 2, tab
    lab = gmm_labels(model, X)
    # collapse to the best 2-way split: map each cluster to its majority pop
    maj = pd.Series(truth).groupby(lab).mean().round().astype(int)
    assert adjusted_rand(maj[lab].to_numpy(), truth) > 0.6
