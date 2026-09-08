r"""V6.5 structure first: model-free market fingerprints, then per-cluster fits.

The V2-V4 parametric layer forced one theta (or one theta per hand-drawn
regime) onto the whole venue, and showed classic mixture symptoms: s pinning
at bounds, zi degeneracy, exponents landing between named candidates. V6.5
asks whether the population is a mixture of walk types and, if so, whether
the fits tighten within the discovered types. Order of operations is
deliberately structure-first, model-second:

(A) Fingerprints. One row per market, built ONLY from the traversal path
    (closes, timestamps, resolution date). Outcome-free by construction:
    the terminal jump and the resolved outcome are never features, so the
    per-cluster budget decomposition in (C) is not circular.

    The CLUSTERED features describe the law of the STANDARDISED increments
    z = dp / gauss_rate(p, tau): the zero-parameter probit rate strips the
    mechanical (p, tau) dependence first, exactly as in the regime map,
    so the clusterer sees the observation layer and the information flow
    rather than where the path happened to live. (The first cut clustered
    raw path-shape features; a supervised probe showed a 97%-separable
    2-population sim that the GMM could not split, because likelihood
    gains on the nuisance size/shape axes drown the dynamics signal.
    State-normalising fixed it: k=2 ARI went 0.05 -> 0.98.)
      xstill   stillness beyond what tick-censored diffusion predicts
      lam      log mean z^2 (average normalised intensity)
      kz       log(3 + excess kurtosis of z) (lumpiness of arrival)
      dlam     log mean z^2, second half of life minus first (timing)
      ar1z     within-market lag-1 autocorr of z (bounce)
    DESCRIPTOR features (reported on cards, never clustered): log_moves,
    log_tres, log_mdt, still, p_ext, p_range, spend_lr (log traversal
    spend over prior budget), f1..f4 (variance shares by life fifth),
    lump, ar1. All stacked groupby aggregations, no per-market fit.

(B) Mixture. Numpy EM-GMM (full covariance, k-means++ init, restarts,
    escalating-jitter Cholesky). Model selection is by held-out
    log-likelihood with the split BY CONTRACT: fit on 70% of markets,
    score the rest, choose the SMALLEST k within SEL_TOL nats of the best
    (k = 1 is genuinely reachable). BIC reported alongside. Labels for
    every market come from the train-fitted model's posterior, so test
    markets never shape the clusters they are assigned to.

(C) Cards. Per cluster: size, unstandardised feature means, category mix,
    and the OUTCOMES (not features): test-A budget slope and terminal-jump
    share of lifetime variance. Adjusted Rand vs category metadata answers
    "did we just rediscover sports vs politics". The per-cluster budget
    slopes decompose the venue-level excess-movement finding.

(D) Per-cluster power family. The V2 (gamma, alpha) fit rerun inside each
    cluster: if exponents separate toward named candidates (binomial 1/1,
    Wright-Fisher 1/0, logit-Brownian 2/1), the venue-level blend was
    aggregation bias.

(E) Per-cluster SLV, gated. Three partitions of the same markets - global
    (1 cell), category groups, clusters - each cell refit with the V3 R1
    SLV on train markets and scored on the SAME held-out segments. The
    clustering earns its keep only if held-out ll/obs beats both global
    and the category baseline. Cells below the step floor fall back to
    the global train theta. This block is the slow one (~10-20 NM fits
    per venue); it writes results_v2/cluster65_progress.json after every
    fit so a long run can be checked from outside.

Oracle gates live in tests/test_cluster_v65.py; main() reruns the sim
oracles before the venues (regimes_v3 pattern). Outputs in results_v2/:
  cluster65_features_{label}.csv    fingerprint table + labels
  cluster65_select_{label}.csv     k table (train ll, held-out ll, BIC)
  cluster65_cards_{label}.csv      per-cluster cards
  cluster65_scalars_{label}.json   chosen k, ARI, budget/power per cluster,
                                   SLV gate table
  cluster65_progress.json          live status (check in during long runs)

Usage:
  python analysis/cluster_v65.py                   # oracles + both venues
  python analysis/cluster_v65.py --venue kalshi
  python analysis/cluster_v65.py --sim-only
  python analysis/cluster_v65.py --skip-slv        # fast blocks A-D only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.config import DATA_V2, RESULTS, SEED                    # noqa: E402
from analysis.core import (P_CLIP, Market, load_markets, panel,      # noqa: E402
                           simulate)
from analysis.empirics_v2 import move_panel, origin_slope, power_fit  # noqa: E402
from analysis.regimes_v3 import (STILL_TICK, slv_batches, slv_fit,    # noqa: E402
                                 slv_loglik, slv_segments, slv_start)

EPS = 1e-12
CLUSTER_FEATURES = ["xstill", "lam", "kz", "dlam", "ar1z"]
DESC_FEATURES = ["log_moves", "log_tres", "log_mdt", "still", "p_ext",
                 "p_range", "spend_lr", "f1", "f2", "f3", "f4", "lump", "ar1"]
FEATURES = CLUSTER_FEATURES + DESC_FEATURES
K_RANGE = range(1, 9)
SEL_TOL = 0.01          # nats/market: smallest k within this of the best
TEST_FRAC = 0.3         # held-out market share (by contract, everywhere)
N_INIT = 4              # EM restarts
AR1_MIN = 4             # moves needed for a within-market autocorr
N_CAT = 4               # top categories kept for the baseline; rest "other"
SLV_MIN_STEPS = 2_000   # train steps below which a cell falls back to global
SLV_MAX_STEPS = 20_000  # subsample cap per cell fit
PROGRESS = "cluster65_progress.json"
_T0 = time.time()


def _progress(stage: str, **kw) -> None:
    RESULTS.mkdir(exist_ok=True)
    obj = {"stage": stage, "elapsed_s": round(time.time() - _T0, 1), **kw}
    with open(RESULTS / PROGRESS, "w") as fh:
        json.dump(obj, fh, indent=1)
    print(f"[{obj['elapsed_s']:8.1f}s] {stage} "
          + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


# ============================================================ (A) fingerprints
def _grouped_ar1(key: pd.Series, x: pd.Series, out_index) -> pd.Series:
    """Within-group lag-1 autocorr of x by grouped moment sums."""
    nxt = key.eq(key.shift(-1))
    d = pd.DataFrame({"key": key, "x": x, "y": x.shift(-1)})[nxt.to_numpy()]
    g = d.groupby("key")
    n = g.size()
    mx, my = g["x"].mean(), g["y"].mean()
    cov = (d["x"] * d["y"]).groupby(d["key"]).sum() / n - mx * my
    var = np.sqrt((g["x"].var(ddof=0) * g["y"].var(ddof=0)).clip(lower=0))
    return (cov / (var + EPS)).where(n >= AR1_MIN) \
        .reindex(out_index).fillna(0.0).clip(-1, 1)


def fingerprints(mkts: list[Market]) -> pd.DataFrame:
    """Outcome-free per-market feature table (uses closes/times only).

    Rows come out sorted by key (groupby order), NOT input order.
    """
    pan = panel(mkts)
    meta = pd.DataFrame({"key": [m.key for m in mkts],
                         "t_res": [max(m.t_res, 1.0) for m in mkts],
                         "p0": [float(m.closes[0]) for m in mkts]})
    pan = pan.merge(meta, on="key", how="inner")
    frac = (pan["t_res"] - pan["tau"]) / pan["t_res"]
    pan["fifth"] = np.clip((5 * frac).astype(int), 0, 4)
    pan["late"] = frac > 0.5
    pan["still"] = pan["dp"].abs() < STILL_TICK
    pan["p_ext"] = (pan["p"] - 0.5).abs()
    # standardised increments under the zero-parameter probit rate
    pc = pan["p"].clip(*P_CLIP)
    sig = np.sqrt((norm.pdf(norm.ppf(pc)) ** 2
                   / np.maximum(pan["tau"], 1.0) * pan["dt"])
                  .clip(lower=1e-12))
    pan["z"] = pan["dp"] / sig
    pan["z2"] = pan["z"] ** 2
    pan["z4"] = pan["z2"] ** 2
    pan["xstill"] = (pan["still"].astype(float)
                     - (2 * norm.cdf(STILL_TICK / sig) - 1))

    g = pan.groupby("key", sort=True)
    m2, m4 = g["z2"].mean(), g["z4"].mean()
    F = pd.DataFrame({
        "n_moves": g.size(), "mdt": g["dt"].mean(), "still": g["still"].mean(),
        "p_ext": g["p_ext"].mean(), "p_range": g["p"].max() - g["p"].min(),
        "tot2": g["dp2"].sum(), "max2": g["dp2"].max(),
        "t_res": g["t_res"].first(), "p0": g["p0"].first(),
        "xstill": g["xstill"].mean(), "lam": np.log(m2 + 1e-9),
        "kz": np.log(3 + (m4 / np.maximum(m2, EPS) ** 2 - 3)
                     .clip(lower=-2.9))})
    F["log_moves"] = np.log(F["n_moves"])
    F["log_tres"] = np.log(F["t_res"])
    F["log_mdt"] = np.log(F["mdt"] + EPS)
    F["spend_lr"] = np.log((F["tot2"] + EPS)
                           / (F["p0"] * (1 - F["p0"]) + EPS))
    F["lump"] = np.log((F["max2"] + EPS) / (F["tot2"] / F["n_moves"] + EPS))

    halves = pan.pivot_table(index="key", columns="late", values="z2",
                             aggfunc="mean").reindex(columns=[False, True])
    F["dlam"] = (np.log(halves[True] + 1e-9)
                 - np.log(halves[False] + 1e-9)) \
        .reindex(F.index).fillna(0.0)

    fifths = pan.pivot_table(index="key", columns="fifth", values="dp2",
                             aggfunc="sum").reindex(columns=range(5)) \
        .fillna(0.0)
    shares = fifths.div(F["tot2"] + EPS, axis=0)
    for j in range(4):
        F[f"f{j + 1}"] = shares[j].reindex(F.index).fillna(0.0)

    F["ar1"] = _grouped_ar1(pan["key"], pan["dp"], F.index)
    F["ar1z"] = _grouped_ar1(pan["key"], pan["z"], F.index)
    return F.reset_index()


def feature_matrix(F: pd.DataFrame) -> np.ndarray:
    """Standardised CLUSTER_FEATURES only; descriptors stay on the cards."""
    X = F[CLUSTER_FEATURES].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.maximum(sd, 1e-9)


# ================================================================= (B) GMM
def _chol(cov: np.ndarray) -> np.ndarray:
    jit = 1e-8 * max(np.trace(cov) / len(cov), 1e-12)
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + jit * np.eye(len(cov)))
        except np.linalg.LinAlgError:
            jit *= 100
    raise np.linalg.LinAlgError("covariance not repairable")


def _comp_logpdf(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """(n, k) log N(x; mu_j, cov_j); loop over k<=8, vectorised over n."""
    n, d = X.shape
    out = np.empty((n, len(mu)))
    for j in range(len(mu)):
        L = _chol(cov[j])
        sol = np.linalg.solve(L, (X - mu[j]).T)
        out[:, j] = (-0.5 * (d * np.log(2 * np.pi) + (sol ** 2).sum(0))
                     - np.log(np.diag(L)).sum())
    return out


def _lse(a: np.ndarray) -> np.ndarray:
    mx = a.max(1, keepdims=True)
    return (mx + np.log(np.exp(a - mx).sum(1, keepdims=True)))[:, 0]


def _kpp(X: np.ndarray, k: int, rng) -> np.ndarray:
    """k-means++ seeding."""
    c = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min([((X - ci) ** 2).sum(1) for ci in c], axis=0)
        c.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(c)


def gmm_fit(X: np.ndarray, k: int, rng, n_init: int = N_INIT,
            max_iter: int = 300, tol: float = 1e-7) -> dict:
    """Full-covariance EM, best of n_init k-means++ starts."""
    n, d = X.shape
    best = None
    for _ in range(n_init):
        centers = _kpp(X, k, rng)
        lab = ((X[:, None, :] - centers[None]) ** 2).sum(2).argmin(1)
        resp = np.full((n, k), 1e-3)
        resp[np.arange(n), lab] = 1.0
        resp /= resp.sum(1, keepdims=True)
        ll_old = -np.inf
        for it in range(max_iter):
            nk = resp.sum(0) + 1e-10
            w = nk / n
            mu = (resp.T @ X) / nk[:, None]
            cov = (np.einsum("nj,nd,ne->jde", resp, X, X) / nk[:, None, None]
                   - mu[:, :, None] * mu[:, None, :])
            lp = _comp_logpdf(X, mu, cov) + np.log(w)
            row = _lse(lp)
            total = float(row.sum())
            resp = np.exp(lp - row[:, None])
            if it > 10 and total - ll_old < tol * n:
                break
            ll_old = total
        if best is None or total > best["ll"]:
            best = {"w": w, "mu": mu, "cov": cov, "ll": total}
    n_par = (k - 1) + k * d + k * d * (d + 1) // 2
    best["k"], best["bic"] = k, -2 * best["ll"] + n_par * np.log(n)
    return best


def gmm_score(model: dict, X: np.ndarray) -> np.ndarray:
    return _lse(_comp_logpdf(X, model["mu"], model["cov"])
                + np.log(model["w"]))


def gmm_labels(model: dict, X: np.ndarray) -> np.ndarray:
    return (_comp_logpdf(X, model["mu"], model["cov"])
            + np.log(model["w"])).argmax(1)


def select_k(X: np.ndarray, rng, k_range=K_RANGE) -> tuple[dict, pd.DataFrame]:
    """Held-out selection: fit on 70% of rows, pick smallest k in tolerance."""
    n = len(X)
    test = rng.random(n) < TEST_FRAC
    rows, models = [], {}
    for k in k_range:
        m = gmm_fit(X[~test], k, rng)
        held = float(gmm_score(m, X[test]).mean())
        models[k] = m
        rows.append({"k": k, "train_ll": m["ll"] / (~test).sum(),
                     "heldout_ll": held, "bic": m["bic"]})
        _progress("select_k", k=k, heldout=round(held, 4))
    tab = pd.DataFrame(rows)
    best = tab["heldout_ll"].max()
    k_star = int(tab.loc[tab["heldout_ll"] >= best - SEL_TOL, "k"].min())
    tab["chosen"] = tab["k"] == k_star
    m = models[k_star]
    m["test_mask"] = test
    return m, tab


# ================================================================ (C) cards
def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    ct = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy(float)
    comb = lambda x: x * (x - 1) / 2                          # noqa: E731
    s_ij, s_a, s_b = comb(ct).sum(), comb(ct.sum(1)).sum(), comb(ct.sum(0)).sum()
    exp = s_a * s_b / comb(ct.sum())
    mx = 0.5 * (s_a + s_b)
    return float((s_ij - exp) / (mx - exp)) if mx != exp else 0.0


def categories(venue: str, keys: pd.Series) -> pd.Series:
    """Coarse category per market key, top N_CAT kept, rest 'other'."""
    if venue == "kalshi":
        cat = pd.read_parquet(DATA_V2 / "catalog" / "kalshi.parquet",
                              columns=["key", "category"])
        s = keys.map(dict(zip(cat.key.astype(str), cat.category)))
    else:
        cat = pd.read_parquet(DATA_V2 / "catalog" / "polymarket.parquet",
                              columns=["key", "event_id"])
        tags = pd.read_parquet(DATA_V2 / "catalog" / "pm_event_tags.parquet")
        lists = [json.loads(ts) if ts else [] for ts in tags["tags"]]
        counts: dict[str, int] = {}
        for ts in lists:
            for t in ts:
                counts[t] = counts.get(t, 0) + 1
        ev = {e: max(ts, key=lambda t: counts[t]) if ts else "other"
              for e, ts in zip(tags["event_id"], lists)}
        e_of = dict(zip(cat.key.astype(str), cat.event_id))
        s = keys.map(lambda k: ev.get(e_of.get(k), "other"))
    s = s.fillna("other")
    top = s.value_counts().head(N_CAT).index
    return s.where(s.isin(top), "other")


def cluster_cards(F: pd.DataFrame, mkts: list[Market],
                  cats: pd.Series | None) -> pd.DataFrame:
    """Per-cluster feature means + outcome readouts (budget slope, terminal)."""
    m_by = {m.key: m for m in mkts}
    x = (F["p0"] * (1 - F["p0"])).to_numpy()
    trav = F["tot2"].to_numpy()
    term = np.array([(m_by[k].final - m_by[k].closes[-1]) ** 2
                     for k in F["key"]])
    rows = []
    for c, g in F.groupby("cluster"):
        i = g.index.to_numpy()
        slope, se = origin_slope(x[i], trav[i] + term[i])
        row = {"cluster": int(c), "n_mkts": len(g),
               "share": len(g) / len(F),
               **{f: float(g[f].mean()) for f in FEATURES},
               "budget_slope": slope, "budget_se": se,
               "term_share": float(term[i].sum()
                                   / (trav[i].sum() + term[i].sum() + EPS))}
        if cats is not None:
            top = cats.loc[g.index].value_counts(normalize=True)
            row["top_cat"] = top.index[0]
            row["top_cat_frac"] = float(top.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster")


# ======================================================= (D) power per cluster
def cluster_power(F: pd.DataFrame, mkts: list[Market]) -> list[dict]:
    m_by = {m.key: m for m in mkts}
    out = []
    for c, g in F.groupby("cluster"):
        pan = move_panel([m_by[k] for k in g["key"]])
        fit = power_fit(pan, bulk=True)
        out.append({"cluster": int(c), "n_mkts": len(g),
                    "gamma": fit.get("gamma", np.nan),
                    "alpha": fit.get("alpha", np.nan),
                    "n_cells": fit.get("n_cells", 0)})
    return out


# ========================================================== (E) SLV partition
def _fit_cell(mkts_tr: list[Market], rect: dict, rng, tag: str) -> dict | None:
    segs = slv_segments(mkts_tr, rect)
    rng.shuffle(segs)
    keep, tot = [], 0
    for s in segs:
        keep.append(s)
        tot += len(s[0]) - 1
        if tot >= SLV_MAX_STEPS:
            break
    if tot < SLV_MIN_STEPS:
        _progress("slv_fit", cell=tag, skipped=f"{tot} steps < floor")
        return None
    batches = slv_batches(keep)
    r = slv_fit(batches, "full", slv_start(batches))
    _progress("slv_fit", cell=tag, steps=tot,
              ll=round(r["ll_per_obs"], 4), secs=round(r["seconds"]))
    return r["theta"]


def slv_gate(F: pd.DataFrame, mkts: list[Market], cats: pd.Series,
             venue: str, rng) -> dict:
    """Held-out ll/obs of global vs category vs cluster SLV partitions."""
    with open(RESULTS / f"regimes_scalars_{venue}.json") as fh:
        rect = json.load(fh)["rect"]
    m_by = {m.key: m for m in mkts}
    test = pd.Series(rng.random(len(F)) < TEST_FRAC, index=F.index)
    parts = {"global": pd.Series("all", index=F.index),
             "category": cats.astype(str),
             "cluster": F["cluster"].astype(str)}
    g_theta = _fit_cell([m_by[k] for k in F.loc[~test, "key"]],
                        rect, rng, "global/all")
    out = {}
    for name, part in parts.items():
        ll_sum, n_sum, cells = 0.0, 0, {}
        for cell in sorted(part.unique()):
            in_cell = part == cell
            theta = (g_theta if name == "global" else
                     _fit_cell([m_by[k] for k in F.loc[in_cell & ~test, "key"]],
                               rect, rng, f"{name}/{cell}")) or g_theta
            segs = slv_segments([m_by[k] for k in F.loc[in_cell & test, "key"]],
                                rect)
            if not segs:
                continue
            ll, n = slv_loglik(theta, slv_batches(segs))
            ll_sum, n_sum = ll_sum + ll, n_sum + n
            cells[cell] = {"ll_per_obs": ll / n if n else np.nan, "n": n}
        out[name] = {"heldout_ll_per_obs": ll_sum / n_sum if n_sum else np.nan,
                     "n_obs": n_sum, "cells": cells}
        _progress("slv_gate", partition=name,
                  heldout=round(out[name]["heldout_ll_per_obs"], 4))
    return out


# ================================================================= drivers
def run_venue(label: str, mkts: list[Market], venue: str | None,
              skip_slv: bool = False, write: bool = True) -> dict:
    rng = np.random.default_rng(SEED)
    _progress("fingerprints", label=label, n_mkts=len(mkts))
    F = fingerprints(mkts)
    X = feature_matrix(F)
    model, tab = select_k(X, rng)
    F["cluster"] = gmm_labels(model, X)
    k_star = model["k"]
    print(f"\n===== {label}: k selection =====")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cats = categories(venue, F["key"]) if venue else None
    cards = cluster_cards(F, mkts, cats)
    print(f"\n===== {label}: {k_star} clusters =====")
    print(cards.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ari = (adjusted_rand(F["cluster"].to_numpy(), cats.to_numpy())
           if cats is not None else np.nan)
    if cats is not None:
        print(f"ARI vs category metadata: {ari:.3f}")

    power = cluster_power(F, mkts)
    print(pd.DataFrame(power).to_string(index=False,
                                        float_format=lambda v: f"{v:.3f}"))
    out = {"k": k_star, "n_mkts": len(F), "ari_vs_category": ari,
           "selection": tab.to_dict("records"),
           "cards": cards.to_dict("records"), "power": power}
    if venue and not skip_slv:
        out["slv_gate"] = slv_gate(F, mkts, cats, venue, rng)
        g = out["slv_gate"]
        print("SLV gate (held-out ll/obs): "
              + "  ".join(f"{k} {v['heldout_ll_per_obs']:.4f}"
                          for k, v in g.items()))
    if write:
        RESULTS.mkdir(exist_ok=True)
        F.to_csv(RESULTS / f"cluster65_features_{label}.csv", index=False)
        tab.to_csv(RESULTS / f"cluster65_select_{label}.csv", index=False)
        cards.to_csv(RESULTS / f"cluster65_cards_{label}.csv", index=False)
        with open(RESULTS / f"cluster65_scalars_{label}.json", "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    _progress("venue_done", label=label, k=k_star)
    return out


def run_oracles() -> None:
    """Recovery on known truths; tests/test_cluster_v65.py mirrors these."""
    print("===== oracle: 2-population sim (expect k>=2, high ARI) =====")
    a = simulate(800, seed=SEED)
    b = simulate(800, stale=0.6, noise=0.02, seed=SEED + 1)
    for m in b:
        m.key = "n_" + m.key
    res = run_venue("sim_2pop", a + b, venue=None, write=True)
    F = pd.read_csv(RESULTS / "cluster65_features_sim_2pop.csv")
    truth = F["key"].str.startswith("n_").astype(int).to_numpy()
    ari = adjusted_rand(F["cluster"].to_numpy(), truth)
    print(f"k={res['k']}  ARI vs truth: {ari:.3f}")
    print("\n===== oracle: 1-population sim (k inflation readout) =====")
    res1 = run_venue("sim_1pop", simulate(1600, seed=SEED), venue=None,
                     write=True)
    print(f"clean-sim chosen k = {res1['k']} "
          "(length/tau heterogeneity is real structure, not a bug)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=["polymarket", "kalshi"])
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--skip-slv", action="store_true")
    args = ap.parse_args(argv)
    if not args.venue:
        run_oracles()
        if args.sim_only:
            return 0
    for v in ([args.venue] if args.venue else ["polymarket", "kalshi"]):
        run_venue(v, load_markets(v), venue=v, skip_slv=args.skip_slv)
    _progress("all_done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
