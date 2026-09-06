"""Empirical regime map of the (p, tau) plane. P0 of the modeling phase.

The daily tests treat the whole panel as one object, but the dynamics are
not one object: mid-p far-from-expiry markets diffuse; decided-but-unresolved
markets (p near 0/1, tau large) sit still and gap; near-expiry markets are
forced to spend their remaining budget in lumps. Before fitting any model,
this script maps where each behaviour lives, with data-driven boundaries.

Cells: price buckets (vol_check.P_BUCKETS) x log-spaced tau bands. Per cell:

  still_frac  : fraction of daily moves with |dp| < half a cent.
  excess_still: still_frac minus the fraction the diffusive (Gaussian-rate)
                model itself predicts, 2*Phi(halftick/sigma) - 1 averaged
                over the cell. A longshot diffusing in genuinely tiny steps
                is NOT frozen; only stillness beyond the diffusion's own
                prediction indicates censored/hazard territory. (Raw
                still_frac mislabels clean simulated longshots as frozen.)
  kurt_z      : excess kurtosis of z = dp/sqrt(gauss rate). High = moves,
                when they happen, are lumps, not diffusion.
  jump_share  : 1 - (pi/2) E|dp_t||dp_{t+1}| / E dp_t^2 on consecutive-day
                pairs (bipower); share of variance not attributable to a
                continuous component.
  ratio_fit   : cell mean dp^2/dt over the venue bulk power-family
                prediction c[p(1-p)]^gamma/tau^alpha; how far the cell is
                from the fitted diffusion.
  budget_share: cell share of the venue's total traversal variance
                sum(dp^2) (terminal resolution jumps are not in the panel).

Classification (cell-local, thresholds printed with the output):
  R2 "itm/hazard"  if excess_still >= 0.3
  R3 "jumpy"       if kurt_z >= 30 (lumpy but active)
  R1 "diffusive"   otherwise

The rectangle summary then reports, per venue, the largest p-range and
tau-range whose cells are majority-R1: the domain where a diffusive SLV fit
is honest, for use by the later phases.

--simulate N runs the clean latent-Gaussian simulator through the same map:
its cells should classify (nearly) all R1, which validates the thresholds.

Usage:
  python analysis/regime_map.py [--venue polymarket|kalshi|both]
                                [--simulate N] [--no-plot]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import vol_check as vc  # noqa: E402

TAU_EDGES = np.array([1.0, 3.0, 7.0, 14.0, 30.0, 90.0, np.inf])
STILL_TICK = 0.005          # half a cent: "nothing happened today"
STILL_THRESH = 0.3          # R2 if stillness exceeds the diffusive prediction by this
KURT_THRESH = 30.0          # R3 if active but lumpy
MIN_CELL = 200

REGIME_NAMES = {1: "R1 diffusive", 2: "R2 itm/hazard", 3: "R3 jumpy"}


def _tau_band(tau: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(TAU_EDGES, tau, side="right") - 1,
                   0, len(TAU_EDGES) - 2)


def _tau_label(j: int) -> str:
    hi = TAU_EDGES[j + 1]
    hi_s = "inf" if np.isinf(hi) else f"{hi:.0f}"
    return f"[{TAU_EDGES[j]:.0f},{hi_s})d"


def build_cells(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-(p bucket, tau band) metrics. panel = vc.candidate_rates output."""
    df = panel.copy()
    df["pi"] = vc._bucket(df["p"].to_numpy())
    df["ti"] = _tau_band(df["tau"].to_numpy())
    from scipy.stats import norm
    df["z"] = df["dp"] / np.sqrt(df["v_gauss"].clip(lower=1e-12))
    df["still"] = df["dp"].abs() < STILL_TICK
    # stillness the diffusive model itself predicts at this (p, tau)
    df["pred_still"] = 2 * norm.cdf(
        STILL_TICK / np.sqrt(df["v_gauss"].clip(lower=1e-12))) - 1
    df["y"] = df["dp2"] / df["dt"]

    # bipower pairs: consecutive daily rows within a market
    df = df.sort_values(["key", "tau"], ascending=[True, False])
    same = (df["key"] == df["key"].shift(-1)) & \
           df["dt"].between(0.9, 1.1) & df["dt"].shift(-1).between(0.9, 1.1)
    df["bp_prod"] = np.where(same, df["dp"].abs() * df["dp"].shift(-1).abs(),
                             np.nan)
    df["bp_dp2"] = np.where(same, df["dp2"], np.nan)

    pf = vc.power_fit(df[(df["p"] >= 0.05) & (df["p"] <= 0.95)])

    rows = []
    for (pi, ti), g in df.groupby(["pi", "ti"]):
        if len(g) < MIN_CELL:
            continue
        z = g["z"].to_numpy()
        z = z[np.isfinite(z)]
        m2 = np.mean((z - z.mean()) ** 2)
        m4 = np.mean((z - z.mean()) ** 4)
        bp = g["bp_prod"].dropna()
        jump = (1 - (np.pi / 2) * bp.sum() / g["bp_dp2"].dropna().sum()
                if len(bp) >= 50 and g["bp_dp2"].dropna().sum() > 0 else np.nan)
        pred = (pf["c"] * (g["p"] * (1 - g["p"])).mean() ** pf["gamma"]
                / g["tau"].mean() ** pf["alpha"])
        rows.append({
            "pi": int(pi), "ti": int(ti),
            "p_bucket": vc._bucket_label(int(pi)), "tau_band": _tau_label(int(ti)),
            "n": len(g), "n_mkts": g["key"].nunique(),
            "still_frac": float(g["still"].mean()),
            "excess_still": float(g["still"].mean() - g["pred_still"].mean()),
            "kurt_z": float(m4 / m2 ** 2 - 3) if m2 > 0 else np.nan,
            "jump_share": float(max(jump, 0.0)) if np.isfinite(jump) else np.nan,
            "ratio_fit": float(g["y"].mean() / pred) if pred > 0 else np.nan,
            "budget_share": float(g["dp2"].sum()),
        })
    out = pd.DataFrame(rows)
    out["budget_share"] /= out["budget_share"].sum()
    out.attrs["power_fit"] = pf
    return out


def classify(cells: pd.DataFrame) -> pd.DataFrame:
    cells = cells.copy()
    r = np.ones(len(cells), int)
    r[cells["kurt_z"] >= KURT_THRESH] = 3
    r[cells["excess_still"] >= STILL_THRESH] = 2   # R2 wins over R3
    cells["regime"] = r
    return cells


def r1_rectangle(cells: pd.DataFrame) -> dict:
    """Largest contiguous p-range / tau-range whose cells are majority R1.

    p-range: scan p buckets, keep those whose long-tau (>= 14d) cells are
    majority R1. tau-range: scan tau bands, keep those whose retained-p cells
    are majority R1.
    """
    long_tau = cells[cells["ti"] >= 2]  # >= 7d as a stable base for the p scan
    p_ok = [pi for pi, g in long_tau.groupby("pi")
            if (g["regime"] == 1).mean() > 0.5]
    mid = cells[cells["pi"].isin(p_ok)]
    t_ok = [ti for ti, g in mid.groupby("ti")
            if (g["regime"] == 1).mean() > 0.5]
    p_lo = vc.P_BUCKETS[min(p_ok)] if p_ok else np.nan
    p_hi = vc.P_BUCKETS[max(p_ok) + 1] if p_ok else np.nan
    tau_lo = TAU_EDGES[min(t_ok)] if t_ok else np.nan
    return {"p_lo": p_lo, "p_hi": p_hi, "tau_lo": tau_lo,
            "p_buckets": p_ok, "tau_bands": t_ok}


def budget_by_regime(cells: pd.DataFrame) -> pd.Series:
    s = cells.groupby("regime")["budget_share"].sum()
    s.index = [REGIME_NAMES[i] for i in s.index]
    return s


def _grid(cells: pd.DataFrame, col: str) -> np.ndarray:
    g = np.full((len(TAU_EDGES) - 1, len(vc.P_BUCKETS) - 1), np.nan)
    for _, r in cells.iterrows():
        g[int(r["ti"]), int(r["pi"])] = r[col]
    return g


def plot_map(maps: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm

    panels = [("excess_still", "excess stillness over diffusive prediction", None),
              ("jump_share", "jump share (bipower)", None),
              ("kurt_z", "excess kurtosis of z (log colour)", "log"),
              ("regime", "regime (R1 diff / R2 itm / R3 jumpy)", "cat")]
    nv = len(maps)
    fig, axes = plt.subplots(nv, 4, figsize=(16, 3.6 * nv), squeeze=False)
    p_labels = [vc._bucket_label(i) for i in range(len(vc.P_BUCKETS) - 1)]
    t_labels = [_tau_label(j) for j in range(len(TAU_EDGES) - 1)]
    for vi, (venue, cells) in enumerate(maps.items()):
        for ci, (col, title, kind) in enumerate(panels):
            ax = axes[vi, ci]
            g = _grid(cells, col)
            if kind == "cat":
                cmap = ListedColormap(["#4c92c3", "#e1a63c", "#c34c4c"])
                im = ax.imshow(g, origin="lower", aspect="auto", cmap=cmap,
                               vmin=1, vmax=3)
            elif kind == "log":
                im = ax.imshow(g, origin="lower", aspect="auto",
                               cmap="viridis", norm=LogNorm(vmin=1, vmax=500))
            else:
                im = ax.imshow(g, origin="lower", aspect="auto",
                               cmap="viridis", vmin=0, vmax=1)
            if kind != "cat":
                fig.colorbar(im, ax=ax, shrink=0.85)
            ax.set_xticks(range(len(p_labels)), p_labels, rotation=45,
                          fontsize=6)
            ax.set_yticks(range(len(t_labels)), t_labels, fontsize=6)
            ax.set_title(f"{venue}: {title}", fontsize=9)
            if ci == 0:
                ax.set_ylabel("tau band")
    fig.suptitle("Regime map: where the walk diffuses, freezes, and jumps",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_venue(name: str, panel: pd.DataFrame) -> pd.DataFrame:
    cells = classify(build_cells(panel))
    pf = cells.attrs["power_fit"]
    print(f"\n=== {name} ===  bulk power fit gamma={pf['gamma']:.2f} "
          f"alpha={pf['alpha']:.2f}")
    show = cells[["p_bucket", "tau_band", "n", "n_mkts", "still_frac",
                  "excess_still", "kurt_z", "jump_share", "ratio_fit",
                  "budget_share", "regime"]].copy()
    show["regime"] = show["regime"].map(REGIME_NAMES)
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nbudget share by regime (traversal variance only):")
    print(budget_by_regime(cells).to_string(float_format=lambda v: f"{v:.3f}"))
    rect = r1_rectangle(cells)
    print(f"\nR1 rectangle for downstream fits: p in [{rect['p_lo']:.2f}, "
          f"{rect['p_hi']:.2f}), tau >= {rect['tau_lo']:.0f}d")
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", default="both",
                    choices=["polymarket", "kalshi", "both"])
    ap.add_argument("--simulate", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    print(f"thresholds: R2 excess_still >= {STILL_THRESH}, "
          f"R3 kurt_z >= {KURT_THRESH}, min cell n = {MIN_CELL}")

    maps: dict[str, pd.DataFrame] = {}
    if args.simulate:
        mkts = vc.simulate(args.simulate, seed=7)
        panel = vc.candidate_rates(vc.daily_panel(mkts))
        maps["sim clean"] = run_venue("sim clean", panel)
    else:
        venues = (["polymarket", "kalshi"] if args.venue == "both"
                  else [args.venue])
        for v in venues:
            mkts = vc.load_markets(v, None)
            panel = vc.candidate_rates(vc.daily_panel(mkts))
            maps[v] = run_venue(v, panel)

    if not args.no_plot and maps:
        out = Path(__file__).resolve().parent.parent / "figures" / "regime_map.png"
        plot_map(maps, out)
        print(f"\nfigure: {out}")
    return maps


if __name__ == "__main__":
    main()
