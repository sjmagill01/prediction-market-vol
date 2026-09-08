"""V6 figures: every README figure regenerated from results_v2/ only.

One palette everywhere: Polymarket blue, Kalshi orange, simulator greys.
Each figure reads the frozen result files (never raw data), so the figures
cannot drift from the numbers the claims registry pins.

Usage: python -m analysis.figures_v6
"""
from __future__ import annotations

import json
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.config import FIGURES, RESULTS  # noqa: E402

C = {"polymarket": "#1f77b4", "kalshi": "#ff7f0e",
     "sim_clean": "#9a9a9a", "sim_noise": "#4d4d4d", "sim_stochvol": "#bdbdbd"}
LBL = {"polymarket": "Polymarket", "kalshi": "Kalshi",
       "sim_clean": "sim: clean martingale", "sim_noise": "sim: +2c noise",
       "sim_stochvol": "sim: stoch. vol"}

plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.titlesize": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def _j(fname: str) -> dict:
    with open(RESULTS / fname) as fh:
        return json.load(fh)


def _mid(bucket: str) -> float:
    lo, hi = bucket.strip("[)").split(",")
    return (float(lo) + float(hi)) / 2


# ------------------------------------------------------------- fig: budget
def fig_budget() -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in ("sim_clean", "sim_noise", "polymarket", "kalshi"):
        df = pd.read_csv(RESULTS / f"empirics_budget_{s}.csv")
        x = df.bucket.map(_mid)
        style = dict(color=C[s], lw=1.8, marker="o", ms=3.5)
        if s.startswith("sim"):
            style.update(lw=1.2, ls="--", marker=None)
        ax.plot(x, df.ratio, label=LBL[s], **style)
        if not s.startswith("sim"):
            ax.fill_between(x, df.ci_lo, df.ci_hi, color=C[s], alpha=0.15,
                            lw=0)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("starting price bucket (midpoint)")
    ax.set_ylabel("realized lifetime variance / p0(1-p0)")
    sc = {v: _j(f"empirics_scalars_{v}.json")["budget_slope"]
          for v in ("polymarket", "kalshi")}
    ax.set_title("Variance budget by starting price "
                 f"(pooled slopes: PM {sc['polymarket']:.2f}, "
                 f"Kalshi {sc['kalshi']:.2f})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "budget.png")
    plt.close(fig)


# --------------------------------------------- fig: state dependence/power
def fig_state_dependence() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    for s in ("sim_clean", "sim_noise", "polymarket", "kalshi"):
        df = pd.read_csv(RESULTS / f"empirics_local_{s}.csv")
        x = df.bucket.map(_mid)
        style = dict(color=C[s], lw=1.8, marker="o", ms=3.5)
        if s.startswith("sim"):
            style.update(lw=1.2, ls="--", marker=None)
        ax1.plot(x, df.ratio_gauss, label=LBL[s], **style)
    ax1.axhline(1.0, color="k", lw=0.8, ls=":")
    ax1.set_yscale("log")
    ax1.set_xlabel("price bucket (midpoint)")
    ax1.set_ylabel("realized / latent-Gaussian rate")
    ax1.set_title("Local rate vs the zero-parameter probit model")
    ax1.legend(frameon=False, fontsize=8)

    marks = {"sim_clean": "s", "sim_noise": "D",
             "polymarket": "o", "kalshi": "o"}
    for s in ("sim_clean", "sim_noise", "polymarket", "kalshi"):
        d = _j(f"empirics_scalars_{s}.json")["power_bulk"]
        ax2.scatter(d["gamma"], d["alpha"], color=C[s], marker=marks[s],
                    s=55, zorder=3, label=LBL[s])
    for g, a, name, off in (
            (1.0, 0.0, "Wright-Fisher", (5, 5)),
            (1.0, 1.0, "binomial", (5, 5)),
            (2.0, 1.0, "logit-Brownian", (-8, -14)),
            (1.6, 1.0, "probit (effective)", (5, 8))):
        ax2.scatter(g, a, color="none", edgecolor="k", s=45, zorder=2)
        ax2.annotate(name, (g, a), textcoords="offset points",
                     xytext=off, fontsize=7)
    ax2.set_xlabel("gamma  (level exponent on p(1-p))")
    ax2.set_ylabel("alpha  (resolution-clock exponent on tau)")
    ax2.set_title("Power family: which walk is it? (bulk fit)")
    ax2.set_xlim(0.2, 2.4)
    ax2.set_ylim(-0.15, 1.3)
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "state_dependence.png")
    plt.close(fig)


# ------------------------------------------------------------ fig: regimes
def fig_regimes() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    cmap = matplotlib.colors.ListedColormap(["#c6dbef", "#fdd49e", "#e34a33"])
    for ax, venue in zip(axes, ("polymarket", "kalshi")):
        df = pd.read_csv(RESULTS / f"regimes_cells_{venue}.csv")
        grid = df.pivot(index="ti", columns="pi", values="regime")
        pb = df.drop_duplicates("pi").sort_values("pi").p_bucket.tolist()
        tb = df.drop_duplicates("ti").sort_values("ti").tau_band.tolist()
        ax.imshow(grid.to_numpy(), cmap=cmap, vmin=1, vmax=3,
                  aspect="auto", origin="lower")
        ax.set_xticks(range(len(pb)), pb, rotation=60, fontsize=6,
                      ha="right")
        ax.set_yticks(range(len(tb)), tb, fontsize=7)
        sc = _j(f"regimes_scalars_{venue}.json")["budget_by_regime"]
        ax.set_title(f"{LBL[venue]}: budget share "
                     f"R1 {sc['R1 diffusive']:.0%} / "
                     f"R2 {sc['R2 itm/hazard']:.0%} / "
                     f"R3 {sc['R3 jumpy']:.0%}")
        ax.set_xlabel("price bucket")
    axes[0].set_ylabel("time-to-resolution band")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in cmap.colors]
    axes[1].legend(handles, ["R1 diffusive", "R2 itm/hazard", "R3 jumpy"],
                   frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "regimes.png")
    plt.close(fig)


# --------------------------------------------------------------- fig: race
def fig_race() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, venue in zip(axes, ("polymarket", "kalshi")):
        d = _j(f"race_summary_{venue}.json")
        models = [m for m in d["models"] if m["model"] != "composite"]
        names = [m["model"] for m in models]
        delta = np.array([m["delta_vs_composite"] for m in models])
        # d_ci_* bounds the bootstrap of (composite - model); the delta's own
        # CI is the sign flip (-d_ci_hi, -d_ci_lo).
        lo = delta - (-np.array([m["d_ci_hi"] for m in models]))
        hi = (-np.array([m["d_ci_lo"] for m in models])) - delta
        y = np.arange(len(models))
        cols = [C[venue] if v >= 0 else "#888888" for v in delta]
        ax.barh(y, delta, xerr=[lo, hi], color=cols, height=0.6,
                error_kw={"lw": 0.8})
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y, names)
        ax.set_title(f"{LBL[venue]} ({d['n_obs']:,} obs)")
        ax.set_xlabel("log score per obs vs composite (higher = better)")
        ax.set_xscale("symlog", linthresh=0.05)
    fig.suptitle("Walk-forward horse race: deltas vs the regime composite "
                 "(95% block-bootstrap CIs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "race.png")
    plt.close(fig)


# ---------------------------------------------------- fig: budget forecast
def fig_budget_v4() -> None:
    engines = [("full_ratio", "SLV composite"), ("zi_ratio", "zi composite"),
               ("clam_ratio", "constant-lambda"), ("real_ratio", "realized")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, venue in zip(axes, ("polymarket", "kalshi")):
        d = _j(f"budget_v4_{venue}.json")
        bands = list(d)
        x = np.arange(len(bands))
        for i, (key, name) in enumerate(engines):
            vals = [d[b][key] for b in bands]
            col = "k" if key == "real_ratio" else C[venue]
            alpha = 1.0 if key == "real_ratio" else 0.35 + 0.25 * i
            ax.bar(x + (i - 1.5) * 0.2, vals, width=0.19, label=name,
                   color=col, alpha=alpha)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_xticks(x, bands)
        ax.set_title(LBL[venue])
        ax.set_xlabel("start state (time to resolution)")
    axes[0].set_ylabel("MC remaining variance / p(1-p)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Budget integration: model engines vs realized continuation",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "budget_v4.png")
    plt.close(fig)


# ------------------------------------------------------------- fig: bridge
def fig_bridge() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    m = pd.read_csv(RESULTS / "bridge_pair_metrics.csv")
    for q, mk in (("exact", "o"), ("approx", "^")):
        sub = m[m.quality == q]
        ax1.scatter(sub.ratio_pm, sub.ratio_k, s=22, marker=mk,
                    color=C["kalshi"] if q == "exact" else "#888888",
                    alpha=0.75, label=f"{q} pair")
    lim = (1e-3, 2e1)
    ax1.plot(lim, lim, color="k", lw=0.8, ls=":")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(*lim)
    ax1.set_ylim(*lim)
    b = _j("bridge_v5.json")
    ax1.set_xlabel("Polymarket budget ratio (same event)")
    ax1.set_ylabel("Kalshi budget ratio (same event)")
    pb = b["paired_budget"]
    ax1.set_title(f"Paired budget: K > PM on {pb['frac_k_gt_pm']:.0%} of "
                  f"pairs (median {pb['median_ratio_of_ratios']:.2f}x)",
                  fontsize=9)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    xc = b["lead_lag_hourly"]["xcorr_dpm_leads_dk"]
    lags = sorted(xc, key=int)
    ax2.bar([int(k) for k in lags], [xc[k] for k in lags],
            color=C["polymarket"], width=0.7)
    ax2.axvline(0, color="k", lw=0.8, ls=":")
    pk = b["lead_lag_hourly"]["pm_predicts_k"]
    kp = b["lead_lag_hourly"]["k_predicts_pm"]
    ax2.set_xlabel("lag (hours; positive = PM change leads Kalshi change)")
    ax2.set_ylabel("cross-correlation")
    ax2.set_title(f"Hourly lead-lag: PM->K coef {pk['cross_lag']:.2f} "
                  f"(t={pk['cross_t']:.0f}) vs K->PM {kp['cross_lag']:.2f}",
                  fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "bridge.png")
    plt.close(fig)


FIGS = [fig_budget, fig_state_dependence, fig_regimes, fig_race,
        fig_budget_v4, fig_bridge]


def main(argv: list[str] | None = None) -> int:
    FIGURES.mkdir(exist_ok=True)
    for fn in FIGS:
        fn()
        print(f"figures_v6: wrote {fn.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
