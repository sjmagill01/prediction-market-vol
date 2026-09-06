"""Generate the README figures into figures/.

Usage: python analysis/readme_figures.py
Heavy (loads both venues' daily panels and the Kalshi hourly panel).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import vol_check as vc            # noqa: E402
from analysis import intraday_decomp as idc     # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

VENUES = {"polymarket": None, "kalshi": None}  # match vol_check CLI defaults


def fig_budget(data: dict) -> None:
    """Realised vs implied lifetime variance: the martingale budget test."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (venue, (a, _)) in zip(axes, data.items()):
        s = a.attrs["slope"]
        ax.scatter(a["iv"], a["rv"], s=6, alpha=0.25, lw=0)
        x = np.linspace(0, 0.25, 50)
        ax.plot(x, x, "k--", lw=1, label="martingale budget (slope 1)")
        ax.plot(x, s * x, "r-", lw=1.2, label=f"fit: slope {s:.2f}")
        ax.set_xlim(0, 0.26)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(r"implied lifetime variance  $p_0(1-p_0)$")
        ax.set_title(f"{venue} ({len(a)} resolved markets)")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel(r"realised lifetime variance  $\sum dp_t^2 + (F-p_T)^2$")
    fig.suptitle("Test A: every market quotes its own implied variance", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "budget.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_state_dependence(data: dict) -> None:
    """Local variance rate vs p, against the candidate walks."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    grid = np.linspace(0.02, 0.98, 200)
    gq = grid * (1 - grid)
    for ax, (venue, (_, panel)) in zip(axes, data.items()):
        bulk = panel[(panel["p"] >= 0.05) & (panel["p"] <= 0.95)]
        pf = vc.power_fit(bulk)
        # empirical: mean tau-scaled dp^2/dt in p bins
        b = bulk.copy()
        b["y"] = b["dp2"] / b["dt"] * b["tau"] ** pf["alpha"]
        b["bin"] = np.digitize(b["p"], np.linspace(0.05, 0.95, 19))
        cell = b.groupby("bin").agg(mp=("p", "mean"), my=("y", "mean"),
                                    n=("y", "size"))
        cell = cell[cell["n"] >= 200]
        ax.plot(cell["mp"], cell["my"], "o", ms=5, color="C0",
                label=r"data: $E[dp^2/dt]\cdot\tau^{\hat\alpha}$")
        z = norm.ppf(grid)

        def match(f):  # geometric-mean level match to the data cells
            fc = np.interp(cell["mp"], grid, f)
            return f * np.exp(np.mean(np.log(cell["my"]) - np.log(fc)))

        for f, lab, stl in ((gq, r"arcsin walk  $p(1-p)$", ":"),
                            (gq ** 2, r"logit walk  $[p(1-p)]^2$", "--"),
                            (norm.pdf(z) ** 2, "probit walk  $\\phi(\\Phi^{-1}(p))^2$", "-.")):
            ax.plot(grid, match(f), stl, lw=1.2, label=lab)
        ax.plot(grid, match(gq ** pf["gamma"]), "r-",
                lw=1.4, label=f"fit: $\\gamma$={pf['gamma']:.2f}, "
                              f"$\\alpha$={pf['alpha']:.2f}")
        ax.set_yscale("log")
        ax.set_xlabel("price p")
        ax.set_title(venue)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("local variance rate (log scale, level-matched)")
    fig.suptitle("Which random walk? state-dependence of the variance rate, "
                 "bulk p in [0.05, 0.95]", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "state_dependence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_intraday() -> None:
    """Kalshi hourly bounce vs jump shares by p bucket."""
    df = idc.load_hourly("kalshi", 50_000)
    pairs = idc.pair_panel(df)
    tab = idc.decompose(pairs, has_quotes=True)
    tab = tab[tab["bucket"] != "all"]
    x = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.27
    ax.bar(x - w, tab["bounce_quote"], w, label="bounce (quote: 1 - RV_mid/RV_trade)")
    ax.bar(x, tab["bounce_roll"], w, label="bounce (Roll: -2$\\gamma_1$/Var)")
    ax.bar(x + w, tab["jump_share_mid"], w, label="jumps (bipower on midpoints)")
    ax.set_xticks(x, tab["bucket"], rotation=30, fontsize=8)
    ax.set_ylabel("share of hourly variance")
    ax.set_title("Kalshi hourly bars: what the walk is made of, by price bucket")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "intraday.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = {}
    for venue, mv in VENUES.items():
        mkts = vc.load_markets(venue, mv)
        a = vc.test_a(mkts)
        panel = vc.candidate_rates(vc.daily_panel(mkts))
        data[venue] = (a, panel)
        print(f"{venue}: {len(a)} markets, slope {a.attrs['slope']:.3f}")
    fig_budget(data)
    print("budget.png")
    fig_state_dependence(data)
    print("state_dependence.png")
    fig_intraday()
    print("intraday.png")


if __name__ == "__main__":
    main()
