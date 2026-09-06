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


def _bucket_ratios(a, rng, n_boot=2000):
    """Per-p0-bucket realised/implied ratio of sums with bootstrap CI."""
    b = vc._bucket(a["p0"].to_numpy())
    out = []
    for i in range(len(vc.P_BUCKETS) - 1):
        g = a[b == i]
        if len(g) < 10:
            out.append((i, np.nan, np.nan, np.nan, len(g)))
            continue
        rv, iv = g["rv"].to_numpy(), g["iv"].to_numpy()
        ratio = rv.sum() / iv.sum()
        idx = rng.integers(0, len(g), (n_boot, len(g)))
        boot = rv[idx].sum(axis=1) / iv[idx].sum(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        out.append((i, ratio, lo, hi, len(g)))
    return out


def fig_budget(data: dict) -> None:
    """The martingale budget by starting price: realised / implied variance.

    A ratio-of-sums per p0 bucket says *where* the excess movement lives;
    a scatter of rv on iv with a fitted line hides it (iv is bounded by
    1/4, the cloud is heteroskedastic, and the slope is one number).
    Benchmarks: 1.0 (any martingale) and the same statistic on simulated
    paths with 2c of additive bid-ask noise.
    """
    rng = np.random.default_rng(7)
    sim = vc.test_a(vc.simulate(600, noise=0.02, seed=7))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    series = [("polymarket", data["polymarket"][0], "C0", -0.15),
              ("kalshi", data["kalshi"][0], "C1", 0.0),
              ("sim: clean walk + 2c bounce", sim, "0.55", 0.15)]
    labels = [vc._bucket_label(i) for i in range(len(vc.P_BUCKETS) - 1)]
    for name, a, color, off in series:
        rows = _bucket_ratios(a, rng)
        x = np.array([r[0] for r in rows], float) + off
        y = np.array([r[1] for r in rows])
        lo = np.array([r[2] for r in rows])
        hi = np.array([r[3] for r in rows])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", ms=5, lw=1.2,
                    capsize=2.5, color=color, label=name)
    ax.axhline(1.0, color="k", ls="--", lw=1,
               label="martingale budget (exact for any dynamics)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)), labels, rotation=30, fontsize=8)
    ax.set_xlabel("starting price bucket $p_0$")
    ax.set_ylabel(r"realised / implied lifetime variance"
                  "\n"
                  r"$\sum dp_t^2 + (F-p_T)^2$  vs  $p_0(1-p_0)$")
    ax.set_title("The budget test: how much more the walk moves than a "
                 "martingale may (95% bootstrap CIs)")
    ax.legend(frameon=False, fontsize=9)
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
                                    sd=("y", "std"), n=("y", "size"))
        cell = cell[cell["n"] >= 200]
        sem = cell["sd"] / np.sqrt(cell["n"])
        ax.errorbar(cell["mp"], cell["my"], yerr=1.96 * sem, fmt="o", ms=5,
                    lw=1, capsize=2, color="C0",
                    label=r"data: $E[dp^2/dt]\cdot\tau^{\hat\alpha}$ (95% CI)")
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
    ax.bar(x + w, tab["jump_share_mid"], w,
           label="jumps (bipower on midpoints, share of bounce-free RV)")
    ax.set_xticks(x, tab["bucket"], rotation=30, fontsize=8)
    ax.set_ylabel("share of hourly variance\n(bounce: of trade RV; jumps: of midpoint RV)")
    ax.set_title("Kalshi hourly bars: what the walk is made of, by price bucket")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "intraday.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_strike_strip() -> None:
    """Strike-strip IV vs time to resolution: median + IQR band per series.

    Replaces per-event spaghetti: the cross-event median with an IQR band
    is the readable statement of the IV term structure.
    """
    from analysis import strike_strip as ss
    panel = ss.build_iv_panel(ss.load_strips())
    kind = {s: ss.SERIES[s][0] for s in ss.SERIES}
    panel["iv"] = np.where(panel["series"].map(kind) == "prop",
                           panel["iv_prop"], panel["iv_norm"])
    show = [("KXINXY", "S&P 500 end-of-year", "annualised IV (prop)"),
            ("KXWTIW", "WTI weekly close", "annualised IV (prop)"),
            ("KXCPIYOY", "CPI YoY %", "normal vol (%-pts / sqrt-yr)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (s, title, ylab) in zip(axes, show):
        sub = panel[panel["series"] == s].copy()
        edges = np.quantile(sub["tau_days"], np.linspace(0, 1, 13))
        sub["bin"] = np.searchsorted(np.unique(edges), sub["tau_days"],
                                     side="right")
        g = sub.groupby("bin")["iv"].agg(med="median",
                                         q25=lambda v: v.quantile(0.25),
                                         q75=lambda v: v.quantile(0.75))
        t = sub.groupby("bin")["tau_days"].median()
        ax.plot(t, g["med"], "o-", ms=4, lw=1.3, color="C0", label="median")
        ax.fill_between(t, g["q25"], g["q75"], alpha=0.25, color="C0",
                        label="IQR across event-days")
        ax.set_title(f"{title} ({sub['event'].nunique()} events, "
                     f"{len(sub)} days)", fontsize=10)
        ax.set_xlabel("days to resolution")
        ax.set_ylabel(ylab)
        ax.invert_xaxis()
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Forward-looking implied vol from bracketed strips", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "strike_strip_iv.png", dpi=120, bbox_inches="tight")
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
    fig_strike_strip()
    print("strike_strip_iv.png")


if __name__ == "__main__":
    main()
