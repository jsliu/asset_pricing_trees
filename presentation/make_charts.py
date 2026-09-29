"""
Charts for the backtest presentation, from result/ and the run's report folder (e.g. result/report_full/).
Run analysis/backtest_report.py first, then from the project root: python presentation/make_charts.py [<universe>] [--region GL] [--vw | --variant TAGS]
PNGs are written to presentation/charts/; make_deck.py draws the same charts into the PDF.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "charts"

from analysis.backtest_report import REGION, UNIVERSE, VARIANT, RET_NAME, long_short   # noqa: E402
from src.constants import Chars, DataPaths, tree_variant        # noqa: E402
from src.preprocessing import read_backtest_data                # noqa: E402

SURFACE = "#f7f6f2"
INK, INK2, GRID = "#1f2a3d", "#556070", "#dcdad2"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]                      # categorical slots 1-3
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue ramp, small -> large
SCORES = ["size_oriented_score", "sdf_weight", "norm_score"]
SCORE_NAMES = {"size_oriented_score": "score", "sdf_weight": "sdf_weight", "norm_score": "norm_score (original)"}
SDF_NAMES = {"Return": "Factor-hedged (Return)", "Return_mkt_adj": "Market-adjusted (Return_mkt_adj)"}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 20, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "legend.frameon": False,
})


def to_dt(idx):
    return pd.to_datetime(pd.Index(idx).astype(str), format="%Y%m%d")


def load_inputs():
    """Everything the charts and the deck need, from the backtest outputs and the report tables."""
    paths = DataPaths()
    rep = paths.result_file("report", REGION, UNIVERSE, VARIANT, ext=None)
    rets = pd.read_csv(paths.result_file("ret", REGION, UNIVERSE, VARIANT), index_col=0)
    scores = pd.read_csv(paths.result_file("score", REGION, UNIVERSE, VARIANT)).set_index(["date", "permno"])

    features = list(Chars().__dict__.values())[:-2]
    data = read_backtest_data(features, RET_NAME, region=REGION, universe=UNIVERSE).swaplevel(0, 1).sort_index()
    ret = data[RET_NAME][data.index.get_level_values("date") >= rets.index.min()]
    ls = pd.DataFrame({c: long_short(scores[c], ret) for c in SCORES if c in scores})

    contrib = pd.read_csv(rep / "sdf_return_by_size_by_period.csv").set_index(["size bucket", "period"])
    return {
        "rets": rets,
        "score_ls": ls,
        "sdf_table": pd.read_csv(rep / "sdf_returns_by_period.csv").set_index(["series", "period"]),
        "loadings": pd.read_csv(rep / "sdf_factor_loadings.csv").set_index(["series", "stat"]),
        "perf": pd.read_csv(rep / "score_long_short_by_period.csv").set_index(["universe", "score", "period"]),
        "costs": pd.read_csv(rep / "score_turnover_costs.csv").set_index(["universe", "score"]),
        "diffs": pd.read_csv(rep / "score_difference_tests.csv").set_index(["universe", "comparison", "period"]),
        "contrib": contrib,
        "contrib_monthly": pd.read_csv(rep / "sdf_return_by_size_monthly.csv", index_col=0),
        "exposure": pd.read_csv(rep / "sdf_weight_by_size.csv", index_col=0),
        "buckets": list(contrib.index.get_level_values("size bucket").unique()),
        "node_betas": pd.read_csv(paths.result_file("node_betas", REGION, UNIVERSE, VARIANT)),
        # optional: from hedge_analysis.py and turnover_controls.py
        "hedge": (pd.read_csv(rep / "hedge_costs_by_period.csv").set_index(["portfolio", "period"])
                  if (rep / "hedge_costs_by_period.csv").exists() else None),
        "controls": (pd.read_csv(rep / "turnover_controls_summary.csv").set_index(["portfolio", "control"])
                     if (rep / "turnover_controls_summary.csv").exists() else None),
        "n_tree_files": len(paths.tree_files(REGION, UNIVERSE, tree_variant(VARIANT))),
        "stocks_with_return": ret.groupby("date").size().median(),
        "universe_size": scores.groupby("date").size().median(),
        "n_features": len(features),
    }


def _end_labels(ax, series, fmt="{:.0f}%", skip=()):
    for col, s in series.items():
        if col not in skip:
            ax.annotate(fmt.format(s.iloc[-1]), (s.index[-1], s.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", color=INK)


def plot_sdf_cumulative(ax, d):
    cum = d["rets"].cumsum() * 100
    cum.index = to_dt(cum.index)
    for col, c in zip(cum.columns, CAT):
        ax.plot(cum.index, cum[col], color=c, linewidth=2.5, label=SDF_NAMES.get(col, col))
    _end_labels(ax, cum)
    ax.set_ylabel("Cumulative return, % (sum of monthly)")
    ax.legend(loc="upper left")
    ax.margins(x=0.08)


def plot_score_cumulative(ax, d):
    cum = d["score_ls"].cumsum() * 100
    cum.index = to_dt(cum.index)
    for col, c in zip(cum.columns, CAT):
        ax.plot(cum.index, cum[col], color=c, linewidth=2.5, label=SCORE_NAMES.get(col, col))
    _end_labels(ax, cum)
    ax.set_ylabel("Cumulative return, % (sum of monthly)")
    ax.legend(loc="upper left")
    ax.margins(x=0.08)


def plot_size_contribution(axes, d):
    """Two panels: contribution to Return_mkt_adj and share of gross weight, by size bucket."""
    b = d["buckets"]
    full = d["contrib"].xs("Full", level="period").loc[b, "ann. return %"]
    share = d["exposure"].loc[b, "share of gross weight %"]
    for ax, vals, title, fmt in [
        (axes[0], full, "Contribution to Return_mkt_adj, % a year", "{:.2f}"),
        (axes[1], share, "Share of the SDF's gross weight, %", "{:.0f}"),
    ]:
        bars = ax.bar(b, vals, color=ORD[:len(b)], width=0.62, edgecolor=SURFACE, linewidth=2)
        for bar, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, max(v, 0)), xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom", color=INK)
        ax.set_title(title, loc="left", color=INK, fontsize="large", pad=16)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(min(0, vals.min() * 1.2), vals.max() * 1.2)


def plot_size_cumulative(ax, d):
    b = d["buckets"]
    cum = d["contrib_monthly"][b].cumsum() * 100
    cum.index = to_dt(cum.index)
    for col, c in zip(b, ORD):
        ax.plot(cum.index, cum[col], color=c, linewidth=2.5, label=col)
    # label the lines at the end, merging those that finish too close together to read
    ends = cum.iloc[-1].sort_values()
    span = cum.max().max() - cum.min().min()
    groups, current = [], [ends.index[0]]
    for prev, col in zip(ends.index[:-1], ends.index[1:]):
        if ends[col] - ends[prev] < 0.04 * span:
            current.append(col)
        else:
            groups.append(current)
            current = [col]
    groups.append(current)
    for g in groups:
        ax.annotate(", ".join(g), (cum.index[-1], ends[g].mean()), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK)
    ax.set_ylabel("Cumulative contribution, %")
    ax.legend(loc="upper left", ncol=2)
    ax.margins(x=0.14)


def plot_size_sharpe(ax, d):
    b = d["buckets"]
    x = np.arange(len(b))
    w = 0.26
    perf = d["perf"]
    allv = [perf.loc[(f"size {q}", sc, "Full"), "Sharpe"] for q in b for sc in SCORES]
    ax.set_ylim(min(0, min(allv) * 1.3), max(allv) * 1.35)     # headroom for the one-row legend
    for i, (score, c) in enumerate(zip(SCORES, CAT)):
        vals = [perf.loc[(f"size {q}", score, "Full"), "Sharpe"] for q in b]
        ax.bar(x + (i - 1) * w, vals, width=w, color=c, edgecolor=SURFACE, linewidth=2, label=SCORE_NAMES[score])
        if i == 0:
            for xi, v in zip(x, vals):
                ax.annotate(f"{v:.2f}", (xi - w, v), xytext=(0, 6 if v >= 0 else -6), textcoords="offset points",
                            ha="center", va="bottom" if v >= 0 else "top", color=INK)
    ax.axhline(0, color=INK2, linewidth=1.5)
    ax.set_xticks(x, b)
    ax.set_ylabel("Sharpe ratio, full sample")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=len(SCORES))


def main():
    OUT.mkdir(exist_ok=True)
    d = load_inputs()
    d["score_ls"].to_csv(OUT / "score_ls_monthly.csv")
    for name, size, draw in [
        ("sdf_cumulative.png", (10.4, 7.0), plot_sdf_cumulative),
        ("score_cumulative.png", (10.4, 7.0), plot_score_cumulative),
        ("size_contribution_cumulative.png", (10.4, 7.0), plot_size_cumulative),
        ("size_sharpe.png", (16.6, 6.2), plot_size_sharpe),
    ]:
        fig, ax = plt.subplots(figsize=size)
        draw(ax, d)
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=100, facecolor=SURFACE)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(16.6, 6.2))
    plot_size_contribution(axes, d)
    fig.tight_layout(w_pad=4)
    fig.savefig(OUT / "size_contribution.png", dpi=100, facecolor=SURFACE)
    plt.close(fig)
    print("charts:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
