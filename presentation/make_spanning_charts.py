# %%
"""
Charts for the factor-hedged score (analysis/factor_spanning.py).

Run analysis/factor_spanning.py for the run first, then from the project root:
    python presentation/make_spanning_charts.py [<universe>] [--variant TAGS]     e.g. ... largecap
PNGs go to presentation/charts/: signal_hedged_cumulative.png, signal_spanning_hedged.png, signal_hedged_sharpe.png.
make_deck.py draws the same charts into the "hedged score" slides.

Kept separate from presentation/make_charts.py (and free of the repo's external data modules) so the hedged-score
charts can be drawn from the saved report tables on their own.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "charts"

from src.constants import DataPaths    # noqa: E402

SURFACE, INK, INK2, GRID = "#f7f6f2", "#1f2a3d", "#556070", "#dcdad2"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]                      # categorical slots 1-3

# applied by main() only, so that importing this module (make_deck.py) has no styling side effects
STYLE = {
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 20, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "legend.frameon": False,
}


def to_dt(idx):
    return pd.to_datetime(pd.Index(idx).astype(str), format="%Y%m%d")


def load_hedged_inputs(region=None, universe=None, variant=None, report_dir=None, score=None, label='Hedged score'):
    """The tables analysis/factor_spanning.py writes for a score (None = the default score), or None if the run
    has not been analysed yet. label names the hedged score in the charts."""
    rep = report_dir or DataPaths().result_file('report', region, universe, variant, ext=None)
    name = (lambda stem: f'{stem}.csv') if score in (None, 'size_oriented_score') else (lambda stem: f'{stem}_{score}.csv')
    if not (rep / name('signal_spanning_sharpe')).exists():
        return None
    decay_file = rep / name('signal_score_decay')
    decomp_file = rep / name('signal_hedge_decomposition')
    return {
        'monthly': pd.read_csv(rep / name('signal_hedged_monthly'), index_col=0)['hedged'],
        'table': pd.read_csv(rep / name('signal_hedged_by_period'), index_col=0),
        'spanning': pd.read_csv(rep / name('signal_spanning_sharpe'), index_col=0),
        'decay': pd.read_csv(decay_file, index_col=0) if decay_file.exists() else None,
        'decay_raw': (pd.read_csv(rep / name('signal_score_decay_unhedged'), index_col=0)
                      if (rep / name('signal_score_decay_unhedged')).exists() else None),
        'decomp': pd.read_csv(decomp_file, index_col=0) if decomp_file.exists() else None,
        'dir': rep,
        'label': label,
    }


def _value_label(ax, rect, value, fmt="{:+.2f}", fontsize=15):
    ax.annotate(fmt.format(value), (rect.get_x() + rect.get_width() / 2, value),
                xytext=(0, 6 if value >= 0 else -6), textcoords="offset points", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=fontsize, color=INK)


def plot_hedged_cumulative(ax, d):
    """Cumulative return of the factor-hedged score, per cent, summed monthly."""
    cum = d['monthly'].cumsum() * 100
    cum.index = to_dt(cum.index)
    ax.plot(cum.index, cum.to_numpy(), color=CAT[0], linewidth=2.5, label=d.get('label', 'Hedged score'))
    ax.axhline(0, color=INK2, linewidth=1)
    ax.annotate(f"{cum.iloc[-1]:.0f}%", (cum.index[-1], cum.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK)
    ax.set_ylabel("Cumulative return, % (sum of monthly)")
    ax.legend(loc="upper left")
    ax.margins(x=0.08)


def plot_spanning_hedged(ax, d):
    """Sharpe ratio of the tangency portfolio of the 9 characteristic factors, alone and with the hedged score added."""
    s = d['spanning']
    periods = ["Full", "pre-2000", "2000-2016"]
    x = np.arange(len(periods))
    w = 0.36
    base = s.loc[periods, 'factors only SR'].to_numpy(dtype=float)
    hedg = s.loc[periods, 'hedged SR'].to_numpy(dtype=float)
    ax.bar(x - w / 2, base, width=w, color=CAT[0], edgecolor=SURFACE, linewidth=2, label='Factors only')
    bars = ax.bar(x + w / 2, hedg, width=w, color=CAT[1], edgecolor=SURFACE, linewidth=2,
                  label="Factors + hedged")
    for rect, v in zip(bars, hedg):
        _value_label(ax, rect, v, "{:.2f}")
    for j, p in enumerate(periods):
        ax.annotate(f"Δ {s.loc[p, 'dSharpe']:+.2f} (t {s.loc[p, 'JK t']:.1f})",
                    (j, max(base[j], hedg[j])), xytext=(0, 22), textcoords="offset points",
                    ha="center", va="bottom", fontsize=14, color=INK2)
    ax.set_ylim(0, max(base.max(), hedg.max()) * 1.45)                 # room for the labels and the legend
    ax.set_xticks(x, ["Full sample", "Pre-2000", "2000-2016"])
    ax.set_ylabel("Sharpe ratio, in-sample tangency")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")


def plot_hedged_sharpe_by_period(ax, d):
    """Sharpe ratio of the hedged score in each decade."""
    t = d['table']
    periods = [p for p in t.index if p != 'Full']
    vals = t.loc[periods, 'Sharpe'].to_numpy(dtype=float)
    bars = ax.bar(np.arange(len(periods)), vals, width=0.52, color=CAT[1], edgecolor=SURFACE, linewidth=2)
    for rect, v in zip(bars, vals):
        _value_label(ax, rect, v)
    ax.axhline(0, color=INK2, linewidth=1.5)
    ax.set_xticks(np.arange(len(periods)), periods)
    ax.set_ylabel("Sharpe ratio, hedged score")
    ax.set_ylim(0, vals.max() * 1.25)
    ax.grid(axis="x", visible=False)


# several hedged scores on one chart: ds is a list of load_hedged_inputs() results, each with its 'label' and 'color'

def plot_hedged_cumulative_multi(ax, ds):
    """Cumulative returns of several hedged scores; the legend gives each one's total."""
    for d in ds:
        cum = d['monthly'].cumsum() * 100
        cum.index = to_dt(cum.index)
        ax.plot(cum.index, cum.to_numpy(), color=d['color'], linewidth=2.3, label=f"{d['label']} ({cum.iloc[-1]:.0f}%)")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylabel("Cumulative return, % (sum of monthly)")
    ax.legend(loc="upper left", fontsize=15)


def plot_spanning_multi(ax, ds):
    """Tangency Sharpe ratio of the factors alone and with each hedged score added, per period."""
    periods = ["Full", "pre-2000", "2000-2016"]
    x = np.arange(len(periods))
    w = 0.8 / (len(ds) + 1)
    base = ds[0]['spanning'].loc[periods, 'factors only SR'].to_numpy(dtype=float)
    ax.bar(x - 0.4 + w / 2, base, width=w, color="#b9c2cf", edgecolor=SURFACE, linewidth=1.5, label='Factors only')
    top = base.max()
    for i, d in enumerate(ds):
        vals = d['spanning'].loc[periods, 'hedged SR'].to_numpy(dtype=float)
        bars = ax.bar(x - 0.4 + w * (i + 1.5), vals, width=w, color=d['color'], edgecolor=SURFACE, linewidth=1.5,
                      label=f"+ {d['label'].lower()}")
        for rect, v in zip(bars, vals):
            _value_label(ax, rect, v, "{:.2f}", fontsize=13)
        top = max(top, vals.max())
    ax.set_ylim(0, top * 1.45)
    ax.set_xticks(x, ["Full sample", "Pre-2000", "2000-2016"])
    ax.set_ylabel("Sharpe ratio, in-sample tangency")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=14, ncol=2)


def plot_hedged_sharpe_by_period_multi(ax, ds):
    """Sharpe ratio of each hedged score in each decade."""
    periods = [p for p in ds[0]['table'].index if p != 'Full']
    x = np.arange(len(periods))
    w = 0.8 / len(ds)
    top = 0
    for i, d in enumerate(ds):
        vals = d['table'].reindex(periods)['Sharpe'].to_numpy(dtype=float)
        bars = ax.bar(x - 0.4 + w * (i + 0.5), vals, width=w, color=d['color'], edgecolor=SURFACE, linewidth=1.5,
                      label=d['label'])
        for rect, v in zip(bars, vals):
            _value_label(ax, rect, v, "{:.2f}", fontsize=12)
        top = max(top, np.nanmax(vals))
    ax.axhline(0, color=INK2, linewidth=1.5)
    ax.set_xticks(x, periods)
    ax.set_ylabel("Sharpe ratio, hedged")
    ax.set_ylim(0, top * 1.35)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=15)


def plot_score_decay_multi(ax, ds):
    """Horizon retention of each score: the held-k-month long/short return as a share of the 1-month return."""
    horizons = [1, 3, 6, 12]
    x = np.arange(len(horizons))
    w = 0.8 / max(len(ds), 1)
    lo, hi = 0.0, 1.0
    for i, d in enumerate(ds):
        if d.get('decay') is None:
            continue
        vals = d['decay'].reindex(horizons)['retention'].to_numpy(dtype=float)
        bars = ax.bar(x - 0.4 + w * (i + 0.5), vals, width=w, color=d['color'], edgecolor=SURFACE, linewidth=1.5,
                      label=d['label'])
        for rect, v in zip(bars, vals):
            _value_label(ax, rect, v, "{:.2f}", fontsize=11)
        lo, hi = min(lo, np.nanmin(vals)), max(hi, np.nanmax(vals))
    ax.axhline(0, color=INK2, linewidth=1.5)
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle='--')
    ax.set_ylim(lo - 0.2 * (hi - lo), hi + 0.15 * (hi - lo))           # room for the value labels
    ax.set_xticks(x, [f"Month {h}" for h in horizons])
    ax.set_ylabel("Return in month k / month-1 return")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=13)   # above the plot


def plot_hedge_decomposition_multi(ax, ds):
    """Sharpe ratio of each score's hedged long/short as size and market beta are added to the hedge."""
    specs = [sp for sp in ['9 factors', '9 + size', '9 + size + beta']   # those the run has (company data: no beta)
             if any(d.get('decomp') is not None and sp in d['decomp'].index for d in ds)]
    x = np.arange(len(specs))
    w = 0.8 / max(len(ds), 1)
    top = 0
    for i, d in enumerate(ds):
        if d.get('decomp') is None:
            continue
        vals = d['decomp'].reindex(specs)['Sharpe'].to_numpy(dtype=float)
        bars = ax.bar(x - 0.4 + w * (i + 0.5), vals, width=w, color=d['color'], edgecolor=SURFACE, linewidth=1.5,
                      label=d['label'])
        for rect, v in zip(bars, vals):
            _value_label(ax, rect, v, "{:.2f}", fontsize=12)
        top = max(top, np.nanmax(vals))
    ax.set_xticks(x, [{"9 factors": "9 factors", "9 + size": "+ size", "9 + size + beta": "+ size + beta"}[sp] for sp in specs])
    ax.set_ylabel("Sharpe ratio, hedged long/short")
    ax.set_ylim(0, top * 1.15)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=13)   # above the plot


# the hedged score by size quintile (analysis/factor_spanning.py, section 6)

def load_hedged_by_size(region=None, universe=None, variant=None, report_dir=None):
    """signal_hedged_by_size.csv of a run (the default score), or None if it has not been computed."""
    rep = report_dir or DataPaths().result_file('report', region, universe, variant, ext=None)
    f = rep / 'signal_hedged_by_size.csv'
    return pd.read_csv(f, index_col=0) if f.exists() else None


def _paired_bars(ax, labels, first, second, names, colors, fmt):
    """Two bars per size quintile, each labelled with its value."""
    x = np.arange(len(labels))
    w = 0.38
    for off, vals, name, color in [(-w / 2, first, names[0], colors[0]), (w / 2, second, names[1], colors[1])]:
        bars = ax.bar(x + off, vals, width=w, color=color, edgecolor=SURFACE, linewidth=1.5, label=name)
        for rect, v in zip(bars, vals):
            _value_label(ax, rect, v, fmt, fontsize=12)
    ax.axhline(0, color=INK2, linewidth=1.2)
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=13)
    lo, hi = min(0, np.nanmin(first), np.nanmin(second)), max(np.nanmax(first), np.nanmax(second))
    ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.15 * (hi - lo))


def plot_size_share(ax, labels, sdf_share, hedged_share):
    """Share of the market-adjusted SDF's and of the hedged score's return from each size quintile, %."""
    _paired_bars(ax, labels, sdf_share, hedged_share, ["Market-adjusted SDF", "Factor-hedged score"],
                 ["#b9c2cf", CAT[0]], "{:.0f}%")
    ax.set_ylabel("Share of the return, %")


def plot_size_sharpe_within(ax, labels, raw, hedged):
    """Sharpe ratio of the score's long/short built within each size quintile, before and after the factor hedge."""
    _paired_bars(ax, labels, raw, hedged, ["Score, unhedged", "Score, factor-hedged"], ["#b9c2cf", CAT[0]], "{:.2f}")
    ax.set_ylabel("Sharpe ratio within the quintile")


def main():
    import matplotlib
    matplotlib.use("Agg")
    plt.rcParams.update(STYLE)
    UNIVERSE, VARIANT = 'full', None
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    UNIVERSE = args[0] if args else 'full'
    VARIANT = None
    if '--variant' in sys.argv:
        VARIANT = sys.argv[sys.argv.index('--variant') + 1]
    OUT.mkdir(exist_ok=True)
    d = load_hedged_inputs(universe=UNIVERSE, variant=VARIANT)
    if d is None:
        raise SystemExit(f'no hedged-score tables for {UNIVERSE}/{VARIANT} - run analysis/factor_spanning.py first')
    for name, size, draw in [("signal_hedged_cumulative.png", (10.4, 6.4), plot_hedged_cumulative),
                             ("signal_spanning_hedged.png", (10.4, 6.4), plot_spanning_hedged),
                             ("signal_hedged_sharpe.png", (10.4, 6.4), plot_hedged_sharpe_by_period)]:
        fig, ax = plt.subplots(figsize=size)
        draw(ax, d)
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=100, facecolor=SURFACE)
        plt.close(fig)
    print("charts:", sorted(p.name for p in OUT.glob("signal_*.png")))


if __name__ == "__main__":
    main()
