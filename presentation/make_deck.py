"""
Builds the backtest presentation as a PDF (16:9 pages) from the backtest outputs and the report tables.
Run backtest_report.py first, then from the project root:
    python presentation/make_deck.py [universe] [--png]
        -> presentation/backtest_{SUFFIX}_{universe}.pdf (universe defaults to 'full');
           --png also writes one PNG per page to presentation/deck_preview/
Numbers, tables and the descriptive sentences are all computed from the current results.
"""
import sys
import textwrap
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import make_charts as mc
from backtest import MIN_NODE_SIZE
from backtest_report import EQUAL_WEIGHTED, N_SIZE_BUCKETS, REG, SUFFIX
from src.constants import Chars, DataPaths, Parameters

W, H, M = 16, 9, 0.9                     # page size and side margin, inches
BG, BAND, CARD, LINE = "#f7f6f2", "#efece4", "#fdfcf9", "#d9d5ca"
NAVY, ACCENT, ACCENT_LIGHT = "#14213d", "#b04a18", "#f2a65a"
TEXT, BODY, MUTED, ON_DARK = "#1f2a3d", "#3e4a5c", "#6b7585", "#c9d3e0"
SERIF = ["Georgia", "DejaVu Serif"]
DEFAULT_SCORE = "size_oriented_score"


def num(v, fmt="{:.2f}"):
    return fmt.format(v).replace("-", "−")


def tstat(v):
    """'t = 1.2' kept on one line when wrapped."""
    return f"t\u00a0=\u00a0{num(v, '{:.1f}')}"


def month(d):
    """YYYYMMDD int as 'Jan 1980'."""
    return f"{pd.to_datetime(str(d), format='%Y%m%d'):%b %Y}"


class Deck:
    def __init__(self, path, preview_dir=None):
        self.pdf = PdfPages(path)
        self.preview_dir = preview_dir
        self.page = 0

    def slide(self, eyebrow=None, title=None, dark=False, footer=True):
        self.page += 1
        fig = plt.figure(figsize=(W, H))
        fig.patch.set_facecolor(NAVY if dark else BG)
        if eyebrow:
            fig.text(M / W, 8.2 / H, eyebrow.upper(), fontsize=13, weight="bold", va="top",
                     color=ACCENT_LIGHT if dark else ACCENT)
        if title:
            fig.text(M / W, 7.8 / H, title, fontsize=32, family=SERIF, weight="bold", va="top",
                     color=BG if dark else NAVY)
        if footer:
            color = "#9fb0c6" if dark else MUTED
            fig.text(M / W, 0.45 / H, "Asset Pricing Trees · backtest review", fontsize=12, color=color)
            fig.text((W - M) / W, 0.45 / H, str(self.page), fontsize=12, color=color, ha="right")
        return fig

    def save(self, fig):
        self.pdf.savefig(fig, facecolor=fig.get_facecolor())
        if self.preview_dir:
            fig.savefig(self.preview_dir / f"page_{self.page:02d}.png", dpi=60, facecolor=fig.get_facecolor())
        plt.close(fig)

    def close(self):
        self.pdf.close()


def text(fig, x, y, s, width, size=16, color=BODY, weight="normal", family=None, line=1.4):
    """Wrapped text with its top-left corner at (x, y) inches; returns the y below it."""
    chars = max(10, int(width * 72 / (size * 0.47)))
    lines = []
    for para in s.split("\n"):
        lines += textwrap.wrap(para, chars) or [""]
    kw = {"family": family} if family else {}
    fig.text(x / W, y / H, "\n".join(lines), fontsize=size, color=color, va="top", linespacing=line, weight=weight, **kw)
    return y - len(lines) * size * line / 72


def box(fig, x, y_top, w, h, fill=CARD, edge=LINE):
    fig.add_artist(Rectangle((x / W, (y_top - h) / H), w / W, h / H, transform=fig.transFigure,
                             facecolor=fill, edgecolor=edge, linewidth=1, zorder=0))


def table(fig, x, y, width, header, rows, widths, align=None, size=15, row_h=0.46, bold=()):
    """Banded table with its top-left corner at (x, y) inches; returns the y below it."""
    align = align or ["left"] + ["right"] * (len(header) - 1)
    lefts = [x + width * sum(widths[:i]) for i in range(len(widths))]

    def cell(yy, i, s, weight, color):
        xx = lefts[i] + 0.12 if align[i] == "left" else lefts[i] + width * widths[i] - 0.12
        fig.text(xx / W, yy / H, s, fontsize=size, ha=align[i], va="center", weight=weight, color=color)

    for i, h in enumerate(header):
        cell(y - row_h / 2, i, h, "bold", NAVY)
    fig.add_artist(Line2D([x / W, (x + width) / W], [(y - row_h) / H] * 2, transform=fig.transFigure, color=LINE))
    for r, row in enumerate(rows):
        top = y - (r + 1) * row_h
        if r % 2 == 0:
            fig.add_artist(Rectangle((x / W, (top - row_h) / H), width / W, row_h / H, transform=fig.transFigure,
                                     facecolor=BAND, edgecolor="none", zorder=0))
        for i, s in enumerate(row):
            cell(top - row_h / 2, i, s, "bold" if r in bold else "normal", TEXT)
    return y - (len(rows) + 1) * row_h


def chart(fig, x, y, w, h):
    """Axes at (x, y) bottom-left, w x h inches."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def build(d, path, preview_dir=None):
    plt.rcParams["font.size"] = 13
    deck = Deck(path, preview_dir)
    rets, sdf, perf, costs, diffs = d["rets"], d["sdf_table"], d["perf"], d["costs"], d["diffs"]
    buckets = d["buckets"]
    y0, y1 = rets.index.min() // 10000, rets.index.max() // 10000
    periods = [p for p in sdf.loc["Return_mkt_adj"].index if p != "Full"]
    mk = sdf.loc[("Return_mkt_adj", "Full")]
    hedged = sdf.loc[("Return", "Full")]
    weighting = "equal" if EQUAL_WEIGHTED else "value"
    n_small = 2
    small_label = f"smallest {n_small * 100 // N_SIZE_BUCKETS}%"
    contrib_full = d["contrib"].xs("Full", level="period")["ann. return %"]
    small_share = contrib_full[buckets[:n_small]].sum() / contrib_full.sum() * 100
    large_weight = d["exposure"].loc[buckets[-n_small:], "share of gross weight %"].sum()
    large_contrib = contrib_full[buckets[-n_small:]].sum()
    cost_all = costs.loc[("all stocks", DEFAULT_SCORE)]
    s50 = perf.loc[("size >= 50th pct", DEFAULT_SCORE, "Full"), "Sharpe"]
    s50_net = costs.loc[("size >= 50th pct", DEFAULT_SCORE), "Sharpe after 10bps"]
    refits = d["node_betas"]["refit_date"].unique()
    features = list(Chars().__dict__.values())[:-2]

    # 1. cover
    fig = deck.slide(dark=True, footer=False)
    fig.text(M / W, 8.1 / H, f"BACKTEST REVIEW · {date.today():%B %Y}".upper(), fontsize=13, weight="bold", color=ACCENT_LIGHT, va="top")
    fig.text(M / W, 5.6 / H, "Asset Pricing Trees", fontsize=64, family=SERIF, weight="bold", color=BG, va="top")
    fig.text(M / W, 4.2 / H, f"Out-of-sample SDF and stock scores, {y0}–{y1}", fontsize=26, color=ON_DARK, va="top")
    fig.text(M / W, 1.0 / H, f"Universe: {REG} · {weighting}-weighted trees · {len(refits)} refits · {len(rets)} months out of sample",
             fontsize=15, color="#9fb0c6")
    deck.save(fig)

    # 2. summary
    fig = deck.slide("Summary", "Headline results")
    cards = [
        (num(mk["Sharpe"]), NAVY, f"Sharpe ratio of the market-adjusted SDF return, {y0}–{y1} ({tstat(mk['t-stat'])})"),
        (num(mk["alpha ann. %"], "{:.1f}") + "%", NAVY, f"Alpha a year against the {len(features)} characteristic factors ({tstat(mk['alpha t (NW)'])})"),
        (f"{small_share:.0f}%", ACCENT, f"Of that return comes from the {small_label} of stocks by market cap"),
        (f"{cost_all['one-way turnover %/mo']:.0f}%", ACCENT, f"One-way monthly turnover of {DEFAULT_SCORE}; break-even cost {cost_all['break-even cost bps']:.0f} bps"),
    ]
    cw = (W - 2 * M - 3 * 0.3) / 4
    for i, (big, color, label) in enumerate(cards):
        x = M + i * (cw + 0.3)
        box(fig, x, 6.9, cw, 3.0)
        fig.text((x + 0.3) / W, 6.55 / H, big, fontsize=44, family=SERIF, weight="bold", color=color, va="top")
        text(fig, x + 0.3, 5.3, label, cw - 0.6, size=15)
    text(fig, M, 3.4, f"Outside the smallest half of the market (size at or above the 50th percentile), the {DEFAULT_SCORE} "
         f"long/short has a Sharpe ratio of {num(s50)} before costs and {num(s50_net)} after 10 bps a trade.",
         W - 2 * M - 1.5, size=18, color=TEXT)
    deck.save(fig)

    # 3. setup
    fig = deck.slide("Setup", "Data and model")
    col_w = (W - 2 * M - 0.8) / 2
    data_items = [
        f"Characteristic files in characteristics/ ({REG} universe); out of sample {y0}–{y1}",
        f"{len(features)} characteristics plus size: {', '.join(features)}",
        f"Median {d['stocks_with_return']:,.0f} stocks a month with returns, {d['universe_size']:,.0f} with every characteristic",
        f"Stock returns net of the {weighting}-weighted market; zero market-cap rows dropped",
    ]
    model_items = [
        f"{d['n_tree_files']} tree sets: size plus each pair of characteristics, depth {Parameters.tree_depth}, "
        f"{Parameters.n_splits}-way splits, nodes of at least {MIN_NODE_SIZE} stocks, {weighting}-weighted",
        f"Elastic-net SDF over {len(Parameters.mean_shrinkage)} mean-shrinkage × {len(Parameters.ridge_lambda)} ridge values, "
        f"{Parameters.cv_splits}-fold time-series CV; the last {Parameters.test_size} months choose the sparsity; "
        f"{Parameters.k_min}–{Parameters.k_max} portfolios",
        f"{len(refits)} refits from {month(refits.min())} to {month(refits.max())}, each on all months before the test month",
    ]
    for x, heading, items in [(M, "Data", data_items), (M + col_w + 0.8, "Model", model_items)]:
        fig.text(x / W, 7.0 / H, heading, fontsize=22, family=SERIF, weight="bold", color=NAVY, va="top")
        y = 6.35
        for item in items:
            fig.text(x / W, y / H, "•", fontsize=16, color=ACCENT, va="top")
            y = text(fig, x + 0.3, y, item, col_w - 0.3, size=16) - 0.2
    deck.save(fig)

    # 4. method
    fig = deck.slide("Method", "How each month is scored, without look-ahead")
    steps = [
        ("Residualise", "Each tree's portfolio returns are regressed on the characteristic factor returns over the training window; residuals r − B·F keep each portfolio's alpha."),
        ("Prune each tree", "An elastic-net SDF is fitted per tree set, in parallel, keeping 5–50 portfolios."),
        ("Combine", "The selected portfolios are pooled and pruned again, giving the final node weights β."),
        ("Score", "β applied to the month's portfolio returns gives the SDF return; spread over each node's stocks it gives the stock scores."),
    ]
    y = 6.9
    for i, (head, body) in enumerate(steps):
        fig.text(M / W, y / H, str(i + 1), fontsize=26, family=SERIF, weight="bold", color=ACCENT, va="top")
        fig.text((M + 0.6) / W, y / H, head, fontsize=17, weight="bold", color=NAVY, va="top")
        y = text(fig, M + 0.6, y - 0.4, body, 7.2, size=15) - 0.3
    x2 = M + 8.6
    box(fig, x2, 6.95, W - M - x2, 4.9)
    fig.text((x2 + 0.35) / W, 6.6 / H, "No look-ahead", fontsize=18, weight="bold", color=NAVY, va="top")
    y = 6.0
    for item in [
        "Models are refitted on all months before the test month; the test month is never in training.",
        "Factor betas are estimated on the same training window and reused until the next refit.",
        "Stock scores use characteristics known at the start of the month.",
        "Months between refits reuse the last fitted models.",
    ]:
        fig.text((x2 + 0.35) / W, y / H, "•", fontsize=15, color=ACCENT, va="top")
        y = text(fig, x2 + 0.65, y, item, W - M - x2 - 1.0, size=15) - 0.15
    deck.save(fig)

    # 5. SDF returns
    fig = deck.slide("SDF returns", f"Out-of-sample SDF returns, {y0}–{y1}")
    mc.plot_sdf_cumulative(chart(fig, M + 0.8, 1.0, 7.6, 5.9), d)
    rows = [
        ["Return, % a year", num(mk["ann. return %"]), num(hedged["ann. return %"])],
        ["Volatility, % a year", num(mk["ann. vol %"]), num(hedged["ann. vol %"])],
        ["Sharpe ratio", num(mk["Sharpe"]), num(hedged["Sharpe"])],
        ["t-stat", num(mk["t-stat"], "{:.1f}"), num(hedged["t-stat"], "{:.1f}")],
        ["Months positive", f"{mk['hit rate %']:.0f}%", f"{hedged['hit rate %']:.0f}%"],
        ["Max drawdown", num(mk["max drawdown %"], "{:.1f}") + "%", num(hedged["max drawdown %"], "{:.1f}") + "%"],
        ["Alpha, % a year (t)", f"{num(mk['alpha ann. %'])} ({mk['alpha t (NW)']:.1f})", f"{num(hedged['alpha ann. %'])} ({hedged['alpha t (NW)']:.1f})"],
        ["Factor R²", num(mk["factor R2"]), num(hedged["factor R2"])],
    ]
    y = table(fig, 9.9, 6.9, W - M - 9.9, ["Full sample", "Market-adj.", "Hedged"], rows, [0.46, 0.27, 0.27], size=14)
    text(fig, 9.9, y - 0.25, "Market-adjusted: SDF weights on tree returns net of the market. Hedged: also net of the factor "
         "exposure estimated in the training window. Alpha against the characteristic factors, Newey-West t.",
         W - M - 9.9, size=12, color=MUTED)
    deck.save(fig)

    # 6. SDF by decade
    by = sdf.loc["Return_mkt_adj"]
    hed = sdf.loc["Return"]
    best, worst = by.loc[periods, "Sharpe"].idxmax(), by.loc[periods, "Sharpe"].idxmin()
    n_pos = int((by.loc[periods, "ann. return %"] > 0).sum())
    fig = deck.slide("SDF by period", f"Positive in {n_pos} of {len(periods)} periods, strongest in the {best}")
    rows = [[p if p != "Full" else f"{y0}–{y1}", num(by.loc[p, "ann. return %"]), num(by.loc[p, "Sharpe"]),
             f"{num(by.loc[p, 'alpha ann. %'])} ({by.loc[p, 'alpha t (NW)']:.1f})", num(by.loc[p, "factor R2"]),
             num(hed.loc[p, "Sharpe"])] for p in ["Full"] + periods]
    y = table(fig, M, 6.9, W - 2 * M, ["Period", "Return, %/yr", "Sharpe", "Alpha, %/yr (t)", "Factor R²", "Hedged Sharpe"],
              rows, [0.2, 0.16, 0.14, 0.2, 0.14, 0.16], size=17, row_h=0.6, bold=(0,))
    last = periods[-1]
    text(fig, M, y - 0.4, f"Market-adjusted SDF return unless stated. Sharpe ratio highest in the {best} ({num(by.loc[best, 'Sharpe'])}) "
         f"and lowest in the {worst} ({num(by.loc[worst, 'Sharpe'])}). In the latest period ({last}) the alpha is "
         f"{num(by.loc[last, 'alpha ann. %'], '{:.1f}')}% a year ({tstat(by.loc[last, 'alpha t (NW)'])}).",
         W - 2 * M - 1.5, size=17)
    deck.save(fig)

    # 7. factor exposure
    lo = d["loadings"].loc["Return_mkt_adj"].drop(columns="const")
    sig = [f for f in lo.columns if abs(lo.loc["t", f]) > 2]
    fig = deck.slide("Factor exposure", "Factor loadings of the market-adjusted SDF return")
    rows = [[f, num(lo.loc["beta", f]), num(lo.loc["t", f], "{:.1f}")] for f in lo.columns]
    table(fig, M, 6.9, 6.5, ["Factor", "Beta", "t"], rows, [0.5, 0.25, 0.25], size=15, row_h=0.5,
          bold=[i for i, f in enumerate(lo.columns) if f in sig])
    box(fig, 8.4, 6.9, W - M - 8.4, 2.2)
    fig.text(8.75 / W, 6.55 / H, num(mk["factor R2"]), fontsize=40, family=SERIF, weight="bold", color=NAVY, va="top")
    text(fig, 8.75, 5.5, f"R² on the {len(lo.columns)} factors: {100 * (1 - mk['factor R2']):.0f}% of the variance is unexplained",
         W - M - 9.1, size=15)
    sig_text = (f"Significant at |t| > 2: {', '.join(sig)}." if sig else "No loading is significant at |t| > 2.")
    text(fig, 8.4, 4.3, sig_text + " Pruning on residual returns keeps the SDF close to factor-neutral without an explicit hedge.",
         W - M - 8.4, size=16)
    deck.save(fig)

    # 8. score definitions
    fig = deck.slide("Stock scores", "Three ways to turn node weights into stock scores")
    defs = [
        ("norm_score", "Node weight × geometric mean of the stock's market-wide characteristic ranks, rank-normalised.", "Original score"),
        ("sdf_weight", "Node weight β spread over the node's stocks by their weight in the node, summed over nodes. Holding it reproduces Return_mkt_adj.", "The SDF portfolio"),
        ("size_oriented_score", "The same β per node, tilted towards the stocks deepest inside the node (ranks within the parent node, in the split's direction).", "Default score"),
        ("size_oriented_norm", "size_oriented_score rank-normalised to ±3.5; stocks in no node sit at a neutral score.", "Combining with factors"),
    ]
    y = 6.9
    fig.text(M / W, y / H, "Score", fontsize=15, weight="bold", color=NAVY, va="top")
    fig.text((M + 3.6) / W, y / H, "How each stock is scored", fontsize=15, weight="bold", color=NAVY, va="top")
    fig.text((M + 11.4) / W, y / H, "Role", fontsize=15, weight="bold", color=NAVY, va="top")
    y -= 0.55
    for i, (name, desc, role) in enumerate(defs):
        lines = textwrap.wrap(desc, int(7.4 * 72 / (15 * 0.47)))
        h = len(lines) * 15 * 1.4 / 72 + 0.35
        if i % 2 == 0:
            box(fig, M, y + 0.15, W - 2 * M, h, fill=BAND, edge="none")
        fig.text((M + 0.12) / W, y / H, name, fontsize=15, weight="bold" if name == DEFAULT_SCORE else "normal", color=TEXT, va="top")
        text(fig, M + 3.6, y, desc, 7.4, size=15, color=TEXT)
        fig.text((M + 11.4) / W, y / H, role, fontsize=15, weight="bold" if name == DEFAULT_SCORE else "normal", color=TEXT, va="top")
        y -= h
    text(fig, M, y - 0.4, "The new scores keep each node's total weight equal to its SDF weight; the original score used raw ranks "
         "whatever side of a split a node was on.", W - 2 * M - 1.5, size=17)
    deck.save(fig)

    # 9. score performance
    fig = deck.slide("Score performance", "Score-weighted long/short, all stocks")
    mc.plot_score_cumulative(chart(fig, M + 0.8, 1.0, 7.6, 5.9), d)
    scores = [s for s in mc.SCORES if (("all stocks", s, "Full") in perf.index)]
    full = {s: perf.loc[("all stocks", s, "Full")] for s in scores}
    rows = [
        ["Return, %/yr"] + [num(full[s]["ann. return %"], "{:.1f}") for s in scores],
        ["Volatility"] + [num(full[s]["ann. vol %"], "{:.1f}") for s in scores],
        ["Sharpe"] + [num(full[s]["Sharpe"]) for s in scores],
        ["Alpha, %/yr"] + [num(full[s]["alpha ann. %"], "{:.1f}") for s in scores],
        ["Alpha t"] + [num(full[s]["alpha t (NW)"], "{:.1f}") for s in scores],
        ["ICIR"] + [num(full[s]["ICIR"]) for s in scores],
        ["Factor R²"] + [num(full[s]["factor R2"]) for s in scores],
    ]
    short = {"size_oriented_score": "Size-orient.", "sdf_weight": "SDF weight", "norm_score": "Original"}
    y = table(fig, 9.9, 6.9, W - M - 9.9, ["All stocks"] + [short[s] for s in scores], rows,
              [0.31] + [0.23] * len(scores), size=13)
    text(fig, 9.9, y - 0.25, "Score-weighted long/short (100% long, 100% short) on same-month returns, before costs.",
         W - M - 9.9, size=12, color=MUTED)
    deck.save(fig)

    # 10. scores by period
    def tilt(p):
        for comp, sign in [(f"{DEFAULT_SCORE} - sdf_weight", 1), (f"sdf_weight - {DEFAULT_SCORE}", -1)]:
            if ("all stocks", comp, p) in diffs.index:
                r = diffs.loc[("all stocks", comp, p)]
                return f"{num(sign * r['mean diff %/mo'], '{:+.2f}')} ({sign * r['t (NW)']:.1f})"
        return "n/a"

    fig = deck.slide("Scores by period", "Score long/short Sharpe ratios by period")
    rows = [[p if p != "Full" else f"{y0}–{y1}"] + [num(perf.loc[("all stocks", s, p), "Sharpe"]) for s in scores]
            + [num(perf.loc[("all stocks", DEFAULT_SCORE, p), "alpha t (NW)"], "{:.1f}"), tilt(p)] for p in ["Full"] + periods]
    y = table(fig, M, 6.9, W - 2 * M, ["Period"] + [short[s] for s in scores] + ["Size-orient. alpha t", "Tilt gain, %/mo (t)"],
              rows, [0.16, 0.15, 0.15, 0.14, 0.18, 0.22], size=17, row_h=0.6, bold=(0,))
    text(fig, M, y - 0.4, f"Tilt gain is {DEFAULT_SCORE} minus sdf_weight, with a Newey-West t: what the within-node tilt adds "
         "to the plain SDF weights.", W - 2 * M - 1.5, size=17)
    deck.save(fig)

    # 11. statement
    fig = deck.slide("Where the return comes from", dark=True)
    fig.text(M / W, 6.9 / H, f"{small_share:.0f}%", fontsize=120, family=SERIF, weight="bold", color=ACCENT_LIGHT, va="top")
    text(fig, M, 4.4, f"of the SDF's market-adjusted return comes from the {small_label} of stocks by market cap",
         11.5, size=28, color=BG, family=SERIF)
    text(fig, M, 2.6, " · ".join(f"{b} {num(contrib_full[b])}% a year" for b in buckets), 13.5, size=16, color=ON_DARK)
    deck.save(fig)

    # 12. size contribution
    fig = deck.slide("Size breakdown", "Contribution and weight by market-cap quintile")
    ax1, ax2 = chart(fig, M + 0.7, 2.0, 6.2, 4.6), chart(fig, M + 8.0, 2.0, 6.2, 4.6)
    mc.plot_size_contribution([ax1, ax2], d)
    text(fig, M, 1.45, f"The largest {n_small * 100 // N_SIZE_BUCKETS}% of stocks hold {large_weight:.0f}% of the SDF's gross weight "
         f"and contribute {num(large_contrib)}% a year.", W - 2 * M, size=17, color=TEXT)
    deck.save(fig)

    # 13. size over time
    fig = deck.slide("Size breakdown over time", "Cumulative contribution by market-cap quintile")
    mc.plot_size_cumulative(chart(fig, M + 0.8, 1.0, 7.6, 5.9), d)
    ex = d["exposure"]
    rows = [[b, f"{ex.loc[b, 'median mkt_cap']:,.0f}", f"{ex.loc[b, 'stocks held/mo']:.0f}", num(contrib_full[b])] for b in buckets]
    y = table(fig, 9.9, 6.9, W - M - 9.9, ["Quintile", "Median cap", "Held", "%/yr"], rows, [0.3, 0.26, 0.2, 0.24], size=14)
    text(fig, 9.9, y - 0.25, "Median market cap in the units of the characteristic files; held = median number of stocks with a "
         "non-zero SDF weight; %/yr = contribution to Return_mkt_adj.", W - M - 9.9, size=12, color=MUTED)
    deck.save(fig)

    # 14. scores within size groups
    fig = deck.slide("Scores within size groups", "Score long/short Sharpe ratio within each quintile")
    mc.plot_size_sharpe(chart(fig, M + 0.7, 1.9, W - 2 * M - 0.8, 4.9), d)
    sh = [perf.loc[(f"size {b}", DEFAULT_SCORE, "Full"), "Sharpe"] for b in buckets]
    text(fig, M, 1.4, f"{DEFAULT_SCORE}: Sharpe ratio {num(sh[0])} in the smallest quintile and {num(sh[-1])} in the largest.",
         W - 2 * M, size=17, color=TEXT)
    deck.save(fig)

    # 15. turnover and costs
    fig = deck.slide("Turnover and costs", f"{DEFAULT_SCORE} after trading costs")
    universes = [u for u in ["all stocks", f"size {buckets[0]}", "size >= 20th pct", "size >= 50th pct", "top 1000"]
                 if (u, DEFAULT_SCORE) in costs.index]
    rows = []
    for u in universes:
        c = costs.loc[(u, DEFAULT_SCORE)]
        label = {"all stocks": "All stocks", "top 1000": "Top 1,000", f"size {buckets[0]}": f"{buckets[0]} only"}.get(u, u)
        rows.append([label.replace("size >=", "Size ≥"),
                     f"{c['stocks scored/mo']:,.0f}", num(perf.loc[(u, DEFAULT_SCORE, "Full"), "Sharpe"]),
                     f"{c['one-way turnover %/mo']:.0f}%", f"{num(c['break-even cost bps'], '{:.0f}')} bps",
                     num(c["Sharpe after 10bps"]), num(c["Sharpe after 25bps"])])
    y = table(fig, M, 6.9, W - 2 * M, ["Universe", "Stocks", "Sharpe", "Turnover", "Break-even", "@ 10 bps", "@ 25 bps"],
              rows, [0.24, 0.12, 0.12, 0.13, 0.15, 0.12, 0.12], size=17, row_h=0.6)
    text(fig, M, y - 0.4, "Turnover is one-way per month, on a full stock × month grid so exits count as sales. Break-even is the "
         "one-way cost per trade that wipes out the return; the last two columns are Sharpe ratios net of that cost per trade.",
         W - 2 * M - 1.5, size=16)
    deck.save(fig)

    # optional: hedged portfolio after costs (hedge_analysis.py)
    if d.get("hedge") is not None:
        hg = d["hedge"]
        alone, hedged_p = "SDF alone (Return_mkt_adj)", "Hedged (Return)"
        fig = deck.slide("Tradable portfolio", "The factor-hedged SDF after trading costs")
        rows = [[p if p != "Full" else f"{y0}–{y1}", num(hg.loc[(alone, p), "Sharpe"]), num(hg.loc[(alone, p), "Sharpe after 10bps"]),
                 num(hg.loc[(hedged_p, p), "Sharpe"]), num(hg.loc[(hedged_p, p), "Sharpe after 10bps"]),
                 num(hg.loc[(hedged_p, p), "Sharpe after 25bps"]), f"{num(hg.loc[(hedged_p, p), 'break-even cost bps'], '{:.0f}')} bps"]
                for p in ["Full"] + periods]
        y = table(fig, M, 6.9, W - 2 * M, ["Period", "SDF alone", "@ 10 bps", "Hedged", "@ 10 bps", "@ 25 bps", "Break-even"],
                  rows, [0.16, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14], size=17, row_h=0.6, bold=(0,))
        f_h = hg.loc[(hedged_p, "Full")]
        text(fig, M, y - 0.4, f"Sharpe ratios. Hedged: the SDF's stock positions minus its factor exposure, held through the "
             f"factor portfolios' stocks, with hedge ratios from the last refit. It turns over "
             f"{f_h['one-way turnover % of book']:.0f}% of its book a month (one-way); costs are per unit traded.",
             W - 2 * M - 1.5, size=16)
        deck.save(fig)

    # optional: turnover controls (turnover_controls.py)
    if d.get("controls") is not None:
        ct = d["controls"].loc["Hedged"]
        base = ct.loc["rebalance 100% a month"]
        fig = deck.slide("Turnover controls", "Slower trading of the hedged portfolio")
        rows = [[c, f"{ct.loc[c, 'one-way turnover % of book']:.0f}%", num(ct.loc[c, "Sharpe"]),
                 f"{num(ct.loc[c, 'break-even cost bps'], '{:.0f}')} bps", num(ct.loc[c, "Sharpe after 10bps"]),
                 num(ct.loc[c, "Sharpe after 25bps"])] for c in ct.index]
        y = table(fig, M, 6.9, W - 2 * M, ["Trading rule", "Turnover", "Sharpe", "Break-even", "@ 10 bps", "@ 25 bps"],
                  rows, [0.3, 0.14, 0.14, 0.15, 0.13, 0.14], size=17, row_h=0.6)
        best10, best25 = ct["Sharpe after 10bps"].idxmax(), ct["Sharpe after 25bps"].idxmax()
        text(fig, M, y - 0.4, f"Monthly rebalancing: Sharpe {num(base['Sharpe after 10bps'])} after 10 bps and "
             f"{num(base['Sharpe after 25bps'])} after 25 bps. Best after 10 bps: {best10} ({num(ct.loc[best10, 'Sharpe after 10bps'])}); "
             f"after 25 bps: {best25} ({num(ct.loc[best25, 'Sharpe after 25bps'])}). Partial rebalancing moves a fraction of the "
             "way to the new target each month; averaging holds the mean of recent targets.", W - 2 * M - 1.5, size=16)
        deck.save(fig)

    # 16. caveats
    fig = deck.slide("Caveats", "Read the numbers with these in mind")
    items = [
        ("Size concentration", f"{small_share:.0f}% of the SDF return comes from the {small_label} of stocks."),
        ("Gross of costs", f"Headline numbers ignore about {cost_all['one-way turnover %/mo']:.0f}% one-way turnover a month."),
        ("Alpha benchmark", "The characteristic factors weight all stocks, so alphas of large-cap portfolios against them can overstate."),
        ("Factor normaliser", "Factor returns and the hedge depend on rank_normalise; results from a stand-in version will differ slightly."),
        ("Selection effect", "Tree portfolios are filtered by their missing-data rate over the full sample."),
        ("One sample", f"The {REG} universe, {y0}–{y1}, one configuration; nothing tested beyond it."),
    ]
    cw = (W - 2 * M - 2 * 0.3) / 3
    for i, (head, body) in enumerate(items):
        x, top = M + (i % 3) * (cw + 0.3), 6.9 - (i // 3) * 2.75
        box(fig, x, top, cw, 2.45)
        fig.text((x + 0.3) / W, (top - 0.3) / H, head, fontsize=17, weight="bold", color=NAVY, va="top")
        text(fig, x + 0.3, top - 0.85, body, cw - 0.6, size=15)
    deck.save(fig)

    deck.close()


if __name__ == "__main__":
    out = mc.REPO / "presentation" / DataPaths().result_file("backtest", REG, SUFFIX, "pdf").name
    preview = None
    if "--png" in sys.argv:
        preview = mc.REPO / "presentation" / "deck_preview"
        preview.mkdir(exist_ok=True)
    build(mc.load_inputs(), out, preview)
    print(f"Saved {out}" + (f" and page previews in {preview}" if preview else ""))
