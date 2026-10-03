"""
Builds the backtest presentation as a PDF (16:9 pages) from the backtest outputs and the report tables.
Run analysis/backtest_report.py first, then from the project root:
    python presentation/make_deck.py [<universe>] [--region GL] [--variant TAGS] [--png]
        -> presentation/[<region>_]backtest[_<universe>][_vw].pdf (universe defaults to 'full');
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
from analysis.backtest_report import EQUAL_WEIGHTED, FEATURES, N_SIZE_BUCKETS, REGION, UNIVERSE, LABEL, VARIANT
from src.constants import DataPaths, Parameters

W, H, M = 16, 9, 0.9                     # page size and side margin, inches
BG, BAND, CARD, LINE = "#f7f6f2", "#efece4", "#fdfcf9", "#d9d5ca"
NAVY, ACCENT, ACCENT_LIGHT = "#14213d", "#b04a18", "#f2a65a"
TEXT, BODY, MUTED, ON_DARK = "#1f2a3d", "#3e4a5c", "#6b7585", "#c9d3e0"
SERIF = ["Georgia", "DejaVu Serif"]
DEFAULT_SCORE = "size_oriented_score"
SCORE_LABEL = "score"           # name of DEFAULT_SCORE on the slides


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


CELL_PAD = 0.12          # inches of padding inside each side of a table cell
CELL_LINE = 1.25         # line spacing of wrapped table cells


def wrap_cell(s, col_width, size, bold=False):
    """Lines of a cell's text wrapped to its column width (a word longer than the column stays whole)."""
    chars = max(4, int((col_width - 2 * CELL_PAD) * 72 / (size * (0.53 if bold else 0.48))))
    return textwrap.wrap(str(s), chars, break_long_words=False) or [""]


def table_height(width, header, rows, widths, size=15, row_h=0.46, bold=()):
    """Height in inches that table() will take, with wrapped cells."""
    line_h = size * CELL_LINE / 72

    def row_height(cells, is_bold):
        lines = max(len(wrap_cell(s, width * widths[i], size, is_bold)) for i, s in enumerate(cells))
        return max(row_h, lines * line_h + (row_h - line_h))

    return row_height(header, True) + sum(row_height(row, r in bold) for r, row in enumerate(rows))


def table(fig, x, y, width, header, rows, widths, align=None, size=15, row_h=0.46, bold=()):
    """
    Banded table with its top-left corner at (x, y) inches; returns the y below it. Text that is too long for its
    column wraps onto more lines, and the row grows to fit (row_h is the height of a one-line row).
    """
    align = align or ["left"] + ["right"] * (len(header) - 1)
    lefts = [x + width * sum(widths[:i]) for i in range(len(widths))]
    line_h = size * CELL_LINE / 72

    def draw_row(top, cells, weight, color, band=False):
        wrapped = [wrap_cell(s, width * widths[i], size, weight == "bold") for i, s in enumerate(cells)]
        height = max(row_h, max(len(w) for w in wrapped) * line_h + (row_h - line_h))
        if band:
            fig.add_artist(Rectangle((x / W, (top - height) / H), width / W, height / H, transform=fig.transFigure,
                                     facecolor=BAND, edgecolor="none", zorder=0))
        for i, lines in enumerate(wrapped):
            xx = lefts[i] + CELL_PAD if align[i] == "left" else lefts[i] + width * widths[i] - CELL_PAD
            fig.text(xx / W, (top - height / 2) / H, "\n".join(lines), fontsize=size, ha=align[i], va="center",
                     multialignment=align[i], linespacing=CELL_LINE, weight=weight, color=color)
        return top - height

    top = draw_row(y, header, "bold", NAVY)
    fig.add_artist(Line2D([x / W, (x + width) / W], [top / H] * 2, transform=fig.transFigure, color=LINE))
    for r, row in enumerate(rows):
        top = draw_row(top, row, "bold" if r in bold else "normal", TEXT, band=(r % 2 == 0))
    return top


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
    cost_all = costs.loc[("all stocks", DEFAULT_SCORE)]
    s50 = perf.loc[("size >= 50th pct", DEFAULT_SCORE, "Full"), "Sharpe"]
    s50_net = costs.loc[("size >= 50th pct", DEFAULT_SCORE), "Sharpe after 10bps"]
    refits = d["node_betas"]["refit_date"].unique()
    features = FEATURES

    # 1. cover
    fig = deck.slide(dark=True, footer=False)
    fig.text(M / W, 8.1 / H, f"BACKTEST REVIEW · {date.today():%B %Y}".upper(), fontsize=13, weight="bold", color=ACCENT_LIGHT, va="top")
    model, model_desc = mc.describe(REGION, UNIVERSE, VARIANT)
    fig.text(M / W, 5.6 / H, "Asset Pricing Trees", fontsize=64, family=SERIF, weight="bold", color=BG, va="top")
    fig.text(M / W, 4.2 / H, f"{model}: out-of-sample SDF and stock scores, {y0}–{y1}", fontsize=26, color=ON_DARK, va="top")
    fig.text(M / W, 1.0 / H, f"{model_desc} · {len(refits)} refits · {len(rets)} months out of sample",
             fontsize=15, color="#9fb0c6")
    deck.save(fig)

    # 2. summary
    fig = deck.slide("Summary", "Headline results")
    cards = [
        (num(mk["Sharpe"]), NAVY, f"Sharpe ratio of the market-adjusted SDF return, {y0}–{y1} ({tstat(mk['t-stat'])})"),
        (num(mk["alpha ann. %"], "{:.1f}") + "%", NAVY, f"Alpha a year against the {len(features)} characteristic factors ({tstat(mk['alpha t (NW)'])})"),
        (f"{small_share:.0f}%", ACCENT, f"Of that return comes from the {small_label} of stocks by market cap"),
        (f"{cost_all['one-way turnover %/mo']:.0f}%", ACCENT, f"One-way monthly turnover of the {SCORE_LABEL}; break-even cost {cost_all['break-even cost bps']:.0f} bps"),
    ]
    cw = (W - 2 * M - 3 * 0.3) / 4
    for i, (big, color, label) in enumerate(cards):
        x = M + i * (cw + 0.3)
        box(fig, x, 6.9, cw, 3.0)
        fig.text((x + 0.3) / W, 6.55 / H, big, fontsize=44, family=SERIF, weight="bold", color=color, va="top")
        text(fig, x + 0.3, 5.3, label, cw - 0.6, size=15)
    text(fig, M, 3.4, f"Outside the smallest half of the market (size at or above the 50th percentile), the {SCORE_LABEL} "
         f"long/short has a Sharpe ratio of {num(s50)} before costs and {num(s50_net)} after 10 bps a trade.",
         W - 2 * M - 1.5, size=18, color=TEXT)
    deck.save(fig)

    # 3. setup
    fig = deck.slide("Setup", "Data and model")
    col_w = (W - 2 * M - 0.8) / 2
    data_items = [
        (f"Characteristic files in characteristics/ ({UNIVERSE} universe)" if UNIVERSE else f"Company factor data, region {REGION}")
        + f"; out of sample {y0}–{y1}",
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
    lo = d["loadings"].loc["Return_mkt_adj"].drop(columns="const")            # market-adjusted SDF
    lh = d["loadings"].loc["Return"].drop(columns="const")                    # factor-hedged SDF
    sig = [f for f in lo.columns if abs(lo.loc["t", f]) > 2]
    sig_h = [f for f in lh.columns if abs(lh.loc["t", f]) > 2]
    fig = deck.slide("Factor exposure", "Factor loadings of the market-adjusted and factor-hedged SDF")
    rows = [[f, num(lo.loc["beta", f]), num(lo.loc["t", f], "{:.1f}"), num(lh.loc["beta", f]), num(lh.loc["t", f], "{:.1f}")]
            for f in lo.columns]
    rows.append(["Factor R²", num(mk["factor R2"]), "", num(hedged["factor R2"]), ""])
    tw = 8.6
    widths_lo = [0.28, 0.18, 0.18, 0.18, 0.18]
    for label_, first in [("Market-adjusted", 1), ("Factor-hedged", 3)]:             # labels over the column groups
        gx = M + tw * sum(widths_lo[:first])
        fig.text((gx + 0.12) / W, 7.05 / H, label_, fontsize=14, weight="bold", color=ACCENT, va="bottom")
        fig.add_artist(Line2D([(gx + 0.1) / W, (gx + tw * sum(widths_lo[first:first + 2]) - 0.1) / W], [7.0 / H, 7.0 / H],
                              color=ACCENT, linewidth=1))
    table(fig, M, 6.9, tw, ["Factor", "Beta", "t", "Beta", "t"], rows, widths_lo, size=14, row_h=0.46,
          bold=[len(rows) - 1])
    x0 = M + tw + 0.5
    for i, (value, label_) in enumerate([(mk["factor R2"], "R² of the market-adjusted SDF on the factors"),
                                         (hedged["factor R2"], "R² of the factor-hedged SDF on the factors")]):
        top = 6.9 - i * 2.05
        box(fig, x0, top, W - M - x0, 1.8)
        fig.text((x0 + 0.3) / W, (top - 0.3) / H, num(value), fontsize=34, family=SERIF, weight="bold",
                 color=NAVY if i == 0 else ACCENT, va="top")
        text(fig, x0 + 0.3, top - 1.2, label_, W - M - x0 - 0.6, size=13)
    sig_text = (f"Market-adjusted: {', '.join(sig)} significant at |t| > 2." if sig else
                "Market-adjusted: no loading significant at |t| > 2.")
    sig_text += (f" Hedged: {', '.join(sig_h)} significant" if sig_h else " Hedged: none significant")
    text(fig, x0, 2.65, sig_text + f", because its hedge ratios come from the training window: the hedge removes most "
         f"of the exposure out of sample (R² {num(mk['factor R2'])} → {num(hedged['factor R2'])}), not all of it.",
         W - M - x0, size=13)
    deck.save(fig)

    # 8. score definitions
    fig = deck.slide("Stock scores", "Three ways to turn node weights into stock scores")
    defs = [
        ("norm_score", "Node weight × geometric mean of the stock's market-wide characteristic ranks, rank-normalised.", "Original score"),
        ("sdf_weight", "Node weight β spread over the node's stocks by their weight in the node, summed over nodes. Holding it reproduces Return_mkt_adj.", "The SDF portfolio"),
        (SCORE_LABEL, "The same β per node, tilted towards the stocks deepest inside the node (ranks within the parent node, in the split's direction). size_oriented_score in the result files.", "Default score"),
        ("score, normalised", "The score rank-normalised to ±3.5; stocks in no node sit at a neutral score.", "Combining with factors"),
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
        fig.text((M + 0.12) / W, y / H, name, fontsize=15, weight="bold" if name == SCORE_LABEL else "normal", color=TEXT, va="top")
        text(fig, M + 3.6, y, desc, 7.4, size=15, color=TEXT)
        fig.text((M + 11.4) / W, y / H, role, fontsize=15, weight="bold" if name == SCORE_LABEL else "normal", color=TEXT, va="top")
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
    short = {"size_oriented_score": "Score", "sdf_weight": "SDF weight", "norm_score": "Original"}
    y = table(fig, 9.9, 6.9, W - M - 9.9, ["All stocks"] + [short[s] for s in scores], rows,
              [0.31] + [0.23] * len(scores), size=13)
    text(fig, 9.9, y - 0.25, "Score-weighted long/short (100% long, 100% short) on same-month returns, before costs.",
         W - M - 9.9, size=12, color=MUTED)
    deck.save(fig)

    # 9b. the three scores with their factor exposure removed (analysis/factor_spanning.py), side by side
    import make_spanning_charts as msc
    hedged = {}                                      # score -> its hedged-score tables, in the order of `scores`
    for s_ in scores:
        h_ = msc.load_hedged_inputs(REGION, UNIVERSE, VARIANT, score=s_, label=short[s_])
        if h_ is not None:
            h_['color'] = mc.CAT[mc.SCORES.index(s_) % len(mc.CAT)]      # the colours of the unhedged chart
            hedged[s_] = h_
    ds = list(hedged.values())
    if ds:
        fig = deck.slide("The hedged scores", "The scores with their factor exposure removed")
        msc.plot_hedged_cumulative_multi(chart(fig, M + 0.2, 3.7, 7.4, 3.5), ds)
        msc.plot_spanning_multi(chart(fig, M + 7.9, 3.7, 7.4, 3.5), ds)

        def gain(h_, period):
            r = h_['spanning'].loc[period]
            return f"{num(r['dSharpe'], '{:+.2f}')} ({r['JK t']:.1f})"

        rows = [[short[s_], num(h_['table'].loc['Full', 'Sharpe']), num(h_['table'].loc['Full', 'alpha t (NW)'], '{:.1f}'),
                 num(h_['table'].loc['Full', 'factor R2']), f"{h_['table'].loc['Full', 'one-way turnover %/mo']:.0f}%",
                 num(h_['table'].loc['Full', 'Sharpe after 10bps']), gain(h_, 'Full'), gain(h_, '2000-2016')]
                for s_, h_ in hedged.items()]
        y = table(fig, M, 3.05, W - 2 * M, ["Hedged", "Sharpe", "Alpha t", "Factor R²", "Turnover", "@ 10 bps",
                                            "Adds to factors (t)", "Adds, 2000–2016 (t)"],
                  rows, [0.14, 0.1, 0.1, 0.11, 0.11, 0.1, 0.17, 0.17], size=14, row_h=0.4)
        text(fig, M, y - 0.15, "Each score's long/short minus its factor exposure, hedged like the SDF: betas at each refit from "
             f"the model's own score rebuilt over the previous 10 years (from {min(h_['monthly'].index.min() for h_ in ds) // 10000}). "
             "'Adds to factors': the rise in "
             "the Sharpe ratio of the factors' in-sample tangency portfolio when the hedged score joins it, with a "
             "Jobson-Korkie t.", W - 2 * M - 1.5, size=11, color=MUTED)
        deck.save(fig)

        # 9c. the hedged scores by period, and after costs
        fig = deck.slide("Hedged scores by period", "Independent of the factors, and after costs")
        msc.plot_hedged_sharpe_by_period_multi(chart(fig, M + 0.2, 2.6, 6.3, 4.2), ds)
        decades = [p_ for p_ in ['Full', '1980s', '1990s', '2000s', '2010s'] if p_ in ds[0]['table'].index]
        rows = [[p_ if p_ != 'Full' else "Full sample"]
                + [num(h_['table'].loc[p_, 'Sharpe']) for h_ in ds]
                + [num(h_['table'].loc[p_, 'Sharpe after 10bps']) for h_ in ds] for p_ in decades]
        tx, tw = 7.5, W - M - 7.5
        widths = [0.16] + [0.84 / (2 * len(ds))] * (2 * len(ds))
        for label_, first in [("Sharpe, gross", 1), ("After 10 bps", 1 + len(ds))]:     # labels over the column groups
            gx = tx + tw * sum(widths[:first])
            fig.text((gx + 0.12) / W, 7.05 / H, label_, fontsize=14, weight="bold", color=ACCENT, va="bottom")
            fig.add_artist(Line2D([(gx + 0.1) / W, (gx + tw * sum(widths[first:first + len(ds)]) - 0.1) / W],
                                  [7.0 / H, 7.0 / H], color=ACCENT, linewidth=1))
        y = table(fig, tx, 6.9, tw, ["Period"] + [short[s_] for s_ in hedged] * 2, rows, widths,
                  size=14, row_h=0.52, bold=(0,))
        text(fig, tx, y - 0.3, "Sharpe ratios of the hedged score long/shorts, before and after 10 bps a trade (hedge "
             "trading included). Hedge ratios at each refit from the model's score rebuilt over the previous 10 years.",
             tw, size=13, color=MUTED)
        deck.save(fig)

        # 9d. does the edge persist? horizon decay of the hedged scores
        with_decay = [h_ for h_ in ds if h_['decay'] is not None]
        if with_decay:
            fig = deck.slide("Does the edge persist?", "How much of the hedged score's return is left months later")
            msc.plot_score_decay_multi(chart(fig, M + 0.2, 2.75, 6.4, 3.75), with_decay)
            horizons = [h for h in [1, 3, 6, 12] if h in with_decay[0]['decay'].index]
            dec0 = with_decay[0]['decay']                      # the score (blue), for the return column and the text
            rows = [[f"Month {h}", f"{num(dec0.loc[h, 'ann. return %'], '{:+.1f}')} ({dec0.loc[h, 't-stat']:.1f})"]
                    + [num(h_['decay'].loc[h, 'rank IC'], '{:+.3f}') for h_ in with_decay] for h in horizons]
            tx, tw = 8.1, W - M - 8.1
            widths = [0.2, 0.26] + [0.54 / len(with_decay)] * len(with_decay)
            gx = tx + tw * sum(widths[:2])                     # label over the rank-IC columns
            fig.text((gx + 0.12) / W, 7.05 / H, "Rank IC with the month-k return", fontsize=13, weight="bold",
                     color=ACCENT, va="bottom")
            fig.add_artist(Line2D([(gx + 0.1) / W, (tx + tw - 0.1) / W], [7.0 / H, 7.0 / H], color=ACCENT, linewidth=1))
            y = table(fig, tx, 6.9, tw, ["", f"Hedged {with_decay[0]['label'].lower()} return, %/yr (t)"] + [h_['label'] for h_ in with_decay],
                      rows, widths, size=14, row_h=0.46)
            text(fig, tx, y - 0.25, "How it is measured: each score's hedged long/short (its positions minus the factor "
                 "hedge) is formed as usual, then the same positions are held and the return earned in month k alone is "
                 "compared with month 1 (left; 1.0 = as much as month 1, below 0 = the positions lose money). The rank IC "
                 "correlates the hedged position weights with each stock's return in month k.", tw, size=12, color=MUTED)
            r = {h: dec0.loc[h] for h in horizons}
            later = [h for h in horizons if h > 1]
            parts = [f"{num(r[h]['ann. return %'], '{:+.1f}')}% a year in month {h} (t {r[h]['t-stat']:.1f})" for h in later]
            held = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
            t_later = [r[h]['t-stat'] for h in later]
            turnover = with_decay[0]['table'].loc['Full', 'one-way turnover %/mo']
            if all(abs(t_) < 2 for t_ in t_later):
                verdict = ("None of the later months is significant: the information is used up within about a month, "
                           f"so the hedged score is rebalanced monthly (turnover {turnover:.0f}% a month, hedge included).")
            elif min(t_later) <= -2:
                verdict = ("The positions then lose money, so the hedged score has to be rebalanced monthly "
                           f"(turnover {turnover:.0f}% a month, hedge included).")
            else:
                verdict = ("The positions keep earning after the first month, so slower rebalancing could keep much of "
                           f"the return at a fraction of the {turnover:.0f}% monthly turnover.")
            raw = with_decay[0].get('decay_raw')
            compare = (f" Unhedged, month 3 is {num(raw.loc[3, 'ann. return %'], '{:+.1f}')}% a year "
                       f"(t {raw.loc[3, 't-stat']:.1f})." if raw is not None and 3 in raw.index else "")
            text(fig, M, 1.95, f"What it shows: the hedged {with_decay[0]['label'].lower()} earns "
                 f"{num(r[1]['ann. return %'], '{:.1f}')}% a year in the month after it is formed (t {r[1]['t-stat']:.1f}); "
                 f"held on, the same positions earn {held}. " + verdict + compare,
                 W - 2 * M, size=14, color=TEXT)
            deck.save(fig)

        # 9d2. trading the hedged score more slowly (turnover_controls.py)
        rep_dir = DataPaths().result_file("report", REGION, UNIVERSE, VARIANT, ext=None)
        if (rep_dir / "turnover_controls_score_by_period.csv").exists():
            tc = pd.read_csv(rep_dir / "turnover_controls_score_by_period.csv").set_index(["control", "period"])
            full_tc, last_p = tc.xs("Full", level="period"), [p_ for p_ in periods if p_ in tc.index.get_level_values("period")][-1]
            last_tc = tc.xs(last_p, level="period")
            fig = deck.slide("Trading the hedged score", "Slower trading keeps most of the hedged score at far lower cost")
            rows = [[c, f"{full_tc.loc[c, 'one-way turnover %/mo']:.0f}%", num(full_tc.loc[c, "Sharpe"]),
                     f"{num(full_tc.loc[c, 'break-even cost bps'], '{:.0f}')} bps", num(full_tc.loc[c, "Sharpe after 10bps"]),
                     num(full_tc.loc[c, "Sharpe after 25bps"]), num(last_tc.loc[c, "Sharpe"]),
                     num(last_tc.loc[c, "Sharpe after 10bps"])] for c in full_tc.index]
            widths_tc = [0.24, 0.1, 0.09, 0.11, 0.1, 0.1, 0.12, 0.14]
            gx = M + (W - 2 * M) * sum(widths_tc[:6])          # label over the latest period's columns
            fig.text((gx + 0.12) / W, 7.05 / H, f"The {last_p}", fontsize=14, weight="bold", color=ACCENT, va="bottom")
            fig.add_artist(Line2D([(gx + 0.1) / W, (W - M - 0.1) / W], [7.0 / H, 7.0 / H], color=ACCENT, linewidth=1))
            y = table(fig, M, 6.9, W - 2 * M, ["Trading rule, whole book", "Turnover", "Sharpe", "Break-even", "@ 10 bps",
                                               "@ 25 bps", "Sharpe", "@ 10 bps"],
                      rows, widths_tc, size=15, row_h=0.48, bold=(1,))
            base, best10 = full_tc.iloc[0], full_tc["Sharpe after 10bps"].idxmax()
            best25 = full_tc["Sharpe after 25bps"].idxmax()
            monthly = full_tc.index[0]
            if best10 == monthly:
                at10 = (f"Monthly rebalancing already gives the best Sharpe ratio after 10 bps "
                        f"({num(base['Sharpe after 10bps'])})")
            else:
                at10 = (f"{best10.capitalize()} cuts turnover from {base['one-way turnover %/mo']:.0f}% to "
                        f"{full_tc.loc[best10, 'one-way turnover %/mo']:.0f}% a month and lifts the Sharpe ratio after "
                        f"10 bps from {num(base['Sharpe after 10bps'])} to {num(full_tc.loc[best10, 'Sharpe after 10bps'])}")
            at25 = ("; it is also best after 25 bps." if best25 == best10 else
                    f"; after 25 bps, {best25} gives {num(full_tc.loc[best25, 'Sharpe after 25bps'])} against "
                    f"{num(base['Sharpe after 25bps'])}.")
            recent = last_tc['Sharpe after 10bps'].max()
            recent_txt = (f" In the {last_p}, though, no rule gets above {num(recent)} after 10 bps: slower trading cuts "
                          "the cost, but the recent gross return is too small to leave much." if recent < 0.5 else
                          f" In the {last_p} the best rule still gives {num(recent)} after 10 bps.")
            text(fig, M, y - 0.3, "Each month the book (score positions and hedge) moves only part of the way to its new "
                 "target, or holds the average of recent targets; turnover is one-way per unit long and unit short. "
                 + at10 + at25 + recent_txt, W - 2 * M - 1.0, size=14, color=TEXT)
            deck.save(fig)

        # 9e. is the alpha a size or market-beta bet?
        with_decomp = [h_ for h_ in ds if h_['decomp'] is not None]
        if with_decomp:
            fig = deck.slide("Size and beta", "Adding size and beta to the hedge leaves the alpha")
            msc.plot_hedge_decomposition_multi(chart(fig, M + 0.2, 2.75, 6.4, 3.75), with_decomp)
            d0 = with_decomp[0]                                # the score (blue), for the table and the text
            specs = [sp for sp in ['9 factors', '9 + size', '9 + size + beta'] if sp in d0['decomp'].index]
            rows = [[sp, num(d0['decomp'].loc[sp, 'Sharpe']), num(d0['decomp'].loc[sp, 'alpha t (NW)'], '{:.1f}'),
                     num(d0['decomp'].loc[sp, 'factor R2']),
                     f"{d0['decomp'].loc[sp, 'size beta']:+.2f} ({d0['decomp'].loc[sp, 'size beta t']:.1f})"]
                    for sp in specs]
            tx, tw = 8.1, W - M - 8.1
            y = table(fig, tx, 6.9, tw, [f"{d0['label']}, hedged on", "Sharpe", "Alpha t", "R²", "Size loading (t)"], rows,
                      [0.3, 0.15, 0.15, 0.13, 0.27], size=14, row_h=0.5)
            text(fig, tx, y - 0.25, ("How it is tested: two factors are added to the hedge, size (long small, short large "
                 "stocks) and market beta (long high-beta, short low-beta)" if '9 + size + beta' in d0['decomp'].index else
                 "How it is tested: a size factor (long small, short large stocks) is added to the hedge; the data has no "
                 "market beta to add") + ", built like the nine, with betas from the same rebuilt score histories. "
                 "Size loading: the hedged score regressed on the size factor.", tw, size=12, color=MUTED)
            has_beta = '9 + size + beta' in d0['decomp'].index         # company data has no market beta
            last = '9 + size + beta' if has_beta else '9 + size'
            a, b, c = (d0['decomp'].loc[sp] for sp in ['9 factors', '9 + size', last])
            others = [h_ for h_ in with_decomp[1:] if last in h_['decomp'].index]
            other_txt = "; ".join(f"{h_['label'].lower()} {num(h_['decomp'].loc['9 factors', 'Sharpe'])} → "
                                  f"{num(h_['decomp'].loc[last, 'Sharpe'])}" for h_ in others)
            text(fig, M, 1.95, f"What it shows: hedging size as well moves the {d0['label'].lower()}'s Sharpe ratio from "
                 f"{num(a['Sharpe'])} to {num(b['Sharpe'])}"
                 + (f", and adding beta to {num(c['Sharpe'])}" if has_beta else "")
                 + f" (alpha t {a['alpha t (NW)']:.1f} → {c['alpha t (NW)']:.1f}). "
                 + (f"Its size loading stays near zero ({c['size beta']:+.2f}, t {c['size beta t']:.1f}): "
                    if abs(c['size beta t']) < 2 else
                    f"Its size loading is small but {'negative' if c['size beta'] < 0 else 'positive'} "
                    f"({c['size beta']:+.2f}, t {c['size beta t']:.1f}), leaning if anything to "
                    f"{'large' if c['size beta'] < 0 else 'small'} stocks: ")
                 + "most of the SDF's gross return comes from smaller "
                 "stocks, but the alpha left after hedging is not a small-stock bet. "
                 + ("" if not has_beta else
                    "Beta takes a little more, so part of the return came with market-beta exposure, but most of it survives"
                    if b['Sharpe'] - c['Sharpe'] > 0.02 else "Adding beta to the hedge changes little")
                 + (f" ({other_txt})." if others else "."), W - 2 * M, size=14, color=TEXT)
            deck.save(fig)

    # 9f. the tree scores added to the existing factor scores (analysis/backtest_analysis.py)
    rep_dir = DataPaths().result_file("report", REGION, UNIVERSE, VARIANT, ext=None)
    if (rep_dir / "combined_scores_summary.csv").exists() and (rep_dir / "combined_scores_pnl.csv").exists():
        comb = pd.read_csv(rep_dir / "combined_scores_summary.csv", index_col=0)
        comb_pnl = pd.read_csv(rep_dir / "combined_scores_pnl.csv", index_col=0)
        trees = [k for k in comb.index if k != "EI alone"]
        palette = dict(zip(["Tree", "Resid", "Hedged", "Original"], ["#2a78d6", "#eda100", "#eb6834", "#1baf7a"]))
        fig = deck.slide("Adding to the factors", "Each tree score combined with the existing factor scores")
        ax = chart(fig, M + 0.2, 2.45, 6.9, 4.1)
        cum = comb_pnl.cumsum() * 100
        cum.index = mc.to_dt(cum.index)
        ax.plot(cum.index, cum["EI"], color="#8a94a3", linewidth=2.6, label=f"EI ({cum['EI'].iloc[-1]:.0f}%)")
        for k in trees:
            col = f"EI + {k}"
            ax.plot(cum.index, cum[col], color=palette.get(k, "#4a3aa7"), linewidth=1.8,
                    label=f"+ {k} ({cum[col].iloc[-1]:.0f}%)")
        ax.set_ylabel("Cumulative return, % (sum of monthly)")
        ax.legend(loc="upper left", fontsize=13)
        rows = [["EI alone", num(comb.loc["EI alone", "IR, EI + tree score"]), "", num(comb.loc["EI alone", "IC"], "{:.3f}"),
                 num(comb.loc["EI alone", "ICIR"]), f"{comb.loc['EI alone', 'turnover']:.0%}"]]
        rows += [[f"+ {k}", num(comb.loc[k, "IR, EI + tree score"]),
                  f"{num(comb.loc[k, 'IR gain over EI'], '{:+.2f}')} ({comb.loc[k, 'gain t-stat']:.1f})",
                  num(comb.loc[k, "IC"], "{:.3f}"), num(comb.loc[k, "ICIR"]), f"{comb.loc[k, 'turnover']:.0%}"]
                 for k in trees]
        tx, tw = 7.9, W - M - 7.9
        y = table(fig, tx, 6.9, tw, ["Score", "IR", "Gain (t)", "IC", "ICIR", "Turnover"], rows,
                  [0.22, 0.13, 0.22, 0.15, 0.13, 0.15], size=14, row_h=0.5, bold=(0,))
        weighting = ("each combination is 80% the production-weighted factor scores and 20% the tree score"
                     if REGION is not None else
                     f"the factor scores and the tree score are averaged with equal weights "
                     f"(tree score 1/{len(FEATURES) + 1})")
        text(fig, tx, y - 0.25, f"EI: the {len(FEATURES)} existing factor scores; {weighting}. IR and gain: the "
             "score-weighted long/short's information ratio and its rise over EI (t of the monthly difference). "
             "Tree, Resid, Hedged, Original: the score, its residual on the factor scores, its factor-hedged positions "
             "and the original score, all rank-normalised.", tw, size=11, color=MUTED)
        best = comb.loc[trees, "IR gain over EI"].idxmax()
        text(fig, M, 1.85, f"What it shows: every tree score adds to the factors, and {best.lower()} adds the most "
             f"(IR {num(comb.loc['EI alone', 'IR, EI + tree score'])} → {num(comb.loc[best, 'IR, EI + tree score'])}, "
             f"IC {num(comb.loc['EI alone', 'IC'], '{:.3f}')} → {num(comb.loc[best, 'IC'], '{:.3f}')}); turnover moves "
             f"from {comb.loc['EI alone', 'turnover']:.0%} to {comb.loc[best, 'turnover']:.0%}."
             if (comb.loc[trees, "IR gain over EI"] > 0).all() else
             f"What it shows: {best.lower()} adds the most to the factors "
             f"(IR {num(comb.loc['EI alone', 'IR, EI + tree score'])} → {num(comb.loc[best, 'IR, EI + tree score'])}); "
             f"{', '.join(k.lower() for k in trees if comb.loc[k, 'IR gain over EI'] <= 0)} do not.",
             W - 2 * M, size=14, color=TEXT)
        deck.save(fig)

        # 9g. the combinations by period
        if (rep_dir / "combined_scores_by_period.csv").exists():
            bp = pd.read_csv(rep_dir / "combined_scores_by_period.csv", index_col=0)
            cols = ["EI"] + [f"EI + {k}" for k in trees]
            fig = deck.slide("Adding to the factors, by period", "EI and each combination, period by period")
            rows = [[p_] + [num(bp.loc[p_, f"IR|{c}"]) for c in cols] + [num(bp.loc[p_, f"IC|{c}"], "{:.3f}") for c in cols]
                    for p_ in bp.index]
            widths_bp = [0.12] + [0.88 / (2 * len(cols))] * (2 * len(cols))
            for label_, first in [("Information ratio", 1), ("IC", 1 + len(cols))]:   # labels over the column groups
                gx = M + (W - 2 * M) * sum(widths_bp[:first])
                fig.text((gx + 0.12) / W, 7.05 / H, label_, fontsize=14, weight="bold", color=ACCENT, va="bottom")
                fig.add_artist(Line2D([(gx + 0.1) / W, (gx + (W - 2 * M) * sum(widths_bp[first:first + len(cols)]) - 0.1) / W],
                                      [7.0 / H, 7.0 / H], color=ACCENT, linewidth=1))
            heads = ["Period"] + ["EI"] + [f"+ {k}" for k in trees]
            y = table(fig, M, 6.9, W - 2 * M, heads + heads[1:], rows, widths_bp, size=13, row_h=0.5, bold=(0,))
            dec = [p_ for p_ in bp.index if p_ != "Full sample"]
            best = comb.loc[trees, "IR gain over EI"].idxmax()
            wins = [p_ for p_ in dec if bp.loc[p_, f"IR|EI + {best}"] > bp.loc[p_, "IR|EI"]]
            last_p = dec[-1]
            text(fig, M, y - 0.3, f"Information ratio and IC of the score-weighted long/short, by period. EI + {best.lower()} beats "
                 f"EI in {len(wins)} of {len(dec)} periods; in the {last_p} its IR is "
                 f"{num(bp.loc[last_p, f'IR|EI + {best}'])} against {num(bp.loc[last_p, 'IR|EI'])} "
                 f"(t of the monthly gain {bp.loc[last_p, f'gain t|{best}']:.1f}).", W - 2 * M - 1.0, size=14, color=TEXT)
            deck.save(fig)

    # 9h. do the scores capture the SDF? (backtest_report.py section 3b; hedged scores from factor_spanning.py)
    corr_file = DataPaths().result_file("report", REGION, UNIVERSE, VARIANT, ext=None) / "score_sdf_correlation.csv"
    if corr_file.exists():
        sc = pd.read_csv(corr_file).set_index(["period", "score"])
        fig = deck.slide("Scores and the SDF", "Do the scores capture the SDF?")
        ax = chart(fig, M + 0.2, 2.45, 6.0, 4.1)
        series = [("Hedged SDF", rets["Return"], NAVY)] + [(f"Hedged {short[s_].lower()}", h_["monthly"], h_["color"])
                                                           for s_, h_ in hedged.items()]
        for name_, ser, col in series:                # the SDF and the long/shorts differ in scale: put all at 10% vol
            cum = (ser * 0.10 / (ser.std() * 12 ** 0.5)).cumsum() * 100
            cum.index = mc.to_dt(cum.index)
            ax.plot(cum.index, cum, color=col, linewidth=2.2, label=f"{name_} ({cum.iloc[-1]:.0f}%)")
        ax.set_ylabel("Cumulative return at 10% vol, %")
        ax.legend(loc="upper left", fontsize=13)
        hsdf = rets["Return"]

        def stats(t):                                 # Sharpe, alpha (t), factor R² from a by-period table row
            return [num(t["Sharpe"]), f"{num(t['alpha ann. %'])} ({t['alpha t (NW)']:.1f})", num(t["factor R2"])]

        rows = [["Hedged SDF"] + stats(sdf.loc[("Return", "Full")]) + ["1.00", "1.00", num(rets["Return_mkt_adj"].corr(hsdf))]]
        rows += [[f"Hedged {short[s_].lower()}"] + stats(hedged[s_]["table"].loc["Full"])
                 + [num(hedged[s_]["monthly"].corr(hsdf.reindex(hedged[s_]["monthly"].index))),
                    num(sc.loc[("Full", s_), "Return_mkt_adj"]), num(sc.loc[("Full", s_), "Return"])]
                 for s_ in hedged if ("Full", s_) in sc.index]
        tx, tw = 7.5, W - M - 7.5
        widths = [0.22, 0.1, 0.17, 0.08, 0.13, 0.17, 0.13]
        for x0, span, label in [(sum(widths[:4]), widths[4], "Hedged corr."), (sum(widths[:5]), sum(widths[5:]), "Unhedged corr.")]:
            fig.text((tx + tw * (x0 + span / 2)) / W, 7.05 / H, label, fontsize=12, weight="bold", color=ACCENT,
                     ha="center", va="bottom")
        y = table(fig, tx, 6.9, tw, ["", "Sharpe", "Alpha %/yr (t)", "R²", "Hedged SDF",
                                     "Mkt-adj. SDF", "Hedged SDF"], rows, widths, size=12, row_h=0.5)
        text(fig, tx, y - 0.25, "Full sample, monthly. Sharpe, alpha (Newey-West t) and factor R² are for the hedged "
             "series. Hedged corr.: the hedged series against the factor-hedged SDF (Return). Unhedged corr.: the score's own "
             "long/short (for the SDF row, the market-adjusted SDF) against each SDF. The chart scales every hedged "
             "series to 10% annual volatility (sum of monthly returns).", tw, size=12, color=MUTED)
        s0 = scores[0]
        text(fig, M, 1.85, f"What it shows: the {short[s0].lower()} long/short moves almost one-for-one with the "
             f"market-adjusted SDF (correlation {num(sc.loc[('Full', s0), 'Return_mkt_adj'])}) but much less with the "
             f"factor-hedged SDF ({num(sc.loc[('Full', s0), 'Return'])}): the score carries the SDF's factor exposure. "
             + (f"Hedged the same way, it tracks the hedged SDF more closely "
                f"({num(hedged[s0]['monthly'].corr(hsdf.reindex(hedged[s0]['monthly'].index)))})."
                if s0 in hedged else ""), W - 2 * M, size=14, color=TEXT)
        deck.save(fig)

    # 10. scores by period
    def tilt(p):
        for comp, sign in [(f"{DEFAULT_SCORE} - sdf_weight", 1), (f"sdf_weight - {DEFAULT_SCORE}", -1)]:
            if ("all stocks", comp, p) in diffs.index:
                r = diffs.loc[("all stocks", comp, p)]
                return f"{num(sign * r['mean diff %/mo'], '{:+.2f}')} ({sign * r['t (NW)']:.1f})"
        return "n/a"

    fig = deck.slide("Scores by period", "Score long/short Sharpe ratios by period")

    def hedged_sharpe(s_, p):
        return num(hedged[s_]['table'].loc[p, 'Sharpe']) if s_ in hedged and p in hedged[s_]['table'].index else "n/a"

    rows = [[p if p != "Full" else f"{y0}–{y1}"] + [num(perf.loc[("all stocks", s, p), "Sharpe"]) for s in scores]
            + [hedged_sharpe(s, p) for s in hedged]
            + [num(perf.loc[("all stocks", DEFAULT_SCORE, p), "alpha t (NW)"], "{:.1f}"), tilt(p)] for p in ["Full"] + periods]
    n_cols = len(scores) + len(hedged)
    y = table(fig, M, 6.9, W - 2 * M, ["Period"] + [short[s] for s in scores] + [f"{short[s]} hedged" for s in hedged]
              + ["Score alpha t", "Tilt gain, %/mo (t)"],
              rows, [0.13] + [0.6 / n_cols] * n_cols + [0.11, 0.16], size=15, row_h=0.56, bold=(0,))
    text(fig, M, y - 0.4, f"Tilt gain is the {SCORE_LABEL} minus sdf_weight, with a Newey-West t: what the within-node tilt adds "
         "to the plain SDF weights. Hedged: each score's long/short minus its factor exposure, with betas at each refit "
         "from the model's score rebuilt over the previous 10 years"
         + (f"; its first row covers {min(h_['monthly'].index.min() for h_ in ds) // 10000}–{y1}."
            if ds and min(h_['monthly'].index.min() for h_ in ds) // 10000 != y0 else "."), W - 2 * M - 1.5, size=15)
    deck.save(fig)

    # 11. statement: the factor-hedged score's alpha is not a size bet
    by_size = msc.load_hedged_by_size(REGION, UNIVERSE, VARIANT)
    dec = hedged[DEFAULT_SCORE]['decomp'] if DEFAULT_SCORE in hedged else None
    fig = deck.slide("Size-neutral alpha" if by_size is not None else "Where the return comes from", dark=True)
    if by_size is not None:
        sh_in = by_size['Sharpe within']
        fig.text(M / W, 6.9 / H, f"{int((sh_in > 0).sum())} of {len(sh_in)}", fontsize=110, family=SERIF, weight="bold",
                 color=ACCENT_LIGHT, va="top")
        text(fig, M, 4.55, "size quintiles in which the factor-hedged score earns on its own: Sharpe ratio "
             f"{num(sh_in.iloc[0])} in the smallest and {num(sh_in.iloc[-1])} in the largest", 12.5, size=26, color=BG,
             family=SERIF)
        if dec is not None:
            text(fig, M, 2.75, f"Its loading on a size factor is {dec.loc['9 factors', 'size beta']:+.2f} "
                 f"(t {dec.loc['9 factors', 'size beta t']:.1f}), and hedging size as well leaves a Sharpe ratio of "
                 f"{num(dec.loc['9 + size', 'Sharpe'])}: the alpha is not a size bet.", 13.5, size=17, color=ON_DARK)
        text(fig, M, 1.75, f"By contrast, {small_share:.0f}% of the market-adjusted SDF's return comes from the "
             f"{small_label} of stocks by market cap.", 13.5, size=14, color="#9fb0c6")
    else:
        fig.text(M / W, 6.9 / H, f"{small_share:.0f}%", fontsize=120, family=SERIF, weight="bold", color=ACCENT_LIGHT,
                 va="top")
        text(fig, M, 4.4, f"of the SDF's market-adjusted return comes from the {small_label} of stocks by market cap",
             11.5, size=28, color=BG, family=SERIF)
    deck.save(fig)

    if by_size is not None:
        labels = list(by_size.index)
        raw_within = [perf.loc[(f"size {b}", DEFAULT_SCORE, "Full"), "Sharpe"] for b in labels]
        hedged_small = by_size['share of return %'].iloc[:n_small].sum()

        # 12. the hedged score by size: where its return comes from, and how it does inside each quintile
        fig = deck.slide("The hedged score by size", "The factor-hedged score earns in every size quintile")
        msc.plot_size_share(chart(fig, M + 0.2, 2.65, 6.6, 3.85), labels,
                            (contrib_full[labels] / contrib_full.sum() * 100).to_numpy(dtype=float),
                            by_size['share of return %'].to_numpy(dtype=float))
        msc.plot_size_sharpe_within(chart(fig, M + 8.0, 2.65, 6.6, 3.85), labels, raw_within,
                                    by_size['Sharpe within'].to_numpy(dtype=float))
        n_earn = int((by_size['Sharpe within'] > 0).sum())
        text(fig, M, 1.85, f"Right: the score's long/short built only from each quintile's stocks. Unhedged its Sharpe ratio "
             f"goes from {num(raw_within[0])} in the smallest to {num(raw_within[-1])} in the largest; hedged on the nine "
             f"factors it earns in {'all five' if n_earn == len(labels) else f'{n_earn} of {len(labels)}'} (Sharpe "
             f"{num(by_size['Sharpe within'].iloc[0])} to {num(by_size['Sharpe within'].iloc[-1])}, alpha t "
             f"{by_size['alpha t within'].min():.1f} to {by_size['alpha t within'].max():.1f}), so the alpha is not a "
             f"size bet. Left: its return still leans to smaller stocks, {hedged_small:.0f}% from the {small_label} "
             f"(the SDF: {small_share:.0f}%), "
             + ("though it earns about the same per unit of risk at every size: the same weight simply earns more in "
                "the more volatile small stocks."
                if by_size['Sharpe within'].max() <= 1.3 * by_size['Sharpe within'].min() else
                "because it earns more per unit of risk there, not because it is long small stocks."),
             W - 2 * M, size=14, color=TEXT)
        deck.save(fig)

        # 13. the hedged score within size groups, after costs
        fig = deck.slide("Hedged score and trading costs", "The factor-hedged score within size groups, after costs")
        hfull = hedged[DEFAULT_SCORE]['table'].loc['Full'] if DEFAULT_SCORE in hedged else None
        rows = []
        if hfull is not None:
            rows.append(["All stocks", f"{cost_all['stocks scored/mo']:,.0f}", num(hfull['Sharpe']),
                         num(hfull['alpha t (NW)'], '{:.1f}'), f"{hfull['one-way turnover %/mo']:.0f}%",
                         f"{num(hfull['break-even cost bps'], '{:.0f}')} bps", num(hfull['Sharpe after 10bps']),
                         num(cost_all['Sharpe after 10bps'])])
        for b in labels:
            r_ = by_size.loc[b]
            raw_net = costs.loc[(f"size {b}", DEFAULT_SCORE), "Sharpe after 10bps"] if (f"size {b}", DEFAULT_SCORE) in costs.index else None
            rows.append([f"{b} only", f"{r_['stocks/mo']:,.0f}", num(r_['Sharpe within']), num(r_['alpha t within'], '{:.1f}'),
                         f"{r_['turnover within %/mo']:.0f}%", f"{num(r_['break-even within bps'], '{:.0f}')} bps",
                         num(r_['Sharpe after 10bps within']), num(raw_net) if raw_net is not None else "n/a"])
        y = table(fig, M, 6.9, W - 2 * M, ["Stocks used", "Stocks/mo", "Sharpe", "Alpha t", "Turnover", "Break-even",
                                           "@ 10 bps", "Unhedged @ 10 bps"],
                  rows, [0.16, 0.11, 0.1, 0.1, 0.11, 0.13, 0.11, 0.18], size=15, row_h=0.5, bold=(0,))
        text(fig, M, y - 0.3, "Each row builds the score's long/short from those stocks only and hedges it on the nine factors "
             "(betas at each refit from the model's score rebuilt over the previous 10 years); turnover and costs include "
             "trading the hedge. Break-even "
             "is the one-way cost per trade that wipes out the return. The last column is the same long/short without the "
             "hedge, for comparison.", W - 2 * M - 1.5, size=14, color=MUTED)
        deck.save(fig)
    else:
        # size breakdown of the SDF when the hedged-score tables are missing
        fig = deck.slide("Size breakdown", "Contribution and weight by market-cap quintile")
        ax1, ax2 = chart(fig, M + 0.7, 2.0, 6.2, 4.6), chart(fig, M + 8.0, 2.0, 6.2, 4.6)
        mc.plot_size_contribution([ax1, ax2], d)
        deck.save(fig)

    # the factor-hedged SDF after costs, and trading it more slowly (hedge_analysis.py, turnover_controls.py)
    if d.get("hedge") is not None:
        hg = d["hedge"]
        alone, hedged_p = "SDF alone (Return_mkt_adj)", "Hedged (Return)"
        fig = deck.slide("Tradable portfolio", "The factor-hedged SDF after costs, and trading it more slowly")
        rows1 = [[p if p != "Full" else f"{y0}–{y1}", num(hg.loc[(alone, p), "Sharpe"]), num(hg.loc[(alone, p), "Sharpe after 10bps"]),
                  num(hg.loc[(hedged_p, p), "Sharpe"]), num(hg.loc[(hedged_p, p), "Sharpe after 10bps"]),
                  num(hg.loc[(hedged_p, p), "Sharpe after 25bps"]), f"{num(hg.loc[(hedged_p, p), 'break-even cost bps'], '{:.0f}')} bps"]
                 for p in ["Full"] + periods]
        head1 = ["Period", "SDF alone", "@ 10 bps", "Hedged", "@ 10 bps", "@ 25 bps", "Break-even"]
        w1 = [0.16, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14]
        ct = d["controls"].loc["Hedged"] if d.get("controls") is not None else None
        if ct is not None:
            rows2 = [[c, f"{ct.loc[c, 'one-way turnover % of book']:.0f}%", num(ct.loc[c, "Sharpe"]),
                      f"{num(ct.loc[c, 'break-even cost bps'], '{:.0f}')} bps", num(ct.loc[c, "Sharpe after 10bps"]),
                      num(ct.loc[c, "Sharpe after 25bps"])] for c in ct.index]
            head2 = ["Trading rule, hedged SDF", "Turnover", "Sharpe", "Break-even", "@ 10 bps", "@ 25 bps"]
            w2 = [0.3, 0.14, 0.14, 0.15, 0.13, 0.14]
        for size_, rh in [(15, 0.44), (14, 0.4), (13, 0.36), (12, 0.33)]:     # largest that fits both tables
            need = table_height(W - 2 * M, head1, rows1, w1, size_, rh) + 0.45
            if ct is not None:
                need += table_height(W - 2 * M, head2, rows2, w2, size_, rh)
            if 6.9 - need >= 1.55:
                break
        y = table(fig, M, 6.9, W - 2 * M, head1, rows1, w1, size=size_, row_h=rh, bold=(0,))
        if ct is not None:
            y = table(fig, M, y - 0.45, W - 2 * M, head2, rows2, w2, size=size_, row_h=rh)
            best10 = ct["Sharpe after 10bps"].idxmax()
            note = ((" Slower trading: monthly rebalancing already gives the best Sharpe after 10 bps "
                     if best10 == ct.index[0] else f" Slower trading: {best10} gives the best Sharpe after 10 bps ")
                    + f"({num(ct.loc[best10, 'Sharpe after 10bps'])}); partial rebalancing moves part of the way to the new "
                    "target each month, averaging holds the mean of recent targets.")
        else:
            note = ""
        f_h = hg.loc[(hedged_p, "Full")]
        text(fig, M, y - 0.25, f"Hedged: the SDF's stock positions minus its factor exposure, held through the factor portfolios' "
             f"stocks with hedge ratios from the last refit; it turns over {f_h['one-way turnover % of book']:.0f}% of its "
             f"book a month (one-way), costs per unit traded." + note, W - 2 * M - 1.5, size=13, color=MUTED)
        deck.save(fig)

    # 16. caveats
    fig = deck.slide("Caveats", "Read the numbers with these in mind")
    items = [
        ("Size concentration", f"{small_share:.0f}% of the market-adjusted SDF return comes from the {small_label} of "
                               "stocks; the hedged score earns in every size quintile, but more in smaller stocks."),
        ("Gross of costs", f"Headline numbers ignore about {cost_all['one-way turnover %/mo']:.0f}% one-way turnover a month."),
        ("Alpha benchmark", "The characteristic factors weight the model's stocks by score, not by market cap, so alphas "
                            "against them can differ from alphas against cap-weighted or standard factors."),
        ("Factor normaliser", "Factor returns, score weights and hedges depend on rank_normalise, and its tie rule matters: "
                              "stocks in no held node all score 0. Another normaliser moves the hedged score's Sharpe by ~0.1."),
        ("Selection effect", "Tree portfolios are filtered by their missing-data rate over the full sample."),
        ("One sample", f"{LABEL}, {y0}–{y1}, one configuration; nothing tested beyond it."),
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
    out = mc.REPO / "presentation" / DataPaths().result_file("backtest", REGION, UNIVERSE, VARIANT, "pdf").name
    preview = None
    if "--png" in sys.argv:
        preview = mc.REPO / "presentation" / "deck_preview"
        preview.mkdir(exist_ok=True)
    build(mc.load_inputs(), out, preview)
    print(f"Saved {out}" + (f" and page previews in {preview}" if preview else ""))
