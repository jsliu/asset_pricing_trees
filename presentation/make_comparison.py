"""
One PDF presentation comparing backtest runs side by side: characteristic-file universes (full, large cap, ...),
company-data regions (GL, US, ...) or both. SDF returns, factor exposure, the hedged portfolio after costs,
turnover controls, stock scores and size concentration. Runs whose results are missing are left out; values a run
does not have (a period outside its sample, an analysis not run) show 'n/a'.

For each run, run analysis/backtest_report.py, hedge_analysis.py and turnover_controls.py first (make_presentation.py does
all of it), then from the project root:
    python presentation/make_comparison.py [--runs RUN ...] [--output NAME.pdf] [--png]
A run is written [REGION@][UNIVERSE][:vw], e.g. full, largecap:vw (value-weighted), GL@ or GL@:vw; without --runs the
runs are RUNS in make_presentation.py. Output: presentation/backtest_comparison.pdf unless --output is given.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_charts as mc
from make_deck import (Deck, M, W, NAVY, ACCENT, ACCENT_LIGHT, TEXT, BODY, ON_DARK, BG, SERIF,
                       text, table, table_height, box, chart, num)
from make_charts import describe
from src.constants import DataPaths

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # categorical 1-8
SCORE = "size_oriented_score"
SCORE_LABEL = "Score"          # name of SCORE on the slides
NA = "n/a"
H_ = 9


def parse_run(token):
    """'[REGION@][UNIVERSE][:vw]' -> (region, universe, variant)."""
    body, _, variant = token.partition(":")
    region, at, universe = body.rpartition("@") if "@" in body else ("", "", body)
    return region or None, universe or None, variant or None


def run_token(region, universe, variant):
    return (f"{region}@" if region else "") + (universe or "") + (f":{variant}" if variant else "")




def load(region, universe, variant):
    """Report tables of one run, or None if the run is missing."""
    paths = DataPaths()
    rep = paths.result_file("report", region, universe, variant, ext=None)
    if not (rep / "sdf_returns_by_period.csv").exists():
        return None

    def opt(name, index):
        f = rep / name
        return pd.read_csv(f).set_index(index) if f.exists() else None

    return {
        "rets": pd.read_csv(paths.result_file("ret", region, universe, variant), index_col=0),
        "sdf": pd.read_csv(rep / "sdf_returns_by_period.csv").set_index(["series", "period"]),
        "loadings": pd.read_csv(rep / "sdf_factor_loadings.csv").set_index(["series", "stat"]),
        "perf": pd.read_csv(rep / "score_long_short_by_period.csv").set_index(["universe", "score", "period"]),
        "costs": pd.read_csv(rep / "score_turnover_costs.csv").set_index(["universe", "score"]),
        "contrib": pd.read_csv(rep / "sdf_return_by_size_by_period.csv").set_index(["size bucket", "period"]),
        "exposure": pd.read_csv(rep / "sdf_weight_by_size.csv", index_col=0),
        "hedge": opt("hedge_costs_by_period.csv", ["portfolio", "period"]),
        "hscore": opt("hedged_score_nodes_by_period.csv", ["portfolio", "period"]),
        "neutral": opt("neutral_score_by_period.csv", ["score", "period"]),
        "nsum": opt("neutral_score_summary.csv", ["Unnamed: 0"]),
        "ratios": opt("hedge_ratios_summary.csv", ["Unnamed: 0"]),
        "controls": opt("turnover_controls_summary.csv", ["portfolio", "control"]),
    }


def value(frame, key, col):
    """Raw value, or None when the table, row or column is missing."""
    try:
        v = frame.loc[key, col]
        return None if pd.isna(v) else float(v)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def get(frame, key, col, fmt="{:.2f}", unit=""):
    """Formatted value, or n/a when it is missing."""
    v = value(frame, key, col)
    return NA if v is None else num(v, fmt) + unit


def listing(runs, fn):
    """'x (run 1), y (run 2), ...' for the findings."""
    return ", ".join(f"{fn(r)} ({r['label']})" for r in runs)


def fit_size(header, rows, widths, space, options=((15, 0.5), (14, 0.44), (13, 0.4), (12, 0.34), (11, 0.3)), bold=()):
    """Largest (font size, row height) in options at which the table is at most `space` inches tall."""
    for size, row_h in options:
        if table_height(W - 2 * M, header, rows, widths, size, row_h, bold) <= space:
            return size, row_h
    return options[-1]


def stacked_tables(fig, first, second, header, widths, top=6.95, bottom=0.85, bold=(0,)):
    """Two titled tables, one under the other, in the largest font and row height that fit above the footer.
    first/second: (title, rows)."""
    width = W - 2 * M
    for size, row_h in [(15, 0.46), (14, 0.42), (13, 0.36), (12, 0.32), (11, 0.28)]:
        need = 0.35 + table_height(width, header, first[1], widths, size, row_h, bold) + 0.65 \
            + table_height(width, header, second[1], widths, size, row_h, bold)
        if top - need >= bottom:
            break
    fig.text(M / W, top / H_, first[0], fontsize=15, weight="bold", color=ACCENT, va="top")
    y = table(fig, M, top - 0.35, width, header, first[1], widths, size=size, row_h=row_h, bold=bold)
    fig.text(M / W, (y - 0.3) / H_, second[0], fontsize=15, weight="bold", color=ACCENT, va="top")
    table(fig, M, y - 0.65, width, header, second[1], widths, size=size, row_h=row_h, bold=bold)


def plot_cumulative(ax, series, labels, colors, ylabel):
    for s, label, c in zip(series, labels, colors):
        cum = s.cumsum() * 100
        cum.index = mc.to_dt(cum.index)
        ax.plot(cum.index, cum, color=c, linewidth=2.2, label=label)
        ax.annotate(f"{cum.iloc[-1]:.0f}%", (cum.index[-1], cum.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=mc.INK)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    ax.margins(x=0.1)


def build(runs, path, preview_dir=None):
    plt.rcParams["font.size"] = 13
    deck = Deck(path, preview_dir)
    labels = [r["label"] for r in runs]
    colors = [r["color"] for r in runs]
    n = len(runs)
    y0 = min(r["d"]["rets"].index.min() for r in runs) // 10000
    y1 = max(r["d"]["rets"].index.max() for r in runs) // 10000
    periods = sorted({p for r in runs for p in r["d"]["sdf"].loc["Return_mkt_adj"].index if p != "Full"})
    last = periods[-1]
    ps = ["Full"] + periods
    per = lambda p: "Full sample" if p == "Full" else p
    col_w = [0.22] + [0.78 / n] * n
    wide_w = [0.3] + [0.7 / n] * n
    mk_key, hk = ("Return_mkt_adj", "Full"), ("Hedged (Return)", "Full")
    hscore = lambda r, p, col, fmt="{:.2f}", unit="": get(r["d"]["hscore"],
                                                          (f"{SCORE} long/short, hedged on rebuilt history", p), col, fmt, unit)
    neutral = lambda r, p, col, fmt="{:.2f}": get(r["d"]["neutral"], ("neutral", p), col, fmt)
    nsum = lambda r, col, fmt="{:.2f}", unit="": get(r["d"]["nsum"], col, "neutral", fmt, unit)
    sdf = lambda r, series, p, col, fmt="{:.2f}": get(r["d"]["sdf"], (series, p), col, fmt)
    hedge = lambda r, key, col, fmt="{:.2f}": get(r["d"]["hedge"], key, col, fmt)
    score = lambda r, p, col, fmt="{:.2f}": get(r["d"]["perf"], ("all stocks", SCORE, p), col, fmt)
    cost = lambda r, col, fmt="{:.2f}", unit="": get(r["d"]["costs"], ("all stocks", SCORE), col, fmt, unit)

    def alpha(r, p):
        a, t = value(r["d"]["sdf"], ("Return_mkt_adj", p), "alpha ann. %"), value(r["d"]["sdf"], ("Return_mkt_adj", p), "alpha t (NW)")
        return NA if a is None else f"{num(a, '{:.1f}')} ({t:.1f})"

    def small_share(r):
        try:
            c = r["d"]["contrib"].xs("Full", level="period")["ann. return %"]
            return f"{c.iloc[:2].sum() / c.sum() * 100:.0f}%"
        except (KeyError, ZeroDivisionError):
            return NA

    # 1. cover
    fig = deck.slide(dark=True, footer=False)
    fig.text(M / W, 8.1 / H_, "BACKTEST REVIEW · COMPARISON", fontsize=13, weight="bold", color=ACCENT_LIGHT, va="top")
    fig.text(M / W, 5.6 / H_, "Asset Pricing Trees", fontsize=64, family=SERIF, weight="bold", color=BG, va="top")
    fig.text(M / W, 4.2 / H_, f"{n} backtest runs compared, {y0}–{y1}", fontsize=26, color=ON_DARK, va="top")
    fig.text(M / W, 1.0 / H_, " · ".join(labels), fontsize=15, color="#9fb0c6")
    deck.save(fig)

    # 2. runs
    fig = deck.slide("Runs", f"The {n} runs compared")
    rows = []
    for r in runs:
        ex, rets = r["d"]["exposure"], r["d"]["rets"]
        rows.append([r["label"], r["desc"], f"{rets.index.min() // 10000}–{rets.index.max() // 10000}",
                     cost(r, "stocks scored/mo", "{:,.0f}"),
                     f"{ex['median mkt_cap'].iloc[0]:,.0f}", f"{ex['median mkt_cap'].iloc[-1]:,.0f}"])
    header = ["Run", "Data and weighting", "Sample", "Stocks", "Smallest Q1", "Largest Q5"]
    widths = [0.15, 0.4, 0.12, 0.1, 0.115, 0.115]
    for size, row_h in [(15, 0.55), (14, 0.48), (13, 0.42), (12, 0.36), (11, 0.32)]:   # largest that leaves room for the note
        if 6.9 - table_height(W - 2 * M, header, rows, widths, size, row_h) >= 1.9:
            break
    y = table(fig, M, 6.9, W - 2 * M, header, rows, widths, align=["left", "left", "left", "right", "right", "right"],
              size=size, row_h=row_h)
    text(fig, M, y - 0.4, "Sample = out-of-sample years. Stocks = median number scored a month. Q1/Q5 = median market cap of the "
         "smallest and largest size quintile within each run, in the units of its data. Every run: yearly refits, no look-ahead.",
         W - 2 * M - 1.5, size=15)
    deck.save(fig)

    # 3. summary
    fig = deck.slide("Summary", "Headline results by run")
    metrics = [
        ("Market-adjusted SDF: Sharpe", lambda r: sdf(r, *mk_key, "Sharpe")),
        ("Alpha, % a year (t)", lambda r: alpha(r, "Full")),
        ("Factor R²", lambda r: sdf(r, *mk_key, "factor R2")),
        ("Hedged SDF: Sharpe", lambda r: sdf(r, "Return", "Full", "Sharpe")),
        ("Hedged: Sharpe after 10 bps", lambda r: hedge(r, hk, "Sharpe after 10bps")),
        ("Hedged: break-even, bps", lambda r: hedge(r, hk, "break-even cost bps", "{:.0f}")),
        ("Score long/short: Sharpe", lambda r: score(r, "Full", "Sharpe")),
        ("Score: Sharpe after 10 bps", lambda r: cost(r, "Sharpe after 10bps")),
        ("Neutral score: Sharpe", lambda r: neutral(r, "Full", "Sharpe")),
        ("Neutral score: Sharpe after 10 bps", lambda r: neutral(r, "Full", "Sharpe after 10bps")),
        ("Hedged score: Sharpe after 10 bps", lambda r: hscore(r, "Full", "Sharpe after 10bps")),
        ("Return from smallest 40%", small_share),
    ]
    rows = [[name] + [f(r) for r in runs] for name, f in metrics]
    table(fig, M, 6.9, W - 2 * M, ["Full sample"] + labels, rows, wide_w, size=15, row_h=0.43, bold=(3, 4, 8, 9, 10))
    deck.save(fig)

    # 4. key findings
    def hedged_net(r):
        v = value(r["d"]["hedge"], hk, "Sharpe after 10bps")
        return -np.inf if v is None else v

    best = max(runs, key=hedged_net)
    fig = deck.slide("Key findings", "Key results by run")
    items = [
        ("Market-adjusted SDF", "Sharpe ratio: " + listing(runs, lambda r: sdf(r, *mk_key, "Sharpe")) + "."),
        ("Factor exposure", "Alpha t against the characteristic factors: "
         + listing(runs, lambda r: sdf(r, *mk_key, "alpha t (NW)", "{:.1f}")) + "."),
        ("Factor-hedged SDF", "Sharpe ratio after removing the factor exposure: "
         + listing(runs, lambda r: sdf(r, "Return", "Full", "Sharpe")) + "."),
        ("Best after costs", f"{best['label']}: hedged Sharpe {hedge(best, hk, 'Sharpe after 10bps')} after 10 bps "
         f"(break-even {hedge(best, hk, 'break-even cost bps', '{:.0f}')} bps)."),
        ("Stock scores", "Factor-neutral score long/short, Sharpe after 10 bps: "
         + listing(runs, lambda r: neutral(r, "Full", "Sharpe after 10bps")) + "."),
        ("Latest period", f"Market-adjusted alpha t in the {last}: "
         + listing(runs, lambda r: sdf(r, "Return_mkt_adj", last, "alpha t (NW)", "{:.1f}")) + "."),
    ]
    cw = (W - 2 * M - 2 * 0.3) / 3
    for i, (head, body) in enumerate(items):
        x, top = M + (i % 3) * (cw + 0.3), 6.9 - (i // 3) * 2.75
        box(fig, x, top, cw, 2.45)
        fig.text((x + 0.3) / W, (top - 0.3) / H_, head, fontsize=17, weight="bold", color=NAVY, va="top")
        text(fig, x + 0.3, top - 0.85, body, cw - 0.6, size=14 if n <= 4 else 13 if n <= 6 else 12)
    deck.save(fig)

    # 5. cumulative SDF returns
    fig = deck.slide("SDF returns", "Cumulative SDF returns: market-adjusted and factor-hedged")
    for x, col, title in [(M + 0.7, "Return_mkt_adj", "Market-adjusted"), (M + 8.0, "Return", "Factor-hedged")]:
        ax = chart(fig, x, 1.1, 6.1, 5.3)
        plot_cumulative(ax, [r["d"]["rets"][col] for r in runs], labels, colors, "Cumulative return, %")
        ax.set_title(title, loc="left", color=mc.INK, fontsize=15, pad=10)
    deck.save(fig)

    # 6. SDF by period
    fig = deck.slide("SDF by period", "Market-adjusted SDF: Sharpe ratio and alpha by period")
    stacked_tables(fig,
                   ("Sharpe ratio", [[per(p)] + [sdf(r, "Return_mkt_adj", p, "Sharpe") for r in runs] for p in ps]),
                   ("Alpha t-stat against the characteristic factors",
                    [[per(p)] + [sdf(r, "Return_mkt_adj", p, "alpha t (NW)", "{:.1f}") for r in runs] for p in ps]),
                   ["Period"] + labels, col_w)
    deck.save(fig)

    # 7. factor exposure
    factors = list(dict.fromkeys(c for r in runs for c in r["d"]["loadings"].columns if c != "const"))
    fig = deck.slide("Factor exposure", "Factor loadings of the market-adjusted SDF (t-stat)")

    def loading(r, f):
        b, t = value(r["d"]["loadings"], ("Return_mkt_adj", "beta"), f), value(r["d"]["loadings"], ("Return_mkt_adj", "t"), f)
        return NA if b is None else f"{num(b)} ({t:.1f})"

    rows = [[f] + [loading(r, f) for r in runs] for f in factors]
    rows.append(["Factor R²"] + [sdf(r, *mk_key, "factor R2") for r in runs])
    size, row_h = fit_size(["Factor"] + labels, rows, col_w, 6.9 - 0.85, bold=(len(factors),))
    table(fig, M, 6.9, W - 2 * M, ["Factor"] + labels, rows, col_w, size=size, row_h=row_h, bold=(len(factors),))
    deck.save(fig)

    # 8. hedged portfolio after costs
    fig = deck.slide("Tradable portfolio", "The factor-hedged SDF after trading costs")
    stats = [("Sharpe (gross)", "Sharpe", "{:.2f}"), ("Return, % a year", "ann. return %", "{:.2f}"),
             ("One-way turnover, % of book", "one-way turnover % of book", "{:.0f}"), ("Break-even cost, bps", "break-even cost bps", "{:.0f}"),
             ("Sharpe after 5 bps", "Sharpe after 5bps", "{:.2f}"), ("Sharpe after 10 bps", "Sharpe after 10bps", "{:.2f}"),
             ("Sharpe after 25 bps", "Sharpe after 25bps", "{:.2f}")]
    rows = [[name] + [hedge(r, hk, col, fmt) for r in runs] for name, col, fmt in stats]
    rows.append(["SDF alone, after 10 bps"] + [hedge(r, ("SDF alone (Return_mkt_adj)", "Full"), "Sharpe after 10bps") for r in runs])
    y = table(fig, M, 6.9, W - 2 * M, ["Full sample"] + labels, rows, wide_w, size=15, row_h=0.5, bold=(5,))
    text(fig, M, y - 0.35, "Hedged: the SDF's stock positions minus its factor exposure, held through the factor portfolios' stocks "
         "with hedge ratios from the last refit (fully out of sample). Costs are one-way, per unit traded.", W - 2 * M - 1.5, size=15)
    deck.save(fig)

    # 9. hedged by period after 10 bps
    fig = deck.slide("Tradable portfolio by period", "Hedged SDF: Sharpe ratio after 10 bps by period")
    stacked_tables(fig,
                   ("Sharpe ratio after 10 bps",
                    [[per(p)] + [hedge(r, ("Hedged (Return)", p), "Sharpe after 10bps") for r in runs] for p in ps]),
                   ("Break-even cost, bps",
                    [[per(p)] + [hedge(r, ("Hedged (Return)", p), "break-even cost bps", "{:.0f}") for r in runs] for p in ps]),
                   ["Period"] + labels, col_w)
    deck.save(fig)

    # 10. hedge ratios
    fig = deck.slide("Hedge", "Factor exposure the hedge removes (mean hedge ratio)")
    rows = [[f] + [get(r["d"]["ratios"], f, "mean h", "{:.3f}") for r in runs] for f in factors]
    size, row_h = fit_size(["Factor"] + labels, rows, col_w, 6.9 - 1.9)          # room for the note below
    y = table(fig, M, 6.9, W - 2 * M, ["Factor"] + labels, rows, col_w, size=size, row_h=row_h)
    text(fig, M, y - 0.35, "Exposure of the SDF to each factor portfolio, averaged over refits; the hedge sells it. A negative value "
         "means the SDF is short that factor.", W - 2 * M - 1.5, size=15)
    deck.save(fig)

    # 11. turnover controls
    fig = deck.slide("Turnover controls", "Slower trading of the hedged portfolio")
    rules = ["rebalance 100% a month", "rebalance 50% a month", "rebalance 33% a month", "rebalance 20% a month",
             "average of last 3 months", "average of last 6 months"]
    rows = [[rule] + [f"{get(r['d']['controls'], ('Hedged', rule), 'Sharpe after 10bps')} / "
                      f"{get(r['d']['controls'], ('Hedged', rule), 'Sharpe after 25bps')}" for r in runs] for rule in rules]
    rows.append(["Turnover at 50% rebalancing"] + [get(r["d"]["controls"], ("Hedged", "rebalance 50% a month"),
                                                       "one-way turnover % of book", "{:.0f}", "%") for r in runs])
    y = table(fig, M, 6.9, W - 2 * M, ["Hedged: Sharpe @10 / @25 bps"] + labels, rows, wide_w, size=15 if n <= 5 else 13, row_h=0.52)
    text(fig, M, y - 0.35, "Partial rebalancing moves a fraction of the way to the new target each month; averaging holds the mean of "
         "recent targets. Slower trading gives up gross return when the signal decays quickly, but it pays at higher costs.",
         W - 2 * M - 1.5, size=15)
    deck.save(fig)

    # 12. stock scores: the factor-hedged score by run (score_hedge.py), with the raw score for comparison
    fig = deck.slide("Stock scores", f"{SCORE_LABEL} long/short hedged against the factors, by run")
    stats = [("Sharpe (gross)", lambda r: hscore(r, "Full", "Sharpe")),
             ("  unhedged", lambda r: score(r, "Full", "Sharpe")),
             ("Alpha t", lambda r: hscore(r, "Full", "alpha t (NW)", "{:.1f}")),
             ("Factor R² left", lambda r: hscore(r, "Full", "factor R2")),
             ("One-way turnover", lambda r: hscore(r, "Full", "one-way turnover %/mo", "{:.0f}", "%")),
             ("Break-even, bps", lambda r: hscore(r, "Full", "break-even cost bps", "{:.0f}")),
             ("Sharpe after 10 bps", lambda r: hscore(r, "Full", "Sharpe after 10bps")),
             ("  unhedged", lambda r: cost(r, "Sharpe after 10bps")),
             (f"Sharpe, {last}", lambda r: hscore(r, last, "Sharpe")),
             (f"Sharpe after 10 bps, {last}", lambda r: hscore(r, last, "Sharpe after 10bps"))]
    rows = [[name] + [f(r) for r in runs] for name, f in stats]
    y = table(fig, M, 6.9, W - 2 * M, ["Full sample"] + labels, rows, wide_w, size=14, row_h=0.4, bold=(0, 6))
    text(fig, M, y - 0.25, "The score's long/short (100% long, 100% short) minus its factor exposure, hedged like the SDF: "
         "betas at each refit from the model's own score rebuilt over the previous 10 years. Turnover and costs include "
         "trading the hedge. 'Unhedged': the same long/short before the hedge.", W - 2 * M - 1.5, size=13)
    deck.save(fig)

    # 12b. factor-neutral score
    fig = deck.slide("Stock scores", f"{SCORE_LABEL} long/short, neutral to the factor characteristics")
    stats = [("Sharpe (gross)", lambda r: neutral(r, "Full", "Sharpe")),
             ("  before neutralising", lambda r: get(r["d"]["neutral"], ("raw", "Full"), "Sharpe")),
             ("Alpha t", lambda r: neutral(r, "Full", "alpha t (NW)", "{:.1f}")),
             ("Factor R² left", lambda r: neutral(r, "Full", "factor R2")),
             ("ICIR", lambda r: neutral(r, "Full", "ICIR")),
             ("One-way turnover", lambda r: nsum(r, "one-way turnover %/mo", "{:.0f}", "%")),
             ("Break-even, bps", lambda r: nsum(r, "break-even bps", "{:.0f}")),
             ("Sharpe after 10 bps", lambda r: neutral(r, "Full", "Sharpe after 10bps")),
             (f"Sharpe after 10 bps, {last}", lambda r: neutral(r, last, "Sharpe after 10bps"))]
    rows = [[name] + [f(r) for r in runs] for name, f in stats]
    y = table(fig, M, 6.9, W - 2 * M, ["Full sample"] + labels, rows, wide_w, size=15, row_h=0.46, bold=(0, 7))
    text(fig, M, y - 0.3, "The other way to remove the factors: each month the score is regressed across stocks on the "
         "nine factor characteristics and the residual is the new score, with no exposure to them and no hedge to trade. "
         "The previous page hedges the score's returns instead.", W - 2 * M - 1.5, size=14)
    deck.save(fig)

    # 13. size concentration
    fig = deck.slide("Size concentration", "Contribution to the market-adjusted SDF by size quintile")
    ax = chart(fig, M + 0.7, 2.0, W - 2 * M - 0.8, 4.8)
    buckets = list(dict.fromkeys(b for r in runs for b in r["d"]["contrib"].index.get_level_values("size bucket")))
    x = np.arange(len(buckets))
    w = 0.8 / n
    for i, r in enumerate(runs):
        c = r["d"]["contrib"].xs("Full", level="period")["ann. return %"].reindex(buckets)
        ax.bar(x + (i - (n - 1) / 2) * w, c.fillna(0), width=w, color=r["color"], edgecolor=mc.SURFACE,
               linewidth=2 if n <= 4 else 1, label=r["label"])
    ax.axhline(0, color=mc.INK2, linewidth=1.2)
    ax.set_xticks(x, buckets)
    ax.set_ylabel("Contribution, % a year")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=min(n, 4))
    text(fig, M, 1.5, "Quintiles are formed within each run's own stocks. Share of the return from the two smallest quintiles: "
         + listing(runs, small_share) + ".", W - 2 * M, size=15, color=TEXT)
    deck.save(fig)

    # 14. caveats and next steps
    fig = deck.slide("Caveats and next steps", "Read the numbers with these in mind")
    items = [
        ("Factor normaliser", "Factor returns and the hedge depend on rank_normalise; results move with its version."),
        ("Cost model", "Flat one-way costs per unit traded; no market impact, short-sale costs or financing."),
        ("Hedge implementation", "The hedge trades the factor portfolios' stocks, which adds turnover and short positions."),
        ("Latest period", f"Check the {last} before relying on the results; nothing is tested beyond {y1}."),
        ("Next: investable trees", "Value weighting, excluding fast-moving characteristics such as short-term reversal from splits."),
        ("Next: live-style test", "Combine the hedged portfolio with the production factors and run it through the optimiser."),
    ]
    cw = (W - 2 * M - 2 * 0.3) / 3
    for i, (head, body) in enumerate(items):
        xx, top = M + (i % 3) * (cw + 0.3), 6.9 - (i // 3) * 2.75
        box(fig, xx, top, cw, 2.45)
        fig.text((xx + 0.3) / W, (top - 0.3) / H_, head, fontsize=17, weight="bold", color=NAVY, va="top")
        text(fig, xx + 0.3, top - 0.85, body, cw - 0.6, size=15, color=BODY)
    deck.save(fig)

    deck.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", metavar="RUN", help="runs to compare, e.g. full largecap:vw GL@ US@")
    parser.add_argument("--output", default="backtest_comparison.pdf", help="PDF file name in presentation/")
    parser.add_argument("--png", action="store_true", help="also save page previews in presentation/deck_preview/")
    args = parser.parse_args()
    if args.runs:
        specs = [parse_run(t) for t in args.runs]
    else:
        from make_presentation import RUNS
        specs = RUNS
    if len(specs) > len(COLORS):
        raise SystemExit(f"At most {len(COLORS)} runs fit in one comparison; got {len(specs)}")

    runs = []
    for (region, universe, variant), color in zip(specs, COLORS):
        label, desc = describe(region, universe, variant)
        d = load(region, universe, variant)
        if d is None:
            print(f"Skipping {label}: no report for {run_token(region, universe, variant)} (run analysis/backtest_report.py first)")
            continue
        runs.append({"label": label, "desc": desc, "color": color, "d": d})
    if not runs:
        raise SystemExit("No runs with results to compare")

    out = mc.REPO / "presentation" / args.output
    preview = None
    if args.png:
        preview = mc.REPO / "presentation" / "deck_preview"
        preview.mkdir(exist_ok=True)
    build(runs, out, preview)
    print(f"Saved {out} ({', '.join(r['label'] for r in runs)})" + (f"; previews in {preview}" if preview else ""))


if __name__ == "__main__":
    main()
