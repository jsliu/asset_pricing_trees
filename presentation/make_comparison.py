"""
One PDF presentation comparing the backtest across universes (full, large cap, large cap 0.01%, value-weighted
large cap): SDF returns, factor exposure, the hedged portfolio after costs, turnover controls, stock scores and
size concentration. Universes whose results are missing are left out; missing optional analyses show 'n/a'.

For each universe run backtest_report.py, hedge_analysis.py and turnover_controls.py first, then from the
project root:
    python presentation/make_comparison.py [--png]      -> presentation/backtest_comparison.pdf
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_charts as mc
from make_deck import (Deck, M, W, NAVY, ACCENT, ACCENT_LIGHT, TEXT, BODY, ON_DARK, BG, SERIF,
                       text, table, box, chart, num)
from src.constants import DataPaths

UNIVERSES = [   # label, universe, run suffix, description
    ("Full", "full", "std", "All stocks, equal-weighted"),
    ("Large cap", "largecap", "std", "Market cap ≥ 0.001% of the total, equal-weighted"),
    ("Large cap 0.01%", "largecap001", "std", "Market cap ≥ 0.01% of the total, equal-weighted"),
    ("Large cap VW", "largecap", "vw", "Market cap ≥ 0.001% of the total, value-weighted"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # categorical slots 1-4, fixed per universe
SCORE = "size_oriented_score"
NA = "n/a"


def load(reg, suffix):
    """Report tables of one universe/run, or None if the run is missing."""
    paths = DataPaths()
    rep = paths.result_file("report", reg, suffix, ext=None)
    if not (rep / "sdf_returns_by_period.csv").exists():
        return None

    def opt(name, index):
        f = rep / name
        return pd.read_csv(f).set_index(index) if f.exists() else None

    return {
        "rets": pd.read_csv(paths.result_file("ret", reg, suffix), index_col=0),
        "sdf": pd.read_csv(rep / "sdf_returns_by_period.csv").set_index(["series", "period"]),
        "loadings": pd.read_csv(rep / "sdf_factor_loadings.csv").set_index(["series", "stat"]),
        "perf": pd.read_csv(rep / "score_long_short_by_period.csv").set_index(["universe", "score", "period"]),
        "costs": pd.read_csv(rep / "score_turnover_costs.csv").set_index(["universe", "score"]),
        "contrib": pd.read_csv(rep / "sdf_return_by_size_by_period.csv").set_index(["size bucket", "period"]),
        "exposure": pd.read_csv(rep / "sdf_weight_by_size.csv", index_col=0),
        "refits": pd.read_csv(paths.result_file("node_betas", reg, suffix))["refit_date"].nunique(),
        "hedge": opt("hedge_costs_by_period.csv", ["portfolio", "period"]),
        "ratios": opt("hedge_ratios_summary.csv", ["Unnamed: 0"]),
        "controls": opt("turnover_controls_summary.csv", ["portfolio", "control"]),
    }


def get(frame, key, col, fmt="{:.2f}"):
    """Formatted value, or n/a when the table or row is missing."""
    try:
        return num(frame.loc[key, col], fmt)
    except (AttributeError, KeyError, TypeError):
        return NA


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
    first = runs[0]["d"]
    y0, y1 = first["rets"].index.min() // 10000, first["rets"].index.max() // 10000
    periods = [p for p in first["sdf"].loc["Return_mkt_adj"].index if p != "Full"]
    col_w = [0.22] + [0.78 / n] * n
    mk = {r["label"]: r["d"]["sdf"].loc[("Return_mkt_adj", "Full")] for r in runs}
    hg = {r["label"]: r["d"]["hedge"] for r in runs}

    def small_share(d):
        c = d["contrib"].xs("Full", level="period")["ann. return %"]
        return c.iloc[:2].sum() / c.sum() * 100

    # 1. cover
    fig = deck.slide(dark=True, footer=False)
    fig.text(M / W, 8.1 / H_, "BACKTEST REVIEW · COMPARISON ACROSS UNIVERSES", fontsize=13, weight="bold", color=ACCENT_LIGHT, va="top")
    fig.text(M / W, 5.6 / H_, "Asset Pricing Trees", fontsize=64, family=SERIF, weight="bold", color=BG, va="top")
    fig.text(M / W, 4.2 / H_, f"Does the signal survive in large, tradable stocks? {y0}–{y1}", fontsize=26, color=ON_DARK, va="top")
    fig.text(M / W, 1.0 / H_, " · ".join(labels), fontsize=15, color="#9fb0c6")
    deck.save(fig)

    # 2. universes
    fig = deck.slide("Universes", f"The {n} universes compared")
    rows = []
    for r in runs:
        d = r["d"]
        ex = d["exposure"]
        rows.append([r["label"], r["desc"], f"{d['costs'].loc[('all stocks', SCORE), 'stocks scored/mo']:,.0f}",
                     f"{ex['median mkt_cap'].iloc[0]:,.0f}", f"{ex['median mkt_cap'].iloc[-1]:,.0f}"])
    y = table(fig, M, 6.9, W - 2 * M, ["Universe", "Rule and weighting", "Stocks", "Smallest Q1", "Largest Q5"],
              rows, [0.17, 0.45, 0.12, 0.13, 0.13], align=["left", "left", "right", "right", "right"], size=16, row_h=0.6)
    text(fig, M, y - 0.4, f"Stocks = median number scored a month (every characteristic available). Q1/Q5 = median market cap of "
         "the smallest and largest size quintile within each universe, in the units of the characteristic files. Every run: "
         f"36 tree sets, yearly refits, out of sample {y0}–{y1}, no look-ahead.", W - 2 * M - 1.5, size=16)
    deck.save(fig)

    # 3. summary
    fig = deck.slide("Summary", "Headline results by universe")
    metrics = [
        ("Market-adjusted SDF: Sharpe", lambda r: num(mk[r["label"]]["Sharpe"])),
        ("Alpha, % a year (t)", lambda r: f"{num(mk[r['label']]['alpha ann. %'], '{:.1f}')} ({mk[r['label']]['alpha t (NW)']:.1f})"),
        ("Factor R²", lambda r: num(mk[r["label"]]["factor R2"])),
        ("Hedged SDF: Sharpe", lambda r: num(r["d"]["sdf"].loc[("Return", "Full"), "Sharpe"])),
        ("Hedged: Sharpe after 10 bps", lambda r: get(hg[r["label"]], ("Hedged (Return)", "Full"), "Sharpe after 10bps")),
        ("Hedged: break-even, bps", lambda r: get(hg[r["label"]], ("Hedged (Return)", "Full"), "break-even cost bps", "{:.0f}")),
        ("Score long/short: Sharpe", lambda r: num(r["d"]["perf"].loc[("all stocks", SCORE, "Full"), "Sharpe"])),
        ("Score: Sharpe after 10 bps", lambda r: num(r["d"]["costs"].loc[("all stocks", SCORE), "Sharpe after 10bps"])),
        ("Return from smallest 40%", lambda r: f"{small_share(r['d']):.0f}%"),
    ]
    rows = [[name] + [f(r) for r in runs] for name, f in metrics]
    table(fig, M, 6.9, W - 2 * M, ["1980–2016"] + labels, rows, [0.3] + [0.7 / n] * n, size=15, row_h=0.5, bold=(3, 4))
    deck.save(fig)

    # 4. conclusions up front
    def hedged_net(r):
        h = hg[r["label"]]
        return h.loc[("Hedged (Return)", "Full"), "Sharpe after 10bps"] if h is not None else -np.inf

    best_hedged = max(runs, key=hedged_net)
    fig = deck.slide("Key findings", "What changes when the universe gets larger")
    full_lbl = runs[0]["label"]
    lc = [r for r in runs if r["reg"] != "full"]
    items = [
        ("Raw SDF weakens", f"Market-adjusted Sharpe {num(mk[full_lbl]['Sharpe'])} in the full universe against "
         + ", ".join(f"{num(mk[r['label']]['Sharpe'])} ({r['label']})" for r in lc) + "."),
        ("Factors explain more", f"Factor R² rises from {num(mk[full_lbl]['factor R2'])} to "
         + ", ".join(num(mk[r['label']]['factor R2']) for r in lc) + "; the alpha stays significant."),
        ("Hedging recovers it", "Removing the SDF's factor exposure gives Sharpe ratios of "
         + ", ".join(f"{num(r['d']['sdf'].loc[('Return', 'Full'), 'Sharpe'])} ({r['label']})" for r in runs) + "."),
        ("Best after costs", f"{best_hedged['label']}: hedged Sharpe "
         f"{get(hg[best_hedged['label']], ('Hedged (Return)', 'Full'), 'Sharpe after 10bps')} after 10 bps."),
        ("Stock scores do not carry it", "Score long/short Sharpe after 10 bps: "
         + ", ".join(f"{num(r['d']['costs'].loc[('all stocks', SCORE), 'Sharpe after 10bps'])} ({r['label']})" for r in runs) + "."),
        ("Recent weakness", f"Market-adjusted alpha t in the {periods[-1]}: "
         + ", ".join(f"{r['d']['sdf'].loc[('Return_mkt_adj', periods[-1]), 'alpha t (NW)']:.1f} ({r['label']})" for r in runs) + "."),
    ]
    cw = (W - 2 * M - 2 * 0.3) / 3
    for i, (head, body) in enumerate(items):
        x, top = M + (i % 3) * (cw + 0.3), 6.9 - (i // 3) * 2.75
        box(fig, x, top, cw, 2.45)
        fig.text((x + 0.3) / W, (top - 0.3) / H_, head, fontsize=17, weight="bold", color=NAVY, va="top")
        text(fig, x + 0.3, top - 0.85, body, cw - 0.6, size=14)
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
    ps = ["Full"] + periods
    per = lambda p: f"{y0}–{y1}" if p == "Full" else p
    fig.text(M / W, 6.95 / H_, "Sharpe ratio", fontsize=15, weight="bold", color=ACCENT, va="top")
    rows = [[per(p)] + [num(r["d"]["sdf"].loc[("Return_mkt_adj", p), "Sharpe"]) for r in runs] for p in ps]
    y = table(fig, M, 6.6, W - 2 * M, ["Period"] + labels, rows, col_w, size=15, row_h=0.42, bold=(0,))
    fig.text(M / W, (y - 0.3) / H_, "Alpha t-stat against the characteristic factors", fontsize=15, weight="bold", color=ACCENT, va="top")
    rows = [[per(p)] + [num(r["d"]["sdf"].loc[("Return_mkt_adj", p), "alpha t (NW)"], "{:.1f}") for r in runs] for p in ps]
    table(fig, M, y - 0.65, W - 2 * M, ["Period"] + labels, rows, col_w, size=15, row_h=0.42, bold=(0,))
    deck.save(fig)

    # 7. factor exposure
    fig = deck.slide("Factor exposure", "Factor loadings of the market-adjusted SDF (t-stat)")
    factors = [c for c in first["loadings"].columns if c != "const"]
    rows = [[f] + [f"{num(r['d']['loadings'].loc[('Return_mkt_adj', 'beta'), f])} ({r['d']['loadings'].loc[('Return_mkt_adj', 't'), f]:.1f})"
                   for r in runs] for f in factors]
    rows.append(["Factor R²"] + [num(mk[r["label"]]["factor R2"]) for r in runs])
    table(fig, M, 6.9, W - 2 * M, ["Factor"] + labels, rows, col_w, size=15, row_h=0.5, bold=(len(factors),))
    deck.save(fig)

    # 8. hedged portfolio after costs
    fig = deck.slide("Tradable portfolio", "The factor-hedged SDF after trading costs")
    hk = "Hedged (Return)"
    stats = [("Sharpe (gross)", "Sharpe", "{:.2f}"), ("Return, % a year", "ann. return %", "{:.2f}"),
             ("One-way turnover, % of book", "one-way turnover % of book", "{:.0f}"), ("Break-even cost, bps", "break-even cost bps", "{:.0f}"),
             ("Sharpe after 5 bps", "Sharpe after 5bps", "{:.2f}"), ("Sharpe after 10 bps", "Sharpe after 10bps", "{:.2f}"),
             ("Sharpe after 25 bps", "Sharpe after 25bps", "{:.2f}")]
    rows = [[name] + [get(hg[r["label"]], (hk, "Full"), col, fmt) for r in runs] for name, col, fmt in stats]
    rows.append(["SDF alone, after 10 bps"] + [get(hg[r["label"]], ("SDF alone (Return_mkt_adj)", "Full"), "Sharpe after 10bps") for r in runs])
    y = table(fig, M, 6.9, W - 2 * M, ["1980–2016"] + labels, rows, [0.3] + [0.7 / n] * n, size=15, row_h=0.5, bold=(5,))
    text(fig, M, y - 0.35, "Hedged: the SDF's stock positions minus its factor exposure, held through the factor portfolios' stocks "
         "with hedge ratios from the last refit (fully out of sample). Costs are one-way, per unit traded.", W - 2 * M - 1.5, size=15)
    deck.save(fig)

    # 9. hedged by period after 10 bps
    fig = deck.slide("Tradable portfolio by period", "Hedged SDF: Sharpe ratio after 10 bps by period")
    rows = [[per(p)] + [get(hg[r["label"]], (hk, p), "Sharpe after 10bps") for r in runs] for p in ps]
    y = table(fig, M, 6.9, W - 2 * M, ["Period"] + labels, rows, col_w, size=16, row_h=0.48, bold=(0,))
    rows = [[per(p)] + [get(hg[r["label"]], (hk, p), "break-even cost bps", "{:.0f}") for r in runs] for p in ps]
    fig.text(M / W, (y - 0.3) / H_, "Break-even cost, bps", fontsize=15, weight="bold", color=ACCENT, va="top")
    table(fig, M, y - 0.65, W - 2 * M, ["Period"] + labels, rows, col_w, size=15, row_h=0.4, bold=(0,))
    deck.save(fig)

    # 10. hedge ratios
    fig = deck.slide("Hedge", "Factor exposure the hedge removes (mean hedge ratio)")
    rows = [[f] + [get(r["d"]["ratios"], f, "mean h", "{:.3f}") for r in runs] for f in factors]
    y = table(fig, M, 6.9, W - 2 * M, ["Factor"] + labels, rows, col_w, size=15, row_h=0.5)
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
                                                       "one-way turnover % of book", "{:.0f}") + "%" for r in runs])
    y = table(fig, M, 6.9, W - 2 * M, ["Hedged: Sharpe @10 / @25 bps"] + labels, rows, [0.3] + [0.7 / n] * n, size=15, row_h=0.52)
    text(fig, M, y - 0.35, "Partial rebalancing moves a fraction of the way to the new target each month; averaging holds the mean of "
         "recent targets. Slower trading gives up gross return because the signal decays quickly, but it pays at higher costs.",
         W - 2 * M - 1.5, size=15)
    deck.save(fig)

    # 12. stock scores
    fig = deck.slide("Stock scores", f"{SCORE} long/short across universes")
    stats = [("Sharpe (gross)", lambda d: num(d["perf"].loc[("all stocks", SCORE, "Full"), "Sharpe"])),
             ("Alpha t", lambda d: num(d["perf"].loc[("all stocks", SCORE, "Full"), "alpha t (NW)"], "{:.1f}")),
             ("ICIR", lambda d: num(d["perf"].loc[("all stocks", SCORE, "Full"), "ICIR"])),
             ("One-way turnover", lambda d: f"{d['costs'].loc[('all stocks', SCORE), 'one-way turnover %/mo']:.0f}%"),
             ("Break-even, bps", lambda d: num(d["costs"].loc[("all stocks", SCORE), "break-even cost bps"], "{:.0f}")),
             ("Sharpe after 10 bps", lambda d: num(d["costs"].loc[("all stocks", SCORE), "Sharpe after 10bps"])),
             (f"Sharpe, {periods[-1]}", lambda d: num(d["perf"].loc[("all stocks", SCORE, periods[-1]), "Sharpe"]))]
    rows = [[name] + [f(r["d"]) for r in runs] for name, f in stats]
    y = table(fig, M, 6.9, W - 2 * M, ["1980–2016"] + labels, rows, [0.3] + [0.7 / n] * n, size=16, row_h=0.55)
    text(fig, M, y - 0.35, "Score-weighted long/short (100% long, 100% short) on same-month returns. The stock scores trade far more "
         "than the hedged SDF positions and lose most of their edge to costs outside the full universe.", W - 2 * M - 1.5, size=15)
    deck.save(fig)

    # 13. size concentration
    fig = deck.slide("Size concentration", "Contribution to the market-adjusted SDF by size quintile")
    ax = chart(fig, M + 0.7, 2.0, W - 2 * M - 0.8, 4.8)
    buckets = list(first["contrib"].index.get_level_values("size bucket").unique())
    x = np.arange(len(buckets))
    w = 0.8 / n
    for i, r in enumerate(runs):
        vals = r["d"]["contrib"].xs("Full", level="period").loc[buckets, "ann. return %"]
        ax.bar(x + (i - (n - 1) / 2) * w, vals, width=w, color=r["color"], edgecolor=mc.SURFACE, linewidth=2, label=r["label"])
    ax.axhline(0, color=mc.INK2, linewidth=1.2)
    ax.set_xticks(x, buckets)
    ax.set_ylabel("Contribution, % a year")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=n)
    text(fig, M, 1.5, "Quintiles are formed within each universe, so Q1 of the large-cap universes holds much larger stocks than Q1 of "
         "the full universe. Share of the return from the two smallest quintiles: "
         + ", ".join(f"{small_share(r['d']):.0f}% ({r['label']})" for r in runs) + ".", W - 2 * M, size=15, color=TEXT)
    deck.save(fig)

    # 14. caveats and next steps
    fig = deck.slide("Caveats and next steps", "Read the numbers with these in mind")
    items = [
        ("Factor normaliser", "Factor returns and the hedge depend on rank_normalise; a stand-in version was used here."),
        ("Cost model", "Flat one-way costs per unit traded; no market impact, short-sale costs or financing."),
        ("Hedge implementation", "The hedge trades the factor portfolios' stocks, which adds turnover and short positions."),
        ("Recent period", f"Every universe is weakest in the {periods[-1]}; test beyond {y1} before relying on it."),
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


H_ = 9

if __name__ == "__main__":
    runs = []
    for (label, reg, suffix, desc), color in zip(UNIVERSES, COLORS):
        d = load(reg, suffix)
        if d is None:
            print(f"Skipping {label}: no results for {reg} ({suffix})")
            continue
        runs.append({"label": label, "reg": reg, "suffix": suffix, "desc": desc, "color": color, "d": d})
    out = mc.REPO / "presentation" / "backtest_comparison.pdf"
    preview = None
    if "--png" in sys.argv:
        preview = mc.REPO / "presentation" / "deck_preview"
        preview.mkdir(exist_ok=True)
    build(runs, out, preview)
    print(f"Saved {out} ({', '.join(r['label'] for r in runs)})" + (f"; previews in {preview}" if preview else ""))
