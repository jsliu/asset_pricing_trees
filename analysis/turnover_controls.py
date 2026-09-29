# %%
"""
Turnover controls for the SDF portfolio, alone and factor-hedged, applied to the stock positions built in
hedge_analysis.py:
    partial rebalancing: each month trade only a fraction lam of the way to the new target,
        W_t = lam * target_t + (1 - lam) * W_(t-1)          (lam = 1: no control)
    signal averaging: hold the average of the last k monthly targets.
Stocks without a return in a month are sold. Weight drift between rebalances is ignored.

Run backtest_report.py for the same universe first, then from the project root:
    python analysis/turnover_controls.py [<universe>] [--region GL] [--vw | --variant TAGS]      e.g. python analysis/turnover_controls.py largecap
Tables are printed and saved to the report folder (turnover_controls_*.csv).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, for src/ and backtest.py

import numpy as np
import pandas as pd

from analysis.backtest_report import REGION, UNIVERSE, VARIANT, LABEL, make_periods
from analysis.hedge_analysis import build_positions, net_stats

LAMBDAS = (1.0, 0.5, 0.33, 0.2)
AVG_MONTHS = (3, 6)


def to_grid(s, dates, stocks):
    """(date, permno) series as a date x stock array, 0 where missing."""
    return s.unstack().reindex(index=dates, columns=stocks).fillna(0).to_numpy()


def partial(target, held, lam):
    out, prev = np.zeros_like(target), np.zeros(target.shape[1])
    for t in range(len(target)):
        prev = (lam * target[t] + (1 - lam) * prev) * held[t]
        out[t] = prev
    return out


def averaged(target, held, k):
    cs = np.cumsum(target, axis=0)
    total = cs.copy()
    total[k:] = cs[k:] - cs[:-k]
    n = np.minimum(np.arange(1, len(target) + 1), k)[:, None]
    return total / n * held


def evaluate(sdf_leg, hedge_leg, sdf_adj, ew_adj, dates, periods, hedged):
    """Return, trading and statistics of holding sdf_leg (minus hedge_leg if hedged)."""
    ret = (sdf_leg * sdf_adj).sum(axis=1) - ((hedge_leg * ew_adj).sum(axis=1) if hedged else 0)
    pos = sdf_leg - hedge_leg if hedged else sdf_leg
    tr = pd.Series(np.abs(np.diff(pos, axis=0)).sum(axis=1), index=dates[1:])
    gross = pd.Series(np.abs(pos).sum(axis=1), index=dates)
    return pd.Series(ret, index=dates), net_stats(pd.Series(ret, index=dates), tr, gross, periods)


# %%
if __name__ == '__main__':
    pd.set_option('display.width', 250)
    p = build_positions(REGION, UNIVERSE, VARIANT)
    dates = np.array(p['dates'])
    stocks = p['sdf_adj'].index.get_level_values('permno').unique().union(p['hedge_w'].index.get_level_values('permno').unique())
    held = to_grid(p['sdf_adj'].notna().astype(float), dates, stocks)
    sdf_adj, ew_adj = to_grid(p['sdf_adj'], dates, stocks), to_grid(p['ew_adj'], dates, stocks)
    sdf_target, hedge_target = to_grid(p['sdf_w'], dates, stocks), to_grid(p['hedge_w'], dates, stocks)
    periods = make_periods(sorted(dates))
    dates_idx = pd.Index(dates, name='date')

    controls = {f'rebalance {lam:.0%} a month': (lambda x, lam=lam: partial(x, held, lam)) for lam in LAMBDAS}
    controls |= {f'average of last {k} months': (lambda x, k=k: averaged(x, held, k)) for k in AVG_MONTHS}

    tables = {}
    for name, control in controls.items():
        sdf_leg, hedge_leg = control(sdf_target), control(hedge_target)
        for portfolio, hedged in [('SDF alone', False), ('Hedged', True)]:
            ret, table = evaluate(sdf_leg, hedge_leg, sdf_adj, ew_adj, dates_idx, periods, hedged)
            tables[(portfolio, name)] = table
            if name == 'rebalance 100% a month':
                target = p['rets']['Return' if hedged else 'Return_mkt_adj']
                print(f'Check: {portfolio}, no control reproduces the backtest return, max abs diff {(ret - target).abs().max():.1e}')

    table = pd.concat(tables, names=['portfolio', 'control', 'period'])
    table.to_csv(p['out_dir'] / 'turnover_controls_by_period.csv')
    full = table.xs('Full', level='period')[['ann. return %', 'Sharpe', 'one-way turnover % of book', 'break-even cost bps',
                                             'Sharpe after 10bps', 'Sharpe after 25bps']]
    decades = table['Sharpe after 10bps'].unstack('period').drop(columns='Full')
    decades.columns = [f'{c} @10bps' for c in decades.columns]
    summary = full.join(decades)
    summary.to_csv(p['out_dir'] / 'turnover_controls_summary.csv')
    print(f'\n==== {LABEL}: turnover controls, full sample; last columns Sharpe after 10 bps by period ====')
    print(summary.round(2).to_string())
    print(f'\nSaved to {p["out_dir"]}')
