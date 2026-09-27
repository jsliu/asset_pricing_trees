# %%
"""
The factor-hedged SDF return ('Return') as a tradable position, and what trading costs leave of it.

'Return' is the SDF minus its factor exposure: Return = Return_mkt_adj - h(d) . F(d), where F are the
characteristic factor returns and h the hedge ratios implied by the betas of the last refit (fixed until the
next refit). This script
    1. recovers h for every refit period from Return_mkt_adj - Return (exact: 9 unknowns, 12 months),
    2. builds the stock positions of the hedged portfolio: sdf_weight - sum_k h_k x (factor k's stock weights),
       and checks that they reproduce 'Return',
    3. measures turnover and the net return after trading costs, for the SDF alone and hedged, per period.

Run backtest_report.py for the same universe first (it saves the factor returns), then from the project root:
    python hedge_analysis.py <universe> [<suffix>]      e.g. python hedge_analysis.py largecap
Tables are printed and saved to the report folder (e.g. result/report_std_largecap/hedge_*.csv).
"""
import numpy as np
import pandas as pd

from backtest_report import REG, SUFFIX, RET_NAME, EQUAL_WEIGHTED, make_periods
from src.constants import Chars, DataPaths
from src.functions import _get_weights
from src.preprocessing import read_char_data

COSTS_BPS = (5, 10, 25)          # one-way cost per unit traded


def refit_blocks(dates, refit_dates):
    """Refit date in force at each date."""
    refit_dates = np.sort(refit_dates)
    return pd.Series(refit_dates[np.searchsorted(refit_dates, dates, side='right') - 1], index=dates)


def hedge_ratios(hedge_ret, factors, blocks):
    """h for each refit period, from hedge_ret = h . F within the period (least squares, no intercept)."""
    rows, max_resid = {}, 0.0
    for refit, idx in blocks.groupby(blocks).groups.items():
        f = factors.loc[idx].to_numpy()
        y = hedge_ret.loc[idx].to_numpy()
        h, *_ = np.linalg.lstsq(f, y, rcond=None)
        max_resid = max(max_resid, np.abs(y - f @ h).max())
        rows[refit] = h
    return pd.DataFrame(rows, index=factors.columns).T, max_resid


def traded(weights):
    """Monthly sum of |weight changes| on a full date x stock grid (stocks leaving count as sells)."""
    grid = weights.unstack(fill_value=0).fillna(0)
    return grid.diff().abs().sum(axis=1).iloc[1:], grid.abs().sum(axis=1)


def net_stats(ret, tr, gross, periods):
    """Gross and after-cost statistics of a return series with its trading, per period."""
    rows = {}
    for name, (a, b) in periods.items():
        r = ret[(ret.index >= a) & (ret.index <= b)]
        t = tr.reindex(r.index).fillna(0)
        g = gross.reindex(r.index)
        out = {
            'ann. return %': r.mean() * 1200,
            'Sharpe': r.mean() / r.std() * np.sqrt(12),
            'gross exposure': g.mean(),
            'one-way turnover % of book': (t / 2 / g).mean() * 100,
            'break-even cost bps': r.mean() / t.mean() * 1e4,
        }
        for bps in COSTS_BPS:
            net = r - t * bps / 1e4
            out[f'Sharpe after {bps}bps'] = net.mean() / net.std() * np.sqrt(12)
        rows[name] = out
    return pd.DataFrame(rows).T


def build_positions(reg, suffix, equal_weighted):
    """
    Everything needed to hold the SDF alone or hedged: the SDF stock weights, the stock weights of the factor
    hedge (sdf position minus hedge = the hedged portfolio), the stock returns net of the markets they are
    measured against, the hedge ratios and the reproduction checks.
    """
    paths = DataPaths()
    out_dir = paths.result_file('report', reg, suffix, ext=None)
    rets = pd.read_csv(paths.result_file('ret', reg, suffix), index_col=0)
    sdf_w = pd.read_csv(paths.result_file('score', reg, suffix), usecols=['date', 'permno', 'sdf_weight']).set_index(['date', 'permno'])['sdf_weight']
    refits = pd.read_csv(paths.result_file('node_betas', reg, suffix))['refit_date'].unique()
    factors = pd.read_csv(out_dir / 'factor_returns.csv', index_col=0)

    features = list(Chars().__dict__.values())[:-2]
    data = read_char_data(features, universe=None if reg == 'full' else reg, ret_name=RET_NAME).swaplevel(0, 1).sort_index()
    data = data[data.index.get_level_values('date') >= rets.index.min()]
    ret = data[RET_NAME]
    ew_market = ret.groupby('date').transform('mean')
    if equal_weighted:
        sdf_market = ew_market
    else:
        sdf_market = (ret * data['mkt_cap']).groupby('date').transform('sum') / data['mkt_cap'].groupby('date').transform('sum')

    # hedge ratios per refit period
    dates = rets.index
    blocks = refit_blocks(dates, refits)
    h, resid = hedge_ratios(rets['Return_mkt_adj'] - rets['Return'], factors.loc[dates], blocks)

    # stock weights of the factor portfolios and of the hedge
    fw = pd.DataFrame({f: data[f].groupby('date').transform(lambda x: _get_weights(x, score_weighted=True)) for f in features})
    factor_gap = ((fw.mul(ret - ew_market, axis=0)).groupby('date').sum().loc[dates] - factors.loc[dates]).abs().max().max()
    h_by_date = h.loc[blocks.to_numpy()].set_axis(dates)
    hedge_w = (fw.fillna(0) * h_by_date.reindex(fw.index.get_level_values('date')).to_numpy()).sum(axis=1)
    hedge_w = hedge_w[hedge_w.index.get_level_values('date').isin(dates)]

    sdf_adj, ew_adj = (ret - sdf_market), (ret - ew_market)
    hedged_ret = ((sdf_w * sdf_adj.reindex(sdf_w.index)).groupby('date').sum()
                  - (hedge_w * ew_adj.reindex(hedge_w.index)).groupby('date').sum())
    return {
        'out_dir': out_dir, 'rets': rets, 'h': h, 'sdf_w': sdf_w, 'hedge_w': hedge_w,
        'sdf_adj': sdf_adj, 'ew_adj': ew_adj, 'dates': dates,
        'checks': {'hedge ratio fit': resid, 'factor weights vs factor returns': factor_gap,
                   'hedged positions vs Return': (hedged_ret - rets['Return']).abs().max()},
    }


# %%
if __name__ == '__main__':
    pd.set_option('display.width', 250)
    p = build_positions(REG, SUFFIX, EQUAL_WEIGHTED)
    for name, gap in p['checks'].items():
        print(f'Check: {name}, max abs diff {gap:.1e}')
    rets, h, out_dir = p['rets'], p['h'], p['out_dir']
    combined = p['sdf_w'].sub(p['hedge_w'], fill_value=0)

    periods = make_periods(sorted(p['dates']))
    tr_sdf, gross_sdf = traded(p['sdf_w'])
    tr_hedged, gross_hedged = traded(combined)
    table = pd.concat({
        'SDF alone (Return_mkt_adj)': net_stats(rets['Return_mkt_adj'], tr_sdf, gross_sdf, periods),
        'Hedged (Return)': net_stats(rets['Return'], tr_hedged, gross_hedged, periods),
    }, names=['portfolio', 'period'])
    ratios = pd.DataFrame({'mean h': h.mean(), 'mean |h|': h.abs().mean(), 'share of refits h > 0 %': (h > 0).mean() * 100})

    table.to_csv(out_dir / 'hedge_costs_by_period.csv')
    ratios.to_csv(out_dir / 'hedge_ratios_summary.csv')
    h.to_csv(out_dir / 'hedge_ratios_by_refit.csv')
    print(f'\n==== {REG} ({SUFFIX}): SDF alone vs factor-hedged, before and after costs ====')
    print(table.round(2).to_string())
    print('\nHedge ratios (factor exposure removed per unit of SDF)')
    print(ratios.round(3).to_string())
    print(f'\nSaved to {out_dir}')
