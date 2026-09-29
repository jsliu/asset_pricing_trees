# %%
"""
Stand-alone performance report for a run_backtest() output in result/:
    ret file    SDF returns ('Return' factor-hedged, 'Return_mkt_adj' market-adjusted)
    score file  stock scores
(names from DataPaths.result_file, e.g. result/ret_largecap.csv, result/ret_largecap_vw.csv or result/GL_ret.csv)

For the full sample and each period (decades by default) it reports return, risk, Sharpe ratio and
alpha against the characteristic factor returns, for
    1. the SDF returns,
    2. the score-weighted long/short of each stock score, in several size universes, with rank IC,
       turnover and break-even trading cost,
    3. the return difference between pairs of scores,
    4. the market-cap breakdown: the SDF's market-adjusted return split by size quintile (sdf_weight x
       market-adjusted stock return, which adds up to Return_mkt_adj), and the score long/short within
       each size quintile.
Tables are printed and saved to the 'report' result folder, e.g. result/report_largecap/.
Run: python analysis/backtest_report.py [<universe>] [--region GL] [--vw] [--variant TAGS], e.g. largecap --vw for
the value-weighted run or largecap --variant slow3_val for an experiment (see run_variant in src/constants.py);
universe defaults to 'full' when no region is given.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, for src/ and backtest.py

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from src.constants import Chars, DataPaths, parse_variant
from src.preprocessing import read_backtest_data
from src.functions import calc_fac_ret, _get_weights



def _run_args():
    """python <script> [<universe>] [--region GL] [--vw] [--variant TAGS]; universe defaults to 'full' when no
    region is given; --vw selects the value-weighted run, --variant any run (e.g. 'vw', 'slow3_val')."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('universe', nargs='?')
    parser.add_argument('--region')
    parser.add_argument('--vw', action='store_true')
    parser.add_argument('--variant')
    args, _ = parser.parse_known_args()
    universe = args.universe if (args.universe or args.region) else 'full'
    return args.region, universe, args.variant or ('vw' if args.vw else None)


REGION, UNIVERSE, VARIANT = _run_args()            # VARIANT: file-name part of the run, e.g. None, 'vw', 'slow3_val'
EQUAL_WEIGHTED = parse_variant(VARIANT)[0]         # market adjustment of the run (backtest.py)
LABEL = DataPaths().label(REGION, UNIVERSE, VARIANT)
RET_NAME = 'gross_returns' if UNIVERSE is None else 'ret'
SCORES = ['sdf_weight', 'size_oriented_score', 'norm_score']
PERIODS = None                  # None = decades, or {'name': (start, end)} with YYYYMMDD ints
N_SIZE_BUCKETS = 5              # market-cap buckets (quintiles) for the size breakdown
NW_LAGS = 6                     # Newey-West lags for alpha t-stats
COSTS_BPS = (10, 25)            # one-way trading costs for the after-cost Sharpe ratios


def make_periods(dates, periods=None):
    """Full sample plus the given periods (decades if None), as {name: (start, end)}."""
    if periods is None:
        decades = sorted({d // 100000 * 10 for d in dates})
        periods = {f'{dec}s': (dec * 10000 + 101, (dec + 9) * 10000 + 1231) for dec in decades}
    return {'Full': (min(dates), max(dates)), **periods}


def perf_stats(x, factors=None):
    """Return/risk statistics of a monthly return series, and its alpha against the factors."""
    x = x.dropna()
    cum = x.cumsum()
    out = {
        'months': len(x),
        'ann. return %': x.mean() * 1200,
        'ann. vol %': x.std() * np.sqrt(12) * 100,
        'Sharpe': x.mean() / x.std() * np.sqrt(12),
        't-stat': x.mean() / x.std() * np.sqrt(len(x)),
        'hit rate %': (x > 0).mean() * 100,
        'max drawdown %': (cum - cum.cummax()).min() * 100,
    }
    if factors is not None:
        m = sm.OLS(x, sm.add_constant(factors.loc[x.index])).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
        resid_vol = np.sqrt(m.scale)
        out.update({
            'alpha ann. %': m.params['const'] * 1200,
            'alpha t (NW)': m.tvalues['const'],
            'residual vol %': resid_vol * np.sqrt(12) * 100,
            'appraisal ratio': m.params['const'] / resid_vol * np.sqrt(12),
            'factor R2': m.rsquared,
        })
    return out


def by_period(x, periods, factors=None):
    """perf_stats of x for each period, one row per period."""
    return pd.DataFrame({
        name: perf_stats(x[(x.index >= start) & (x.index <= end)], factors)
        for name, (start, end) in periods.items()
    }).T


def factor_loadings(x, factors):
    """Full-sample factor betas and Newey-West t-stats of x."""
    x = x.dropna()
    m = sm.OLS(x, sm.add_constant(factors.loc[x.index])).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    return pd.DataFrame({'beta': m.params, 't': m.tvalues}).T


def long_short(score, ret):
    """Score-weighted long/short return (same construction as backtest_analysis.py)."""
    s = score.dropna().rename('score')
    return calc_fac_ret(s, ret, date_col='date', score_weighted=True)['score']


def rank_ic(score, ret):
    """Monthly Spearman correlation between the score and the same-month stock return."""
    both = pd.concat([score.rename('score'), ret.rename('ret')], axis=1, join='inner').dropna()
    return both.groupby('date').apply(lambda g: g['score'].corr(g['ret'], method='spearman'))


def traded(score):
    """Monthly sum of |weight changes| of the score-weighted long/short (gross 2), on a full date x stock
    grid so that stocks leaving the portfolio count as sells."""
    w = score.dropna().groupby('date').transform(lambda x: _get_weights(x, score_weighted=True))
    return w.unstack(fill_value=0).fillna(0).diff().abs().sum(axis=1).iloc[1:]


def size_buckets(mkt_cap, n=N_SIZE_BUCKETS):
    """Market-cap bucket of each stock each month, among all stocks with a return: Q1 smallest ... Qn largest."""
    labels = [f'Q{i + 1}' + (' small' if i == 0 else ' large' if i == n - 1 else '') for i in range(n)]
    pct = mkt_cap.groupby('date').rank(pct=True)
    return pd.cut(pct, np.linspace(0, 1, n + 1), labels=labels, include_lowest=True)


def size_attribution(weights, ret_adj, buckets):
    """
    Split the SDF's market-adjusted return by size bucket: sum of weight x market-adjusted return of the
    stocks in each bucket (columns add up to the SDF return), and the bucket's share of the portfolio.
    """
    df = pd.concat([weights.rename('w'), ret_adj.rename('r'), buckets.rename('bucket')], axis=1, join='inner')
    df['contrib'] = df['w'] * df['r']
    g = df.groupby(['date', 'bucket'], observed=False)
    contrib = g['contrib'].sum().unstack()
    gross = df['w'].abs().groupby([df.index.get_level_values('date'), df['bucket']], observed=False).sum().unstack()
    exposure = pd.DataFrame({
        'share of gross weight %': (gross.div(gross.sum(axis=1), axis=0) * 100).mean(),
        'net weight %': g['w'].sum().unstack().mean() * 100,
        'stocks held/mo': (df['w'] != 0).groupby([df.index.get_level_values('date'), df['bucket']], observed=False).sum().unstack().median(),
    })
    return contrib, exposure


def diff_test(a, b, periods):
    """Mean monthly return difference a - b, with Newey-West t-stat, per period."""
    diff = (a - b).dropna()
    rows = {}
    for name, (start, end) in periods.items():
        dp = diff[(diff.index >= start) & (diff.index <= end)]
        m = sm.OLS(dp, np.ones(len(dp))).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
        rows[name] = {'mean diff %/mo': dp.mean() * 100, 't (NW)': m.tvalues.iloc[0], 'corr': a.loc[dp.index].corr(b.loc[dp.index])}
    return pd.DataFrame(rows).T


# %%
if __name__ == '__main__':
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    paths = DataPaths()
    out_dir = paths.result_file('report', REGION, UNIVERSE, VARIANT, ext=None)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- inputs ----------------
    rets = pd.read_csv(paths.result_file('ret', REGION, UNIVERSE, VARIANT), index_col=0)
    scores = pd.read_csv(paths.result_file('score', REGION, UNIVERSE, VARIANT)).set_index(['date', 'permno'])
    start = rets.index.min()

    features = list(Chars().__dict__.values())[:-2]
    data = read_backtest_data(features, RET_NAME, region=REGION, universe=UNIVERSE).swaplevel(0, 1).sort_index()
    print('Computing factor returns')
    factors = calc_fac_ret(data[features], data[RET_NAME], date_col='date', score_weighted=True)
    factors.to_csv(out_dir / 'factor_returns.csv')

    data = data[data.index.get_level_values('date') >= start]
    ret = data[RET_NAME]
    periods = make_periods(sorted(rets.index), PERIODS)

    # ---------------- check: sdf_weight reproduces Return_mkt_adj ----------------
    if 'sdf_weight' in scores:
        if EQUAL_WEIGHTED:
            market = ret.groupby('date').transform('mean')
        else:
            market = (ret * data['mkt_cap']).groupby('date').transform('sum') / data['mkt_cap'].groupby('date').transform('sum')
        w = scores['sdf_weight'].fillna(0)
        ret_adj = ret - market
        held = (w * ret_adj.reindex(w.index)).groupby('date').sum()
        gap = (held - rets['Return_mkt_adj'].loc[held.index]).abs().max()
        print(f'Check: sdf_weight portfolio vs Return_mkt_adj, max abs diff {gap:.1e}')

    buckets = size_buckets(data['mkt_cap'])

    # ---------------- 1. SDF returns ----------------
    sdf_tables = {col: by_period(rets[col], periods, factors) for col in rets.columns}
    sdf_table = pd.concat(sdf_tables, names=['series', 'period'])
    sdf_loadings = pd.concat({col: factor_loadings(rets[col], factors) for col in rets.columns}, names=['series', 'stat'])
    sdf_table.to_csv(out_dir / 'sdf_returns_by_period.csv')
    sdf_loadings.to_csv(out_dir / 'sdf_factor_loadings.csv')
    print('\n==== 1. SDF returns ====')
    print(sdf_table.round(2).to_string())
    print('\nFactor loadings (full sample)')
    print(sdf_loadings.round(2).to_string())

    # ---------------- 2. stock-score long/short by size universe ----------------
    size_pct = data['mkt_cap'].groupby('date').rank(pct=True)
    size_rank = data['mkt_cap'].groupby('date').rank(ascending=False)
    universes = {
        'all stocks': data.index,
        'size >= 20th pct': data.index[size_pct >= 0.2],
        'size >= 50th pct': data.index[size_pct >= 0.5],
        'top 1000': data.index[size_rank <= 1000],
        **{f'size {b}': data.index[buckets == b] for b in buckets.cat.categories},
    }
    score_cols = [c for c in SCORES if c in scores]

    perf, costs, diffs, ls_all = {}, {}, {}, {}
    for uni, idx in universes.items():
        ret_u = ret.loc[idx]
        ls_u = {}
        for col in score_cols:
            print(f'Scoring {col} in {uni}')
            s = scores[col].dropna()
            s = s.loc[s.index.intersection(idx)]
            ls = long_short(s, ret_u)
            ic = rank_ic(s, ret_u)
            tr = traded(s).reindex(ls.index).fillna(0)
            ls_u[col] = ls

            table = by_period(ls, periods, factors)
            ic_by = {name: ic[(ic.index >= a) & (ic.index <= b)] for name, (a, b) in periods.items()}
            table['rank IC'] = pd.Series({k: v.mean() for k, v in ic_by.items()})
            table['ICIR'] = pd.Series({k: v.mean() / v.std() * np.sqrt(12) for k, v in ic_by.items()})
            perf[(uni, col)] = table

            cost = {
                'stocks scored/mo': s.groupby('date').size().median(),
                'one-way turnover %/mo': tr.mean() / 2 * 100,
                'break-even cost bps': ls.mean() / tr.mean() * 1e4,
            }
            for bps in COSTS_BPS:
                net = ls - tr * bps / 1e4
                cost[f'Sharpe after {bps}bps'] = net.mean() / net.std() * np.sqrt(12)
            costs[(uni, col)] = cost
        for a, b in combinations(score_cols, 2):
            diffs[(uni, f'{a} - {b}')] = diff_test(ls_u[a], ls_u[b], periods)
        ls_all[uni] = ls_u

    perf_table = pd.concat(perf, names=['universe', 'score', 'period'])
    cost_table = pd.DataFrame(costs).T.rename_axis(['universe', 'score'])
    diff_table = pd.concat(diffs, names=['universe', 'comparison', 'period'])
    perf_table.to_csv(out_dir / 'score_long_short_by_period.csv')
    cost_table.to_csv(out_dir / 'score_turnover_costs.csv')
    diff_table.to_csv(out_dir / 'score_difference_tests.csv')

    print('\n==== 2. Stock-score long/short ====')
    print(perf_table.round(2).to_string())
    print('\nTurnover and trading costs (full sample)')
    print(cost_table.round(2).to_string())
    print('\n==== 3. Score differences ====')
    print(diff_table.round(2).to_string())

    # ---------------- 4. market-cap breakdown ----------------
    print('\n==== 4. Market-cap breakdown ====')
    if 'sdf_weight' in scores:
        contrib, exposure = size_attribution(scores['sdf_weight'].fillna(0), ret_adj, buckets)
        exposure['median mkt_cap'] = data['mkt_cap'].groupby(buckets, observed=False).median()
        gap = (contrib.sum(axis=1) - rets['Return_mkt_adj'].loc[contrib.index]).abs().max()
        print(f'Check: size-bucket contributions add up to Return_mkt_adj, max abs diff {gap:.1e}')
        contrib_table = pd.concat({b: by_period(contrib[b], periods, factors) for b in contrib.columns}, names=['size bucket', 'period'])
        contrib.to_csv(out_dir / 'sdf_return_by_size_monthly.csv')
        exposure.to_csv(out_dir / 'sdf_weight_by_size.csv')
        contrib_table.to_csv(out_dir / 'sdf_return_by_size_by_period.csv')
        print('\nSDF portfolio (sdf_weight) by size bucket')
        print(exposure.round(2).to_string())
        print('\nContribution of each size bucket to Return_mkt_adj')
        print(contrib_table.round(2).to_string())

    quintiles = [u for u in universes if u.startswith('size Q')]
    for stat in ['Sharpe', 'alpha t (NW)']:
        view = perf_table.loc[quintiles, stat].unstack('period')[list(periods)]
        print(f'\nScore long/short within each size bucket: {stat}')
        print(view.round(2).to_string())

    # ---------------- plots ----------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    cum = rets.cumsum()
    cum.index = pd.to_datetime(cum.index.astype(str), format='%Y%m%d')
    cum.plot(ax=axes[0], title=f'{LABEL}: SDF cumulative returns')
    ls_cum = pd.DataFrame(ls_all['all stocks']).cumsum()
    ls_cum.index = pd.to_datetime(ls_cum.index.astype(str), format='%Y%m%d')
    ls_cum.plot(ax=axes[1], title=f'{LABEL}: score long/short, all stocks')
    if 'sdf_weight' in scores:
        contrib_cum = contrib.cumsum()
        contrib_cum.index = pd.to_datetime(contrib_cum.index.astype(str), format='%Y%m%d')
        contrib_cum.plot(ax=axes[2], title=f'{LABEL}: Return_mkt_adj by size bucket (cumulative)')
    fig.tight_layout()
    fig.savefig(out_dir / 'cumulative_returns.png', dpi=120)
    plt.show()

    print(f'\nTables and plot saved to {out_dir}')
