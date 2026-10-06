# %%
"""
Hedge ratios for the stock scores' long/shorts, estimated the way the SDF's are: from the factor exposure of the
portfolio the current model holds, over its training history.

At each refit the SDF's betas come from the tree portfolios' past returns, and those portfolios re-sort the stocks
into the nodes every month. This script does the same for the scores: with the node weights (beta) of the refit, it
recomputes the scores (SCORES, backtest.score_stocks) in each of the previous HEDGE_WINDOW months (all there are, if fewer)
from that month's stocks and characteristics, takes their score-weighted long/short returns - over all stocks and
within each market-cap quintile - and regresses them on the factors. Only months before the refit are used.

Hedge ratios are estimated for every score and universe on the run's factors, and for all stocks also with size and
with size and market beta added (HEDGE_SPECS; size for every run, market beta for the characteristic files). They are saved in long form to
hedge_betas_rebuilt.csv, which analysis/factor_spanning.py uses for all its hedged-score analyses. The default score's
hedged long/short is also saved as hedged_score_nodes_by_period.csv.

Run backtest.py, backtest_report.py and hedge_analysis.py for the same run first, then from the project root:
    python analysis/hedge_score.py [<universe>] [--region GL] [--variant TAGS]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, for src/ and backtest.py

import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

from analysis.backtest_report import REGION, UNIVERSE, VARIANT, LABEL, RET_NAME, make_periods, size_buckets
from analysis.factor_spanning import HEDGE_SPECS, extra_factors
from analysis.hedge_analysis import SCORE, build_positions, score_weights, hedge_score, score_stats
from build_trees import prepare_data
from src.constants import Columns, DataPaths, TREE_SETUPS, factor_chars, parse_variant
from src.preprocessing import read_backtest_data
from src.score_hedging import HEDGE_WINDOW, enough_months, rebuilt_history
from src.utils import build_comb

SCORES = ['size_oriented_score', 'final_score_norm', 'sdf_weight']


def refit_betas(refit, node_betas, combo_wei, comb_by_date, ret_by_date, bucket_by_date, factor_sets, dates):
    """Hedge ratios at one refit for every score, universe and specification (long rows), and the fit R2s."""
    months = [m for m in dates if m < refit and m in factor_sets['factors'].index][-HEDGE_WINDOW:]
    if not enough_months(len(months), factor_sets['factors'].shape[1]):
        return [], []
    hist = rebuilt_history(comb_by_date, ret_by_date, node_betas, combo_wei, months, SCORES, bucket_by_date)
    rows, fits = [], []
    for (score, universe), y in hist.items():
        for spec, facs in factor_sets.items():
            if universe != 'all' and spec != 'factors':
                continue
            both = pd.concat([y.rename('y'), facs], axis=1, join='inner').dropna()
            if not enough_months(len(both), facs.shape[1]):
                continue
            fit = sm.OLS(both['y'], sm.add_constant(both.drop(columns='y'))).fit()
            rows += [{'refit': refit, 'score': score, 'universe': universe, 'spec': spec, 'factor': f, 'beta': v}
                     for f, v in fit.params.drop('const').items()]
            fits.append({'refit': refit, 'score': score, 'universe': universe, 'spec': spec, 'R2': fit.rsquared,
                         'months': len(both)})
    return rows, fits


# %%
if __name__ == '__main__':
    pd.set_option('display.width', 250)
    paths = DataPaths()
    equal_weighted, tree_tag, prune_tag = parse_variant(VARIANT)
    tree_chars = TREE_SETUPS[tree_tag]['chars']
    out_dir = paths.result_file('report', REGION, UNIVERSE, VARIANT, ext=None)
    factors = pd.read_csv(out_dir / 'factor_returns.csv', index_col=0)

    # the stock data the backtest scores stocks on, from the first month
    features = factor_chars(tree_tag)
    data = read_backtest_data(list(dict.fromkeys(features + (tree_chars or []))), RET_NAME, region=REGION, universe=UNIVERSE)
    data_pl, ret_and_mcap = prepare_data(data, ret_name=RET_NAME, factors=None, equal_weighted=equal_weighted)
    comb_pl = build_comb(data=data_pl, merged_df=ret_and_mcap, features=(tree_chars or features) + ['lme'])
    comb_by_date = {(k[0] if isinstance(k, tuple) else k): v
                    for k, v in comb_pl.partition_by(Columns.date_col, as_dict=True).items()}
    by_date = data.swaplevel(0, 1).sort_index()
    ret_by_date = {d: g.droplevel('date') for d, g in by_date[RET_NAME].groupby(level='date')}
    bucket_by_date = {d: g.droplevel('date') for d, g in size_buckets(by_date['mkt_cap']).groupby(level='date', observed=True)}
    dates = sorted(comb_by_date)
    # build each index's lookup table now: pandas builds it lazily on first use, which is not thread-safe, and the
    # refits below run in threads (a half-built table can make an index look non-unique)
    for series in list(ret_by_date.values()) + list(bucket_by_date.values()):
        assert series.index.is_unique

    # the factor sets: the run's factors, and with size and (where the data has it) market beta added
    factor_sets = {'factors': factors}
    extra_rets, _, _ = extra_factors(REGION, UNIVERSE, VARIANT)
    for spec, added in HEDGE_SPECS.items():
        if added and all(a in extra_rets for a in added):
            factor_sets[spec] = pd.concat([factors, extra_rets[added]], axis=1, join='inner')

    # node weights of every refit: the SDF's, and the original score's if the backtest saved them
    keys = ['combination', 'port', 'node']
    nb = pd.read_csv(paths.result_file('node_betas', REGION, UNIVERSE, VARIANT))
    by_refit = {r: g.set_index(keys)['beta'] for r, g in nb.groupby('refit_date')}
    combo_file = paths.result_file('combo_weights', REGION, UNIVERSE, VARIANT)
    if combo_file.exists():
        combo = {r: g.set_index(keys)['weight'] for r, g in pd.read_csv(combo_file).groupby('refit_date')}
    elif prune_tag is None:
        # without validated pruning the original score's weights are the SDF's: prune() and calc_sharpe() both take
        # the model with the best Sharpe ratio on the training window's last months (checked on a rerun backtest)
        combo = by_refit
    else:
        combo = {}
        print(f'no {combo_file.name} (rerun backtest.py to save it): the original score is left out')

    start = time.time()
    results = Parallel(n_jobs=4, prefer='threads', verbose=5)(
        delayed(refit_betas)(r, b, combo.get(r), comb_by_date, ret_by_date, bucket_by_date, factor_sets, dates)
        for r, b in by_refit.items())
    betas = pd.DataFrame([row for rows, _ in results for row in rows])
    fits = pd.DataFrame([f for _, fs in results for f in fs])
    betas.to_csv(out_dir / 'hedge_betas_rebuilt.csv', index=False)
    fits.to_csv(out_dir / 'hedge_betas_rebuilt_fit.csv', index=False)
    print(f'Hedge ratios for {betas["refit"].nunique()} refits in {time.time() - start:.0f}s; mean R2 of the rebuilt '
          'histories on the factors:')
    print(fits.groupby(['score', 'spec'])['R2'].mean().round(2).to_string())

    # the default score hedged on the run's factors with these ratios (the comparison deck's hedged score)
    h = betas[(betas['score'] == SCORE) & (betas['universe'] == 'all') & (betas['spec'] == 'factors')]
    h = h.pivot(index='refit', columns='factor', values='beta')
    p = build_positions(REGION, UNIVERSE, VARIANT)
    w, ls = score_weights(p)
    hedged, w_hedged = hedge_score(p, w, ls, h)
    periods = make_periods(sorted(hedged.index))
    in_hedge = w.index.get_level_values('date').isin(hedged.index)
    table = pd.concat({
        f'{SCORE} long/short': score_stats(ls.loc[hedged.index], w[in_hedge], p['factors'], periods),
        f'{SCORE} long/short, hedged on rebuilt history': score_stats(hedged, w_hedged, p['factors'], periods),
    }, names=['portfolio', 'period'])
    h.to_csv(out_dir / 'hedged_score_nodes_ratios_by_refit.csv')
    table.to_csv(out_dir / 'hedged_score_nodes_by_period.csv')
    cols = ['Sharpe', 'alpha t (NW)', 'factor R2', 'gross exposure', 'one-way turnover %/mo', 'break-even cost bps',
            'Sharpe after 10bps']
    print(f'\n==== {LABEL}: {SCORE} long/short, hedged on its rebuilt history (from {hedged.index.min()}) ====')
    print(table[cols].round(2).to_string())
    print(f'\nSaved to {out_dir}')
