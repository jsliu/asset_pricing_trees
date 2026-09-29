# %%
"""
The stock score's long/short hedged the way the SDF is: with the factor betas of the portfolio the current model
holds, over its training history.

At each refit the SDF's betas come from the tree portfolios' past returns, and those portfolios re-sort the stocks
into the nodes every month. This script does the same for the score (SCORE): with the node weights (beta) of the
refit, it recomputes the score in each of the previous HEDGE_WINDOW months (at least HEDGE_MIN) from that month's
stocks and characteristics (backtest.score_stocks), takes its score-weighted long/short return, and regresses those
returns on the factors. Only months before the refit are used. The hedged long/short, its positions, turnover and
costs are then computed as in hedge_analysis.py.

Run backtest.py, backtest_report.py and hedge_analysis.py for the same run first, then from the project root:
    python analysis/score_hedge.py [<universe>] [--region GL] [--vw | --variant TAGS]
Saves hedged_score_nodes_by_period.csv and hedged_score_nodes_ratios_by_refit.csv to the report folder.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, for src/ and backtest.py

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

from analysis.backtest_report import REGION, UNIVERSE, VARIANT, LABEL, RET_NAME, make_periods
from analysis.hedge_analysis import SCORE, build_positions, score_weights, hedge_score, score_stats
from backtest import score_stocks
from build_trees import prepare_data
from src.constants import Chars, Columns, DataPaths, TREE_SETUPS, parse_variant
from src.functions import _get_weights
from src.preprocessing import read_backtest_data
from src.utils import build_comb

HEDGE_WINDOW, HEDGE_MIN = 120, 60   # months of rebuilt score history the betas are estimated on


def rebuilt_long_short(comb_by_date, ret_by_date, node_betas, months):
    """Score-weighted long/short return in each of `months`, of the score the model with node_betas gives."""
    out = {}
    for m in months:
        comb_m = comb_by_date.get(m)
        if comb_m is None:
            continue
        s = score_stocks(comb_m, node_betas).select(Columns.id_col, SCORE).to_pandas().set_index(Columns.id_col)[SCORE]
        w = pd.Series(np.asarray(_get_weights(s, score_weighted=True), dtype=float), index=s.index).dropna()
        r = ret_by_date[m].reindex(w.index).dropna()
        w = w.loc[r.index]
        out[m] = float((w * (r - r.mean())).sum())
    return pd.Series(out)


def refit_betas(refit, node_betas, comb_by_date, ret_by_date, factors, dates):
    """Factor betas (regression with intercept) of the rebuilt score long/short over the months before refit."""
    months = [m for m in dates if m < refit and m in factors.index][-HEDGE_WINDOW:]
    if len(months) < HEDGE_MIN:
        return refit, None, None
    ls = rebuilt_long_short(comb_by_date, ret_by_date, node_betas, months)
    fit = sm.OLS(ls, sm.add_constant(factors.loc[ls.index])).fit()
    return refit, fit.params.drop('const'), fit.rsquared


# %%
if __name__ == '__main__':
    pd.set_option('display.width', 250)
    paths = DataPaths()
    equal_weighted, tree_tag, _ = parse_variant(VARIANT)
    tree_chars = TREE_SETUPS[tree_tag]['chars']
    out_dir = paths.result_file('report', REGION, UNIVERSE, VARIANT, ext=None)
    factors = pd.read_csv(out_dir / 'factor_returns.csv', index_col=0)

    # the stock data the backtest scores stocks on, from the first month
    features = list(Chars().__dict__.values())[:-2]
    data = read_backtest_data(list(dict.fromkeys(features + (tree_chars or []))), RET_NAME, region=REGION, universe=UNIVERSE)
    data_pl, ret_and_mcap = prepare_data(data, ret_name=RET_NAME, factors=None, equal_weighted=equal_weighted)
    comb_pl = build_comb(data=data_pl, merged_df=ret_and_mcap, features=(tree_chars or features) + ['lme'])
    comb_by_date = {(k[0] if isinstance(k, tuple) else k): v
                    for k, v in comb_pl.partition_by(Columns.date_col, as_dict=True).items()}
    ret = data[RET_NAME]
    ret_by_date = {d: g.droplevel('date') for d, g in ret.swaplevel(0, 1).groupby(level='date')}
    dates = sorted(comb_by_date)

    # node weights of every refit
    nb = pd.read_csv(paths.result_file('node_betas', REGION, UNIVERSE, VARIANT))
    by_refit = {r: g.set_index(['combination', 'port', 'node'])['beta'] for r, g in nb.groupby('refit_date')}

    start = time.time()
    results = Parallel(n_jobs=4, prefer='threads', verbose=5)(
        delayed(refit_betas)(r, b, comb_by_date, ret_by_date, factors, dates) for r, b in by_refit.items())
    h = pd.DataFrame({r: b for r, b, _ in results if b is not None}).T.sort_index()
    fit_r2 = pd.Series({r: r2 for r, _, r2 in results if r2 is not None}).sort_index()
    print(f'Betas for {len(h)} refits in {time.time() - start:.0f}s; R2 of the rebuilt history on the factors: '
          f'mean {fit_r2.mean():.2f}')

    # hedge with these betas, and compare with hedging on the score's own past returns (hedge_analysis.py)
    p = build_positions(REGION, UNIVERSE, VARIANT)
    w, ls = score_weights(p)
    hedged, w_hedged = hedge_score(p, w, ls, h)
    periods = make_periods(sorted(hedged.index))
    in_hedge = w.index.get_level_values('date').isin(hedged.index)
    table = pd.concat({
        f'{SCORE} long/short': score_stats(ls.loc[hedged.index], w[in_hedge], p['factors'], periods),
        f'{SCORE} long/short, hedged on rebuilt history': score_stats(hedged, w_hedged, p['factors'], periods),
    }, names=['portfolio', 'period'])
    h.assign(history_r2=fit_r2).to_csv(out_dir / 'hedged_score_nodes_ratios_by_refit.csv')
    table.to_csv(out_dir / 'hedged_score_nodes_by_period.csv')

    old = pd.read_csv(out_dir / 'hedged_score_by_period.csv').set_index(['portfolio', 'period'])
    cols = ['Sharpe', 'alpha t (NW)', 'factor R2', 'gross exposure', 'one-way turnover %/mo', 'break-even cost bps',
            'Sharpe after 10bps']
    print(f'\n==== {LABEL}: {SCORE} long/short, hedged on its rebuilt history (from {hedged.index.min()}) ====')
    print(table[cols].round(2).to_string())
    print('\nHedged on its own past returns (hedge_analysis.py, from 1983)')
    print(old.loc[f'{SCORE} long/short, hedged', cols].round(2).to_string())
    print(f'\nSaved to {out_dir}')
