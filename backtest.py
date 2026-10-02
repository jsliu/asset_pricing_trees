# %%
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import norm

from prune_trees import prune, to_pandas, factor_betas, residualize_portfolios
from build_trees import prepare_data
from src.tree_scores import MIN_NODE_SIZE, calc_sharpe, score_stocks    # noqa: F401 (MIN_NODE_SIZE: make_deck.py)
from src.constants import DataPaths, Columns, Years, TREE_SETUPS, PRUNE_SETUPS, run_variant, factor_chars
from src.functions import calc_fac_ret
from src.preprocessing import read_backtest_data
from src.score_hedging import HEDGE_MIN, HEDGE_WINDOW, hedge_ratios, neutral_scores, rebuilt_history
from src.utils import build_comb


def rank_normalize(x):
    n = len(x)

    # equivalent to rank("min")
    u = (x.rank(method="min") - 0.5) / n

    return norm.ppf(u)

def r_squared(X, y):
    X = sm.add_constant(X)
    res = sm.OLS(y, X).fit()
    return res.rsquared_adj


def _select_portfolios(train_val_portfolio, prune_kwargs=None):
    """Prune one tree and return the column positions of its selected portfolios."""
    best_model, overall_model = prune(train_val_portfolio, n_jobs=1, **(prune_kwargs or {}))
    return np.nonzero(overall_model.betas[best_model, :])[0]

def _residual_returns(tree, rows):
    """Residual returns of the tree's `rows`, using the factor betas estimated at its last refit."""
    return residualize_portfolios(tree["df"].iloc[rows], tree["factors"].iloc[rows], betas=tree["betas"])

def _market_adjusted_returns(trees, d, columns):
    """Date d returns of the given portfolios before factor residualisation (stock returns net of the market)."""
    ret = pd.concat([tree["df"].iloc[slice(*tree["date_bounds"][d])] for tree in trees], axis=1)
    return ret.loc[:, ~ret.columns.duplicated()].reindex(columns=columns, fill_value=0)



SCORE = 'size_oriented_score'      # the score the residual and hedged versions are made of


def _with_neutral_scores(final_df, chars, h):
    """final_df (one date's stock scores) with the residual and, given hedge ratios h, the hedged score added
    (src.score_hedging.neutral_scores); stocks only in the factor portfolios join with the other scores empty."""
    s = final_df.select(Columns.id_col, SCORE).to_pandas().set_index(Columns.id_col)[SCORE].dropna()
    extra = neutral_scores(s, chars, h).rename_axis(Columns.id_col).reset_index()
    extra = pl.from_pandas(extra).with_columns(pl.col(Columns.id_col).cast(final_df.schema[Columns.id_col]))
    return final_df.join(extra, on=Columns.id_col, how='full', coalesce=True)


def _refit_period(d, refit_freq):
    """Period that date d (YYYYMMDD int) falls in; the models are refitted whenever it changes."""
    year, month = d // 10000, d // 100 % 100
    if refit_freq == 'Y':
        return year
    if refit_freq == 'Q':
        return year, (month - 1) // 3
    return d


def run_backtest(region=None, universe=None, ret_name='gross_returns', start_year=None, refit_freq='Y',
                 equal_weighted=True, tree_tag=None, prune_tag=None, neutral=True):
    """
    Run the tree backtest for a region of the company data (e.g. 'GL') or a universe of the characteristic files
    (e.g. 'largecap'); saves scores/returns to paths.output and returns (rets, stk_score).

    rets has two columns, both the SDF (pruned weights) applied to date d's portfolio returns:
    'Return' uses the factor-residualised returns the model is fitted on, 'Return_mkt_adj' the
    market-adjusted returns before residualisation.

    stk_score has the stock scores of score_stocks() per date for every stock in the universe:
    'final_score'/'norm_score' (original), 'sdf_weight' (sum(sdf_weight x market-adjusted return)
    reproduces 'Return_mkt_adj'), 'size_oriented_score' (SDF node weights tilted towards the stocks most
    firmly inside each node) and 'size_oriented_norm' (size_oriented_score rank-normalised to combine
    with factor scores).
    With neutral=True it also has the score with the existing factors taken out (src/score_hedging.py):
    'size_oriented_resid' - the residual of a cross-sectional regression of size_oriented_score on the factor scores
    (each factor characteristic rank-normalised that month) - and 'size_oriented_hedged' - the score's long/short
    positions minus their factor exposure, with hedge ratios estimated at each refit from the model's score rebuilt
    over the previous HEDGE_WINDOW months - each with a rank-normalised version ('..._norm'). The hedged positions
    include the factor portfolios' stocks, so a date's rows are the scored stocks plus those; the hedge ratios of every
    refit are saved as the 'score_hedge_ratios' result file.

    The SDF node weights (beta) of every refit are saved as the 'node_betas' result file (DataPaths.result_file), and
    the node weights the original score uses (from calc_sharpe) as the 'combo_weights' result file, so that the scores
    can be recomputed for any month with a refit's model (analysis/hedge_score.py).

    equal_weighted: equal- or value-weighted market adjustment and node weights. Like the tree files, the outputs of
    a value-weighted run end in '_vw' (e.g. result/ret_largecap_vw.csv) and it reads the '_vw' trees
    (build_trees.py with EQUAL_WEIGHTED = False).

    tree_tag: a tree set-up in TREE_SETUPS (characteristics and depth of the trees, built by build_trees.py with the
    same TREE_TAG); prune_tag: a pruning set-up in PRUNE_SETUPS. Both are added to the output file names,
    e.g. result/ret_largecap_slow3_val.csv.

    refit_freq: 'Y' (yearly), 'Q' (quarterly) or 'M' (every date) - how often the tree and final models are
    re-estimated. Between refits the last fitted models are applied to each new date.
    """
    if refit_freq not in ('Y', 'Q', 'M'):
        raise ValueError(f"refit_freq must be 'Y', 'Q' or 'M', got {refit_freq!r}")
    years = Years()
    paths = DataPaths()
    start_year = years.min_year if start_year is None else start_year

    variant = run_variant(equal_weighted, tree_tag, prune_tag)
    label = paths.label(region, universe, variant)
    tree_chars = TREE_SETUPS[tree_tag]['chars']
    prune_kwargs = PRUNE_SETUPS[prune_tag]
    print(f"Loading base characteristics in {label}")

    features = factor_chars(tree_tag)                  # the factor portfolios' characteristics
    # the factor characteristics, and those of the trees when they are others
    data = read_backtest_data(list(dict.fromkeys(features + (tree_chars or []))), ret_name, region=region, universe=universe)
    # data = read_db_data(region_=reg, features=features, ret_name=Columns.returns_col, data_saved=data_saved)
    # data = read_big_universe(ret_name=ret_name, features=features)
    factor_returns = calc_fac_ret(data[features], data[ret_name], date_col="date", score_weighted=True)

    data_pl, ret_and_mcap = prepare_data(data, ret_name=ret_name, factors=None, equal_weighted=equal_weighted)

    comb_pl = build_comb(
        data=data_pl,
        merged_df=ret_and_mcap,
        features=(tree_chars or features) + ['lme'],     # the characteristics the trees were built from
    )

    # -----------------------------
    # 1. Load all tree files once
    # ----------------------------- 
    tree_data = {}
    all_dates = set()

    for tree_file_path in paths.tree_files(region, universe, run_variant(equal_weighted, tree_tag)):

        # raw portfolio returns; they are residualised on the factors at each refit using training rows only
        tree_pd = to_pandas(tree_file_path)
        dates_idx = tree_pd.index.get_level_values(Columns.date_col)

        # --- Stable-sort rows by date so each date is a contiguous block ---
        # train window = rows [0, start) (dates before d), test = rows [start, end) (date d)
        order = np.argsort(dates_idx.to_numpy(), kind="stable")
        df = tree_pd.iloc[order]
        sorted_dates, starts, counts = np.unique(dates_idx.to_numpy()[order], return_index=True, return_counts=True)
        date_bounds = {d: (s, s + c) for d, s, c in zip(sorted_dates.tolist(), starts, counts)}

        # store
        tree_data[tree_file_path.name] = {
            "df": df,
            "factors": factor_returns.loc[dates_idx.to_numpy()[order]],
            "date_bounds": date_bounds,
            "betas": None,
        }

        all_dates.update(sorted_dates)

    # Sort dates once
    dates = sorted(all_dates)

    # -----------------------------
    # 2. Backtest loop
    # -----------------------------
    rets = pd.DataFrame(index=dates, columns=['Return', 'Return_mkt_adj'], dtype=float)
    stock_scores = []
    refit_betas, refit_combo = [], []

    # stock-level data split by date once, so node lookups only touch one cross-section
    comb_by_date = {
        (k[0] if isinstance(k, tuple) else k): v
        for k, v in comb_pl.partition_by(Columns.date_col, as_dict=True).items()
    }

    # stock data by date: the factor characteristics for the residual and hedged scores, the returns for the
    # rebuilt score histories the hedge ratios come from
    if neutral:
        data_by_date = {dt: g.droplevel('date') for dt, g in data.swaplevel(0, 1).sort_index().groupby(level='date')}
        ret_by_date = {dt: g[ret_name] for dt, g in data_by_date.items()}
        for series in ret_by_date.values():                   # build the index lookups once
            assert series.index.is_unique
        score_dates = sorted(comb_by_date)
    hedge, refit_hedge = None, []

    # one worker pool reused across dates; each worker runs its grid search single-threaded
    parallel = Parallel(n_jobs=-1)
    fitted_period = None

    for d in tqdm(dates, desc=f"{label} Backtest", unit=Columns.date_col):

        if d // 10000 < start_year:
        # if d < 20070330:
            continue

        # print(d)
        trees_d = [obj for obj in tree_data.values() if d in obj["date_bounds"]]
        if not trees_d:
            continue

        if _refit_period(d, refit_freq) != fitted_period:
            # fit only on dates before d, so date d stays out of sample
            train_list, test_list = [], []
            for obj in trees_d:
                start, end = obj["date_bounds"][d]
                obj["betas"] = factor_betas(obj["df"].iloc[:start], obj["factors"].iloc[:start])
                train_list.append(_residual_returns(obj, slice(0, start)))
                test_list.append(_residual_returns(obj, slice(start, end)))

            # prune each tree in parallel and keep only its selected portfolios
            selected = parallel(delayed(_select_portfolios)(train, prune_kwargs) for train in train_list)
            train_list = [train.iloc[:, idx] for train, idx in zip(train_list, selected)]
            test_list = [test.iloc[:, idx] for test, idx in zip(test_list, selected)]

            # concat once
            train_portfolios = pd.concat(train_list, axis=1)
            test_portfolios = pd.concat(test_list, axis=1)

            # deduplication
            all_test_portfolios = test_portfolios.loc[
                :, ~(test_portfolios.T.duplicated() | test_portfolios.columns.duplicated())
            ]

            all_train_portfolios = train_portfolios.loc[
                :, ~train_portfolios.columns.duplicated()
            ]

            all_train_portfolios = all_train_portfolios[all_test_portfolios.columns]

            # final model
            final_best_model, final_model = prune(all_train_portfolios, **prune_kwargs)

            final_sharpes, final_combo_wei, final_port = calc_sharpe(
                all_train_portfolios, final_model
            )
            fitted_period = _refit_period(d, refit_freq)
            refit_betas.append(
                pd.Series(final_model.betas[final_best_model], index=final_model.feature_weights.index, name='beta')
                .loc[lambda x: x != 0].reset_index().assign(refit_date=d)
            )
            refit_combo.append(final_combo_wei.rename('weight').loc[lambda x: x != 0].reset_index().assign(refit_date=d))
            if neutral:
                # hedge ratios of the score this model gives, from its rebuilt history over the months before d
                months = [m for m in score_dates if m < d][-HEDGE_WINDOW:]
                refit_nodes = pd.Series(final_model.betas[final_best_model], index=final_model.feature_weights.index)
                history = rebuilt_history(comb_by_date, ret_by_date, refit_nodes, None, months, [SCORE])
                hedge = hedge_ratios(history[(SCORE, 'all')], factor_returns, HEDGE_MIN) if len(history) else None
                if hedge is not None:
                    refit_hedge.append(hedge.rename(d))
        else:
            # reuse the last fit: take this date's returns of the portfolios the final model was fitted on
            test_portfolios = pd.concat(
                [_residual_returns(obj, slice(*obj["date_bounds"][d])) for obj in trees_d if obj["betas"] is not None],
                axis=1,
            )
            all_test_portfolios = test_portfolios.loc[:, ~test_portfolios.columns.duplicated()].reindex(
                columns=final_model.feature_weights.index, fill_value=0
            )

        node_betas = pd.Series(final_model.betas[final_best_model], index=final_model.feature_weights.index)
        comb_d = comb_by_date.get(d)
        final_df = score_stocks(comb_d, node_betas, final_combo_wei) if comb_d is not None else pl.DataFrame()
        if neutral and comb_d is not None and d in data_by_date:
            final_df = _with_neutral_scores(final_df, data_by_date[d][features], hedge)

        final_df = final_df.with_columns(
            pl.lit(d).alias(Columns.date_col)
        )

        stock_scores.append(final_df)
        sdf = final_model.predict(all_test_portfolios)
        rets.loc[d, 'Return'] = sdf[final_best_model].iloc[0]
        mkt_adj_sdf = final_model.predict(_market_adjusted_returns(trees_d, d, final_model.feature_weights.index))
        rets.loc[d, 'Return_mkt_adj'] = mkt_adj_sdf[final_best_model].iloc[0]



    stk_score = pl.concat(stock_scores, how='diagonal_relaxed')      # dates before the first hedge lack its columns
    stk_score.write_csv(paths.result_file('score', region, universe, variant))
    pd.concat(refit_betas).to_csv(paths.result_file('node_betas', region, universe, variant), index=False)
    pd.concat(refit_combo).to_csv(paths.result_file('combo_weights', region, universe, variant), index=False)
    if refit_hedge:
        pd.DataFrame(refit_hedge).rename_axis('refit_date').to_csv(
            paths.result_file('score_hedge_ratios', region, universe, variant))

    rets = rets.dropna()
    rets.cumsum().plot()
    plt.title(f"{label} cumulative returns")
    plt.show()

    rets.to_csv(paths.result_file('ret', region, universe, variant))

    return rets, stk_score


# %%
# Running backtest
if __name__ == '__main__':
    sns.set_theme()
    # returns column: 'gross_returns' in the company data, 'ret' in the characteristic files (set per run below)
    # company data: regions = ['GL', 'US', ...] with universes = [None]
    # characteristic files: regions = [None] with universes = ['full', 'largecap', 'largecap001']
    regions = [None]
    universes = ['largecap', ]
    EQUAL_WEIGHTED = True      # False: reads the '_vw' trees and names its outputs with '_vw', e.g. ret_largecap_vw.csv
    TREE_TAG = 'slow4'         # tree set-up in TREE_SETUPS, e.g. 'slow3' (trees built with the same TREE_TAG)
    PRUNE_TAG = 'val'          # pruning set-up in PRUNE_SETUPS, e.g. 'val' (validated pruning on a rolling window)
    for region, universe in product(regions, universes):
        ret_name = 'gross_returns' if universe is None else 'ret'
        run_backtest(region, universe, ret_name=ret_name, start_year=1980, refit_freq='Y',
                     equal_weighted=EQUAL_WEIGHTED, tree_tag=TREE_TAG, prune_tag=PRUNE_TAG)

# %%
