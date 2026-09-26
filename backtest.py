# %%
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import norm
from AlphaWorkshop.src.main.python.alphaworkshop.functions import rank_normalise

from plot_test_sr import calc_sharpe
from prune_trees import prune, to_pandas, factor_betas, residualize_portfolios
from build_trees_pl import prepare_data
from stock_portfolio_pl import get_stocks_in_node, compute_node_scores
from src.constants import DataPaths, Parameters, Columns, Chars, Years
from src.functions import calc_fac_ret
from src.preprocessing import read_ei_data, read_all_data, read_char_data
from src.utils import build_comb_pl


def rank_normalize(x):
    n = len(x)

    # equivalent to rank("min")
    u = (x.rank(method="min") - 0.5) / n

    return norm.ppf(u)

def r_squared(X, y):
    X = sm.add_constant(X)
    res = sm.OLS(y, X).fit()
    return res.rsquared_adj

# tree_portfolio's min_node_size (src/utils.py): smaller nodes get no portfolio return
MIN_NODE_SIZE = 50

def _select_portfolios(train_val_portfolio):
    """Prune one tree and return the column positions of its selected portfolios."""
    best_model, overall_model = prune(train_val_portfolio, n_jobs=1)
    return np.nonzero(overall_model.betas[best_model, :])[0]

def _residual_returns(tree, rows):
    """Residual returns of the tree's `rows`, using the factor betas estimated at its last refit."""
    return residualize_portfolios(tree["df"].iloc[rows], tree["factors"].iloc[rows], betas=tree["betas"])

def _market_adjusted_returns(trees, d, columns):
    """Date d returns of the given portfolios before factor residualisation (stock returns net of the market)."""
    ret = pd.concat([tree["df"].iloc[slice(*tree["date_bounds"][d])] for tree in trees], axis=1)
    return ret.loc[:, ~ret.columns.duplicated()].reindex(columns=columns, fill_value=0)

def score_stocks(comb_d, node_betas, combo_wei=None):
    """
    Stock scores on one date from the nodes held by the SDF (node_betas) and, if given, by the
    original score (combo_wei); both are Series indexed by (tree_key, port_col, node_id).

    final_score/norm_score: original score, node weight x geometric mean of market-wide characteristic ranks.
    sdf_weight: the stock's weight in the SDF, sum over nodes of beta x the stock's weight in the node.
    size_oriented_score: the same node weights beta, spread inside each node in proportion to
        (weight in node x oriented_score), so each node still totals beta but the stocks sitting most
        firmly inside it (by characteristic ranks inside the parent node) get more of it.
    size_oriented_norm: size_oriented_score rank-normalised like the factor scores (see _normalise_scores).

    Every stock of the date's universe (comb_d) is returned; sdf_weight and size_oriented_score are 0 for
    stocks in no held node, final_score/norm_score are null for stocks the original score does not cover.
    """
    combo_wei = pd.Series(dtype=float) if combo_wei is None else combo_wei
    node_keys = list(dict.fromkeys(
        [k for k, w in combo_wei.items() if w != 0] + [k for k, b in node_betas.items() if b != 0]
    ))

    dfs = []
    # tree splits are computed within each date, so growing the tree on one date alone gives the same nodes
    for tree_key, port_col, node_id in node_keys:
        key = (tree_key, port_col, node_id)

        df_node = get_stocks_in_node(
            comb_d,
            tree_key,
            port_col,
            int(node_id),
            Parameters.n_splits
        )

        if df_node.is_empty():
            continue

        df_scores = compute_node_scores(
            df_node=df_node,
            tree_key=tree_key,
            port_col=port_col,
            node_id=int(node_id),
            n_split=Parameters.n_splits,
        )

        w = combo_wei.get(key, 0.0)
        # the node's tree portfolio has no return below MIN_NODE_SIZE stocks, so the SDF holds nothing there
        b_held = node_betas.get(key, 0.0) if df_node.height >= MIN_NODE_SIZE else 0.0
        in_node = pl.col(Columns.size_col) / pl.col(Columns.size_col).sum()
        tilted = in_node * pl.col("oriented_score")

        dfs.append(
            df_scores.select(
                Columns.id_col,
                pl.lit(w != 0).alias("in_old_score"),
                (pl.col("factor_score") * w).alias("weighted_score"),
                (in_node * b_held).alias("sdf_weight"),
                (tilted / tilted.sum() * b_held).alias("size_oriented_score"),
            )
        )

    final_df = comb_d.select(Columns.id_col)
    if not dfs:
        return _normalise_scores(final_df.with_columns(
            pl.lit(None, pl.Float64).alias("final_score"), pl.lit(None, pl.Float64).alias("norm_score"),
            pl.lit(0.0).alias("sdf_weight"), pl.lit(0.0).alias("size_oriented_score"),
        ))

    node_rows = pl.concat(dfs)
    scores = node_rows.group_by(Columns.id_col).agg(pl.col("sdf_weight").sum(), pl.col("size_oriented_score").sum())

    old = (
        node_rows
        .filter(pl.col("in_old_score"))
        .group_by(Columns.id_col)
        .agg(
            pl.col("weighted_score").sum().alias("final_score")
        )
    )
    n = old.height
    u = ((old["final_score"].rank("min") - 0.5) / n).to_numpy()
    old = old.with_columns(pl.Series("norm_score", norm.ppf(u), dtype=pl.Float64))

    final_df = (
        final_df
        .join(old, on=Columns.id_col, how="left")
        .join(scores, on=Columns.id_col, how="left")
        .with_columns(pl.col("sdf_weight").fill_null(0.0), pl.col("size_oriented_score").fill_null(0.0))
    )
    return _normalise_scores(final_df)

def _normalise_scores(scores):
    """
    size_oriented_norm: size_oriented_score across the whole universe (0 = not held, i.e. neutral),
    rank-normalised with the same rank_normalise(cutoff_std=3.5) as the factor scores.
    """
    z = rank_normalise(scores["size_oriented_score"].to_pandas(), cutoff_std=3.5)
    return scores.with_columns(pl.Series("size_oriented_norm", np.asarray(z, dtype=float)))

def _refit_period(d, refit_freq):
    """Period that date d (YYYYMMDD int) falls in; the models are refitted whenever it changes."""
    year, month = d // 10000, d // 100 % 100
    if refit_freq == 'Y':
        return year
    if refit_freq == 'Q':
        return year, (month - 1) // 3
    return d


def run_backtest(reg, ret_name='gross_returns', suffix='std', start_year=None, refit_freq='Y'):
    """
    Run the tree backtest for one region; saves scores/returns to paths.output and returns (rets, stk_score).

    rets has two columns, both the SDF (pruned weights) applied to date d's portfolio returns:
    'Return' uses the factor-residualised returns the model is fitted on, 'Return_mkt_adj' the
    market-adjusted returns before residualisation.

    stk_score has the stock scores of score_stocks() per date for every stock in the universe:
    'final_score'/'norm_score' (original), 'sdf_weight' (sum(sdf_weight x market-adjusted return)
    reproduces 'Return_mkt_adj'), 'size_oriented_score' (SDF node weights tilted towards the stocks most
    firmly inside each node) and 'size_oriented_norm' (size_oriented_score rank-normalised to combine
    with factor scores).

    The SDF node weights (beta) of every refit are saved to {reg}_node_betas_{suffix}.csv.

    refit_freq: 'Y' (yearly), 'Q' (quarterly) or 'M' (every date) - how often the tree and final models are
    re-estimated. Between refits the last fitted models are applied to each new date.
    """
    if refit_freq not in ('Y', 'Q', 'M'):
        raise ValueError(f"refit_freq must be 'Y', 'Q' or 'M', got {refit_freq!r}")
    chars = Chars()
    years = Years()
    paths = DataPaths()
    start_year = years.min_year if start_year is None else start_year

    print(f"Loading base characteristics in {reg}")

    features = list(chars.__dict__.values())[:-2]
    if reg == "ALL":
        data, CHARAS_LIST, _ = read_all_data(target=ret_name, ei_factors=features)
    elif reg in ("full", "largecap", "largecap001"):
        data = read_char_data(features, universe=None if reg == "full" else reg, ret_name=ret_name)
    else:
        data, _, CHARAS_LIST, _, _ = read_ei_data(region_=reg, target=ret_name, ei_factors=features)
    # data = read_db_data(region_=reg, features=features, ret_name=Columns.returns_col, data_saved=data_saved)
    # data = read_big_universe(ret_name=ret_name, features=features)
    factor_returns = calc_fac_ret(data[features], data[ret_name], date_col="date", score_weighted=True)

    data_pl, ret_and_mcap = prepare_data(data, ret_name=ret_name, factors=None, equal_weighted=True)

    comb_pl = build_comb_pl(
        data=data_pl,
        merged_df=ret_and_mcap,
        features=features+['lme'],
    )

    # -----------------------------
    # 1. Load all tree files once
    # ----------------------------- 
    tree_data = {}
    all_dates = set()

    for tree_file_path in paths.processed_data.iterdir():
        if tree_file_path.suffix != '.parquet':
            continue
        if reg not in tree_file_path.name:
            continue

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
    refit_betas = []

    # stock-level data split by date once, so node lookups only touch one cross-section
    comb_by_date = {
        (k[0] if isinstance(k, tuple) else k): v
        for k, v in comb_pl.partition_by(Columns.date_col, as_dict=True).items()
    }

    # one worker pool reused across dates; each worker runs its grid search single-threaded
    parallel = Parallel(n_jobs=-1)
    fitted_period = None

    for d in tqdm(dates, desc=f"{reg} Backtest", unit=Columns.date_col):

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
            selected = parallel(delayed(_select_portfolios)(train) for train in train_list)
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
            final_best_model, final_model = prune(all_train_portfolios)

            final_sharpes, final_combo_wei, final_port = calc_sharpe(
                all_train_portfolios, final_model
            )
            fitted_period = _refit_period(d, refit_freq)
            refit_betas.append(
                pd.Series(final_model.betas[final_best_model], index=final_model.feature_weights.index, name='beta')
                .loc[lambda x: x != 0].reset_index().assign(refit_date=d)
            )
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

        final_df = final_df.with_columns(
            pl.lit(d).alias(Columns.date_col)
        )

        stock_scores.append(final_df)
        sdf = final_model.predict(all_test_portfolios)
        rets.loc[d, 'Return'] = sdf[final_best_model].iloc[0]
        mkt_adj_sdf = final_model.predict(_market_adjusted_returns(trees_d, d, final_model.feature_weights.index))
        rets.loc[d, 'Return_mkt_adj'] = mkt_adj_sdf[final_best_model].iloc[0]



    stk_score = pl.concat(stock_scores)
    stk_score.write_csv(paths.output / f'{reg}_score_{suffix}.csv')
    pd.concat(refit_betas).to_csv(paths.output / f'{reg}_node_betas_{suffix}.csv', index=False)

    rets = rets.dropna()
    rets.cumsum().plot()
    plt.title(f"{reg} cumulative returns")
    plt.show()

    rets.to_csv(paths.output / f'{reg}_ret_{suffix}.csv')

    return rets, stk_score


# %%
# Running backtest
if __name__ == '__main__':
    sns.set_theme()
    suffix = 'std'
    # ret_name = 'Universe Returns'
    # ret_name = 'gross_returns'
    ret_name = 'ret'
    # regions = ['US', 'EU', 'UK', 'JP', 'AP', 'EM']
    # regions = ['GL', ]
    regions = ['full', ]
    for reg in regions:
        run_backtest(reg, ret_name=ret_name, suffix=suffix, start_year=1980, refit_freq='Y')

# %%
