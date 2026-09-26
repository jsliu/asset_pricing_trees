# %%
import pandas as pd
import polars as pl
import numpy as np
import pickle
import logging
import warnings

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.exceptions import ConvergenceWarning
from src.preprocessing import cap_weight, read_ei_data, read_db_data, read_big_universe, read_all_data
from src.functions import calc_fac_ret
from src.constants import Chars, DataPaths, Parameters, Columns
from src.model import TreeElastic
from tqdm import tqdm

import numpy as np
import pandas as pd

def factor_betas(tree_ret, factor_ret):
    """
    Estimate the factor model (intercept + factors) for many portfolio returns.

    Parameters
    ----------
    tree_ret : pd.DataFrame
        T x N node portfolio returns
    factor_ret : pd.DataFrame
        T x K factor returns

    Returns
    -------
    np.ndarray
        (K+1) x N coefficients, intercept in the first row
    """

    # align dates
    common_idx = tree_ret.index.intersection(factor_ret.index)
    Y = tree_ret.loc[common_idx].to_numpy()
    F = factor_ret.loc[common_idx].to_numpy()

    # add intercept
    X = np.column_stack([np.ones(len(F)), F])  # T x (K+1)

    # solve all regressions simultaneously
    return np.linalg.lstsq(X, Y, rcond=None)[0]   # (K+1) x N


def residualize_portfolios(tree_ret, factor_ret, betas=None):
    """
    Residualize many portfolio returns against a factor model: returns minus the factor part B·F.
    The intercept is used when estimating B but not subtracted, so the residuals keep each
    portfolio's alpha (the average return not explained by the factors).

    Parameters
    ----------
    tree_ret : pd.DataFrame
        T x N node portfolio returns
    factor_ret : pd.DataFrame
        T x K factor returns
    betas : np.ndarray, optional
        (K+1) x N coefficients from factor_betas(). Pass betas estimated on past data to
        residualize later dates without look-ahead. If None, they are estimated on these same
        dates (in-sample).

    Returns
    -------
    pd.DataFrame
        Residual returns (alpha + noise), same shape as tree_ret
    """
    if betas is None:
        betas = factor_betas(tree_ret, factor_ret)

    # align dates
    common_idx = tree_ret.index.intersection(factor_ret.index)
    Y = tree_ret.loc[common_idx].to_numpy()
    F = factor_ret.loc[common_idx].to_numpy()

    # factor part only; betas[0] is the intercept (alpha), which stays in the residual
    resid = Y - F @ betas[1:]

    return pd.DataFrame(
        resid,
        index=common_idx,
        columns=tree_ret.columns,
    )


def to_pandas(tree_file_path, missing_rate=0.1):
    logging.info(f"Loading {str(tree_file_path)}")
    tree_portfolio = pl.read_parquet(tree_file_path)

    logging.info("Selecting only returns")
    tree_portfolio_pd = (
        tree_portfolio
        .pivot(
            values=Columns.w_returns_col,
            index=Columns.date_col,
            on=[Columns.comb_col, Columns.port_col, Columns.node_col],
            aggregate_function="first",
        )
        .sort(Columns.date_col)
    ).to_pandas().set_index(Columns.date_col)

    tree_portfolio_pd.columns = [
        c.replace("{", "").replace("}", "").replace('"', "")
        for c in tree_portfolio_pd.columns
    ]

    tree_portfolio_pd.columns = pd.MultiIndex.from_tuples(
        [
            tuple(c.split(",")) for c in tree_portfolio_pd.columns
        ],
        names=[Columns.comb_col, Columns.port_col, Columns.node_col],
    )

    tp = tree_portfolio_pd.loc[:, ~tree_portfolio_pd.T.duplicated()]
    valid_col = tp.isna().sum() / tp.shape[0] < missing_rate
    return tp.loc[:, tp.columns[valid_col]].fillna(0)


def prune(tree_returns, n_jobs=-1):
    logging.info('Splitting data')

    train_val_portfolios, test_portfolios = train_test_split(
        tree_returns, 
        test_size=Parameters.test_size, 
        shuffle=False
        )

    test_portfolios = test_portfolios.dropna(axis=1, how="all").fillna(0)
    # valid_rows = train_val_portfolios[test_portfolios.columns].isna().sum(axis=1) / train_val_portfolios.shape[1] < 0.1
    # train_val_portfolios = train_val_portfolios[test_portfolios.columns][valid_rows].fillna(0)
    train_val_portfolios = train_val_portfolios[test_portfolios.columns].fillna(0)

    param_grid = {
        'mean_shrinkage': Parameters.mean_shrinkage,
        'ridge_lambda': Parameters.ridge_lambda
    }
    tscv = TimeSeriesSplit(n_splits=Parameters.cv_splits)
    tree_model = TreeElastic(k_min=Parameters.k_min, k_max=Parameters.k_max)
    cv_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, verbose=0, cv=tscv, n_jobs=n_jobs)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        cv_search.fit(train_val_portfolios)

    # overall_model.fit(train_val_portfolios)
    overall_model = cv_search.best_estimator_
    sdf = overall_model.predict(test_portfolios)
    sharpe_ratio = sdf.mean() / (sdf.std() + 1e-20)
    best_models = np.argmax(sharpe_ratio)
    return best_models, overall_model

# %%
if __name__ == '__main__':
    chars = Chars()
    paths = DataPaths()
    ret_name = "gross_returns"
    # ret_name = "Universe Returns"
    # regions = ['UK', 'EU', 'AP', 'JP', 'EM']
    regions = ['US', ]
    for reg in regions:
        print(f"Read EI factors in {reg}")
        features = list(chars.__dict__.values())[:-2]
        if reg == "ALL":
            data, CHARAS_LIST, _ = read_all_data(target=ret_name, ei_factors=features)
        else:
            data, _, CHARAS_LIST, _, _ = read_ei_data(region_=reg, target=ret_name, ei_factors=features)
        # data = read_big_universe(ret_name=ret_name, features=features, features_direction=[1, -1, -1, 1, 1, -1, -1, -1, 1, -1])
        factor_returns = calc_fac_ret(data[features], data[ret_name], date_col="date", score_weighted=True)

        print(f"Pruning tree in {reg}")
        files = list(paths.processed_data.iterdir())
        for tree_file_path in tqdm(files):
            if tree_file_path.suffix != '.parquet':
                continue
            if reg not in tree_file_path.name:
                continue

            node_returns = to_pandas(tree_file_path)
            node_resid = residualize_portfolios(tree_ret=node_returns, factor_ret=factor_returns)
            _, overall_model = prune(node_resid)

            model_output_name = f"{tree_file_path.with_suffix('').name}{paths.sep}{paths.model_suffix}"
            with open(paths.model_dumps / model_output_name, 'wb') as f:
                pickle.dump(overall_model, f)


# %%
