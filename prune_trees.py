# %%
import pandas as pd
import polars as pl
import numpy as np
import pickle
import logging
import warnings

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.exceptions import ConvergenceWarning
from src.constants import Chars, DataPaths, Parameters, Columns
from src.model import TreeElastic
from tqdm import tqdm

def to_pandas(tree_file_path):
    logging.info(f"Loading {str(tree_file_path)}")
    tree_portfolio = pl.read_parquet(tree_file_path)

    logging.info("Selecting only returns")
    # ret_indexes = tree_portfolio.index.get_level_values(Columns.features_col) == Columns.w_returns_col
    # tree_portfolio = tree_portfolio[ret_indexes]
    
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
    return tree_portfolio_pd.loc[:, ~tree_portfolio_pd.T.duplicated()]



def prune(tree_portfolio):
    logging.info('Splitting data')
    train_val_portfolios, test_portfolios = train_test_split(
        tree_portfolio, 
        test_size=Parameters.test_size, 
        shuffle=False
        )

    test_portfolios = test_portfolios.dropna(axis=1, how="all").fillna(0)
    train_val_portfolios = train_val_portfolios[test_portfolios.columns].fillna(0)

    param_grid = {
        'mean_shrinkage': Parameters.mean_shrinkage,
        'ridge_lambda': Parameters.ridge_lambda
    }
    tscv = TimeSeriesSplit(n_splits=Parameters.cv_splits)
    tree_model = TreeElastic(k_min=Parameters.k_min, k_max=Parameters.k_max)
    cv_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, verbose=0, cv=tscv, n_jobs=-1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        cv_search.fit(train_val_portfolios)

    # overall_model.fit(train_val_portfolios)
    overall_model = cv_search.best_estimator_
    sdf = overall_model.predict(test_portfolios)
    sharpe = sdf.mean() / (sdf.std() + 1e-20)
    best_models = np.argmax(sharpe)
    return best_models, overall_model

# %%
if __name__ == '__main__':
    chars = Chars()
    paths = DataPaths()
    regions = ['GL', 'US', 'UK', 'EU', 'AP', 'JP', 'EM']
    for reg in regions:
        print(f"Pruning tree in {reg}")
        for tree_file_path in tqdm(paths.processed_data.iterdir()):
            if tree_file_path.suffix != '.parquet':
                continue
            if reg not in tree_file_path.name:
                continue

            tree_portfolio_pd = to_pandas(tree_file_path)
            _, overall_model = prune(tree_portfolio_pd)

            model_output_name = f"{tree_file_path.with_suffix('').name}{paths.sep}{paths.model_suffix}"
            with open(paths.model_dumps / model_output_name, 'wb') as f:
                pickle.dump(overall_model, f)


# %%
