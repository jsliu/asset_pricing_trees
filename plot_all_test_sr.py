# %%
import pandas as pd
import pickle
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.constants import DataPaths, Parameters, Columns, Chars


def find_max_sharpe(feature_combination):
    paths = DataPaths()
    tree_portfolio = pd.read_pickle(paths.processed_data / f"{feature_combination}.pkl")

    logging.info("Selecting only returns")
    ret_indexes = tree_portfolio.index.get_level_values(Columns.features_col) == Columns.w_returns_col
    tree_portfolio = tree_portfolio[ret_indexes]

    logging.info('Splitting data')
    _, test_portfolios = train_test_split(
        tree_portfolio, test_size=Parameters.test_size, shuffle=False)

    logging.info('Loading model dump')
    model_output_name = f"{feature_combination}{paths.sep}{paths.model_suffix}"
    with open(paths.model_dumps / model_output_name, 'rb') as f:
        ap_tree_model = pickle.load(f)

    sdf = ap_tree_model.predict(test_portfolios)
    sharpe_values = np.mean(sdf, axis=0) / (np.std(sdf, axis=0) + 1e-20)
    k_nonzero = np.sum(ap_tree_model.betas != 0, axis=1)

    df_plot = pd.DataFrame({
        "k_nonzero": k_nonzero,
        "Sharpe": sharpe_values
    })

    return np.max(sharpe_values)

# %%
if __name__ == '__main__':
    sns.set_theme()
    chars = Chars()
    paths = DataPaths()
    sharpe = {}
    for i, tree_file_path in enumerate(tqdm(paths.processed_data.iterdir())):
        if tree_file_path.suffix != '.pkl':
            continue
        feature_combination = tree_file_path.stem
        logging.info(f"Reading {feature_combination} ...")
        sharpe[str(i)] = find_max_sharpe(feature_combination)
    sharpes = pd.DataFrame(sharpe, index=['SR']).T.sort_values('SR').reset_index()
    # sharpes.loc[:, 'index'] = sharpes['index'].astype(str)
    sharpes.plot(x='index', y='SR', legend=False, xlabel='Index', ylabel='Sharpe Ratio')
    plt.show()
# %%
