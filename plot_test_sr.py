# %%
import pandas as pd
import pickle
import logging
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.constants import DataPaths, Parameters, Columns


def calc_sharpe(feature_combination):
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

    mask = df_plot["Sharpe"] == df_plot["Sharpe"].max()
    max_idx = df_plot.index[mask]
    if len(max_idx) > 1:
        best_combo = tree_portfolio.columns[np.unique(np.nonzero(ap_tree_model.betas[max_idx, :][1]))]
        sdf_wei = ap_tree_model.betas[max_idx, :][1]
    else:
        best_combo = tree_portfolio.columns[np.nonzero(ap_tree_model.betas[max_idx, :])[1]]
        sdf_wei = ap_tree_model.betas[max_idx, :]
    w = sdf_wei[np.nonzero(sdf_wei)]
    # w = w - np.mean(w)
    # combo_wei = pd.Series(w/np.sum(np.abs(w)), index=best_combo, name='weight')
    combo_wei = pd.Series(w, index=best_combo, name='weight')
    tree_portfolio[best_combo].to_parquet(paths.combo_data / f"{feature_combination}.parquet")
    combo_wei.to_frame().to_parquet(paths.combo_weight /f"{feature_combination}.parquet")
    return df_plot, best_combo, combo_wei

# %%
if __name__ == '__main__':
    # feature_combination='val_fcf_rank_lme'
    # feature_combination='val_trd_fcf_rank'
    feature_combination='trd_sen_lme'
    sharpes, best_combo, combo_wei = calc_sharpe(feature_combination)
    sns.lineplot(data=sharpes, x="k_nonzero", y="Sharpe", markers="o")    
    plt.xlabel("Number of non-zero betas (k)")
    plt.ylabel("Test Sharpe")
    plt.title(f"Test Sharpe vs Sparsity: {feature_combination}")
    plt.tight_layout()
    plt.show()

# %%
