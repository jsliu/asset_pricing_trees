# %%
import pandas as pd
import pickle
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.constants import DataPaths, Parameters
from prune_trees import prune, to_pandas


def calc_sharpe(tree_portfolio, ap_tree_model):

    logging.info('Splitting data')
    _, test_portfolios = train_test_split(
        tree_portfolio, test_size=Parameters.test_size, shuffle=False)

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
    return df_plot, combo_wei

# %%
if __name__ == '__main__':
    # feature_combination='val_fcf_rank_lme'
    # feature_combination='val_trd_fcf_rank'
    # feature_combination='trd_sen_lme'
    # sharpes, best_combo, combo_wei = calc_sharpe(feature_combination)
    # sns.lineplot(data=sharpes, x="k_nonzero", y="Sharpe", markers="o")    
    # plt.xlabel("Number of non-zero betas (k)")
    # plt.ylabel("Test Sharpe")
    # plt.title(f"Test Sharpe vs Sparsity: {feature_combination}")
    # plt.tight_layout()
    # plt.show()
    reg = 'GL'
    paths = DataPaths()
    all_combos = pd.DataFrame()
    all_portfolios = pd.DataFrame()
    for tree_file_path in tqdm(paths.processed_data.iterdir()):
        if tree_file_path.suffix != '.parquet':
            continue
        if reg not in tree_file_path.name:
            continue
        
        # logging.info('Loading model dump')
        feature_combination = tree_file_path.stem
        model_output_name = f"{feature_combination}{paths.sep}{paths.model_suffix}"
        with open(paths.model_dumps / model_output_name, 'rb') as f:
            ap_tree_model = pickle.load(f)
        tree_portfolio = to_pandas(tree_file_path)
        sharpes, combo_wei = calc_sharpe(tree_portfolio, ap_tree_model)
        all_combos = pd.concat([all_combos, combo_wei])
        all_portfolios = pd.concat([all_portfolios, tree_portfolio[combo_wei.index]], axis=1)
    
    all_train_portfolios, all_test_portfolios = train_test_split(all_portfolios, test_size=Parameters.test_size, shuffle=False)
    final_best_model, final_model = prune(all_train_portfolios)
    # sdf = final_model.predict(all_test_portfolios)
    all_sharpes, all_combo_wei = calc_sharpe(all_portfolios, final_model)
    sns.lineplot(data=all_sharpes, x="k_nonzero", y="Sharpe", markers="o")    
    plt.xlabel("Number of non-zero weights (k)")
    plt.ylabel("Test Sharpe")
    plt.title(f"Test Sharpe vs Sparsity: {feature_combination}")
    plt.tight_layout()
    plt.show()

# %%
