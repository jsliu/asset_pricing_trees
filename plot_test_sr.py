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
from src.functions import scale
from prune_trees import prune, to_pandas


def plot_sharpe(data):
    ax = sns.lineplot(data=data, x="k_nonzero", y="Sharpe", markers="o")

    # Extract plotted line
    line = ax.lines[0]
    x_data = line.get_xdata()
    y_data = line.get_ydata()

    # Find max y and corresponding x
    idx_max = y_data.argmax()
    x_star = x_data[idx_max]
    y_star = y_data[idx_max]

    # Draw vertical dashed line
    ax.axvline(x=x_star, linestyle="--", color="red")

    # Highlight the max point
    ax.scatter(x_star, y_star, color="red", zorder=3)

    # Labels
    ax.set_xlabel("Number of non-zero weights (k)")
    ax.set_ylabel("Test Sharpe")
    ax.set_title(f"{reg}: Test Sharpe vs Sparsity")

    plt.tight_layout()
    plt.show()


def calc_sharpe(tree_portfolio, ap_tree_model, use_test_data=True):

    logging.info('Splitting data')
    train_portfolios, test_portfolios = train_test_split(
        tree_portfolio, test_size=Parameters.test_size, shuffle=False)

    test_portfolios= test_portfolios.dropna(axis=1, how="all").fillna(0)
    train_portfolios = train_portfolios[test_portfolios.columns].fillna(0)
    
    if use_test_data:
        sdf = ap_tree_model.predict(test_portfolios)
    else:
        sdf = ap_tree_model.predict(train_portfolios)
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
    all_combo_wei = {}
    regions = ['GL', 'US', 'UK', 'EU', 'AP', 'JP', 'EM']
    # regions = ['GL', ]
    paths = DataPaths()
    for reg in regions:
        print(f"Processing in {reg}")
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
            sharpes, combo_wei = calc_sharpe(tree_portfolio, ap_tree_model, use_test_data=False)
            all_combos = pd.concat([all_combos, combo_wei])
            all_portfolios = pd.concat([all_portfolios, tree_portfolio[combo_wei.index]], axis=1)
        
        # some columns eventhough they have same column name, the values are not identical, but 99% correlated
        all_portfolios = all_portfolios.loc[:, ~(all_portfolios.T.duplicated() | all_portfolios.columns.duplicated())]
        _, final_model = prune(all_portfolios)
        # sdf = final_model.predict(all_test_portfolios)
        all_sharpes, all_wei = calc_sharpe(all_portfolios, final_model, use_test_data=True)
        all_combo_wei[reg] = scale(all_wei)
        plot_sharpe(all_sharpes)

# %%