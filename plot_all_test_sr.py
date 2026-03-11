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
from src.stock_portfolio import get_B_for_best_combo_over_time
from src.preprocessing import read_ei_data
from src.functions import calc_fac_ret, summary
from plot_test_sr import calc_sharpe


# %%
if __name__ == '__main__':
    sns.set_theme()
    chars = Chars()
    paths = DataPaths()
    sharpe, B = {}, {}
    reg = 'GL'
    features = list(chars.__dict__.values())[:-2]
    data, _, CHARAS_LIST, _ = read_ei_data(region_=reg, target=Columns.returns_col, ei_factors=features)
    data.loc[:, 'lme'] = np.log(data[Columns.size_col])
    stk_wei = dict()
    for i, tree_file_path in enumerate(tqdm(paths.processed_data.iterdir())):
        if tree_file_path.suffix != '.pkl':
            continue
        feature_combination = tree_file_path.stem
        logging.info(f"Reading {feature_combination} ...")
        max_sharpe, best_combo, combo_wei = calc_sharpe(feature_combination)
        sharpe[str(i)] = {"Combo": feature_combination, "SR": max_sharpe["Sharpe"].max()}
        B = get_B_for_best_combo_over_time(data, best_combo=best_combo, weight_col=Columns.size_col, n_split=Parameters.n_splits, )
        stk_wei[feature_combination] = B.dot(combo_wei)
    stock_weights = pd.DataFrame(stk_wei).mean(axis=1)
    sharpes = pd.DataFrame(sharpe).T.sort_values('SR').reset_index()
    sharpes.plot(x='index', y='SR', legend=False, xlabel='Combo', ylabel='Sharpe Ratio')
    plt.show()

    ap_pnl = calc_fac_ret(stock_weights, data['gross_returns'].swaplevel(0, 1), q=5, date_col='date')
    ei_pnl = calc_fac_ret(data[features].mean(axis=1).swaplevel(0, 1), data['gross_returns'].swaplevel(0, 1), q=5, date_col='date')
    pnl = pd.concat([ap_pnl, ei_pnl], axis=1)
    pnl.index = pd.to_datetime(pnl.index, format="%Y%m%d")
    pnl.columns = ['Tree', 'EI']
    pnl.cumsum().plot()
    plt.show()
    print(summary(pnl, ann_factor=12, sorted=False))
# %%
