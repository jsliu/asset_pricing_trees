# %%
import pandas as pd
import polars as pl
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from tqdm import tqdm

from prune_trees import prune, to_pandas
from src.constants import DataPaths, Parameters, Columns, Chars, Years
from src.functions import calc_fac_ret, summary
from src.preprocessing import read_ei_data


# %%
# Running backtest
if __name__ == '__main__':
    sns.set_theme()
    chars = Chars()
    years = Years()
    paths = DataPaths()
    # regions = ['GL', 'US', 'EU', 'UK', 'JP', 'AP', 'EM']
    regions = ['US', ]
    for reg in regions:
        print(f'Processing {reg}')
        features = list(chars.__dict__.values())[:-2]
        data, _, CHARAS_LIST, _ = read_ei_data(region_=reg, target=Columns.returns_col, ei_factors=features)
        data.loc[:, 'lme'] = np.log(data[Columns.size_col])
        dates = pd.to_datetime(data.index.get_level_values('date').unique(), format="%Y%m%d")[:-1]

        # run backtest 
        pnl = {}
        rets = pd.Series(index=dates)
        for i, d in enumerate(dates):
            if d.year < years.min_year:
                continue 
            print(f"Processing {d} ...")
            train_portfolios, test_portfolios = pd.DataFrame(), pd.DataFrame()
            for tree_file_path in tqdm(paths.processed_data.iterdir()):
                if tree_file_path.suffix != '.parquet':
                    continue
                if reg not in tree_file_path.name:
                    continue
                
                # feature_combination = tree_file_path.stem
                # logging.info(f"Reading {feature_combination} ...")
                # tree_portfolio = pl.read_parquet(paths.processed_data / f"{reg}_{feature_combination}.parquet")
                tree_portfolio = to_pandas(tree_file_path)
                
                logging.info('Splitting data')
                train_val_portfolio = tree_portfolio.loc[tree_portfolio.index.get_level_values('date').intersection(dates[:i])]
                test_portfolio = tree_portfolio.loc[[d]]
                best_model, overall_model = prune(train_val_portfolio)
                train_portfolios = pd.concat([train_portfolios, train_val_portfolio.iloc[:, np.nonzero(overall_model.betas[best_model, :])[0]]], axis=1)
                test_portfolios = pd.concat([test_portfolios, test_portfolio.iloc[:, np.nonzero(overall_model.betas[best_model, :])[0]]], axis=1)

            all_train_portfolios = train_portfolios.loc[:, ~train_portfolios.columns.duplicated()]
            all_test_portfolios = test_portfolios.loc[:, ~test_portfolios.columns.duplicated()]
            final_best_model, final_model = prune(all_train_portfolios)
            sdf = final_model.predict(all_test_portfolios)
            rets.loc[d] = sdf[final_best_model].to_numpy()
        rets.dropna().cumsum().plot()
        plt.show()
        rets.to_csv(paths.output / f'{reg}_ret.csv')
    # %% 
    # w = w - np.mean(w)
    # combo_wei = pd.Series(w/np.sum(np.abs(w)), index=best_combo, name='weight')
    sns.set_theme()
    chars = Chars()
    years = Years()
    paths = DataPaths()
    # regions = ['GL', 'US', 'EU', 'UK', 'JP', 'AP', 'EM']
    regions = ['GL', 'US']
    for reg in regions:
        print(f'Processing {reg}')
        features = list(chars.__dict__.values())[:-2]
        data, _, CHARAS_LIST, _ = read_ei_data(region_=reg, target=Columns.returns_col, ei_factors=features)
        data.loc[:, 'lme'] = np.log(data[Columns.size_col])
        
        ai_pnl = pd.read_csv(paths.output / f"{reg}_ret.csv").set_index('date')
        ai_pnl.index = pd.to_datetime(ai_pnl.index, format="%Y-%m-%d")
        ei_pnl = calc_fac_ret(data[features].mean(axis=1).swaplevel(0, 1), data['gross_returns'].swaplevel(0, 1), q=5, date_col='date', score_weighted=True)
        ei_pnl.index = pd.to_datetime(ei_pnl.index, format="%Y%m%d")
        pnl = pd.concat([ai_pnl, ei_pnl], axis=1).dropna()
        comb_pnl = pnl.mean(axis=1)
        pnl = pd.concat([pnl, comb_pnl], axis=1)
        pnl.columns = ['Tree', 'EI', 'Combined']
        pnl.cumsum().plot(title=f'{reg}')
        plt.show()
        print(summary(pnl, ann_factor=12, sorted=False))
# %%
