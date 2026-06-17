# %%
import logging
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm

from plot_test_sr import calc_sharpe
from prune_trees import prune, to_pandas
from stock_portfolio_pl import get_node_stocks_cached, get_stocks_in_node, compute_node_scores
from src.constants import DataPaths, Parameters, Columns, Chars, Years
from src.functions import calc_fac_ret, calc_turnover, calc_decay, calc_factor_exposure, calc_group_exposure, summary
from src.preprocessing import read_ei_data, cap_weight
from src.utils import build_comb_pl



# %%
# Running backtest
if __name__ == '__main__':
    sns.set_theme()
    chars = Chars()
    years = Years()
    paths = DataPaths()
    # regions = ['US', 'EU', 'UK', 'JP', 'AP', 'EM']
    regions = ['US', ]
    for reg in regions:
 
        print(f"Loading base characteristics in {reg}")

        features = list(chars.__dict__.values())[:-2]
        data, _, CHARAS_LIST, _ = read_ei_data(region_=reg, target='gross_returns', ei_factors=features)
        # data = read_db_data(region_=reg, features=features, ret_name=Columns.returns_col, data_saved=data_saved)
        # data = read_big_universe(ret_name='gross_returns', features=features)
        ret_df = data['gross_returns'] - data[['gross_returns', Columns.size_col]].groupby(Columns.date_col).apply(lambda x: x.prod(axis=1).sum() / x[Columns.size_col].sum())
        ret_df.name = Columns.returns_col

        print(f"Transform base Size feature into quantiles")
        raw_size_df = data[Columns.size_col].groupby('date').transform(lambda x: cap_weight(x))
        raw_size_df.name = Columns.size_col
        lme_df = np.log(data[Columns.size_col])
        lme_df.name = chars.lme

        print(f"Stack raw Size and Returns variables together")
        data = pl.from_pandas(pd.concat([data.drop(columns=['gross_returns']), lme_df], axis=1).reset_index())
        merged_df = pl.from_pandas(pd.concat([raw_size_df, ret_df], axis=1).reset_index())

        
        comb_pl = build_comb_pl(
            data=data,
            merged_df=merged_df,
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

            df = to_pandas(tree_file_path)
            dates_idx = df.index.get_level_values(Columns.date_col)

            
            # --- Build date -> positions mapping ---
            date_to_pos = {}
            for pos, d in enumerate(dates_idx):
                date_to_pos.setdefault(d, []).append(pos)

            # Convert to numpy arrays (faster iloc)
            date_to_pos = {k: np.array(v, dtype=np.int32) for k, v in date_to_pos.items()}

            # --- Build cumulative (expanding) indices ---
            sorted_dates = sorted(date_to_pos.keys())
            cumulative_map = {}
            running_idx = []

            for d in sorted_dates:
                running_idx.extend(date_to_pos[d])
                cumulative_map[d] = np.array(running_idx.copy(), dtype=np.int32)

            # store
            tree_data[tree_file_path.name] = {
                "df": df,
                "date_map": date_to_pos,
                "cum_map": cumulative_map
            }

            all_dates.update(sorted_dates)

        # Sort dates once
        dates = sorted(all_dates)

        # -----------------------------
        # 2. Backtest loop
        # -----------------------------
        rets = pd.Series(index=dates, name='Return', dtype=float)
        stock_scores = []

        for d in tqdm(dates, desc=f"{reg} Backtest", unit=Columns.date_col):

            if d // 10000 < years.min_year:
                continue

            train_list = []
            test_list = []

            for name, obj in tree_data.items():
                # print(name)
                df = obj["df"]
                date_map = obj["date_map"]
                cum_map = obj["cum_map"]

                if d not in date_map:
                    continue

                train_idx = cum_map[d]
                test_idx = date_map[d]

                if len(train_idx) == 0:
                    continue

                train_val_portfolio = df.iloc[train_idx]
                test_portfolio = df.iloc[test_idx]

                best_model, overall_model = prune(train_val_portfolio)

                selected_idx = np.nonzero(overall_model.betas[best_model, :])[0]

                train_list.append(train_val_portfolio.iloc[:, selected_idx])
                test_list.append(test_portfolio.iloc[:, selected_idx])

            if not test_list:
                continue

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

            final_sharpes, final_combo_wei = calc_sharpe(
                all_train_portfolios, final_model
            )

            dfs = []
            for (tree_key, port_col, node_id), w in zip(final_combo_wei.index, final_combo_wei.to_numpy()):

                if w == 0:
                    continue

                df_node = get_node_stocks_cached(
                    comb_pl,
                    tree_key,
                    port_col,
                    int(node_id),
                    Parameters.n_splits
                )

                df_node = df_node.filter(pl.col(Columns.date_col)==d)
                if df_node.is_empty():
                    continue
                
                df_scores = compute_node_scores(
                    df_node=df_node,
                    tree_key=tree_key,
                    port_col=port_col,
                )

                
                df_scores = df_scores.with_columns(
                    (pl.col("factor_score") * w).alias("weighted_score")
                )

                dfs.append(
                    df_scores.select([Columns.id_col, "weighted_score"])
                )

            if len(dfs) > 0:
                final_df = (
                    pl.concat(dfs)
                    .group_by(Columns.id_col)
                    .agg(
                        pl.col("weighted_score").sum().alias("final_score")
                    )
                )
            else:
                final_df = pl.DataFrame()

            n = final_df.height

            final_df = final_df.with_columns([
                (
                    (pl.col("final_score").rank("min") - 0.5) / n
                ).alias("u")
            ]).with_columns(
                pl.col("u")
                .map_elements(norm.ppf)
                .alias("norm_score")
            ).drop("u")   
            
            final_df = final_df.with_columns(
                pl.lit(d).alias(Columns.date_col)
            )

            stock_scores.append(final_df)
            sdf = final_model.predict(all_test_portfolios)
            rets.loc[d] = sdf[final_best_model].to_numpy()



        stk_score = pl.concat(stock_scores)
        # stk_score.write_csv(paths.output / f'{reg}_score.csv')
        
        rets = rets.dropna()
        rets.cumsum().plot()
        plt.title(f"{reg} cumulative returns")
        plt.show()

        # rets.to_csv(paths.output / f'{reg}_ret.csv')
    
    # %% 
    sns.set_theme()
    chars = Chars()
    years = Years()
    paths = DataPaths()
    # regions = ['GL', 'US', 'EU', 'UK', 'JP', 'AP', ]
    regions = ['GL', ]
    for reg in regions:
        print(f'Processing {reg}')
        features = list(chars.__dict__.values())[:-2]
        data, _, CHARAS_LIST, _ = read_ei_data(region_=reg, target='gross_returns', ei_factors=features)
        data.loc[:, 'lme'] = np.log(data[Columns.size_col])
        ai_pnl = pd.read_csv(paths.output / f"{reg}_ret.csv")
        ai_pnl.columns = ['Date', 'Return']
        ai_pnl = ai_pnl.set_index('Date')
        ei_pnl = calc_fac_ret(data[features].mean(axis=1).swaplevel(0, 1), data['gross_returns'].swaplevel(0, 1), q=5, date_col='date', score_weighted=True)
        pnl = pd.concat([ai_pnl, ei_pnl], axis=1, sort=True)
        pnl.index = pd.to_datetime(pnl.index, format="%Y%m%d")
        dates = pnl.index
        pnl = pnl.loc[dates[dates >= '2006-01-31']].fillna(0)
        pnl.columns = ['Tree', 'EI']
        pnl['Combined'] = pnl['EI'] * 0.7 + pnl['Tree'] * 0.3 
        # pnl = pd.concat([pnl, comb_pnl], axis=1)
        pnl.cumsum().plot(title=f'{reg}')
        plt.show()
        print(summary(pnl, ann_factor=12, sorted=False))
        print(pnl.corr())
    
# %%
    sns.set_theme()
    chars = Chars()
    years = Years()
    paths = DataPaths()
    # regions = ['GL', 'US', 'EU', 'UK', 'JP', 'AP',]
    regions = ['US', ]
    for reg in regions:
        print(f'Processing {reg}')
        features = list(chars.__dict__.values())[:-2]
        data, _, CHARAS_LIST, _ = read_ei_data(region_=reg, target='gross_returns', ei_factors=features)
        data.loc[:, 'lme'] = np.log(data[Columns.size_col])

        ai_score = pd.read_csv(paths.output / f"{reg}_score.csv").set_index(['date', 'permno'])['norm_score']
        ai_score.name = 'Tree'
        ei_score = data[features].swaplevel(0, 1)
        combined_score = ei_score.merge(ai_score, left_index=True, right_index=True, how='left')
        dates = combined_score.index.get_level_values('date').unique()
        valid_dates = ~combined_score['Tree'].groupby('date').apply(lambda x: x.isna().all())
        combined_score = combined_score.loc[dates[valid_dates]]
        ai_pnl = calc_fac_ret(ai_score, data['gross_returns'].swaplevel(0, 1), q=5, date_col='date', score_weighted=True)
        ei_pnl = calc_fac_ret(ei_score.mean(axis=1), data['gross_returns'].swaplevel(0, 1), q=5, date_col='date', score_weighted=True)
        combined_pnl = calc_fac_ret(combined_score.mean(axis=1), data['gross_returns'].swaplevel(0, 1), q=5, date_col='date', score_weighted=True)
        pnl = pd.concat([ai_pnl, ei_pnl, combined_pnl], axis=1).dropna()
        pnl.index = pd.to_datetime(pnl.index, format="%Y%m%d")
        pnl.columns = ['Tree', 'EI', 'Combined']
        pnl.cumsum().plot(title=f'{reg}')
        plt.show()
        print(summary(pnl, ann_factor=12, sorted=False))

        roll_perf = pnl.rolling(12).sum()
        roll_perf.index = pd.to_datetime(roll_perf.index, format="%Y%m%d")
        roll_perf.plot(title=f"{reg}")
        plt.show()

        factor_pnl = calc_fac_ret(combined_score, data['gross_returns'].swaplevel(0, 1), q=5, date_col='date', score_weighted=True)
        factor_pnl.index = pd.to_datetime(factor_pnl.index, format="%Y%m%d")
        factor_pnl.cumsum().plot(title=f"{reg}")
        plt.show()
        print(summary(factor_pnl, ann_factor=12, sorted=False))
        print(factor_pnl.corr())

        factor_turnover = combined_score.apply(lambda x: calc_turnover(x, stock_id='permno', date_col='date'))
        factor_turnover.index = pd.to_datetime(factor_turnover.index, format="%Y%m%d")
        avg_to = factor_turnover.mean()
        ax = factor_turnover.plot(title=f"{reg}")
        labels = [
            f"{col}: {avg_to[col]*100:.1f}%"
            for col in factor_turnover.columns
        ]
        ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

        factor_decay = combined_score.apply(lambda x: calc_decay(x, date_col='date'))
        factor_decay.plot(title=f"{reg}").legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

        factor_exposure = combined_score.groupby('date').apply(lambda x: x.iloc[:, :-1].apply(lambda y: calc_factor_exposure(x.iloc[:, -1], factor=y)))
        factor_exposure.index = pd.to_datetime(factor_exposure.index, format="%Y%m%d")
        avg_exp = factor_exposure.mean()
        ax = factor_exposure.plot(title=f"{reg}")
        labels = [
            f"{col}: {avg_exp[col]:.2f}"
            for col in factor_exposure.columns
        ]
        ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()
        
        plt.figure(figsize=(10,6))
        corr_by_date = combined_score.groupby('date').corr()
        cols = combined_score.columns
        for i in range(len(cols)-1):
            s = corr_by_date.loc[(slice(None), cols[i]), 'Tree']
            s.index = s.index.droplevel(1)
            s.index = pd.to_datetime(s.index, format="%Y%m%d")
            avg_corr = s.mean()
            plt.plot(s, label=f"{cols[i]}-tree: {avg_corr*100:.1f}%")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f"{reg}")
        plt.show()
# %%
