# %%
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm

from plot_test_sr import calc_sharpe
from prune_trees import prune, to_pandas, residualize_portfolios
from build_trees_pl import prepare_data
from stock_portfolio_pl import get_node_stocks_cached, compute_node_scores
from src.constants import DataPaths, Parameters, Columns, Chars, Years
from src.functions import calc_fac_ret, calc_turnover, calc_decay, calc_factor_exposure, calc_group_exposure, summary
from src.preprocessing import read_ei_data, read_big_universe, read_all_data
from src.utils import build_comb_pl


def rank_normalize(x):
    n = len(x)

    # equivalent to rank("min")
    u = (x.rank(method="min") - 0.5) / n

    return norm.ppf(u)


# %%
# Running backtest
if __name__ == '__main__':
    sns.set_theme()
    chars = Chars()
    years = Years()
    paths = DataPaths()
    suffix = 'resid2'
    # ret_name = 'Universe Returns'
    ret_name = 'gross_returns'
    # regions = ['GL', 'US', 'EU', 'UK', 'JP', 'AP', 'EM']
    regions = ['GL', ]
    for reg in regions:
 
        print(f"Loading base characteristics in {reg}")

        features = list(chars.__dict__.values())[:-2]
        if reg == "ALL":
            data, CHARAS_LIST, _ = read_all_data(target=ret_name, ei_factors=features)
        else:
            data, _, CHARAS_LIST, _, _ = read_ei_data(region_=reg, target=ret_name, ei_factors=features)
        # data = read_db_data(region_=reg, features=features, ret_name=Columns.returns_col, data_saved=data_saved)
        # data = read_big_universe(ret_name=ret_name, features=features)
        factor_returns = calc_fac_ret(data[features], data[ret_name], date_col="date", score_weighted=True)

        data_pl, ret_and_mcap = prepare_data(data, ret_name=ret_name, factors=None, equal_weighted=False)
        
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

            #  calculate residual returns
            tree_pd = to_pandas(tree_file_path)
            dates_idx = tree_pd.index.get_level_values(Columns.date_col)
            valid_factors = factor_returns.loc[dates_idx]
            df = residualize_portfolios(tree_pd, valid_factors)
            
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
            # if d < 20070330:
                continue

            # print(d)
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
                # train_list.append(train_val_portfolio)
                # test_list.append(test_portfolio)

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

            final_sharpes, final_combo_wei, final_port = calc_sharpe(
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
        stk_score.write_csv(paths.output / f'{reg}_score_{suffix}.csv')
        
        rets = rets.dropna()
        rets.cumsum().plot()
        plt.title(f"{reg} cumulative returns")
        plt.show()

        rets.to_csv(paths.output / f'{reg}_ret_{suffix}.csv')
    
    # %% 
    sns.set_theme()
    chars = Chars()
    years = Years()
    paths = DataPaths()
    suffix = 'resid2'
    ret_name = 'gross_returns'
    # regions = ['GL', 'US', 'EU', 'UK', 'JP', 'AP', 'EM']
    regions = ['ALL', ]
    for reg in regions:
        print(f'Processing {reg}')
        # features = list(chars.__dict__.values())[:-2]
        features = ['val', 'qual', 'fcf_rank', 'trd', 'sen']
        if reg == "ALL":
            data, CHARAS_LIST, _ = read_all_data(target=ret_name, ei_factors=features)
        else:
            data, _, CHARAS_LIST, _, _ = read_ei_data(region_=reg, target=ret_name, ei_factors=features)
        data.loc[:, 'lme'] = np.log(data[Columns.size_col])
        ai_pnl = pd.read_csv(paths.output / f"{reg}_ret_{suffix}.csv")
        ai_pnl.columns = ['Date', 'Return']
        ai_pnl = ai_pnl.set_index('Date')
        ei_pnl = calc_fac_ret(data[features].mean(axis=1).swaplevel(0, 1), data[ret_name].swaplevel(0, 1), date_col='date', score_weighted=True)
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
    suffix = 'resid2'
    ret_name = 'gross_returns'
    factor_weights = {
        'ALL': {'fcf_rank': 0.08, 'qual': 0.37, 'sen': 0.22, 'trd': 0.15, 'val': 0.18},
        'AP': {'fcf_rank': 0.085, 'qual': 0.365, 'sen': 0.18, 'trd': 0.2, 'val': 0.17},
        'EM': {'fcf_rank': 0.085, 'qual': 0.385, 'sen': 0.18, 'trd': 0.2, 'val': 0.15},
        'EU': {'fcf_rank': 0.085, 'qual': 0.365, 'sen': 0.19, 'trd': 0.19, 'val': 0.17},
        'GL': {'fcf_rank': 0.08, 'qual': 0.37, 'sen': 0.22, 'trd': 0.15, 'val': 0.18},
        'JP': {'fcf_rank': 0.25, 'qual': 0.25, 'sen': 0.125, 'trd': 0.125, 'val': 0.25},
        'UK': {'fcf_rank': 0.07, 'qual': 0.36, 'sen': 0.22, 'trd': 0.17, 'val': 0.18},
        'US': {'fcf_rank': 0.06, 'qual': 0.4, 'sen': 0.18, 'trd': 0.18, 'val': 0.18},
    }
    # regions = ['GL', 'US', 'EU', 'UK', 'JP', 'AP', 'EM']
    regions = ['ALL', ]
    for reg in regions:
        print(f'Processing {reg}')
        features = list(chars.__dict__.values())[:-2]
        # features = ['val', 'qual', 'fcf_rank', 'trd', 'sen']
        if reg == "ALL":
            data, CHARAS_LIST, stock_info = read_all_data(target=ret_name, ei_factors=features)
        else:
            data, _, CHARAS_LIST, _, stock_info = read_ei_data(region_=reg, target=ret_name, ei_factors=features)
        data.loc[:, 'lme'] = np.log(data[Columns.size_col])

        ai_score = pd.read_csv(paths.output / f"{reg}_score_{suffix}.csv").set_index(['date', 'permno'])['norm_score']
        ai_score.name = 'Tree'
        ei_score = data[features].swaplevel(0, 1)
        combined_score = ei_score.merge(ai_score, left_index=True, right_index=True, how='left')
        dates = combined_score.index.get_level_values('date').unique()
        valid_dates = ~combined_score['Tree'].groupby('date').apply(lambda x: x.isna().all())
        combined_score = combined_score.loc[dates[valid_dates]]
        ai_pnl = calc_fac_ret(ai_score, data[ret_name].swaplevel(0, 1), date_col='date', score_weighted=True)
        ei_pnl = calc_fac_ret(ei_score.mean(axis=1), data[ret_name].swaplevel(0, 1), date_col='date', score_weighted=True)
        combined_pnl = calc_fac_ret(combined_score.mean(axis=1), data[ret_name].swaplevel(0, 1), date_col='date', score_weighted=True)
        pnl = pd.concat([ai_pnl, ei_pnl, combined_pnl], axis=1).dropna()
        pnl.index = pd.to_datetime(pnl.index, format="%Y%m%d")
        pnl.columns = ['Tree', 'EI', 'Combined']
        pnl.cumsum().plot(title=f'{reg}: Performance').legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()
        print(summary(pnl, ann_factor=12, sorted=False))

        roll_perf = pnl.rolling(12).sum()
        roll_perf.index = pd.to_datetime(roll_perf.index, format="%Y%m%d")
        roll_perf.plot(title=f"{reg}: Rolling 1-year")
        plt.show()
        
        roll_perf = pnl.rolling(36).sum()
        roll_perf.index = pd.to_datetime(roll_perf.index, format="%Y%m%d")
        roll_perf.plot(title=f"{reg}: Rolling 3-year")
        plt.show()

        factor_pnl = calc_fac_ret(combined_score, data['gross_returns'].swaplevel(0, 1), date_col='date', score_weighted=True)
        factor_pnl.index = pd.to_datetime(factor_pnl.index, format="%Y%m%d")
        factor_pnl.cumsum().plot(title=f"{reg}: Factor Performance").legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()
        print(summary(factor_pnl, ann_factor=12, sorted=False))
        print(factor_pnl.corr())

        grp_exp = pd.concat([ai_score, stock_info['sector'].swaplevel(0, 1)], axis=1, join='inner').groupby('date').apply(lambda x: calc_group_exposure(x.iloc[:, 0], groups=x.iloc[:, 1]))
        grp_exp_df = grp_exp.unstack()
        grp_exp_df.mean().plot(kind='bar', title=f"{reg}: Average Exposure")
        grp_exp_df.index = pd.to_datetime(grp_exp_df.index, format="%Y%m%d")
        grp_exp_df.plot(title=f"{reg}: Sector Exposure").legend(loc='upper left', bbox_to_anchor=(1, 1))

        factor_turnover = combined_score.apply(lambda x: calc_turnover(x, stock_id='permno', date_col='date'))
        factor_turnover.index = pd.to_datetime(factor_turnover.index, format="%Y%m%d")
        avg_to = factor_turnover.mean()
        ax = factor_turnover.plot(title=f"{reg}: Turnover")
        labels = [
            f"{col}: {avg_to[col]*100:.1f}%"
            for col in factor_turnover.columns
        ]
        ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

        factor_decay = combined_score.apply(lambda x: calc_decay(x, date_col='date'))
        factor_decay.plot(title=f"{reg}: Decay").legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

        factor_exposure = combined_score.groupby('date').apply(lambda x: x.iloc[:, :-1].apply(lambda y: calc_factor_exposure(x.iloc[:, -1], factor=y)))
        factor_exposure.index = pd.to_datetime(factor_exposure.index, format="%Y%m%d")
        avg_exp = factor_exposure.mean()
        ax = factor_exposure.plot(title=f"{reg}: Factor Exposure")
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
        plt.title(f"{reg}: Correlation")
        plt.show()

        fac_wei = pd.Series(factor_weights[reg])
        ei_alpha = combined_score.iloc[:,:-1].mean(axis=1)
        prod_alpha = combined_score.iloc[:, :-1].fillna(0).dot(fac_wei)*0.8 + combined_score.iloc[:, -1].fillna(0)*0.2
        new_alpha = combined_score.mean(axis=1)
        alpha = pd.concat([combined_score.iloc[:, -1], ei_alpha, new_alpha, prod_alpha], axis=1)
        alpha.columns = ['tree', 'ew_ei', 'ew_all', 'prod']
        alpha.to_csv(f'result/{reg}_all_score_{suffix}.csv')
    # %%
    sns.set_theme()
    chars = Chars()
    paths = DataPaths()
    reg = "GL"
    ret_name = "Universe Returns"
    groups = ['region', 'ind']
    ai_features = list(chars.__dict__.values())[:-2]
    ai_data = read_big_universe(ret_name=ret_name, features=ai_features, add_groups=True)
    ai_score = pd.read_csv(paths.output / f"{reg}_score_big_univ.csv").set_index(['date', 'permno'])['final_score']
    ai_score.name = 'Tree'
    score_and_ret = pd.concat([ai_data[[ret_name] + groups].swaplevel(0, 1), ai_score], axis=1, join='outer')
    score_and_ret["alpha"] = score_and_ret.groupby(["date"])["Tree"].transform(rank_normalize)
    score_and_ret["region_industry_neutral"] = score_and_ret.groupby(["date", "region", "ind"])["Tree"].transform(rank_normalize)
    alpha_scores = score_and_ret[['alpha', 'region_industry_neutral']]
    pnl = calc_fac_ret(alpha_scores, score_and_ret[ret_name]/100, q=5, date_col='date', score_weighted=True)
    pnl.index = pd.to_datetime(pnl.index, format="%Y%m%d")
    pnl.cumsum().plot()
    plt.show()
    print(summary(pnl, ann_factor=12, sorted=False))
    to = alpha_scores.apply(lambda x: calc_turnover(x, date_col='date', stock_id='permno'))
    to.index = pd.to_datetime(to.index, format="%Y%m%d")
    to.plot()
    plt.show()
    decay = alpha_scores.apply(lambda x: calc_decay(x, date_col='date'))
    decay.plot()
    plt.show()
    factor_scores = ai_data[["by", "vol", "mom_1y1m", "mkt_cap"]].groupby('date').transform(lambda x: rank_normalize(x))
    scores1 = factor_scores.merge(alpha_scores['alpha'], left_index=True, right_index=True)
    factor_exposure1 = scores1.groupby('date').apply(lambda x: x.iloc[:, :-1].apply(lambda y: calc_factor_exposure(x.iloc[:, -1], factor=y)))
    factor_exposure1.index = pd.to_datetime(factor_exposure1.index, format="%Y%m%d")
    avg_exp = factor_exposure1.mean()
    avg_exp.plot(kind='bar')
    ax = factor_exposure1.plot(title=f"{reg}: Factor Exposure: alpha")
    labels = [
        f"{col}: {avg_exp[col]:.2f}"
        for col in factor_scores.columns
    ]
    ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


    scores2 = factor_scores.merge(alpha_scores['region_industry_neutral'], left_index=True, right_index=True)
    factor_exposure2 = scores2.groupby('date').apply(lambda x: x.iloc[:, :-1].apply(lambda y: calc_factor_exposure(x.iloc[:, -1], factor=y)))
    factor_exposure2.index = pd.to_datetime(factor_exposure2.index, format="%Y%m%d")
    avg_exp = factor_exposure2.mean()
    avg_exp.plot(kind='bar')
    ax = factor_exposure2.plot(title=f"{reg}: Factor Exposure: region_industry_neutral")
    labels = [
        f"{col}: {avg_exp[col]:.2f}"
        for col in factor_scores.columns
    ]
    ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    
    all_scores = factor_scores.swaplevel(0, 1).merge(alpha_scores, left_index=True, right_index=True, how='left')
    all_pnls = calc_fac_ret(all_scores, ai_data[ret_name].swaplevel(0, 1)/100, q=5, date_col='date')
    all_pnls.index = pd.to_datetime(all_pnls.index, format="%Y%m%d")
    all_pnls.cumsum().plot()
    plt.show()
    print(summary(all_pnls, ann_factor=12, sorted=False))
    decays = all_scores.apply(lambda x: calc_decay(x, date_col='date'))
    decays.plot()
    plt.show()

# %%
