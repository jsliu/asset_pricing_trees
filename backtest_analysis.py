# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

from backtest import r_squared, rank_normalize
from src.constants import DataPaths, Columns, Chars, Years
from src.functions import calc_fac_ret, calc_turnover, calc_decay, calc_factor_exposure, grouped_exposure, summary, get_residuals
from src.preprocessing import read_ei_data, read_big_universe, read_all_data

# %% 
# performance at portfolio level
sns.set_theme()
chars = Chars()
years = Years()
paths = DataPaths()
suffix = 'std'
ret_name = 'gross_returns'
regions = ['GL', 'US', 'EU', 'UK', 'JP', 'AP', 'EM']
# regions = ['US', ]
for reg in regions:
    print(f'Processing {reg}')
    # features = list(chars.__dict__.values())[:-2]
    features = ['val', 'qual', 'fcf_rank', 'trd', 'sen']
    if reg == "ALL":
        data, CHARAS_LIST, _ = read_all_data(target=ret_name, ei_factors=features)
    else:
        data, _, CHARAS_LIST, _, _ = read_ei_data(region_=reg, target=ret_name, ei_factors=features)
    data.loc[:, 'lme'] = np.log(data[Columns.size_col])
    data = data.swaplevel(0, 1)

    ai_pnl = pd.read_csv(paths.output / f"{reg}_ret_{suffix}.csv", index_col=0)[['Return']]
    ai_pnl.index.name = 'Date'
    ei_pnl = calc_fac_ret(data[features].mean(axis=1), data[ret_name], date_col='date', score_weighted=True)
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
# performance analysis
sns.set_theme()
chars = Chars()
years = Years()
paths = DataPaths()
suffix = 'sub_fac2'
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
# quant_winter_start = 20180101
# quant_winter_end = 20210101
# quant_winter_start = 20210101
# quant_winter_end = 20230101
quant_winter_start = 20060601
quant_winter_end = 20260501
# regions = ['GL', 'US', 'EU', 'UK', 'JP', 'AP', 'EM']
# regions = ['GL', 'US', 'EU', 'UK']
regions = ['GL', ]
# regions = ['JP', 'AP']
for reg in regions:
    print(f'Processing {reg}')
    # features = list(chars.__dict__.values())[:-2]
    features = ['val', 'qual', 'fcf_rank', 'trd', 'sen']
    if reg == "ALL":
        data, CHARAS_LIST, stock_info = read_all_data(target=ret_name, ei_factors=features)
    else:
        data, _, CHARAS_LIST, _, stock_info = read_ei_data(region_=reg, target=ret_name, ei_factors=features)
    data.loc[:, 'lme'] = np.log(data[Columns.size_col])
    data = data.swaplevel(0, 1)
    stock_info = stock_info.swaplevel(0, 1)

    # tree score: SDF node weights tilted towards the stocks most firmly inside each node, rank-normalised per date
    ai_score = pd.read_csv(paths.output / f"{reg}_score_{suffix}.csv").set_index(['date', 'permno'])['size_oriented_norm']
    ai_score.name = 'Tree'
    ei_score = data[features]

    # select quant winter
    dates = data.index.levels[0]
    quant_winter = dates[(dates >= quant_winter_start) & (dates < quant_winter_end)]

    data = data.loc[quant_winter]
    ai_score = ai_score.loc[quant_winter]
    ei_score = ei_score.loc[quant_winter]
    combined_score = ei_score.merge(ai_score, left_index=True, right_index=True, how='left')
    dates = combined_score.index.get_level_values('date').unique()
    valid_dates = ~combined_score['Tree'].groupby('date').apply(lambda x: x.isna().all())
    combined_score = combined_score.loc[dates[valid_dates]]
    ai_pnl = calc_fac_ret(ai_score, data[ret_name], date_col='date', score_weighted=True)
    ei_pnl = calc_fac_ret(ei_score.mean(axis=1), data[ret_name], date_col='date', score_weighted=True)
    combined_pnl = calc_fac_ret(combined_score.mean(axis=1), data[ret_name], date_col='date', score_weighted=True)
    pnl = pd.concat([ai_pnl, ei_pnl, combined_pnl], axis=1).dropna()
    pnl.index = pd.to_datetime(pnl.index, format="%Y%m%d")
    pnl.columns = ['Tree', 'EI', 'Combined']
    pnl.cumsum().plot(title=f'{reg}: Performance').legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    print(summary(pnl, ann_factor=12, sorted=False))
    print(pnl.corr())

    roll_perf = pnl.rolling(12).sum()
    roll_perf.index = pd.to_datetime(roll_perf.index, format="%Y%m%d")
    roll_perf.plot(title=f"{reg}: Rolling 1-year")
    plt.show()

    roll_perf = pnl.rolling(36).sum()
    roll_perf.index = pd.to_datetime(roll_perf.index, format="%Y%m%d")
    roll_perf.plot(title=f"{reg}: Rolling 3-year")
    plt.show()

    factor_pnl = calc_fac_ret(combined_score, data['gross_returns'], date_col='date', score_weighted=True)
    factor_pnl.index = pd.to_datetime(factor_pnl.index, format="%Y%m%d")
    factor_pnl.cumsum().plot(title=f"{reg}: Factor Performance").legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    print(summary(factor_pnl, ann_factor=12, sorted=False))
    print(factor_pnl.corr())

    # tree_grp_exp = pd.concat([ai_score, stock_info['sector']], axis=1, join='inner').groupby('date').apply(lambda x: calc_group_exposure(x.iloc[:, 0], groups=x.iloc[:, 1]))
    # ei_grp_exp = pd.concat([ei_score.mean(axis=1), stock_info['sector']], axis=1, join='inner').groupby('date').apply(lambda x: calc_group_exposure(x.iloc[:, 0], groups=x.iloc[:, 1]))
    # combined_grp_exp = pd.concat([combined_score.mean(axis=1), stock_info['sector']], axis=1, join='inner').groupby('date').apply(lambda x: calc_group_exposure(x.iloc[:, 0], groups=x.iloc[:, 1]))
    tree_grp_exp = grouped_exposure(pd.concat([ai_score, stock_info['sector']], axis=1, join='inner'))
    ei_grp_exp = grouped_exposure(pd.concat([ei_score.mean(axis=1), stock_info['sector']], axis=1, join='inner'))
    combined_grp_exp = grouped_exposure(pd.concat([combined_score.mean(axis=1), stock_info['sector']], axis=1, join='inner'))
    grp_exp = pd.concat([tree_grp_exp, ei_grp_exp, combined_grp_exp], axis=1)
    grp_exp.columns = ['Tree', 'EI', 'Combined']
    grp_exp.groupby('sector').mean().plot(kind='bar', title=f"{reg}: Average Exposure")
    tree_grp_exp_df = tree_grp_exp.unstack()
    tree_grp_exp_df.index = pd.to_datetime(tree_grp_exp_df.index, format="%Y%m%d")
    ax = tree_grp_exp_df.plot(title=f"{reg}: Sector Exposure")
    avg_exp = tree_grp_exp_df.mean()
    labels = [
        f"{col}: {avg_exp[col]*100:.1f}%"
        for col in tree_grp_exp_df.columns
    ]
    ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))

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
    ei_prod = combined_score.iloc[:, :-1].fillna(0).dot(fac_wei)
    prod_alpha = combined_score.iloc[:, :-1].fillna(0).dot(fac_wei)*0.8 + combined_score.iloc[:, -1].fillna(0)*0.2
    new_alpha = combined_score.mean(axis=1)
    alpha = pd.concat([ei_prod, combined_score.iloc[:, -1], ei_alpha, new_alpha, prod_alpha], axis=1)
    alpha.columns = ['ei', 'tree', 'ew_ei', 'ew_all', 'prod']
    alpha.to_csv(f'result/{reg}_all_score_{suffix}.csv')

    r2_tree = pd.concat([ai_score, data['gross_returns'].groupby('permno').shift()], axis=1).dropna().groupby('date').apply(lambda x: r_squared(x.iloc[:, :-1], x.iloc[:, -1]))
    r2_ei = ei_score.merge(data['gross_returns'].groupby('permno').shift(), left_index=True, right_index=True).dropna().groupby('date').apply(lambda x: r_squared(x.iloc[:, :-1], x.iloc[:, -1]))
    r2_combined = combined_score.merge(data['gross_returns'].groupby('permno').shift(), left_index=True, right_index=True).dropna().groupby('date').apply(lambda x: r_squared(x.iloc[:, :-1], x.iloc[:, -1]))
    r2 = pd.concat([r2_tree, r2_ei, r2_combined], axis=1)
    r2.index = pd.to_datetime(r2.index, format="%Y%m%d")
    r2.columns = ['Tree', 'EI', 'Combined']
    ax = r2.plot(title=f"{reg}: R2")
    labels = [
        f"{col}: {r2[col].mean():.2f}"
        for col in r2.columns
    ]
    ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    stats, pvalue = ttest_rel(r2['EI'], r2['Combined'])
    print(f"Stats: {stats}, P-value: {pvalue}")

    tree_residuals = get_residuals(combined_score, factor_names=features, return_name='Tree', date_name='date')
    resid_ic = pd.concat([tree_residuals, data['gross_returns']], axis=1).groupby('date').apply(lambda x: x.corr().iloc[0, 1])
    tree_ic = pd.concat([ai_score, data['gross_returns']], axis=1).groupby('date').apply(lambda x: x.corr().iloc[0, 1])
    alpha_ic = pd.concat([tree_ic, resid_ic], axis=1)
    alpha_ic.index = pd.to_datetime(alpha_ic.index, format="%Y%m%d")
    alpha_ic.columns = ['Tree', 'Residual']
    ax = alpha_ic.plot(title=f"{reg}: IC")
    labels = [
        f"{col}: {alpha_ic[col].mean():.2f}"
        for col in alpha_ic.columns
    ]
    ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

    # ei_ic = pd.concat([ei_score.mean(axis=1), data['gross_returns']], axis=1).groupby('date').apply(lambda x: x.corr().iloc[0, 1])
    # tree_ic = pd.concat([ai_score, data['gross_returns']], axis=1).groupby('date').apply(lambda x: x.corr().iloc[0, 1])
    # combined_ic = pd.concat([combined_score.mean(axis=1), data['gross_returns']], axis=1).groupby('date').apply(lambda x: x.corr().iloc[0, 1])
    # ic = pd.concat([tree_ic, ei_ic, combined_ic], axis=1)
    # ic.index = pd.to_datetime(alpha_ic.index, format="%Y%m%d")
    # ic.columns = ['Tree', 'EI', 'Combined']
    ic = alpha.apply(lambda x: pd.concat([x, data['gross_returns']], axis=1).groupby('date').apply(lambda x: x.corr().iloc[0, 1]))
    ic.index = pd.to_datetime(ic.index, format="%Y%m%d")
    ax = ic.plot(title=f"{reg}: IC")
    labels = [
        f"{col}: {ic[col].mean():.2f}"
        for col in ic.columns
    ]
    ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    stats, pvalue = ttest_rel(ic['ei'], ic['prod'])
    print(f"Stats: {stats}, P-value: {pvalue}")
    print(pd.DataFrame(ic.mean()/ic.std(), columns=["ICIR"]))

    bt_rets = pd.read_excel(rf"J:\Quant\Enhanced Index\Team\Zhen\projects\ap_trees\Backtest\backtest_pickles\better_beta_{reg}_prod_2026-09-08_full_constraints_{suffix}_summary.xlsx", sheet_name="rets_active").set_index("dates")
    bt_rets = bt_rets.loc[quant_winter]
    bt_rets.index = pd.to_datetime(bt_rets.index, format="%Y%m%d")
    ei_alpha = ei_pnl.copy()
    ei_alpha.index = pd.to_datetime(ei_alpha.index, format="%Y%m%d")
    models = bt_rets.columns
    r2_ts = pd.DataFrame(index=models, columns=['tree2EI', 'Tree', 'EI', 'Combined'])
    for m in models:
        r2_ts.loc[m, 'tree2EI'] = r_squared(factor_pnl[features], factor_pnl['Tree'])
        r2_ts.loc[m, 'Tree'] = r_squared(factor_pnl["Tree"], bt_rets[m])
        r2_ts.loc[m, 'EI'] = r_squared(factor_pnl[features], bt_rets[m])
        r2_ts.loc[m, 'Combined'] = r_squared(factor_pnl, bt_rets[m])
        # r2_ts.loc[m, 'Tree'] = r_squared(factor_pnl["Tree"], ei_alpha)
        # r2_ts.loc[m, 'EI'] = r_squared(factor_pnl[features], ei_alpha)
        # r2_ts.loc[m, 'Combined'] = r_squared(factor_pnl, ei_alpha)
    print(r2_ts)
    # r2_ts.to_csv(f"{reg}_r2.csv")
    bt_rets.cumsum().plot(title=f"{reg}")
    plt.show()
    print(summary(bt_rets, sorted=False, ann_factor=12))

    bt_pickle = pd.read_pickle(rf"J:\Quant\Enhanced Index\Team\Zhen\projects\ap_trees\Backtest\backtest_pickles\better_beta_{reg}_prod_2026-09-08_full_constraints_{suffix}.pickle")
    models = dict(zip(bt_pickle.keys(), alpha.columns))
    corr = {} 
    for m, a in models.items():
        alpha_a = alpha[a]
        alpha_a.index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(alpha_a.index.get_level_values('date'), format="%Y%m%d"),
                alpha_a.index.get_level_values('permno')
            ],
            names=bt_pickle[m]['optimal_weight'].index.names
        )
        corr[a] = pd.concat([alpha_a, bt_pickle[m]['optimal_weight']], axis=1, join='inner').groupby('dates').apply(lambda x: x.corr().iloc[0, 1])
    tc = pd.DataFrame(corr)
    ax = tc.plot(title=f"{reg}: TC")
    labels = [
        f"{col}: {tc[col].mean():.2f}"
        for col in tc.columns
    ]
    ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    stats, pvalue = ttest_rel(tc['ei'], tc['prod'])
    print(f"Stats: {stats}, P-value: {pvalue}")

# %%
# sns.set_theme()
# chars = Chars()
# paths = DataPaths()
# reg = "GL"
# ret_name = "Universe Returns"
# groups = ['region', 'ind']
# ai_features = list(chars.__dict__.values())[:-2]
# ai_data = read_big_universe(ret_name=ret_name, features=ai_features, add_groups=True)
# ai_score = pd.read_csv(paths.output / f"{reg}_score_big_univ.csv").set_index(['date', 'permno'])['final_score']
# ai_score.name = 'Tree'
# score_and_ret = pd.concat([ai_data[[ret_name] + groups].swaplevel(0, 1), ai_score], axis=1, join='outer')
# score_and_ret["alpha"] = score_and_ret.groupby(["date"])["Tree"].transform(rank_normalize)
# score_and_ret["region_industry_neutral"] = score_and_ret.groupby(["date", "region", "ind"])["Tree"].transform(rank_normalize)
# alpha_scores = score_and_ret[['alpha', 'region_industry_neutral']]
# pnl = calc_fac_ret(alpha_scores, score_and_ret[ret_name]/100, q=5, date_col='date', score_weighted=True)
# pnl.index = pd.to_datetime(pnl.index, format="%Y%m%d")
# pnl.cumsum().plot()
# plt.show()
# print(summary(pnl, ann_factor=12, sorted=False))
# to = alpha_scores.apply(lambda x: calc_turnover(x, date_col='date', stock_id='permno'))
# to.index = pd.to_datetime(to.index, format="%Y%m%d")
# to.plot()
# plt.show()
# decay = alpha_scores.apply(lambda x: calc_decay(x, date_col='date'))
# decay.plot()
# plt.show()
# factor_scores = ai_data[["by", "vol", "mom_1y1m", "mkt_cap"]].groupby('date').transform(lambda x: rank_normalize(x))
# scores1 = factor_scores.merge(alpha_scores['alpha'], left_index=True, right_index=True)
# factor_exposure1 = scores1.groupby('date').apply(lambda x: x.iloc[:, :-1].apply(lambda y: calc_factor_exposure(x.iloc[:, -1], factor=y)))
# factor_exposure1.index = pd.to_datetime(factor_exposure1.index, format="%Y%m%d")
# avg_exp = factor_exposure1.mean()
# avg_exp.plot(kind='bar')
# ax = factor_exposure1.plot(title=f"{reg}: Factor Exposure: alpha")
# labels = [
#     f"{col}: {avg_exp[col]:.2f}"
#     for col in factor_scores.columns
# ]
# ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
# plt.show()


# scores2 = factor_scores.merge(alpha_scores['region_industry_neutral'], left_index=True, right_index=True)
# factor_exposure2 = scores2.groupby('date').apply(lambda x: x.iloc[:, :-1].apply(lambda y: calc_factor_exposure(x.iloc[:, -1], factor=y)))
# factor_exposure2.index = pd.to_datetime(factor_exposure2.index, format="%Y%m%d")
# avg_exp = factor_exposure2.mean()
# avg_exp.plot(kind='bar')
# ax = factor_exposure2.plot(title=f"{reg}: Factor Exposure: region_industry_neutral")
# labels = [
#     f"{col}: {avg_exp[col]:.2f}"
#     for col in factor_scores.columns
# ]
# ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
# plt.show()

# all_scores = factor_scores.swaplevel(0, 1).merge(alpha_scores, left_index=True, right_index=True, how='left')
# all_pnls = calc_fac_ret(all_scores, ai_data[ret_name].swaplevel(0, 1)/100, q=5, date_col='date')
# all_pnls.index = pd.to_datetime(all_pnls.index, format="%Y%m%d")
# all_pnls.cumsum().plot()
# plt.show()
# print(summary(all_pnls, ann_factor=12, sorted=False))
# decays = all_scores.apply(lambda x: calc_decay(x, date_col='date'))
# decays.plot()
# plt.show()

# %%
