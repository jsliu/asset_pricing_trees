# %%
# read factor files
import sys
import pathlib

working_dir = str(pathlib.Path(__file__).parent.parent.parent.parent) + r"\code\\"
sys.path.append(working_dir)
working_dir = str(pathlib.Path(__file__).parent.parent.parent.parent) + r'\code\PortfolioConstruction\src\main\python\\'
sys.path.append(working_dir)
working_dir = str(pathlib.Path(__file__).parent.parent.parent.parent) + r'\code\PortfolioConstruction\src\main\python\quasar\database_sql\\'
sys.path.append(working_dir)

from AlphaWorkshop.src.main.python.alphaworkshop.functions import rank_normalise
from AIalpha.src.main.python.aialpha.file_process import read_factor_data
from connector import DatabaseConnector
from sklearn.impute import KNNImputer

import AIalpha.src.main.python.aialpha.functions as f
import pandas as pd
import numpy as np


def average(y, type='ewm', halflife=None):
    if isinstance(y, pd.Series):
        y = y.to_frame()
    if type=='ewm':
        if halflife is None:
            halflife = 6
        y_avg = y.ewm(halflife=halflife).mean().iloc[-1, :]
    if type=='ma':
        y_avg = y.mean()
    return y_avg

def std(y, type='ewm', halflife=None):
    if isinstance(y, pd.Series):
        y = y.to_frame()
    if type=='ewm':
        if halflife is None:
            halflife = 6
        y_std = y.ewm(halflife=halflife).std().iloc[-1, :]
    if type=='ma' :
        y_std = y.std()
    return y_std

# this is wrong, can't apply wem to geometric accumulative
# def geom_return(y, type='ewm', halflife=None):
#     if isinstance(y, pd.Series):
#         y = y.to_frame()
#     # p = 1 + r
#     p = y.divide(100).add(1)
#     if type == 'ewm':
#         if halflife is None:
#             halflife = 6
#         ewma = p.ewm(halflife=halflife).mean()
#         geom_ret = np.exp(np.log(ewma))
#     if type == 'ma':
#         geom_ret = np.exp(np.log(p))
#     return geom_ret.iloc[-1, :]
    

def orthogonal(x):
    notnan_cols = (x.notna() & x!=0).any(axis=0)
    x2 = x.loc[:, notnan_cols].replace(np.nan, 0.0).to_numpy()
    U, s, V = np.linalg.svd(x2)
    invs = np.diag(1/s)
    t = V.transpose() @ invs @ V
    orthogX = pd.DataFrame(index=x.index, columns=x.columns, dtype="float64")
    orthogX.loc[:, notnan_cols] = x2 @ t
    return orthogX


def readEIfactors(
    region_="EU",
    country=None,
    factor_names=[
        "cap_structure",
        "growth",
        "profitability",
        "accrual",
        "investment",
        "dy_rank",
        "by_rank",
        "fy1_ey_rank",
        "ee_no_fin_rank",
        "fcf_rank",
        "senstock",
        "senind",
        "trdstock",
        "trdind",
    ],
):
    factor_data = pd.read_parquet(
        rf"J:\Quant\Enhanced Index\Research\MATLAB\database\factors_processed\{region_}\factors_processed_backtest.parquet"
    )
    factor_data['dates'] = factor_data.dates.dt.strftime("%Y%m%d").astype(int)
    factor_data = factor_data.set_index(["factset_perm_id", "dates"])
    stock_info_data = pd.read_parquet(
        rf"J:\Quant\Enhanced Index\Research\MATLAB\database\parquet\{region_}\bm_{region_.lower()}_full.parquet"
    )
    stock_info_data['dates'] = stock_info_data.dates.dt.strftime("%Y%m%d").astype(int)
    stock_info = (
        stock_info_data.groupby("dates")
        .apply(
            lambda x: x.drop_duplicates(subset=["factset_perm_id"]),
            include_groups=False,
        )
        .reset_index()
        .set_index(["factset_perm_id", "dates"])
    )
    if country is not None:
        factor_data = factor_data.loc[stock_info['country_exposure']==country]
        stock_info = stock_info.loc[stock_info['country_exposure']==country]
    if factor_names is None:
        ei_factors = factor_data.loc[factor_data['non_inv_trust_mask'].notna(), :]
    else:
        ei_factors = factor_data.loc[factor_data['non_inv_trust_mask'].notna(), factor_names]
    # return ei_factors, stock_info.loc[factor_data['non_inv_trust_mask'].notna(), ["gross_returns", "price_ret_stdev20d", "mkt_cap", "country_exposure"]], factor_data
    ei_factors = ei_factors.rename_axis(index={'factset_perm_id': 'permno', 'dates': 'date'})
    si = stock_info[["gross_returns", "price_ret_stdev20d", "mkt_cap", "country_exposure"]].rename_axis(index={'factset_perm_id': 'permno', 'dates': 'date'})
    return ei_factors, si

def extract_trade_data(data, sub_factors, ret_names=[]):
    # remove stocks don't have market cap data
    data = data[data.mkt_cap.notna()]
    # columns needed for rebalancing
    basic_cols = [
        "Region",
        "sedol",
        "sector",
        "SectorCode",
        "country_exposure",
        "Return",
        "ind",
        "adv_20d",
        "adv_6m",
        "mkt_cap",
        # temporarily remove vol for test
        "vol",
    ]
    basic_data = data[basic_cols].copy()
    basic_data.rename(
        columns={
            "adv_20d": "trading_adv",
            "adv_6m": "holding_adv",
            "mkt_cap": "Market_Capitalization",
            "vol": "ann_vol",
        },
        inplace=True,
    )

    # impute_data = f.group_replace_nan(data, ret_names + sub_factors)
    # ret_scores = f.standardize_factors(impute_data, ret_names, ('dates',), cutoff=3.5)
    # factor_scores = f.standardize_factors(impute_data, sub_factors, ('dates',), cutoff=3.5)
    # zscores = ret_scores.merge(factor_scores, on=['dates', 'sedol'])
    # if ret_names == ['Return']:
    # zscores = data[['dates', 'factset_perm_id'] + sub_factors]
    # trade_data = basic_data.merge(zscores[['dates', 'factset_perm_id'] + sub_factors], on=['dates', 'factset_perm_id'])
    # else:
    zscores = data[sub_factors]
    trade_data = basic_data.merge(
        zscores[sub_factors], left_index=True, right_index=True
    )
    return trade_data

def readAIfactors(
    region_="EU",
    ret_names = ['X1MFwdReturnLoc'],
    ai_drop_factor_names=[
        # duplicated or highly correlated to ei factors:
        #  1.value
        # 'by', 'dy', 'ey', 'ee', 'fy1_ey',
        #  2. stock sentiment
        # 'earn_rev',
        #  3. free cash flow
        # 'fcf',
        #  4. cap_structure
        # 'F_CHG_LEVER_LEVEL', 'F_CHG_LIQUID_LEVEL',
        #  5. investment
        # 'capex', 'capex_trend', 'capex_growth',
        #  6. growth
        # 'F_CHG_MARGIN_LEVEL', 'F_CHG_ROA_LEVEL', 'F_CHG_TURN_LEVEL',
        #  7. accrual
        # 'F_ACCRUAL_LEVEL',

        # other factors which are correlated to each other
        # "by_3y_avg",
        # "dy_3y_avg",
        # "ey_3y_avg",
        # "ee_3y_avg",
        # "earn_rev",
        # "earn_rev_fy2",
        # "mom_1y",
        # "em_12m",
        # "est_eps_ltg",
        # "mom_1m", "mom_3m", "mom_6m", "mom_1y1m", "mom_2y", "mom_3y", "mom_5y", "price_mom_1w",

        # higly correlated quality factors
        # "Total_Debt_Total_Capital",
        # "Common_Equity_Total_Capital",
        # "Pfrd_Stock_Total_Capital",
        # "Sales_Inven_Turnover",
        # "Total_Debt_Equity",
        # "chg_debt",
        # "shares",
        # "Cash_Divid_Cash_Flow",
        # "employees",

        # factors don't make too much sense
        # "Net_Sales_Gross_Fix_Ast",
        # "Payable_Turnover",
        # "Receiv_Turnover",
        "Inven_Turnover",
        # "EST_SALES_GROWTH_FY2",
        # "sales_gr_5y",
        # "eps_gr_5y",
        # "Cash_Flow_Sales",
        "up_fy1_l",
        "dn_fy1_l",
        "tt_fy1_l",
        "up_fy1_s",
        "dn_fy1_s",
        "tt_fy1_s",
        "up_fy2_l",
        "dn_fy2_l",
        "tt_fy2_l",
        "up_fy2_s",
        "dn_fy2_s",
        "tt_fy2_s",
        "up_fy3_l",
        "dn_fy3_l",
        "tt_fy3_l",
        "up_fy3_s",
        "dn_fy3_s",
        "tt_fy3_s",
        # "X12MDwVol",
        # "X12MUpVol",
        # "vol",
        # "price_ret_stdev20d",
        # "X1MVolDaily",
        "vol_1y",
        # "X3MBETALOCALBENCHDaily",

        # "X36MBETAWorld",
        # "X3MILLIQ",
        # "beta_local_3y",
        "price",
        "price_gbp",
        "price_2dayLag",
        # "bm_expanded",
        # "X1MFwdReturnLoc,
        "X3MFwdReturnLoc",
        "X6MFwdReturnLoc",

        # non-numeric factors
        "acquisition_target",
        "currency_iso", 
        "trading_lot_size", 
        # "region_fsreg_gem", 
        "em_dev_flag", 
    ],
    no_sector_neutral=[
        "price_mom_1w",
        "mom_1m",
        "mom_3m",
        "mom_6m",
        "mom_2y",
        "mom_1y1m",
        "earn_rev",
        "em",
        "em_12m",
        # "price",
        "price_ret_stdev20d",
        # "vol",
        # "mkt_cap",
    ],
    standardize=False
):
    # using the data from EI model to make sure they have same benchmark
    stock_info_data = pd.read_parquet(
        rf"J:\Quant\Enhanced Index\Research\MATLAB\database\parquet\{region_}\bm_{region_.lower()}_full.parquet"
    )
    # changes dates to int format YYYYMMDD to adapt for autoencoder paper
    stock_info_data['dates'] = stock_info_data.dates.dt.strftime("%Y%m%d").astype(int)
    stock_info = (
        stock_info_data.groupby("dates")
        .apply(
            lambda x: x.drop_duplicates(subset=["factset_perm_id"]),
            include_groups=False,
        )
        .reset_index()
        .set_index(["dates", "factset_perm_id"])
    )
    stock_returns = stock_info["gross_returns"]
    # get factor data from AI model
    # folder=r'J:\Quant\Enhanced Index\Research\MATLAB\database\parquet\csv'
    # ai_data = read_factor_data(rf'{folder}\mtec4.csv')
    ai_data = read_factor_data(
        r"G:\Quant\Enhanced Index\Research\MATLAB\database\parquet\csv\test_ai_facs.csv"
    ).reset_index()
    # changes dates to int format YYYYMMDD to adapt for autoencoder paper
    ai_data['dates'] = ai_data.dates.dt.strftime("%Y%m%d").astype(int)
    all_data = ai_data.drop(columns=ai_drop_factor_names).set_index(["dates", "factset_perm_id"])
    # factors to be used
    factor_names = [x for x in all_data.drop(columns=['country_exposure']).columns[21:-2] if x not in ret_names]
    trade_data = extract_trade_data(all_data, factor_names)
    # combine with regional stock return data to get regional factor data
    regional_data = trade_data.merge(
        stock_returns[stock_info['uk_inv_trust'] == False], left_index=True, right_index=True, how="right"
    ).dropna(how="all")
    if standardize:
        ai_factors_sn = f.standardize_factors(
            regional_data,
            np.setdiff1d(factor_names, ["X1MFwdReturnLoc"] + no_sector_neutral),
            (
                "dates",
                "sector",
            ),
            cutoff=3.5,
        ).dropna(how="all")
        ai_factors_nsn = f.standardize_factors(
            regional_data, no_sector_neutral, ("dates",), cutoff=3.5
        ).dropna(how="all")
        ai_factors = ai_factors_sn.merge(ai_factors_nsn, left_index=True, right_index=True)
    else:
        ai_factors = regional_data.drop(columns=['Market_Capitalization', 'ann_vol'])
    
    out = ai_factors.loc[ai_factors['X1MFwdReturnLoc'].notna()].swaplevel(0, 1)
    out = out.rename_axis(index={'dates': 'date', 'factset_perm_id': 'permno'})
    return out


def _regression(X, y, scaled=True):
    out = np.empty(X.shape[1])
    out[:] = np.nan
    notna_X = X.notna().any().to_numpy()
    notna_y = y.notna().to_numpy()
    if sum(notna_y) > 0:
        r = y[notna_y]
        A = X.loc[notna_y, notna_X].fillna(0).to_numpy()
        if scaled:
            out[notna_X] = np.linalg.inv(A.T @ A) @ A.T @ r
        else:
            out[notna_X] = A.T @ r
    else:
        out = np.nan
    return pd.Series(out, index=X.columns)


def get_residuals(data, factor_names, return_name="gross_returns", intercept=True, shift=True, **kwargs):
    data2 = data.copy()
    if intercept:
        data2.insert(0, "intercept", 1.0)
        factor_names = ["intercept"] + list(factor_names)
    resids = data2.groupby("dates", group_keys=False).apply(
        lambda x: x[return_name] - x[factor_names].fillna(0) @ _regression(x[factor_names], x[return_name], **kwargs)
    )
    if shift:
        resids = resids.groupby('factset_perm_id', observed=False).shift(1).dropna()
    else:
        resids = resids.dropna()
    return resids


def get_managed_portfolio(
    data, factor_names, return_name="gross_returns", intercept=True, shift=True, **kwargs
):
    data2 = data.copy()
    if intercept:
        data2.insert(0, "intercept", 1.0)
        factor_names = ["intercept"] + list(factor_names)
    man_port = data2.groupby("dates").apply(
        lambda x: _regression(x[factor_names], x[return_name], **kwargs)
    )
    if shift:
        man_port = man_port.shift(1).dropna(how='all')
    else:
        man_port = man_port.dropna(how='all')
    return man_port


def benchmark_backtest(factor_scores, betas, intercept=True):
    idx = pd.IndexSlice
    dates = betas.index
    pred_stk_ret = pd.DataFrame()
    for i, d in enumerate(dates):
        # print(f'Running on {d}')
        if intercept:
            fd = factor_scores.loc[idx[d, :], :].replace(np.nan, 0)
            fd.insert(0, "intercept", 1)
            y_hat = fd @ betas.loc[d, :].repalce(np.nan, 0)
        else:
            y_hat = factor_scores.loc[idx[d, :], :].replace(np.nan, 0) @ betas.loc[d, :].replance(np.nan, 0)
        pred_stk_ret = pd.concat([pred_stk_ret, y_hat.to_frame()])
    return pred_stk_ret

# my version
def mydrawdown(data):
    perf = data.add(1).cumprod()
    peak = perf[0]
    dd = np.zeros(len(perf))
    for i in range(1, len(perf)):
        curr_dd = (perf[i] - peak) / peak
        if perf[i] <= peak:
            dd[i] = curr_dd
        else:
            peak = perf[i]
            dd[i] = 0

    return pd.DataFrame(dd, data.index)


# easier version
def drawdown(rets):
    prices = rets.add(1).cumprod()
    max_price = prices.cummax()
    dd = (prices - max_price) / max_price
    return dd


def _qcut(x, q=5):
    not_na = ~x.isna()
    if np.sum(not_na.to_numpy()):
        qx = x.copy()
        qx[not_na] = pd.qcut(x[not_na], q=q, labels=False, duplicates="drop")
        labels = qx[not_na].unique()
        if max(labels) < q - 1:
            qx[qx == max(labels)] = q - 1
    else:
        qx = pd.Series(np.nan, index=x.index)
    return qx

def _fac_wei(x):
    x_r = rank_normalise(x, cutoff_std=3.5)
    x_r = x_r - np.nanmean(x_r)
    wei = (x_r / np.nansum(np.abs(x_r))) * 2
    return wei


# def _factor_return(factor_, return_, q, date_col="dates", ret_col="gross_returns", long=False, short=False, score_weighted=True):
#     if score_weighted:
#         w = factor_.groupby(date_col).transform(lambda x: _fac_wei(x.to_numpy()))
#         wd = pd.concat([w, return_], axis=1, join="inner").dropna()
#         rets = wd.prod(axis=1).groupby(date_col).sum()
#         return rets    
#     else:
#         qx = factor_.groupby(date_col).transform(_qcut, q=q)
#         fd = pd.concat([qx, return_], axis=1, join="inner")
#         top = fd.loc[fd[qx.name] == q - 1, ret_col].groupby(date_col).mean() - fd[ret_col].groupby(date_col).mean()
#         bottom = fd.loc[fd[qx.name] == 0, ret_col].groupby(date_col).mean() - fd[ret_col].groupby(date_col).mean()

#         if long:
#             return top
    
#         if short:
#             return bottom

#         return top - bottom


def _factor_return(return_, weights, date_col="dates", long=False, short=False):
        excess_return = return_.groupby(date_col).transform(lambda x: x-x.mean())
        wd = pd.concat([weights, excess_return], axis=1, join="inner").dropna()
        rets = wd.prod(axis=1).groupby(date_col).sum()
        
        if long:
            rets = wd.loc[wd.iloc[:, 0] > 0, :].prod(axis=1).groupby(date_col).sum()
    
        if short:
            rets = -wd.loc[wd.iloc[:, 0] < 0, :].prod(axis=1).groupby(date_col).sum()
        
        return rets


def calc_fac_ret(
    factor_score, stock_return, q, date_col="dates", long=False, short=False, score_weighted=True, screen_factor=None
):
    if isinstance(factor_score, pd.Series):
        factor_score = factor_score.to_frame()

    if screen_factor is None:
        weights = factor_score.groupby(date_col).transform(lambda x: _get_weights(x, score_weighted=score_weighted, q=q))    
    else:
        scores = pd.concat([factor_score, screen_factor], axis=1)
        weights = scores.iloc[:, :-1].apply(lambda x: _get_weights_by_screen(x, screen_factor=scores.iloc[:, -1], score_weighted=score_weighted, q=q, date_col=date_col))    
    
    fac_ret = weights.apply(
        lambda x: _factor_return(
            stock_return, x, date_col=date_col, long=long, short=short
        )
    )
    return fac_ret.dropna(how="all")

def _get_weights(alpha, score_weighted=False, q=5):
    if alpha.isna().all() or len(alpha[alpha.notna()].unique()) == 1:
        w=np.empty_like(alpha)
        w[:] = np.nan
    else:
        if score_weighted:
            w = _fac_wei(alpha)
        else:
            qts = _qcut(alpha, q=q)
            w = np.zeros_like(alpha)
            w[qts==np.nanmin(qts)] = -1.0 / sum(qts==np.nanmin(qts))
            w[qts==np.nanmax(qts)] = 1.0 / sum(qts==np.nanmax(qts))
    return w

def _get_weights_by_screen(alpha_factor, screen_factor, score_weighted=True, q=5, date_col='dates'):
    port_weights = alpha_factor.groupby(date_col).transform(lambda x: _get_weights(x, score_weighted=score_weighted, q=q))
    prev_weights = port_weights.groupby('factset_perm_id', observed=True).shift(1).fillna(0)
    delta = port_weights - prev_weights
    qcut = screen_factor.groupby(date_col).transform(lambda x: _qcut(x.rank(method='first'), q=q))
    port_weights.loc[(qcut == 0) & (delta > 0)] = prev_weights.loc[(qcut == 0) & (delta > 0)]
    port_weights.loc[(qcut == q-1) & (delta < 0)] = prev_weights.loc[(qcut == q-1) & (delta < 0)]
    return port_weights
    

def alpha_by_screen(alpha_factor, screen_factor, q=5, date_col='dates'):
    alpha_factor_copy = alpha_factor.fillna(0)
    prev_alpha = alpha_factor_copy.groupby('factset_perm_id', observed=True).shift(1).fillna(0)
    delta = alpha_factor - prev_alpha
    qcut = screen_factor.groupby(date_col).transform(lambda x: _qcut(x.rank(method='first'), q=q))
    alpha_factor_copy.loc[(qcut == 0) & (delta > 0)] = prev_alpha.loc[(qcut == 0) & (delta > 0)]
    alpha_factor_copy.loc[(qcut == q-1) & (delta < 0)] = prev_alpha.loc[(qcut == q-1) & (delta < 0)]
    return alpha_factor_copy

def calc_group_exposure(alpha, groups, group_name='sector'):
    port_weights = _get_weights(alpha)
    exposures = {}
    for g in groups.unique():
        exposures[g] = port_weights[groups==g].sum()
    out = pd.Series(exposures)
    out.index.name = group_name
    return out

def calc_factor_exposure(alpha, factor):
    port_weights = _get_weights(alpha)
    exposures = factor.fillna(0) @ port_weights
    return exposures

def calc_turnover(alpha_factor, date_col='dates', stock_id='factset_perm_id', oneway=True, q=5, score_weighted=True, screen_factor=None):
    """
    Calculate one-way turnover, which is the minimum of buy and sell
    """
    if screen_factor is None:
        port_weights = alpha_factor.groupby(date_col).transform(lambda x: _get_weights(x, score_weighted=score_weighted, q=q))
    else:
        scores = pd.concat([alpha_factor, screen_factor], axis=1)
        port_weights = scores.iloc[:, :-1].apply(lambda x: _get_weights_by_screen(x, screen_factor=scores.iloc[:, -1], score_weighted=score_weighted, q=q, date_col=date_col)).squeeze()    
    
    prev_weights = port_weights.groupby(stock_id, group_keys=False, observed=True).shift(1).fillna(0)
    if oneway:
        to = port_weights.subtract(prev_weights).groupby(date_col).apply(lambda x: min(np.abs(x[x>0].sum()), np.abs(x[x<0].sum())))
    else:
        to = port_weights.subtract(prev_weights).abs().groupby(date_col).sum().squeeze()
    return to

def calc_decay(alpha_factor, date_col='dates', max_lag=12, screen_factor=None):
    dates = alpha_factor.index.get_level_values(date_col).unique()
    decays = np.zeros(max_lag+1)
    decays[0] = 1
    L = len(dates)
    if screen_factor is None:
        port_weights = alpha_factor.groupby(date_col).transform(lambda x: _get_weights(x))
    else:
        scores = pd.concat([alpha_factor, screen_factor], axis=1)
        port_weights = scores.iloc[:, :-1].apply(lambda x: _get_weights_by_screen(x, screen_factor=scores.iloc[:, -1], date_col=date_col)).squeeze()    
    
    for l in range(1, max_lag+1):
        for i in range(L-l):
            corr = pd.concat([port_weights[dates[i]], port_weights[dates[i+l]]], axis=1).corr().iloc[0, 1] 
            if np.isnan(corr):
                corr = 0
            decays[l] += corr / (L-l)
    return pd.Series(decays)



def summary(strategy, ann_factor=252, sorted=True):
    ann_ret = strategy.mean() * ann_factor
    ann_std = strategy.std() * np.sqrt(ann_factor)
    ir = ann_ret / ann_std
    mdd = drawdown(strategy).min()
    stats = pd.concat([ann_ret, ann_std, ir, mdd], axis=1)
    stats.columns = ["Return", "Risk", "IR", "MDD"]
    if sorted:
        result = stats.sort_values(by=["IR"])
    else:
        result = stats
    return result


def trimOutlier(x, sigma=None):
    # date = x.index.get_level_values('dates')[0]
    # print(f'{date}:{x.name}')
    if sigma is None:
        sigma = 4.5
    if ~x.isna().all():
        med = x.median()
        mad = np.abs(x - med).median()
        lower = med - sigma * mad
        upper = med + sigma * mad
        x[(x < lower) | (x > upper)] = np.nan
    return x

def standardize_factors(factor_data, sub_factors, std_func, group=("dates",), **kwargs):
    cat = list(group)
    fdata = factor_data[sub_factors].groupby("dates").transform(trimOutlier)
    impute_data = fdata.groupby("dates", group_keys=False).apply(lambda x: impute_xs(x), include_groups=False).dropna(how='all')
    if len(cat) > 1:
        impute_data = impute_data.merge(factor_data[cat[1:]], left_index=True, right_index=True).reset_index().set_index(list(factor_data.index.names) + cat[1:])
        zscore = impute_data.groupby(cat, observed=True).transform(std_func, **kwargs).reset_index().set_index(factor_data.index.names).drop(columns=cat[1:])
    else:
        zscore = impute_data.groupby(cat, observed=True).transform(std_func, **kwargs)
    # out = pd.concat([factor_data[['dates', 'sedol']], zscore], axis=1)
    return zscore.dropna(how='all')

def impute_xs(data, K=4):
    """
    Cross sectional imputation
    """
    knn = KNNImputer(n_neighbors=K)
    imputed_data = np.full_like(data, np.nan)
    valid_col = data.notna().any()
    if valid_col.any():
        imputed_data[:, valid_col] = knn.fit_transform(data.loc[:, valid_col])
    return pd.DataFrame(imputed_data, index=data.index, columns=data.columns)

def not_in_list(a, b):
    if isinstance(b, list):
        return [element not in b for element in a]
    else:
        return [element != b for element in a]


def read_from_db(factor_names, region_, from_date, to_date):
    if region_ == 'EU':
        r_ = 'EUR'
        fund = 'MSCIEURXUK'
        curr = 'gbp'
    if region_ == 'UK':
        r_ = 'UK'
        fund = 'FTALLSH'
        curr = 'gbp'
    if region_ == 'US':
        r_ = 'NAM'
        fund = 'SAP500'
        curr = 'usd'
    if region_ == 'JP':
        r_ = 'JAP'
        fund = 'MSCIJP'
        curr = 'usd'
    if region_ == 'AP':
        r_ = 'FEAC'
        fund = 'MSAPFXJ'
        curr = 'usd'
    if region_ == 'GL':
        r_ = 'GLO'
        fund = 'MSWRLD'
        curr = 'usd'
    if region_ == 'EM':
        r_ = 'GEM'
        fund = 'MSEMMF'
        curr = 'usd'

    dbc = DatabaseConnector()
    data = dbc.get_df_perf_view_optimal(region=r_, funds=[fund], currs=[curr, 'local'], factors=factor_names, from_date=from_date, to_date=to_date, freq='daily', lags=0)
    data = data[data['factset_perm_id'] != 'Missing']
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index(['date', 'factset_perm_id']).sort_index()
    duplicated = data.index.duplicated(keep='first')
    data = data[~duplicated]
    return data, fund, curr
    
# %%