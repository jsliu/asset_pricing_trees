# data prepropcessing
import pandas as pd
import numpy as np

from datetime import date
from sklearn.preprocessing import scale, LabelEncoder
from src.functions import readAIfactors, readEIfactors, read_from_db
from src.constants import DataPaths

def read_ai_data(region_, data_saved, end_date=None, target='X1MFwdReturnLoc', mean_features=False):
    # region_ = 'US'
    # target = 'X1MFwdReturnLoc'
    # data_saved = False
    path = DataPaths()
    if data_saved:
        data_nn = pd.read_parquet(path.input_data / f"{region_}.parquet")
        all_dates = data_nn.index.get_level_values('date').unique()
        if end_date is not None:
            data_nn = data_nn.loc[all_dates[all_dates <= end_date]]
        features = list(data_nn)
        features.remove(target)
        features.remove('ind_code')
        features.remove('sec_code')
    else:
        le = LabelEncoder()
        ai_data = readAIfactors(region_=region_, ret_names=[target,])
        all_dates = ai_data.index.get_level_values('date').unique()
        if end_date is not None:
            ai_data = ai_data.loc[all_dates[all_dates <= end_date]]
        ind_code = pd.Series(le.fit_transform(ai_data['ind']), index=ai_data.index, name='ind_code')
        # sec_code = pd.Series(le.fit_transform(ai_data['sector']), index=ai_data.index, name='sec_code')
        sec_code = ai_data['SectorCode']
        sec_code.name = 'sec_code'
        country_code = pd.Series(le.fit_transform(ai_data['country_exposure']), index=ai_data.index, name='country_code')
        region_code = pd.Series(le.fit_transform(ai_data['region_fsreg_gem']), index=ai_data.index, name='region_code')

        # split as train and test so that we can create labels in train data
        # dates = ai_data.index.get_level_values('dates').unique()    
        # train_dates, test_dates = dates[:-1], dates[-1]
        # data4train, data4test = ai_data.loc[train_dates], ai_data.loc[[test_dates]]

        # create lables
        # q_ret = data4train[target].groupby('dates').transform(lambda x: pd.qcut(x, q=q, labels=False))
        # idx = (q_ret == 0) | (q_ret == q-1)
        # data4train = data4train[idx]
        # change to use 8 if you removed vol factor, otherwise 9
        if target == 'X1MFwdReturnLoc':
            features = list(ai_data.drop(columns=[target, 'gross_returns', 'region_fsreg_gem']).columns[7:])
        if target == 'gross_returns':
            features = list(ai_data.drop(columns=[target, 'X1MFwdReturnLoc', 'region_fsreg_gem']).columns[7:])

        # labels = q_ret[idx]
        # labels[labels==q-1] = 1
        # labels.name = label
        # n_data = standardize_factors(pd.concat([data4train, data4test]), features, std_func=minmax_scale, group=('dates', ), feature_range=(-1, 1))
        # standardize returns
        # returns = ai_data[target].groupby('dates').transform(lambda x: scale(x, with_std=False))
        returns = ai_data[target]
        returns = returns[returns != 0]

        # out_y = pd.concat([returns, labels], axis=1, join='inner')
        
        data_nn = ai_data[features].merge(pd.concat([ind_code, sec_code, country_code, region_code], axis=1), left_index=True, right_index=True, how='right')
        data_nn = data_nn.merge(returns, left_index=True, right_index=True, how='left')
        # all_data = pd.concat([data_nn, data4test[features].merge(ind_code[test_dates], right_index=True, left_index=True)])
        data_nn.to_parquet(path.input_data / f"{region_}.parquet")
        features = list(data_nn)
        features.remove(target)
        features.remove('ind_code')
        features.remove('sec_code')
        features.remove('country_code')
        features.remove('region_code')
    
    if mean_features:
        data_nn, features = add_mean_features(data_nn, features, target)
    
    dates = data_nn.index.get_level_values('date').unique()    
    train_dates, test_dates = dates[:-1], dates[-1]
    # currently i put all the data as training data
    # train_dates, test_dates = dates, dates[-1]
    idx = pd.IndexSlice
    train = data_nn.loc[idx[:, train_dates], :]
    test = data_nn.loc[idx[:, test_dates], :]
    
    # remove na
    train.loc[:, features] = train[features].fillna(train[features].mean())
    test.loc[:, features] = test[features].fillna(test[features].mean())
    return train, test, features, test_dates

def create_labels(stock_returns, q=3):
    # q_ret = stock_returns.groupby('dates').transform(lambda x: pd.qcut(x, q=q, labels=False))
    q_ret = pd.qcut(stock_returns, q=q, labels=False)
    qs = q_ret[(q_ret==0) | (q_ret==q-1)]
    qs[qs==q-1] = 1
    return qs

def create_weights(region_, rebal_date, data, ts=False):
    similarity = pd.read_csv(rf'G:\Quant\Enhanced Index\Research\Zhen\projects\mlp\sim_dist_daily_return\{region_}_{rebal_date.strftime("%Y%m")}.csv')
    # similarity = pd.read_csv(rf'C:\Users\zhen.liu\Projects\MachineLearning\sim_dist_daily_return\{region_}_{rebal_date.strftime("%Y%m")}.csv')
    similarity['dates2'] = pd.to_datetime(similarity['dates']).dt.to_period('M')
    # similarity.set_index('dates', inplace=True)
    data2 = data.copy()
    data2.reset_index(inplace=True)
    dates = data2['dates']
    data2['dates2'] = dates.dt.to_period('M')
    weights = data2.merge(similarity.drop(columns='dates'), on='dates2', how='left')[['dates', 'dates2', 'factset_perm_id', 'similarity']]
    # weights.loc[:, 'dates'] = dates
    weights.drop(columns=['dates2'], inplace=True)
    weights.rename(columns={'similarity': 'weights'}, inplace=True)
    weights.set_index(['dates', 'factset_perm_id'], inplace=True)
    if ts:
        weights = weights.groupby('dates', group_keys=False).apply(lambda x: x.mean())
    return weights.dropna().squeeze()

def create_old_weights(region_, rebal_date, data, ts=False):
    similarity = pd.read_csv(rf'G:\Quant\Enhanced Index\Research\Zhen\mtec_data\factor_sim_dist_backtest_market_return_macro\{region_}_{rebal_date.strftime("%Y%m")}.csv')
    # similarity = pd.read_csv(rf'G:\Quant\Enhanced Index\Research\Zhen\projects\mlp\sim_dist\{region_}_{rebal_date.strftime("%Y%m")}.csv')
    similarity['dates2'] = pd.to_datetime(similarity['dates']).dt.to_period('M')
    similarity.columns = ['dates', 'similarity', 'dates2']
    # similarity.set_index('dates', inplace=True)
    data.reset_index(inplace=True)
    dates = data['dates']
    data['dates2'] = dates.dt.to_period('M')
    weights = data.merge(similarity.drop(columns='dates'), on='dates2', how='left')[['dates', 'dates2', 'factset_perm_id', 'similarity']]
    # weights.loc[:, 'dates'] = dates
    weights.drop(columns=['dates2'], inplace=True)
    weights.rename(columns={'similarity': 'weights'}, inplace=True)
    weights.set_index(['dates', 'factset_perm_id'], inplace=True)
    if ts:
        weights = weights.groupby('dates', group_keys=False).apply(lambda x: x.mean())
    return weights.dropna().squeeze()

def add_mean_features(data, features, target, group_code='ind_code'):
    data_features = data[features]
    mean_features = data_features.groupby('factset_perm_id', observed=True).transform(lambda x: x.rolling(window=12, min_periods=6).mean()).dropna(how='all')
    var_features = (data_features - mean_features).dropna(how='all')
    mean_features.columns = ['mean_' + f for f in features]
    var_features.columns = ['var_' + f for f in features]
    if group_code is None:
        new_data = pd.concat([mean_features, var_features, data[target]], axis=1, join='inner')
    else:
        new_data = pd.concat([mean_features, var_features, data[['ind_code', target]]], axis=1, join='inner')
    new_features = list(mean_features.columns) + list(var_features.columns)
    return new_data, new_features


def dataframe_to_2d_array(df, to_array=False):
    """
    Convert a multilevel index DataFrame with index (n_time, n_samples) and n_factors columns to a 2D array.
    """
    # Pivot the DataFrame to get a 2D array (n_samples, n_time * n_factors)
    df_pivot = df.unstack(level=0)
    
    # Convert the DataFrame to a NumPy array
    if to_array:
        data_array = df_pivot.values
    else:
        data_array = df_pivot
    
    return data_array


def dataframe_to_3d_array(df):
    """
    Pivot the DataFrame to get a 3D array (n_time, n_samples, n_factors)
    """
    df_pivot = df.unstack()
    
    # Convert the DataFrame to a NumPy array
    n_factors = len(df.columns)
    n_time = len(df.index.levels[0])
    n_samples = len(df.index.levels[1])
    # adjust dimension
    data_array = df_pivot.values.reshape(n_time, n_factors, n_samples).transpose(2, 0, 1)
    
    return data_array


def fill_daily_na(data, factors):
    """
    we are filling the dailyd data, so the missing data is filled with monthly average
    """
    data.loc[:, 'yearmonth'] = data['date'].dt.to_period('M')
    data = data.set_index(['yearmonth', 'factset_perm_id'])
    data.loc[:, factors] = data[factors].groupby(['yearmonth', 'factset_perm_id']).transform(lambda x: x.fillna(x.mean()))
    return data.reset_index().drop(columns=['yearmonth'])
        

def read_ei_data(region_, end_date=None, target='gross_returns', ei_factors=None):    
    if ei_factors:
        factor_data, stock_info, = readEIfactors(region_, factor_names=ei_factors)
        features = ei_factors
    else:
        factor_data, stock_info, = readEIfactors(region_)
        features = factor_data.columns
    # returns = stock_info[target].groupby('dates').transform(lambda x: scale(x, with_std=False))
    ret_and_size = stock_info[[target, "mkt_cap"]]
    data = factor_data[features].merge(ret_and_size, left_index=True, right_index=True, how='inner')
    
    all_dates = data.index.get_level_values('date').unique()   
    if end_date is not None:
        data = data.loc[all_dates[all_dates <= end_date]]
        dates = data.index.get_level_values('date').unique()
    else:
        dates = all_dates
    train_dates, test_dates = dates[:-1], dates[-1]
    idx = pd.IndexSlice
    train = data.loc[idx[:, train_dates], :]
    test = data.loc[idx[:, test_dates], :]

    # remove stocks don't have returns
    train_no_na = train.loc[train[target].notna(), :]
    # # valid_date = train_no_na.groupby('date').apply(lambda x: x.apply(lambda y: y.notna().sum()/len(y) > 0.5)).prod(axis=1)
    # vd_idx = valid_date[valid_date==1].index
    # valid_train = train_no_na.loc[idx[:, vd_idx], :]
    # valid_train = valid_train.fillna(valid_train.groupby('date').mean())
    valid_train = train_no_na
    return valid_train, test, list(features) + ['mkt_cap'], test_dates


def calculate_rolling_ir(returns, lookback_period, ann_factor=12):
    rolling_mean = returns.rolling(window=lookback_period).mean() * ann_factor
    rolling_std = returns.rolling(window=lookback_period).std() * np.sqrt(ann_factor)
    rolling_ir = rolling_mean / rolling_std
    return rolling_ir


def read_db_data(region_, features, ret_name, freq='M', data_saved=True):
    path = DataPaths()
    if data_saved:
        data = pd.read_parquet(path.input_data / f"{region_}_db.parquet")
    else:
        data, fund, curr = read_from_db(features, region_=region_, from_date='20090101', to_date=date.today().strftime("%Y%m%d"))
        data.to_parquet(path.input_data / f"{region_}_db.parquet")
    
    data = data.rename_axis(index={'factset_perm_id': 'permno'})
    data = data.swaplevel(0, 1)
    if freq != 'D':
        if freq == "M":
            window = 20
        elif freq == "W":
            window = 5
        data[ret_name] = data[ret_name].groupby('permno', group_keys=False).apply(lambda x: x.rolling(window).sum().shift(-window+1), include_groups=False)

    return data[data[ret_name].notna()]


def read_big_universe(features, ret_name="X1MFwdReturnLoc"):
    data = pd.read_csv(r"G:\Quant\Enhanced Index\Research\MATLAB\database\parquet\csv\mtec4.csv",
                       na_values=["NA", "NAN"],
                       keep_default_na=True).set_index(['Period (YYYYMMDD)', 'factset_perm_id'])
    data = data[features + ['mkt_cap', ret_name]].swaplevel(0, 1)
    data = data.rename_axis(index = {'Period (YYYYMMDD)': 'date', 'factset_perm_id': 'permno'})
    dates = data.index.levels[1]
    idx = pd.IndexSlice
    return data.loc[idx[:, dates[:-2]], :]
