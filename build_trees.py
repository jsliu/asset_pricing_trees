# %%
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from datetime import date

from src.utils import build_tree_portfolio
from src.tree_portfolio import build_tree_portfolio_best_split
from src.constants import Columns, Chars, DataPaths, Parameters
from src.preprocessing import read_ai_data, read_ei_data, read_db_data, read_big_universe



if __name__ == '__main__':
    chars = Chars()
    paths = DataPaths()

    reg = 'GL'
    data_saved = True
    logging.info(f"Loading base characteristics")
    print(f"Loading base characteristics in {reg}")

    features = list(chars.__dict__.values())[:-2]
    # data, _, CHARAS_LIST, _ = read_ei_data(region_=reg, target=Columns.returns_col, ei_factors=features)
    # data, _, CHARAS_LIST, _ = read_ai_data(region_=reg, data_saved=data_saved, target=Columns.returns_col, mean_features=False)
    # data = read_db_data(region_=reg, features=features, ret_name=Columns.returns_col, data_saved=data_saved)
    data = read_big_universe(ret_name=Columns.returns_col, features=features)
    ret_df = data[Columns.returns_col]
    # we don't have risk free return, but it close to 0, so we ignore 
    # raw_lme_df = read_rename_df(paths.input_data / f"{chars.lme}.csv")
    # lme_df = pd.read_csv(paths.input_data / f"{chars.lme}.csv", names=[Columns.size_col])
    # ret_df = unstack_df(read_rename_df(paths.input_data / f"{paths.returns_file_name}.csv"), paths.returns_file_name)
    # ret_df = pd.read_csv(paths.input_data / f"{paths.returns_file_name}.csv", names=[paths.returns_file_name])
    # rf_factor_df = pd.read_csv(paths.input_data / f"{paths.rf_factor_file_name}.csv", names=[Columns.returns_col])
    # rf_factor_df[Columns.returns_col] = rf_factor_df[Columns.returns_col].apply(lambda x: x / 100)

    logging.info(f"Transform base Size feature into quantiles")
    print(f"Transform base Size feature into quantiles")
    raw_size_df = data[Columns.size_col]
    raw_size_df.name = Columns.size_col
    lme_df = np.log(raw_size_df)
    lme_df.name = chars.lme
    # quantile_lme_df = rows_to_quantiles(raw_lme_df.copy())

    logging.info(f"Stack raw Size and Returns variables together")
    print(f"Stack raw Size and Returns variables together")
    data = pd.concat([data, lme_df,], axis=1)
    merged_df = pd.concat([
        # unstack_df(raw_lme_df, Columns.size_col),
        # unstack_df(quantile_lme_df, chars.lme),
        raw_size_df,
        ret_df], axis=1)

    logging.info(f"Start building the AP trees given the combinations of features")
    print(f"Start building the AP trees given the combinations of features")
    
    # Actually we don't need to always split on size
    for char_comb in tqdm(chars.combinations_of_chars(k=Parameters.n_chars, exclude_chars=[chars.lme, chars.returns])):
        feature_sequence = [chars.lme] + list(char_comb)
        output_file_name = f"{paths.sep}".join(feature_sequence)
        comb_df = data[feature_sequence].groupby('date').transform(lambda x: x.rank(method="min", pct=True))
        comb_df = pd.concat([merged_df, comb_df], axis=1)
        comb_df = comb_df[~(comb_df.isna().sum(axis=1).astype(bool))]
        comb_df.reset_index(inplace=True)

        # In the original implementation, the year start from 1964, thus excluding 1963 from the dataset
        comb_df[Columns.date_col] = pd.to_datetime(comb_df[Columns.date_col], format="%Y%m%d")
        comb_df = comb_df[comb_df[Columns.date_col].apply(lambda x: x.year != 1963)]

        # Start building the tree portfolios
        portfolio = build_tree_portfolio(comb_df, feature_sequence, n_split=Parameters.n_splits, tree_depth=Parameters.tree_depth)
        # portfolio = build_tree_portfolio_best_split(comb_df=comb_df,
        #                                            feature_pool=feature_sequence,
        #                                            report_features=feature_sequence,
        #                                            n_split=Parameters.n_splits, 
        #                                            tree_depth=Parameters.tree_depth,
        #                                            date_col=Columns.date_col,
        #                                            ret_col=Columns.returns_col,
        #                                            w_col=Columns.size_col,
        #                                            mean_shrink=0.05,
        #                                            ridge=1e-4,
        #                                            allow_feature_reuse=True,
        #                                            min_leaf_obs=50,
        #                                            min_T=24,
        # )

        # Get returns excess variable, but we use zero to replace it
        # ret_mask = portfolio.index.get_level_values(Columns.features_col) == Columns.w_returns_col
        # portfolio.iloc[ret_mask, :] = portfolio.iloc[ret_mask, :].sub(
        #     rf_factor_df[portfolio.iloc[ret_mask, :].index.get_level_values(Columns.date_col)].tolist(), axis=0)

        # Remove the trees that are solely based on the single characteristics
        # (all combinations in max port are the same)
        mask_one = (
                portfolio.columns.get_level_values(Columns.port_col) ==
                f"{Columns.port_col}{Columns.col_sep}{Parameters.tree_depth}")
        mask_two = portfolio.columns.get_level_values(Columns.comb_col).isin(
            ['_'.join([v] * Parameters.tree_depth) for v in feature_sequence])
        mask = np.logical_and(mask_one, mask_two)
        portfolio = portfolio.iloc[:, ~mask]
        portfolio.to_pickle(paths.processed_data / f"{output_file_name}.pkl")

# %%
