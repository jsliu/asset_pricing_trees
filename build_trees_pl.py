# %%
import numpy as np
import pandas as pd
import logging
import polars as pl
from tqdm import tqdm
# from datetime import date
# from itertools import combinations

from src.utils import build_tree_portfolio, build_comb_pl
from src.constants import Columns, Chars, DataPaths, Parameters
from src.functions import get_residuals
from src.preprocessing import cap_weight, read_ei_data, read_db_data, read_big_universe, read_all_data, read_char_data




def prepare_data(data, ret_name, factors=None, equal_weighted=True):
    chars = Chars()
    print(f"Transform base Size feature into quantiles")
    # wei_df = pd.Series(1, index=data.index) if equal_weighted else data[Columns.size_col].groupby('date').transform(lambda x: cap_weight(x))
    wei_df = pd.Series(1, index=data.index) if equal_weighted else data[Columns.size_col]
    wei_df.name = Columns.size_col
    lme_df = -np.log(data[Columns.size_col])
    lme_df.name = chars.lme
    ret_df = data[ret_name] - pd.concat([data[ret_name], wei_df], axis=1).groupby(Columns.date_col).apply(lambda x: x.prod(axis=1).sum() / x[Columns.size_col].sum())
    ret_df.name = Columns.returns_col

    if factors is not None:
        regress_data = data[factors].merge(ret_df, right_index=True, left_index=True)
        ret_df = get_residuals(regress_data, factor_names=factors, return_name=Columns.returns_col, date_name=Columns.date_col, id_name=Columns.id_col)

    print(f"Stack raw Size and Returns variables together")
    data_pl = pl.from_pandas(pd.concat([data.drop(columns=[ret_name]), lme_df], axis=1).reset_index())
    merged_df = pl.from_pandas(pd.concat([wei_df, ret_df], axis=1).reset_index())
    return data_pl, merged_df

def run_pipeline(
    reg: str,
    data: pl.DataFrame,
    merged_df: pl.DataFrame,
    exclude_chars: list,
):
    paths = DataPaths()

    # --------------------------------------------------
    # 1. Generate combinations
    # --------------------------------------------------
    char_combs = list(chars.combinations_of_chars(
        k=Parameters.n_chars,
        exclude_chars=exclude_chars,
    ))
    char_combs = [(chars.lme, ) + s for s in char_combs]

    # --------------------------------------------------
    # 2. Build full feature list (ONCE)
    # --------------------------------------------------
    all_features = list(set().union(*char_combs))

    # --------------------------------------------------
    # 3. Build comb_pl ONCE (KEY IMPROVEMENT)
    # --------------------------------------------------
    full_comb_pl = build_comb_pl(
        data=data,
        merged_df=merged_df,
        features=all_features,
    )

    # --------------------------------------------------
    # 4. Loop through combinations (lightweight now)
    # --------------------------------------------------
    for char_comb in tqdm(char_combs, desc=f"{reg} pipeline"):

        feature_sequence = list(char_comb)
        output_file_name = f"{paths.sep}".join(feature_sequence)

        # only select relevant columns (FAST)
        comb_pl = full_comb_pl.select(
            [Columns.date_col, Columns.id_col, Columns.returns_col, Columns.size_col] + feature_sequence
        )

        # --------------------------------------------------
        # 5. Build tree portfolios
        # --------------------------------------------------
        portfolio = build_tree_portfolio(
            comb_pl,
            feature_sequence,
            n_split=Parameters.n_splits,
            tree_depth=Parameters.tree_depth
        )

        # --------------------------------------------------
        # 6. Filtering logic
        # --------------------------------------------------
        depth_col = f"{Columns.port_col}{Columns.col_sep}{Parameters.tree_depth}"

        mask_one = pl.col(Columns.port_col) == depth_col
        mask_two = pl.col(Columns.comb_col).is_in(
            [Columns.col_sep.join([v] * Parameters.tree_depth)
             for v in feature_sequence]
        )
        mask_three = pl.col(Columns.port_col) == f"{Columns.port_col}{Columns.col_sep}0"

        portfolio = portfolio.filter(
            ~((mask_one & mask_two) | mask_three)
        )

        # --------------------------------------------------
        # 7. Save
        # --------------------------------------------------
        portfolio.write_parquet(
            paths.processed_data / f"{reg}_{output_file_name}.parquet"
        )


# %%
if __name__ == '__main__':
    chars = Chars()
    # paths = DataPaths()

    # data_saved = False
    # ret_name = "Universe Returns"
    # ret_name = "gross_returns"
    ret_name = "ret"
    # regions = ['US', 'UK', 'EU', 'AP', 'JP', 'EM']
    # regions = ['US', ]
    regions = ['full', ]
    for reg in regions:
        logging.info(f"Loading base characteristics")
        print(f"Loading base characteristics in {reg}")

        features = list(chars.__dict__.values())[:-2]
        if reg == "ALL":
            data, CHARAS_LIST, _ = read_all_data(target=ret_name, ei_factors=features)
            data1 = []
            data2 = {}
            for reg2 in  ['GL', 'US', 'UK', 'EU', 'AP', 'JP', 'EM']:
                data2[reg2], _, CHARAS_LIST, _, _ = read_ei_data(region_=reg2, target=ret_name, ei_factors=features)
                data1.append(data2[reg2])
            data = pd.concat(data1)
            data = data.loc[~data.index.duplicated()]
        elif reg in ("full", "largecap", "largecap001"):
            data = read_char_data(features, universe=None if reg == "full" else reg, ret_name=ret_name)
        else:
            data, _, CHARAS_LIST, _, _ = read_ei_data(region_=reg, target=ret_name, ei_factors=features)
        # data = read_db_data(region_=reg, features=features, ret_name=ret_name, data_saved=data_saved)
        # data = read_big_universe(ret_name=ret_name, features=features, features_direction=[1, -1, -1, 1, 1, -1, -1, -1, 1, -1])

        data_pl, ret_and_mcap = prepare_data(data, ret_name=ret_name, factors=None, equal_weighted=True)
        logging.info(f"Start building the trees in {reg} given the combinations of features")
        print(f"Start building the trees in {reg} given the combinations of features")
        # if alwasy use lme as a feature, exlucde it firslty
        run_pipeline(reg, data_pl, ret_and_mcap, exclude_chars=[chars.returns, chars.lme])
# %%
