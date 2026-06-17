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
from src.preprocessing import cap_weight, read_ei_data, read_db_data, read_big_universe


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

    # --------------------------------------------------
    # 2. Build full feature list (ONCE)
    # --------------------------------------------------
    all_features = list(set().union(*char_combs))

    # --------------------------------------------------
    # 3. Build comb_pl ONCE (KEY IMPROVEMENT)
    # --------------------------------------------------
    # # Convert char_combs into "best_combos" format expected by build_comb_pl
    # # (we only use feature strings inside)
    # best_combos = [
    #     (Columns.col_sep.join(comb), None, None)
    #     for comb in char_combs
    # ]

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


'''
def run_pipeline(reg: str, data: pl.DataFrame, merged_df: pl.DataFrame, exclude_chars: list):

    # chars = Chars()
    paths = DataPaths()
    char_combs = list(chars.combinations_of_chars(
        k=Parameters.n_chars,
        exclude_chars=exclude_chars,
    ))

    all_features = list(set().union(*char_combs))

    ranked_data = (
        data.lazy()
        .select([Columns.date_col, Columns.id_col] + all_features)
        .with_columns([
            (
                (pl.col(f).rank("min").over(Columns.date_col) - 1)
                / (pl.col(f).count().over(Columns.date_col) - 1)
            ).alias(f)
            for f in all_features
        ])
        .collect()
    )

    for char_comb in tqdm(char_combs):

        feature_sequence = list(char_comb)
        output_file_name = f"{paths.sep}".join(feature_sequence)

        comb_pl = (
            ranked_data
            .select([Columns.date_col, Columns.id_col] + feature_sequence)
            .join(merged_df, on=[Columns.date_col, Columns.id_col], how="inner")
            .drop_nulls()
            .with_columns(pl.len().over(Columns.date_col).alias("group_size_check"))
            .filter(pl.col("group_size_check") > 100)
            .drop("group_size_check")
        )
        
        portfolio = build_tree_portfolio(
            comb_pl,
            feature_sequence,
            n_split=Parameters.n_splits,
            tree_depth=Parameters.tree_depth
        )

        depth_col = f"{Columns.port_col}{Columns.col_sep}{Parameters.tree_depth}"

        mask_one = pl.col(Columns.port_col) == depth_col
        mask_two = pl.col(Columns.comb_col).is_in(
            [Columns.col_sep.join([v] * Parameters.tree_depth) for v in feature_sequence]
        )
        mask_three = pl.col(Columns.port_col) == f"{Columns.port_col}{Columns.col_sep}0"
        portfolio = portfolio.filter(~((mask_one & mask_two) | mask_three))

        portfolio.write_parquet(
            paths.processed_data / f"{reg}_{output_file_name}.parquet"
        )
'''
# %%
if __name__ == '__main__':
    chars = Chars()
    # paths = DataPaths()

    data_saved = False
    regions = ['GL', 'US', 'UK', 'EU', 'AP', 'JP', 'EM']
    # regions = ['GL', ]
    for reg in regions:
        logging.info(f"Loading base characteristics")
        print(f"Loading base characteristics in {reg}")

        features = list(chars.__dict__.values())[:-2]
        data, _, CHARAS_LIST, _ = read_ei_data(region_=reg, target='gross_returns', ei_factors=features)
        # data = read_db_data(region_=reg, features=features, ret_name=Columns.returns_col, data_saved=data_saved)
        # data = read_big_universe(ret_name='gross_returns', features=features)
        ret_df = data['gross_returns'] - data[['gross_returns', Columns.size_col]].groupby(Columns.date_col).apply(lambda x: x.prod(axis=1).sum() / x[Columns.size_col].sum())
        ret_df.name = Columns.returns_col

        logging.info(f"Transform base Size feature into quantiles")
        print(f"Transform base Size feature into quantiles")
        raw_size_df = data[Columns.size_col].groupby('date').transform(lambda x: cap_weight(x))
        raw_size_df.name = Columns.size_col
        lme_df = np.log(data[Columns.size_col])
        lme_df.name = chars.lme

        logging.info(f"Stack raw Size and Returns variables together")
        print(f"Stack raw Size and Returns variables together")
        data = pl.from_pandas(pd.concat([data.drop(columns=['gross_returns']), lme_df], axis=1).reset_index())
        merged_df = pl.from_pandas(pd.concat([raw_size_df, ret_df], axis=1).reset_index())

        logging.info(f"Start building the trees in {reg} given the combinations of features")
        print(f"Start building the trees in {reg} given the combinations of features")
        run_pipeline(reg, data, merged_df, exclude_chars=[chars.returns, ])
# %%
