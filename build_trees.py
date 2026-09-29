# %%
import numpy as np
import pandas as pd
import logging
from itertools import product
import polars as pl
from tqdm import tqdm
# from datetime import date
# from itertools import combinations

from src.utils import build_tree_portfolio, build_comb
from src.constants import Columns, Chars, DataPaths, Parameters, TREE_SETUPS, run_variant
from src.functions import get_residuals
from src.preprocessing import cap_weight, read_db_data, read_big_universe, read_backtest_data




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
    data: pl.DataFrame,
    merged_df: pl.DataFrame,
    exclude_chars: list,
    region: str = None,
    universe: str = None,
    variant: str = None,
    tree_chars: list = None,
    depth: int = Parameters.tree_depth,
):
    """Build and save the tree portfolios; tree_chars limits the characteristics (None = all), depth the tree depth."""
    paths = DataPaths()

    # --------------------------------------------------
    # 1. Generate combinations
    # --------------------------------------------------
    char_combs = list(chars.combinations_of_chars(
        k=Parameters.n_chars,
        exclude_chars=exclude_chars,
        include_chars=tree_chars,
    ))
    char_combs = [(chars.lme, ) + s for s in char_combs]

    # --------------------------------------------------
    # 2. Build full feature list (ONCE)
    # --------------------------------------------------
    all_features = list(set().union(*char_combs))

    # --------------------------------------------------
    # 3. Build comb_pl ONCE (KEY IMPROVEMENT)
    # --------------------------------------------------
    full_comb_pl = build_comb(
        data=data,
        merged_df=merged_df,
        features=all_features,
    )

    # --------------------------------------------------
    # 4. Loop through combinations (lightweight now)
    # --------------------------------------------------
    for char_comb in tqdm(char_combs, desc=f"{paths.label(region, universe)} pipeline"):

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
            tree_depth=depth
        )

        # --------------------------------------------------
        # 6. Filtering logic
        # --------------------------------------------------
        depth_col = f"{Columns.port_col}{Columns.col_sep}{depth}"

        mask_one = pl.col(Columns.port_col) == depth_col
        mask_two = pl.col(Columns.comb_col).is_in(
            [Columns.col_sep.join([v] * depth)
             for v in feature_sequence]
        )
        mask_three = pl.col(Columns.port_col) == f"{Columns.port_col}{Columns.col_sep}0"

        portfolio = portfolio.filter(
            ~((mask_one & mask_two) | mask_three)
        )

        # --------------------------------------------------
        # 7. Save
        # --------------------------------------------------
        portfolio.write_parquet(paths.tree_file(output_file_name, region, universe, variant))


# %%
if __name__ == '__main__':
    chars = Chars()
    # paths = DataPaths()

    # data_saved = False
    # returns column: "gross_returns" in the company data, "ret" in the characteristic files (set per run below)
    paths = DataPaths()
    # company data: regions = ['GL', 'US', ...] (or 'ALL') with universes = [None]
    # characteristic files: regions = [None] with universes = ['full', 'largecap', 'largecap001']
    regions = [None]
    universes = ['largecap', ]
    EQUAL_WEIGHTED = True      # False: value-weighted trees, saved as <comb>_<universe>_vw.parquet
    TREE_TAG = None            # a tree set-up in TREE_SETUPS (src/constants.py), e.g. 'slow3': slow characteristics,
                               # depth 3, saved as <comb>_<universe>_slow3.parquet
    setup = TREE_SETUPS[TREE_TAG]
    for region, universe in product(regions, universes):
        label = paths.label(region, universe)
        ret_name = "gross_returns" if universe is None else "ret"
        logging.info(f"Loading base characteristics")
        print(f"Loading base characteristics in {label}")

        features = list(chars.__dict__.values())[:-2]
        data = read_backtest_data(list(dict.fromkeys(features + (setup['chars'] or []))), ret_name, region=region, universe=universe)
        # data = read_db_data(region_=region, features=features, ret_name=ret_name, data_saved=data_saved)
        # data = read_big_universe(ret_name=ret_name, features=features, features_direction=[1, -1, -1, 1, 1, -1, -1, -1, 1, -1])

        data_pl, ret_and_mcap = prepare_data(data, ret_name=ret_name, factors=None, equal_weighted=EQUAL_WEIGHTED)
        logging.info(f"Start building the trees in {label} given the combinations of features")
        print(f"Start building the trees in {label} given the combinations of features")
        # if alwasy use lme as a feature, exlucde it firslty
        run_pipeline(data_pl, ret_and_mcap, exclude_chars=[chars.returns, chars.lme], region=region, universe=universe,
                     variant=run_variant(EQUAL_WEIGHTED, TREE_TAG), tree_chars=setup['chars'], depth=setup['depth'])
# %%
