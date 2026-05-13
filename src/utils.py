from typing import List
from pathlib import Path
from tqdm import tqdm
from itertools import product
from src.constants import Columns

import pandas as pd
import polars as pl
import numpy as np

import os


def rows_to_quantiles(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Each row contains values - convert them to the quantiles.
    E.g. row [1, 5, 2, 10, 7] will become [0, 0.5, 0.25, 1, 0.75]

    Parameters
    ----------
    input_df

    Returns
    -------

    """
    non_date_cols = input_df.columns.difference([Columns.date_col])
    sum_of_not_nan_per_row = (~input_df[non_date_cols].isna()).sum(axis=1)
    input_df.loc[:, non_date_cols] = input_df[non_date_cols].rank(1)
    # Remove 1 from numerator and denominator in order to receive quantile from 0 to 1, inclusively.
    input_df.loc[:, non_date_cols] = (input_df[non_date_cols] - 1).div(sum_of_not_nan_per_row - 1, axis=0)
    return input_df


def read_rename_df(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise ValueError(f"File {input_path} does not exist.")
    df = pd.read_csv(input_path)
    # Get name of the file without extension in an Upper form
    df.columns = [c.lower() for c in df.columns]
    stem = f"{input_path.stem.lower()}."
    df.columns = [c if stem not in c else c.replace(stem, '') for c in df.columns]
    stem = stem.replace('_', '.')
    df.columns = [c if stem not in c else c.replace(stem, '') for c in df.columns]
    df = process_date_column(df)
    return df


def unstack_df(input_df: pd.DataFrame, name: str) -> pd.DataFrame:
    input_df = input_df.set_index(Columns.date_col).unstack().to_frame(name)
    input_df.index.names = [Columns.permno_col, Columns.date_col]
    input_df = input_df[~input_df[name].isna()]
    return input_df


def process_date_column(input_df: pd.DataFrame, col_name: str = 'date') -> pd.DataFrame:
    # input_df.loc[:, col_name] = pd.to_datetime(input_df[col_name].astype(str), format='%Y%m%d')
    # store original column order
    original_cols = list(input_df.columns)

    # convert date column safely
    col_idx = original_cols.index(col_name)
    dates = pd.to_datetime(input_df[col_name].astype(str), format='%Y%m%d')
    input_df.drop(columns=[col_name], inplace=True)
    input_df.insert(col_idx, col_name, dates)
    return input_df


def get_custom_ranks(input_series: pd.Series) -> pd.Series:
    return input_series.groupby(input_series).cumcount() + input_series


def recursive_tree_grows(input_df: pd.DataFrame, n_split: int, col_idx: int = 0):
    if col_idx >= input_df.shape[1]:
        return input_df
    
    node_col = input_df.columns[col_idx]
    input_df.loc[:, node_col] = pd.qcut(get_custom_ranks(input_df.loc[:, node_col]), n_split, labels=False, duplicates='drop')
    col_idx += 1
    # Recursively apply to each group
    grouped_dfs = []
    for _, group in input_df.groupby(node_col, group_keys=False):
        grouped_dfs.append(recursive_tree_grows(group, n_split, col_idx))
    # Concatenate all groups back together
    return pd.concat(grouped_dfs, axis=0)


# non recursive version
def tree_grows(input_df: pd.DataFrame, n_split: int):
    df = input_df.copy()
    original_cols = list(df.columns)

    for col_idx, node_col in enumerate(original_cols):

        if col_idx == 0:
            df[node_col] = pd.qcut(
                get_custom_ranks(df[node_col]),
                n_split,
                labels=False
            )
        else:
            # split with previous column
            prev_col = original_cols[:col_idx]

            df[node_col] = (
                df
                .groupby(prev_col)[node_col]
                .transform(
                    lambda x: pd.qcut(
                        get_custom_ranks(x),
                        n_split,
                        labels=False
                    )
                )
            )

    return df


def add_portfolio_cols(input_df: pd.DataFrame, feature_sequence: List[str], tree_splits_features: List[str], n_split: int = 2):
    tree_df = input_df[tree_splits_features].copy()
    tree_df.columns = [f"{Columns.node_col}{Columns.col_sep}{i}" for i in range(len(tree_splits_features))]
    
    # tree_df.loc[:, f"{Columns.node_col}{Columns.col_sep}0"] = 0
    # print(tree_splits_features)
    
    tree_df = recursive_tree_grows(tree_df, n_split=n_split)
    # tree_df = tree_grows(tree_df, n_split=n_split)
    
    input_df = input_df.join(tree_df)
    feat_agg_func = {f: ['min', 'max'] for f in set(feature_sequence)}
    features_dict = dict()
    for i_seq in range(len(tree_splits_features)+1):
        port_col = f"{Columns.port_col}{Columns.col_sep}{i_seq}"
        input_df[port_col] = 1
        for k_subseq in range(i_seq):
            input_df.loc[:, port_col] = (input_df[port_col] +
                                         (tree_df[f"{Columns.node_col}{Columns.col_sep}{k_subseq}"]
                                          * (n_split ** (i_seq-k_subseq - 1))))
        min_max_feature_df = input_df.groupby(port_col).agg(feat_agg_func)
        min_max_feature_df.columns = list(map(Columns.col_sep.join, min_max_feature_df.columns.values))
        features_dict[port_col] = pd.concat([
            input_df.groupby(port_col, group_keys=False).apply(lambda sub_df: get_ret_val(sub_df), include_groups=False).to_frame(Columns.w_returns_col),
            min_max_feature_df], axis=1)
    features_df = pd.concat(features_dict, axis=0)
    features_df.columns.name = Columns.features_col
    features_df.index.names = [Columns.port_col, Columns.node_col]
    features_df = features_df.T
    return features_df


def get_ret_val(input_df: pd.DataFrame):
    return np.dot(
        input_df[Columns.returns_col].values,
        input_df[Columns.size_col].values) / input_df[Columns.size_col].sum()


# def tree_portfolio(
#         comb_df: pd.DataFrame, 
#         feature_sequence: List[str], 
#         n_split: int = 2, 
#         tree_depth: int = 4):
#     all_trees_portfolio_dict = dict()
#     for char_product in tqdm(product(feature_sequence, repeat=tree_depth)):
#         # Create grouping by months
#         one_tree = comb_df.groupby(Columns.date_col).apply(
#             lambda x: add_portfolio_cols(x, feature_sequence, list(char_product), n_split))
#         all_trees_portfolio_dict[Columns.col_sep.join(char_product)] = one_tree

#     # Here groupind date/month/port/node and each feature value
#     # In R, for each feature there is a separate matrix with months as Rows, and (port + node) as columns
#     # in the same order as in our data
#     return all_trees_portfolio_dict


def tree_portfolio(
    comb_pl: pl.DataFrame,
    feature_sequence,
    n_split: int,
    tree_depth: int,
    min_node_size: int = 50
) -> pl.DataFrame:
    """
    Build tree portfolio using pure Polars logic.
    """
    all_trees = {}

    for char_prodcut in product(feature_sequence, repeat=tree_depth):
        tree_key = Columns.col_sep.join(char_prodcut)
        # --------------------------------------------------
        # 1. Build tree nodes (hierarchical buckets)
        # --------------------------------------------------

        tree_df = tree_grows_pl(
            comb_pl.select(
                [Columns.date_col] + 
                [
                    pl.col(f).alias(f"{Columns.node_col}{Columns.col_sep}{i}")
                    for i, f in enumerate(char_prodcut)
                ]
            ),
            n_split=n_split,
            date_col=Columns.date_col
        )

        df = comb_pl.with_columns(tree_df)

        # --------------------------------------------------
        # 2. Prepare aggregation expressions
        # --------------------------------------------------
        agg_exprs = []
        for f in set(feature_sequence):
            agg_exprs.append(
                pl.col(f).min().alias(f"{f}{Columns.col_sep}min")
            )
            agg_exprs.append(
                pl.col(f).max().alias(f"{f}{Columns.col_sep}max")
            )

        port_dict = {}

        # --------------------------------------------------
        # 3. Loop over tree depth (portfolio levels)
        # --------------------------------------------------
        for i_seq in range(tree_depth + 1):

            port_col = f"{Columns.port_col}{Columns.col_sep}{i_seq}"
            # port_col = Columns.port_col

            # Build portfolio ID vectorially
            expr = pl.lit(1)
            for k_subseq in range(i_seq):
                node_col = f"{Columns.node_col}{Columns.col_sep}{k_subseq}"
                expr = expr + pl.col(node_col) * (n_split ** (i_seq - k_subseq - 1))

            # df_i = df.with_columns(expr.alias(Columns.node_col))
            df_i = (
                df
                .with_columns(expr.alias(Columns.node_col))
                .with_columns(
                    pl.count().over([Columns.date_col, Columns.node_col]).alias("node_count")
                )
                .filter(pl.col("node_count") >= min_node_size)
            )          

            # --------------------------------------------------
            # 4. Polars-native aggregation (FAST)
            # --------------------------------------------------
            grouped = (
                df_i
                .group_by([Columns.date_col, Columns.node_col])
                .agg([
                    # weighted return (your get_ret_val)
                    (
                        (pl.col(Columns.returns_col) * pl.col(Columns.size_col)).sum()
                        / pl.col(Columns.size_col).sum()
                    ).alias(Columns.w_returns_col),

                    *agg_exprs,
                ])
                # .with_columns(
                #     pl.lit(i_seq).alias(Columns.node_col)
                # )
            )

            port_dict[port_col] = grouped

        # all_trees[tree_key] = pl.concat(results)
        all_trees[tree_key] = pl.concat([
            df.with_columns(pl.lit(key).alias(Columns.port_col)) for key, df in port_dict.items()
        ], how='vertical')
    return all_trees


def build_tree_portfolio(comb_df: pd.DataFrame, feature_sequence: List[str], n_split: int = 2, tree_depth: int = 4):
    comb_pl = pl.from_pandas(comb_df)
    portfolio_dict = tree_portfolio(comb_pl, feature_sequence, n_split, tree_depth)
    portfolio = pl.concat([
            df.with_columns(pl.lit(key).alias(Columns.comb_col)) for key, df in portfolio_dict.items()
        ], how='vertical')
    # portfolio = pd.concat(portfolio_dict, axis=1).T.drop_duplicates().T
    # portfolio.index.names = [Columns.date_col, Columns.features_col]
    # portfolio.columns.names = [Columns.comb_col, Columns.port_col, Columns.node_col]
    return portfolio


# def add_portfolio_cols_pl(
#     df: pl.DataFrame,
#     feature_sequence,
#     tree_splits_features,
#     n_split: int = 2,
# ) -> pl.DataFrame:

#     # --- 1. Build tree_df ---
#     tree_df = (
#         df.select(tree_splits_features)
#         .rename({
#             f: f"{Columns.node_col}{Columns.col_sep}{i}"
#             for i, f in enumerate(tree_splits_features)
#         })
#     )

#     tree_df = tree_grows_pl(tree_df, n_split=n_split)

#     df = df.with_columns(tree_df)

#     # --- 2. Prepare feature aggregations ---
#     agg_exprs = []
#     for f in set(feature_sequence):
#         agg_exprs.append(pl.col(f).min().alias(f"{f}{Columns.col_sep}min"))
#         agg_exprs.append(pl.col(f).max().alias(f"{f}{Columns.col_sep}max"))

#     result_frames = []

#     # --- 3. Loop over portfolio depth ---
#     for i_seq in range(len(tree_splits_features) + 1):

#         port_col = f"{Columns.port_col}{Columns.col_sep}{i_seq}"

#         # Build portfolio id vectorially
#         expr = pl.lit(1)
#         for k_subseq in range(i_seq):
#             node_col = f"{Columns.node_col}{Columns.col_sep}{k_subseq}"
#             expr = expr + pl.col(node_col) * (n_split ** (i_seq - k_subseq - 1))

#         df_i = df.with_columns(expr.alias(port_col))

#         # --- 4. Aggregations (Polars-native) ---
#         grouped = (
#             df_i
#             .group_by(port_col)
#             .agg([
#                 (
#                     (pl.col(Columns.returns_col) * pl.col(Columns.size_col)).sum()
#                     / pl.col(Columns.size_col).sum()
#                 ).alias(Columns.w_returns_col),
#                 *agg_exprs,
#             ])
#         )

#         result_frames.append(grouped)

#    # --- 5. Combine results ---
#   return pl.concat(result_frames)

def quantile_bucket(expr: pl.Expr, n_split: int, groups) -> pl.Expr:
    return(
        (expr.rank("average").over(groups) / (pl.count().over(groups) + 1))
        .mul(n_split)
        .floor()
        .clip(0, n_split - 1)
        .cast(pl.Int32)
    )


def tree_grows_pl(
    df: pl.DataFrame,
    n_split: int,
    date_col: str,
) -> pl.DataFrame:
    
    out = df.clone()

    node_cols = [c for c in out.columns if c != date_col]
    for i, col in enumerate(node_cols):

        if i == 0:
            # Global split
            out = out.with_columns(
                (
                    quantile_bucket(
                        pl.col(col),
                        n_split,
                        groups=date_col
                    )
                ).alias(col)
            )

        else:
            # Split within previous path using window functions
            prev_cols = [date_col] + node_cols[:i]

            out = out.with_columns(
                (
                    quantile_bucket(
                        pl.col(col),
                        n_split,
                        groups=prev_cols
                    )
                ).alias(col)
            )

    return out