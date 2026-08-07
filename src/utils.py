from typing import List
from pathlib import Path
from tqdm import tqdm
from itertools import product
from src.constants import Columns

import pandas as pd
import polars as pl
import numpy as np


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


def get_ret_val(input_df: pd.DataFrame):
    return np.dot(
        input_df[Columns.returns_col].values,
        input_df[Columns.size_col].values) / input_df[Columns.size_col].sum()


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
            )

            port_dict[port_col] = grouped

        # all_trees[tree_key] = pl.concat(results)
        all_trees[tree_key] = pl.concat([
            df.with_columns(pl.lit(key).alias(Columns.port_col)) for key, df in port_dict.items()
        ], how='vertical')
    return all_trees


def build_tree_portfolio(comb_pl: pl.DataFrame, feature_sequence: List[str], n_split: int = 2, tree_depth: int = 4):
    # comb_pl = pl.from_pandas(comb_df)
    portfolio_dict = tree_portfolio(comb_pl, feature_sequence, n_split, tree_depth)
    portfolio = pl.concat([
            df.with_columns(pl.lit(key).alias(Columns.comb_col)) for key, df in portfolio_dict.items()
        ], how='vertical')
    return portfolio


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


def build_comb_pl(data, merged_df, features):

    # all_features = list({
    #     f
    #     for comb, _, _ in best_combos
    #     for f in comb.split(Columns.col_sep)
    # })

    ranked_data = (
        data.lazy()
        .select([Columns.date_col, Columns.id_col] + features)
        .with_columns([
            (
                (pl.col(f).rank("min").over(Columns.date_col) - 1)
                / (pl.col(f).is_not_null().sum().over(Columns.date_col) - 1)
            ).alias(f)
            for f in features
        ])
        .collect()
    )

    comb_pl = (
        ranked_data
        .join(merged_df, on=[Columns.date_col, Columns.id_col], how="inner")
        .drop_nulls()
        .with_columns(pl.len().over(Columns.date_col).alias("group_size_check"))
        .filter(pl.col("group_size_check") > 100)
        .drop("group_size_check")
    )

    return comb_pl
