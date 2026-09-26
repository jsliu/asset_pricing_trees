# %%
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from build_trees_pl import prepare_data
from src.utils import tree_grows_pl
from src.constants import Columns, Chars
from src.preprocessing import read_ei_data, read_big_universe

def run_pipeline(data: pl.DataFrame, merged_df: pl.DataFrame, best_combos: list):

    # chars = Chars()
    # paths = DataPaths()

    all_features = list({
        f
        for comb, _, _ in best_combos
        for f in comb.split(Columns.col_sep)
    })

    ranked_data = (
        data.lazy()
        .select([Columns.date_col, Columns.id_col] + all_features)
        .with_columns([
            (
                (pl.col(f).rank("min").over(Columns.date_col) - 1)
                / (pl.col(f).is_not_null().sum().over(Columns.date_col) - 1)
            ).alias(f)
            for f in all_features
        ])
        .collect()
    )
    
    comb_pl = (
        ranked_data
        .select([Columns.date_col, Columns.id_col] + all_features)
        .join(merged_df, on=[Columns.date_col, Columns.id_col], how="inner")
        .drop_nulls()
        .with_columns(pl.len().over(Columns.date_col).alias("group_size_check"))
        .filter(pl.col("group_size_check") > 100)
        .drop("group_size_check")
    )

    for comb, port, node in tqdm(best_combos):

        df_node = get_stocks_in_node(
            comb_pl=comb_pl,
            tree_key=comb,
            port_col=port,
            node_id=node,
            n_split=2
        )

        # df_node = get_continuous_hierarchical_exposure(
        #     comb_pl=comb_pl,
        #     tree_key=comb,
        #     port_col=port,
        #     node_id=node,
        #     n_split=2,
        #     keep_node_cols=True
        # )

        return df_node


def get_node_stocks_cached(comb_pl, tree_key, port_col, node_id, n_split):

    node_cache = {}
    key = (tree_key, port_col, node_id)

    if key not in node_cache:
        node_cache[key] = get_stocks_in_node(
            comb_pl=comb_pl,
            tree_key=tree_key,
            port_col=port_col,
            node_id=node_id,
            n_split=n_split,
        )

    return node_cache[key]


def get_stocks_in_node(
    comb_pl: pl.DataFrame,
    tree_key: str,
    port_col: str,
    node_id: int,
    n_split: int,
) -> pl.DataFrame:

    # --------------------------------------------------
    # 1. Parse tree_key → feature sequence
    # --------------------------------------------------
    feature_seq = tree_key.split(Columns.col_sep)

    # --------------------------------------------------
    # 2. Build tree nodes (same as tree_portfolio)
    # --------------------------------------------------
    tree_df = tree_grows_pl(
        comb_pl.select(
            [Columns.date_col] +
            [
                pl.col(f).alias(f"{Columns.node_col}{Columns.col_sep}{i+1}")
                for i, f in enumerate(feature_seq)
            ]
        ),
        n_split=n_split,
        date_col=Columns.date_col
    )

    # attach nodes
    df = comb_pl.with_columns(tree_df)

    # --------------------------------------------------
    # 3. Determine depth from port_col
    # --------------------------------------------------
    depth = int(port_col.split(Columns.col_sep)[-1])
    used_feature = list(dict.fromkeys(feature_seq[:depth]))

    # --------------------------------------------------
    # 4. Build portfolio ID (same formula)
    # --------------------------------------------------
    expr = pl.lit(1)
    for k in range(depth):
        node_col_k = f"{Columns.node_col}{Columns.col_sep}{k+1}"
        expr = expr + pl.col(node_col_k) * (n_split ** (depth - k - 1))

    df = df.with_columns(expr.alias("target_port"))

    # rank of each split characteristic inside its parent node (the groups the tree splits on),
    # same convention as quantile_bucket: rank / (count + 1)
    df = df.with_columns([
        (
            pl.col(feature_seq[k]).rank("average").over(groups) / (pl.len().over(groups) + 1)
        ).alias(f"parent_rank{Columns.col_sep}{k+1}")
        for k in range(depth)
        for groups in [[Columns.date_col] + [f"{Columns.node_col}{Columns.col_sep}{j+1}" for j in range(k)]]
    ])

    # --------------------------------------------------
    # 5. Filter target node
    # --------------------------------------------------
    df_node = df.filter(pl.col("target_port") == node_id)

    # return result (size is kept for value weighting within the node)
    return (
        df_node.select(
            [Columns.id_col, Columns.date_col, Columns.size_col] + used_feature
            + [f"parent_rank{Columns.col_sep}{k+1}" for k in range(depth)]
        )
    )


def node_buckets(node_id: int, depth: int, n_split: int) -> list:
    """
    Bucket (0 = lowest) the node falls in at each split, first split first.
    Inverts the portfolio id formula: node_id = 1 + sum_k bucket_k * n_split ** (depth - k - 1).
    """
    buckets, rest = [], node_id - 1
    for _ in range(depth):
        buckets.append(rest % n_split)
        rest //= n_split
    return buckets[::-1]


def compute_node_scores(
    df_node: pl.DataFrame,
    tree_key: str,
    port_col: str,
    node_id: int = None,
    n_split: int = 2,
) -> pl.DataFrame:
    """
    factor_score: geometric mean of the stock's market-wide characteristic ranks along the node's path.
    oriented_score (when node_id is given): geometric mean of the stock's ranks inside each parent node
    (from get_stocks_in_node), oriented to the side of the split the node is on (rank for the top bucket,
    1 - rank for the bottom one), so it measures how firmly the stock sits inside the node.
    """

    id_col = Columns.id_col

    # --------------------------------------------------
    # 1. determine features used (with duplicates!)
    # --------------------------------------------------
    full_seq = tree_key.split(Columns.col_sep)
    depth = int(port_col.split(Columns.col_sep)[-1])

    used_seq = full_seq[:depth]

    # --------------------------------------------------
    # 2. build product expression ✅
    # --------------------------------------------------
    score_expr = pl.lit(1.0)

    for f in used_seq:
        score_expr = score_expr * pl.col(f)

    df_node = df_node.with_columns(
        score_expr.pow(1/len(used_seq)).alias("factor_score")
    )

    # --------------------------------------------------
    # 3. direction-aware score
    # --------------------------------------------------
    if node_id is not None:
        oriented_expr = pl.lit(1.0)
        for k, bucket in enumerate(node_buckets(node_id, depth, n_split)):
            position = bucket / (n_split - 1)   # 0 = bottom bucket, 1 = top bucket
            parent_rank = pl.col(f"parent_rank{Columns.col_sep}{k+1}")
            oriented_expr = oriented_expr * (1 - (parent_rank - position).abs())

        df_node = df_node.with_columns(
            oriented_expr.pow(1/len(used_seq)).alias("oriented_score")
        )

    return df_node

# %%
if __name__ == "__main__":

    tuples = [
        # ('accrual/accrual/accrual/lme', 'port/3', 8),
        ('accrual/growth/profitability/lme', 'port/3', 3),
        # ('accrual/accrual/accrual/lme', 'port/3', 1),
        # ('qual/fcf_rank/lme/qual',  'port/4', 12),
        # ('val/val/lme/qual', 'port/2', 1),
        # ('trd/qual/sen/qual', 'port/3', 8)
    ]

    best_combos = pd.MultiIndex.from_tuples(
        tuples,
        names=['combination', 'port', 'node']
    )

    chars = Chars()
    # paths = DataPaths()

    data_saved = False
    # regions = ['GL', 'US', 'UK', 'EU', 'AP', 'JP', 'EM']
    regions = ['GL', ]
    for reg in regions:
        print(f"Loading base characteristics in {reg}")
        ret_name = 'gross_returns'
        features = list(chars.__dict__.values())[:-2]
        data, _, CHARAS_LIST, _, _ = read_ei_data(region_=reg, target=ret_name, ei_factors=features)
        # data = read_db_data(region_=reg, features=features, ret_name=Columns.returns_col, data_saved=data_saved)
        # data = read_big_universe(ret_name=ret_name, features=features)
        data_pl, ret_and_wei = prepare_data(data, ret_name=ret_name, equal_weighted=True)
        df_node = run_pipeline(data_pl, ret_and_wei, best_combos)
        print(df_node)

# %%
