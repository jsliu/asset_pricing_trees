"""
The stock-level pieces of the AP-tree backtest: which stocks sit in a tree node on a date, how they are scored within
it, and the node weights of the original score (calc_sharpe). Moved here from stock_portfolio_pl.py and
plot_test_sr.py (now in obsolete/), unchanged.
"""
import logging

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from src.constants import Columns, Parameters
from src.utils import tree_grows


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
    tree_df = tree_grows(
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


def calc_sharpe(tree_portfolio, ap_tree_model, use_test_data=True):

    logging.info('Splitting data')
    train_portfolios, test_portfolios = train_test_split(
        tree_portfolio, test_size=Parameters.test_size, shuffle=False)

    test_portfolios= test_portfolios.dropna(axis=1, how="all").fillna(0)
    train_portfolios = train_portfolios[test_portfolios.columns].fillna(0)
    
    if use_test_data:
        sdf = ap_tree_model.predict(test_portfolios)
    else:
        sdf = ap_tree_model.predict(train_portfolios)
    sharpe_values = np.mean(sdf, axis=0) / (np.std(sdf, axis=0) + 1e-20)
    k_nonzero = np.sum(ap_tree_model.betas != 0, axis=1)

    df_plot = pd.DataFrame({
        "k_nonzero": k_nonzero,
        "Sharpe": sharpe_values
    })

    mask = df_plot["Sharpe"] == df_plot["Sharpe"].max()
    max_idx = df_plot.index[mask]
    if len(max_idx) > 1:
        best_combo = tree_portfolio.columns[np.unique(np.nonzero(ap_tree_model.betas[max_idx, :][1]))]
        best_port = tree_portfolio.iloc[:, np.unique(np.nonzero(ap_tree_model.betas[max_idx, :][1]))]
        sdf_wei = ap_tree_model.betas[max_idx, :][1]
    else:
        best_combo = tree_portfolio.columns[np.nonzero(ap_tree_model.betas[max_idx, :])[1]]
        best_port = tree_portfolio.iloc[:, np.nonzero(ap_tree_model.betas[max_idx, :])[1]]
        sdf_wei = ap_tree_model.betas[max_idx, :]
    w = sdf_wei[np.nonzero(sdf_wei)]
    combo_wei = pd.Series(w, index=best_combo, name='weight')
    return df_plot, combo_wei, best_port
