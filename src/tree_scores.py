"""
The stock-level pieces of the AP-tree backtest: which stocks sit in a tree node on a date, how they are scored within
it, the stock scores of a model on a date (score_stocks), and the node weights of the original score (calc_sharpe).
Moved here unchanged from stock_portfolio_pl.py and plot_test_sr.py (now in obsolete/) and backtest.py.
"""
import logging

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from src.constants import Columns, Parameters
from src.functions import rank_normalise
from src.utils import tree_grows

# tree_portfolio's min_node_size (src/utils.py): smaller nodes get no portfolio return
MIN_NODE_SIZE = 50


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


def score_stocks(comb_d, node_betas, combo_wei=None):
    """
    Stock scores on one date from the nodes held by the SDF (node_betas) and, if given, by the
    original score (combo_wei); both are Series indexed by (tree_key, port_col, node_id).

    final_score/final_score_norm: original score, node weight x geometric mean of market-wide characteristic ranks.
    sdf_weight: the stock's weight in the SDF, sum over nodes of beta x the stock's weight in the node.
    size_oriented_score: the same node weights beta, spread inside each node in proportion to
        (weight in node x oriented_score), so each node still totals beta but the stocks sitting most
        firmly inside it (by characteristic ranks inside the parent node) get more of it.
    size_oriented_norm: size_oriented_score rank-normalised like the factor scores (see _normalise_scores).

    Every stock of the date's universe (comb_d) is returned; sdf_weight and size_oriented_score are 0 for
    stocks in no held node, final_score/final_score_norm are null for stocks the original score does not cover.
    """
    combo_wei = pd.Series(dtype=float) if combo_wei is None else combo_wei
    node_keys = list(dict.fromkeys(
        [k for k, w in combo_wei.items() if w != 0] + [k for k, b in node_betas.items() if b != 0]
    ))

    dfs = []
    # tree splits are computed within each date, so growing the tree on one date alone gives the same nodes
    for tree_key, port_col, node_id in node_keys:
        key = (tree_key, port_col, node_id)

        df_node = get_stocks_in_node(
            comb_d,
            tree_key,
            port_col,
            int(node_id),
            Parameters.n_splits
        )

        if df_node.is_empty():
            continue

        df_scores = compute_node_scores(
            df_node=df_node,
            tree_key=tree_key,
            port_col=port_col,
            node_id=int(node_id),
            n_split=Parameters.n_splits,
        )

        w = combo_wei.get(key, 0.0)
        # the node's tree portfolio has no return below MIN_NODE_SIZE stocks, so the SDF holds nothing there
        b_held = node_betas.get(key, 0.0) if df_node.height >= MIN_NODE_SIZE else 0.0
        in_node = pl.col(Columns.size_col) / pl.col(Columns.size_col).sum()
        tilted = in_node * pl.col("oriented_score")

        dfs.append(
            df_scores.select(
                Columns.id_col,
                pl.lit(w != 0).alias("in_old_score"),
                (pl.col("factor_score") * w).alias("weighted_score"),
                (in_node * b_held).alias("sdf_weight"),
                (tilted / tilted.sum() * b_held).alias("size_oriented_score"),
            )
        )

    final_df = comb_d.select(Columns.id_col)
    if not dfs:
        return _normalise_scores(final_df.with_columns(
            pl.lit(None, pl.Float64).alias("final_score"), pl.lit(None, pl.Float64).alias("final_score_norm"),
            pl.lit(0.0).alias("sdf_weight"), pl.lit(0.0).alias("size_oriented_score"),
        ))

    node_rows = pl.concat(dfs)
    scores = node_rows.group_by(Columns.id_col).agg(pl.col("sdf_weight").sum(), pl.col("size_oriented_score").sum())

    old = (
        node_rows
        .filter(pl.col("in_old_score"))
        .group_by(Columns.id_col)
        .agg(
            pl.col("weighted_score").sum().alias("final_score")
        )
    )
    n = old.height
    u = ((old["final_score"].rank("min") - 0.5) / n).to_numpy()
    old = old.with_columns(pl.Series("final_score_norm", norm.ppf(u), dtype=pl.Float64))

    final_df = (
        final_df
        .join(old, on=Columns.id_col, how="left")
        .join(scores, on=Columns.id_col, how="left")
        .with_columns(pl.col("sdf_weight").fill_null(0.0), pl.col("size_oriented_score").fill_null(0.0))
    )
    return _normalise_scores(final_df)


def _normalise_scores(scores):
    """
    size_oriented_norm: size_oriented_score across the whole universe (0 = not held, i.e. neutral),
    rank-normalised with the same rank_normalise(cutoff_std=3.5) as the factor scores.
    """
    z = rank_normalise(scores["size_oriented_score"].to_pandas(), cutoff_std=3.5)
    return scores.with_columns(pl.Series("size_oriented_norm", np.asarray(z, dtype=float)))
