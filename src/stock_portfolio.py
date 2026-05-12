# %%
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.utils import recursive_tree_grows
from src.constants import Columns, Chars
from src.preprocessing import read_ei_data


# ----------------------------- core building blocks -----------------------------

def _build_node_columns(input_df: pd.DataFrame,
                        split_features: List[str],
                        n_split: int = 2) -> pd.DataFrame:
    """
    Recreate 'node|k' integer labels (0..n_split-1) for each split level, exactly
    as in your add_portfolio_cols -> recursive_tree_grows path.
    """
    tree_df = input_df[split_features].dropna().copy()
    tree_df.groupby('date').transform(lambda x: x.rank(method='min', pct=True))
    tree_df.columns = [f"{Columns.node_col}{Columns.col_sep}{i}" for i in range(len(split_features))]
    # IMPORTANT: this must call YOUR splitter
    tree_df = recursive_tree_grows(tree_df, n_split=n_split)
    return tree_df


def _attach_port_labels(input_df: pd.DataFrame,
                        tree_df: pd.DataFrame,
                        n_split: int) -> pd.DataFrame:
    """
    Attach per-depth 'port|d' labels using the SAME base-n encoding as add_portfolio_cols:
        port|i = 1 + sum_{k=0..i-1} node|k * n_split^(i-k-1)
    """
    out = input_df.copy()
    D = tree_df.shape[1]
    for i_seq in range(D + 1):
        port_col = f"{Columns.port_col}{Columns.col_sep}{i_seq}"
        out[port_col] = 1
        for k_subseq in range(i_seq):
            out[port_col] = (
                out[port_col]
                + tree_df[f"{Columns.node_col}{Columns.col_sep}{k_subseq}"] * (n_split ** (i_seq - k_subseq - 1))
            )
    return out


def _weights_in_group(idx: pd.Index,
                      weight_s: Optional[pd.Series],
                      all_names: pd.Index) -> pd.Series:
    """
    Compute constituent weights for one node: value-weight if weight_s is provided,
    else equal-weight. Returns a Series indexed by all_names with column sum == 1
    (or 0 if group is empty).
    """
    w = pd.Series(0.0, index=all_names)
    if len(idx) == 0:
        return w
    if weight_s is None:
        w.loc[idx] = 1.0 / len(idx)
        return w
    m = weight_s.reindex(idx).fillna(0.0)
    s = m.sum()
    if s > 0:
        w.loc[idx] = m / s
    return w


# ----------------------------- single-date retrieval -----------------------------

def get_B_for_best_combo_at_date(
    df_t: pd.DataFrame,
    best_combo: pd.MultiIndex,
    combo_to_sequence: Dict[str, List[str]],
    weight_col: Optional[str] = None,
    n_split: int = 2,
    min_node_size: int = 0,
) -> Dict[Tuple[str, str, int], pd.Series]:
    """
    Build stock-weight vectors (a column of B_t) for each (combination, port, node) in best_combo
    for a SINGLE snapshot df_t (one month). Returns a dict mapping the tuple to a Series of weights.

    Parameters
    ----------
    df_t : pd.DataFrame
        Stock snapshot at time t. Index = stock IDs. Must include all split features used by each combination.
        If value-weights are desired, include `weight_col` (or Columns.size_col) in df_t.
    best_combo : pd.MultiIndex
        As provided in your example, with levels: (combination, port, node).
        'port' strings like 'port_4' indicate the depth.
    combo_to_sequence : dict
        Map from `combination` string to the ordered list of feature names used at each depth,
        e.g., {'qual_qual_qual_fcf_rank': ['qual','qual','qual','fcf_rank'], ...}
        (We avoid parsing by '_' because some feature names may themselves contain underscores.)
    weight_col : str or None
        Column for value-weights (e.g., 'mcap'). If None, fall back to Columns.size_col; if not found, use equal-weights.
    n_split : int
        Branching factor; AP-Trees use 2 (binary).
    min_node_size : int
        Optionally drop nodes smaller than this.

    Returns
    -------
    dict[(combination, port, node)] -> pd.Series of stock weights (sum to 1 within node)
    """
    if df_t.index.duplicated().any():
        raise ValueError("df_t index contains duplicates; expected unique stocks at a single date.")

    # choose the weighting series
    wcol = (weight_col or getattr(Columns, "size_col", None))
    weight_s = df_t[wcol].astype(float) if (wcol in df_t.columns) else None

    # parse best_combo into (comb, depth, port_label, node_id)
    parsed = []
    for comb, port, node in best_combo:
        try:
            depth = int(str(port).split(Columns.col_sep)[-1])  # 'port_4' -> 4
        except Exception as e:
            raise ValueError(f"Could not parse depth from port label '{port}'. Expected 'port_<int>'.") from e
        parsed.append((comb, depth, port, int(node)))

    # group by (combination, depth) to reuse computations
    out: Dict[Tuple[str, str, int], pd.Series] = {}
    all_names = df_t.index

    for comb, depth in sorted({(c, d) for (c, d, _, _) in parsed}):
        if comb not in combo_to_sequence:
            raise KeyError(f"combo_to_sequence has no entry for '{comb}'. Provide the ordered split features.")
        sequence = combo_to_sequence[comb]

        if df_t[sequence].dropna().empty:
            continue

        # 1) Recompute node|k labels using YOUR splitter
        tree_df = _build_node_columns(df_t, sequence, n_split=n_split)

        # 2) Build per-depth 'port|d' labels exactly like add_portfolio_cols
        df_with_ports = _attach_port_labels(df_t, tree_df, n_split=n_split)

        # 3) Membership at requested depth
        port_col = f"{Columns.port_col}{Columns.col_sep}{depth}"   # e.g. 'port|4'
        groups = df_with_ports.groupby(port_col).groups            # dict: node_id -> Int64Index of stocks

        # 4) Build weight vector for each requested node under this (comb, depth)
        for (_, _, port_label, node_id) in [x for x in parsed if x[0] == comb and x[1] == depth]:
            if node_id not in groups:
                out[(comb, port_label, node_id)] = pd.Series(0.0, index=all_names, name=f"{port_label}{Columns.col_sep}{node_id}")
                continue
            idx = pd.Index(groups[node_id])
            if min_node_size and (len(idx) < min_node_size):
                out[(comb, port_label, node_id)] = pd.Series(0.0, index=all_names, name=f"{port_label}{Columns.col_sep}{node_id}")
                continue

            w = _weights_in_group(idx, weight_s, all_names)
            s = w.sum()
            if s > 0:
                w = w / s
            w.name = f"{port_label}{Columns.col_sep}{node_id}"
            out[(comb, port_label, node_id)] = w.droplevel('date')

    return pd.DataFrame(out)


def _convert_combo_to_seq(combos: pd.MultiIndex):
    combo2seq = {}
    for combo in combos.get_level_values("combination"):
        combo2seq[combo] = combo.split(Columns.col_sep)
    return combo2seq


# returns: {date -> {(combination, port, node) -> stock-weight Series}}
def get_B_for_best_combo_over_time(
    comb_df: pd.DataFrame,
    best_combo: pd.MultiIndex,
    weight_col: Optional[str] = None,
    n_split: int = 2,
    min_node_size: int = 0
) -> Dict[pd.Timestamp, Dict[Tuple[str, str, int], pd.Series]]:
    out_ts = {}
    combo_to_sequence = _convert_combo_to_seq(best_combo)
    for dt, df_t in comb_df.groupby(Columns.date_col):
        out_ts[dt] = get_B_for_best_combo_at_date(
            df_t=df_t,
            best_combo=best_combo,
            combo_to_sequence=combo_to_sequence,
            weight_col=weight_col,
            n_split=n_split,
            min_node_size=min_node_size
        )
    out_df = pd.concat(out_ts)
    out_df.index = out_df.index.set_names('date', level=0)
    return out_df

# %%
if __name__ == "__main__":
    # 1) Define the ordered split features for each combination in best_combo.
    combo_to_sequence = {
        'qual_qual_qual_fcf_rank':             ['qual','qual','qual','fcf_rank'],
        'qual_fcf_rank_lme_qual':              ['qual','fcf_rank','lme','qual'],
        'fcf_rank_qual_fcf_rank_fcf_rank':     ['fcf_rank','qual','fcf_rank','fcf_rank'],
        'lme_lme_qual_qual':                   ['lme','lme','qual','qual'],  # depth=4; 'port_2' will use depth=2
    }


    tuples = [
        ('qual_qual_qual_fcf_rank', 'port_4', 15),
        ('qual_fcf_rank_lme_qual',  'port_4', 12),
        ('fcf_rank_qual_fcf_rank_fcf_rank', 'port_4', 8),
        ('lme_lme_qual_qual', 'port_2', 3)
    ]

    best_combo = pd.MultiIndex.from_tuples(
        tuples,
        names=['combination', 'port', 'node']
    )

    reg = 'GL'
    chars = Chars()
    features = list(chars.__dict__.values())[:-2]
    data, _, CHARAS_LIST, _ = read_ei_data(region_=reg, target=Columns.returns_col, ei_factors=features)
    ret_df = data[Columns.returns_col]
    raw_size_df = data[Columns.size_col]
    raw_size_df.name = Columns.size_col
    lme_df = np.log(raw_size_df)
    lme_df.name = chars.lme
    data = pd.concat([data, lme_df,], axis=1)

    # 2) One-month snapshot df_t: index=stock IDs, columns include ALL used features + market cap (e.g., Columns.size_col)
    idx = pd.IndexSlice
    df_t = data.loc[idx[:, 20260130], :]

    # 3) Build B columns (stock weights) for those nodes at that date
    B_cols = get_B_for_best_combo_at_date(
        df_t=df_t,
        best_combo=best_combo,
        combo_to_sequence=combo_to_sequence,
        weight_col=Columns.size_col,  # e.g., 'mcap' if Columns.size_col == 'mcap'
        n_split=2,
        min_node_size=20
    )

    # Inspect one node’s constituents
    key = ('qual_qual_qual_fcf_rank', 'port_4', 15)
    w_node = B_cols[key].loc[lambda s: s > 0].sort_values(ascending=False)
    print(w_node.head(10))

    for key in B_cols.keys():
        print(f"{key}: {B_cols[key].sum()}")
# %%
