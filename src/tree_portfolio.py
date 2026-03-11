import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from constants import Columns

# -------------------- Low-level helpers --------------------
def _vw_ret_by_date(df: pd.DataFrame,
                    date_col: str,
                    ret_col: str,
                    w_col: Optional[str]) -> pd.Series:
    """
    Value-weighted (or equal-weighted) return per date.
    """
    if df.empty:
        return pd.Series(dtype=float)
    if w_col is None:
        return df.groupby(date_col, observed=True)[ret_col].mean()
    g = df.groupby(date_col, observed=True)
    num = g.apply(lambda x: (x[ret_col] * x[w_col]).sum(), include_groups=False)
    den = g[w_col].sum().replace(0.0, np.nan)
    return (num / den).fillna(0.0)


def _sharpe2(mu: np.ndarray,
             cov: np.ndarray,
             mean_shrink: float = 0.0,
             ridge: float = 0.0) -> float:
    """
    Sharpe^2 = mu' (Sigma + ridge*I)^{-1} mu with mean-shrink on mu.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    mu_s = (1.0 - mean_shrink) * mu
    if ridge > 0:
        cov = cov + ridge * np.eye(cov.shape[0], dtype=float)
    try:
        z = np.linalg.solve(cov, mu_s)
        return float(mu_s @ z)
    except np.linalg.LinAlgError:
        z = np.linalg.pinv(cov) @ mu_s
        return float(mu_s @ z)


def _sdf_gain(basis_df: Optional[pd.DataFrame],
              children: List[pd.Series],
              mean_shrink: float,
              ridge: float,
              min_T: int = 24) -> float:
    """
    Δ Sharpe^2 from adding all child series to the current basis.
    """
    if basis_df is None or basis_df.shape[1] == 0:
        M = pd.concat(children, axis=1).dropna()
        if M.shape[0] < min_T or M.shape[1] == 0:
            return -np.inf
        mu = M.mean().values
        cov = np.cov(M.values, rowvar=False)
        return _sharpe2(mu, cov, mean_shrink, ridge)

    M_old = basis_df.dropna()
    if M_old.shape[0] < min_T or M_old.shape[1] == 0:
        return -np.inf
    mu_old = M_old.mean().values
    cov_old = np.cov(M_old.values, rowvar=False)
    sh2_old = _sharpe2(mu_old, cov_old, mean_shrink, ridge)

    M_new = pd.concat([M_old] + children, axis=1).dropna()
    if M_new.shape[0] < min_T:
        return -np.inf
    mu_new = M_new.mean().values
    cov_new = np.cov(M_new.values, rowvar=False)
    sh2_new = _sharpe2(mu_new, cov_new, mean_shrink, ridge)

    return sh2_new - sh2_old


def _feature_cuts_for_S(n_split: int) -> List[float]:
    """
    Default equally spaced quantiles for S-way split:
    e.g., S=4 -> [0.25, 0.5, 0.75]
    """
    return [k / n_split for k in range(1, n_split)]


def _assign_bins_by_date_rank(sub: pd.DataFrame,
                              date_col: str,
                              feat: str,
                              cut_points: List[float]) -> pd.Series:
    """
    Compute within-date percentile ranks and map to 0..S-1 bins.
    """
    # rank in (0,1], method average stable
    ranks = sub.groupby(date_col, observed=True)[feat].rank(pct=True, method="average")
    # convert to bins: <=cut1 -> 0, (cut1,cut2] -> 1, ... > last -> S-1
    bins = pd.Series(np.searchsorted(cut_points, ranks, side="right"), index=sub.index, dtype=int)
    return bins


def _children_returns_from_bins(sub: pd.DataFrame,
                                bins: pd.Series,
                                n_split: int,
                                date_col: str,
                                ret_col: str,
                                w_col: Optional[str]) -> List[pd.Series]:
    """
    Value-weighted returns per bin (0..S-1), each as T×1 Series.
    """
    out = []
    for b in range(n_split):
        mask = (bins == b)
        if not mask.any():
            out.append(pd.Series(dtype=float))
            continue
        out.append(_vw_ret_by_date(sub.loc[mask], date_col, ret_col, w_col))
    return out


def _best_split_at_node_multi(panel: pd.DataFrame,
                              features: List[str],
                              n_split: int,
                              date_col: str,
                              ret_col: str,
                              w_col: Optional[str],
                              basis_df: Optional[pd.DataFrame],
                              mean_shrink: float,
                              ridge: float,
                              min_leaf_obs: int,
                              min_T: int,
                              custom_cuts: Optional[Dict[str, List[float]]] = None
                              ) -> Tuple[Optional[str], Optional[List[float]], List[pd.Series], float]:
    """
    Exhaustive search over (feature, S-way cut) and return:
      (feature*, cut_points*, [r_0..r_{S-1}], gain)
    cut_points*: list of (S-1) quantiles in (0,1).
    """
    best = dict(f=None, cuts=None, children=None, gain=-np.inf)

    for f in features:
        sub = panel[[date_col, f, ret_col] + ([w_col] if w_col else [])].dropna(subset=[f, ret_col]).copy()
        if sub.empty:
            continue

        # choose cut list for this feature
        cuts = custom_cuts.get(f) if (custom_cuts and f in custom_cuts) else _feature_cuts_for_S(n_split)
        if len(cuts) != (n_split - 1):
            raise ValueError(f"Feature '{f}' needs {n_split-1} cuts, got {len(cuts)}")

        # map to 0..S-1 bins
        bins = _assign_bins_by_date_rank(sub, date_col, f, cuts)

        # leaf-size guard (optional)
        valid = True
        for b in range(n_split):
            if (bins == b).sum() < min_leaf_obs:
                valid = False
                break
        if not valid:
            continue

        # children returns
        children = _children_returns_from_bins(sub, bins, n_split, date_col, ret_col, w_col)
        # If any child is empty, skip
        if any(ch.empty for ch in children):
            continue

        gain = _sdf_gain(basis_df, children, mean_shrink, ridge, min_T=min_T)
        if gain > best["gain"]:
            best.update(f=f, cuts=cuts, children=children, gain=gain)

    if best["f"] is None:
        return None, None, [], -np.inf
    return best["f"], best["cuts"], best["children"], best["gain"]


def recursive_tree_grows_bestsplit(
    input_df: pd.DataFrame,
    n_split: int,
    features: List[str],
    *,
    date_col: str = Columns.date_col,
    ret_col: str = Columns.returns_col,
    w_col: Optional[str] = Columns.size_col,
    depth: int = 0,
    max_depth: int = 3,
    mean_shrink: float = 0.0,
    ridge: float = 0.0,
    basis_df: Optional[pd.DataFrame] = None,
    allow_feature_reuse: bool = True,
    min_leaf_obs: int = 50,
    min_T: int = 24,
    custom_cuts: Optional[Dict[str, List[float]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-way best-split AP-Tree.
    Writes node columns as f"{Columns.node_col}{Columns.col_sep}{depth}" with values in {0,...,n_split-1}.
    Returns (panel_out, basis_out). Also records chosen feature order in attrs["chosen_features"].
    """
    panel = input_df.copy()

    # keep trace of chosen order in attrs (initialize at root)
    if depth == 0 and "chosen_features" not in panel.attrs:
        panel.attrs["chosen_features"] = []

    if depth >= max_depth:
        return panel, (basis_df if basis_df is not None else pd.DataFrame())

    feature_pool = features if allow_feature_reuse else features[:]  # (optionally remove chosen later)

    # choose best feature + cuts (S-way)
    f_star, cuts_star, children, gain = _best_split_at_node_multi(
        panel, feature_pool, n_split, date_col, ret_col, w_col,
        basis_df, mean_shrink, ridge, min_leaf_obs, min_T, custom_cuts
    )
    node_col = f"{Columns.node_col}{Columns.col_sep}{depth}"

    if f_star is None or not np.isfinite(gain):
        # cannot split further → constant branch value to keep structure
        panel.loc[:, node_col] = 0
        return panel, (basis_df if basis_df is not None else pd.DataFrame())

    # assign bins for the chosen feature/cuts
    sub = panel[[date_col, f_star]].copy()
    bins = _assign_bins_by_date_rank(sub, date_col, f_star, cuts_star)
    panel.loc[:, node_col] = bins.values

    # append children series into basis
    child_cols = [f"{node_col}:{b}" for b in range(n_split)]
    if basis_df is None or basis_df.empty:
        basis_new = pd.concat([ch.rename(child_cols[i]) for i, ch in enumerate(children)], axis=1)
    else:
        basis_new = pd.concat([basis_df] + [children[i].rename(child_cols[i]) for i in range(n_split)], axis=1)

    # record chosen feature order for naming the top column level later
    chosen = panel.attrs.get("chosen_features", [])
    chosen = chosen + [f_star]
    panel.attrs["chosen_features"] = chosen

    # recurse into each child bin
    out_parts = []
    basis_parts = []
    for b in range(n_split):
        child_panel = panel.loc[panel[node_col] == b].copy()
        if child_panel.empty:
            continue
        next_features = features if allow_feature_reuse else [f for f in features if f != f_star]

        child_out, child_basis = recursive_tree_grows_bestsplit(
            input_df=child_panel,
            n_split=n_split,
            features=next_features,
            date_col=date_col,
            ret_col=ret_col,
            w_col=w_col,
            depth=depth + 1,
            max_depth=max_depth,
            mean_shrink=mean_shrink,
            ridge=ridge,
            basis_df=basis_new,
            allow_feature_reuse=allow_feature_reuse,
            min_leaf_obs=min_leaf_obs,
            min_T=min_T,
            custom_cuts=custom_cuts
        )
        out_parts.append(child_out)
        if child_basis is not None and not child_basis.empty:
            basis_parts.append(child_basis)

    panel_out = pd.concat(out_parts, axis=0) if out_parts else panel

    # merge basis pieces (identical column names; extend row coverage)
    if basis_parts:
        basis_out = basis_parts[0]
        for B in basis_parts[1:]:
            basis_out = basis_out.combine_first(B)
    else:
        basis_out = basis_new

    # propagate chosen features up
    panel_out.attrs["chosen_features"] = chosen
    return panel_out, basis_out


def fit_best_split_nodes(
    df: pd.DataFrame,
    features_pool: List[str],
    tree_depth: int,
    *,
    n_split: int = 2,
    date_col: str = Columns.date_col,
    ret_col: str = Columns.returns_col,
    w_col: Optional[str] = Columns.size_col,
    mean_shrink: float = 0.0,
    ridge: float = 0.0,
    allow_feature_reuse: bool = True,
    min_leaf_obs: int = 50,
    min_T: int = 24,
    custom_cuts: Optional[Dict[str, List[float]]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fits the multi-way best-split tree once on the full panel (time-series objective),
    returns panel with node:{d} columns and chosen feature order.
    """
    panel_out, _ = recursive_tree_grows_bestsplit(
        input_df=df,
        n_split=n_split,
        features=features_pool,
        date_col=date_col,
        ret_col=ret_col,
        w_col=w_col,
        depth=0,
        max_depth=tree_depth,
        mean_shrink=mean_shrink,
        ridge=ridge,
        basis_df=None,
        allow_feature_reuse=allow_feature_reuse,
        min_leaf_obs=min_leaf_obs,
        min_T=min_T,
        custom_cuts=custom_cuts
    )
    chosen_features = panel_out.attrs.get("chosen_features", ["best"] * tree_depth)
    return panel_out, chosen_features


def _get_weighted_ret(month_df: pd.DataFrame) -> float:
    w = month_df[Columns.size_col].values
    r = month_df[Columns.returns_col].values
    s = w.sum()
    return float(np.dot(r, w) / s) if s != 0 else np.nan


def add_portfolio_cols_from_nodes(
    input_df: pd.DataFrame,
    report_features: List[str],
    tree_depth: int,
    *,
    n_split: int = 2
) -> pd.DataFrame:
    """
    EXACT monthly block shape as your original:
      - After outer concat, columns → MultiIndex [comb, port, node]
      - Here we produce (port, node) and transpose to match.
    """
    df = input_df.copy()
    features_dict: Dict[str, pd.DataFrame] = {}

    for i_seq in range(tree_depth + 1):
        port_col = f"{Columns.port_col}{Columns.col_sep}{i_seq}"
        df.loc[:, port_col] = pd.Series(1, index=df.index, dtype='Int64')
        for k_subseq in range(i_seq):
            node_k = f"{Columns.node_col}{Columns.col_sep}{k_subseq}"
            df.loc[:, port_col] = df[port_col].astype('Int64') + (df[node_k].astype('Int64') * (n_split ** (i_seq - k_subseq - 1)))

        # weighted return per port
        ret_by_port = (df.groupby(port_col, observed=True)
                         .apply(_get_weighted_ret, include_groups=False)
                         .rename(Columns.w_returns_col)
                         .to_frame())

        # min/max of reporting features
        agg = {f: ['min', 'max'] for f in set(report_features)}
        minmax = df.groupby(port_col, observed=True).agg(agg)
        minmax.columns = [f"{f}_{stat}" for (f, stat) in minmax.columns.to_flat_index()]

        block = pd.concat([ret_by_port, minmax], axis=1)
        features_dict[port_col] = block

    features_df = pd.concat(features_dict, axis=0)
    features_df.columns.name = Columns.features_col
    features_df.index.names = [Columns.port_col, Columns.node_col]
    features_df = features_df.T
    return features_df


def tree_portfolio_best_split(
    comb_df: pd.DataFrame,
    feature_pool: List[str],
    report_features: List[str],
    *,
    n_split: int = 2,
    tree_depth: int = 4,
    date_col: str = Columns.date_col,
    ret_col: str = Columns.returns_col,
    w_col: Optional[str] = Columns.size_col,
    mean_shrink: float = 0.0,
    ridge: float = 0.0,
    allow_feature_reuse: bool = True,
    min_leaf_obs: int = 50,
    min_T: int = 24,
    custom_cuts: Optional[Dict[str, List[float]]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fit best-split once; then per-month aggregation to your panel format.
    Returns {comb_name: monthly_panel}.
    """
    df_nodes, chosen_order = fit_best_split_nodes(
        comb_df, feature_pool, tree_depth,
        n_split=n_split, date_col=date_col, ret_col=ret_col, w_col=w_col,
        mean_shrink=mean_shrink, ridge=ridge,
        allow_feature_reuse=allow_feature_reuse,
        min_leaf_obs=min_leaf_obs, min_T=min_T,
        custom_cuts=custom_cuts
    )
    # top-level "combination" name (just like your old join over factor order)
    comb_name = Columns.col_sep.join(chosen_order)

    def _per_month(month_df: pd.DataFrame) -> pd.DataFrame:
        return add_portfolio_cols_from_nodes(
            month_df, report_features=report_features, tree_depth=tree_depth, n_split=n_split
        )

    monthly = (df_nodes.groupby(date_col, observed=True).apply(_per_month, include_groups=False))
    # index: (date, features); columns: (port, node)

    return {comb_name: monthly}


def build_tree_portfolio_best_split(
    comb_df: pd.DataFrame,
    feature_pool: List[str],
    report_features: List[str],
    *,
    n_split: int = 2,
    tree_depth: int = 4,
    **kwargs
) -> pd.DataFrame:
    """
    Final panel:
      columns: MultiIndex [comb, port, node]
      index:   MultiIndex [date, features]
    """
    portfolios = tree_portfolio_best_split(
        comb_df=comb_df,
        feature_pool=feature_pool,
        report_features=report_features,
        n_split=n_split,
        tree_depth=tree_depth,
        **kwargs
    )
    portfolio = pd.concat(portfolios, axis=1).T.drop_duplicates().T
    portfolio.index.names = [Columns.date_col, Columns.features_col]
    portfolio.columns.names = [Columns.comb_col, Columns.port_col, Columns.node_col]
    return portfolio
