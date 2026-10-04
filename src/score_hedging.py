"""
Stock scores with the existing factors taken out, two ways:

    residual   the score regressed across stocks (with an intercept) on the factor scores - each factor characteristic
               rank-normalised that month, as the factor portfolios use it - keeping the residual (residual_score)
    hedged     the score's long/short positions minus its factor exposure: score weights - h . factor weights, with
               the hedge ratios h estimated at each refit from the model's score rebuilt over the months before it
               (rebuilt_history, hedge_ratios) - the way the SDF's own hedge uses its portfolios' history

Used by backtest.py (the scores of every date) and analysis/hedge_score.py (hedge ratios for the analyses).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.constants import Columns
from src.functions import _get_weights, rank_normalise
from src.tree_scores import score_stocks

MIN_STOCKS = 30                     # fewest scored stocks for a month's long/short
# months of rebuilt score history the hedge ratios are estimated on: the last HEDGE_WINDOW, or all there are when
# the data is shorter (e.g. a region whose history starts a few years before the backtest)
HEDGE_WINDOW = 120


def score_weights(s):
    """Score-weighted long/short weights of scores s (rank-normalised, 100% long and 100% short), NaN dropped."""
    return pd.Series(np.asarray(_get_weights(s, score_weighted=True), dtype=float), index=s.index).dropna()


def long_short(s, r):
    """Score-weighted long/short return of scores s on returns r (both indexed by permno), net of their mean."""
    if len(s) < MIN_STOCKS or s.nunique() < 2:
        return np.nan
    w = score_weights(s)
    rr = r.reindex(w.index).dropna()
    w = w.loc[rr.index]
    return float((w * (rr - rr.mean())).sum())


def rebuilt_history(comb_by_date, ret_by_date, node_betas, combo_wei, months, scores, bucket_by_date=None):
    """
    Long/short returns in each of `months` of the scores the model with node_betas (and combo_wei, the original
    score's node weights; None leaves that score out) gives, over all stocks and, with bucket_by_date, within each
    size quintile: DataFrame of months x (score, universe).
    """
    out = {}
    for m in months:
        comb_m = comb_by_date.get(m)
        if comb_m is None:
            continue
        sc = score_stocks(comb_m, node_betas, combo_wei).select(Columns.id_col, *scores).to_pandas().set_index(Columns.id_col)
        r, b = ret_by_date[m], (bucket_by_date or {}).get(m)
        row = {}
        for score in scores:
            s = sc[score].dropna()
            row[(score, 'all')] = long_short(s, r)
            if b is not None:
                for q in b.cat.categories:
                    row[(score, q)] = long_short(s[s.index.isin(b.index[b == q])], r)
        out[m] = row
    return pd.DataFrame(out).T


def enough_months(n, n_factors):
    """Whether n months can estimate hedge ratios on n_factors factors: more months than coefficients."""
    return n > n_factors + 1             # ponytail: any short history is used; few months give noisy ratios


def hedge_ratios(y, factors):
    """Factor betas (regression with intercept) of the monthly return series y, or None with too few months."""
    both = pd.concat([y.rename('y'), factors], axis=1, join='inner').dropna()
    if not enough_months(len(both), factors.shape[1]):
        return None
    return sm.OLS(both['y'], sm.add_constant(both.drop(columns='y'))).fit().params.drop('const')


def residual_score(s, exposures):
    """Residual of the cross-sectional regression (with intercept) of scores s on the exposures (stocks x factors)."""
    x = exposures.reindex(s.index).fillna(0).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(s)), x])
    beta = np.linalg.lstsq(X, s.to_numpy(dtype=float), rcond=None)[0]
    return pd.Series(s.to_numpy(dtype=float) - X @ beta, index=s.index)


def factor_scores(chars):
    """The factor portfolios' view of one month's characteristics (stocks x factors): the rank-normalised exposures
    (missing = 0, neutral) and the stock weights (score-weighted long/short, as calc_fac_ret builds the factors)."""
    exposures = chars.apply(lambda c: pd.Series(np.asarray(rank_normalise(c, cutoff_std=3.5), dtype=float),
                                                index=c.index)).fillna(0)
    weights = chars.apply(lambda c: pd.Series(np.asarray(_get_weights(c, score_weighted=True), dtype=float),
                                              index=c.index))
    return exposures, weights


def neutral_scores(s, chars, h=None, name='size_oriented'):
    """
    For one month's scores s (indexed by permno) and characteristics chars (stocks x factors): the residual score and,
    with hedge ratios h (a Series over the factors), the hedged score's positions, each with a rank-normalised version
    (cutoff +-3.5). Columns <name>_resid(_norm) and <name>_hedged(_norm); the hedged positions cover the factor
    portfolios' stocks as well, so their rows are the union of both.
    """
    exposures, weights = factor_scores(chars)
    resid = residual_score(s, exposures)
    out = {f'{name}_resid': resid,
           f'{name}_resid_norm': pd.Series(np.asarray(rank_normalise(resid, cutoff_std=3.5), dtype=float), index=resid.index)}
    if h is not None:
        stocks = s.index.union(weights.index)
        hedged = (score_weights(s).reindex(stocks).fillna(0)
                  - weights.reindex(stocks).fillna(0)[list(h.index)].to_numpy() @ h.to_numpy())
        hedged = pd.Series(hedged, index=stocks)
        out[f'{name}_hedged'] = hedged
        out[f'{name}_hedged_norm'] = pd.Series(np.asarray(rank_normalise(hedged, cutoff_std=3.5), dtype=float),
                                               index=stocks)
    return pd.DataFrame(out)
