# %%
"""
How much the tree scores add to the existing factor scores, for any run: company data (a region) or the
characteristic files (a universe).

Each tree score in TREE_SCORES (the score, its residual on the factor scores, its factor-hedged positions and the
original score, all rank-normalised) is combined with the run's factor scores, and the combinations are compared with
the factor scores alone (EI):
    characteristic files   the factor scores are the factor characteristics rank-normalised each month and oriented
                           (factor_signs: x -1 for those expected to pay when low); EI and each combination are
                           equal-weighted (the tree score has weight 1/(number of factors + 1))
    company data           the factor scores are the company factor scores as they are; EI is weighted with the
                           region's production weights (FACTOR_WEIGHTS) and each combination is
                           (1 - TREE_WEIGHT) x EI + TREE_WEIGHT x tree score
The factors are the run's factor characteristics (factor_chars of its tree set-up, as in the backtest).

Reported, for EI and each combination: performance (score-weighted long/short, with the SDF and each tree score alone
for reference) and rolling returns, IC and ICIR, cross-sectional R2 of returns on the scores (with paired t-tests
against EI), turnover, decay; each tree score's factor exposure, its correlation with the factor scores and the IC of
its residual on them. Company data only: sector exposure and, when the optimiser backtests exist (BT_DIR), their R2 on
the factor returns and the transfer coefficient of each score.
The comparison table and the monthly returns go to the run's report folder (combined_scores_summary.csv,
combined_scores_pnl.csv) and the combined scores to
result/[<region>_]combined_scores[_<universe>][_<variant>].csv.

The residual and hedged scores come from the score file when the backtest wrote them; otherwise they are computed
with the backtest's function (src.score_hedging) and the hedge ratios of analysis/hedge_score.py (run it first); a
score that cannot be had is left out.

Run from the project root (or cell by cell in an IDE, which uses the defaults of the argument parsers):
    python analysis/backtest_analysis.py largecap --variant val
    python analysis/backtest_analysis.py --region GL --variant EI_sub_val --start 20060601
    python analysis/backtest_analysis.py full --no-plots
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, for src/ and backtest.py

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

from analysis.backtest_report import FEATURES, LABEL, REGION, RET_NAME, UNIVERSE, VARIANT
from backtest import r_squared
from src.constants import DataPaths, factor_signs, parse_variant
from src.functions import (calc_fac_ret, calc_turnover, calc_decay, calc_factor_exposure, grouped_exposure, summary,
                           get_residuals, rank_normalise)
from src.preprocessing import read_backtest_data, read_ei_data, read_all_data
from src.score_hedging import neutral_scores

TREE_SCORES = {                                        # name in the tables -> column of the score file
    'Tree': 'size_oriented_norm',                      # the score, rank-normalised
    'Resid': 'size_oriented_resid_norm',               # its residual on the factor scores
    'Hedged': 'size_oriented_hedged_norm',             # its factor-hedged positions
    'Original': 'norm_score',                          # the original score
}
# company data: production weights of the factor scores, by region, and the tree score's weight in a combination
FACTOR_WEIGHTS = {
    'ALL': {'fcf_rank': 0.08, 'qual': 0.37, 'sen': 0.22, 'trd': 0.15, 'val': 0.18},
    'AP': {'fcf_rank': 0.085, 'qual': 0.365, 'sen': 0.18, 'trd': 0.2, 'val': 0.17},
    'EM': {'fcf_rank': 0.085, 'qual': 0.385, 'sen': 0.18, 'trd': 0.2, 'val': 0.15},
    'EU': {'fcf_rank': 0.085, 'qual': 0.365, 'sen': 0.19, 'trd': 0.19, 'val': 0.17},
    'GL': {'fcf_rank': 0.08, 'qual': 0.37, 'sen': 0.22, 'trd': 0.15, 'val': 0.18},
    'JP': {'fcf_rank': 0.25, 'qual': 0.25, 'sen': 0.125, 'trd': 0.125, 'val': 0.25},
    'UK': {'fcf_rank': 0.07, 'qual': 0.36, 'sen': 0.22, 'trd': 0.17, 'val': 0.18},
    'US': {'fcf_rank': 0.06, 'qual': 0.4, 'sen': 0.18, 'trd': 0.18, 'val': 0.18},
}
TREE_WEIGHT = 0.2
# company data: the optimiser backtests, read when they exist ({region}, {tag} filled in)
BT_DIR = r"J:\Quant\Enhanced Index\Team\Zhen\projects\ap_trees\Backtest\backtest_pickles"
BT_NAME = "better_beta_{region}_prod_2026-09-08_full_constraints_{tag}"
BT_MODELS = None        # optimiser model name -> column of the combined-scores file; None pairs them in order


def _analysis_args():
    """The options of this script on top of backtest_report.py's run arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--start', type=int, help='first date, YYYYMMDD')
    parser.add_argument('--end', type=int, help='last date, YYYYMMDD')
    parser.add_argument('--bt-tag', default='sub_fac2', help='tag of the optimiser backtest files (company data)')
    parser.add_argument('--no-plots', action='store_true', help='print and save the tables only')
    args, _ = parser.parse_known_args()
    return args


ARGS = _analysis_args()
COMPANY = REGION is not None
if ARGS.no_plots:
    plt.show = lambda *a, **k: plt.close('all')


def load_data():
    """The run's stock panel by (date, permno), with the company stock information (sectors) for a region."""
    stock_info = None
    if REGION == 'ALL':
        data, _, stock_info = read_all_data(target=RET_NAME, ei_factors=FEATURES)
    elif COMPANY:
        data, _, _, _, stock_info = read_ei_data(region_=REGION, target=RET_NAME, ei_factors=FEATURES)
    else:
        data = read_backtest_data(FEATURES, RET_NAME, universe=UNIVERSE)
    data = data.swaplevel(0, 1).sort_index()
    if stock_info is not None:
        stock_info = stock_info.swaplevel(0, 1).sort_index()
    return data, stock_info


def factor_score_panel(data):
    """The factor scores: the company scores as they are, or each characteristic rank-normalised within each month and
    oriented so that high is expected to pay (factor_signs)."""
    if COMPANY:
        return data[FEATURES]
    scores = data[FEATURES].groupby('date').transform(
        lambda c: pd.Series(np.asarray(rank_normalise(c, cutoff_std=3.5), dtype=float), index=c.index))
    return scores * pd.Series(factor_signs(parse_variant(VARIANT)[1]))


def tree_score_panel(data):
    """The tree scores of TREE_SCORES by (date, permno), those that can be had (see the module docstring)."""
    paths = DataPaths()
    scores = pd.read_csv(paths.result_file('score', REGION, UNIVERSE, VARIANT)).set_index(['date', 'permno'])
    missing = [c for c in TREE_SCORES.values() if c not in scores]
    hedge_file = paths.result_file('report', REGION, UNIVERSE, VARIANT, ext=None) / 'hedge_betas_rebuilt.csv'
    if missing:
        h = None
        if hedge_file.exists():
            h = pd.read_csv(hedge_file).query(
                "score == 'size_oriented_score' and universe == 'all' and spec == '9 factors'").pivot(
                index='refit', columns='factor', values='beta')
        print(f'{missing} not in the score file: computed from size_oriented_score'
              + ('' if h is not None else f' (no {hedge_file.name}: the hedged score is left out)'))
        refits = np.sort(h.index.to_numpy()) if h is not None else np.array([])
        chars_by_date = {d: g.droplevel('date')[FEATURES] for d, g in data.groupby(level='date')}
        extra = {}
        for d, s in scores['size_oriented_score'].dropna().groupby(level='date'):
            if d not in chars_by_date:
                continue
            k = np.searchsorted(refits, d, side='right') - 1          # the refit in force at d
            extra[d] = neutral_scores(s.droplevel('date'), chars_by_date[d], h.loc[refits[k]] if k >= 0 else None)
        extra = pd.concat(extra, names=['date', 'permno'])
        scores = scores.join(extra[[c for c in missing if c in extra]], how='outer')
    available = {k: c for k, c in TREE_SCORES.items() if c in scores}
    out = scores[list(available.values())]
    out.columns = list(available)
    return out


def combine(factor_scores, tree):
    """EI and EI + each tree score: equal-weighted (characteristic files) or production-weighted (company data)."""
    if COMPANY:
        w = pd.Series(FACTOR_WEIGHTS[REGION]).reindex(FEATURES).fillna(0)
        ei = factor_scores.fillna(0) @ w
        combined = {k: (1 - TREE_WEIGHT) * ei + TREE_WEIGHT * tree[k].fillna(0) for k in tree}
    else:
        ei = factor_scores.mean(axis=1)
        combined = {k: pd.concat([factor_scores, tree[k]], axis=1).mean(axis=1) for k in tree}
    return ei.rename('EI'), pd.DataFrame(combined)


def ic_series(score, ret):
    """Monthly correlation of a score with the return."""
    return pd.concat([score, ret], axis=1).dropna().groupby('date').apply(lambda x: x.corr().iloc[0, 1])


def as_dates(frame):
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    return frame


def legend_with_means(ax, frame, fmt):
    ax.legend([f"{c}: {fmt.format(frame[c].mean())}" for c in frame.columns], loc='upper left', bbox_to_anchor=(1, 1))


def paired_t(a, b, name):
    both = pd.concat([a, b], axis=1).dropna()
    stats, pvalue = ttest_rel(both.iloc[:, 0], both.iloc[:, 1])
    print(f'{name}: t {stats:.2f}, p-value {pvalue:.3f}')


# %%
# ---- inputs ----
sns.set_theme()
paths = DataPaths()
data, stock_info = load_data()
ret = data[RET_NAME]
factor_scores = factor_score_panel(data)
tree = tree_score_panel(data).reindex(factor_scores.index)

# the months every tree score covers, within --start/--end, so the combinations are compared over the same sample
covered = tree.notna().groupby('date').any()
months = covered.index[covered.all(axis=1)]
months = months[(months >= (ARGS.start or 0)) & (months <= (ARGS.end or 99999999))]
factor_scores, tree, ret = factor_scores.loc[months], tree.loc[months], ret.loc[months]
ei, combined = combine(factor_scores, tree)
names = list(tree.columns)
weighting = (f'production weights, tree score {TREE_WEIGHT:.0%}' if COMPANY else
             f'equal weights, tree score 1/{len(FEATURES) + 1}')
print(f'{LABEL}: {len(months)} months {months.min()}-{months.max()}; {len(FEATURES)} factors; {weighting}; '
      f'tree scores {names}')

# %%
# ---- performance ----
sdf = pd.read_csv(paths.result_file('ret', REGION, UNIVERSE, VARIANT), index_col=0)['Return'].rename('SDF')
pnl = calc_fac_ret(pd.concat([ei, tree.add_prefix('alone: '), combined.add_prefix('EI + ')], axis=1), ret,
                   date_col='date', score_weighted=True)
pnl = as_dates(pnl.join(sdf, how='left'))
cols_main = ['EI'] + [f'EI + {k}' for k in names]
pnl[cols_main].cumsum().plot(title=f'{LABEL}: EI and EI + each tree score').legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
stats = summary(pnl, ann_factor=12, sorted=False)
print(stats)
print(pnl[cols_main + ['SDF']].corr())
for window, name in [(12, '1-year'), (36, '3-year')]:
    pnl[cols_main].rolling(window).sum().plot(title=f'{LABEL}: Rolling {name}').legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
gain = pnl[[f'EI + {k}' for k in names]].sub(pnl['EI'], axis=0)
gain.columns = names
gain.cumsum().plot(title=f'{LABEL}: Combined minus EI, cumulative').legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# %%
# ---- IC ----
ic = as_dates(pd.DataFrame({c: ic_series(s, ret) for c, s in [('EI', ei)] + [(f'EI + {k}', combined[k]) for k in names]}))
legend_with_means(ic.plot(title=f'{LABEL}: IC'), ic, '{:.3f}')
plt.show()
icir = ic.mean() / ic.std() * np.sqrt(12)
print(pd.DataFrame({'IC': ic.mean(), 'ICIR (annual)': icir}))
for k in names:
    paired_t(ic['EI'], ic[f'EI + {k}'], f'IC, EI vs EI + {k}')

# %%
# ---- cross-sectional R2 of returns on the scores ----
# characteristic files: the return at a date is the next month's; company data: as the earlier analysis did, the
# return shifted by one month per stock
r2_ret = ret.groupby('permno').shift() if COMPANY else ret
panel = pd.concat([factor_scores, tree, r2_ret.rename('r2_ret')], axis=1)
r2 = {'EI': panel.dropna(subset=FEATURES + ['r2_ret']).groupby('date').apply(lambda x: r_squared(x[FEATURES], x['r2_ret']))}
for k in names:
    r2[f'EI + {k}'] = panel.dropna(subset=FEATURES + [k, 'r2_ret']).groupby('date').apply(
        lambda x: r_squared(x[FEATURES + [k]], x['r2_ret']))
r2 = as_dates(pd.DataFrame(r2))
legend_with_means(r2.plot(title=f'{LABEL}: R2'), r2, '{:.4f}')
plt.show()
for k in names:
    paired_t(r2['EI'], r2[f'EI + {k}'], f'R2, EI vs EI + {k}')

# %%
# ---- turnover and decay ----
scores_main = pd.concat([ei, combined.add_prefix('EI + ')], axis=1)
turnover = as_dates(scores_main.apply(lambda x: calc_turnover(x.dropna(), stock_id='permno', date_col='date')))
legend_with_means(turnover.plot(title=f'{LABEL}: Turnover'), turnover, '{:.1%}')
plt.show()
decay = scores_main.apply(lambda x: calc_decay(x.dropna(), date_col='date'))
decay.plot(title=f'{LABEL}: Decay').legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# %%
# ---- the tree scores against the factors: exposure, correlation, residual IC ----
exposure = {k: pd.concat([factor_scores, tree[k]], axis=1).dropna(subset=[k]).groupby('date').apply(
    lambda x: x[FEATURES].apply(lambda y: calc_factor_exposure(x[k], factor=y))) for k in names}
avg_exposure = pd.DataFrame({k: e.mean() for k, e in exposure.items()})
avg_exposure.plot(kind='bar', title=f'{LABEL}: Average factor exposure of each tree score')
plt.show()
print(avg_exposure.round(3))
for k, e in exposure.items():
    e = as_dates(e)
    legend_with_means(e.plot(title=f'{LABEL}: Factor exposure of {k}'), e, '{:.2f}')
    plt.show()
corr = pd.DataFrame({k: pd.concat([factor_scores, tree[k]], axis=1).dropna(subset=[k]).groupby('date').apply(
    lambda x: x[FEATURES].corrwith(x[k])).mean() for k in names})
print('Mean correlation of each tree score with the factor scores')
print(corr.round(3))
resid_ic = {}
for k in names:
    resid = get_residuals(pd.concat([factor_scores, tree[k]], axis=1).dropna(subset=[k]), factor_names=FEATURES,
                          return_name=k, date_name='date')
    resid_ic[k] = pd.Series({'IC': ic_series(tree[k], ret).mean(), 'residual IC': ic_series(resid, ret).mean()})
print('IC of each tree score and of its residual on the factor scores')
print(pd.DataFrame(resid_ic).T.round(4))

# %%
# ---- company data: sector exposure ----
if stock_info is not None and 'sector' in stock_info:
    sector = stock_info['sector']
    grp_exp = pd.concat({c: grouped_exposure(pd.concat([s, sector], axis=1, join='inner'))
                         for c, s in [('EI', ei)] + [(f'EI + {k}', combined[k]) for k in names]
                         + [(k, tree[k]) for k in names]}, axis=1)
    grp_exp[cols_main].groupby('sector').mean().plot(kind='bar', title=f"{LABEL}: Average sector exposure")
    plt.show()
    for k in names:
        frame = as_dates(grp_exp[k].unstack())
        legend_with_means(frame.plot(title=f"{LABEL}: Sector exposure of {k}"), frame, '{:.1%}')
        plt.show()

# %%
# ---- one table to compare the combinations; the combined scores for the optimiser ----
table = pd.DataFrame({k: {
    'IR, tree score alone': stats.loc[f'alone: {k}', 'IR'],
    'IR, EI + tree score': stats.loc[f'EI + {k}', 'IR'],
    'IR gain over EI': stats.loc[f'EI + {k}', 'IR'] - stats.loc['EI', 'IR'],
    'gain t-stat': gain[k].mean() / gain[k].std() * np.sqrt(gain[k].count()),
    'corr with EI pnl': pnl['EI'].corr(pnl[f'EI + {k}']),
    'IC': ic[f'EI + {k}'].mean(), 'ICIR': icir[f'EI + {k}'],
    'R2': r2[f'EI + {k}'].mean(), 'turnover': turnover[f'EI + {k}'].mean(),
    'decay, 3 months': decay.loc[3, f'EI + {k}'],
} for k in names}).T
table.loc['EI alone'] = {'IR, EI + tree score': stats.loc['EI', 'IR'], 'IC': ic['EI'].mean(), 'ICIR': icir['EI'],
                         'R2': r2['EI'].mean(), 'turnover': turnover['EI'].mean(), 'decay, 3 months': decay.loc[3, 'EI']}
print(f'\n==== {LABEL}: the combinations ({weighting}) ====')
print(table.round(3).to_string())
report_dir = paths.result_file('report', REGION, UNIVERSE, VARIANT, ext=None)
report_dir.mkdir(parents=True, exist_ok=True)
table.to_csv(report_dir / 'combined_scores_summary.csv')
alpha = pd.concat([ei.rename('ei')] + [combined[k].rename(f'ei+{k.lower()}') for k in names], axis=1)
alpha.to_csv(paths.result_file('combined_scores', REGION, UNIVERSE, VARIANT))
pnl_out = pnl[cols_main].copy()                        # monthly returns of EI and each combination, for the deck
pnl_out.index = pnl_out.index.strftime('%Y%m%d').astype(int).rename('date')
pnl_out.to_csv(report_dir / 'combined_scores_pnl.csv')
print(f'Saved {report_dir / "combined_scores_summary.csv"} and {paths.result_file("combined_scores", REGION, UNIVERSE, VARIANT)}')

# %%
# ---- company data: the optimiser backtests, if they exist ----
bt_base = Path(BT_DIR) / BT_NAME.format(region=REGION, tag=ARGS.bt_tag) if COMPANY else None
if bt_base is not None and Path(f'{bt_base}_summary.xlsx').exists():
    bt_rets = pd.read_excel(f'{bt_base}_summary.xlsx', sheet_name="rets_active").set_index("dates")
    bt_rets = as_dates(bt_rets.loc[bt_rets.index.isin(months)])
    factor_pnl = as_dates(calc_fac_ret(pd.concat([factor_scores, tree], axis=1), data[RET_NAME].loc[months],
                                       date_col='date', score_weighted=True))
    r2_ts = pd.DataFrame({m: {'EI': r_squared(factor_pnl[FEATURES], bt_rets[m]),
                              **{f'EI + {k}': r_squared(factor_pnl[FEATURES + [k]], bt_rets[m]) for k in names}}
                          for m in bt_rets.columns}).T
    print('R2 of each optimiser backtest on the factor returns')
    print(r2_ts.round(3))
    bt_rets.cumsum().plot(title=f"{LABEL}: optimiser backtests")
    plt.show()
    print(summary(bt_rets, sorted=False, ann_factor=12))

    bt_pickle = pd.read_pickle(f'{bt_base}.pickle')
    models = BT_MODELS or dict(zip(bt_pickle.keys(), alpha.columns))
    print(f'Transfer coefficient: optimiser model -> score {models}')
    tc = {}
    for m, a in models.items():
        alpha_a = alpha[a].copy()
        alpha_a.index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(alpha_a.index.get_level_values('date').astype(str), format="%Y%m%d"),
             alpha_a.index.get_level_values('permno')], names=bt_pickle[m]['optimal_weight'].index.names)
        tc[a] = pd.concat([alpha_a, bt_pickle[m]['optimal_weight']], axis=1, join='inner').groupby(
            level=0).apply(lambda x: x.corr().iloc[0, 1])
    tc = pd.DataFrame(tc)
    legend_with_means(tc.plot(title=f"{LABEL}: TC"), tc, '{:.2f}')
    plt.show()
    for c in tc.columns[1:]:
        paired_t(tc[tc.columns[0]], tc[c], f'TC, {tc.columns[0]} vs {c}')
elif COMPANY:
    print(f'No optimiser backtests at {bt_base}_summary.xlsx: that part is skipped')

# %%
# sns.set_theme()
# chars = Chars()
# paths = DataPaths()
# reg = "GL"
# ret_name = "Universe Returns"
# groups = ['region', 'ind']
# ai_features = list(chars.__dict__.values())[:-2]
# ai_data = read_big_universe(ret_name=ret_name, features=ai_features, add_groups=True)
# ai_score = pd.read_csv(paths.output / f"{reg}_score_big_univ.csv").set_index(['date', 'permno'])['final_score']
# ai_score.name = 'Tree'
# score_and_ret = pd.concat([ai_data[[ret_name] + groups].swaplevel(0, 1), ai_score], axis=1, join='outer')
# score_and_ret["alpha"] = score_and_ret.groupby(["date"])["Tree"].transform(rank_normalize)
# score_and_ret["region_industry_neutral"] = score_and_ret.groupby(["date", "region", "ind"])["Tree"].transform(rank_normalize)
# alpha_scores = score_and_ret[['alpha', 'region_industry_neutral']]
# pnl = calc_fac_ret(alpha_scores, score_and_ret[ret_name]/100, q=5, date_col='date', score_weighted=True)
# pnl.index = pd.to_datetime(pnl.index, format="%Y%m%d")
# pnl.cumsum().plot()
# plt.show()
# print(summary(pnl, ann_factor=12, sorted=False))
# to = alpha_scores.apply(lambda x: calc_turnover(x, date_col='date', stock_id='permno'))
# to.index = pd.to_datetime(to.index, format="%Y%m%d")
# to.plot()
# plt.show()
# decay = alpha_scores.apply(lambda x: calc_decay(x, date_col='date'))
# decay.plot()
# plt.show()
# factor_scores = ai_data[["by", "vol", "mom_1y1m", "mkt_cap"]].groupby('date').transform(lambda x: rank_normalize(x))
# scores1 = factor_scores.merge(alpha_scores['alpha'], left_index=True, right_index=True)
# factor_exposure1 = scores1.groupby('date').apply(lambda x: x.iloc[:, :-1].apply(lambda y: calc_factor_exposure(x.iloc[:, -1], factor=y)))
# factor_exposure1.index = pd.to_datetime(factor_exposure1.index, format="%Y%m%d")
# avg_exp = factor_exposure1.mean()
# avg_exp.plot(kind='bar')
# ax = factor_exposure1.plot(title=f"{reg}: Factor Exposure: alpha")
# labels = [
#     f"{col}: {avg_exp[col]:.2f}"
#     for col in factor_scores.columns
# ]
# ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
# plt.show()


# scores2 = factor_scores.merge(alpha_scores['region_industry_neutral'], left_index=True, right_index=True)
# factor_exposure2 = scores2.groupby('date').apply(lambda x: x.iloc[:, :-1].apply(lambda y: calc_factor_exposure(x.iloc[:, -1], factor=y)))
# factor_exposure2.index = pd.to_datetime(factor_exposure2.index, format="%Y%m%d")
# avg_exp = factor_exposure2.mean()
# avg_exp.plot(kind='bar')
# ax = factor_exposure2.plot(title=f"{reg}: Factor Exposure: region_industry_neutral")
# labels = [
#     f"{col}: {avg_exp[col]:.2f}"
#     for col in factor_scores.columns
# ]
# ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
# plt.show()

# all_scores = factor_scores.swaplevel(0, 1).merge(alpha_scores, left_index=True, right_index=True, how='left')
# all_pnls = calc_fac_ret(all_scores, ai_data[ret_name].swaplevel(0, 1)/100, q=5, date_col='date')
# all_pnls.index = pd.to_datetime(all_pnls.index, format="%Y%m%d")
# all_pnls.cumsum().plot()
# plt.show()
# print(summary(all_pnls, ann_factor=12, sorted=False))
# decays = all_scores.apply(lambda x: calc_decay(x, date_col='date'))
# decays.plot()
# plt.show()

# %%
