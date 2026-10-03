# %%
"""
Factor-neutral stock score. Each month the tree score (size_oriented_score) is regressed across the scored stocks on
the factor scores - the characteristics behind the factor portfolios, each rank-normalised like the factors - and the
residual, the part of the score they do not explain, is rank-normalised: size_oriented_resid_norm. This is
src.score_hedging.neutral_scores, the same function backtest.py uses for the columns of the same names.

The raw and neutral scores are then compared: score-weighted long/short return, Sharpe ratio, alpha and R2 against
the factor returns, rank IC, turnover and trading costs (overall and by period), and the transfer coefficient and
information ratio in a long-only portfolio against the cap-weighted benchmark (30% active share).

Run backtest_report.py for the run first (it saves the factor returns), then from the project root:
    python analysis/neutralize_scores.py [<universe>] [--region GL] [--variant TAGS]      e.g. python analysis/neutralize_scores.py largecap
Saves the neutral score to result/[<region>_]score_neutral[_<universe>][_<variant>].csv (date, permno,
size_oriented_resid, size_oriented_resid_norm), e.g. result/score_neutral_largecap_slow3_val.csv, and the
comparison to the run's report folder (neutral_score_by_period.csv, neutral_score_summary.csv).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, for src/ and backtest.py

import numpy as np
import pandas as pd

from analysis.backtest_report import (FEATURES, REGION, UNIVERSE, VARIANT, LABEL, RET_NAME, COSTS_BPS, make_periods, by_period,
                                      long_short, rank_ic, traded)
from analysis.transfer_coefficient import grid, long_only
from src.constants import DataPaths
from src.preprocessing import read_backtest_data
from src.score_hedging import factor_scores, neutral_scores

SCORE = 'size_oriented_score'          # regressed on the factor scores
RAW = 'size_oriented_norm'             # the score before neutralising, for the comparison (same ranks as SCORE)
NEUTRAL = 'size_oriented_resid_norm'
ACTIVE_SHARE = 0.30


def long_only_stats(score, data, dates):
    """Mean transfer coefficient and information ratio of a long-only portfolio built from the score."""
    stocks = data.index.get_level_values('permno').unique()
    cap, ret = grid(data['mkt_cap'], dates, stocks), grid(data[RET_NAME], dates, stocks)
    held = grid(data[RET_NAME].notna().astype(float), dates, stocks)
    b = cap * held / (cap * held).sum(axis=1, keepdims=True)
    z = grid(score, dates, stocks) * held
    w = np.vstack([long_only(b[t], z[t], ACTIVE_SHARE) for t in range(len(dates))])
    active = w - b
    tc = np.nanmean([np.corrcoef(active[t], z[t])[0, 1] for t in range(len(dates))])
    r = (active * ret).sum(axis=1)
    t = np.r_[0, np.abs(np.diff(active, axis=0)).sum(axis=1)]
    out = {'long-only TC': tc, 'long-only IR': r.mean() / r.std() * np.sqrt(12)}
    for bps in COSTS_BPS:
        net = r - t * bps / 1e4
        out[f'long-only IR after {bps}bps'] = net.mean() / net.std() * np.sqrt(12)
    return out


# %%
if __name__ == '__main__':
    pd.set_option('display.width', 250)
    paths = DataPaths()
    report_dir = paths.result_file('report', REGION, UNIVERSE, VARIANT, ext=None)
    features = FEATURES

    scores = pd.read_csv(paths.result_file('score', REGION, UNIVERSE, VARIANT), usecols=['date', 'permno', SCORE, RAW])
    scores = scores.dropna(subset=[SCORE]).set_index(['date', 'permno'])
    data = read_backtest_data(features, RET_NAME, region=REGION, universe=UNIVERSE).swaplevel(0, 1).sort_index()
    data = data[data.index.get_level_values('date') >= scores.index.get_level_values('date').min()]
    chars_by_date = {d: g.droplevel('date')[features] for d, g in data.groupby(level='date')}

    # ---- 1. neutralise, month by month ----
    resid, worst = {}, 0.0
    for d, s in scores[SCORE].groupby(level='date'):
        if d not in chars_by_date:
            continue
        s = s.droplevel('date')
        resid[d] = neutral_scores(s, chars_by_date[d])
        exposures, _ = factor_scores(chars_by_date[d])
        worst = max(worst, exposures.reindex(s.index).fillna(0).corrwith(resid[d]['size_oriented_resid']).abs().max())
    panel = scores.join(pd.concat(resid, names=['date', 'permno']), how='inner')
    raw_vs_neutral = panel.groupby('date').apply(lambda g: g[RAW].corr(g[NEUTRAL])).mean()
    print(f'Check: residual vs factor scores, max |corr| {worst:.1e}; raw vs neutral score, mean corr {raw_vs_neutral:.2f}')
    out_file = paths.result_file('score_neutral', REGION, UNIVERSE, VARIANT)
    panel[['size_oriented_resid', NEUTRAL]].reset_index().to_csv(out_file, index=False)

    # ---- 2. compare raw and neutral ----
    factors = pd.read_csv(report_dir / 'factor_returns.csv', index_col=0)
    ret = data[RET_NAME]
    dates = panel.index.get_level_values('date').unique().sort_values()
    periods = make_periods(list(dates))
    tables, summary = {}, {}
    for name, col in [('raw', RAW), ('neutral', NEUTRAL)]:
        s = panel[col]
        ls = long_short(s, ret)
        ic = rank_ic(s, ret)
        tr = traded(s).reindex(ls.index).fillna(0)
        table = by_period(ls, periods, factors)
        table['rank IC'] = pd.Series({p: ic[(ic.index >= a) & (ic.index <= b)].mean() for p, (a, b) in periods.items()})
        table['ICIR'] = pd.Series({p: (lambda v: v.mean() / v.std() * np.sqrt(12))(ic[(ic.index >= a) & (ic.index <= b)])
                                   for p, (a, b) in periods.items()})
        for bps in COSTS_BPS:
            table[f'Sharpe after {bps}bps'] = pd.Series({
                p: (lambda r: r.mean() / r.std() * np.sqrt(12))((ls - tr * bps / 1e4)[(ls.index >= a) & (ls.index <= b)])
                for p, (a, b) in periods.items()})
        tables[name] = table
        full = table.loc['Full']
        summary[name] = {
            'Sharpe': full['Sharpe'], 'alpha ann. %': full['alpha ann. %'], 'alpha t (NW)': full['alpha t (NW)'],
            'factor R2': full['factor R2'], 'rank IC': full['rank IC'], 'ICIR': full['ICIR'],
            'one-way turnover %/mo': tr.mean() / 2 * 100, 'break-even bps': ls.mean() / tr.mean() * 1e4,
            **{f'Sharpe after {bps}bps': full[f'Sharpe after {bps}bps'] for bps in COSTS_BPS},
            **long_only_stats(s, data, dates),
        }
        print(f'{name}: done')

    by_p = pd.concat(tables, names=['score', 'period'])
    summ = pd.DataFrame(summary)
    by_p.to_csv(report_dir / 'neutral_score_by_period.csv')
    summ.to_csv(report_dir / 'neutral_score_summary.csv')
    print(f'\n==== {LABEL}: raw vs factor-neutral score ====')
    print(summ.round(2).to_string())
    print('\nBy period')
    print(by_p[['Sharpe', 'alpha t (NW)', 'factor R2', 'rank IC', 'Sharpe after 10bps']].round(2).to_string())
    print(f'\nNeutral score saved to {out_file}; tables to {report_dir}')
