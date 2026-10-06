# %%
"""
Does the factor-hedged stock score give an alpha source independent of the characteristic factors?

The raw score's long/short is largely the factors it is built from (factor R2 ~ 0.75), so this analysis works with
the *hedged* score: the score's long/short minus its factor exposure, as the backtest writes it (the score file's
size_oriented_hedged, final_score_hedged and sdf_weight_hedged positions; hedge ratios estimated at each refit the
way the SDF's are, from the model's own score rebuilt over the previous 10 years). That leaves the part of the score
the factors do not explain. The hedges with size and beta added (5.) and within each size quintile (6.) use the
ratios of analysis/hedge_score.py (hedge_betas_rebuilt.csv; run it first).
Three questions are answered:

    1. Standalone        Sharpe ratio, return, alpha against the factors (which is now its own return), the residual
                         factor R2, turnover and trading costs, by period   -> signal_hedged_by_period.csv
    2. Spanning          whether adding the hedged score to a tangency portfolio of the 9 characteristic factors
                         improves its Sharpe (Jobson-Korkie/Memmel t-test)  -> signal_spanning_sharpe.csv
    3. History           the monthly hedged-score return                    -> signal_hedged_monthly.csv
The same is done for the other stock scores in SCORES (the original score and sdf_weight), saved with the score's
name added, e.g. signal_spanning_sharpe_final_score_norm.csv (see output_name). For the default score also:
    6. By size           the hedged positions' return by the size quintile of the stocks held, and the score
                         built and hedged within each quintile   -> signal_hedged_by_size.csv (+ _monthly.csv)

Run backtest_report.py, hedge_analysis.py and hedge_score.py for the run first, then from the project root:
    python analysis/factor_spanning.py [<universe>] [--variant TAGS]     e.g. ... largecap
Tables are printed and saved to the run's report folder, e.g. result/report_largecap/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, for src/ and analysis/

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.constants import DataPaths, factor_chars, parse_variant
from src.functions import _get_weights
from src.preprocessing import read_backtest_data
from analysis.backtest_report import make_periods, size_buckets
from analysis.hedge_analysis import build_positions, score_weights, hedge_score, score_stats


def _run_args():
    """python <script> [<universe>] [--region GL] [--variant TAGS]; universe defaults to 'full' (the same
    convention as analysis/backtest_report.py)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('universe', nargs='?')
    parser.add_argument('--region')
    parser.add_argument('--variant')
    args, _ = parser.parse_known_args()
    universe = args.universe if (args.universe or args.region) else 'full'
    return args.region, universe, args.variant


SCORES = ['size_oriented_score', 'final_score_norm', 'sdf_weight']     # the first is the default score


def output_name(stem, score):
    """File name of a table: <stem>.csv for the default score, <stem>_<score>.csv for the others."""
    return f'{stem}.csv' if score == SCORES[0] else f'{stem}_{score}.csv'


def sharpe(x):
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(12)


def tangency_returns(rets):
    """Return series and weights of the full-sample tangency (maximum-Sharpe) portfolio of a monthly return matrix.
    The weights are in-sample, which is standard for a spanning comparison."""
    w = np.linalg.pinv(rets.cov().to_numpy(dtype=float)) @ rets.mean().to_numpy(dtype=float)
    return pd.Series(rets.to_numpy(dtype=float) @ w, index=rets.index), pd.Series(w, index=rets.columns)


def jobson_korkie(r1, r2):
    """Sharpe-ratio difference of two monthly return series with the Memmel (2003) corrected Jobson-Korkie
    t-statistic. Returns (annualised Sharpe difference, t-statistic of the difference)."""
    a, b = r1.align(r2, join='inner')
    sr1, sr2, rho, t = a.mean() / a.std(), b.mean() / b.std(), a.corr(b), len(a)
    theta = 2 * (1 - rho) + 0.5 * (sr1 ** 2 + sr2 ** 2 - 2 * rho ** 2 * sr1 * sr2)
    return (sr1 - sr2) * np.sqrt(12), (sr1 - sr2) / np.sqrt(theta / t)


def span_periods(dates):
    """Full sample and its two halves, split at the start of the year nearest the middle of the data, e.g.
    1980-1997 / 1998-2016 or 2006-2015 / 2016-2026 (the tangency weights are in-sample either way)."""
    y0, y1 = min(dates) // 10000, max(dates) // 10000
    mid = (y0 + y1 + 1) // 2
    return {'Full': (min(dates), max(dates)), f'{y0}-{mid - 1}': (min(dates), (mid - 1) * 10000 + 1231),
            f'{mid}-{y1}': (mid * 10000 + 101, max(dates))}


def score_decay_table(score, w, ret, horizons=(1, 3, 6, 12)):
    """Horizon retention of a score: its score-weighted long/short (weights `w`) held fixed for k months, and its
    rank IC with the return k months later. Returns at date t are the following month's return (as in
    analysis/characteristic_screen.py). retention = mean(k-month return) / mean(1-month return)."""
    sw = score.unstack()
    ww = w.unstack().reindex(index=sw.index, columns=sw.columns)
    rw = ret.unstack().reindex(index=sw.index, columns=sw.columns)
    ls_by_k, ic_by_k = {}, {}
    for k in horizons:
        r = rw.shift(-(k - 1))                                    # return k months after the signal date
        excess = r.sub(r.where(ww.notna()).mean(axis=1), axis=0)  # net of the scored stocks' mean
        ls_by_k[k] = (ww * excess).sum(axis=1, min_count=1).dropna()
        ics = []
        for t in r.index:
            both = pd.concat([sw.loc[t].rename('s'), r.loc[t].rename('y')], axis=1).dropna()
            if len(both) > 30:
                ics.append(both['s'].corr(both['y'], method='spearman'))
        ic_by_k[k] = pd.Series(ics, dtype=float)
    base = ls_by_k[horizons[0]].mean()
    rows = {}
    for k in horizons:
        ls, ic = ls_by_k[k], ic_by_k[k]
        rows[k] = {'ann. return %': ls.mean() * 1200, 't-stat': ls.mean() / ls.std() * np.sqrt(len(ls)),
                   'Sharpe': ls.mean() / ls.std() * np.sqrt(12),
                   'retention': ls.mean() / base if base else np.nan,
                   'rank IC': ic.mean(), 'ICIR': ic.mean() / ic.std() * np.sqrt(12) if ic.std() > 0 else np.nan}
    return pd.DataFrame(rows).T.rename_axis('horizon months')


def factor_return(s, ret):
    """Score-weighted long/short return of a characteristic series s (date, permno), the same construction as the
    saved factor returns: net of the cross-sectional mean, weights sum|w| = 2."""
    w = s.dropna().groupby('date').transform(lambda x: _get_weights(x, score_weighted=True))
    excess = ret - ret.groupby('date').transform('mean')
    df = pd.concat([w.rename('w'), excess.rename('e')], axis=1, join='inner').dropna()
    return (df['w'] * df['e']).groupby('date').sum()


def extra_factors(region, universe, variant=None):
    """The size (small minus big) and market-beta (high minus low) factors the saved factor set leaves out, over the
    whole sample: (returns with a column per factor, their stock weights, the (date, permno) data read). Size comes
    from market cap, so every run has it; market beta is a characteristic of the characteristic files, so company
    data (a region) has size only."""
    ret_name = 'gross_returns' if universe is None else 'ret'
    chars = factor_chars(parse_variant(variant)[1]) + ([] if region else ['beta'])
    extra = read_backtest_data(chars, ret_name, region, universe).swaplevel(0, 1).sort_index()
    sources = {'size': -extra['mkt_cap'], **({'beta': extra['beta']} if 'beta' in extra else {})}
    rets = pd.DataFrame({k: factor_return(v, extra[ret_name]) for k, v in sources.items()})
    weights = pd.DataFrame({k: v.groupby('date').transform(lambda x: _get_weights(x, score_weighted=True))
                            for k, v in sources.items()})
    return rets, weights, extra


# hedge specifications: name -> the extra factors added to the run's own factors (however many it has)
HEDGE_SPECS = {'factors': [], '+ size': ['size'], '+ size + beta': ['size', 'beta']}


# the backtest's hedged positions of each score (columns of the score file)
HEDGED_COL = {'size_oriented_score': 'size_oriented_hedged', 'final_score_norm': 'final_score_hedged',
              'sdf_weight': 'sdf_weight_hedged'}


def backtest_hedged(p, score):
    """(monthly return, stock positions) of the score's factor-hedged positions as backtest.py wrote them, over the
    run's dates; None when the score file has no such column (a backtest from before it was added)."""
    col = HEDGED_COL[score]
    if col not in pd.read_csv(p['scores_file'], nrows=0).columns:
        return None
    w = pd.read_csv(p['scores_file'], usecols=['date', 'permno', col]).set_index(['date', 'permno'])[col].dropna()
    w = w[w.index.get_level_values('date').isin(p['dates'])]
    return (w * p['ret'].reindex(w.index)).groupby('date').sum(), w


def rebuilt_betas(out_dir, score, universe='all', spec='factors'):
    """Hedge ratios per refit (refit date x factor) that analysis/hedge_score.py estimated from each model's rebuilt
    score history, for a score, a universe ('all' or a size quintile) and a hedge specification."""
    b = pd.read_csv(out_dir / 'hedge_betas_rebuilt.csv')
    b = b[(b['score'] == score) & (b['universe'] == universe) & (b['spec'] == spec)]
    return b.pivot(index='refit', columns='factor', values='beta')


# %%
if __name__ == '__main__':
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    REGION, UNIVERSE, VARIANT = _run_args()
    LABEL = DataPaths().label(REGION, UNIVERSE, VARIANT)
    paths = DataPaths()
    out_dir = paths.result_file('report', REGION, UNIVERSE, VARIANT, ext=None)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (out_dir / 'hedge_betas_rebuilt.csv').exists():
        raise SystemExit(f'no hedge ratios in {out_dir} - run analysis/hedge_score.py for this run first')
    p = build_positions(REGION, UNIVERSE, VARIANT)
    factors = p['factors']
    # extra hedge factors the saved factor set excludes (factor_chars drops size): size and, where the data has it,
    # market beta; a specification needing a factor the run lacks is left out
    extra_rets, fw_extra, extra = extra_factors(REGION, UNIVERSE, VARIANT)
    size_f = extra_rets['size']
    fw_extra = fw_extra[fw_extra.index.get_level_values('date') >= p['dates'].min()]
    specs = {spec: (pd.concat([factors, extra_rets[added]], axis=1, join='inner') if added else factors,
                    pd.concat([p['fw'], fw_extra[added]], axis=1) if added else p['fw'])
             for spec, added in HEDGE_SPECS.items() if all(a in extra_rets for a in added)}
    for score in SCORES:
        # ---------------- the hedged score, as the backtest wrote it ----------------
        bh = backtest_hedged(p, score)
        if bh is None:
            # a score file from before the backtest wrote this hedged score: leave the score out (rerun backtest.py)
            # and remove any tables left from an earlier run
            print(f'\n######## {LABEL}, {score}: no {HEDGED_COL[score]} in the score file (rerun backtest.py), left out')
            for stem in ['signal_hedged_by_period', 'signal_spanning_sharpe', 'signal_hedged_monthly',
                         'signal_score_decay', 'signal_hedge_decomposition']:
                (out_dir / output_name(stem, score)).unlink(missing_ok=True)
            continue
        w, ls = score_weights(p, score)                          # unhedged, for the size/beta hedges and the decay
        hedged, w_hedged = bh                                    # hedged return series + its stock positions
        print(f'\n######## {LABEL}, {score}: hedged over {len(hedged)} months {hedged.index.min()}-{hedged.index.max()}')

        # ---------------- 1. standalone quality, by period ----------------
        periods = make_periods(sorted(hedged.index))
        table = score_stats(hedged, w_hedged, factors, periods)
        table.to_csv(out_dir / output_name('signal_hedged_by_period', score))
        print('\n==== 1. Hedged score: return, alpha against the factors, turnover and costs ====')
        print(table[['Sharpe', 'ann. return %', 'alpha ann. %', 'alpha t (NW)', 'factor R2',
                     'one-way turnover %/mo', 'break-even cost bps', 'Sharpe after 10bps']].round(2).to_string())
        check = out_dir / 'hedged_score_nodes_by_period.csv'
        if score == SCORES[0] and check.exists():            # the same series from hedge_score.py, for verification
            old = pd.read_csv(check).set_index(['portfolio', 'period'])
            old = old.loc['size_oriented_score long/short, hedged on rebuilt history']
            print('\nCheck vs hedge_score.py (same series)')
            print(old[['Sharpe', 'alpha t (NW)', 'factor R2']].round(2).to_string())

        # ---------------- 2. spanning: does the hedged score improve the factor tangency? ----------------
        rows = {}
        for name, (a, b) in span_periods(hedged.index).items():
            f = factors[(factors.index >= a) & (factors.index <= b)]
            base, _ = tangency_returns(f)
            ext = pd.concat([f, hedged.rename('hedged')], axis=1, join='inner').dropna()
            r, w2 = tangency_returns(ext)
            ds, z = jobson_korkie(r, base.reindex(r.index))
            rows[name] = {'factors only SR': sharpe(base), 'hedged SR': sharpe(r), 'dSharpe': ds, 'JK t': z,
                          'weight on hedged': w2['hedged']}
        spanning = pd.DataFrame(rows).T.rename_axis('period')
        spanning.to_csv(out_dir / output_name('signal_spanning_sharpe', score))
        print('\n==== 2. Spanning: tangency portfolio of the factors, with and without the hedged score ====')
        print(spanning.round(3).to_string())

        # ---------------- 3. the monthly series, for the deck's chart ----------------
        hedged.rename('hedged').to_csv(out_dir / output_name('signal_hedged_monthly', score))

        # ---------------- 4. horizon retention: does the information persist? ----------------
        # the hedged positions (score weights minus the hedge) held for k months, their rank IC with the month-k
        # return; the unhedged score's for comparison
        score_series = pd.read_csv(paths.result_file('score', REGION, UNIVERSE, VARIANT),
                                   usecols=['date', 'permno', score]).set_index(['date', 'permno'])[score].dropna()
        decay = score_decay_table(w_hedged, w_hedged, p['ret'], horizons=(1, 3, 6, 12))
        decay.to_csv(out_dir / output_name('signal_score_decay', score))
        decay_raw = score_decay_table(score_series, w, p['ret'], horizons=(1, 3, 6, 12))
        decay_raw.to_csv(out_dir / output_name('signal_score_decay_unhedged', score))
        print('\n==== 4. Horizon retention of the hedged score (unhedged below) ====')
        print(decay.round(3).to_string())
        print(decay_raw.round(3).to_string())

        # ---------------- 5. does the alpha survive size and beta hedging? ----------------
        decomp = {}
        for label_, (facs, fws) in specs.items():
            pe = dict(p)
            pe['factors'], pe['fw'] = facs, fws
            hb = rebuilt_betas(out_dir, score, 'all', label_)
            if hb.empty:                                     # hedge_score.py has no ratios for this specification
                continue
            hd, wd = hedge_score(pe, w, ls, hb)
            full = score_stats(hd, wd, factors, {'Full': (hd.index.min(), hd.index.max())}).loc['Full']
            hd2 = hd.dropna()
            m = sm.OLS(hd2, sm.add_constant(size_f.loc[hd2.index])).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
            decomp[label_] = {'Sharpe': full['Sharpe'], 'alpha t (NW)': full['alpha t (NW)'],
                              'factor R2': full['factor R2'], 'size beta': m.params['size'],
                              'size beta t': m.tvalues['size'], 'one-way turnover %/mo': full['one-way turnover %/mo']}
        decomp = pd.DataFrame(decomp).T
        decomp.to_csv(out_dir / output_name('signal_hedge_decomposition', score))
        print('\n==== 5. Hedged score: adding size and beta to the hedge ====')
        print(decomp.round(3).to_string())

        # ---------------- 6. the hedged score by size quintile (the default score) ----------------
        if score == SCORES[0]:
            buckets = size_buckets(extra['mkt_cap'])                         # (date, permno) -> Q1 small ... Q5 large
            # a. where its return comes from: the hedged positions' contribution by the size of the stocks held
            b_ = buckets.reindex(w_hedged.index).to_numpy()
            keys = [w_hedged.index.get_level_values('date'), b_]
            contrib = (w_hedged * p['ret'].reindex(w_hedged.index)).groupby(keys, observed=True).sum().unstack()
            gross = w_hedged.abs().groupby(keys, observed=True).sum().unstack()
            gap = (contrib.sum(axis=1) - hedged.reindex(contrib.index)).abs().max()
            print(f'\nCheck: size contributions vs hedged score return, max abs diff {gap:.1e}')
            contrib.to_csv(out_dir / 'signal_hedged_by_size_monthly.csv')
            # b. the score's long/short built and hedged within each size quintile on its own
            rows = {}
            for b in buckets.cat.categories:
                s_b = score_series[score_series.index.isin(buckets.index[buckets == b])]
                w_b = s_b.groupby('date').transform(lambda x: _get_weights(x, score_weighted=True))
                w_b = w_b[w_b.index.get_level_values('date').isin(p['dates'])]
                ls_b = (w_b * p['ret'].reindex(w_b.index)).groupby('date').sum().reindex(p['dates']).fillna(0)
                hd_b, wh_b = hedge_score(p, w_b, ls_b, rebuilt_betas(out_dir, score, b))
                st = score_stats(hd_b, wh_b, factors, {'Full': (hd_b.index.min(), hd_b.index.max())}).loc['Full']
                rows[b] = {'contribution %/yr': contrib[b].mean() * 1200,
                           'share of return %': contrib[b].mean() / contrib.sum(axis=1).mean() * 100,
                           'share of gross weight %': gross[b].mean() / gross.sum(axis=1).mean() * 100,
                           'stocks/mo': s_b.groupby('date').size().median(),
                           'Sharpe within': st['Sharpe'], 'alpha t within': st['alpha t (NW)'],
                           'turnover within %/mo': st['one-way turnover %/mo'],
                           'break-even within bps': st['break-even cost bps'],
                           'Sharpe after 10bps within': st['Sharpe after 10bps']}
            by_size = pd.DataFrame(rows).T.rename_axis('size bucket')
            by_size.to_csv(out_dir / 'signal_hedged_by_size.csv')
            print('\n==== 6. Hedged score by size quintile ====')
            print(by_size.round(2).to_string())
    print(f'\nHedged scores saved to {out_dir}')
