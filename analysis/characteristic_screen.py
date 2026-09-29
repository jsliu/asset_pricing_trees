# %%
"""
Screens every characteristic file in characteristics/ as a long/short signal in the large-cap universe, to choose
the characteristics the trees split on.

For each characteristic, each month: the stocks in the large-cap universe (a return in RET_largecap.csv and a positive
market cap in LME_largecap.csv) are given score weights as the characteristic factors are (rank-normalised, demeaned,
100% long and 100% short, src.functions._fac_wei), and hold for the next month's return (RET at the same date is the
return of the following month). The sign of each characteristic is the sign of its full-sample mean long/short
return, so every signal is read in its profitable direction.

Measures:
    performance        Sharpe ratio and t-stat, full sample and by period
    decay over time    Sharpe ratio 2000-2016 against before 2000
    decay over horizon the same weights held k = 1, 3, 6, 12 months later: mean return relative to k = 1 (retention)
    turnover           one-way monthly turnover of the weights
Ranking: the average rank of full-sample Sharpe, Sharpe 2000-2016 and 6-month retention. The 9 are then picked in
that order, skipping any whose long/short returns correlate above MAX_CORR with one already picked.

The choice uses the whole sample, backtest years included, so it is in-sample for a backtest from 1980: the
1964-1979 column shows how each looked before the backtest starts.

Run from the project root:
    python analysis/characteristic_screen.py
Saves result/characteristic_screen_largecap.csv and result/characteristic_screen_largecap_corr.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, for src/

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.constants import Chars, DataPaths
from src.functions import _fac_wei

UNIVERSE = 'largecap'
N_PICK = 9
MAX_CORR = 0.6
HORIZONS = (1, 3, 6, 12)
PERIODS = {'1964-1979': (0, 19791231), '1980s': (19800101, 19891231), '1990s': (19900101, 19991231),
           '2000s': (20000101, 20091231), '2010-2016': (20100101, 20991231)}
# files that are not a single characteristic, duplicates, or size (the trees always split on size)
SKIP = {'ret', 'lme', 'date', 'rf_factor', 'threshold0001', 'npformartinlettau', 'charall_na_rm', 'charall_na_rm_count',
        'charall_largecap_1000', 'resd_var'}


def read_wide(path, permnos=None):
    """date x permno table of one characteristic file, +-inf as missing."""
    wide = pd.read_csv(path, index_col=0)
    wide.columns = wide.columns.str.split('.').str[-1].astype(int)
    if permnos is not None:
        wide = wide.reindex(columns=permnos)
    return wide.replace([np.inf, -np.inf], np.nan).astype(float)


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(12) if len(x) > 12 else np.nan


def screen_one(name, path, ret, universe):
    """Monthly long/short returns (held 1..12 months later) and statistics of one characteristic."""
    char = read_wide(path, ret.columns).reindex(ret.index).where(universe)
    values = char.to_numpy()
    weights = np.full_like(values, np.nan)
    for i, row in enumerate(values):
        ok = ~np.isnan(row)
        if ok.sum() >= 50 and np.unique(row[ok]).size > 1:
            weights[i, ok] = _fac_wei(row[ok])
    w = pd.DataFrame(weights, index=ret.index, columns=ret.columns)

    ls = {}
    for k in HORIZONS:
        r = ret.shift(-(k - 1))                               # return k months after the signal date
        r = r.sub(r.where(w.notna()).mean(axis=1), axis=0)    # excess of the scored stocks' mean
        ls[k] = (w * r).sum(axis=1, min_count=1)
    ls = pd.DataFrame(ls).dropna(subset=[1])
    if len(ls) < 120:
        return name, None, None
    sign = np.sign(ls[1].mean())
    ls = ls * sign

    one = ls[1]
    turnover = w.fillna(0).diff().abs().sum(axis=1).iloc[1:].reindex(one.index).mean() / 2
    stats = {
        'sign': int(sign),
        'first date': int(one.index.min()),
        'stocks/mo': int(w.notna().sum(axis=1).loc[one.index].median()),
        'Sharpe': sharpe(one),
        't-stat': one.mean() / one.std() * np.sqrt(len(one)),
        'ann. return %': one.mean() * 1200,
        **{f'Sharpe {p}': sharpe(one[(one.index >= a) & (one.index <= b)]) for p, (a, b) in PERIODS.items()},
        'Sharpe pre-2000': sharpe(one[one.index < 20000101]),
        'Sharpe 2000-2016': sharpe(one[one.index >= 20000101]),
        **{f'retention {k}m': ls[k].mean() / one.mean() for k in HORIZONS[1:]},
        'Sharpe held 6m later': sharpe(ls[6].dropna()),
        'one-way turnover %/mo': turnover * 100,
    }
    return name, stats, one


# %%
if __name__ == '__main__':
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    paths = DataPaths()
    folder = paths.input_data
    ret = read_wide(folder / f'RET_{UNIVERSE}.csv')
    mcap = read_wide(folder / f'LME_{UNIVERSE}.csv', ret.columns).reindex(ret.index)
    universe = ret.notna() & (mcap > 0)
    keep = universe.columns[universe.any()]                   # stocks ever in the universe
    universe, ret = universe[keep], ret[keep].where(universe[keep])
    print(f'{UNIVERSE}: {ret.shape[1]} stocks, {ret.index.min()}-{ret.index.max()}, median {universe.sum(axis=1).median():.0f} a month')

    files = sorted(f for f in folder.glob('*.csv') if '_largecap' not in f.stem.lower() and f.stem.lower() not in SKIP)
    print(f'Screening {len(files)} characteristics')
    results = Parallel(n_jobs=4, verbose=5)(delayed(screen_one)(f.stem, f, ret, universe) for f in files)

    stats = pd.DataFrame({name: s for name, s, _ in results if s is not None}).T
    returns = pd.DataFrame({name: r for name, _, r in results if r is not None})
    skipped = [name for name, s, _ in results if s is None]
    if skipped:
        print('Too little data, left out:', ', '.join(skipped))

    ranks = pd.DataFrame({c: stats[c].astype(float).rank(ascending=False)
                          for c in ['Sharpe', 'Sharpe 2000-2016', 'retention 6m']})
    stats['rank score'] = ranks.mean(axis=1)
    stats = stats.sort_values('rank score')

    corr = returns.corr()
    picked, dropped = [], {}
    for name in stats.index:
        clash = [p for p in picked if abs(corr.loc[name, p]) > MAX_CORR]
        if clash:
            dropped[name] = clash[0]
        else:
            picked.append(name)
        if len(picked) == N_PICK:
            break
    stats['picked'] = stats.index.isin(picked)
    stats['too close to'] = pd.Series(dropped)
    current = {c.lower() for c in Chars().__dict__.values()}
    stats['in current 9'] = stats.index.str.lower().isin(current)

    out = paths.output
    stats.to_csv(out / f'characteristic_screen_{UNIVERSE}.csv')
    corr.loc[picked, picked].to_csv(out / f'characteristic_screen_{UNIVERSE}_corr.csv')

    show = ['sign', 'first date', 'Sharpe', 't-stat', 'Sharpe 1964-1979', 'Sharpe pre-2000', 'Sharpe 2000-2016',
            'retention 3m', 'retention 6m', 'retention 12m', 'one-way turnover %/mo', 'rank score', 'picked',
            'too close to', 'in current 9']
    print(f'\n==== {UNIVERSE}: characteristic long/short, best first ====')
    print(stats[show].round(2).to_string())
    print(f'\nPicked ({N_PICK}, long/short correlation at most {MAX_CORR}): {", ".join(picked)}')
    print('\nCorrelation of the picked long/short returns')
    print(corr.loc[picked, picked].round(2).to_string())
    cur = [c for c in stats.index if stats.loc[c, 'in current 9']]
    print(f'\nCurrent 9: {", ".join(cur)}')
    print(f'Saved to {out}')
