# %%
"""
Transfer-coefficient proxy: how much of a stock score survives a long-only, benchmark-relative portfolio, the
binding constraint an optimizer faces (no risk model, sector or turnover constraints here).

Each month, for a universe's market-cap-weighted benchmark b and a score z (0 for stocks without a score):
    long-only portfolio   w = max(b + c * z, 0), renormalised to sum to 1,
                          with c set so the active share sum|w - b| / 2 equals a target;
    unconstrained         a = c' * (z - mean z), scaled to the same active share (long/short, for reference);
    TC                    cross-sectional correlation between the active weights w - b and z.
Reports TC, the rank IC of the score, the active return, tracking error and information ratio (gross and after
trading costs), for each run and active-share target, overall and by period.

Run from the project root after the backtests, e.g.
    python analysis/transfer_coefficient.py                                  largecap, equal- and value-weighted runs
    python analysis/transfer_coefficient.py --universe largecap001 --runs ew
Output: printed and saved to result/transfer_coefficient_<universe>.csv (and _by_period.csv).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, for src/ and backtest.py

import argparse

import numpy as np
import pandas as pd

from analysis.backtest_report import make_periods
from src.constants import DataPaths, factor_chars, run_variant
from src.preprocessing import read_backtest_data

TARGETS = (0.15, 0.30, 0.50)     # active share of the long-only portfolio
COSTS_BPS = (10, 25)


def grid(s, dates, stocks, fill=0.0):
    return s.unstack().reindex(index=dates, columns=stocks).fillna(fill).to_numpy()


def long_only(b, z, target, iters=60):
    """w = max(b + c z, 0) / sum, with c found by bisection so that the active share is `target` (one month)."""
    def weights(c):
        w = np.maximum(b + c * z, 0)
        return w / w.sum()

    def share(c):
        return np.abs(weights(c) - b).sum() / 2

    lo, hi = 0.0, 1e-4
    while share(hi) < target and hi < 1e6:
        hi *= 2
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if share(mid) < target else (lo, mid)
    return weights(hi)


def evaluate(active, ret, dates):
    """Monthly active return and the trading of the active positions."""
    r = pd.Series((active * ret).sum(axis=1), index=dates)
    traded = pd.Series(np.r_[np.nan, np.abs(np.diff(active, axis=0)).sum(axis=1)], index=dates)
    return r, traded


def stats(r, traded, tc=None, ic=None):
    t = traded.fillna(0)
    out = {
        'active return %/yr': r.mean() * 1200,
        'tracking error %/yr': r.std() * np.sqrt(12) * 100,
        'IR': r.mean() / r.std() * np.sqrt(12),
        'one-way turnover %/mo': t.mean() / 2 * 100,
        'break-even bps': r.mean() / t.mean() * 1e4,
    }
    for bps in COSTS_BPS:
        net = r - t * bps / 1e4
        out[f'IR after {bps}bps'] = net.mean() / net.std() * np.sqrt(12)
    if tc is not None:
        out['TC'] = tc.mean()
    if ic is not None:
        out['rank IC'] = ic.mean()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--universe', default='largecap')
    parser.add_argument('--region', default=None)
    parser.add_argument('--runs', nargs='+', default=['ew', 'vw'], choices=['ew', 'vw'],
                        help="runs to compare: 'ew' (equal-weighted) and/or 'vw' (value-weighted)")
    parser.add_argument('--score', default='size_oriented_norm')
    args = parser.parse_args()
    pd.set_option('display.width', 250)

    paths = DataPaths()
    features = factor_chars()                    # the ew and vw runs use the default trees
    ret_name = 'gross_returns' if args.universe is None else 'ret'
    data = read_backtest_data(features, ret_name, region=args.region, universe=args.universe).swaplevel(0, 1).sort_index()

    results, by_period = {}, {}
    for run in args.runs:
        f = paths.result_file('score', args.region, args.universe, run_variant(run == 'ew'))
        if not f.exists():
            print(f'{run}: no scores ({f}), skipped')
            continue
        z_all = pd.read_csv(f, usecols=['date', 'permno', args.score]).set_index(['date', 'permno'])[args.score]
        d = data[data.index.get_level_values('date').isin(z_all.index.get_level_values('date').unique())]
        dates = d.index.get_level_values('date').unique().sort_values()
        stocks = d.index.get_level_values('permno').unique()
        cap = grid(d['mkt_cap'], dates, stocks)
        ret = grid(d[ret_name], dates, stocks)
        held = grid(d[ret_name].notna().astype(float), dates, stocks)
        b = cap * held / (cap * held).sum(axis=1, keepdims=True)                 # cap-weighted benchmark
        z = grid(z_all, dates, stocks) * held                                    # 0 for stocks without a score
        periods = make_periods(list(dates))

        # rank IC of the score among scored stocks
        zr = pd.concat([z_all.rename('z'), d[ret_name].rename('r')], axis=1, join='inner').dropna()
        ic = zr.groupby('date').apply(lambda g: g['z'].corr(g['r'], method='spearman')).reindex(dates)

        for target in TARGETS:
            w = np.vstack([long_only(b[t], z[t], target) for t in range(len(dates))])
            active = w - b
            tc = pd.Series([np.corrcoef(active[t], z[t])[0, 1] for t in range(len(dates))], index=dates)
            r_lo, tr_lo = evaluate(active, ret, dates)

            zc = (z - (z * held).sum(axis=1, keepdims=True) / held.sum(axis=1, keepdims=True)) * held
            a_u = zc * (target / (np.abs(zc).sum(axis=1, keepdims=True) / 2))
            r_u, tr_u = evaluate(a_u, ret, dates)

            key = (run, f'{target:.0%}')
            results[key + ('long-only',)] = stats(r_lo, tr_lo, tc, ic) | {'holdings': (w > 0).sum(axis=1).mean()}
            results[key + ('unconstrained',)] = stats(r_u, tr_u, None, ic)
            for name, (a, e) in periods.items():
                m = (dates >= a) & (dates <= e)
                by_period[key + (name,)] = {'TC': tc[m].mean(), 'IR long-only': stats(r_lo[m], tr_lo[m])['IR'],
                                            'IR long-only after 10bps': stats(r_lo[m], tr_lo[m])['IR after 10bps'],
                                            'IR unconstrained': stats(r_u[m], tr_u[m])['IR']}
        print(f'{run}: done ({len(dates)} months, median {int(held.sum(axis=1).mean())} stocks, '
              f'{int((z != 0).sum(axis=1).mean())} scored)')

    table = pd.DataFrame(results).T.rename_axis(['run', 'active share', 'portfolio'])
    periods_t = pd.DataFrame(by_period).T.rename_axis(['run', 'active share', 'period'])
    name = paths.label(args.region, args.universe)
    table.to_csv(paths.output / f'transfer_coefficient_{name}.csv')
    periods_t.to_csv(paths.output / f'transfer_coefficient_{name}_by_period.csv')
    print(f'\n==== {name}: {args.score} in a long-only portfolio vs cap-weighted benchmark ====')
    print(table.round(2).to_string())
    print('\nBy period')
    print(periods_t.round(2).to_string())


if __name__ == '__main__':
    main()
