# %%
"""
Runs the reporting pipeline and builds the comparison presentation, presentation/backtest_comparison.pdf: the runs
side by side. The detailed deck of any one model is make_detail.py.

For each run in RUNS whose backtest results exist (e.g. result/ret_largecap.csv or ret_largecap_vw.csv, from
backtest.py):
    1. analysis/backtest_report.py      performance tables and factor returns   -> e.g. result/report_largecap/
    2. analysis/hedge_analysis.py       factor-hedged portfolio after costs
    3. analysis/neutralize_scores.py    the score made neutral to the factor scores (size_oriented_resid_norm)
    4. analysis/hedge_score.py          hedge ratios for the scores from each model's rebuilt history (slow)
    5. analysis/factor_spanning.py      the factor-hedged scores: spanning, by period, horizon, size and beta, by size
    6. analysis/turnover_controls.py    partial rebalancing and signal averaging, hedged SDF and hedged score
then presentation/make_comparison.py puts all runs side by side.

Run from the project root:
    python make_presentation.py                        all runs in RUNS, every step
    python make_presentation.py --skip-existing        reuse the analyses already done
    python make_presentation.py --only largecap largecap_vw
    python make_presentation.py --png                  also page previews in presentation/deck_preview/
    python make_detail.py largecap_val                 one model in detail, its own PDF (see make_detail.py)

Company data (regions), instead of RUNS:
    python make_presentation.py --regions GL                        one region
    python make_presentation.py --regions GL US EU UK JP AP EM      several regions, side by side in one PDF
    python make_presentation.py --regions GL US --variant vw        value-weighted runs of those regions
    python make_presentation.py --regions GL US --variant EI_sub_val   any other variant of those regions
These write presentation/backtest_comparison_regions.pdf (--output to change it). The regions' backtests must
exist first: in backtest.py (and build_trees.py) set regions = ['GL', 'US', ...] and universes = [None].
One comparison holds at most 8 runs.
Each step's output goes to result/logs/<step>_<run>.log.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from src.constants import DataPaths

# region (company data, e.g. 'GL'), universe (characteristic files), variant (None = equal-weighted baseline; else the
# run's file-name part, e.g. 'vw' value-weighted, 'slow3' slow-characteristic depth-3 trees, 'val' validated pruning -
# see run_variant in src/constants.py); the comparison deck shows these runs as its columns (at most 8)
RUNS = [
    (None, 'full', None),
    (None, 'largecap', None),
    (None, 'largecap001', None),
    (None, 'largecap', 'vw'),
    (None, 'largecap', 'slow3'),
    (None, 'largecap', 'val'),
    (None, 'largecap', 'slow3_val'),
    (None, 'largecap', 'screen3_val'),
]
# Company data, set here instead of using --regions, for example:
#   one region:        RUNS = [('GL', None, None)]
#   several regions:   RUNS = [(region, None, None) for region in ['GL', 'US', 'EU', 'UK', 'JP', 'AP', 'EM']]
#   regions and universes side by side (up to 8 runs):
#                      RUNS = [('GL', None, None), ('US', None, None), (None, 'largecap', None)]

ROOT = Path(__file__).resolve().parent
STEPS = [   # script, file in the report folder it writes last
    ('analysis/backtest_report.py', 'sdf_returns_by_period.csv'),
    ('analysis/hedge_analysis.py', 'hedged_score_by_period.csv'),
    ('analysis/neutralize_scores.py', 'neutral_score_summary.csv'),
    ('analysis/hedge_score.py', 'hedge_betas_rebuilt_fit.csv'),      # hedge ratios the next steps use
    ('analysis/factor_spanning.py', 'signal_hedged_by_size.csv'),
    ('analysis/turnover_controls.py', 'turnover_controls_score_summary.csv'),
]


def run_name(region, universe, variant):
    """Short name of a run, e.g. 'largecap', 'largecap_vw', 'GL'."""
    return DataPaths().label(region, universe, variant)


def run_token(region, universe, variant):
    """A run as make_comparison.py --runs takes it: [REGION@][UNIVERSE][:vw]."""
    return (f'{region}@' if region else '') + (universe or '') + (f':{variant}' if variant else '')


def run_args(region, universe, variant):
    """Command-line arguments the report scripts take for a run."""
    args = [universe] if universe else []
    if region:
        args += ['--region', region]
    return args + (['--variant', variant] if variant else [])


def run_step(script, args, log_file, env):
    """Run one script from the project root, output to its log; returns True on success."""
    start = time.time()
    with open(log_file, 'w') as log:
        result = subprocess.run([sys.executable, script, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    status = 'done' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
    print(f'    {script:38s} {status:18s} {time.time() - start:5.0f}s   log: {log_file.relative_to(ROOT)}')
    if result.returncode != 0:
        tail = log_file.read_text().splitlines()[-15:]
        print('      ' + '\n      '.join(tail))
    return result.returncode == 0


def run_analyses(region, universe, variant, logs, env, skip_existing=False, steps=None):
    """The analysis steps (STEPS, or `steps`: (script, file it writes last[, extra arguments])) for one run, stopping
    at the first failure; returns the failed script or None."""
    name = run_name(region, universe, variant)
    report_dir = ROOT / DataPaths().result_file('report', region, universe, variant, ext=None)
    for script, step_output, *extra in (steps or STEPS):
        if skip_existing and step_output and (report_dir / step_output).exists():
            print(f'    {script:38s} {"exists, skipped":18s}')
            continue
        args = run_args(region, universe, variant) + (extra[0] if extra else [])
        if not run_step(script, args, logs / f'{Path(script).stem}_{name}.log', env):
            return script                               # later steps need this one's output
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-existing', action='store_true', help='skip steps whose output already exists')
    parser.add_argument('--only', nargs='+', metavar='RUN', help='only these runs, e.g. full largecap_vw GL')
    parser.add_argument('--regions', nargs='+', metavar='REGION', help='company-data regions to run instead of RUNS, e.g. GL US EU')
    parser.add_argument('--variant', help="with --regions: the variant of those regions' runs, e.g. vw or EI_sub_val")
    parser.add_argument('--output', help='PDF name in presentation/ (default: backtest_comparison.pdf, '
                                         'or backtest_comparison_regions.pdf with --regions)')
    parser.add_argument('--png', action='store_true', help='also save page previews of the PDF')
    args = parser.parse_args()

    paths = DataPaths()
    logs = ROOT / paths.output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'MPLBACKEND': 'Agg'}          # charts are saved, not shown
    failed = []
    runs = [(region, None, args.variant) for region in args.regions] if args.regions else RUNS
    output = args.output or ('backtest_comparison_regions.pdf' if args.regions else 'backtest_comparison.pdf')

    for region, universe, variant in runs:
        name = run_name(region, universe, variant)
        if args.only and name not in args.only:
            continue
        if not (ROOT / paths.result_file('ret', region, universe, variant)).exists():
            print(f'{name}: no backtest results ({paths.result_file("ret", region, universe, variant)}), skipped')
            continue
        print(f'{name}:')
        failed_step = run_analyses(region, universe, variant, logs, env, args.skip_existing)
        if failed_step:
            failed.append(f'{name}: {failed_step}')

    # the comparison shows every run with results, including those --only left out of this session
    compared = [run_token(region, universe, variant) for region, universe, variant in runs
                if (ROOT / paths.result_file('report', region, universe, variant, ext=None) / STEPS[0][1]).exists()]
    print('comparison:')
    if not compared:
        print('    no run has results to compare, skipped')
    else:
        compare_args = ['--runs', *compared, '--output', output] + (['--png'] if args.png else [])
        ok = run_step('presentation/make_comparison.py', compare_args, logs / 'make_comparison.log', env)
        if ok:
            print('    ' + (logs / 'make_comparison.log').read_text().strip().splitlines()[-1])
        else:
            failed.append('comparison: presentation/make_comparison.py')

    if failed:
        print('\nFailed steps:\n  ' + '\n  '.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
