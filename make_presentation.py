# %%
"""
Runs the reporting pipeline and builds the comparison presentation, presentation/backtest_comparison.pdf.

For each run in RUNS whose backtest results exist (result/ret_<suffix>_<universe>.csv, from backtest.py):
    1. backtest_report.py      performance tables and factor returns   -> result/report_<suffix>_<universe>/
    2. hedge_analysis.py       factor-hedged portfolio after costs
    3. turnover_controls.py    partial rebalancing and signal averaging
    (4. presentation/make_deck.py, a PDF for the run alone, with --single-decks)
then presentation/make_comparison.py puts all runs side by side in one PDF.

Run from the project root:
    python make_presentation.py                        all runs in RUNS, every step
    python make_presentation.py --skip-existing        reuse the analyses already done
    python make_presentation.py --only largecap largecap_vw
    python make_presentation.py --single-decks --png   also one PDF per run, and page previews

Company data (regions), instead of RUNS:
    python make_presentation.py --regions GL                        one region
    python make_presentation.py --regions GL US EU UK JP AP EM      several regions, side by side in one PDF
    python make_presentation.py --regions GL US --suffix vw         value-weighted runs of those regions
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

# region (company data, e.g. 'GL'), universe (characteristic files), run suffix ('std' or 'vw' = value-weighted);
# keep in line with UNIVERSES in presentation/make_comparison.py, which sets the deck's columns
RUNS = [
    (None, 'full', 'std'),
    (None, 'largecap', 'std'),
    (None, 'largecap001', 'std'),
    (None, 'largecap', 'vw'),
]
# Company data, set here instead of using --regions, for example:
#   one region:        RUNS = [('GL', None, 'std')]
#   several regions:   RUNS = [(region, None, 'std') for region in ['GL', 'US', 'EU', 'UK', 'JP', 'AP', 'EM']]
#   regions and universes side by side (up to 8 runs):
#                      RUNS = [('GL', None, 'std'), ('US', None, 'std'), (None, 'largecap', 'std')]

ROOT = Path(__file__).resolve().parent
STEPS = [   # script, file in the report folder it writes last
    ('backtest_report.py', 'sdf_returns_by_period.csv'),
    ('hedge_analysis.py', 'hedge_costs_by_period.csv'),
    ('turnover_controls.py', 'turnover_controls_summary.csv'),
]


def run_name(region, universe, suffix):
    """Short name of a run, e.g. 'largecap', 'largecap_vw', 'GL'."""
    return DataPaths().label(region, universe) + ('' if suffix == 'std' else f'_{suffix}')


def run_token(region, universe, suffix):
    """A run as make_comparison.py --runs takes it: [REGION@][UNIVERSE][:SUFFIX]."""
    return (f'{region}@' if region else '') + (universe or '') + f':{suffix}'


def run_args(region, universe, suffix):
    """Command-line arguments the report scripts take for a run."""
    args = [universe] if universe else []
    if region:
        args += ['--region', region]
    return args + ['--suffix', suffix]


def run_step(script, args, log_file, env):
    """Run one script from the project root, output to its log; returns True on success."""
    start = time.time()
    with open(log_file, 'w') as log:
        result = subprocess.run([sys.executable, script, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    status = 'done' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
    print(f'    {script:32s} {status:18s} {time.time() - start:5.0f}s   log: {log_file.relative_to(ROOT)}')
    if result.returncode != 0:
        tail = log_file.read_text().splitlines()[-15:]
        print('      ' + '\n      '.join(tail))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-existing', action='store_true', help='skip steps whose output already exists')
    parser.add_argument('--only', nargs='+', metavar='RUN', help='only these runs, e.g. full largecap_vw GL')
    parser.add_argument('--regions', nargs='+', metavar='REGION', help='company-data regions to run instead of RUNS, e.g. GL US EU')
    parser.add_argument('--suffix', default='std', help="run suffix for --regions: 'std' or 'vw' (value-weighted)")
    parser.add_argument('--output', help='comparison PDF name in presentation/ (default: backtest_comparison.pdf, '
                                         'or backtest_comparison_regions.pdf with --regions)')
    parser.add_argument('--single-decks', action='store_true', help='also build one PDF per run (make_deck.py)')
    parser.add_argument('--png', action='store_true', help='also save page previews of the PDFs')
    args = parser.parse_args()

    paths = DataPaths()
    logs = ROOT / paths.output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'MPLBACKEND': 'Agg'}          # charts are saved, not shown
    failed, compared = [], []
    runs = [(region, None, args.suffix) for region in args.regions] if args.regions else RUNS
    output = args.output or ('backtest_comparison_regions.pdf' if args.regions else 'backtest_comparison.pdf')

    for region, universe, suffix in runs:
        name = run_name(region, universe, suffix)
        if args.only and name not in args.only:
            continue
        if not (ROOT / paths.result_file('ret', suffix, region, universe)).exists():
            print(f'{name}: no backtest results ({paths.result_file("ret", suffix, region, universe)}), skipped')
            continue
        print(f'{name}:')
        report_dir = ROOT / paths.result_file('report', suffix, region, universe, ext=None)
        steps = STEPS + ([('presentation/make_deck.py', None)] if args.single_decks else [])
        for script, step_output in steps:
            if args.skip_existing and step_output and (report_dir / step_output).exists():
                print(f'    {script:32s} {"exists, skipped":18s}')
                continue
            extra = ['--png'] if (args.png and script.endswith('make_deck.py')) else []
            ok = run_step(script, run_args(region, universe, suffix) + extra, logs / f'{Path(script).stem}_{name}.log', env)
            if not ok:
                failed.append(f'{name}: {script}')
                break                                   # later steps need this one's output
        if (report_dir / STEPS[0][1]).exists():
            compared.append(run_token(region, universe, suffix))

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
