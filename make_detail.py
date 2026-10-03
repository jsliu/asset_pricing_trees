# %%
"""
The detailed presentation of one model, or several, each as its own PDF: presentation/backtest_<model>.pdf, e.g.
backtest_largecap.pdf or backtest_largecap_val.pdf (presentation/make_deck.py). make_presentation.py compares the
models side by side.

A model is any backtest with results in result/, named in one of three ways:
    its file-name label             largecap, largecap_val, largecap_slow3_val, full, largecap001, GL, GL_vw
    [REGION@][UNIVERSE][:variant]   largecap:val, full, GL@, GL@:vw   (as make_comparison.py --runs takes it)
    a run in make_presentation.RUNS by its label
Before the deck, the analysis steps the deck reads (DETAIL_STEPS: make_presentation.STEPS and
analysis/backtest_analysis.py, the tree scores combined with the factor scores) are run for the model: all of them, only
the missing ones (--skip-existing), or none (--deck-only).

Run from the project root:
    python make_detail.py --model largecap                                the large-cap model, every analysis step, then the deck
    python make_detail.py --model largecap_val full --skip-existing       two models, only the missing analyses
    python make_detail.py --model GL@:vw --deck-only --png                a region's value-weighted run, deck only, with page previews

The two ways of naming the same model ([REGION@] marks a region and can be left out without one; :variant is the run's
file-name suffix, as in the result file names):
    model                                       [REGION@][UNIVERSE][:variant]   file-name label
    large cap, val                              largecap:val                    largecap_val
    large cap, baseline                         largecap                        largecap
    large cap, value-weighted slow d3 val       largecap:vw_slow3_val           largecap_vw_slow3_val
    region GL, baseline                         GL@                             GL
    region GL, value-weighted                   GL@:vw                          GL_vw
    region GL, EI_sub trees, val                GL@:EI_sub_val                  GL_EI_sub_val
e.g. python make_detail.py --model largecap:val is the same as python make_detail.py --model largecap_val.

Company data (a region, no universe; the results are named [REGION_]kind[_variant], e.g. result/GL_ret_EI_sub_val.csv):
    python make_detail.py --model GL                                      region GL, default trees and pruning   (result/GL_ret.csv)
    python make_detail.py --model GL_EI_sub_val                           region GL, EI_sub trees, validated pruning
    python make_detail.py --model GL@:EI_sub_val                          the same model as [REGION@][UNIVERSE][:variant]
    python make_detail.py --model GL_EI US_EI --skip-existing             two regions' EI models, only the missing analyses
The variant's tree set-up (e.g. EI_sub) must be in TREE_SETUPS (src/constants.py), which also gives the model's factor
characteristics; reading company data needs the AIalpha/connector packages. Company data has no market beta, so its
size-and-beta page adds size to the hedge only.
Each step's output goes to result/logs/<step>_<model>.log.
"""
import argparse
import os
import sys

from make_presentation import RUNS, ROOT, STEPS, run_analyses, run_args, run_name, run_step
from src.constants import DataPaths, parse_variant

# the comparison's analysis steps, then the tree scores combined with the factor scores (its own slide)
DETAIL_STEPS = STEPS + [('analysis/backtest_analysis.py', 'combined_scores_pnl.csv', ['--no-plots'])]
UNIVERSES = ['largecap001', 'largecap', 'full']       # characteristic-file universes, longest name first


def parse_model(token):
    """(region, universe, variant) of a model named by its label, a [REGION@][UNIVERSE][:variant] token or a RUNS
    entry's label. Raises ValueError when the name cannot be read."""
    for run in RUNS:
        if run_name(*run) == token:
            return run
    if '@' in token or ':' in token:
        body, _, variant = token.partition(':')
        region, _, universe = body.rpartition('@') if '@' in body else ('', '', body)
        model = region or None, universe or None, variant or None
    else:                                             # a file-name label: [region_]universe[_variant]
        parts = token.split('_')
        at = next((i for i, part in enumerate(parts) if part in UNIVERSES), None)
        if at is None:                                # a region's run: REGION[_variant]
            model = parts[0], None, '_'.join(parts[1:]) or None
        else:
            model = '_'.join(parts[:at]) or None, parts[at], '_'.join(parts[at + 1:]) or None
    parse_variant(model[2])                           # raises on an unknown variant
    if model[0] is None and model[1] is None:
        raise ValueError(f'no universe or region in {token!r}')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', '--models', dest='models', nargs='+', required=True, metavar='MODEL',
                        help='one or more models, e.g. largecap, largecap_val, full, GL, GL@:vw')
    steps = parser.add_mutually_exclusive_group()
    steps.add_argument('--skip-existing', action='store_true', help='run only the analysis steps whose output is missing')
    steps.add_argument('--deck-only', action='store_true', help='no analysis steps, only the deck')
    parser.add_argument('--png', action='store_true', help='also save page previews in presentation/deck_preview/')
    args = parser.parse_args()

    paths = DataPaths()
    logs = ROOT / paths.output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'MPLBACKEND': 'Agg'}         # charts are saved, not shown
    failed = []
    for token in args.models:
        try:
            region, universe, variant = parse_model(token)
        except ValueError as e:
            print(f'{token}: {e}')
            failed.append(token)
            continue
        name = run_name(region, universe, variant)
        print(f'{name}:')
        if not (ROOT / paths.result_file('ret', region, universe, variant)).exists():
            print(f'    no backtest results ({paths.result_file("ret", region, universe, variant)}), skipped')
            failed.append(name)
            continue
        if not args.deck_only:
            failed_step = run_analyses(region, universe, variant, logs, env, args.skip_existing, DETAIL_STEPS)
            if failed_step:
                failed.append(f'{name}: {failed_step}')
                continue
        deck_args = run_args(region, universe, variant) + (['--png'] if args.png else [])
        if run_step('presentation/make_deck.py', deck_args, logs / f'make_deck_{name}.log', env):
            print(f'    Saved {ROOT / "presentation" / paths.result_file("backtest", region, universe, variant, ext="pdf").name}')
        else:
            failed.append(f'{name}: presentation/make_deck.py')

    if failed:
        print('\nFailed:\n  ' + '\n  '.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
