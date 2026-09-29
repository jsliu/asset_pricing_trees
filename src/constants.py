from typing import List, Union, ClassVar
from pathlib import Path
from dataclasses import dataclass
from itertools import combinations
from collections.abc import Iterator

import numpy as np

@dataclass(frozen=True)
class Parameters:
    tree_depth: int = 4
    n_splits: int = 2
    n_chars: int = 2
    mean_shrinkage: ClassVar[np.ndarray] = np.arange(0.1, 1, 0.1)
    ridge_lambda: ClassVar[np.ndarray] = 0.1 ** np.arange(1, 9, 1)
    cv_splits: int = 2
    k_min: int = 5
    k_max: int = 50
    # test_size: int = 826
    # test_size: int = 36
    test_size: int = 120


@dataclass(frozen=True)
class Columns:
    date_col: str = 'date'
    # size_col: str = 'market_value_gbp'
    size_col: str = 'mkt_cap'
    id_col: str = 'permno'
    returns_col: str = 'ret'
    # returns_col: str = 'total_return_1d_gbp'
    # returns_col: str = 'excess_returns'
    # returns_col: str = 'X1MFwdReturnLoc'
    w_returns_col: str = 'weighted_ret'
    col_sep: str = '/'
    node_col: str = 'node'
    port_col: str = 'port'
    features_col: str = 'features'
    comb_col: str = 'combination'


@dataclass(init=False)
class Years:
    min_year: int = 1963
    max_year: int = 2016
    __sort_index: int = 0

    def get_month_number(self, year: int, month: int = 0):
        return 12 * (year - self.min_year) + month

    def iterate_over_month(self, year: int):
        for m in range(1, 13):
            yield 12 * (year - self.min_year) + m

    def __iter__(self):
        self._sort_index = 0
        return self

    def __next__(self):
        result = self.min_year + self._sort_index
        if result > self.max_year:
            raise StopIteration
        self._sort_index += 1
        return result


@dataclass(frozen=True, init=True)
class Chars:
    ac: str = 'ac'
    beme: str = 'beme'
    idiovol: str = 'idiovol'
    r12_2: str = 'r12_2'
    op: str = 'op'
    investment: str = 'investment'
    st_rev: str = 'st_rev'
    lt_rev: str = 'lt_rev'
    lrunover: str = 'lturnover'

    # val: str = 'val'
    # qual: str = 'qual'
    # trd: str = 'trd'
    # sen: str = 'sen'
    # fcf: str = 'fcf_rank'
    
    # capital_structure: str = "cap_structure"
    # growth: str = "growth"
    # profitability: str = "profitability"
    # accrual: str = "accrual"
    # investment: str = "investment"
    # dividend_yield: str = "dy_rank"
    # book_yield: str = "by_rank"
    # forward_earnings_yield: str = "fy1_ey_rank"
    # ebidta_to_ev: str = "ee_no_fin_rank"
    # free_cash_flow: str = "fcf_rank"
    # stock_sentiment: str = "senstock"
    # industry_sentiment: str = "senind"
    # stock_trend: str = "trdstock"
    # industry_trend: str = "trdind"

    # book_yield: str = "by"
    # accrual: str = "accrual_level"
    # vol: str = "vol"
    # momemtum: str = "mom_1y1m"
    # op: str = "Oper_Income_Total_Capital"
    # turnover: str = "turnover_3m"
    # reversal: str = "mom_1m"
    # lt_rev: str = "mom_6m"
    # inven_turnover: str = "Inven_Turnover" 
    # illiquid: str = 'X3MILLIQ'
    lme: str = "lme"
    returns: str = 'ret'

    @property
    def base_char(self):
        return self.lme

    def combinations_with_size(self, k: int = 2) -> Iterator[tuple[str, str, str]]:
        """
        Provides generator of combinations with size characteristic (LME) with length k.

        Parameters
        ----------
        k: int, length of combination

        Returns
        -------
            Generator of combinations
        """
        for comb in combinations([k for k, v in self.__dict__.items() if v != self.returns], k):
            # yield self.lme, comb[0], comb[1]
            yield comb

    def combinations_of_chars(self, k: int = 2, exclude_chars: Union[List[str], str] = None,
                              include_chars: Union[List[str], str] = None) -> Iterator[tuple[str, str, str]]:
        """
        Provides generator of combinations of characteristics.

        Parameters
        ----------
        k:  int
            length of combination
        exclude_chars:
            List of characteristics not to include in combinations
        include_chars:
            List of characteristics to include in combinations

        Returns
        -------
            Generator of combinations
        """
        if exclude_chars is None:
            exclude_chars = list()
        if include_chars is None:
            include_chars = list(self.__dict__.values())
        if not isinstance(exclude_chars, list):
            exclude_chars = [exclude_chars]
        if not isinstance(include_chars, list):
            include_chars = [include_chars]
        # include_chars may name characteristics outside Chars (any file in characteristics/)
        for comb in combinations([v for v in include_chars if v not in exclude_chars], k):
            yield comb


def tree_combinations(k: int = Parameters.n_chars, chars: list = None) -> list:
    """Feature sequences of the tree sets: size first, then each combination of k other characteristics
    (of `chars` if given, else of every characteristic in Chars)."""
    c = Chars()
    return [(c.lme,) + comb for comb in c.combinations_of_chars(k=k, exclude_chars=[c.returns, c.lme], include_chars=chars)]


# Tree set-ups: tag in the tree and output file names -> characteristics the trees are built from (None = every
# characteristic in Chars) and tree depth. Tags have no '_', so a file name can be split back into its parts.
TREE_SETUPS = {
    None: {'chars': None, 'depth': Parameters.tree_depth},
    'slow3': {'chars': ['ac', 'beme', 'r12_2', 'op', 'investment', 'lt_rev'], 'depth': 3},   # slow-moving, shallower
    # best large-cap long/short with the least decay (analysis/characteristic_screen.py), SUV swapped for beme
    'screen3': {'chars': ['investment', 's2p', 'prof', 'beme', 'cf', 'ol', 'noa', 'd2a', 'lt_rev'], 'depth': 3},
}
# Pruning set-ups: tag in the output file names -> options of prune_trees.prune()
PRUNE_SETUPS = {
    None: {},                                                # sparsity picked on the last test_size months
    'val': {'validated': True, 'window': 240, 'folds': 3},   # rolling 20-year window, validated choice, refit on all
}


def run_variant(equal_weighted: bool = True, tree_tag: str = None, prune_tag: str = None) -> str:
    """File-name part of a run: None, or e.g. 'vw', 'slow3', 'slow3_val', 'vw_slow3_val'."""
    return '_'.join(p for p in ['vw' if not equal_weighted else None, tree_tag, prune_tag] if p) or None


def parse_variant(variant: str) -> tuple:
    """(equal_weighted, tree_tag, prune_tag) of a run's file-name part."""
    parts = variant.split('_') if variant else []
    tree = next((p for p in parts if p in TREE_SETUPS), None)
    prune = next((p for p in parts if p in PRUNE_SETUPS), None)
    return 'vw' not in parts, tree, prune


def tree_variant(variant: str) -> str:
    """File-name part of the trees a run reads (the pruning tag is not part of the tree file names)."""
    equal_weighted, tree, _ = parse_variant(variant)
    return run_variant(equal_weighted, tree)


@dataclass(frozen=True, init=False)
class DataPaths:
    input_data: Path = Path('characteristics')
    output: Path = Path('result')
    sep: str = '_'
    returns_file_name: str = 'ret'
    rf_factor_file_name: str = 'rf_factor'
    model_dumps: Path = Path('model_dumps')
    processed_data: Path = Path('processed_data')
    model_suffix: str = 'model.pkl'
    def _join(self, *parts) -> str:
        return self.sep.join(p for p in parts if p)

    def label(self, region: str = None, universe: str = None, variant: str = None) -> str:
        """Name of a run for messages and titles, e.g. 'GL', 'largecap' or 'largecap_vw'."""
        return self._join(region, universe, variant)

    def tree_file(self, comb: str, region: str = None, universe: str = None, variant: str = None) -> Path:
        """
        Tree portfolio file: processed_data/[<region>_]<comb>[_<universe>][_<variant>].parquet, e.g.
        GL_lme_ac_beme.parquet (company data), lme_ac_beme_largecap.parquet (characteristic files),
        lme_ac_beme_largecap_vw.parquet (value-weighted trees).
        """
        return self.processed_data / f"{self._join(region, comb, universe, variant)}.parquet"

    def tree_files(self, region: str = None, universe: str = None, variant: str = None) -> list:
        """The existing tree files of a region/universe/variant, one per feature combination of the variant's tree
        set-up (tree_combinations)."""
        chars = TREE_SETUPS[parse_variant(variant)[1]]['chars']
        files = [self.tree_file(self.sep.join(seq), region, universe, variant) for seq in tree_combinations(chars=chars)]
        return [f for f in files if f.exists()]

    def result_file(self, kind: str, region: str = None, universe: str = None, variant: str = None,
                    ext: str = 'csv') -> Path:
        """Backtest output, named like the tree files: result/[<region>_]<kind>[_<universe>][_<variant>].<ext>,
        e.g. result/GL_ret.csv, result/ret_largecap.csv, result/ret_largecap_vw.csv; ext=None for a folder."""
        name = self._join(region, kind, universe, variant)
        return self.output / (f"{name}.{ext}" if ext else name)

    def merge_tuple(self, input_tuple: tuple[str, str]) -> str:
        return self.sep.join(input_tuple)

