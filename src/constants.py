from typing import List, Union, ClassVar
from pathlib import Path
from dataclasses import dataclass
from itertools import combinations
from collections.abc import Iterator

import numpy as np

@dataclass(frozen=True)
class Parameters:
    tree_depth: int = 4
    n_splits: int = 3
    n_chars: int = 2
    mean_shrinkage: ClassVar[np.ndarray] = np.arange(0.5, 0.9, 0.1)
    ridge_lambda: ClassVar[np.ndarray] = 0.1 ** np.arange(6, 8, 0.5)
    cv_splits: int = 3
    k_min: int = 5
    k_max: int = 50
    test_size: int = 120


@dataclass(frozen=True)
class Columns:
    date_col: str = 'date'
    size_col: str = 'mkt_cap'
    permno_col: str = 'permno'
    returns_col: str = 'gross_returns'
    w_returns_col: str = 'weighted_ret'
    col_sep: str = '_'
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
    # ac: str = 'ac'
    # beme: str = 'beme'
    # idiovol: str = 'idiovol'
    # lme: str = 'lme'
    # r12_2: str = 'r12_2'
    # op: str = 'op'
    # investment: str = 'investment'
    # st_rev: str = 'st_rev'
    # lt_rev: str = 'lt_rev'
    # lrunover: str = 'lturnover'

    val: str = 'val'
    qual: str = 'qual'
    trd: str = 'trd'
    sen: str = 'sen'
    fcf: str = 'fcf_rank'
    lme: str = 'lme'
    
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
    # lme: str = "lme"
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
        for comb in combinations([v for v in self.__dict__.values() if v not in exclude_chars and v in include_chars], k):
            yield comb


@dataclass(frozen=True, init=False)
class DataPaths:
    input_data: Path = Path('characteristics')
    sep: str = '_'
    returns_file_name: str = 'ret'
    rf_factor_file_name: str = 'rf_factor'
    model_dumps: Path = Path('model_dumps')
    processed_data: Path = Path('processed_data')
    model_suffix: str = 'model.pkl'

    def merge_tuple(self, input_tuple: tuple[str, str]) -> str:
        return self.sep.join(input_tuple)

