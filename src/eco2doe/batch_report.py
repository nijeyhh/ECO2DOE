"""ECO2 일괄 계산 결과(batchreport.tab) 해석."""

import dataclasses as dc
import functools
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

if TYPE_CHECKING:
    from pathlib import Path

METADATA = 'data/BatchReportMeta.csv'


def read_metadata(
    path: str | Path = METADATA,
    category: tuple[str, str, str] = ('항목(대)', '항목(중)', '항목(소)'),
):
    v = (
        pl
        .read_csv(path)
        .with_columns(pl.col(category).fill_null(''))
        .with_columns(
            variable=pl
            .concat_str(category, separator='/')
            .str.strip_chars_end('/')
            .replace({'': None}),
            unit=pl.col('단위').replace({'': None, '-': None}),
        )
        .drop_nulls('variable')
    )
    assert v.select(pl.col('variable').is_unique().all()).item()

    return v


@dc.dataclass(frozen=True)
class BatchReport:
    source: str | Path
    _metadata: str | Path | pl.DataFrame = METADATA

    @functools.cached_property
    def metadata(self):
        return (
            self._metadata
            if isinstance(self._metadata, pl.DataFrame)
            else read_metadata(self._metadata)
        )

    @functools.cached_property
    def wide_data(self):
        data = pl.read_csv(self.source, separator='\t', has_header=False)
        cols = ['case_name', *self.metadata['variable'].to_list()]

        if len(cols) < data.width:
            cols = [*cols, *(f'unknown{x + 1}' for x in range(data.width - len(cols)))]

        data.columns = cols
        data = (
            (data)
            .with_row_index('case')
            .with_columns(
                (cs.contains('면적') & cs.string())
                .str.replace_all(',', '')
                .cast(pl.Float64)
            )
        )
        nulls = (
            data
            .count()
            .unpivot()
            .filter(pl.col('value') == 0)
            .select('variable')
            .to_series()
            .to_list()
        )
        if nulls:
            data = data.with_columns(pl.col(nulls).cast(pl.Float64))
        return data

    @functools.cached_property
    def long_data(self):
        cols = [x for x in self.metadata.columns if x not in {'variable', 'unit'}]
        cols = ['case', 'case_name', *cols, 'variable', 'value', 'unit']

        return (
            self.wide_data
            .unpivot(index=['case', 'case_name'])
            .drop_nulls('value')
            .join(self.metadata, on='variable', how='left', validate='m:1')
            .select(*cols)
        )
