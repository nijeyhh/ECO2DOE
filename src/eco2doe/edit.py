import dataclasses as dc
import functools
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import more_itertools as mi
import polars as pl
import structlog
from eco2 import editor
from rich.progress import track

from eco2doe import batch_report
from eco2doe.design import Design

if TYPE_CHECKING:
    from eco2doe.design import Case

BoilerUsage = Literal['space', 'water']

app = cyclopts.App(
    config=cyclopts.config.Toml(
        'conf.toml',
        allow_unknown=True,
        use_commands_as_keys=False,
    ),
    result_action=['call_if_callable', 'print_non_int_sys_exit'],
)
logger = structlog.stdlib.get_logger()


@dc.dataclass(frozen=True)
class Editor(editor.Eco2Editor):
    case: Case

    cop_reduction: float = 0.45
    SURFACE: ClassVar[dict[str, editor.SurfaceType]] = {
        'u_wall': '외벽(벽체)',
        'u_roof': '외벽(지붕)',
        'u_floor': '외벽(바닥)',
    }

    def _iter(self, path: str):
        return tuple(self.xml.ds.iterfind(path))

    def _count_heating_zone(self, code: str, usage: BoilerUsage):
        tag = {'space': '열생산난방생산기기', 'water': '열생산급탕생산기기'}[usage]
        return sum(1 for x in self._iter('tbl_zone') if x.findtext(tag) == code)

    def _set_boiler_efficiency(self):
        usage: BoilerUsage
        match self.case.variable:
            case 'boiler_space':
                usage = 'space'
            case 'boiler_water':
                usage = 'water'
            case _:
                raise ValueError(self.case)

        value = f'{self.case.adjusted:.3f}'

        for e in self._iter('tbl_nanbangkiki'):
            if not self._count_heating_zone(e.findtext('code'), usage):
                continue

            editor.set_child_text(e, '정격보일러효율', value)

    def _set_heating_hp_cop(self):
        elec = {
            'ehp_heating': True,
            'ghp_heating': False,
        }[self.case.variable]

        v7 = f'{self.case.adjusted:.3f}'
        v10 = f'{self.case.adjusted * self.cop_reduction:.3f}'

        for e in self._iter('tbl_nanbangkiki'):
            if e.findtext('열생산기기방식') != '히트펌프':
                continue
            if elec is not (e.findtext('히트연료') == '전기'):
                continue

            editor.set_child_text(e, '히트난방정격7', v7)
            editor.set_child_text(e, '히트난방정격10', v10)

    def _set_cooling_hp_cop(self):
        type_ = {
            'ehp_cooling': '압축식',
            'ghp_cooling': '압축식(LNG)',
        }[self.case.variable]

        value = f'{self.case.adjusted:.3f}'

        for e in self._iter('tbl_nangbangkiki'):
            if e.findtext('냉동기방식') != type_:
                continue

            editor.set_child_text(e, '열성능비', value)

    def _set_cooling_apsorption_cop(self):
        if self.case.variable != 'absorption':
            raise ValueError(self.case)

        value = f'{self.case.adjusted:.3f}'

        for e in self._iter('tbl_nangbangkiki'):
            if e.findtext('냉동기방식') != '흡수식':
                continue

            editor.set_child_text(e, '열성능비', value)

    def _set_heat_exchanger_efficiency(self):
        tag = {
            'recovery_heating': '열회수율',
            'recovery_colling': '열회수율냉',
        }[self.case.variable]

        value = f'{self.case.adjusted:.2f}'

        for e in self._iter('tbl_kongjo'):
            if e.findtext('열교환기유형') != '전열교환':
                continue

            editor.set_child_text(e, tag, value)

    def _edit(self):
        if t := self.SURFACE.get(self.case.variable):
            self.xml.set_walls(uvalue=self.case.adjusted, surface_type=t)
            return

        match self.case.variable:
            case 'u_window':
                self.xml.set_windows(uvalue=self.case.adjusted)
            case 'shgc':
                self.xml.set_windows(shgc=self.case.adjusted)
            case 'boiler_space' | 'boiler_water':
                self._set_boiler_efficiency()
            case 'ehp_heating' | 'ghp_heating':
                self._set_heating_hp_cop()
            case 'ehp_cooling' | 'ghp_cooling':
                self._set_cooling_hp_cop()
            case 'absorption':
                self._set_cooling_apsorption_cop()
            case 'light_density':
                self.xml.set_elements(
                    'tbl_zone/조명에너지부하율입력치', f'{self.case.adjusted:.3f}'
                )
            case 'recovery_heating' | 'recovery_colling':
                self._set_heat_exchanger_efficiency()
            case _:
                raise ValueError(self.case)

    def __call__(self, dst: str | Path):
        self._edit()
        self.write(dst, dsr=False)


@functools.lru_cache
def _find_model(path: Path, app_number: int):
    return mi.one(path.glob(f'{app_number}*.tpl*'))


@app.command
@dc.dataclass(frozen=True)
class Edit:
    src: Path
    dst: Path
    cases: Path

    batch: bool = True
    """건물(Application Number)별로 폴더 구분"""

    skip_zero: bool = True

    @functools.cached_property
    def design(self):
        return Design.create(cases=self.cases)

    @functools.cached_property
    def _eco2(self):
        d = self.dst / 'ECO2'
        d.mkdir(exist_ok=True)
        return d

    def mkdir_batch(self):
        for app_number in self.design.app_numbers:
            self._eco2.joinpath(str(app_number)).mkdir(exist_ok=True)

    def _dst(self, case: Case, idx: int, width: int):
        d = self._eco2 / f'{case.application_number}' if self.batch else self._eco2
        return d / f'{idx:0{width}d}.{case.name()}.tplx'

    def __call__(self):
        self.dst.mkdir(exist_ok=True)
        if self.batch:
            self.mkdir_batch()

        cases = self.design.dataframe()
        cases.write_excel(self.dst / 'cases.xlsx', column_widths=150)
        cases.write_parquet(self.dst / 'cases.parquet')

        width = max(4, len(str(self.design.count)))
        for idx, case in enumerate(track(self.design.iter(), total=self.design.count)):
            if self.skip_zero and case.reference == 0:
                continue

            logger.info('%04d. %s', idx, case)

            src = _find_model(self.src, case.application_number)
            editor = Editor(
                src=src,
                case=case,
                cop_reduction=self.design.cop_reduction,
            )
            editor(self._dst(case, idx, width))


@app.command
@dc.dataclass(frozen=True)
class Report:
    dst: Path

    @functools.cached_property
    def _cases(self):
        return (
            pl
            .scan_parquet(self.dst / 'cases.parquet')
            .rename({
                'application_number': 'app_number',
                'variable': 'design_variable',
                'label': 'design_label',
            })
            .with_columns(pl.col('app_number').cast(pl.UInt32))
            .collect()
        )

    @functools.cached_property
    def _meta(self):
        return batch_report.read_metadata()

    def read_reports(self):
        reports = [
            batch_report.BatchReport(x, _metadata=self._meta)
            for x in self.dst.glob('**/batchreport.tab')
        ]
        long = pl.concat([x.long_data for x in reports])
        wide = pl.concat([x.wide_data for x in reports], how='vertical_relaxed')

        return long, wide

    @staticmethod
    def _prep_report_columns(data: pl.DataFrame):
        data = data.rename({'index': 'report_index'}, strict=False)
        columns = data.columns
        return (
            data
            .with_columns(
                m=pl.col('case_name').str.extract_groups(
                    r'^(?<index>\d+)\.(?<app_number>\d+)'
                    r'_(?<design_variable>[\w_]+)=(?<scale_factor>[\d\.]+)\.tplx$'
                )
            )
            .unnest('m')
            .select(
                pl.col('index').cast(pl.UInt32),
                pl.col('app_number').cast(pl.UInt32),
                'design_variable',
                pl.col('scale_factor').cast(pl.Float64),
                *columns,
            )
            .drop('case')
        )

    def _join(self, data: pl.DataFrame):
        cols = self._cases.columns
        return (
            self
            ._prep_report_columns(data)
            .join(
                self._cases,
                on=['index', 'app_number', 'design_variable', 'scale_factor'],
                how='left',
                validate='m:1',
            )
            .select(*cols, pl.all().exclude(cols))
        )

    def __call__(self):
        long, wide = self.read_reports()

        wide = self._join(wide)
        wide.write_parquet(self.dst / 'report-wide.parquet')
        wide.write_excel(self.dst / 'report-wide.xlsx', column_widths=100)

        long = self._join(long)
        long.write_parquet(self.dst / 'report-long.parquet')
        long.write_excel(self.dst / 'report-long.xlsx', column_widths=100)


if __name__ == '__main__':
    app()
