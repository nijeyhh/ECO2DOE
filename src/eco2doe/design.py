import dataclasses as dc
import functools
import itertools
import tomllib
from pathlib import Path

import polars as pl

SCALE_FACTORS: tuple[float, ...] = (1.3, 1.2, 1.1, 0.9, 0.8, 0.7)


@dc.dataclass(frozen=True)
class Variables:
    application_number: str = '신청번호'

    u_wall: str = '외벽 평균 열관류율'
    u_roof: str = '지붕 평균 열관류율'
    u_floor: str = '바닥 평균 열관류율_직접'
    u_window: str = '창호 평균 열관류율'
    shgc: str = 'SHGC 평균'

    boiler_efficiency: str = '온열설비 효율_보일러'
    ehp_cop_heating: str = '온열설비 효율_히트펌프 전기'
    ghp_cop_heating: str = '온열설비 효율_히트펌프 가스'
    ehp_cop_cooling: str = '냉열설비 효율_압축식'
    ghp_cop_cooling: str = '냉열설비 효율_압축식(LNG)'
    absorption_efficiency: str = '냉열설비 효율_흡수식(없음)'
    boiler_water_efficiency: str = '급탕설비 효율_보일러'
    light_density: str = '평균 조명밀도'
    heat_recovery_heating: str = '평균열회수율_전열교환기'
    heat_recovery_colling: str = '평균열회수율냉_전열교환기'

    @classmethod
    def count(cls):
        return len(dc.fields(cls)) - 1

    def numerics(self):
        return tuple(
            (k, v) for k, v in dc.asdict(self).items() if v != self.application_number
        )


@dc.dataclass(frozen=True)
class Case:
    application_number: int
    variable: str
    label: str

    scale_factor: float
    reference: float
    adjusted: float  # scale_factor * reference


@dc.dataclass(frozen=True)
class Design:
    var: Variables
    models: pl.DataFrame

    scale_factors: tuple[float, ...] = SCALE_FACTORS

    @classmethod
    def create(cls, cases: str | Path, conf: str | Path = 'design.toml'):
        c = tomllib.loads(Path(conf).read_text('UTF-8'))
        var = Variables(**c['variable'])
        models = pl.read_excel(
            cases,
            read_options={'header_row': 1},
            columns=dc.astuple(var),
        )
        if not models[var.application_number].is_unique().all():
            msg = '신청번호 중복'
            raise ValueError(msg)

        return cls(
            var=var,
            models=models,
            scale_factors=c.get('scale_factors', SCALE_FACTORS),
        )

    @functools.cached_property
    def count(self):
        return self.var.count() * self.models.height * len(self.scale_factors)

    def iter(self):
        indices = self.models[self.var.application_number].to_list()
        app_number = pl.col(self.var.application_number)

        for idx in indices:
            row = self.models.row(by_predicate=app_number == idx, named=True)

            for (var, label), factor in itertools.product(
                self.var.numerics(), self.scale_factors
            ):
                reference = row[label]
                adjusted = reference * factor
                yield Case(
                    application_number=idx,
                    variable=var,
                    label=label,
                    scale_factor=factor,
                    reference=reference,
                    adjusted=adjusted,
                )


if __name__ == '__main__':
    import cyclopts
    import rich

    console = rich.get_console()
    app = cyclopts.App(
        config=cyclopts.config.Toml('conf.toml', allow_unknown=True),
        console=console,
    )

    @app.default
    def main(cases: Path):
        design = Design.create(cases=cases)
        console.print(design)

        for case, _ in zip(design.iter(), range(10), strict=False):
            console.print(case)

    app()
