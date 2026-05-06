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

    boiler_space: str = '온열설비 효율_보일러'
    ehp_heating: str = '온열설비 효율_히트펌프 전기'
    ghp_heating: str = '온열설비 효율_히트펌프 가스'

    ehp_cooling: str = '냉열설비 효율_압축식'
    ghp_cooling: str = '냉열설비 효율_압축식(LNG)'
    absorption: str = '냉열설비 효율_흡수식(없음)'

    boiler_water: str = '급탕설비 효율_보일러'
    light_density: str = '평균 조명밀도'

    recovery_heating: str = '평균열회수율_전열교환기'
    recovery_cooling: str = '평균열회수율냉_전열교환기'

    ahu_recovery_heating: str = '평균열회수율_AHU'
    ahu_recovery_cooling: str = '평균열회수율냉_AHU'
    ahu_supply_fan: str = '총효율급기팬_AHU'
    ahu_exhaust_fan: str = '총효율배기팬_AHU'

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

    def name(self):
        return f'{self.application_number}_{self.variable}={self.scale_factor}'


@dc.dataclass(frozen=True)
class Design:
    var: Variables
    models: pl.DataFrame

    scale_factors: tuple[float, ...] = SCALE_FACTORS
    cop_reduction: float = 0.45

    @classmethod
    def create(cls, cases: str | Path, conf: str | Path = 'design.toml'):
        c = tomllib.loads(Path(conf).read_text('UTF-8'))
        var = Variables(**c['variable'])
        numerics = [x[1] for x in var.numerics()]
        models = (
            (pl)
            .read_excel(
                cases,
                read_options={'header_row': 1},
                columns=dc.astuple(var),
            )
            .with_columns(pl.col(numerics).cast(pl.Float64, strict=False).fill_null(0))
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

    @functools.cached_property
    def app_numbers(self):
        return self.models[self.var.application_number].to_list()

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
                    reference=round(reference, 6),
                    adjusted=round(adjusted, 6),
                )

    def dataframe(self):
        return (
            pl
            .from_dicts([dc.asdict(x) for x in self.iter()])
            .with_row_index()
            .with_columns(
                limit=pl.col('variable').replace_strict(
                    {
                        'boiler_space': 100,
                        'boiler_water': 100,
                        'recovery_heating': 1,
                        'recovery_cooling': 1,
                    },
                    default=None,
                    return_dtype=pl.Float64,
                )
            )
            .with_columns(exceed=pl.col('adjusted') > pl.col('limit'))
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
