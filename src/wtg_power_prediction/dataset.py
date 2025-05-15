from pathlib import Path

import kagglehub
import polars as pl


def load_training_dataset(*, force_download: bool = False) -> pl.LazyFrame:
    file_path = kagglehub.competition_download(
        handle="hill-of-towie-wind-turbine-power-prediction",
        path="training_dataset.parquet",
        force_download=force_download,
    )
    return pl.scan_parquet(Path(file_path))


def remove_target_turbine_signals(df: pl.DataFrame, *, target_turbine: int = 1) -> pl.DataFrame:
    return df.with_columns(pl.col(f"wtc_ActPower_mean;{target_turbine}").alias("target")).select(
        [col for col in df.columns if not col.endswith(f";{target_turbine}")]
    )
