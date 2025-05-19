from pathlib import Path

import kagglehub
import polars as pl
import requests
from kagglehub.config import DEFAULT_CACHE_FOLDER

CACHE_DIR = Path(DEFAULT_CACHE_FOLDER) / "competitions" / "hill-of-towie-wind-turbine-power-prediction"


def load_training_dataset(*, force_download: bool = False) -> pl.LazyFrame:
    file_path = kagglehub.competition_download(
        handle="hill-of-towie-wind-turbine-power-prediction",
        path="training_dataset.parquet",
        force_download=force_download,
    )
    return pl.scan_parquet(Path(file_path))


def load_submission_dataset(*, force_download: bool = False) -> pl.LazyFrame:
    file_path = kagglehub.competition_download(
        handle="hill-of-towie-wind-turbine-power-prediction",
        path="submission_dataset.parquet",
        force_download=force_download,
    )
    return pl.scan_parquet(Path(file_path))


def load_turbine_metadata(*, force_download: bool = False) -> pl.LazyFrame:
    file_path = CACHE_DIR / "turbine_metadata.csv"
    if not file_path.exists() or force_download:
        response = requests.get(
            "https://zenodo.org/records/14870023/files/Hill_of_Towie_turbine_metadata.csv?download=1",
            headers={"Accept": "text/csv"},
            timeout=10,
        )
        response.raise_for_status()
        file_path.write_text(response.content.decode("utf-8-sig"), encoding="utf-8")
    return pl.scan_csv(file_path)


def remove_target_turbine_signals(df: pl.DataFrame, *, target_turbine: int = 1) -> pl.DataFrame:
    return df.with_columns(pl.col(f"wtc_ActPower_mean;{target_turbine}").alias("target")).select(
        [col for col in df.columns if not col.endswith(f";{target_turbine}")]
    )
