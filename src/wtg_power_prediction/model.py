import datetime as dt
import math
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
from flaml.automl import AutoML

from wtg_power_prediction.sun import SunPosition


class AutoMLResult:
    def __init__(
        self,
        automl: AutoML,
        feature_importance: pl.DataFrame,
        fig_corr: plt.Figure,
        fig_fi: plt.Figure,
        prediction: pl.Series,
    ) -> None:
        self.automl = automl
        self.feature_importance = feature_importance
        self.fig_corr = fig_corr
        self.fig_fi = fig_fi
        self.prediction = prediction


class WtgPowerPredictionModel:
    def __init__(
        self,
        *,
        latitude: float,
        longitude: float,
        validation_start: dt.datetime,
        time_budget_engineering_s: int = 180,
        time_budget_power_s: int = 180,
    ) -> None:
        self._sun_position = SunPosition(latitude=latitude, longitude=longitude)
        self._test_wtg = 1
        self._ref_wtgs = [2, 3, 4, 5, 7]
        self._wtgs = [self._test_wtg, *self._ref_wtgs]
        self.validation_start = validation_start
        self.models: dict[str, AutoMLResult] = {}
        self.time_budget_engineering_s = time_budget_engineering_s
        self.time_budget_power_s = time_budget_power_s

    def fit(self, X: pl.DataFrame, y: pl.Series) -> "WtgPowerPredictionModel":  # noqa: N803
        df = self.preprocess(X, wtgs=self._wtgs)
        y = y.filter(df.select("is_valid").to_series())
        df = df.filter(pl.col("is_valid"))
        automl_result_ws = self.fit_test_turbine_feature(
            df,
            target_col=f"wtc_AcWindSp_mean;{self._test_wtg}",
            unit="m/s",
            extra_cols=[],
            time_budget_s=self.time_budget_engineering_s,
        )
        self.models["wind_speed"] = automl_result_ws
        df = df.with_columns(engineered_wind_speed=automl_result_ws.prediction, target=y)
        automl_result_power = self.fit_test_turbine_feature(
            df,
            target_col="target",
            unit="kW",
            extra_cols=["engineered_wind_speed"],
            time_budget_s=self.time_budget_power_s,
        )
        self.models["power"] = automl_result_power
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:  # noqa: N803
        df = self.preprocess(X, wtgs=self._ref_wtgs)
        df = self.select_features(df, target_col=None)
        engineered_ws = self.models["wind_speed"].automl.predict(df.to_pandas())
        df = df.with_columns(engineered_wind_speed=pl.Series(values=engineered_ws))
        prediction = self.models["power"].automl.predict(df.to_pandas())
        return pl.Series(values=prediction)

    def preprocess(self, df: pl.DataFrame, *, wtgs: list[int]) -> pl.DataFrame:
        return (
            df.lazy()
            .with_columns(
                pl.col("TimeStamp_StartFormat")
                .sub(dt.datetime(2016, 1, 1, tzinfo=dt.UTC))
                .dt.total_seconds()
                .alias("seconds_since_2016"),
                *[
                    pl.col(f"wtc_ScYawPos_mean;{wtg}").radians().sin().alias(f"wtc_ScYawPos_mean_sin;{wtg}")
                    for wtg in wtgs
                ],
                *[
                    pl.col(f"wtc_ScYawPos_mean;{wtg}").radians().cos().alias(f"wtc_ScYawPos_mean_cos;{wtg}")
                    for wtg in wtgs
                ],
                pl.col("TimeStamp_StartFormat").dt.minute().mul(2 * math.pi / 60).sin().alias("minutes_sin"),
                pl.col("TimeStamp_StartFormat").dt.minute().mul(2 * math.pi / 60).cos().alias("minutes_cos"),
                pl.col("TimeStamp_StartFormat").dt.hour().mul(2 * math.pi / 24).sin().alias("hours_sin"),
                pl.col("TimeStamp_StartFormat").dt.hour().mul(2 * math.pi / 24).cos().alias("hours_cos"),
                pl.col("TimeStamp_StartFormat").dt.ordinal_day().mul(2 * math.pi / 365).sin().alias("days_sin"),
                pl.col("TimeStamp_StartFormat").dt.ordinal_day().mul(2 * math.pi / 365).cos().alias("days_cos"),
                pl.col("TimeStamp_StartFormat").dt.month().mul(2 * math.pi / 12).sin().alias("months_sin"),
                pl.col("TimeStamp_StartFormat").dt.month().mul(2 * math.pi / 12).cos().alias("months_cos"),
                pl.concat_list([pl.col(f"wtc_AmbieTmp_mean;{wtg}") for wtg in self._ref_wtgs])
                .list.mean()
                .alias("ambient_temp_mean"),
            )
            .collect()
            .with_columns(
                pl.col("TimeStamp_StartFormat")
                .map_elements(lambda ts: self._sun_position.altitude(timestamp_utc=ts), return_dtype=pl.Float64)
                .mul(180 / math.pi)
                .alias("sun_altitude"),
            )
        )

    def select_features(
        self, df: pl.DataFrame, *, target_col: str | None = None, extra_cols: list[str] | None = None
    ) -> pl.DataFrame:
        cols = [
            pl.col("TimeStamp_StartFormat"),
            *[pl.col(f"wtc_AcWindSp_mean;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_AcWindSp_stddev;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_AcWindSp_min;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_AcWindSp_max;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_ScYawPos_mean_sin;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_ScYawPos_mean_cos;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_ScYawPos_stddev;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_ScReToOp_timeon;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_ActPower_mean;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_ActPower_stddev;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_ActPower_min;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_ActPower_max;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_GenRpm_mean;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            *[pl.col(f"wtc_PitcPosA_mean;{ref_wtg}") for ref_wtg in self._ref_wtgs],
            pl.col("ambient_temp_mean"),
            pl.col("sun_altitude"),
            pl.col("seconds_since_2016"),
            pl.col("hours_sin"),
            pl.col("hours_cos"),
            pl.col("days_sin"),
            pl.col("days_cos"),
            pl.col("months_sin"),
            pl.col("months_cos"),
        ]

        if extra_cols is not None:
            cols.extend([pl.col(col) for col in extra_cols])

        if target_col is not None:
            cols.append(pl.col(target_col))

        return df.select(*cols)

    def get_train_and_validation_sets(
        self, df: pl.DataFrame, *, target_col: str
    ) -> tuple[pl.DataFrame, pl.Series, pl.DataFrame, pl.Series]:
        df_x_train = (
            df.filter(pl.col("TimeStamp_StartFormat").lt(self.validation_start))
            .select(pl.exclude(target_col))
            .drop("TimeStamp_StartFormat")
        )
        df_y_train = df.filter(pl.col("TimeStamp_StartFormat").lt(self.validation_start)).select(target_col).to_series()
        df_x_test = (
            df.filter(pl.col("TimeStamp_StartFormat").ge(self.validation_start))
            .select(pl.exclude(target_col))
            .drop("TimeStamp_StartFormat")
        )
        df_y_test = df.filter(pl.col("TimeStamp_StartFormat").ge(self.validation_start)).select(target_col).to_series()
        return df_x_train, df_y_train, df_x_test, df_y_test

    def automl_fit(self, *, df_x_train: pl.DataFrame, df_y_train: pl.Series, name: str, time_budget_s: int) -> AutoML:
        automl = AutoML()
        automl_settings = {
            "time_budget": time_budget_s,
            "task": "regression",
            "metric": "mae",
            "estimator_list": [
                "xgboost",
            ],
            "log_file_name": f"{name}.log",
            "seed": 42,
            "eval_method": "cv",
            "n_splits": 5,
            "split_type": "time",
            "early_stop": True,
        }
        automl.fit(
            X_train=df_x_train.to_pandas(),
            y_train=df_y_train.to_pandas(),
            **automl_settings,
        )
        return automl

    def evaluate_automl_fit(
        self,
        automl: AutoML,
        *,
        df_x_train: pl.DataFrame,
        df_y_train: pl.Series,
        df_x_test: pl.DataFrame,
        df_y_test: pl.Series,
        variable: str,
        unit: str,
    ) -> plt.Figure:
        train_prediction = pl.Series(values=automl.predict(df_x_train.to_pandas()))
        test_prediction = pl.Series(values=automl.predict(df_x_test.to_pandas()))
        mae_train = abs(df_y_train - train_prediction).mean()
        mae_test = abs(df_y_test - test_prediction).mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.scatter(
            x=df_y_train.to_numpy().flatten(),
            y=train_prediction.to_numpy().flatten(),
            alpha=0.1,
        )
        ax1.text(0.05, 0.95, f"MAE: {mae_train:.2f} {unit}", ha="left", va="top", transform=ax1.transAxes)  # type: ignore[str-bytes-safe]
        ax1.set_xlabel(f"True {variable} ({unit})")
        ax1.set_ylabel(f"Predicted {variable} ({unit})")
        ax1.set_title("Train set")
        ax1.grid(visible=True)

        ax2.scatter(
            x=df_y_test.to_numpy().flatten(),
            y=test_prediction.to_numpy().flatten(),
            alpha=0.1,
        )
        ax2.text(0.05, 0.95, f"MAE: {mae_test:.2f} {unit}", ha="left", va="top", transform=ax2.transAxes)  # type: ignore[str-bytes-safe]
        ax2.set_xlabel(f"True {variable} ({unit})")
        ax2.set_ylabel(f"Predicted {variable} ({unit})")
        ax2.set_title("Validation set")
        ax2.grid(visible=True)

        plt.close(fig)

        return fig

    def get_feature_importance(self, automl: AutoML) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Name": automl.feature_names_in_,
                "Importance": automl.feature_importances_,
            },
        ).sort("Importance")

    def plot_feature_importance(self, feature_importance: pl.DataFrame, *, top_n: int | None = None) -> plt.Figure:
        if top_n is not None:
            feature_importance = feature_importance.head(top_n)
        fig, ax = plt.subplots(figsize=(10, 0.17 * len(feature_importance)))
        bars = ax.barh(
            y=feature_importance.select("Name").to_numpy().flatten(),
            width=feature_importance.select("Importance").to_numpy().flatten(),
            alpha=0.7,
        )
        ax.bar_label(bars)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        plt.close(fig)

        return fig

    def get_best_config(self, automl: AutoML) -> dict[str, Any]:
        return automl.best_config

    def fit_test_turbine_feature(
        self, df: pl.DataFrame, *, target_col: str, unit: str, extra_cols: list[str], time_budget_s: int
    ) -> AutoMLResult:
        df = self.select_features(df, target_col=target_col, extra_cols=extra_cols)
        df_x_train, df_y_train, df_x_test, df_y_test = self.get_train_and_validation_sets(df, target_col=target_col)
        automl = self.automl_fit(
            df_x_train=df_x_train, df_y_train=df_y_train, name=target_col, time_budget_s=time_budget_s
        )
        fig_corr = self.evaluate_automl_fit(
            automl=automl,
            df_x_train=df_x_train,
            df_y_train=df_y_train,
            df_x_test=df_x_test,
            df_y_test=df_y_test,
            variable=target_col,
            unit=unit,
        )
        feature_importance = self.get_feature_importance(automl)
        fig_fi = self.plot_feature_importance(feature_importance)
        prediction = automl.predict(df.select(pl.exclude(target_col)).drop("TimeStamp_StartFormat").to_pandas())
        return AutoMLResult(
            automl=automl,
            feature_importance=feature_importance,
            fig_corr=fig_corr,
            fig_fi=fig_fi,
            prediction=pl.Series(values=prediction),
        )
