import polars as pl
from polars.testing import assert_frame_equal

from wtg_power_prediction.dataset import remove_target_turbine_signals


def test_remove_target_turbine_signals() -> None:
    data = {
        "wtc_ActPower_mean;1": [1, 2, 3],
        "wtc_ActPower_mean;2": [4, 5, 6],
        "some_other_signal;1": [7, 8, 9],
        "some_other_signal;2": [10, 11, 12],
        "target": [1, 2, 3],
    }
    df = pl.DataFrame(data)

    expected1 = pl.DataFrame(
        {
            "wtc_ActPower_mean;2": [4, 5, 6],
            "some_other_signal;2": [10, 11, 12],
            "target": [1, 2, 3],
        }
    )

    expected2 = pl.DataFrame(
        {
            "wtc_ActPower_mean;1": [1, 2, 3],
            "some_other_signal;1": [7, 8, 9],
            "target": [4, 5, 6],
        }
    )

    actual1 = remove_target_turbine_signals(df, target_turbine=1)
    actual2 = remove_target_turbine_signals(df, target_turbine=2)

    assert_frame_equal(actual1, expected1)
    assert_frame_equal(actual2, expected2)
