import os

import pandas as pd
import pytest

from src.cycling_dynamics.load_data import load_fit_file

# Assuming the FIT files are in a 'test_data' directory
FIT_FILES = [
    "alex_24HOP_2024-02-17-185919-ELEMNT ROAM 3FF5-53-0.fit",
    "alex_24HOP_2024-02-17-185919-ELEMNT_ROAM_3FF5-53-0.fit",
    "alex_24HOP_2024-02-18-002255-ELEMNT ROAM 3FF5-54-0.fit",
    "alex_24HOP_2024-02-18-002255-ELEMNT_ROAM_3FF5-54-0.fit",
    "vincent_lap_1_24HOP_14012433014_ACTIVITY.fit",
    "vincent_lap_2_24HOP_14010180820_ACTIVITY.fit",
    "vincent_mtb_id_surface_14517623900_ACTIVITY.fit",
]
TEST_DATA_DIR = "test_data"


@pytest.fixture(params=FIT_FILES)
def fit_file_path(request):
    return os.path.join(TEST_DATA_DIR, f"{request.param}")


def test_load_fit_file_success(fit_file_path):
    df = load_fit_file(fit_file_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Check for expected columns
    expected_columns = ["timestamp", "position_lat", "position_long", "altitude", "speed"]
    for col in expected_columns:
        assert col in df.columns, f"Expected column {col} not found in DataFrame"

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert pd.api.types.is_float_dtype(df["position_lat"])
    assert pd.api.types.is_float_dtype(df["position_long"])
    assert pd.api.types.is_float_dtype(df["altitude"])
    assert pd.api.types.is_float_dtype(df["speed"])

    # Check value ranges
    assert df["position_lat"].min() >= -90
    assert df["position_lat"].max() <= 90
    assert df["position_long"].min() >= -180
    assert df["position_long"].max() <= 180
    assert df["speed"].min() >= 0


# def test_load_fit_file_add_metrics(fit_file_path):
#     df_with_metrics = load_fit_file(fit_file_path, add_metrics=True)
#     df_without_metrics = load_fit_file(fit_file_path, add_metrics=False)
#
#     assert len(df_with_metrics.columns) > len(df_without_metrics.columns)
#
#     # Check for additional metrics (adjust based on what metrics you add)
#     additional_metrics = ['speed_rolling_avg']  # Add more as you implement them
#     for metric in additional_metrics:
#         assert metric in df_with_metrics.columns
#         assert metric not in df_without_metrics.columns


# @pytest.mark.parametrize("rolling_window", [10, 30, 60])
# def test_load_fit_file_rolling_window(fit_file_path, rolling_window):
#     df = load_fit_file(fit_file_path, add_metrics=True, rolling_window=rolling_window)
#
#     # Check if rolling average is calculated correctly
#     if 'speed_rolling_avg' in df.columns:
#         # The first rolling_window - 1 values should be NaN
#         assert df['speed_rolling_avg'].iloc[:rolling_window-1].isna().all()
#         # The rest should not be NaN
#         assert not df['speed_rolling_avg'].iloc[rolling_window:].isna().any()


def test_load_fit_file_data_integrity(fit_file_path):
    df = load_fit_file(fit_file_path)

    # Check for monotonically increasing timestamps
    assert df["timestamp"].is_monotonic_increasing

    # Check for no duplicate timestamps
    assert not df["timestamp"].duplicated().any()

    # Check for reasonable speed values (e.g., below 100 m/s which is 360 km/h)
    assert df["speed"].max() < 100


if __name__ == "__main__":
    pytest.main([__file__])
