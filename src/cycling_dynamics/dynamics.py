"""Dynamic simulations."""

import numpy as np
import pandas as pd


def simulator(
    df: pd.DataFrame,
    smoothing: int = 3,
    rider_weight: float = 65.0,
    bike_weight: float = 5.0,
    wind_speed: float = 0,
    wind_direction: int = 0,
    temperature: float = 30,
    drag_coefficient: float = 0.8,
    frontal_area: float = 0.565,
    rolling_resistance: float = 0.005,
    efficiency_loss: float = 0.04,
) -> pd.DataFrame:
    """Calculate the components of power loss at each point and estimated total power needed.

    df: Usually from a FIT file or other GPS file. Required columns [distance, speed and/or time, altitude]
    """
    try:
        assert all([c in df.columns for c in ["seconds", "distance", "altitude"]])
    except AssertionError:
        raise AssertionError("Missing columns in dataframe. Must have 'seconds', 'distance', and 'altitude'")

    CdA = drag_coefficient * frontal_area

    df["effective_wind_speed"] = np.cos(np.radians(wind_direction)) * wind_speed

    # Components of power, watts
    df["air_drag_watts"] = (
        0.5 * CdA * df["air_density"] * np.square(df["speed"] + df["effective_wind_speed"]) * df["speed"]
    )
    df["climbing_watts"] = (bike_weight + rider_weight) * 9.8067 * np.sin(np.arctan(df["slope"])) * df["speed"]
    df["rolling_watts"] = (
        np.cos(np.arctan(df["slope"])) * 9.8067 * (bike_weight + rider_weight) * rolling_resistance * df["speed"]
    )
    # TODO: Diff should use smooting.
    df["acceleration_watts"] = (bike_weight + rider_weight) * (df["speed"].diff() / df["seconds"].diff()) * df["speed"]
    df["est_power_no_loss"] = df[["air_drag_watts", "climbing_watts", "rolling_watts", "acceleration_watts"]].sum(
        axis="columns"
    )
    df["est_power"] = df["est_power_no_loss"] / (1 - efficiency_loss)
    df["efficiency_loss_watts"] = df["est_power_no_loss"] - df["est_power"]
    df["est_power_no_acceleration"] = (df["est_power_no_loss"] - df["acceleration_watts"]) / (1 - efficiency_loss)
    df["power_error"] = df["est_power"] - df["power"]

    if smoothing > 0:
        df["speed_smoothed"] = df["speed"].rolling(window=smoothing, center=True).mean()
        df["slope_smoothed"] = df["slope"].rolling(window=smoothing, center=True).mean()
        df["power_smoothed"] = df["power"].rolling(window=smoothing, center=True).mean()
        df["air_drag_watts_smoothed"] = df["air_drag_watts"].rolling(window=smoothing, center=True).mean()
        df["climbing_watts_smoothed"] = df["climbing_watts"].rolling(window=smoothing, center=True).mean()
        df["rolling_watts_smoothed"] = df["rolling_watts"].rolling(window=smoothing, center=True).mean()
        df["est_power_smoothed"] = df["est_power"].rolling(window=smoothing, center=True).mean()
        df["efficiency_loss_watts_smoothed"] = df["efficiency_loss_watts"].rolling(window=smoothing, center=True).mean()
        df["acceleration_watts_smoothed"] = df["acceleration_watts"].rolling(window=smoothing, center=True).mean()
        df["est_power_no_acceleration_smoothed"] = (
            df["est_power_no_acceleration"].rolling(window=smoothing, center=True).mean()
        )
        df["power_error_smoothed"] = df["est_power_smoothed"] - df["power_smoothed"]

    return df
