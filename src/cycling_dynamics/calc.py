"""Python module to contain base calculating functions."""

import logging
from math import asin, atan, cos, radians, sin, sqrt, tan

import numpy as np
import pandas as pd


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points in a sphere, given longitudes and latitudes.

     Referance: https://en.wikipedia.org/wiki/Haversine_formula.

    We know that the globe is "sort of" spherical, so a path between two points
    isn't exactly a straight line. We need to account for the Earth's curvature
    when calculating distance from point A to B. This effect is negligible for
    small distances but adds up as distance increases. The Haversine method treats
    the earth as a sphere which allows us to "project" the two points A and B
    onto the surface of that sphere and approximate the spherical distance between
    them. Since the Earth is not a perfect sphere, other methods which model the
    Earth's ellipsoidal nature are more accurate but a quick and modifiable
    computation like Haversine can be handy for shorter range distances.

    Args:
        lat1, lon1: latitude and longitude of coordinate 1
        lat2, lon2: latitude and longitude of coordinate 2
    Returns:
        geographical distance between two points in metres
    >>> from collections import namedtuple
    >>> point_2d = namedtuple("point_2d", "lat lon")
    >>> SAN_FRANCISCO = point_2d(37.774856, -122.424227)
    >>> YOSEMITE = point_2d(37.864742, -119.537521)
    >>> f"{haversine_distance(*SAN_FRANCISCO, *YOSEMITE):0,.0f} meters"
    '254,352 meters'

    """
    # CONSTANTS per WGS84 https://en.wikipedia.org/wiki/World_Geodetic_System
    # Distance in metres(m)
    AXIS_A = 6378137.0
    AXIS_B = 6356752.314245
    RADIUS = 6378137
    # Equation parameters
    # Equation https://en.wikipedia.org/wiki/Haversine_formula#Formulation
    flattening = (AXIS_A - AXIS_B) / AXIS_A
    phi_1 = atan((1 - flattening) * tan(radians(lat1)))
    phi_2 = atan((1 - flattening) * tan(radians(lat2)))
    lambda_1 = radians(lon1)
    lambda_2 = radians(lon2)
    # Equation
    sin_sq_phi = sin((phi_2 - phi_1) / 2)
    sin_sq_lambda = sin((lambda_2 - lambda_1) / 2)
    # Square both values
    sin_sq_phi *= sin_sq_phi
    sin_sq_lambda *= sin_sq_lambda
    h_value = sqrt(sin_sq_phi + (cos(phi_1) * cos(phi_2) * sin_sq_lambda))
    return 2 * RADIUS * asin(h_value)


def angle_type(a, b, c):
    """Calculate the cosine of the angle using Law of Cosines and determine the angle type.

    If the angle between sides a and b acute, obtuse or 90 degrees (right angle)
     a: The original control point to next point on path
     b: The original control point to point on other path
     c: Second point on path the other path point.
    """
    angle = (a**2 + b**2 - c**2) / (2 * a * b)

    # Determine the angle type
    if angle > 0:  # After control point
        return "after"
    elif angle < 0:  # Before control point
        return "before"
    else:
        return "beside"  # cos_C == 0 implies a right angle


def add_metrics(df: pd.DataFrame, rolling_window: int = 30, ftp: int | None = None) -> pd.DataFrame:
    """Add metrics to the dataframe"""
    logging.info("Adding metrics")
    # Speed
    df[f"speed {rolling_window}sec"] = df["speed"].rolling(window=rolling_window).mean()

    # Efficiency
    df[f"speed per watt {rolling_window}sec"] = (
        df["speed"].rolling(window=rolling_window).mean() / df["power"].rolling(window=rolling_window).mean()
    )
    df[f"speed sqrd per watt {rolling_window}sec"] = (
        df["speed"].rolling(window=rolling_window).mean() ** 2 / df["power"].rolling(window=rolling_window).mean()
    )

    # power
    df["np"] = (df["power"] ** 4).rolling(window=30).mean() ** 0.25
    if ftp is not None:
        df = intensity_factor(df, ftp=ftp)
        df = total_training_stress(df, ftp=ftp)
        logging.info(f"Total Training Stress (TSS): {df['TSS'].iloc[-1]:0.2f} kcal/hr (FTP: {ftp})")
    return df


def total_training_stress(df: pd.DataFrame, ftp: int) -> pd.DataFrame:
    """Calculat Total Training Stress."""
    df["TSS"] = (df["power"] * df["IF"] * df["seconds"] / ftp / 3600).cumsum()
    return df


def normalized_power(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the normalized power of a ride."""
    df["np"] = (df["power"] ** 4).rolling(window=30).mean() ** 0.25
    return df


def intensity_factor(df: pd.DataFrame, ftp: int) -> pd.DataFrame:
    """Calculate the intensity factor of a ride."""
    df["IF"] = ((df["power"] ** 4).rolling(window=30).mean() ** 0.25) / ftp
    return df


def vam(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate vertical ascent rate."""
    # TODO Improve calculation by using 3 points
    df["vam"] = (df["altitude"].diff() / df.seconds.diff()) * 3600
    return df


def slope(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the slope."""
    # TODO Improve calculation by using 3 points
    df["slope"] = df["altitude"].diff() / df.distance.diff()
    df["slope_3sec"] = df["slope"].rolling(window=3, center=True).mean()
    df["slope_3sec"] = df["slope_3sec"].fillna(df["slope_3sec"])
    return df


def zero_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Create or reset seconds callumn starting at zero."""
    # TODO: make sure it is sorted
    df["seconds"] = df["timestamp"].sub(df["timestamp"].min()).dt.total_seconds()
    # if "seconds" not in df.columns:
    #     df["seconds"] = pd.to_datetime(df.index, unit="s", origin="unix").astype(int) // 10**9
    # df["seconds"] = df["seconds"] - df.seconds.min()
    return df


def speed(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate speed if missing."""
    # TODO Improve calculation by using 3 points
    df["speed"] = df.distance.diff() / df.time.diff()

def speed_3sec(df: pd.DataFrame) -> pd.DataFrame:
    df["speed_3sec"] = df["speed"].rolling(window=3, center=True).mean()
    df['speed_3sec'] = df['speed_moving_avg'].fillna(df['speed'])
    return df


def air_density(df: pd.DataFrame, temperature: float = 30) -> pd.DataFrame:
    """Calculate air density."""
    if temperature not in df.columns:
        logging.info("Using default temperature of 30 degrees C")
        df["temperature"] = 30
    df["air_density"] = (
        (101325 / (287.05 * 273.15))
        * (273.15 / (df["temperature"] + 273.15))
        * np.exp((-101325 / (287.05 * 273.15)) * 9.8067 * (df["altitude"] / 101325))
    )
    return df
