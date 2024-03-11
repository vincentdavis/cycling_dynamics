import logging

import pandas as pd

from src.cycling_dynamics.calc import haversine_distance

logging.basicConfig(level=logging.INFO)


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
        df["IF"] = df["np"] / ftp
        df["TSS"] = (df["power"] * df["IF"] * df["seconds"] / ftp / 3600).cumsum()
    return df


def create_grouped_segments(
    tracks: list[pd.DataFrame, ...], start_distance: float, length: float
) -> list[pd.DataFrame]:
    """Compare multiple activities over a segment.
    Tracks: 2 or more activities in the form of a dataframe.
    Track[0] is treated as the control/base track.
    start_point: The "distance" withing track[0] to use as the control point. You can use the exact or the closest will
    be found.
    length: The length of the segment from the control points
    """
    # start_idx = control[control['distance']==start_point]['distance'].sub(start_point).abs().idxmin()
    start_idx = tracks[0]["distance"].sub(start_distance).abs().idxmin()
    logging.info(f"Start index: {start_idx}")
    end_idx = tracks[0]["distance"].sub(start_distance + length).abs().idxmin()
    logging.info(f"End index: {start_idx}")
    start_point = tracks[0].iloc[start_idx][["position_lat", "position_long", "distance"]].values.tolist()
    logging.info(f"Start point lat, lon, distance: {start_point}")
    end_point = tracks[0].iloc[end_idx][["position_lat", "position_long", "distance"]].values.tolist()
    logging.info(f"End point lat, lon, distance: {end_point}")

    # create trimmed track
    control = tracks[0].loc[start_idx:end_idx].copy()
    control.reset_index(drop=True, inplace=True)
    logging.info(f"Distance: {control['distance'].min()}, {control['distance'].max()}")
    control["distance"] = control["distance"] - start_point[2]
    logging.info(f"Distance: {control['distance'].min()}, {control['distance'].max()}")
    control["seconds"] = control["timestamp"].sub(control["timestamp"].min()).dt.total_seconds()
    control["ride"] = 0

    trimmed_tracks = [control]
    for i, track in enumerate(tracks[1:]):
        logging.info(f"Track {i + 1}")
        track["distance_to_start"] = track.apply(
            lambda row: haversine_distance(start_point[0], start_point[1], row["position_lat"], row["position_long"]),
            axis=1,
        )
        match_start_idx = track["distance_to_start"].idxmin()
        logging.info(f"Match start index: {match_start_idx}")
        match_start_point = track.iloc[match_start_idx][["position_lat", "position_long", "distance"]].values.tolist()
        logging.info(f"Match start point lat, lon, distance: {match_start_point}")

        track["distance_to_end"] = track.apply(
            lambda row: haversine_distance(end_point[0], end_point[1], row["position_lat"], row["position_long"]),
            axis=1,
        )
        match_end_idx = track["distance_to_end"].idxmin()
        logging.info(f"Match start index: {match_end_idx}")
        match_end_point = track.iloc[match_end_idx][["position_lat", "position_long", "distance"]].values.tolist()
        logging.info(f"Match end point lat, lon, distance: {match_end_point}")

        trimmed_track = track.loc[match_start_idx:match_end_idx].copy()
        trimmed_track.reset_index(drop=True, inplace=True)
        trimmed_track["distance"] = trimmed_track["distance"] - match_start_point[2]
        trimmed_track["ride"] = i + 1
        trimmed_track["seconds"] = trimmed_track["timestamp"].sub(trimmed_track["timestamp"].min()).dt.total_seconds()
        trimmed_tracks.append(trimmed_track)

    return trimmed_tracks


def normalized_power(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the normalized power of a ride"""
    df["np"] = (df["power"] ** 4).rolling(window=30).mean() ** 0.25
    return df["np"]


def intensity_factor(df: pd.DataFrame, ftp: int) -> pd.DataFrame:
    """Calculate the intensity factor of a ride"""
    df["IF"] = ((df["power"] ** 4).rolling(window=30).mean() ** 0.25) / ftp
    return df
