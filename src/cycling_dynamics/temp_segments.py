import numpy as np
import pandas as pd

# Constants
LAT_COL = "position_lat"
LON_COL = "position_long"
DIST_COL = "distance"
TIME_COL = "timestamp"


def create_grouped_segments(tracks: list[pd.DataFrame], start_distance: float, length: float) -> list[pd.DataFrame]:
    """Compare multiple activities over a segment.

    Args:
        tracks: List of 2 or more activity DataFrames. tracks[0] is treated as the control/base track.
        start_distance: The "distance" within track[0] to use as the control point.
        length: The length of the segment from the control points.

    Returns:
        List of trimmed and aligned DataFrames for each track.

    Raises:
        ValueError: If tracks list is empty or contains invalid DataFrames.

    """
    if not tracks or not all(isinstance(t, pd.DataFrame) for t in tracks):
        raise ValueError("tracks must be a non-empty list of pandas DataFrames")

    required_columns = [LAT_COL, LON_COL, DIST_COL, TIME_COL]
    for i, track in enumerate(tracks):
        if not all(col in track.columns for col in required_columns):
            raise ValueError(f"Track {i} is missing required columns. Required: {required_columns}")

    control = tracks[0]
    start_idx, end_idx = get_segment_indices(control, start_distance, length)
    start_point = control.iloc[start_idx][[LAT_COL, LON_COL, DIST_COL]].values
    end_point = control.iloc[end_idx][[LAT_COL, LON_COL, DIST_COL]].values

    logging.info(f"Start point lat, lon, distance: {start_point}")
    logging.info(f"End point lat, lon, distance: {end_point}")

    trimmed_tracks = [process_control_track(control, start_idx, end_idx, start_point[2])]

    for i, track in enumerate(tracks[1:], 1):
        trimmed_track = process_track(track, start_point, end_point, i)
        trimmed_tracks.append(trimmed_track)

    return trimmed_tracks


def get_segment_indices(track: pd.DataFrame, start_distance: float, length: float) -> tuple[int, int]:
    start_idx = (track[DIST_COL] - start_distance).abs().idxmin()
    end_idx = (track[DIST_COL] - (start_distance + length)).abs().idxmin()
    return start_idx, end_idx


def process_control_track(track: pd.DataFrame, start_idx: int, end_idx: int, start_distance: float) -> pd.DataFrame:
    control = track.loc[start_idx:end_idx].copy().reset_index(drop=True)
    control[DIST_COL] -= start_distance
    control["seconds"] = (control[TIME_COL] - control[TIME_COL].min()).dt.total_seconds()
    control["ride"] = 0
    return control


def process_track(track: pd.DataFrame, start_point: np.ndarray, end_point: np.ndarray, ride_num: int) -> pd.DataFrame:
    logging.info(f"Processing Track {ride_num}")

    start_idx, end_idx = find_matching_points(track, start_point, end_point)

    trimmed_track = track.loc[start_idx:end_idx].copy().reset_index(drop=True)
    trimmed_track[DIST_COL] -= trimmed_track[DIST_COL].iloc[0]
    trimmed_track["ride"] = ride_num
    trimmed_track["seconds"] = (trimmed_track[TIME_COL] - trimmed_track[TIME_COL].min()).dt.total_seconds()

    return trimmed_track


def find_matching_points(track: pd.DataFrame, start_point: np.ndarray, end_point: np.ndarray) -> tuple[int, int]:
    start_idx = track.apply(
        lambda row: haversine_distance(start_point[0], start_point[1], row[LAT_COL], row[LON_COL]), axis=1
    ).idxmin()
    end_idx = track.apply(
        lambda row: haversine_distance(end_point[0], end_point[1], row[LAT_COL], row[LON_COL]), axis=1
    ).idxmin()
    return start_idx, end_idx


# Assuming haversine_distance function is defined elsewhere
