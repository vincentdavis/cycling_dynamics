import logging

import pandas as pd
from garmin_fit_sdk import Decoder, Stream

logging.basicConfig(level=logging.INFO)


def load_fit_file(file_path: str, add_metrics: bool = True, rolling_window: int = 30) -> pd.DataFrame:
    """
    Load a FIT file and return its data as a pandas DataFrame.

    Args:
        file_path (str): Path to the FIT file.
        add_metrics (bool, optional): Flag to add additional metrics. Defaults to True.
        rolling_window (int, optional): Window size for rolling calculations. Defaults to 30.

    Returns:
        pd.DataFrame: DataFrame containing the FIT file data.

    Raises:
        ValueError: If the file is not a valid FIT file or if there are decoding errors.
    """
    logging.info("Loading FIT file")

    stream = Stream.from_file(file_path)
    decoder = Decoder(stream)

    if not decoder.is_fit():
        logging.error(f"File is not FIT: {file_path}")
        raise ValueError(f"File is not a valid FIT file: {file_path}")

    messages, errors = decoder.read()

    if errors:
        logging.error(f"Errors: {errors}")
        raise ValueError(f"Errors occurred while decoding the FIT file: {errors}")

    df = pd.DataFrame(messages["record_mesgs"])

    # Convert lat/long to decimal degrees
    for col in ["position_lat", "position_long"]:
        if col in df.columns:
            df[col] = df[col] / 1e7

    if "enhanced_speed" in df.columns and "speed" in df.columns:
        logging.info("Using enhanced speed")
        df.drop(columns=["speed"], inplace=True)
    df.rename(columns={"enhanced_speed": "speed"}, inplace=True)
    if "enhanced_altitude" in df.columns and "altitude" in df.columns:
        logging.info("Using enhanced altitude")
        df.drop(columns=["altitude"], inplace=True)
    df.rename(columns={"enhanced_altitude": "altitude"}, inplace=True)
    return df
