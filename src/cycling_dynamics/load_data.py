import logging

import pandas as pd
from garmin_fit_sdk import Decoder, Stream

logging.basicConfig(level=logging.INFO)


def load_fit_file(file_path, add_metrics=True, rolling_window=30) -> pd.DataFrame:
    logging.info("Loading FIT file")
    stream = Stream.from_file(file_path)
    decoder = Decoder(stream)
    if not decoder.is_fit():
        logging.error(f"File is not FIT: {file_path}")
        raise f"File is not FIT: {file_path}"
    messages, errors = decoder.read()
    if errors:
        logging.error(f"Errors: {errors}")
        raise f"Errors: {errors}"
    df = pd.DataFrame(messages["record_mesgs"])
    df["position_lat"] = df["position_lat"] / 1e7
    df["position_long"] = df["position_long"] / 1e7
    if "enhanced_speed" in df.columns and "speed" in df.columns:
        logging.info("Using enhanced speed")
        df.drop(columns=["speed"], inplace=True)
    df.rename(columns={"enhanced_speed": "speed"}, inplace=True)
    if "enhanced_altitude" in df.columns and "altitude" in df.columns:
        logging.info("Using enhanced altitude")
        df.drop(columns=["altitude"], inplace=True)
    df.rename(columns={"enhanced_altitude": "altitude"}, inplace=True)
    return df
