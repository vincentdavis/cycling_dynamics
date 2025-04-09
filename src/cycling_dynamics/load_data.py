import logging

import pandas as pd
from garmin_fit_sdk import Decoder, Stream

from cycling_dynamics import calc
from cycling_dynamics.calc import slope

logging.basicConfig(level=logging.INFO)


def load_fit_file(
    file_path: str, add_fields: tuple[str] = ("seconds", "slope", "vam", "speed", "normalized_power", "air_density")
) -> pd.DataFrame:
    """Load a FIT file and return its data as a pandas DataFrame.

    Args:
        file_path (str): Path to the FIT file.
        add_fields (bool, optional): Flag to add additional metrics. Defaults to True.

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

    if 'altitude' not in df.columns:
        df['altitude'] = 0

    # Convert lat/long to decimal degrees
    for col in ["position_lat", "position_long"]:
        if col in df.columns:
            df[col] = df[col] / 1e7

    enhanced_cols = ["enhanced_distance", "enhanced_speed", "enhanced_altitude"]
    for col in enhanced_cols:
        if col in df.columns:
            logging.info(f"Using {col}")
            if col.split("_")[1] in df.columns:
                df.drop(columns=[col.split("_")[1]], inplace=True)
            df.rename(columns={col: col.split("_")[1]}, inplace=True)

    if "seconds" in add_fields and "seconds" not in df.columns:
        logging.info("Using calculated seconds")
        df = calc.zero_seconds(df)
    if "speed" in add_fields and "speed" not in df.columns:
        logging.info("Using calculated speed")
        df = calc.speed(df)
    if "slope" in add_fields and "slope" not in df.columns:
        logging.info("Using calculated slope")
        df = slope(df)
    if "vam" in add_fields and "vam" not in df.columns:
        logging.info("Using calculated vertical acceleration magnitude")
        df = calc.vam(df)
    if "normalized_power" in add_fields and "normalized_power" not in df.columns:
        logging.info("Using calculated normalized power")
        df = calc.normalized_power(df)
    if "air_density" in add_fields and "air_density" not in df.columns:
        logging.info("Using calculated air density")
        df = calc.air_density(df)

    return df
