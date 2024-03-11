import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)


def _interpolate_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate the power curve to 1 second intervals"""
    max_time = df["seconds"].max()
    df.set_index("seconds", inplace=True)
    # fill in missing seconds
    new_index = pd.Index(range(1, max_time + 1), name="seconds")
    df = df.reindex(new_index)
    # Interpolate power for missing seconds
    df["power"] = df["power"].interpolate(method="linear")
    df.reset_index(inplace=True)
    return df


def _calculate_ramp_power(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the ramp power for each second"""
    df["ramp_power"] = 0.0
    for idx in df.index:
        if idx == 0:
            df.loc[idx, "ramp_power"] = df.loc[idx, "power"]
        else:
            df.loc[idx, "ramp_power"] = df.loc[idx, "power"] * df.loc[idx, "seconds"] - df.loc[: idx - 1][
                "ramp_power"
            ].sum().clip(min=0)
    return df


def convert_user_critical_power(profile: dict[int, float]) -> tuple[pd.DataFrame, dict[int, float]]:
    """Convert a user defined critical power to a dict of duration: power for each second
    Returns: a df and a dict"""
    logging.info("Convert user defined critical power")
    try:
        assert profile[1] >= 0
    except AssertionError:
        raise ValueError("The profile must start with 1 second")

    df = pd.DataFrame([(k, v) for k, v in profile.items()], columns=["seconds", "power"])
    df_cp = _interpolate_curve(df)
    cp = df_cp.set_index("seconds")["power"].to_dict()
    return df_cp, cp


def ramp_test_activity(
    profile: dict[int, float], segment_time: int = 30, test_length: int = 1200, ftp: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a power profile to a ramp test workout
    The last 30sec is always 1 second ramp.
    profile: list of tuples of seconds and power [(1, 100), (2, 90), (3, 85), (4, 80)] must start with 1sec
    segment_time: int, the time in seconds of each segment of the ramp test workout, starting after 30sec
    :return: full dataframe, workout segment dataframe, workout segment dataframe with power per ftp
    """
    df, cp = convert_user_critical_power(profile)
    try:
        assert max(profile.keys()) >= test_length
    except AssertionError:
        raise ValueError("The profile must end with with a time >= test_length")
    df = pd.DataFrame(profile, columns=["seconds", "power"])
    df = _interpolate_curve(df)
    df = df[df.seconds <= test_length].copy()

    df = _calculate_ramp_power(df)

    df["bins"] = df.apply(lambda row: row.name // segment_time + 30 if int(row.name) > 30 else row.name, axis=1)
    df["bin_power"] = df.groupby("bins")["ramp_power"].transform("mean").round(0)
    df["bin_time"] = df.groupby("bins")["seconds"].transform("count")
    df["power_per_ftp"] = df["power"] / ftp
    df["ramp_power_per_ftp"] = df["ramp_power"] / ftp
    df["bin_power_per_ftp"] = df["bin_power"] / ftp
    df["WKO Critical Power"] = df["bin_power"].expanding().mean()
    df_wko = df.drop_duplicates(subset=["bins"], keep="first")
    df_wko = df_wko[["bins", "bin_time", "bin_power", "bin_power_per_ftp"]].copy()
    df_wko.rename(
        columns={"bin_power": "power", "bin_time": "duration", "bin_power_per_ftp": "power%ftp"}, inplace=True
    )
    df_wko.sort_values(by="bins", ascending=False, inplace=True)
    df_wko.reset_index(drop=True, inplace=True)
    df_wko["segment"] = df_wko.index + 1
    return df, df_wko[["segment", "duration", "power", "power%ftp"]]


def make_zwo_from_ramp(workout: pd.DataFrame, filename: str | None, name: str, ftp: int | None = 1):
    xml = ""
    xml += "<workout_file>\n"
    xml += "  <author>Vincent Davis</author>\n"
    xml += f"  <name>Most Painful Ramp Test {name}</name>\n"
    xml += "  <description>A ramp test based on power profile</description>\n"
    xml += "  <sportType>bike</sportType>\n"
    xml += "  <tags></tags>\n"
    if ftp:
        xml += f"  <ftpOverride>{ftp}</ftpOverride>\n"
    xml += "  <workout>\n"
    for r in workout.to_dict(orient="records"):
        d = r["duration"]
        p = r["power%ftp"]
        xml += f'      <SteadyState Duration="{d}" Power="{p}"/>\n'
    xml += "    </workout>\n"
    xml += "</workout_file>\n"
    if filename:
        with open(filename, "w") as f:
            f.write(xml)
    return xml


def critical_power(df: pd.DataFrame) -> dict[int, float]:
    """Calculate the critical power of a ride
    The critical power is the highest power that observed for a given duration.
    returns:
    dict of duration: power starting at 1sec to the full length f the dataframe.
    """
    logging.info("calculate critical power")
    cp = {s: df["power"].rolling(window=s).mean().max() for s in range(1, len(df) + 1)}
    cp = {k: v for k, v in cp.items() if v >= 0}
    max_sec = max(list(cp.keys()))
    logging.info(f"Critical power: 1sec = {cp[1]}, {max_sec}sec = {cp[max_sec]}")
    return cp


def get_power_roll_avg_df(df: pd.DataFrame, max_window: int = 1200) -> pd.DataFrame:
    """Add rolling average power to the dataframe
    df: dataframe with a power column
    """
    max_window = min(max_window, len(df))
    rolling_power = [df["power"].rolling(window=rolling_window).mean() for rolling_window in range(1, max_window + 1)]
    rolling_power_names = [n for n in range(1, max_window + 1)]
    rolling_power_df = pd.concat(rolling_power, keys=rolling_power_names, axis=1)
    rolling_power_df.dropna(axis=1, how="all", inplace=True)
    rolling_power_df.columns = pd.concat([df[["timestamp", "power"]], rolling_power_df], axis=1)
    return rolling_power_df


def get_critical_power_intesity(df: pd.DataFrame, cp: dict[int, float] | bool = False) -> pd.DataFrame:
    """Add critical power intesity to the dataframe
    df: dataframe with a power column
    cp: list of lists of duration and power or user defined critical power

    """
    if not critical_power:
        df, cp = critical_power(df)
    df = _interpolate_curve(df)
    return df
