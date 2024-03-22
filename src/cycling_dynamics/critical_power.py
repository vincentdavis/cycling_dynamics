"""Critical power related code"""

import logging
import warnings
from dataclasses import asdict, dataclass

import pandas as pd

from src.cycling_dynamics.load_data import load_fit_file

logging.basicConfig(level=logging.INFO)


@dataclass
class CPPoint:
    """Dataclass to hold critical power data"""

    seconds: int
    idx: int  # index location of the cp in the dataframe
    cp: float
    std: float
    max: float
    min: float
    slope: float
    window: pd.DataFrame  # copy of the window data
    extra_cols: list[str]  # list of extra columns to calculate
    extra_data: dict[str:list]  # dict of data. Column names with list of values.
    intensity: float = 0
    chr: float = 0
    chr_std: float = 0
    chr_max: float = 0
    chr_min: float = 0


def _calculate_ramp_power(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the ramp power for each second
    df is a dataframe of a 1sec resolution CP curve"""
    df["ramp_power"] = 0.0
    for idx in df.index:
        if idx == 0:
            df.loc[idx, "ramp_power"] = df.loc[idx, "power"]
        else:
            df.loc[idx, "ramp_power"] = df.loc[idx, "power"] * df.loc[idx, "seconds"] - df.loc[: idx - 1][
                "ramp_power"
            ].sum().clip(min=0)
    return df


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


class CriticalPower:
    """Critical Power class
    activity: Activity dataframe, file path or None
    cp_user: dict of duration: power, User defined critical power or None. This gets converted to a dataframe and dict
    with 1sec resolution
    max_window: int, the maximum duration to calculate, interpolate critical power, default is 1200 seconds
    """

    def __init__(
        self,
        activity: pd.DataFrame | str | None = None,
        cp_user: dict[int, float] | None = None,
        max_window: int = 1200,
    ):
        self.activity = load_fit_file(activity) if isinstance(activity, str) else activity
        self.max_window = max_window
        self.cp_points: dict[int, CPPoint] = {}
        self.cp_df: pd.DataFrame = pd.DataFrame()
        self.cp_defined_df = None  # this is the user defined CP
        self.cp_defined_dict = None  # this is the user defined CP
        self.ramp_test_df: pd.DataFrame | None = None
        self.ramp_test_wko: pd.DataFrame | None = None
        self.activity_percent_cp: float | None = None
        self.df_rolling: pd.DataFrame | None = None

        if cp_user is not None:
            self._convert_cp_defined(cp_user)

    def _convert_cp_defined(self, cp_user: dict[int, float]):
        """Convert a user defined critical power to a dict and Dataframe interpolated to every second.
        Returns: a df and a dict"""
        logging.info("Convert cp_defined critical power")
        try:
            assert cp_user[1] >= 0
        except AssertionError as err:
            logging.error(f"Critical power must be greater than 0: {err}")
            raise AssertionError("The profile must start with 1 second") from err
        logging.info(f"Last max window of profile: {max(cp_user.keys())}")

        df = pd.DataFrame([(k, v) for k, v in cp_user.items()], columns=["seconds", "power"])
        self.cp_defined_df = _interpolate_curve(df)
        self.cp_defined_dict = self.cp_defined_df.set_index("seconds")["power"].to_dict()

    def calculate_cp(self):
        """Calculate the critical power of an activity"""
        logging.info(f"Calculate critical power up to {self.max_window} seconds")
        df_data = []
        for s in range(1, self.max_window + 1):
            # Power based CP
            self.activity[f"{s}_mean"] = self.activity["power"].rolling(window=s).mean()
            if not self.activity[f"{s}_mean"].isnull().all():
                idx = self.activity[f"{s}_mean"].idxmax()
                window = self.activity.loc[idx - s + 1 : idx].copy().reset_index(drop=True)
                cp_w = window["power"].mean()
                std_w = window["power"].std()
                max_w = window["power"].max()
                min_w = window["power"].min()
                slope_w = window["power"].loc[s // 2 : s].mean() - window["power"].loc[: s // 2].mean()
                try:
                    chr_w = window["heart_rate"].mean()
                    chr_std_w = window["heart_rate"].std()
                    chr_max_w = window["heart_rate"].max()
                    chr_min_w = window["heart_rate"].min()
                except Exception as err:
                    logging.warning("Could not calculate heart rate metrics")
                    logging.warning(err)
                    chr_w = 0
                    chr_std_w = 0
                    chr_max_w = 0
                    chr_min_w = 0

                cols = [
                    col
                    for col in window.columns
                    if col
                    not in ["seconds", "power", "timestamp", "position_lat", "position_long", f"{s}_mean", "heart_rate"]
                    and not isinstance(col, int)
                ]
                extra_data = {}
                for c in cols:
                    try:
                        extra_data[f"{c}_mean"] = window[c].mean()
                        extra_data[f"{c}_std"] = window[c].std()
                        extra_data[f"{c}_max"] = window[c].max()
                        extra_data[f"{c}_min"] = window[c].min()
                    except Exception as err:
                        logging.warning(f"Could not calculate metrics for {c}")
                        logging.warning(err)
                cpp = CPPoint(
                    seconds=s,
                    idx=idx,
                    window=window,
                    cp=cp_w,
                    std=std_w,
                    max=max_w,
                    min=min_w,
                    slope=slope_w,
                    extra_cols=cols,
                    extra_data=extra_data,
                    chr=chr_w,
                    chr_std=chr_std_w,
                    chr_max=chr_max_w,
                    chr_min=chr_min_w,
                )
                self.cp_points[s] = cpp
                df_data_row = {
                    k: v for k, v in asdict(cpp).items() if k in ["seconds", "idx", "cp", "std", "max", "min", "slope"]
                }
                df_data_row.update(extra_data)
                df_data.append(df_data_row)
            self.activity.drop(columns=f"{s}_mean", inplace=True)
        self.cp_df = pd.DataFrame(df_data)

    def get_power_roll_avg_df(self, window: int = 1200) -> pd.DataFrame:
        """Add rolling average power to the dataframe
        df: dataframe with a power column
        """
        max_window = min(window, len(self.activity))
        rolling_power = [
            self.activity["power"].rolling(window=rolling_window, min_periods=1).mean()
            for rolling_window in range(1, max_window + 1)
        ]
        rolling_power_names = [f"power_{n}sec" for n in range(1, max_window + 1)]
        rolling_power_df = pd.concat(rolling_power, keys=rolling_power_names, axis=1)
        rolling_power_df.dropna(axis=1, how="all", inplace=True)
        rolling_power_df = pd.concat([self.activity[["timestamp", "power"]], rolling_power_df], axis=1)
        return rolling_power_df

    def cp_intensity(self, cp_activity=True, length: int = 1200) -> tuple[float, pd.DataFrame]:
        """Add critical power intensity to the activity dataframe
        df: dataframe with a power column
        cp_user: True=use user defined critical power, False=use calculated critical power from activity
        length: int, the length of CP curve, default is 1200 seconds
        """
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        logging.info("Calculate critical power intensity")
        if cp_activity and not self.cp_points:
            logging.info("Calculate critical power from ride data")
            self.calculate_cp()
        logging.info("Calculate critical power intensity")
        df_rolling = self.get_power_roll_avg_df(length)
        logging.info("Calculate critical power intensity")
        for col in [col for col in df_rolling.columns if "power_" in col]:  # 1 second to 1200 seconds
            # Calculate rolling average for the interval
            interval = int(col.split("_")[1].replace("sec", ""))
            if cp_activity:
                df_rolling[f"percent_cp_{interval}sec"] = df_rolling[col] / self.cp_points[interval].cp
            else:
                df_rolling[f"percent_cp_{interval}sec"] = df_rolling[col] / self.cp_defined_dict[interval]
        df_rolling["percent_total_cp"] = (
            df_rolling[[c for c in df_rolling.columns if "percent_cp_" in c]].sum(axis=1) / length
        )
        self.activity_percent_cp = df_rolling["percent_total_cp"].mean()
        self.df_rolling = df_rolling[[col for col in df_rolling.columns if "percent_" in col]]

        return self.activity_percent_cp, self.df_rolling

    def ramp_test_activity(
        self, segment_time: int = 30, test_length: int = 1200, ftp: int = 1
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert a power profile to a ramp test workout
        The last 30sec is always 1 second ramp.
        profile: list of tuples of seconds and power [(1, 100), (2, 90), (3, 85), (4, 80)] must start with 1sec
        segment_time: int, the time in seconds of each segment of the ramp test workout, starting after 30sec
        :return: full dataframe, workout segment dataframe, workout segment dataframe with power per ftp
        """
        try:
            assert max(self.cp_defined_dict.keys()) >= test_length
        except AssertionError as err:
            raise ValueError("The profile must end with with a time >= test_length") from err

        df = self.cp_defined_df[self.cp_defined_df.seconds <= test_length].copy()

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
        self.ramp_test_df = df
        self.ramp_test_wko = df_wko[["segment", "duration", "power", "power%ftp"]]
        return df, df_wko[["segment", "duration", "power", "power%ftp"]]

    def make_zwo_from_ramp(self, workout: pd.DataFrame, filename: str | None, name: str, ftp: int | None = 1):
        xml = "<?xml version='1.0' encoding='UTF-8'?>\n"
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
