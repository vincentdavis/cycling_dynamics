"""Main dynamics models"""
import logging
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

logging.basicConfig(level=logging.INFO)


@dataclass
class Run:
    """Run model
    :param
    data: pandas dataframe with the run data
    mass_change: Choose some starting mass. This should be the differance in mass.
    :return
    """

    data: pd.DataFrame
    mass: float  # differance from base mass
    name: str = "Run"


class RollDown:
    """Roll Down model
    Simple models ot compare single metric, Air resistance, Rolling resistance, change across multiple runs.
    Assuming
    :param
    runs: list of runs, down hill rolls with the changes in the component. The mass_change must be 0 for the first run.
    metric: metric to compare [cda, crr]
    :return
    percentage_change: percentage change in the metric
    """

    def __init__(self, runs: Iterable[Run], metric: str):
        self.runs = runs
        self.metric = metric

    def set_start(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find the point of the run, when movement starts and create a new column with the time from the start"""
        first_non_zero_index = df["speed"].gt(0).idxmax()
        # Make sure we are moving.
        try:
            assert (
                df.iloc[first_non_zero_index - 1]["speed"]
                < df.iloc[first_non_zero_index]["speed"]
                < df.iloc[first_non_zero_index + 1]["speed"]
            )
        except Exception as e:
            logging.error(
                f"Failed to find the start of the run. Check the data. {df.iloc[first_non_zero_index-1]['speed']},"
                f"{df.iloc[first_non_zero_index]['speed']}, {df.iloc[first_non_zero_index+1]['speed']}"
            )
            raise e

        filtered_df = df.loc[first_non_zero_index - 1 :]
        filtered_df.reset_index(drop=True, inplace=True)
        return filtered_df

    def merge_runs(self):
        """Merge the runs into a single dataframe"""

        data = pd.merge([self.set_start(run.data) for run in self.runs])
        return data


class SimulateRollDownDifferance:
    """Simulate Roll Down model
    Simple models ot compare single metric, Air resistance, Rolling resistance, change across multiple runs.
    Assuming
    :param
    runs: list of runs, down hill rolls with the changes in the component.
    metric: metric to compare [cda, crr]
    :return
    percentage_change: percentage change in the metric
    """

    def __init__(self, runs: list | tuple, mass_change: float, metric: str):
        self.runs = runs
        self.mass_change = mass_change
        self.metric = metric

    def sim_step(self, dt, cda1, cda2, crr1, crr2):
        pass

    def percentage_change(self):
        """Calculate the percentage change in the metric"""
        initial_metric = self.runs[0].metrics[self.metric]
        final_metric = self.runs[-1].metrics[self.metric]
        percentage_change = (final_metric - initial_metric) / initial_metric * 100
        return percentage_change


class CompareEfficiancy:
    """Compare the efficiancy of two runs
    :param
    run1: Run
    run2: Run
    :return
    percentage_change: percentage change in the metric
    """

    def __init__(self, run1: Run, run2: Run):
        self.run1 = run1
        self.run2 = run2

    def set_to_grid(self, resolution: float):
        pass
