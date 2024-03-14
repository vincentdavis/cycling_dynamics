"""A collection of plots for visualizing cycling dynamics."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_critical_power_intensity(
    df, width: int = 15, intervals: list[int, ...] = [15, 30, 60, 120, 300, 600, 900, 1200]
) -> px.bar:
    """Plot the critical power intensity
    df: this is the dataframe from get_critical_power_intensity(df, cp=None)
    """
    if "seconds" not in df.columns:
        df["seconds"] = df.index + 1
    number_of_intervals = len(intervals)
    diff_cols_bar = [f"percent_cp_{interval}sec" for interval in intervals]
    diff_cols_bar.reverse()

    every_x_row = df.iloc[::width].copy()
    every_x_row["percent_total_cp"] = every_x_row["percent_total_cp"] * number_of_intervals
    fig = px.bar(
        every_x_row,
        x="seconds",
        y=diff_cols_bar,
        title="Critical Power Intensity",
        labels={"value": "Percent Critical Power", "variable": "% Critical Power"},
        color_discrete_sequence=px.colors.sequential.Plasma,
    )
    fig.add_scatter(
        x=every_x_row["seconds"],
        y=every_x_row["percent_total_cp"],
        mode="lines",
        name="percent_total_cp",
        line=dict(color="red", width=2),
    )
    return fig


def plot_activity_critical_power(df: pd.DataFrame) -> go.Figure:
    """Plot the critical power of an activity
    df: with df being the critical power dataframe from critical_power(df)['df']
    returns: fig
    """
    fig = go.Figure()

    upper_std = df["cp"] + df["std"]
    lower_std = df["cp"] - df["std"]

    fig.add_trace(
        go.Scatter(x=df["seconds"], y=df["cp"], fill=None, mode="lines", line_color="red", name="Critical Power")
    )
    fig.add_trace(
        go.Scatter(x=df["seconds"], y=upper_std, fill="tonexty", mode="lines", line_color="lightblue", name="STD")
    )
    fig.add_trace(
        go.Scatter(x=df["seconds"], y=lower_std, fill="tonexty", mode="lines", line_color="lightblue", name="STD")
    )
    fig.update({"layout": {"title": "Critical Power", "xaxis": {"title": "Time (s)"}, "yaxis": {"title": "Power (W)"}}})

    return fig
