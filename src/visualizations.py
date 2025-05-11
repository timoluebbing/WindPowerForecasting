import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tueplots import bundles
from tueplots.constants.color import rgb


def setup_tueplots(column="full", n_rows=1, n_cols=1, use_tex=True):
    plt.rcParams.update(
        bundles.icml2022(
            column=column,
            nrows=n_rows,
            ncols=n_cols,
            usetex=use_tex,
        )
    )

def plot_wind_power_composition(
    dates, wind_offshore, wind_onshore, moving_average_window=None
):
    """
    Plot the composition of wind power in Germany, including offshore and onshore wind power,
    as well as their moving averages.

    Args:
        dates (pd.Series): Timestamps for the data points.
        wind_offshore (pd.Series): Wind power generated offshore.
        wind_onshore (pd.Series): Wind power generated onshore.
        moving_average_window (int, optional): The window size for the moving average. Defaults to None.

    Returns:
        plt.Figure: The figure object containing the plot.
    """
    time_interval = (
        f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    locator = ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator=locator))

    ax.plot(dates, wind_offshore, label="Wind Offshore [MW]", color="orange", alpha=0.7)
    ax.plot(
        dates,
        wind_offshore.rolling(window=moving_average_window).mean(),
        label=f"Wind Offshore [MW] - MA ({moving_average_window} hours)",
        color="red",
        alpha=0.9,
    )
    ax.plot(
        dates,
        wind_onshore + wind_offshore,
        label="Wind Sum [MW]",
        color="green",
        alpha=0.7,
    )
    ax.plot(
        dates,
        (wind_onshore + wind_offshore).rolling(window=moving_average_window).mean(),
        label=f"Wind Sum [MW] - MA ({moving_average_window} hours)",
        color="darkgreen",
        alpha=0.9,
    )
    ax.fill_between(
        dates,
        wind_offshore,
        wind_onshore + wind_offshore,
        color="lightgreen",
        alpha=0.5,
        label="Wind Onshore Contribution",
    )
    ax.fill_between(
        dates,
        0,
        wind_offshore,
        color="lightcoral",
        alpha=0.5,
        label="Wind Offshore Contribution",
    )

    ax.set_title(f"Composition of Wind Power in Germany [{time_interval}]")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Wind Power [MW]")
    ax.legend()
    plt.xticks(rotation=45)
    plt.show()

    return fig
