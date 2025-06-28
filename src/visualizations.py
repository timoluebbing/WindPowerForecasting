import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tueplots import bundles
from tueplots.constants.color import rgb


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",  # or 'sans-serif' or other LaTeX fonts
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)


def setup_tueplots(rel_width=1.5, n_rows=1, n_cols=1, use_tex=True):
    plt.rcParams.update(
        bundles.neurips2024(
            rel_width=rel_width,
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

    fig, ax = plt.subplots(figsize=(12, 5))
    locator = ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator=locator))

    ax.plot(
        dates,
        wind_offshore,
        label=r"Wind Offshore [MW]",
        color=rgb.tue_ocre,
        alpha=0.7,
    )
    ax.plot(
        dates,
        wind_offshore.rolling(window=moving_average_window).mean(),
        label=rf"Wind Offshore [MW] - MA ({moving_average_window} hours)",
        color=rgb.tue_red,
        alpha=0.9,
    )
    
    ax.plot(
        dates,
        wind_onshore + wind_offshore,
        label=r"Wind Sum [MW]",
        color=rgb.tue_darkblue,
        alpha=0.7,
    )
    ax.plot(
        dates,
        (wind_onshore + wind_offshore).rolling(window=moving_average_window).mean(),
        label=rf"Wind Sum [MW] - MA ({moving_average_window} hours)",
        color="black",
        alpha=0.9,
    )
    
    ax.fill_between(
        dates,
        wind_offshore,
        wind_onshore + wind_offshore,
        color=rgb.tue_lightblue,
        alpha=0.5,
        label=r"Wind Onshore Contribution",
    )
    ax.fill_between(
        dates,
        0,
        wind_offshore,
        color=rgb.tue_orange,
        alpha=0.5,
        label=r"Wind Offshore Contribution",
    )

    ax.set_title(r"Composition of Wind Power in Germany")
    ax.set_xlabel(r"Time (hours)")
    ax.set_ylabel(r"Wind Power [MW]")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return fig


def plot_wind_speed_components_no_split(weather: pd.DataFrame) -> None:
    u10 = weather["u10"]
    u100 = weather["u100"]
    v10 = weather["v10"]
    v100 = weather["v100"]

    # Plot the data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    locator1 = ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator1))
    locator2 = ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator2))

    # Plot first subplot - u components
    ax1.plot(weather.index, u10, label="u10 [m/s]", color="blue", alpha=0.5)
    ax1.plot(weather.index, u100, label="u100 [m/s]", color="green", alpha=0.5)
    ax1.plot(weather.index, u10.rolling(window=24*7).mean(), label="u10 Weekly MA [m/s]", color="darkblue", alpha=1)
    ax1.plot(weather.index, u100.rolling(window=24*7).mean(), label="u100 Weekly MA [m/s]", color="darkgreen", alpha=1)
    ax1.set_title("East-West Wind Speed Components (u)")
    ax1.set_ylabel("Wind Speed [m/s]")
    ax1.legend()

    # Plot second subplot - v components
    ax2.plot(weather.index, v10, label="v10 [m/s]", color="yellow", alpha=0.5)
    ax2.plot(weather.index, v100, label="v100 [m/s]", color="red", alpha=0.5)
    ax2.plot(weather.index, v10.rolling(window=24*7).mean(), label="v10 Weekly MA [m/s]", color="darkorange", alpha=1)
    ax2.plot(weather.index, v100.rolling(window=24*7).mean(), label="v100 Weekly MA [m/s]", color="black", alpha=1)
    ax2.set_title("North-South Wind Speed Components (v)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Wind Speed [m/s]")
    ax2.legend()

    plt.tight_layout()
    plt.show()
    

def plot_wind_speed_components(weather: pd.DataFrame, rolling_window: int = 24 * 7 * 4) -> None:
    u10_south = weather["u10_south"]
    u100_south = weather["u100_south"]
    v10_south = weather["v10_south"]
    v100_south = weather["v100_south"]

    u10_north = weather["u10_north"]
    u100_north = weather["u100_north"]
    v10_north = weather["v10_north"]
    v100_north = weather["v100_north"]

    # setup_tueplots(column="full", n_rows=2, n_cols=1, use_tex=True)
    # Plot the data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    locator1 = ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator1))
    locator2 = ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator2))
    # Plot first subplot - u components
    ax1.plot(
        weather.index,
        u10_north.rolling(window=rolling_window).mean(),
        label=r"u10 North Monthly MA [m/s]",
        color=rgb.tue_blue,
        alpha=1,
    )
    ax1.plot(
        weather.index,
        u100_north.rolling(window=rolling_window).mean(),
        label=r"u100 North Monthly MA [m/s]",
        color=rgb.tue_darkblue,
        alpha=1,
    )
    ax1.plot(
        weather.index,
        u10_south.rolling(window=rolling_window).mean(),
        label=r"u10 South Monthly MA [m/s]",
        color=rgb.tue_orange,
        alpha=1,
    )
    ax1.plot(
        weather.index,
        u100_south.rolling(window=rolling_window).mean(),
        label=r"u100 South Monthly MA [m/s]",
        color=rgb.tue_ocre,
        alpha=1,
    )
    ax1.axhline(
        0, color="black", linestyle="--", linewidth=1, alpha=0.7
    )
    ax1.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.set_title(r"East-West Wind Speed Components (u)")
    ax1.set_ylabel(r"Wind Speed [m/s]")
    ax1.legend()

    # Plot second subplot - v components
    ax2.plot(
        weather.index,
        v10_north.rolling(window=rolling_window).mean(),
        label=r"v10 North Monthly MA [m/s]",
        color=rgb.tue_blue,
        alpha=1,
    )
    ax2.plot(
        weather.index,
        v100_north.rolling(window=rolling_window).mean(),
        label=r"v100 North Monthly MA [m/s]",
        color=rgb.tue_darkblue,
        alpha=1,
    )
    ax2.plot(
        weather.index,
        v10_south.rolling(window=rolling_window).mean(),
        label=r"v10 South Monthly MA [m/s]",
        color=rgb.tue_orange,
        alpha=1,
    )
    ax2.plot(
        weather.index,
        v100_south.rolling(window=rolling_window).mean(),
        label=r"v100 South Monthly MA [m/s]",
        color=rgb.tue_ocre,
        alpha=1,
    )
    ax2.axhline(
        0, color="black", linestyle="--", linewidth=1, alpha=0.7
    )
    ax2.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.set_title(r"North-South Wind Speed Components (v)")
    ax2.set_xlabel(r"Time")
    ax2.set_ylabel(r"Wind Speed [m/s]")
    ax2.legend()
    plt.tight_layout()
    plt.show()