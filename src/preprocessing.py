import pandas as pd
import numpy as np

from enum import Enum

from sklearn.cluster import KMeans


class Column(Enum):
    DATE_FROM = "Date from"
    DATE_TO = "Date to"
    WIND_OFFSHORE = "Wind Offshore [MW] "
    WIND_ONSHORE = "Wind Onshore [MW]"
    WIND = "Wind Sum [MW]"


def preprocess_supply_data(df: pd.DataFrame, resample: str = "h") -> pd.DataFrame:
    # convert date columns to datetime
    df[Column.DATE_FROM.value] = pd.to_datetime(
        df[Column.DATE_FROM.value], format="%d.%m.%y %H:%M"
    )
    df[Column.DATE_TO.value] = pd.to_datetime(
        df[Column.DATE_TO.value], format="%d.%m.%y %H:%M"
    )
    df.set_index(Column.DATE_FROM.value, inplace=True)
    df.sort_index(inplace=True)

    # Columns to exclude from numeric conversion
    non_numeric_cols = [Column.DATE_FROM.value, Column.DATE_TO.value]

    # Convert relevant columns to numeric
    for col in df.columns:
        if col not in non_numeric_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")  # error handling --> NaN

    # 15 minutes intervals summed up to 1 hour intervals
    df = df.resample(resample).sum(numeric_only=True)

    # Sum up wind offshore and onshore
    df[Column.WIND.value] = (
        df[Column.WIND_OFFSHORE.value] + df[Column.WIND_ONSHORE.value]
    )

    return df


def weather_germany_mean(weather: pd.DataFrame) -> pd.DataFrame:
    numeric_weather_cols = weather.select_dtypes(include=np.number).columns.tolist()

    # Remove lat/lon from mean calculation --> mean weather data in Germany
    numeric_weather_cols = [
        col for col in numeric_weather_cols if col not in ["latitude", "longitude"]
    ]

    weather_aggregated_mean = (
        weather.groupby("time")[numeric_weather_cols].mean().fillna(0)
    )

    return weather_aggregated_mean

def weather_north_south_means(
    weather: pd.DataFrame, lat_threshold: float
) -> pd.DataFrame:
    """
    Split weather data into northern and southern Germany and compute means for
    all numerical weather features.
    
    Args:
        weather (pd.DataFrame): Weather data with latitude and longitude columns.
    Returns:
        pd.DataFrame: Mean weather data for northern and southern Germany.
    """
    weather = weather.reset_index()  # ensure 'time' column exists for grouping
    numeric_weather_cols = weather.select_dtypes(include=np.number).columns.tolist()
    numeric_weather_cols = [
        col for col in numeric_weather_cols if col not in ["latitude", "longitude"]
    ]
    
    weather["is_north"] = weather["latitude"] >= lat_threshold
    weather_split_means = (
        weather.groupby(["is_north", "time"])[numeric_weather_cols].mean().fillna(0)
    )
    weather = weather_split_means.unstack(level="is_north")
    weather.columns = [
        f"{col}_{'north' if is_north else 'south'}" for col, is_north in weather.columns
    ]
    
    return weather


def weather_germany_clustered(weather: pd.DataFrame, k_clusters: int) -> pd.DataFrame:
    """
    Cluster weather data based on latitude and longitude.
    Args:
        weather (pd.DataFrame): Weather data with latitude and longitude columns.
        k_clusters (int): Number of clusters to form.
    Returns:
        pd.DataFrame: Weather data with cluster labels.
    """
    kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init="auto")

    # Get unique lat/lon combinations and cluster them
    locations = weather[["latitude", "longitude"]].drop_duplicates()
    clusters = kmeans.fit_predict(locations[["latitude", "longitude"]])
    locations["cluster"] = clusters

    # Merge clusters back while preserving the index
    weather_clustered = weather.reset_index()
    weather_clustered = weather_clustered.merge(
        locations, on=["latitude", "longitude"], how="left"
    )
    weather_clustered.set_index("time", inplace=True)

    numeric_weather_cols = weather.select_dtypes(include=np.number).columns.tolist()
    numeric_weather_cols = [
        col for col in numeric_weather_cols if col not in ["latitude", "longitude"]
    ]
    weather_clustered_aggregated = (
        weather_clustered.groupby(["time", "cluster"])[numeric_weather_cols]
        .mean()
        .fillna(0)
    )
    weather = weather_clustered_aggregated.unstack(level="cluster")
    weather.columns = weather.columns.map(lambda x: f"{x[0]}_cluster_{x[1]}")

    return weather


def create_time_features(df: pd.DataFrame):
    """
    Create time-based features from the datetime index of the dataframe.
    Applying sine and cosine transformations for cyclical time features
    already scales the features for further processing.

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        DataFrame with time features added
    """
    df_with_features = df.copy()

    # Hour of day not needed for the given task and hourly lags included
    # df_with_features["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    # df_with_features["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    df_with_features["dayofweek_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df_with_features["dayofweek_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    df_with_features["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df_with_features["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    df_with_features["dayofyear_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df_with_features["dayofyear_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365)

    return df_with_features


def create_sliding_window_data(
    data: pd.DataFrame,
    history: int,
    forecast_horizon: int,
    target_column: str = Column.WIND.value,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create sliding window data for time series forecasting, returning pandas DataFrames.

    Args:
        data: DataFrame with a DatetimeIndex and columns for the target and other features.
        history: Number of past time steps of the target_column to use as input lags.
        forecast_horizon: Number of future time steps to predict for the target_column.
        target_column: Column name of the target variable.

    Returns:
        X_df: pd.DataFrame, indexed by the timestamp of the last observation in the
              history window. Columns include lagged target values (e.g., 'lag_1', 'lag_2')
              and current values of other feature columns.
        y_df: pd.DataFrame, indexed similarly to X_df. Columns include future target
              values (e.g., 't+1', 't+2').
    """
    other_feature_columns = [col for col in data.columns if col != target_column]

    X_data_list = []
    y_data_list = []
    index_list = []

    n_total_timesteps = len(data)
    max_start_idx = n_total_timesteps - history - forecast_horizon + 1

    for start in range(max_start_idx):
        history_end_idx = start + history
        forecast_end_idx = history_end_idx + forecast_horizon

        current_sample_timestamp = data.index[history_end_idx - 1]
        index_list.append(current_sample_timestamp)

        x_sample_dict = {}

        # Lagged target features: target_values from t-history+1, ..., t
        # lag_1 --> data[target_column].iloc[history_end_idx - 1] (most recent)
        # lag_history --> data[target_column].iloc[start] (oldest in window)
        target_window_values = data[target_column].iloc[start:history_end_idx].values
        for h in range(history):
            x_sample_dict[f"lag_{history - h}"] = target_window_values[h]

        # Current other features (e.g., weather, time features) at time t
        if other_feature_columns:
            current_other_features = data[other_feature_columns].iloc[
                history_end_idx - 1
            ]
            x_sample_dict.update(current_other_features.to_dict())

        X_data_list.append(x_sample_dict)

        # Future target values: target_values from t+1, ..., t+forecast_horizon
        future_target_values = (
            data[target_column].iloc[history_end_idx:forecast_end_idx].values
        )
        y_sample_dict = {
            f"t+{h+1}": future_target_values[h] for h in range(forecast_horizon)
        }
        y_data_list.append(y_sample_dict)

    X_df = pd.DataFrame(X_data_list, index=pd.Index(index_list, name="timestamp"))
    y_df = pd.DataFrame(y_data_list, index=pd.Index(index_list, name="timestamp"))

    return X_df, y_df


def create_sliding_window_data_numpy(
    data: pd.DataFrame,
    history: int,
    forecast_horizon: int,
    target_col: str = Column.WIND.value,
):
    """
    Create multivariate sliding-window input/output for time series forecasting.

    Args:
        data: DataFrame with time-index and columns = [weather features ..., target]
        history: Number of past time steps to use as input
        forecast_horizon: Number of future time steps to predict
        target_col: Name of the target column

    Returns:
        X: np.ndarray, shape (samples, history, n_features) where n_features = lagged target + current weather
        y: np.ndarray, shape (samples, forecast_horizon) for the target values
    """
    # Weather features (all columns except the target)
    feature_cols = [c for c in data.columns if c != target_col]

    X_windows = []
    y_windows = []

    n_total = len(data)
    max_start = n_total - history - forecast_horizon + 1

    for start in range(max_start):
        end_history = start + history

        # Lagged target values for past 'history' steps
        target_lags = (
            data[target_col].iloc[start:end_history].values
        )  # shape = (history,)

        # Weather features at the current time step (aligned with the end of the history window)
        current_features = (
            data[feature_cols].iloc[end_history - 1].values
        )  # shape = (n_features,)

        # Concatenate lagged target and current weather features
        window_X = np.concatenate(
            [target_lags, current_features]
        )  # shape = (history + n_features,)

        # Future target values for forecast horizon
        window_y = (
            data[target_col].iloc[end_history : end_history + forecast_horizon].values
        )

        X_windows.append(window_X)
        y_windows.append(window_y)

    X = np.array(X_windows)  # shape = (samples, history + n_features)
    y = np.array(y_windows)  # shape = (samples, forecast_horizon)

    return X, y
