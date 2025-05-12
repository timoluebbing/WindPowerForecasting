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
    df[Column.WIND.value] = df[Column.WIND_OFFSHORE.value] + df[Column.WIND_ONSHORE.value]

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