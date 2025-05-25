# ---- This parses the data csvs and assembles them into data frames to be used for building the sliding window
#      train/test set. ----
import os.path
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
import tsfel
from typing import Set, List, Tuple
from timemachine.preprocessing.seasonality import cyclical_calendar_encoding



# specify the paths to the data csvs
cepro_data_file_path = os.path.join(os.path.dirname(__file__), 'market_data.csv')
cepro_types_file_path = os.path.join(os.path.dirname(__file__), 'market_data_types.csv')
bmrs_data_file_path = os.path.join(os.path.dirname(__file__), 'bmrs_elexon.csv')
commodity_data_file_path = os.path.join(os.path.dirname(__file__), 'commodity_data.csv')


def fetch_data_cepro(sample_granularity : str = '1h', clip=(pl.datetime(2024,1,1), pl.datetime(2025,1,1))) -> tuple:
    """
    Parses the Cepro data csv into a time series dataframe and a types dataframe.

    Args:
        sample_granularity (str): The time step size to resample the data to.
        clip (Tuple[datetime, datetime]): Datetime interval of data (inclusive).
    Returns:
        Tuple[pl.DatafFrame, pl.Dataframe]
        The time series data and types table.
    """
    if os.path.exists(cepro_data_file_path):
        print("File exists!")
    else:
        print("File does not exist. Check the path.")

    df = pl.read_csv(cepro_data_file_path)
    types_df = pl.read_csv(cepro_types_file_path)

    # cast the datetime column
    df = df.with_columns(pl.col("time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S.%f %z"))

    df.with_columns(pl.col("type").cast(pl.Int64))
    df.with_columns(pl.col("value").cast(pl.Float32))

    # ensure types are contiguous and sorted by time
    df = df.sort(['type', 'time']).set_sorted('time')

    # separate each type into a column
    df = df.pivot(values='value', index='time', columns='type', aggregate_function='first')
    
    # cast into 32-bit floats
    for type_id in types_df['id'].unique():
        df = df.with_columns(pl.col(f'{type_id}').cast(pl.Float32))

    # cyclical seasonal encodings
    df = cyclical_calendar_encoding(df=df, datetime_col='time', period='year', step='1mo')
    df = cyclical_calendar_encoding(df=df, datetime_col='time', period='week', step='1d')
    df = cyclical_calendar_encoding(df=df, datetime_col='time', period='day', step='1ms')

    # clip date range
    if clip:
        start_date = clip[0]
        end_date = clip[1]
        df = df.filter((pl.col("time") >= start_date) & (pl.col("time") <= end_date))

    # clean up and fill null values
    df = df.sort('time')
    df = df.fill_null(strategy='forward')

    # resample to granularity
    df = df.upsample('time', every=sample_granularity)
    return df, types_df


def fetch_data_bmrs(sample_granularity : str = '1h', clip=(pl.datetime(2024,1,1), pl.datetime(2025,1,1))):
    """
    Parses the Elexon BRMS platform data csv into a dataframe.

    Args:
        clip (Tuple[datetime, datetime]): Datetime interval of data (inclusive).
    Returns:
        pl.DataFrame The time series data.
    """
    df = pl.read_csv(bmrs_data_file_path)

    if 'start_time' in df.columns:  # omit start time col
        df = df.drop('start_time')

    df = df.with_columns(pl.col("time").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%f"))  # cast time col to Datetime
    df = df.select(df.columns).filter(pl.col("time").cast(pl.Datetime).is_between(clip[0], clip[1]))  # clip the date range

    df = df.sort('time')  # clean up
    df = df.fill_null(strategy='forward')
    df = df.upsample('time', every=sample_granularity, maintain_order=True).fill_null(strategy="forward")  # upsample

    return df


def fetch_data_commodities(clip, sample_granularity):
    """
    Parse the gas price csv into a dataframe.

    Args:
        sample_granularity (str): The time step size to resample the data to.
        clip (Tuple[datetime, datetime]): Datetime interval of data (inclusive).
    Returns:
        pl.DataFrame The time series data.
    """
    df = pl.read_csv(commodity_data_file_path)

    df = df.with_columns(pl.col("Date").str.strptime(pl.Datetime, format="%Y-%m-%d"))   # cast time col to Datetime
    df = df.select(df.columns).filter(pl.col("Date").cast(pl.Datetime).is_between(clip[0], clip[1])) # clip the date range

    df = df.upsample('Date', every=sample_granularity, maintain_order=True).fill_null(strategy="forward")
    df = df.rename({'Date': 'time'})

    return df


def fetch_full_dataset(clip, sample_granularity='1h'):
    """
    Returns the combined time series dataset for training the forecasting models, including:
        - Cepro price history
        - Elexon BMRS transparency time series
        - Gas futures prices

    Args:
        sample_granularity (str): The time step size to resample the data to.
        clip (Tuple[datetime, datetime]): Datetime interval of data (inclusive).
    Returns:
        pl.DataFrame The combined time series dataset.
    """
    df, types_df = fetch_data_cepro(sample_granularity=sample_granularity, clip=clip)
    bmrs_df = fetch_data_bmrs(clip=clip)
    commodities_df = fetch_data_commodities(clip=clip, sample_granularity=sample_granularity)
    df = df.join(bmrs_df, on='time', how='inner')
    df = df.join(commodities_df, on='time', how='inner')

    return df


def add_tsfel_features(df, config, columns, window_size):
    """
    Uses TSFEL to add time series transformations defined in the config file
    to the selected columns of df.

    Args:
        df (pl.DataFrame): The time series data frame
        config (str | Path): Path to feature config
        columns (Set[str]): Column name selection to generate features for
    Returns:
        pl.DataFrame, List[str]
        The dataframe with added feature columns and the names of the new columns
    """
    cfg = tsfel.load_json(config)
    enriched_cols = []
    selection = df.select(columns)
    selection = selection.to_pandas()
    if selection.shape[0] > 0:
        overlap = 1 - 1 / window_size
        X = tsfel.time_series_features_extractor(
            config=cfg,
            timeseries=selection,
            window_size=window_size,
            overlap=overlap
        )
        pad_size = window_size - 1
        first_row = X.iloc[0]
        enriched_cols = X.columns.tolist()
        padding = pd.DataFrame([first_row] * pad_size, columns=X.columns)
        X = pd.concat([padding, X], ignore_index=True)
        df = pl.concat([pl.from_pandas(X), df], how='horizontal')
    return df, enriched_cols
