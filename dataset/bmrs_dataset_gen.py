# ---- This script is for building time series datasets from the Elexon Insights transparency data API. ----

from datetime import datetime
from datetime import timedelta
import pandas as pd
import polars as pl
from elexonpy.api_client import ApiClient
from elexonpy.api.demand_forecast_api import DemandForecastApi
from elexonpy.api.generation_forecast_api import GenerationForecastApi
from elexonpy.api.generation_api import GenerationApi
from elexonpy.api.temperature_api import TemperatureApi
from elexonpy.api.system_forecast_api import SystemForecastApi
from elexonpy.api.system_api import SystemApi

from_date = datetime(2022, 1, 1)
to_date = datetime(2023, 1, 1)

api_client = ApiClient()
demand_forecast_api = DemandForecastApi(api_client)
forecast_api = GenerationForecastApi(api_client)
temperature_api = TemperatureApi(api_client) # not working
system_forecast_api = SystemForecastApi(api_client)
system_api = SystemApi(api_client)
generation_api = GenerationApi(api_client)


# The API limits requests to 7 days of data, so we need to download and process each week of the desired interval.
DATE_STEP = timedelta(days=7)
step_start = from_date
step_end = step_start + DATE_STEP

time_base = pl.datetime_range(
    start=from_date,
    end=to_date,
    interval="30m",
    eager=True
).to_frame(name="time")

wind_solar_dfs = []
agg_gen_dfs = []
demand_dfs = []
demand_forecast_dfs = []
system_dfs = []
generation_type_dfs = []
# temp_df = temperature_api.temperature_get(_from=from_date, to=to_date, format='dataframe')

while step_end <= to_date:
    # ---- DAY-AHEAD WIND & SOLAR FORECASTS ----
    print(f'{step_start} -> {step_end}')
    ws_df = forecast_api.forecast_generation_wind_and_solar_day_ahead_get(
        _from=step_start,
        to=step_end,
        process_type='day ahead',
        format='dataframe'
    )
    ws_df = pl.from_pandas(ws_df)
    ws_df = (ws_df
        .select(['start_time', 'quantity', 'business_type'])
        .with_columns(pl.col("start_time").cast(pl.Datetime("us")).alias("time"))
        .filter(pl.col('time').is_between(step_start, step_end))
        .pivot(
            values='quantity',
            on='business_type',
            index='time',
            aggregate_function='first'
        ).select([
            'time',
            pl.col('Solar generation').alias('solar_forecast').fill_null(0),
            pl.col('Wind generation').alias('wind_forecast').fill_null(0)
        ])
    )
    wind_solar_dfs.append(ws_df)
    ...
    # s_df = system_api.system_frequency_get(
    #     _from=step_start,
    #     to=step_end,
    #     format='dataframe'
    # )
    # s_df = pl.from_pandas(s_df)
    # s_df = (s_df
    #     .select(['measurement_time', 'frequency'])
    #     .with_columns(pl.col("measurement_time").cast(pl.Datetime("us")).alias("time"))
    #     .filter(pl.col('time').is_between(step_start, step_end))
    # )
    # system_dfs.append(s_df)
    # sf_df = system_forecast_api.forecast_system_loss_of_load_get(
    #     _from=step_start,
    #     to=step_end,
    #     # process_type='day ahead',
    #     format='dataframe'
    # )
    # sf_df = pl.from_pandas(ws_df)
    # sf_df = (sf_df
    #          .select(['start_time', 'forecast_horizon', 'derated_margin'])
    #         .with_columns(pl.col("start_time").cast(pl.Datetime("us")).alias("time"))
    #         .filter(pl.col('time').is_between(step_start, step_end) & pl.col('forecast_horizon'))
    # )
    # print(sf_df.head(n=50))
    # ---------------------------------
    g_df = generation_api.generation_actual_per_type_get(
        _from=step_start,
        to=step_end,
        format='dataframe'
    )
    g_df = pl.from_pandas(g_df)
    g_df = (g_df
        .select(['start_time', 'data'])
        .with_columns(pl.col("start_time").cast(pl.Datetime("us")).alias("time"))
        .filter(pl.col('time').is_between(step_start, step_end))
    )
    # tricky nested list of structs here
    g_df_exploded = g_df.explode('data')
    g_df_typed = g_df_exploded.select([
        pl.col("*").exclude("data"),
        pl.col("data").struct.field("psr_type").alias("psr_type"),
        pl.col("data").struct.field("quantity").alias("quantity")
    ])
    g_df = g_df_typed.pivot(
        index=[col for col in g_df.columns if col != "data"],  # All original columns except 'data'
        columns="psr_type",
        values="quantity"
    )
    generation_type_dfs.append(g_df)
    step_start += DATE_STEP
    step_end += DATE_STEP

    # ---- DAY-AHEAD AGGREGATED GENERATION FORECAST (DAG/B1430) ----
    # ag_df = forecast_api.forecast_generation_day_ahead_get(
    #     _from=step_start,
    #     to=step_end,
    #     format='dataframe'
    # )
    # ag_df = pl.from_pandas(ag_df)
    # ag_df = (ag_df
    #     .select(['start_time', 'quantity'])
    #     .with_columns(pl.col("start_time").cast(pl.Datetime("us")).alias("time"))
    #     .filter(pl.col('time').is_between(step_start, step_end))
    #     .rename({'quantity': 'agg_gen_forecast'})
    #     .select(['time', 'agg_gen_forecast'])
    # )
    # agg_gen_dfs.append(ag_df)
    # --------------------------------------------------------

DATE_STEP = timedelta(days=1)
step_start = from_date
step_end = step_start + DATE_STEP
while step_end <= to_date:
    print(f'Getting day-ahead demand forecasts from {step_start} -> {step_end}')
    d_df = demand_forecast_api.forecast_demand_day_ahead_history_get(
        publish_time=step_start,
        format='dataframe'
    )
    d_df = pl.from_pandas(d_df)
    d_df = (d_df
        .select(['start_time', 'transmission_system_demand', 'national_demand'])
        .with_columns(pl.col("start_time").cast(pl.Datetime("us")).alias("time"))
        .filter(pl.col('time').is_between(step_start, step_end))
        .select(['time', 'transmission_system_demand', 'national_demand'])
        .select(
            pl.col("time").cast(pl.Datetime),
            pl.col("transmission_system_demand").cast(pl.Float32),
            pl.col("national_demand").cast(pl.Float32),
        )
    )
    demand_dfs.append(d_df)
    # -------------------------

    step_start += DATE_STEP
    step_end += DATE_STEP

# Combine all chunks for each data type
wind_solar_full = pl.concat(wind_solar_dfs, how='diagonal')
# agg_gen_full = pl.concat(agg_gen_dfs, how='diagonal')
demand_full = pl.concat(demand_dfs, how='diagonal')
# system_full = pl.concat(system_dfs, how='diagonal')
generation_type_full = pl.concat(generation_type_dfs, how='diagonal')
demand_forecast_full = pl.concat(demand_forecast_dfs, how='diagonal')

# Merge everything onto the time base
dataset_df = (time_base
    .join(wind_solar_full, on='time', how='left')
    # .join(agg_gen_full, on='time', how='left')
    .join(demand_full, on='time', how='left')
    .join(generation_type_full, on='time', how='left')
    .join(demand_forecast_full, on='time', how='left')
    # .join(system_full, on='time', how='left')
    .sort('time')
    .fill_null(strategy='forward')
    .unique(subset=['time'], keep='first')
)

dataset_df.write_csv('bmrs_elexon_new.csv')