# --- This is for getting futures prices for Dutch TTF natural gas usnig yfinance. ----

import pandas as pd
import datetime
from datetime import timedelta
import yfinance as yf
import joblib
from os import path


def check_coverage(df, start_date, end_date, sampling_rate):
    """
    Check if a pandas DataFrame has complete coverage between two dates at specified sampling rate.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index to check
    start_date : str or datetime-like
        Start date for the coverage check
    end_date : str or datetime-like
        End date for the coverage check
    sampling_rate : str
        Sampling rate as string (e.g., '1min', '1h', '1d')
    
    Returns:
    --------
    bool
        True if complete coverage exists, False otherwise
    dict
        Additional information about the coverage check
    """
    # Ensure the DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    
    # Convert start and end dates to pandas Timestamps if they're strings
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Create an expected date range with the specified sampling rate
    expected_range = pd.date_range(start=start, end=end, freq=sampling_rate)
    
    # Filter the DataFrame to only include rows within the specified date range
    df_filtered = df[(df.index >= start) & (df.index <= end)]
    
    # Check if all expected timestamps exist in the DataFrame index
    missing_timestamps = expected_range.difference(df_filtered.index)
    
    # Calculate coverage statistics
    total_expected = len(expected_range)
    total_actual = len(df_filtered)
    total_missing = len(missing_timestamps)
    coverage_percent = (total_actual / total_expected * 100) if total_expected > 0 else 0
    
    # Check if there's complete coverage
    has_complete_coverage = len(missing_timestamps) == 0
    
    result_info = {
        "has_complete_coverage": has_complete_coverage,
        "expected_points": total_expected,
        "actual_points": total_actual,
        "missing_points": total_missing,
        "coverage_percentage": coverage_percent,
        "sampling_rate": sampling_rate,
        "sampling_frequency": pd.tseries.frequencies.to_offset(sampling_rate),
        "start_date": start,
        "end_date": end
    }
    
    # Add first few missing timestamps if any exist
    if not has_complete_coverage:
        result_info["example_missing_timestamps"] = missing_timestamps[:5].tolist()
    
    return has_complete_coverage, result_info



def fetch_tickers(
        tickers : set,
        start : datetime.datetime,
        end : datetime.datetime,
        interval : str,
        reload=False,
    ):
    """
    Basic YFinance wrapper with caching.
    Usually periods over 60 days will be rejected.
    Always in UTC time.

    Args:
        tickers (set): Set of ticker strings.
        interval (str): Interval string (e.g. '1m, 3h, 1d, 5mo')
        start (datetime.datetime): Start date
        end (datetime.datetime): End date
        reload (bool): Force to clear cache and redownload. 
    
    Returns:
        dict[pd.DataFrame]: Dictionary of dataframes for each ticker.
    """
    cache_name = f"{__file__[:-3]}_cache"
    data = {}
    cached_data = {}

    # use existing cache
    if path.exists(cache_name) and not reload:
        cached_data = joblib.load(cache_name)

        # fill in missing tickers
        missing_tickers = tickers.difference(set(cached_data.keys()))
        if missing_tickers:
            print(f"Fetching {len(missing_tickers)} non-cached tickers.")
            for tck in missing_tickers:
                try:
                    cached_data[tck] = yf.download(tck, start=start, end=end, interval=interval)
                except:
                    print(f"Failed to download {tck} at interval:{interval} over {start}->{end}.")
            joblib.dump(cached_data, cache_name)
        
        # ensure start-end date is met with interval sampling rate
        updated = False
        for tck in tickers.difference(missing_tickers):
            full_coverage, _ = check_coverage(cached_data[tck], start, end, interval)
            if not full_coverage:
                updated = True
                print(f"Updating {tck} coverage...")
                try:
                    df = yf.download(tck, start=start, end=end, interval=interval)
                except:
                    print(f"Failed to download {tck} at interval:{interval} over {start}->{end}.")
                if df is not None:
                    cached_data[tck] = pd.concat(
                        [cached_data[tck], df],
                        axis=0
                    ).loc[~pd.concat([cached_data[tck], df]).index.duplicated(keep='last')]
                    cached_data[tck] = cached_data[tck].sort_index()
                
        if updated:
            joblib.dump(cached_data, cache_name)
    # create new cache
    else:
        print("Making time series cache.")
        for tck in tickers:
            try:
                cached_data[tck] = yf.download(tck, start=start, end=end, interval=interval)
            except:
                print(f"Failed to download {tck} at interval:{interval} over {start}->{end}.")
        joblib.dump(cached_data, cache_name)

    for tck in tickers:
        data[tck] = cached_data[tck].loc[(cached_data[tck].index >= start) & (cached_data[tck].index <= end)]

    return data



def build_commodity_dataset(clip, file):
    commodity_tickers = {'TTF=F', 'NG=F'}


    start_date = clip[0]
    end_date = clip[1]
    period_delta = timedelta(days=7)
    step_start = start_date

    df_parts = []

    while step_start < end_date:
        dfs = fetch_tickers(
            tickers=commodity_tickers,
            interval='2m',
            start=step_start,
            end=step_start+period_delta,
            reload=True
        )

        dfs = [dfs[tck] for tck in dfs]
        # flatten the multi index columns
        for df in dfs:
            df.columns = ['_'.join(reversed(col)).strip() for col in df.columns.values]
        part = dfs[0]
        if len(dfs) > 1:
            for df in dfs[1:]:
                part = part.join(df, how='left')

        part = part.rename({'Datetime': 'time'})
        part = part.loc[(part.index >= step_start) & (part.index <= (step_start + period_delta))]
        df_parts.append(part)

        step_start += period_delta
    
    df = pd.concat(df_parts, axis=0)
    df.to_csv(file)

    df = df.ffill()
    return df


if __name__ == '__main__':
    build_commodity_dataset(
        clip=(datetime.datetime(2024,1,1), datetime.datetime(2025,1,1)),
        file='commodity_data.csv'    
    )