import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button
import datetime as dt
import dataset.data as data
import tsfel
import timemachine.preprocessing.rolling_features as rf

import matplotlib.ticker as ticker
import matplotlib.dates as mdates


TARGET_VARIABLE = '9'

df = data.fetch_full_dataset(clip=(pl.datetime(2024,1,1), pl.datetime(2025,1,1)), sample_granularity='1h')
columns = df.columns

window = 12
feat_df = pl.DataFrame()
# feat_df = feat_df.with_columns(
#     pl.Series(
#         rf.rolling_std(df[TARGET_VARIABLE].to_numpy(), window=24*7))
#         .alias(f'{TARGET_VARIABLE}_std'))
# feat_df = feat_df.with_columns(
#     pl.Series(
#         rf.rolling_hurst(df[TARGET_VARIABLE].to_numpy(), window=24*7))
#         .alias(f'{TARGET_VARIABLE}_slope'))


enrichment_cols = [
    '9',
]

cfg = tsfel.load_json('feature_set_statistical.json')
feat = df[enrichment_cols].to_pandas()
window_size = 24 * 7
X = tsfel.time_series_features_extractor(cfg, feat, window_size=window_size, overlap=1-1/(window_size))
pad_size = window_size - 1
first_row = X.iloc[0]
padding = pd.DataFrame([first_row] * pad_size, columns=X.columns)
X = pd.concat([padding, X], ignore_index=True)
enriched_cols = X.columns.tolist()
df = pl.concat([pl.from_pandas(X), df], how='horizontal')

# plt.style.use('classic')
fig, ax1 = plt.subplots(1, 1, figsize=(5, 2.5))
# ax2 = ax1.twinx()
plt.xticks(rotation=45)
plt.grid(True, which="both", axis="x", linestyle="--", linewidth=0.5)

feat_idx = 0

def update_plot():
    ax1.clear()
    # ax2.clear()

    # ax1.plot(df['time'], df[TARGET_VARIABLE], lw=.5, marker='x', markersize=1, label=TARGET_VARIABLE)
    for ecol in enriched_cols:
        ax1.plot(df['time'], df[ecol], lw=1, label=df.columns[feat_idx])

    ax1.grid(True, lw=.25)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))

    plt.tight_layout()
    # plt.legend()
    plt.draw()

def on_key(event):
    global feat_idx
    if event.key == 'right':
        feat_idx += 1
    elif event.key == 'left':
        feat_idx -= 1
    update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)
update_plot()
plt.show()