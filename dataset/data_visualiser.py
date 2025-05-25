# --- This is for visualising the time series dataset ---
# USAGE: Left and right arrows to switch between columns.

import polars as pl
import matplotlib.pyplot as plt
import datetime as dt
import data

import matplotlib.ticker as ticker
import matplotlib.dates as mdates

df = data.fetch_full_dataset(clip=(pl.datetime(2022,1,1), pl.datetime(2025,1,1)), sample_granularity='30m')

columns = df.columns

min_date = dt.datetime(2022,1,1)
max_date = dt.datetime(2025,1,1)


plt.style.use('classic')
fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
ax2 = ax1.twinx()
ax3 = ax1.twinx()
plt.xticks(rotation=45)
plt.grid(True, which="both", axis="x", linestyle="--", linewidth=0.5)

type_idx = 0

def update_plot():
    ax1.clear()

    view_col = columns[type_idx]

    ax1.plot(df['time'], df[view_col], lw=.5, marker='x', markersize=1)
    ax1.set_title(f"{type_idx}: {view_col}")
    ax1.grid(True, lw=.25)
    ax1.set_xlim([min_date, max_date])
    ax2.set_ylim([-1, 1])
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))

    plt.tight_layout()
    plt.draw()

def on_key(event):
    global type_idx
    if event.key == 'right':
        type_idx += 1
    elif event.key == 'left':
        type_idx -= 1
    update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)
update_plot()
plt.legend()
plt.show()