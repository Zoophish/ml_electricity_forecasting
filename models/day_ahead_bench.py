import polars as pl
from datetime import timedelta, datetime
import random

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from tslearn.metrics import SoftDTWLossPyTorch 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from timemachine.models.stateful_lstm import StatefulLSTM
from timemachine.models.cnn_lstm import CNNLSTM
from timemachine.models.mlp import HourglassMLP

from timemachine.training.early_stopper import EarlyStopper
from timemachine.preprocessing.sliding_window import SlidingWindowDataset, transform_batch
from timemachine.training.loss import MAPELoss, MAELoss

from dataset import data



if __name__ == '__main__':

    # ---- SCRIPT OPTIONS ----
    MODE = 'Train'  # either train some new parameters or 'Load' a parameter file
    PARAM_CACHE_LOAD_PATH = 'params.mdl'  # load path
    PARAM_CACHE_SAVE_PATH = 'params.mdl'  # save path

    INTERACTIVE_PLOT = True  # show interactive plot for day ahead forecasts

    DUMP_RESULTS = False  # dump forecast and errors for each day in dataset
    RESULTS_DUMP_NAME = "results"
    PLOT_FIGURE = False  # figure for the report


    # seed for psuedo-random processes, makes results repeatable
    random.seed(10108)


    # ---- HYPERPARAMETERS ----
    TRAINING_RATIO = 0.8
    EPOCHS = 200  # maximum number of epochs

    # architecture-specific parameters tha generally work well
    # best params from the optimisation are available in the report results
    # lstm parameters
    HIDDEN_SIZE_PER_INPUT = 8
    HIDDEN_LAYERS = 1
    BIDIRECTIONAL = True
    POOL = 'max'
    # cnnlstm parameters
    CONV1_FILTERS = 47
    CONV2_FILTERS = 64
    CONV_KERNEL_SIZE = 3
    
    DROPOUT = 0.289

    # training parameters
    BATCH_SIZE = 204
    L2_LAMBDA = .668
    GRAD_CLIP_THRESHOLD = 1.151
    LEARNING_RATE = 0.00824

    # early stopping
    PATIENCE = 30  # number of epochs to stop if there is no significant change
    MIN_DELTA = .25

    LOOKBACK_HOURS = 24  # lookback window size



    # prepare data
    clip = ( pl.datetime(2022,1,1), pl.datetime(2025,1,1) )
    df = data.fetch_full_dataset(clip=clip, sample_granularity='1h')
    print(df.head())

    # selection of variables (columns) to use as model inputs
    FEATURE_COLS = [
        '9',
        '4','5','8','10', 'solar_forecast', 'wind_forecast',
        'transmission_system_demand', 'national_demand',
        'year_sin', 'year_cos',
        'week_sin', 'week_cos',
        'day_sin', 'day_cos',
        'Biomass', 'Fossil Gas', 'Hydro Pumped Storage', 'Hydro Run-of-river and poundage',
        'Nuclear', 'Other', 'Solar', 'Wind Offshore', 'Wind Onshore',
        'NG=F_Close', 'NG=F_Volume'
    ]
    # selection of variables to forecast
    TARGET_COLS = ['9',]

    # time series feature 'enrichment' using the TSFEL library
    enrichment_cols = []
    feature_config = './features/feature_set_temporal.json'
    df, new_cols = data.add_tsfel_features(
        df=df,
        config=feature_config,
        columns=enrichment_cols,
        window_size=24*7
    )
    FEATURE_COLS += new_cols  # add the new features


    INPUT_SIZE = len(FEATURE_COLS)
    HIDDEN_SIZE = HIDDEN_SIZE_PER_INPUT * INPUT_SIZE
    OUTPUT_SIZE = 24  # 24 hours ahead

    # make a training example for every unique day
    unique_dates = df["time"].dt.date().unique().sort()
    unique_dates = unique_dates.cast(pl.Datetime())

    # isolate the train and test set from each other
    min_df_dt = df['time'].min()
    max_df_dt = df['time'].max()
    partition_dt = min_df_dt + (max_df_dt - min_df_dt) * TRAINING_RATIO
    train_df = df.filter(pl.col('time') < partition_dt)
    test_df = df.filter(pl.col('time') >= partition_dt)

    # create the scalers
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()
    input_scaler.fit(train_df[FEATURE_COLS].to_numpy())
    output_scaler.fit(train_df[TARGET_COLS].to_numpy())

    # set up the sliding window datasets & loaders
    train_dataset = SlidingWindowDataset(
        df=train_df,
        timestamp_col='time',
        lookback=timedelta(hours=LOOKBACK_HOURS),
        horizon=timedelta(hours=24),
        stride=timedelta(hours=24),
        lookback_steps=LOOKBACK_HOURS,
        horizon_steps=24,
        lookback_cols=FEATURE_COLS,
        target_cols=TARGET_COLS
    )
    test_dataset = SlidingWindowDataset(
        df=test_df,
        timestamp_col='time',
        lookback=timedelta(hours=LOOKBACK_HOURS),
        horizon=timedelta(hours=24),
        stride=timedelta(hours=24),
        lookback_steps=LOOKBACK_HOURS,
        horizon_steps=24,
        lookback_cols=FEATURE_COLS,
        target_cols=TARGET_COLS
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # choose the model here
    model = StatefulLSTM(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_size=HIDDEN_SIZE_PER_INPUT * INPUT_SIZE,
        layers=1,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        pool=POOL
    )
    # model = CNNLSTM(
    #     input_size=INPUT_SIZE,
    #     output_size=OUTPUT_SIZE,
    #     hidden_size=16 * INPUT_SIZE,
    #     lstm_hidden_layers=1,
    #     bidirectional=False,
    #     dropout=DROPOUT,
    #     output_depth=0,
    #     output_size_decay=0.5,
    #     conv1_filters=47,
    #     conv2_filters=64,
    #     conv_kernel_size=3,
    #     use_pooling=False,
    #     pool_kernel_size=2,
    #     pool_stride=2
    # )
    # model = HourglassMLP(
    #     input_size=INPUT_SIZE * LOOKBACK_HOURS,
    #     output_size=OUTPUT_SIZE,
    #     hidden_size_fac=5,
    #     hidden_layers=12,
    #     dropout=DROPOUT,
    # )

    print(model)

    if MODE == 'Train':
        # loss function
        train_criterion = torch.nn.MSELoss()

        # other error metrics
        mse_criterion = torch.nn.MSELoss()
        mae_criterion = MAELoss()
        mape_criterion = MAPELoss()

        # optimiser
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=L2_LAMBDA,
            amsgrad=True
        )

        # reduces learning rate when the loss plateaus
        scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, min_lr=1e-6)
        # stops training if test loss doesn't derease by some threshold after patience epochs
        early_stopper = EarlyStopper(patience=PATIENCE, min_delta=MIN_DELTA)

        try:
            writer = SummaryWriter()  # lets us view the training process in TensorBoard (VSCode supports this)

            # keep track of best model state
            best_test_mse = float('inf')
            best_test_mae = float('inf')
            best_test_mape = float('inf')

            # an epoch is a single pass over the entire (training) dataset
            for epoch in range(EPOCHS):
                model.train()  # enables automatic gradients in PyTorch for backprop
                optimiser.zero_grad()  # reset the optimiser's parameter gradients
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    # scale the batches
                    X_batch = transform_batch(X_batch, input_scaler.transform)
                    y_batch = transform_batch(y_batch, output_scaler.transform)

                    # forward pass / prediction
                    predictions = model(X_batch)
                    # compute training loss (the mean averages the batch)
                    loss = train_criterion(predictions, y_batch).mean()

                    # backprop/parameter adjustment happens here:
                    optimiser.zero_grad()  # clear out the old gradients
                    loss.backward()  # compute new gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_THRESHOLD)  # clip steep gradients
                    optimiser.step()  # nudge the parameters according to gradients
                    train_loss += loss.item()

                train_loss /= len(train_loader)
                # test set metrics
                test_loss = 0  # usually same as train loss
                test_mape = 0  # other metrics...
                test_mae = 0
                test_mse = 0

                model.eval()  # set torch modules to evaluation mode (affects things like dropout)
                with torch.no_grad():  # disable automatic gradients since backprop isn't used here (reduces memory)
                    for X_batch, y_batch in test_loader:
                        # scaled batches
                        X_batch_scaled = transform_batch(X_batch, input_scaler.transform)
                        y_batch_scaled = transform_batch(y_batch, output_scaler.transform)

                        predictions = model(X_batch_scaled)
                        test_loss += train_criterion(predictions, y_batch_scaled).mean().item()

                        # the test loss metrics are performed in the original data scale
                        predictions = transform_batch(predictions, output_scaler.inverse_transform)
                        test_mape += mape_criterion(predictions, y_batch).item()
                        test_mae += mae_criterion(predictions, y_batch).item()
                        test_mse += mse_criterion(predictions, y_batch).item()
                    
                    # get mean test errors
                    test_loss /= len(test_loader); test_mape /= len(test_loader); test_mae /= len(test_loader); test_mse /= len(test_loader)

                scheduler.step(test_mape)  # provide error to the scheduler
                
                # display epoch information
                print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Test MSE: {test_mse:.8f}, Test MAPE: {test_mape:.2f} Test MAE: {test_mae:.4f}")

                # keep track of best test errors
                if test_mse < best_test_mse:
                    best_test_mse = test_mse
                    best_model_state = model.state_dict()  # cache the best model parameters
                if test_mae < best_test_mae:
                    best_test_mae = test_mae
                if test_mape < best_test_mape:
                    best_test_mape = test_mape
                if early_stopper.early_stop(test_mae):
                    print(f"Stopping early at patience={early_stopper.counter} epochs.")
                    break

        # allow for manual intervention (CTRL+C) during training loop
        except KeyboardInterrupt:
            print("Training interrupted.")
        
        finally:
            writer.close()  # stop streaming to TensorBoard

            model.load_state_dict(best_model_state)  # load the best parameters
            torch.save(model.state_dict(), PARAM_CACHE_SAVE_PATH)  # save the parameters
            print(f"Best MAPE: {best_test_mape:.2f}% Best MAE: {best_test_mae:.4f} Best MSE: {best_test_mse:.8f}")
    
    # load the model prameters from a file
    elif MODE == 'Load':
        model.load_state_dict(torch.load(PARAM_CACHE_LOAD_PATH))
        model.eval()
    

    # ---- everthing below is for plotting / benchmarking ----
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import matplotlib.ticker as ticker
    import matplotlib.dates as mdates
    import os

    if INTERACTIVE_PLOT:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        plt.xticks(rotation=45)
        plt.grid(True, which="both", axis="x", linestyle="--", linewidth=0.5)

        day_idx = 0
        view_date = datetime(2022, 1, 1) + timedelta(days=365*3*TRAINING_RATIO)

        def update_plot():
            # get historical window and target day
            lookback_df = df.filter(
                (pl.col("time").dt.replace_time_zone(None) >= (view_date - timedelta(hours=LOOKBACK_HOURS))) &
                (pl.col("time").dt.date() < view_date)
            )
            target_day_df = df.filter(
                pl.col("time").dt.date() == view_date
            )

            X = torch.tensor(lookback_df[FEATURE_COLS].to_numpy(), dtype=torch.float32).unsqueeze(0)
            X = transform_batch(X, input_scaler.transform)

            model.eval()
            with torch.no_grad():
                predictions = model(X)
                predictions = transform_batch(predictions, output_scaler.inverse_transform)

            ax1.clear()
            ax1.set_title(f"{view_date}", fontsize=10)
            ax1.plot(target_day_df['time'], predictions[0, :, 0], lw=.7, color="black", markersize=1, label='Forecast', linestyle='--')
            ax1.plot(target_day_df['time'], target_day_df[TARGET_COLS], lw=.7, color="black", markersize=1, label='Target')
            
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

            ax1.tick_params(axis="x", rotation=0, length=3, labelsize=8)
            ax1.tick_params(axis="y", labelsize=8)

            ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax1.tick_params(which="both", direction="in", length=3)

            ax1.grid(True, lw=.25)
            ax1.set_xlabel("Date Time", fontsize=8)
            ax1.set_ylabel(r"Price (GBP MWh$^{-1}$)", fontsize=8)
            ax1.legend(frameon=False, loc="center left", fontsize=8, bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.draw()

        def on_key(event):
            global view_date
            if event.key == 'right':
                view_date = view_date + timedelta(days=1)
            elif event.key == 'left':
                view_date = view_date - timedelta(days=1)
            update_plot()

        fig.canvas.mpl_connect('key_press_event', on_key)
        update_plot()
        plt.show()



    # test final model and dump results for Diebold-Mariano test
    if DUMP_RESULTS:
        import joblib

        mape_criterion = MAPELoss()  # error metrics...
        mae_criterion = MAELoss()
        mse_criterion = torch.nn.MSELoss()
        sft_dtw_criterion = SoftDTWLossPyTorch(gamma=1, normalize=True)

        start_date = datetime(2022, 1, 1)
        end_date = datetime(2025, 1, 1)

        test_date = start_date
        test_results = {}
        with torch.no_grad():
            while test_date < (end_date - timedelta(days=1)):  # iterate over each day
                test_date += timedelta(days=1)
                if (test_date - timedelta(hours=LOOKBACK_HOURS) < datetime(2022, 1, 1)):
                    continue
                
                # get historical window and target day
                lookback_df = df.filter(
                    (pl.col("time").dt.replace_time_zone(None) >= (test_date - timedelta(hours=LOOKBACK_HOURS))) &
                    (pl.col("time").dt.date() < test_date)
                )
                target_day_df = df.filter(
                    pl.col("time").dt.date() == test_date
                )

                # ensure windows are the right size
                if lookback_df.height != LOOKBACK_HOURS or target_day_df.height != 24:
                    continue

                # create input 'batch' with one item, hence just unsqueeze the first dimension
                X = torch.tensor(lookback_df[FEATURE_COLS].to_numpy(), dtype=torch.float32).unsqueeze(0)
                X = transform_batch(X, input_scaler.transform)
                y = torch.tensor(target_day_df[TARGET_COLS].to_numpy(), dtype=torch.float32).unsqueeze(0)
                
                # forecast and scale back
                predictions = model(X)
                predictions = transform_batch(predictions, output_scaler.inverse_transform)
                
                # compute errors
                test_mape = mape_criterion(predictions, y).item()
                test_mae = mae_criterion(predictions, y).item()
                test_mse = mse_criterion(predictions, y).item()

                # save information related to the current date
                test_results[test_date] = {
                    "predictions": predictions[0, :].numpy(),
                    "target": y[0, :].numpy(),
                    "mape": test_mape,
                    "mae": test_mae,
                    "mse": test_mse,
                    "sft_dtw": abs(sft_dtw_criterion(predictions, y).item())
                }
        # dump to a file
        joblib.dump(test_results, RESULTS_DUMP_NAME + ".pkl")

        # this was used to plot figures for the report
        if PLOT_FIGURE:
            def sample_dates(start_date, end_date, n, seed=42):
                """Randomly sample n unique dates between start_date and end_date (inclusive)."""
                random.seed(seed)
                delta = (end_date - start_date).days
                all_days = [start_date + timedelta(days=i) for i in range(delta + 1)]
                return random.sample(all_days, n)

            start = datetime(2023, 1, 1)
            end = datetime(2025, 12, 27)
            n_samples = 9

            # get the dates with the highest MSE
            highest_mse = sorted(test_results.items(), key=lambda x: x[1]['mse'], reverse=True)[:n_samples]
            highest_mae = sorted(test_results.items(), key=lambda x: x[1]['mae'], reverse=True)[:n_samples]
            highest_mape = sorted(test_results.items(), key=lambda x: x[1]['mape'], reverse=True)[:n_samples]
            worst_soft_dtw = sorted(test_results.items(), key=lambda x: x[1]['sft_dtw'], reverse=True)[:n_samples]
            # dates_to_plot = [highest_mape[i][0] for i in range(n_samples)]
            dates_to_plot = sample_dates(start, end, n_samples)

            ncols = 3
            nrows = (len(dates_to_plot) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.4, 1.0 * nrows), sharey=True)
            axes = axes.flatten()

            for i, view_date in enumerate(dates_to_plot):
                ax = axes[i]

                lookback_df = df.filter(
                    (pl.col("time").dt.replace_time_zone(None) >= (view_date - timedelta(hours=LOOKBACK_HOURS))) &
                    (pl.col("time").dt.date() < view_date)
                )
                target_day_df = df.filter(
                    pl.col("time").dt.date() == view_date
                )

                if lookback_df.height != LOOKBACK_HOURS or target_day_df.height != 24:
                    continue

                X = torch.tensor(lookback_df[FEATURE_COLS].to_numpy(), dtype=torch.float32).unsqueeze(0)
                X = transform_batch(X, input_scaler.transform)

                model.eval()
                with torch.no_grad():
                    predictions = model(X)
                    predictions = transform_batch(predictions, output_scaler.inverse_transform)

                ax.plot(target_day_df['time'], predictions[0, :, 0], lw=0.7, linestyle='-', color='red', label="Forecast")
                ax.plot(target_day_df['time'], target_day_df[TARGET_COLS], lw=0.7, linestyle='--', color='blue', label="Target")

                ax.set_title(view_date.strftime("%Y-%m-%d"), fontsize=9)
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
                ax.tick_params(axis="x", labelsize=7, rotation=0)
                ax.tick_params(axis="y", labelsize=7)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax.grid(True, lw=0.25, linestyle="--")
                if i // ncols == nrows - 1:
                    ax.set_xlabel("Hour", fontsize=8)
                if i % ncols == 0 and i / ncols == 2:
                    ax.set_ylabel("Price (GBP MWh$^{-1}$)", fontsize=8)

            # hide unused subplots
            for j in range(len(dates_to_plot), len(axes)):
                fig.delaxes(axes[j])

            # shared legend
            handles, labels = axes[0].get_legend_handles_labels()

            fig.legend(
                handles, labels,
                loc='lower center',
                ncol=2,
                frameon=False,
                fontsize=8,
                bbox_to_anchor=(0.5, -0.02)
            )

            plt.tight_layout(rect=[0, 0.03, 1, 1])  # leave space for legend
            plt.subplots_adjust(hspace=0.9, wspace=0.1)
            plt.show()
