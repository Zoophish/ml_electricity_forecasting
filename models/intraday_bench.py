import polars as pl
from datetime import timedelta, datetime

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from timemachine.models.mlp import HourglassMLP

from timemachine.preprocessing.sliding_window import SlidingWindowDataset, transform_batch
import timemachine.preprocessing.rolling_features as rf

from timemachine.training.early_stopper import EarlyStopper
from timemachine.training.loss import MAPELoss, MAELoss
import dataset.data as data

import matplotlib.pyplot as plt


if __name__ == '__main__':

    # ---- SCRIPT OPTIONS ----
    MODE = 'Load'
    PARAM_CACHE_PATH = 'Intraday.mdl'
    PARAM_SAVE_PATH = 'Intraday.mdl'

    INTERACTIVE_PLOT = True  # show interactive plot

    # ---- HYPERPARAMETERS ----
    TRAINING_RATIO = 0.8
    EPOCHS = 200

    # mlp parameters
    HIDDEN_SIZE_FAC = 2  # factor of input layer size
    HIDDEN_LAYERS = 3
    DROPOUT = 0.5

    INFERENCE_MC_SAMPLES = 128  # number of stochastic samples for MC dropout

    # training parameters
    BATCH_SIZE = 64
    L2_LAMBDA = 1.0
    GRAD_CLIP_THRESHOLD = 2.65
    LEARNING_RATE = 1e-4

    # early stopping
    PATIENCE = 60  # number of epochs to stop if there is no significant change in test loss
    MIN_DELTA = .05

    # define input and ouput window sizes
    LOOKBACK_HOURS = 24
    HORIZON_HOURS = 6


    # prepare data
    clip = ( pl.datetime(2022,1,1), pl.datetime(2025,1,1) )
    df = data.fetch_full_dataset(clip=clip, sample_granularity='1h')

    # predicting the log returns can sometimes yield better results than the raw price
    df = df.with_columns(
        pl.Series(rf.log_returns(df['10'].to_numpy(), eps=1e-3)).alias('10_log_returns'),
    )
    print(df.head())


    FEATURE_COLS = [
        '10',
        '10_log_returns',
        '9',
        '4',
        '5',
        'year_sin', 'year_cos',
        'week_sin', 'week_cos',
        'day_sin', 'day_cos',
    ]
    TARGET_COLS = ['10',]  # can change between imbalance/continuous intraday here

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

    # isolate the train and test set from each other
    min_df_dt = df['time'].min()
    max_df_dt = df['time'].max()
    partition_dt = min_df_dt + (max_df_dt - min_df_dt) * TRAINING_RATIO
    train_df = df.filter(pl.col('time') < partition_dt)
    test_df = df.filter(pl.col('time') >= partition_dt)

    # create the scalers
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    input_scaler.fit(train_df[FEATURE_COLS].to_numpy())
    output_scaler.fit(train_df[TARGET_COLS].to_numpy())

    # set up the sliding window datasets & loaders
    train_dataset = SlidingWindowDataset(
        df=train_df,
        timestamp_col='time',
        lookback=timedelta(hours=LOOKBACK_HOURS),
        horizon=timedelta(hours=HORIZON_HOURS),
        stride=timedelta(hours=3),
        lookback_steps=LOOKBACK_HOURS,
        horizon_steps=HORIZON_HOURS,
        lookback_cols=FEATURE_COLS,
        target_cols=TARGET_COLS
    )
    test_dataset = SlidingWindowDataset(
        df=test_df,
        timestamp_col='time',
        lookback=timedelta(hours=LOOKBACK_HOURS),
        horizon=timedelta(hours=HORIZON_HOURS),
        stride=timedelta(hours=3),
        lookback_steps=LOOKBACK_HOURS,
        horizon_steps=HORIZON_HOURS,
        lookback_cols=FEATURE_COLS,
        target_cols=TARGET_COLS
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


    INPUT_SIZE = len(FEATURE_COLS)
    # output size is x2 here because for every point output, there is a 'variance head'
    OUTPUT_SIZE = HORIZON_HOURS * 2

    # create the mlp
    model = HourglassMLP(
        input_size=INPUT_SIZE*LOOKBACK_HOURS,
        hidden_size_fac=HIDDEN_SIZE_FAC,
        hidden_layers=HIDDEN_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
        format='batch-time-channel'
    )

    print(model)

    if MODE == 'Train':
        # loss criterion
        criterion = torch.nn.MSELoss()
        test_criterion = torch.nn.MSELoss()
        mae_criterion = MAELoss()
        mape_criterion = MAPELoss()

        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=L2_LAMBDA,
            amsgrad=True
        )

        scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, min_lr=1e-6)
        early_stopper = EarlyStopper(patience=PATIENCE, min_delta=MIN_DELTA)

        try:
            writer = SummaryWriter()
            best_test_nll = float('inf')
            for epoch in range(EPOCHS):
                model.train()
                optimiser.zero_grad()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    # scale the batches
                    X_batch = transform_batch(X_batch, input_scaler.transform)
                    y_batch = transform_batch(y_batch, output_scaler.transform)

                    output = model(X_batch)
                    mu, log_var = output[:, :HORIZON_HOURS, :], output[:, HORIZON_HOURS:, :]

                    sigma2 = torch.exp(log_var)  # variance head
                    sigma2 = torch.clamp(sigma2, min=1e-6)  # avoid numerical issues
                    nll = 0.5 * log_var + (y_batch - mu)**2 / (2*sigma2)
                    loss = nll.mean()
                    train_loss += loss.item()

                    # backward pass
                    optimiser.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_THRESHOLD)
                    optimiser.step()
                
                train_loss /= len(train_loader)

                test_loss = 0
                mape = 0
                mae = 0
                model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch_scaled = transform_batch(X_batch, input_scaler.transform)
                        y_batch = transform_batch(y_batch, output_scaler.transform)

                        output = model(X_batch_scaled)
                        mu, log_var = output[:, :HORIZON_HOURS, :], output[:, HORIZON_HOURS:, :]
                        sigma2 = torch.exp(log_var)  # variance head
                        sigma2 = torch.clamp(sigma2, min=1e-6)

                        # heteroskedastic gaussian negative log-likelihood
                        nll = 0.5 * log_var + (y_batch - mu)**2 / (2*sigma2)
                        test_loss += nll.mean().item()

                        # rescale to get correct MAPE
                        mu = transform_batch(mu, output_scaler.transform)
                        mape += mape_criterion(mu, y_batch).item()
                        mae += mae_criterion(mu, y_batch).item()

                    # mean test errors
                    test_loss /= len(test_loader); mape /= len(test_loader); mae /= len(test_loader)

                scheduler.step(mape)
                current_lr = optimiser.param_groups[0]['lr']
                
                print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Test MAPE: {mape:.2f} Test MAE: {mae:.4f}")
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/test', test_loss, epoch)
                writer.add_scalar('MAPE/test', mape, epoch)
                writer.add_scalar('LearningRate', current_lr, epoch)

                if test_loss < best_test_nll:
                    best_test_nll = test_loss
                    best_model_state = model.state_dict()
                if early_stopper.early_stop(mape):
                    print(f"Stopping early at patience={early_stopper.counter} epochs.")
                    break

        except KeyboardInterrupt:
            print("Training interrupted.")
        finally:
            model.load_state_dict(best_model_state)
            writer.close()
            # save the parameters
            torch.save(model.state_dict(), PARAM_SAVE_PATH)
            print(f"Best NLL: {best_test_nll:.4f}")
    elif MODE == 'Load':
        model.load_state_dict(torch.load(PARAM_CACHE_PATH))
        model.eval()


    # ---- Interactive Plotting ----
    if INTERACTIVE_PLOT:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

        plt.xticks(rotation=45)
        plt.grid(True, which="both", axis="x", linestyle="--", linewidth=0.5)

        day_idx = 0
        timedelta_step = 60
        center_datetime = partition_dt

        def update_plot():
            day_df = df.filter(
                (pl.col("time").dt.replace_time_zone(None) >= (center_datetime - timedelta(hours=LOOKBACK_HOURS*4)))
                & (pl.col("time").dt.replace_time_zone(None) <= (center_datetime + timedelta(hours=LOOKBACK_HOURS*4)))
            )

            # get the lookback and target dataframes
            lookback_df = df.filter(
                (pl.col("time").dt.replace_time_zone(None) > (center_datetime - timedelta(hours=LOOKBACK_HOURS)))
                & (pl.col("time").dt.replace_time_zone(None) <= center_datetime)
            )
            target_df = df.filter(
                (pl.col("time").dt.replace_time_zone(None) >= center_datetime)
                & (pl.col("time").dt.replace_time_zone(None) < (center_datetime + timedelta(hours=HORIZON_HOURS)))
            )


            X_batch = torch.tensor(lookback_df[FEATURE_COLS].to_numpy()[None, :], dtype=torch.float32)
            X_batch = transform_batch(X_batch, input_scaler.transform)

            # for each sample, store its point forecasts (mu) and aleatoric variance forecasts (var)
            mcd_mu = torch.zeros( [INFERENCE_MC_SAMPLES, HORIZON_HOURS] )
            mcd_var = torch.zeros( [INFERENCE_MC_SAMPLES, HORIZON_HOURS] )

            # droput gets disabled by default with .eval(), so this re-enables it
            def enable_dropout(module):
                for m in module.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()

            model.eval()
            with torch.no_grad():
                enable_dropout(model)
                for sample in range(INFERENCE_MC_SAMPLES):
                    output = model(X_batch)
                    mu, log_var = output[0, :HORIZON_HOURS, 0], output[0, HORIZON_HOURS:, 0]
                    sigma2 = torch.exp(log_var)  # variance head
                    sigma = sigma2.sqrt()  # convert to std deviation
                    # variance needs to be transformed differently to mu
                    transformed_mu = mu * output_scaler.scale_ + output_scaler.mean_
                    transformed_sigma = sigma * output_scaler.scale_
                    transformed_sigma2 = transformed_sigma**2
                    # store this sample
                    mcd_mu[sample, :] = transformed_mu[:]
                    mcd_var[sample, :] = transformed_sigma2[:]

            mu_hat = mcd_mu.mean(dim=0)  # get the mean of point forecasts
            var_ep = mcd_mu.var(dim=0)  # get the variance of point forecasts (epistemic uncertainty)
            std_ep = var_ep.sqrt()  # convert to standard deviation

            var_al = mcd_var.mean(dim=0)  # get the mean aleatoric variance forecasts
            std_al = var_al.sqrt()  # convert to standard deviation

            ax1.clear()
            ax1.set_title(f"{center_datetime}")

            ax1.plot(day_df['time'], day_df[TARGET_COLS], lw=0.8, marker='o', color='g', markersize=1, label=TARGET_COLS)

            ax1.plot(lookback_df['time'], lookback_df[TARGET_COLS], lw=1.5, linestyle='--', color='b', alpha=0.4, markersize=1, label='Lookback Window')

            ax1.plot(target_df['time'], target_df[TARGET_COLS], lw=1, linestyle='--', color='orange', markersize=1, label='Target')


            times = target_df['time']

            ax1.plot(target_df['time'], mu_hat, color='red', label='Mean')

            ax1.fill_between(
                target_df['time'],
                mu_hat - std_ep,
                mu_hat + std_ep,
                color='red',
                alpha=0.1,
                label=f'Epistemic ±1σ'
            )

            ax1.fill_between(
                times,
                mu_hat - 1*std_al,
                mu_hat + 1*std_al,
                color='blue',
                alpha=0.1,
                label='Aleatoric ±1σ'
            )

            ax1.grid(True, lw=.25)
            ax1.legend()
            plt.tight_layout()
            plt.draw()

        def on_key(event):
            global center_datetime, timedelta_step
            if event.key == 'right':
                center_datetime = center_datetime + timedelta(minutes=timedelta_step)
            elif event.key == 'left':
                center_datetime = center_datetime - timedelta(minutes=timedelta_step)
            elif event.key == 'up':
                timedelta_step += 30
            elif event.key == 'down':
                timedelta_step -= 30
            update_plot()

        fig.canvas.mpl_connect('key_press_event', on_key)
        update_plot()
        plt.legend()
        plt.show()


    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.dates as mdates
    import random

    def sample_dates(start_date, end_date, n, seed=45):
        """Randomly sample n unique dates between start_date and end_date (inclusive)."""
        random.seed(seed)
        delta = (end_date - start_date).days
        all_days = [start_date + timedelta(days=i) for i in range(delta + 1)]
        sampled_days =  random.sample(all_days, n)
        sampled_days = [dt + timedelta(hours=random.randint(0, 24)) for dt in sampled_days]
        return sampled_days

    start = datetime(2024, 5, 1)
    end = datetime(2025, 1, 1)
    n_samples = 6
    dates_to_plot = sample_dates(start, end, n_samples, 21)

    # === Plotting Parameters ===
    ncols = 3
    nrows = (len(dates_to_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.4, 1.5 * nrows), sharey=False)
    axes = axes.flatten()

    # === Plot Loop ===
    for i, center_datetime in enumerate(dates_to_plot):
        ax = axes[i]

        # Filter data
        day_df = df.filter(
            (pl.col("time").dt.replace_time_zone(None) >= (center_datetime - timedelta(hours=12)))
            & (pl.col("time").dt.replace_time_zone(None) <= (center_datetime + timedelta(hours=12)))
        )

        # get the lookback
        lookback_df = df.filter(
            (pl.col("time").dt.replace_time_zone(None) > (center_datetime - timedelta(hours=LOOKBACK_HOURS)))
            & (pl.col("time").dt.replace_time_zone(None) <= center_datetime)
        )
        target_df = df.filter(
            (pl.col("time").dt.replace_time_zone(None) >= center_datetime)
            & (pl.col("time").dt.replace_time_zone(None) < (center_datetime + timedelta(hours=HORIZON_HOURS)))
        )

        X_batch = torch.tensor(lookback_df[FEATURE_COLS].to_numpy()[None, :], dtype=torch.float32)
        X_batch = transform_batch(X_batch, input_scaler.transform)

        def enable_dropout(module):
            """ Function to enable the dropout layers during test-time """
            for m in module.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

        mcd_mu = torch.zeros( [INFERENCE_MC_SAMPLES, HORIZON_HOURS] )
        mcd_var = torch.zeros( [INFERENCE_MC_SAMPLES, HORIZON_HOURS] )

        model.eval()
        with torch.no_grad():
            enable_dropout(model)
            # set_dropout(model, DROPOUT)
            for sample in range(INFERENCE_MC_SAMPLES):  # you could batch this
                output = model(
                    X_batch
                )
                mu, log_var = output[0, :HORIZON_HOURS, 0], output[0, HORIZON_HOURS:, 0]
                sigma2 = torch.exp(log_var)  # variance head
                sigma = sigma2.sqrt()  # convert to std deviation
                transformed_mu = mu * output_scaler.scale_ + output_scaler.mean_
                transformed_sigma = sigma * output_scaler.scale_
                transformed_sigma2 = transformed_sigma**2
                mcd_mu[sample, :] = transformed_mu[:]
                mcd_var[sample, :] = transformed_sigma2[:]

        ax.clear()
        ax.set_title(f"{center_datetime}")

        ax.plot(day_df['time'], day_df[TARGET_COLS], lw=0.7, color='b', label="Target", linestyle='--')

        ax.axvspan(day_df['time'].min(), lookback_df['time'].max(), color='grey', alpha=0.1, hatch="////", lw=0.5, label='Lookback Window')

        # epistemic uncertainty
        mu_hat = mcd_mu.mean(dim=0)
        var_ep = mcd_mu.var(dim=0)
        std_ep = var_ep.sqrt()

        var_al = mcd_var.mean(dim=0)
        std_al = var_al.sqrt()

        std_tot = (var_ep + var_al).sqrt()

        times = target_df['time']

        ax.plot(target_df['time'], mu_hat, color='red', label='Mean', lw=0.7)

        ax.fill_between(
            target_df['time'],
            mu_hat - std_ep,
            mu_hat + std_ep,
            color='red',
            alpha=0.2,
            label=f'Epistemic ±1σ',
            lw=0.2
        )

        ax.fill_between(
            times,
            mu_hat - 1*std_al,
            mu_hat + 1*std_al,
            color='green',
            alpha=0.2,
            label='Aleatoric ±1σ',
            lw=0.2
        )

        ax.set_title(center_datetime.strftime("%Y-%m-%d %H:%M"), fontsize=9)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        ax.tick_params(axis="x", labelsize=7, rotation=0)
        ax.tick_params(axis="y", labelsize=7)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, lw=0.25, linestyle="--")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Hour", fontsize=8)
        if i == 0 and i / ncols == 1:
            ax.set_ylabel("Price (GBP MWh$^{-1}$)", fontsize=8)

    # === Hide Unused Subplots ===
    for j in range(len(dates_to_plot), len(axes)):
        fig.delaxes(axes[j])

    # === Shared Legend ===
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=2,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.00)
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])  # leave space for legend
    plt.subplots_adjust(bottom=0.3, hspace=0.55, wspace=0.3)
    plt.show()
