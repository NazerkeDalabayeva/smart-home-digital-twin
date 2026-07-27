# =========================================================
# MSc Data Science Project – Decomposition-Based Forecasting
# Trend + Seasonality + ARIMA Residual Forecast
# With Baseline Comparison and Metrics
# =========================================================

import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


plt.rcParams["figure.figsize"] = (12, 4)

# =========================================================
# 1. LOAD DATA
# =========================================================

folder = r"/Users/nazerke/Downloads/farm_csvs_2026-04-03"
files = glob(os.path.join(folder, "farm_telemetry_2026-*.csv"))

if not files:
    raise FileNotFoundError("No CSV files found. Check your folder path.")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp_utc"])
df = df.sort_values("timestamp_utc")
df = df.set_index("timestamp_utc")

# Hourly data for forecasting
df_hourly = df.select_dtypes(include="number").resample("1h").mean()

# Fill small gaps
df_hourly = df_hourly.asfreq("1h").interpolate(method="time")

# =========================================================
# 2. METRIC FUNCTION
# =========================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - (ss_res / ss_tot)

    return mae, rmse, r2


# =========================================================
# 3. DECOMPOSITION-BASED ARIMA FORECASTING FUNCTION
# =========================================================

def decomposition_arima_forecast(
    series,
    name,
    period=24,
    test_steps=24,
    arima_order=(1, 0, 1),
    trend_window=48
):
    """
    Decomposition-based forecasting:

    y_t = T_t + S_t + R_t

    Forecast:
    y_hat = forecast_trend + repeated_seasonal_pattern + ARIMA_forecast_residual

    period=24 means daily seasonality for hourly data.
    test_steps=24 means last 24 hours are used as test data.
    """

    series = series.dropna()

    if len(series) < (2 * period + test_steps):
        raise ValueError(f"Not enough data for {name}. Need at least two seasonal cycles plus test data.")

    # ------------------------------
    # Train-test split
    # ------------------------------
    train = series.iloc[:-test_steps]
    test = series.iloc[-test_steps:]

    # ------------------------------
    # Seasonal decomposition on TRAIN only
    # Important: avoid using test data during decomposition
    # ------------------------------
    decomposition = seasonal_decompose(
        train,
        model="additive",
        period=period,
        extrapolate_trend="freq"
    )

    trend = decomposition.trend.dropna()
    seasonal = decomposition.seasonal.dropna()
    residual = decomposition.resid.dropna()

    # ------------------------------
    # Forecast seasonal component
    # Repeat the last full daily seasonal cycle
    # ------------------------------
    last_seasonal_cycle = seasonal.iloc[-period:].values

    seasonal_forecast_values = np.resize(last_seasonal_cycle, test_steps)
    seasonal_forecast = pd.Series(
        seasonal_forecast_values,
        index=test.index,
        name="seasonal_forecast"
    )

    # ------------------------------
    # Forecast trend component
    # Simple linear extrapolation from recent trend values
    # ------------------------------
    recent_trend = trend.dropna().iloc[-trend_window:]

    x = np.arange(len(recent_trend))
    y = recent_trend.values

    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(recent_trend), len(recent_trend) + test_steps)
    trend_forecast_values = intercept + slope * future_x

    trend_forecast = pd.Series(
        trend_forecast_values,
        index=test.index,
        name="trend_forecast"
    )

    # ------------------------------
    # Forecast residual component using ARIMA
    # ------------------------------
    residual_model = ARIMA(residual, order=arima_order)
    residual_fit = residual_model.fit()

    residual_forecast_values = residual_fit.forecast(steps=test_steps)
    residual_forecast = pd.Series(
        residual_forecast_values.values,
        index=test.index,
        name="residual_forecast"
    )

    # ------------------------------
    # Reconstruct final forecast
    # ------------------------------
    final_forecast = trend_forecast + seasonal_forecast + residual_forecast
    final_forecast.name = "decomposition_arima_forecast"

    # ------------------------------
    # Baselines
    # ------------------------------

    # Naive forecast: next value = last training value
    naive_forecast = pd.Series(
        [train.iloc[-1]] * test_steps,
        index=test.index,
        name="naive_forecast"
    )

    # Seasonal naive forecast: repeat last 24 hours
    seasonal_naive_values = np.resize(train.iloc[-period:].values, test_steps)
    seasonal_naive_forecast = pd.Series(
        seasonal_naive_values,
        index=test.index,
        name="seasonal_naive_forecast"
    )

    # ------------------------------
    # Metrics
    # ------------------------------
    mae_model, rmse_model, r2_model = calculate_metrics(test, final_forecast)
    mae_naive, rmse_naive, r2_naive = calculate_metrics(test, naive_forecast)
    mae_snaive, rmse_snaive, r2_snaive = calculate_metrics(test, seasonal_naive_forecast)

    results = pd.DataFrame({
        "Model": [
            "Naive persistence",
            "Seasonal naive",
            "Decomposition + ARIMA residual"
        ],
        "Target": [name, name, name],
        "MAE": [mae_naive, mae_snaive, mae_model],
        "RMSE": [rmse_naive, rmse_snaive, rmse_model],
        "R2": [r2_naive, r2_snaive, r2_model]
    })

    print("\n=================================================")
    print(f"Forecasting results for {name}")
    print("=================================================")
    print(results.round(3))

    print("\nARIMA residual model summary:")
    print(f"AIC: {residual_fit.aic:.3f}")
    print(f"BIC: {residual_fit.bic:.3f}")

    # ------------------------------
    # Plot final forecast
    # ------------------------------
    plt.figure(figsize=(12, 4))
    plt.plot(series.index, series, label="Actual", alpha=0.45)
    plt.plot(test.index, test, label="Test actual", linewidth=2)
    plt.plot(test.index, final_forecast, label="Decomposition + ARIMA residual forecast", linestyle="--", linewidth=2)
    plt.plot(test.index, seasonal_naive_forecast, label="Seasonal naive baseline", linestyle=":", linewidth=2)

    plt.title(f"Decomposition-Based Forecast vs Actual ({name})")
    plt.xlabel("Time")
    plt.ylabel(name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # Plot decomposition components
    # ------------------------------
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"Training Decomposition for {name}", y=1.02)
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # Plot reconstructed components for test period
    # ------------------------------
    plt.figure(figsize=(12, 4))
    plt.plot(test.index, trend_forecast, label="Forecast trend")
    plt.plot(test.index, seasonal_forecast, label="Forecast seasonal component")
    plt.plot(test.index, residual_forecast, label="Forecast residual")
    plt.title(f"Forecast Components for {name}")
    plt.xlabel("Time")
    plt.ylabel("Component value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results, final_forecast


# =========================================================
# 4. RUN DECOMPOSITION-BASED FORECASTING
# =========================================================

all_results = []

targets = {
    "Temperature (°C)": "temp_c",
    "Humidity (%)": "rh_percent",
    "RSSI (dBm)": "rssi_dbm"
}

for label, col in targets.items():
    if col in df_hourly.columns:
        try:
            results, forecast = decomposition_arima_forecast(
                series=df_hourly[col],
                name=label,
                period=24,          # daily seasonality for hourly data
                test_steps=24,      # last 24 hours as test set
                arima_order=(1, 0, 1),
                trend_window=48
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nCould not forecast {label}: {e}")

# =========================================================
# 5. COMBINED RESULTS TABLE
# =========================================================

if all_results:
    final_results = pd.concat(all_results, ignore_index=True)

    print("\n=================================================")
    print("Combined forecasting results")
    print("=================================================")
    print(final_results.round(3))

# =========================================================
# END
# =========================================================