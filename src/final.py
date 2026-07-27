# =========================================================
# MSc Data Science Project – Full Telemetry Analysis
# =========================================================

import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore

plt.rcParams["figure.figsize"] = (12, 4)

# =========================================================
# 1. LOAD & PREPARE DATA
# =========================================================

folder = r"C:\Users\NDalabayeva\Downloads\data"
files = glob(os.path.join(folder, "farm_telemetry_2026-*.csv"))

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
df = df.sort_values("timestamp_utc")
df = df.set_index("timestamp_utc")

# =========================================================
# 2. RESAMPLING FOR TIME-SERIES
# =========================================================

df_1min = df.select_dtypes(include="number").resample("1min").mean()
df_hourly = df.select_dtypes(include="number").resample("1h").mean()

# =========================================================
# 3. DESCRIPTIVE STATISTICS
# =========================================================

env_vars = [
    "temp_c", "rh_percent", "light_adc",
    "soil_adc", "water_adc", "rain_adc",
    "rssi_dbm", "distance_cm"
]

desc_stats = df_1min[env_vars].describe().T
print(desc_stats)

df_1min[env_vars].plot(kind="box", subplots=True, layout=(4, 2), sharex=False)
plt.tight_layout()
plt.show()

df_1min[env_vars].hist(bins=50)
plt.tight_layout()
plt.show()

# =========================================================
# 6. CORRELATION & MULTIVARIATE ANALYSIS
# =========================================================

# Excluding constant variables from meaningful correlation discussion:
# soil_adc, water_adc, and rain_adc are constant zero in this dataset.

corr_vars = [
    "temp_c", "rh_percent", "light_adc", "rssi_dbm"
]

corr_matrix = df_1min[corr_vars].corr(method="spearman")

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Spearman Correlation Matrix")
plt.show()

# =========================================================
# 6A. NORMALIZED LAGGED CORRELATION ANALYSIS
# =========================================================

lag_hours = 24
lags = range(0, lag_hours + 1)

pairs = {
    "Temp → Humidity": ("temp_c", "rh_percent"),
    "Light → Temperature": ("light_adc", "temp_c"),
    "Light → Humidity": ("light_adc", "rh_percent"),
}

plt.figure()

for label, (leading_var, response_var) in pairs.items():

    pair_df = df_hourly[[leading_var, response_var]].dropna()

    # Z-score normalization
    pair_df[leading_var] = zscore(pair_df[leading_var])
    pair_df[response_var] = zscore(pair_df[response_var])

    lag_corr = [
        pair_df[leading_var].shift(lag).corr(pair_df[response_var])
        for lag in lags
    ]

    best_lag = lags[np.nanargmax(np.abs(lag_corr))]
    best_corr = lag_corr[best_lag]

    print(f"{label}: strongest correlation = {best_corr:.3f} at lag {best_lag} hours")

    plt.plot(lags, lag_corr, marker="o", label=label)

plt.title("Normalized Lagged Correlation Analysis")
plt.xlabel("Lag (hours)")
plt.ylabel("Correlation")
plt.legend()
plt.grid(True)
plt.show()

# =========================================================
# 7. OPTIONAL – ANOMALY DETECTION (RESIDUAL-BASED)
# =========================================================

light_series = df_1min["light_adc"].dropna()
light_decomp = seasonal_decompose(light_series, period=1440)

resid = light_decomp.resid
threshold = 3 * resid.std()

anomalies = resid[np.abs(resid) > threshold]

light_series.plot()
plt.scatter(anomalies.index, light_series.loc[anomalies.index], color="red")
plt.title("Light Sensor – Anomaly Detection")
plt.show()

rssi_series = df_1min["rssi_dbm"].dropna()
rssi_decomp = seasonal_decompose(rssi_series, period=1440)

resid = rssi_decomp.resid
threshold = 3 * resid.std()

anomalies = resid[np.abs(resid) > threshold]

rssi_series.plot()
plt.scatter(anomalies.index, rssi_series.loc[anomalies.index], color="red")
plt.title("RSSI Sensor – Anomaly Detection")
plt.show()

# =========================================================
# END OF PIPELINE
# =========================================================