# =========================================================
# Predictive Maintenance – Multi-Parameter Telemetry Health
# =========================================================

import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 4)

# =========================================================
# 1. LOAD DATA
# =========================================================

folder = r"C:\Users\NDalabayeva\Downloads\data"
files = glob(os.path.join(folder, "farm_telemetry_2026-*.csv"))

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
df = df.sort_values("timestamp_utc").set_index("timestamp_utc")

df_hourly = df.select_dtypes(include="number").resample("1h").mean()

# =========================================================
# 2. COMMUNICATION HEALTH – RSSI
# =========================================================

rssi = df_hourly["rssi_dbm"].dropna()
rssi_roll = rssi.rolling(24)

rssi_mean = rssi_roll.mean()
rssi_std = rssi_roll.std()

plt.plot(rssi, alpha=0.4, label="RSSI (hourly)")
plt.plot(rssi_mean, label="24h Rolling Mean", linewidth=2)
plt.axhline(-75, color="red", linestyle="--", label="Weak signal threshold")
plt.title("RSSI-based communication health monitoring")
plt.ylabel("RSSI (dBm)")
plt.legend()
plt.show()

rssi_alerts = rssi_mean < -75
print("RSSI degradation detected at:")
print(rssi_alerts[rssi_alerts].index)

# =========================================================
# 3. TEMPERATURE & HUMIDITY SENSOR STABILITY
# =========================================================

def variance_alert(series, name):
    roll_var = series.rolling(24).var()
    upper = roll_var.mean() + 3 * roll_var.std()
    lower = roll_var.mean() * 0.1  # variance collapse = sensor stuck

    plt.plot(roll_var, label="Rolling Variance")
    plt.axhline(upper, color="red", linestyle="--", label="High variance")
    plt.axhline(lower, color="orange", linestyle="--", label="Low variance")
    plt.title(f"{name} – Variance-Based Health Monitoring")
    plt.legend()
    plt.show()

    alerts = (roll_var > upper) | (roll_var < lower)
    print(f"{name} variance alerts:")
    print(alerts[alerts].index)

variance_alert(df_hourly["temp_c"], "Temperature")
variance_alert(df_hourly["rh_percent"], "Humidity")

# =========================================================
# 4. LIGHT SENSOR – ADC FLAT-LINE & SATURATION
# =========================================================

light = df_hourly["light_adc"].dropna()
light_diff = light.diff().abs()

flatline = light_diff.rolling(24).mean() < 5   # ADC noise only
saturation = light > 4000                      # near max ADC

plt.plot(light, alpha=0.4, label="Light ADC")
plt.plot(light_diff.rolling(24).mean(), label="Rolling |Δ|", linewidth=2)
plt.title("Light sensor flat-line and saturation monitoring")
plt.legend()
plt.show()

print("Light sensor flat-line periods:")
print(flatline[flatline].index)

print("Light sensor saturation periods:")
print(saturation[saturation].index)

# =========================================================
# 5. DISTANCE SENSOR – ECHO RELIABILITY
# =========================================================

distance = df_hourly["distance_cm"]

invalid = (distance < 2) | (distance > 400)
high_variance = distance.rolling(24).var() > 1.5

plt.plot(distance, label="Distance (cm)")
plt.title("Ultrasonic Distance – Health Monitoring")
plt.ylabel("cm")
plt.legend()
plt.show()

print("Distance sensor invalid readings:")
print(distance[invalid].index)

print("Distance sensor instability detected at:")
print(high_variance[high_variance].index)

# =========================================================
# 6. SEQUENCE NUMBER – DEVICE REBOOTS
# =========================================================

df["seq_diff"] = df["seq"].diff()

reboots = df[df["seq_diff"] < 0]
drops = df[df["seq_diff"] > 1]

print(f"Device reboots detected: {len(reboots)}")
print(reboots.index)

print(f"Transmission drops detected: {len(drops)}")
print(drops.index)

# =========================================================
# END OF PREDICTIVE MAINTENANCE PIPELINE
# =========================================================