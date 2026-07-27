from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Folder with your CSV files
folder = Path(r"/Users/nazerke/Downloads/farm_csvs_2026-04-03")

# Choose the day where button was pressed
target_day = "2026-03-21"   # change this to your detected day

# Read all CSV files from the folder
csv_files = list(folder.glob("*.csv"))

df_list = []
for file in csv_files:
    temp = pd.read_csv(file)
    temp["source_file"] = file.name
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

# Prepare timestamp
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp_utc"])
df = df.sort_values("timestamp_utc").reset_index(drop=True)

# Make sure button is numeric
df["button"] = pd.to_numeric(df["button"], errors="coerce").fillna(1).astype(int)

# Filter one day
start = pd.Timestamp(target_day, tz="UTC")
end = start + pd.Timedelta(days=1)

day_df = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] < end)].copy()

# Detect button press events: 1 -> 0
# Because 1 = not pressed, 0 = pressed
day_df["button_press_event"] = (
    (day_df["button"] == 0) &
    (day_df["button"].shift(1).fillna(1) == 1)
)

# Print detected press times
press_times = day_df.loc[day_df["button_press_event"], "timestamp_utc"]

print(f"Button press events detected on {target_day}: {len(press_times)}")
print(press_times)

# Plot
plt.figure(figsize=(12, 4))

plt.step(
    day_df["timestamp_utc"],
    day_df["button"],
    where="post",
    label="Button state"
)

# Mark press moments
plt.scatter(
    press_times,
    [0] * len(press_times),
    marker="o",
    s=80,
    label="Detected button press"
)

plt.ylim(-0.2, 1.2)
plt.yticks([0, 1], ["Pressed (0)", "Not pressed (1)"])

plt.xlabel("Time (UTC)")
plt.ylabel("Button state")
plt.title(f"Button State on {target_day}")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.grid(True, axis="x", alpha=0.3)
plt.legend()
plt.tight_layout()

# Save high-quality image for thesis
output_path = folder / f"button_state_{target_day}.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Saved figure to: {output_path}")