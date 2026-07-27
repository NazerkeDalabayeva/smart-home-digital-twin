from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Folder where your CSV files are located
folder = Path("/Users/nazerke/Downloads/farm_csvs_2026-04-03")

# Choose the day where PIR motion was detected
target_day = "2026-04-03"   # change this to your detected day

# Read all CSV files from folder and subfolders
csv_files = list(folder.rglob("*.csv"))

print("CSV files found:")
for f in csv_files:
    print(f)

if not csv_files:
    raise FileNotFoundError("No CSV files found. Check your folder path.")

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

# Make PIR numeric
df["pir_motion"] = pd.to_numeric(df["pir_motion"], errors="coerce").fillna(0).astype(int)

# Filter one day
start = pd.Timestamp(target_day, tz="UTC")
end = start + pd.Timedelta(days=1)

day_df = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] < end)].copy()

if day_df.empty:
    raise ValueError(f"No data found for {target_day}. Check the date.")

# Detect PIR motion event: 0 -> 1
# 0 = no motion, 1 = motion detected
day_df["pir_motion_event"] = (
    (day_df["pir_motion"] == 1) &
    (day_df["pir_motion"].shift(1).fillna(0) == 0)
)

# Print detected PIR motion times
motion_times = day_df.loc[day_df["pir_motion_event"], "timestamp_utc"]

print(f"\nPIR motion events detected on {target_day}: {len(motion_times)}")
print(motion_times)

# Plot
plt.figure(figsize=(12, 4))

plt.step(
    day_df["timestamp_utc"],
    day_df["pir_motion"],
    where="post",
    label="PIR motion state"
)

# Mark detected motion starts
plt.scatter(
    motion_times,
    [1] * len(motion_times),
    marker="o",
    s=80,
    label="Detected PIR motion"
)

plt.ylim(-0.2, 1.2)
plt.yticks([0, 1], ["No motion (0)", "Motion detected (1)"])

plt.xlabel("Time (UTC)")
plt.ylabel("PIR motion state")
plt.title(f"PIR Motion Detection on {target_day}")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.grid(True, axis="x", alpha=0.3)
plt.legend()
plt.tight_layout()

# Save high-quality image for thesis
output_path = folder / f"pir_motion_{target_day}.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nSaved figure to: {output_path}")