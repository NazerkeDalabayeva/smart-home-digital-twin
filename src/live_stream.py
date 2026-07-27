import os
import json
import ssl
from datetime import datetime, timezone

import pandas as pd
import paho.mqtt.client as mqtt

from rich.console import Console
from rich.table import Table
from rich.live import Live


# ==============================
# MQTT CONFIG (HiveMQ Cloud)
# ==============================

MQTT_BROKER = "49bfd52aaf7440d3bda3b1ebd299a9ce.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "farm/farm01/telemetry"

MQTT_USERNAME = "**********"
MQTT_PASSWORD = "**********"


# ==============================
# CSV CONFIG (ONE FILE ONLY)
# ==============================

RAW_FILE = "/Users/nazerke/Desktop/UofA/MSc Data Science project/farm_telemetry_2026-04-27.csv"

CSV_COLUMNS = [
    "timestamp_utc",
    "seq",
    "temp_c",
    "rh_percent",
    "light_adc",
    "soil_adc",
    "water_adc",
    "rain_adc",
    "pir_motion",
    "button",
    "distance_cm",
    "rssi_dbm",
]

# Create CSV with header if not exists
if not os.path.exists(RAW_FILE):
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(RAW_FILE, index=False)


# ==============================
# RICH LIVE TABLE
# ==============================

console = Console()
latest_row = {"status": "Waiting for messages..."}


def build_table():
    table = Table(title="Live Smart Farm Data")
    table.add_column("Field")
    table.add_column("Value")

    for k, v in latest_row.items():
        table.add_row(str(k), str(v))
    return table


# ==============================
# MQTT CALLBACKS
# ==============================

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        console.print("[green]Connected successfully to HiveMQ Cloud![/green]")
        client.subscribe(MQTT_TOPIC)
        console.print(f"[cyan]Subscribed to: {MQTT_TOPIC}[/cyan]")
    else:
        console.print(f"[red]Connection failed with code {rc}[/red]")


def on_message(client, userdata, msg):
    global latest_row

    try:
        data = json.loads(msg.payload.decode())
    except Exception:
        return  # Skip invalid JSON

    sensors = data.get("sensors", {})
    meta = data.get("meta", {})

    timestamp = datetime.now(timezone.utc).isoformat()

    row = {
        "timestamp_utc": timestamp,
        "seq": data.get("seq"),
        "temp_c": sensors.get("temp_c"),
        "rh_percent": sensors.get("rh_percent"),
        "light_adc": sensors.get("light_adc"),
        "soil_adc": sensors.get("soil_adc"),
        "water_adc": sensors.get("water_adc"),
        "rain_adc": sensors.get("rain_adc"),
        "pir_motion": sensors.get("pir_motion"),
        "button": sensors.get("button"),
        "distance_cm": sensors.get("distance_cm"),
        "rssi_dbm": meta.get("rssi_dbm"),
    }

    latest_row = row

    # Append to CSV
    pd.DataFrame([row]).to_csv(
        RAW_FILE,
        mode="a",
        header=False,
        index=False
    )


# ==============================
# MAIN
# ==============================

def main():
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    # TLS required for HiveMQ Cloud
    client.tls_set(
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLS
    )

    client.on_connect = on_connect
    client.on_message = on_message

    console.print("[yellow]Connecting to HiveMQ Cloud...[/yellow]")
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

    client.loop_start()

    with Live(build_table(), refresh_per_second=2, console=console) as live:
        try:
            while True:
                live.update(build_table())
        except KeyboardInterrupt:
            console.print("\n[red]Stopping stream...[/red]")
            client.loop_stop()
            client.disconnect()


if __name__ == "__main__":
    main()
