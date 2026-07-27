# Smart Home Digital Twin

## MSc Data Science Dissertation | University of Aberdeen

An end-to-end IoT and Data Science project that integrates real-time sensor monitoring, time-series analytics, anomaly detection, forecasting, and a digital twin framework for indoor climate and energy monitoring.

## Project Overview

This project develops a Smart Home Digital Twin using an ESP32-based sensing system that continuously collects environmental telemetry and streams data through MQTT to cloud infrastructure for storage, monitoring, and analysis.

The system combines:

- IoT sensing
- Real-time data streaming
- PostgreSQL data storage
- Grafana dashboards
- Mobile monitoring
- Time-series analytics
- Predictive monitoring
- Forecasting models

## System Architecture

ESP32 Sensors

↓

MQTT (HiveMQ)

↓

Python Ingestion Service

↓

PostgreSQL Database

↓

Grafana Dashboard & Mobile App

↓

Time-Series Analytics & Forecasting

## Technologies Used

### Programming

- Python
- Swift

### Data Science

- Pandas
- NumPy
- SciPy
- Statsmodels
- Matplotlib
- Seaborn

### IoT & Cloud

- ESP32
- MQTT
- HiveMQ

### Data Infrastructure

- PostgreSQL
- Grafana

## Features

### Real-Time Monitoring

- Live MQTT telemetry streaming
- Grafana dashboard visualisation
- Mobile monitoring application

### Time-Series Analytics

- Descriptive statistics
- Seasonal decomposition
- Correlation analysis
- Lagged cross-correlation analysis

### Sensor Reliability Analysis

- Residual-based anomaly detection
- RSSI communication monitoring
- Sensor health monitoring
- Event-based signal analysis

### Forecasting

- ARIMA forecasting
- Baseline model comparison
- Performance evaluation using MAE, RMSE and R²

## Key Findings

- Strong daily seasonality was observed in light intensity and temperature.
- Peak correlation between light intensity and temperature occurred approximately 5–6 hours later, indicating delayed thermal response.
- RSSI anomaly detection identified periods of communication degradation.
- Rule-based predictive monitoring successfully identified potential sensor instability.
- The project demonstrates a practical implementation of a simplified Digital Twin framework for smart-home environments.

## Repository Structure

```
smart-home-digital-twin/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── live_stream.py
│   ├── final.py
│   ├── forecasting_models.py
│   ├── predictive_maintenance.py
│   ├── button_press_analysis.py
│   └── pir_motion_analysis.py
│
├── thesis/
│   ├── MSc_Project_Thesis.pdf
│   └── Supplementary_Material.pdf
│
├── images/
│   ├── architecture.png
│   ├── grafana_dashboard.png
│   └── mobile_app.png
│
└── data/
    └── sample_data.csv
```

## Results

The project demonstrates how IoT telemetry can be transformed into actionable insights using Data Science, enabling:

- Environmental monitoring
- Communication health assessment
- Sensor anomaly detection
- Early predictive maintenance
- Digital twin development

## Dissertation

## Author

Nazerke Dalabayeva

Field Engineer transitioning into Data Science, with interests in IoT Analytics, Time-Series Forecasting, Machine Learning, Cloud Technologies, and Digital Twin Systems.
