# Sleep Quality Model Card

**Version:** 1.0.0
**Created:** April 25, 2025

## Description
The Sleep Quality Model predicts sleep efficiency scores and calculates comprehensive sleep quality metrics based on user sleep patterns, wearable device data, and environmental factors.

## Model Information
**Type:** LSTM Neural Network
**Input Features:**
- sleep_duration_hours
- sleep_efficiency
- time_in_bed_hours
- sleep_onset_latency_minutes
- awakenings_count
- total_awake_minutes
- deep_sleep_percentage
- rem_sleep_percentage
- light_sleep_percentage
- heart_rate_variability
- average_heart_rate
- min_heart_rate
- max_heart_rate
- movement_intensity
- bedtime_consistency
- waketime_consistency
- duration_consistency
- day_of_week_sin
- day_of_week_cos
- hour_sin
- hour_cos
- month_sin
- month_cos

**Output:** Sleep Quality Score (0-100)

## Performance Metrics
- **RMSE:** 0.068
- **MAE:** 0.052
- **R²:** 0.873

## Limitations
- Requires at least 7 days of consecutive sleep data for accurate predictions
- May not be accurate for users with highly irregular sleep patterns
- Not clinically validated for sleep disorder diagnosis
- Assumes regular data collection from wearable devices
- Precision varies based on wearable device type

## Intended Use
- Predict sleep quality based on user-reported and wearable device data
- Identify sleep patterns and trends
- Support personalized sleep recommendations
- Track sleep quality improvements over time
- Correlate sleep quality with environmental and behavioral factors

## Training Data Characteristics
- **Number of Users:** 100
- **Days per User:** 90
- **Demographics:** Simulated diverse population
- **Sleep Patterns:**
  - Normal Sleeper Pattern
  - Insomnia Pattern
  - Shift Worker Pattern
  - Oversleeper Pattern
  - Highly Variable Sleeper

## Hyperparameters
- **hidden_size:** 128
- **num_layers:** 2
- **dropout:** 0.2
- **learning_rate:** 0.001
- **batch_size:** 64
- **epochs:** 50

## Ethical Considerations
- Not intended to replace medical diagnosis or treatment
- Privacy considerations for handling sensitive sleep data
- Should be used as a supportive tool rather than a definitive assessment