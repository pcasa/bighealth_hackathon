# sleep_quality Model Card

**Version:** 1.0.0
**Created:** 2025-04-26 11:31:31

## Description
Model for sleep_quality

## Model Information
**Type:** lstm
**Input Features:**
- sleep_duration_hours
- sleep_efficiency
- awakenings_count
- total_awake_minutes
- deep_sleep_percentage
- rem_sleep_percentage
- sleep_onset_latency_minutes
- heart_rate_variability
- average_heart_rate
**Output:** Sleep Quality Score

## Performance Metrics
- **final_train_loss:** 0.17845587238308294
- **final_val_loss:** 0.528221070766449
- **features_used:** ['sleep_duration_hours', 'sleep_efficiency', 'awakenings_count', 'total_awake_minutes', 'deep_sleep_percentage', 'rem_sleep_percentage', 'sleep_onset_latency_minutes', 'heart_rate_variability', 'average_heart_rate']

## Limitations
- Limited to the patterns in the training data
- May not generalize to users with rare sleep disorders
- Assumes regular data collection from wearable devices

## Intended Use
- Predict sleep quality based on user and wearable data
- Identify sleep patterns and anomalies
- Provide recommendations for improved sleep

## Training Data Characteristics
- **Number of Users:** 500
- **Days per User:** 68
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
