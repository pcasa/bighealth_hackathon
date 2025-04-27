# sleep_quality Model Card

**Version:** 1.0.0
**Created:** 2025-04-27 17:22:52

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
- **final_train_loss:** 0.2059290714403001
- **final_val_loss:** 0.46653300523757935
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
- **Days per User:** 86
- **Demographics:** Simulated diverse population
- **Sleep Patterns:**
  - Normal Sleeper Pattern
  - Insomnia Pattern
  - Shift Worker Pattern
  - Oversleeper Pattern
  - Highly Variable Sleeper

## Predictive Capabilities

### Sleep Quality Predictions
- Sleep efficiency trends over time based on user-reported data
- Probability of experiencing insomnia on upcoming nights
- Estimated sleep quality for the coming night based on recent patterns
- Expected subjective ratings if certain behaviors are modified

### Pattern Recognition
- Sleep consistency patterns and how they affect overall sleep quality
- Identification of insomnia triggers based on correlations in the data
- Detection of severe insomnia episodes before they become chronic
- Recognition of improvement trends even when subjective perception lags

### Personalized Insights
- Most effective sleep window based on recorded sleep efficiency data
- Optimal bedtime and wake time for maximum sleep quality
- Personal threshold for sleep onset latency that predicts a good night's sleep
- Impact of awakenings on overall sleep quality for the individual

### Behavior Impact Assessment
- Expected improvement if sleep consistency is increased
- Predicted benefits of reducing time in bed for those with extended wake times
- Forecasted sleep efficiency changes with various interventions
- Projected recovery time after periods of severe insomnia

### Long-term Predictions
- Risk of developing chronic insomnia based on current patterns
- Expected timeline for improvement with consistent sleep practices
- Likelihood of relapse based on pattern recognition
- Long-term sleep health trajectory with and without intervention

## Hyperparameters
- **hidden_size:** 128
- **num_layers:** 2
- **dropout:** 0.2
- **learning_rate:** 0.001
- **batch_size:** 64
- **epochs:** 50
