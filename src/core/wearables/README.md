# Wearable Data Transformer

A set of transformers for converting wearable device sleep data into a standardized format for use with the Sleep Insights application.

## Overview

This package provides transformers for the most common wearable devices that track sleep data:

- Apple Watch / Apple Health
- Fitbit
- Samsung Galaxy Watch / Samsung Health
- Google Pixel Watch / Google Fit

Each transformer converts device-specific sleep data format into a standardized format used by our sleep analysis system. The transformers handle different units, field names, and data structures to produce a consistent output.

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Requirements

- Python 3.8+
- pandas
- numpy
- pydantic

## Usage

### Basic Usage

```python
from wearable_transformer_manager import WearableTransformerManager

# Initialize the transformer manager
transformer_manager = WearableTransformerManager()

# Transform Apple Watch data
transformed_data = transformer_manager.transform_data(
    apple_watch_data,  # DataFrame or list of dictionaries
    'apple_watch',     # Device type
    users_df           # Optional: DataFrame with user information
)

# Save transformed data
transformed_data.to_csv('transformed_sleep_data.csv', index=False)
```

### Batch Transformation

You can transform data from multiple device types at once:

```python
# Data from multiple device types
device_data = {
    'apple_watch': apple_watch_data,
    'fitbit': fitbit_data,
    'samsung_watch': samsung_watch_data,
    'google_watch': google_watch_data
}

# Batch transform and combine
combined_data = transformer_manager.batch_transform(device_data, users_df)
```

### Adding Custom Transformers

You can add support for additional wearable devices by creating a custom transformer:

```python
from wearable_base_transformer import BaseWearableTransformer

# Create a custom transformer
class MyDeviceTransformer(BaseWearableTransformer):
    def __init__(self):
        super().__init__()
        self.device_type = "my_device"
    
    def _transform_data(self, data_df, users_df=None):
        # Implement device-specific transformation logic here
        # ...
        return transformed_df

# Register the custom transformer with the manager
transformer_manager = WearableTransformerManager()
transformer_manager.add_custom_transformer("my_device", MyDeviceTransformer())
```

## Standardized Output Format

The transformers convert device-specific data to a standardized format with the following fields:

| Field Name | Type | Description |
|------------|------|-------------|
| `user_id` | string | Unique identifier for the user |
| `date` | string | Date of the sleep record (YYYY-MM-DD) |
| `device_type` | string | Type of wearable device |
| `device_bedtime` | string | Timestamp when user went to bed (YYYY-MM-DD HH:MM:SS) |
| `device_sleep_onset` | string | Timestamp when user fell asleep (YYYY-MM-DD HH:MM:SS) |
| `device_wake_time` | string | Timestamp when user woke up (YYYY-MM-DD HH:MM:SS) |
| `device_sleep_duration` | float | Total sleep duration in hours |
| `time_in_bed_hours` | float | Total time in bed in hours |
| `deep_sleep_percentage` | float | Percentage of time in deep sleep (0-1) |
| `light_sleep_percentage` | float | Percentage of time in light sleep (0-1) |
| `rem_sleep_percentage` | float | Percentage of time in REM sleep (0-1) |
| `awake_percentage` | float | Percentage of time awake during sleep period (0-1) |
| `sleep_efficiency` | float | Sleep efficiency (sleep time / time in bed) (0-1) |
| `awakenings_count` | int | Number of times the user woke up during sleep |
| `average_heart_rate` | float | Average heart rate during sleep |
| `min_heart_rate` | float | Minimum heart rate during sleep |
| `max_heart_rate` | float | Maximum heart rate during sleep |
| `heart_rate_variability` | float | Heart rate variability during sleep |
| `blood_oxygen` | float | Average blood oxygen level (SpO2) during sleep |

## Input Data Formats

### Apple Watch Data Format

Apple Watch data can be provided in several formats:

1. Apple Health Export XML
2. Pre-processed CSV/DataFrame with fields like:
   - `bedtime`, `wake_time`
   - `sleep_duration_seconds`, `deep_sleep_seconds`, etc.
   - `avg_heart_rate`, `hrv_ms`, etc.

### Fitbit Data Format

Fitbit data can be provided in several formats:

1. Fitbit JSON export
2. Pre-processed CSV/DataFrame with fields like:
   - `start_time`, `end_time`
   - `minutes_in_bed`, `minutes_asleep`
   - `deep_minutes`, `light_minutes`, `rem_minutes`, etc.
   - `average_hr`, `hr_variability`, etc.

### Samsung Watch Data Format

Samsung data can be provided in several formats:

1. Samsung Health CSV export
2. Samsung Health JSON export
3. Pre-processed CSV/DataFrame with fields like:
   - `sleep_start`, `sleep_end`
   - `total_sleep_time` (seconds)
   - `deep_sleep`, `light_sleep`, `rem_sleep`, etc.
   - `avg_hr`, `hrv`, `spo2_avg`, etc.

### Google Watch Data Format

Google data can be provided in several formats:

1. Google Fit JSON export
2. Pre-processed CSV/DataFrame with fields like:
   - `sleep_start_time`, `sleep_end_time`
   - `sleep_duration` (ms)
   - `deep_sleep_duration`, `light_sleep_duration`, etc.
   - `average_heart_rate`, `heart_rate_variability`, etc.

## Running the Example

The repository includes an example script that demonstrates how to use the transformers:

```bash
python usage_example.py
```

This will:
1. Create sample data for different device types
2. Transform the data to the standardized format
3. Combine data from all devices
4. Save the transformed data to the `data/transformed` directory

## Adding Support for New Devices

To add support for a new wearable device:

1. Create a new transformer class that inherits from `BaseWearableTransformer`
2. Implement the `_transform_data` method to convert device-specific data to the standardized format
3. Register the new transformer with the `WearableTransformerManager`

## Error Handling

The transformers include robust error handling to deal with:

- Missing fields
- Different data formats (seconds, minutes, hours)
- Inconsistent field names
- Invalid values

If a record cannot be transformed properly, it is skipped with an error message, allowing the transformation to continue with other records.

## Logging

The transformers use Python's logging module to provide detailed information about the transformation process. You can configure the logging level as needed:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

To see more detailed debug information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

Contributions are welcome! If you'd like to add support for additional devices or improve the existing transformers, please submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.