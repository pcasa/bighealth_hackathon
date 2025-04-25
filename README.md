# Sleep Insights App

A comprehensive sleep analysis application to help people with insomnia and sleep disorders. This application collects sleep data from multiple sources, analyzes patterns, generates personalized recommendations, and provides insights into sleep quality.

# Pytorch
This needs to be installed seperately!

```
uv pip install torch
```

## Project Overview

Sleep Insights App processes two main data sources:
1. Base sleep information input by users (bedtime, wake time, sleep quality ratings, etc.)
2. Sleep and health data from wearable devices (Apple Watch, Google Watch, Fitbit, Samsung Watch)

The application:
- Calculates a sleep score based on multiple factors
- Identifies patterns and trends in sleep behavior
- Detects anomalies in sleep patterns
- Provides personalized recommendations
- Analyzes sleep cycles and quality
- Correlates external factors with sleep quality

## Core Data Structures

### User Profile
- User ID
- Age
- Gender
- Typical sleep schedule (night sleeper, shift worker, etc.)
- Sleep consistency level
- Data entry consistency
- Predefined sleep issues
- Device type (Apple Watch, Google Watch, Fitbit, Samsung Watch)

### Daily Sleep Record
- Date
- User-entered data:
  - Bedtime (when they got into bed)
  - Attempted sleep time
  - Perceived sleep onset time
  - Number of remembered awakenings
  - Estimated time awake during night
  - Final wake time
  - Out of bed time
  - Subjective sleep quality rating (1-10)
  - Notes/factors (optional: stress, alcohol, caffeine, etc.)

### Wearable Device Data
- Device-specific identifiers
- Sleep session data:
  - Detected in-bed time
  - Detected sleep onset
  - Sleep stages (Deep, Light, REM, Awake) with timestamps
  - Movement intensity during sleep
  - Awakenings (count and duration)
  - Detected wake time
  - Total sleep duration
- Heart rate data:
  - Resting heart rate
  - Average heart rate during sleep
  - Heart rate variability
  - Time-series heart rate readings
- Additional metrics (depending on device):
  - Blood oxygen levels
  - Respiration rate
  - Skin temperature

## Sleep Pattern Variations

The system models several sleep pattern types:

### Normal Sleeper Pattern
- Sleep onset within 10-20 minutes of attempting sleep
- 7-8 hours total sleep
- 1-2 brief awakenings
- Consistent bedtime and wake time (±30 min)
- Regular sleep stages with normal proportions
- Sleep efficiency 85-95%

### Insomnia Pattern
- Extended sleep onset (30+ minutes)
- Frequent awakenings (3-7 per night)
- Longer wake periods during the night (10-60+ minutes)
- Reduced total sleep time (4-6 hours)
- Higher heart rate during attempted sleep
- Lower proportion of deep sleep
- Sleep efficiency 60-80%
- Subjective ratings often lower than objective measurements

### Shift Worker Pattern
- Sleep during daytime hours
- More fragmented sleep
- Shorter total duration (5-7 hours)
- Less consistent day-to-day timing
- Reduced REM and deep sleep proportions
- Higher heart rate during sleep
- Sleep efficiency 70-85%

### Oversleeper Pattern
- Extended time in bed (9-11 hours)
- Longer than average total sleep time
- Normal or slightly lower sleep efficiency
- Potentially more REM sleep in later sleep cycles
- Lower average heart rate during sleep

### Highly Variable Sleeper
- Inconsistent bed and wake times (varies by 1-3 hours)
- Variable sleep duration night to night
- Fluctuating sleep quality
- Inconsistent sleep stage distribution
- More irregular heart rate patterns during sleep

### Device-Specific Variation Factors
- Different sampling rates for heart rate
- Varying sensitivity in movement detection
- Different sleep stage classification algorithms
- Precision differences in sleep onset detection
- Different reporting of sleep metrics

## Project Workflows

### Generating Training Data
Uses `scripts/generate_training_data.py` to:
- Generate user profiles with various sleep patterns
- Simulate both user-entered sleep data and wearable device data
- Create realistic patterns including missing entries
- Export data to CSV files in the data/raw directory

### Training Models
Uses `scripts/train_models.py` to:
- Import the CSV data from data/raw
- Preprocess the data and engineer features
- Train multiple models (sleep quality, anomaly detection, pattern analysis)
- Save trained models to the models/ directory with metadata

### Validating Models
Uses `scripts/validate_models.py` to:
- Load trained models from the models/ directory
- Import test data (either a holdout portion or separately generated data)
- Run validation tests and calculate performance metrics
- Generate validation reports with charts and statistics

### n8n Integration
The n8n_workflows/ directory contains workflows that:
- Accept input data (either manual entries or wearable data)
- Call Python API endpoints to process the data
- Run the data through trained models
- Log outputs with user IDs
- Return results or recommendations to users

## Advanced Features

### Transfer Learning Module
Adapts pre-trained sleep models to new users with minimal data by:
- Implementing domain adaptation methods
- Supporting fine-tuning with minimal user data
- Preserving privacy while adapting models

### Batch Processing
Efficiently handles large historical data imports by:
- Processing data in configurable batch sizes
- Supporting parallel processing when possible
- Including checkpointing to resume interrupted processing

### Data Drift Detection
Monitors input data for changes that might affect model performance:
- Implements both univariate and multivariate drift detection
- Sets configurable alerting thresholds
- Tracks feature importance in drift detection
- Provides reports on which features are drifting

### Model Cards
Comprehensive documentation for each model including:
- Model purpose and intended use cases
- Training data characteristics and limitations
- Performance metrics across different user groups
- Known limitations and biases
- Version history and recommendations for retraining

### Docker Deployment
Containerized setup for easy local deployment:
- Creates reproducible environment with all dependencies
- Configures network settings for n8n integration
- Sets up volume mounts for persistent data and models
- Supports scaling for batch processing

## Technology Stack

- Python for core implementation
- PyTorch for machine learning components
- UV for package management (instead of pip)
- n8n for workflow automation
- Docker for containerized deployment

## Getting Started

1. Clone this repository
2. Install UV package manager
3. Install dependencies: `uv pip install -r requirements.txt`
4. Generate training data: `python scripts/generate_training_data.py`
5. Train models: `python scripts/train_models.py`
6. Validate models: `python scripts/validate_models.py`
7. Run Docker: `docker-compose up`
8. Import n8n workflows from n8n_workflows/ directory

## Directory Structure

```
sleep-insight-app/
│
├── README.md                           # Project documentation
├── requirements.txt                    # Dependencies for UV installation
├── Dockerfile                          # Docker container configuration
├── docker-compose.yml                  # Multi-container Docker setup
│
├── config/                             # Configuration files
├── data/                               # Data storage
├── models/                             # Trained model storage
│   └── model_cards/                    # Documentation of models
│
├── src/                                # Source code
│   ├── data_generation/                # Data generation modules
│   ├── data_processing/                # Data processing modules
│   ├── models/                         # Model implementation
│   ├── monitoring/                     # Monitoring systems
│   ├── utils/                          # Utility functions
│   └── api/                            # API integration
│
├── notebooks/                          # Jupyter notebooks
├── scripts/                            # Executable scripts
└── n8n_workflows/                      # n8n workflow definitions
```