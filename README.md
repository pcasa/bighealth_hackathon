# Sleep Insights App

A comprehensive sleep analysis application to help people with insomnia and sleep disorders. This application collects sleep data from multiple sources, analyzes patterns, generates personalized recommendations, and provides insights into sleep quality.

## Project based on UV

Combines pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more.
See more at https://docs.astral.sh/uv/
```
pip install uv
```

## Pytorch

**This needs to be installed seperately!**

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

# Sleep Insights App - Recommendation System

The Sleep Insights App includes a personalized recommendation system that analyzes sleep patterns from user-submitted form data and provides tailored guidance to help users improve their sleep quality.

## Form Data Collection

Our system primarily relies on daily sleep form submissions that capture:

- Time the user went to bed
- Time the user attempted to fall asleep
- Number of awakenings during the night
- Time spent awake during the night
- Wake-up time
- Time the user got out of bed
- Subjective sleep quality rating (1-10)
- Whether the user slept at all (for severe insomnia tracking)

This user-reported data provides valuable insights without requiring wearable devices, making the app accessible to all users.

## Recommendation Features

### Progress Tracking
- Analyzes sleep efficiency trends over time
- Identifies improvement and regression patterns
- Calculates consistency in sleep timing and tracking
- Detects severe insomnia patterns (multiple nights without sleep)

### Personalized Messaging
- Adapts message tone based on user's progress
- Provides encouragement during regression periods
- Celebrates improvements and achievements
- Offers actionable suggestions relevant to detected patterns
- Provides specialized support for severe insomnia cases

### Varied Content
- Maintains engagement by avoiding repetitive messages
- References specific sleep metrics to provide context
- Progressively introduces sleep hygiene concepts

## How Recommendations Work

The recommendation engine follows this process:

1. **Data Collection**: Sleep form data is submitted by users
2. **Data Processing**: The system preprocesses and normalizes the form data
3. **Progress Analysis**: Recent sleep trends are analyzed (typically 7-14 days)
4. **Pattern Identification**: Sleep patterns and consistency are evaluated
5. **Message Selection**: An appropriate recommendation is selected based on:
   - Current sleep trend (improving, regressing, stable)
   - Presence of no-sleep nights (severe insomnia detection)
   - User's tracking consistency
   - Previously sent messages (to avoid repetition)
6. **Personalization**: The message is personalized with the user's specific metrics
7. **Delivery**: Recommendations are delivered through the user interface

## Sample Recommendations

Different messages are provided based on the user's current status:

**For Improving Sleep:**
> "Fantastic progress! Your self-reported sleep quality has improved to 7/10. Whatever you're doing is working - keep it up!"

**For Regressing Sleep:**
> "We all have ups and downs with sleep. Your recent ratings show a temporary dip, but remember you've achieved better sleep before! Consider returning to the habits that worked well previously."

**For Sleepless Nights:**
> "You've reported 3 nights without sleep recently. This is a significant pattern that deserves professional attention - please consider consulting with a healthcare provider specializing in sleep."

## Integration with n8n

The recommendation system integrates with n8n workflows to:
1. Process incoming sleep form submissions
2. Store the data in a centralized repository
3. Generate recommendations
4. Deliver recommendations through preferred channels

## Configuration

The recommendation system can be configured through:
- `config/recommendations_config.yaml`: General settings for analysis and delivery
- `config/message_templates.json`: The library of recommendation messages

## Using the Recommendation API

To generate recommendations programmatically:

```python
from src.models.recommendation_engine import SleepRecommendationEngine

# Initialize the engine
recommendation_engine = SleepRecommendationEngine()

# Analyze a user's progress
progress_data = recommendation_engine.analyze_progress(user_id, sleep_data)

# Generate a personalized recommendation
message = recommendation_engine.generate_recommendation(user_id, progress_data)

```

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