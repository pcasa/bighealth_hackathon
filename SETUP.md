# Sleep Insights App Setup Guide

This guide explains how to set up the Sleep Insights App for development or demonstration purposes.

## Overview

The Sleep Insights App requires several setup steps before it's ready for development:

1. **Data Generation**: Create synthetic sleep data for training and testing
2. **Model Training**: Train machine learning models on the generated data
3. **Enhanced Demo**: Generate a more comprehensive demo dataset with user profiles
4. **Data Analysis**: Run various analytics tools on the enhanced dataset
5. **API Server**: Start the API server for development

## Setup Options

There are three ways to set up the Sleep Insights App:

1. **Shell Script** (`setup.sh`): A simple Bash script that runs all steps in sequence
2. **Python Script** (`setup.py`): A more flexible Python script with command-line options
3. **Docker Setup** (`docker-setup.sh`): Uses Docker containers for setup and development

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Docker & Docker Compose (for Docker setup only)

## Option 1: Using the Shell Script

The shell script is the simplest option for setting up the app.

```bash
# Make the script executable
chmod +x setup.sh

# Run the setup
./setup.sh
```

The script will:
- Create necessary directories
- Install dependencies from requirements.txt
- Generate training data
- Train machine learning models
- Run the enhanced demo
- Analyze the demo data
- Generate sample user insights
- Offer to start the API server

## Option 2: Using the Python Script

The Python script offers more flexibility with command-line options.

```bash
# Run the full setup
python setup.py

# Or customize the setup with options
python setup.py --skip-training --start-api

# To see all available options
python setup.py --help
```

Available options:
- `--skip-training`: Skip the model training step (use if you already have models)
- `--skip-demo`: Skip running the enhanced demo
- `--skip-analysis`: Skip the analysis steps
- `--start-api`: Start the API server after setup

## Option 3: Using Docker Setup

The Docker setup script uses containers for isolation and reproducibility.

```bash
# Make the script executable
chmod +x docker-setup.sh

# Run the Docker setup
./docker-setup.sh
```

This will:
- Create necessary directories and Docker configuration files
- Build a Docker image with all dependencies
- Run each setup step in separate containers
- Mount volumes to persist data between container runs
- Offer to start the API server in a container

## Directory Structure

After running any of the setup scripts, the following directory structure will be created:

```
sleep-insights-app/
├── data/
│   ├── raw/                    # Training data
│   ├── processed/              # Processed data
│   ├── enhanced_demo/          # Enhanced demo data
│   │   ├── data/               # Demo user and sleep data
│   │   ├── visualizations/     # Demo visualizations
│   │   └── recommendations/    # Demo recommendations
│   └── logs/                   # Data processing logs
├── models/                     # Trained ML models
│   ├── sleep_quality_model.pt  # Sleep quality prediction model
│   └── model_cards/            # Model documentation
├── reports/                    # Analysis reports
│   ├── enhanced_analysis/      # Demographic analysis reports
│   └── user_insights/          # Sample user insights
└── logs/                       # Setup and runtime logs
```

## Next Steps

After completing the setup, you can:

1. **Start the API server**:
   ```bash
   python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   Or with Docker:
   ```bash
   docker-compose -f docker-compose.setup.yml up api
   ```

2. **Explore the API documentation**:
   - Open `http://localhost:8000/docs` in your browser

3. **View generated reports**:
   - Check the demo visualizations in `data/enhanced_demo/visualizations/`
   - Read the analysis reports in `reports/enhanced_analysis/`
   - Review sample user insights in `reports/user_insights/`

4. **Start development**:
   - Modify the API endpoints in `src/api/`
   - Enhance the recommendation engine in `src/core/recommendation/`
   - Improve sleep score calculation in `src/core/models/`

## Troubleshooting

If you encounter issues during setup:

1. **Missing dependencies**:
   - Ensure you have Python 3.9+ installed
   - Check if all dependencies in `requirements.txt` are installed
   - Try installing PyTorch separately: `pip install torch`

2. **Model training failures**:
   - The app will use fallback methods if model training fails
   - Check the training logs in `logs/train_models.log`
   - Try increasing the memory limit if using Docker

3. **API server not starting**:
   - Verify that port 8000 is not in use by another application
   - Check if the `src/api/main.py` file exists and is properly formatted
   - Ensure the working directory is set correctly

4. **No data generated**:
   - Check file permissions in the data directory
   - Review generation logs in `logs/data_generation.log`
   - Try running the data generation script manually

## Additional Resources

- API Documentation: Available at `http://localhost:8000/docs`
- Code Documentation: See docstrings in source files
- Model Information: Review model cards in `models/model_cards/`