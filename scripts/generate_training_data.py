# scripts/generate_training_data.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to generate synthetic training data for the Sleep Insights App.
This generates both user profiles and corresponding sleep data patterns.
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation.user_generator import UserGenerator
from src.data_generation.sleep_data_generator import SleepDataGenerator
from src.data_generation.wearable_data_generator import WearableDataGenerator
from src.data_generation.external_factors_generator import ExternalFactorsGenerator
from src.data_generation.sleep_data_generator import SleepDataGenerator
from src.data_generation.wearable_data_generator import WearableDataGenerator


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/generate_training_data.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate synthetic training data for Sleep Insights App')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='src/config/data_generation_config.yaml',
        help='Path to data generation configuration file'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/raw',
        help='Directory to save generated data'
    )
    
    parser.add_argument(
        '--num-users', 
        type=int, 
        default=None,
        help='Number of users to generate (overrides config file)'
    )
    
    parser.add_argument(
        '--days-per-user', 
        type=int, 
        default=None,
        help='Number of days of data per user (overrides config file)'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def create_output_directories(output_dir):
    """Create output directories if they don't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

def main():
    """Main function to generate training data."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.num_users is not None:
        config['user_profiles']['count'] = args.num_users
    
    if args.days_per_user is not None:
        # Ensure days_per_user exists in config
        if 'sleep_data_generation' not in config:
            config['sleep_data_generation'] = {}
        config['sleep_data_generation']['days_per_user'] = args.days_per_user
    
    # Create output directories
    create_output_directories(args.output_dir)
    
    # Initialize generators
    user_generator = UserGenerator(config_path=args.config)
    sleep_data_generator = SleepDataGenerator(config_path='src/config/data_generation_config.yaml')
    wearable_data_generator = WearableDataGenerator(config_path='src/config/data_generation_config.yaml', 
                                          device_config_path='src/config/device_profiles.yaml')

    external_factors_generator = ExternalFactorsGenerator()
    
    # Generate user profiles
    logger.info(f"Generating {config['user_profiles']['count']} user profiles...")
    users_df = user_generator.generate_users()
    
    # Save user profiles
    users_df.to_csv(os.path.join(args.output_dir, 'users.csv'), index=False)
    logger.info(f"Saved {len(users_df)} user profiles to {os.path.join(args.output_dir, 'users.csv')}")
    
    # Generate sleep data
    logger.info("Generating sleep data for users...")
    sleep_data_df = sleep_data_generator.generate_sleep_data(users_df)
    sleep_data_df.to_csv(os.path.join(args.output_dir, 'sleep_data.csv'), index=False)
    logger.info(f"Saved {len(sleep_data_df)} sleep records to {os.path.join(args.output_dir, 'sleep_data.csv')}")
    
    # Generate wearable data
    logger.info("Generating wearable data...")
    wearable_data_df = wearable_data_generator.generate_wearable_data(sleep_data_df, users_df)
    wearable_data_df.to_csv(os.path.join(args.output_dir, 'wearable_data.csv'), index=False)
    logger.info(f"Saved {len(wearable_data_df)} wearable records to {os.path.join(args.output_dir, 'wearable_data.csv')}")
    
    # Generate external factors data
    logger.info("Generating external factors data...")
    # Get date range from sleep data
    start_date = pd.to_datetime(sleep_data_df['date']).min()
    end_date = pd.to_datetime(sleep_data_df['date']).max()
    
    # Generate weather data
    weather_data = external_factors_generator.generate_weather_data(start_date, end_date)
    
    # Generate activity data
    activity_data = external_factors_generator.generate_activity_data(sleep_data_df, users_df)
    
    # Combine external data
    external_factors_df = pd.merge(
        weather_data,
        activity_data,
        on='date',
        how='outer'
    )
    
    external_factors_df.to_csv(os.path.join(args.output_dir, 'external_factors.csv'), index=False)
    logger.info(f"Saved {len(external_factors_df)} external factor records to {os.path.join(args.output_dir, 'external_factors.csv')}")
    
    logger.info("Data generation complete!")

if __name__ == "__main__":
    main()