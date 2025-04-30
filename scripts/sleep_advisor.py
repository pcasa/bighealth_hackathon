#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to analyze sleep data from user forms and generate personalized recommendations.
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.data_processing.preprocessing import Preprocessor
from src.core.models.sleep_quality import SleepQualityModel
from core.recommendation.recommendation_engine import SleepRecommendationEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sleep_advisor.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate sleep recommendations from form data')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/recommendations_config.yaml',
        help='Path to recommendations configuration file'
    )
    
    parser.add_argument(
        '--form-data-file', 
        type=str, 
        default='data/raw/sleep_form_data.csv',
        help='File containing the user-submitted sleep form data'
    )
    
    parser.add_argument(
        '--historical-data', 
        type=str, 
        default='data/processed/historical_sleep_data.csv',
        help='File containing historical sleep data'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/recommendations',
        help='Directory to save recommendations'
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

def load_form_data(file_path):
    """Load sleep form data submitted by users."""
    if not os.path.exists(file_path):
        logger.error(f"Form data file not found: {file_path}")
        return None
    
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded {len(data)} records from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading form data: {str(e)}")
        return None

def preprocess_form_data(form_data):
    """Preprocess the form data for analysis."""
    # Create a copy to avoid modifying the original
    processed_data = form_data.copy()
    
    # Convert date columns to datetime
    date_columns = [col for col in processed_data.columns if 'time' in col.lower() or 'date' in col.lower()]
    for col in date_columns:
        if col in processed_data.columns:
            try:
                processed_data[col] = pd.to_datetime(processed_data[col])
            except:
                logger.warning(f"Could not convert column {col} to datetime")
    
    # Calculate sleep efficiency if not present
    if 'sleep_efficiency' not in processed_data.columns:
        # Time in bed
        if 'time_in_bed_hours' not in processed_data.columns and 'bedtime' in processed_data.columns and 'out_bed_time' in processed_data.columns:
            processed_data['time_in_bed_hours'] = (processed_data['out_bed_time'] - processed_data['bedtime']).dt.total_seconds() / 3600
        
        # Estimated sleep duration (time in bed minus sleep onset latency minus time awake)
        if 'sleep_duration_hours' not in processed_data.columns:
            if 'sleep_time' in processed_data.columns and 'wake_time' in processed_data.columns:
                sleep_duration = (processed_data['wake_time'] - processed_data['sleep_time']).dt.total_seconds() / 3600
                
                # Subtract time awake if available
                if 'time_awake_minutes' in processed_data.columns:
                    sleep_duration -= processed_data['time_awake_minutes'] / 60
                
                processed_data['sleep_duration_hours'] = sleep_duration
        
        # Now calculate efficiency
        if 'time_in_bed_hours' in processed_data.columns and 'sleep_duration_hours' in processed_data.columns:
            processed_data['sleep_efficiency'] = processed_data['sleep_duration_hours'] / processed_data['time_in_bed_hours']
            # Cap at 1.0 (100% efficiency)
            processed_data['sleep_efficiency'] = processed_data['sleep_efficiency'].clip(0, 1)
    
    return processed_data

def combine_with_historical_data(new_data, historical_file):
    """Combine new form data with historical data."""
    if not os.path.exists(historical_file):
        logger.info(f"No historical data file found at {historical_file}. Using only new data.")
        return new_data
    
    try:
        historical_data = pd.read_csv(historical_file)
        logger.info(f"Loaded {len(historical_data)} historical records")
        
        # Convert date columns to datetime for proper comparison
        for df in [new_data, historical_data]:
            if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
        
        # Identify duplicate entries (same user and date)
        if 'user_id' in new_data.columns and 'date' in new_data.columns:
            # Create a key for comparison
            new_data['user_date_key'] = new_data['user_id'] + '_' + new_data['date'].dt.strftime('%Y-%m-%d')
            historical_data['user_date_key'] = historical_data['user_id'] + '_' + historical_data['date'].dt.strftime('%Y-%m-%d')
            
            # Remove historical entries that exist in new data
            duplicate_keys = new_data['user_date_key'].values
            historical_data = historical_data[~historical_data['user_date_key'].isin(duplicate_keys)]
            
            # Remove the temporary key
            historical_data = historical_data.drop('user_date_key', axis=1)
            new_data = new_data.drop('user_date_key', axis=1)
        
        # Combine datasets
        combined_data = pd.concat([historical_data, new_data], ignore_index=True)
        logger.info(f"Combined data has {len(combined_data)} records")
        
        # Save updated historical data
        combined_data.to_csv(historical_file, index=False)
        logger.info(f"Updated historical data saved to {historical_file}")
        
        return combined_data
        
    except Exception as e:
        logger.error(f"Error combining with historical data: {str(e)}")
        return new_data

def get_active_user_ids(data, days_threshold=30):
    """Get list of active users (users with data in the past X days)."""
    if data is None or len(data) == 0:
        return []
    
    # Ensure date is in datetime format
    if 'date' in data.columns:
        if not pd.api.types.is_datetime64_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
    
    # Calculate cutoff date
    today = datetime.now()
    cutoff_date = today - timedelta(days=days_threshold)
    
    # Filter to recent data
    recent_data = data[data['date'] >= cutoff_date]
    
    # Get unique user IDs
    active_users = recent_data['user_id'].unique().tolist()
    logger.info(f"Found {len(active_users)} active users in the past {days_threshold} days")
    return active_users

def get_user_sleep_data(user_id, data):
    """Get sleep data for a specific user."""
    user_data = data[data['user_id'] == user_id].copy()
    
    # Ensure data is sorted by date
    if 'date' in user_data.columns:
        if not pd.api.types.is_datetime64_dtype(user_data['date']):
            user_data['date'] = pd.to_datetime(user_data['date'])
        user_data = user_data.sort_values('date')
    
    return user_data

def store_recommendation(user_id, message, output_dir, timestamp=None):
    """Store a recommendation for a user."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create user recommendations file path
    user_file = os.path.join(output_dir, f"{user_id}_recommendations.csv")
    
    # Create new recommendation entry
    recommendation = {
        'user_id': user_id,
        'timestamp': timestamp,
        'message': message
    }
    
    # Create or append to user recommendations file
    if os.path.exists(user_file):
        # Append to existing file
        recommendations_df = pd.read_csv(user_file)
        recommendations_df = pd.concat([recommendations_df, pd.DataFrame([recommendation])], ignore_index=True)
    else:
        # Create new file
        recommendations_df = pd.DataFrame([recommendation])
    
    # Save to file
    recommendations_df.to_csv(user_file, index=False)
    logger.info(f"Stored recommendation for user {user_id}")
    
    # Also save to a combined recommendations file
    combined_file = os.path.join(output_dir, "all_recommendations.csv")
    
    if os.path.exists(combined_file):
        # Append to existing file
        combined_df = pd.read_csv(combined_file)
        combined_df = pd.concat([combined_df, pd.DataFrame([recommendation])], ignore_index=True)
    else:
        # Create new file
        combined_df = pd.DataFrame([recommendation])
    
    combined_df.to_csv(combined_file, index=False)
    
    return True

def main():
    """Main function to analyze sleep form data and generate recommendations."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    create_output_directories(args.output_dir)
    
    # Initialize recommendation engine
    recommendation_engine = SleepRecommendationEngine(args.config)
    
    # Load message history if available
    history_path = os.path.join(os.path.dirname(args.output_dir), 'user_message_history.json')
    if os.path.exists(history_path):
        try:
            recommendation_engine.load_history(history_path)
            logger.info(f"Loaded message history from {history_path}")
        except Exception as e:
            logger.error(f"Error loading message history: {str(e)}")
    
    # Load form data
    form_data = load_form_data(args.form_data_file)
    if form_data is None:
        logger.error("No form data available. Exiting.")
        return
    
    # Preprocess form data
    processed_data = preprocess_form_data(form_data)
    
    # Combine with historical data
    combined_data = combine_with_historical_data(processed_data, args.historical_data)
    
    # Get active users
    active_user_ids = get_active_user_ids(combined_data)
    logger.info(f"Processing {len(active_user_ids)} active users")
    
    # Process each active user
    for user_id in active_user_ids:
        logger.info(f"Processing user {user_id}")
        
        # Get user's sleep data
        user_sleep_data = get_user_sleep_data(user_id, combined_data)
        
        # Check if we have enough data
        if len(user_sleep_data) < 3:
            logger.info(f"Not enough data for user {user_id}. Skipping.")
            continue
        
        try:
            # Analyze progress
            progress_data = recommendation_engine.analyze_progress(user_id, user_sleep_data)
            
            # Generate recommendation
            message = recommendation_engine.generate_recommendation(user_id, progress_data)
            
            # Store recommendation
            store_recommendation(user_id, message, args.output_dir)
            
        except Exception as e:
            logger.error(f"Error processing user {user_id}: {str(e)}")
            continue
    
    # Save updated message history
    try:
        recommendation_engine.save_history(history_path)
        logger.info(f"Saved updated message history to {history_path}")
    except Exception as e:
        logger.error(f"Error saving message history: {str(e)}")
    
    logger.info("Sleep analysis and recommendation generation complete!")

if __name__ == "__main__":
    main()