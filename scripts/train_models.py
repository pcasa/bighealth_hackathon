#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to train machine learning models for the Sleep Insights App.
This handles preprocessing, feature engineering, and model training.
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.data_processing.preprocessing import Preprocessor
from src.core.data_processing.feature_engineering import FeatureEngineering
from src.core.models.sleep_quality import SleepQualityModel
from src.core.models.transfer_learning import TransferLearning

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_models.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train machine learning models for Sleep Insights App')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/model_config.yaml',
        help='Path to model configuration file'
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data/raw',
        help='Directory containing the raw data'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Proportion of data to use for testing'
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
    Path(os.path.join(output_dir, 'model_cards')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'user_models')).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

def load_data(data_dir):
    """Load data from CSV files."""
    try:
        users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
        sleep_data_df = pd.read_csv(os.path.join(data_dir, 'sleep_data.csv'))
        
        logger.info(f"Loaded {len(users_df)} users")
        logger.info(f"Loaded {len(sleep_data_df)} sleep records")
        
        # Attempt to load wearable data
        try:
            wearable_data_df = pd.read_csv(os.path.join(data_dir, 'wearable_data.csv'))
            logger.info(f"Loaded {len(wearable_data_df)} wearable records")
        except Exception as e:
            logger.warning(f"Could not load wearable data: {str(e)}")
            wearable_data_df = None
        
        # Attempt to load external factors data
        try:
            external_factors_df = pd.read_csv(os.path.join(data_dir, 'external_factors.csv'))
            logger.info(f"Loaded {len(external_factors_df)} external factor records")
        except Exception as e:
            logger.warning(f"Could not load external factors data: {str(e)}")
            external_factors_df = None
            
        return users_df, sleep_data_df, wearable_data_df, external_factors_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def add_missing_features(data):
    """Add missing features required by the model."""
    
    # 1. Add age_normalized if missing
    if 'age' in data.columns and 'age_normalized' not in data.columns:
        data['age_normalized'] = data['age'] / 100.0
        logger.info("Added missing feature: age_normalized")
    
    # 2. Add profession features if missing
    if 'profession' in data.columns and 'profession_category' not in data.columns:
        from src.utils.constants import profession_categories
        
        def categorize_profession(profession):
            for category, keywords in profession_categories.items():
                if isinstance(profession, str) and any(keyword.lower() in profession.lower() for keyword in keywords):
                    return category
            return "other"
        
        data['profession_category'] = data['profession'].apply(categorize_profession)
        logger.info("Added missing feature: profession_category")
        
        # Add profession one-hot encoding
        prof_categories = ['healthcare', 'tech', 'service', 'education', 'office', 'other']
        for category in prof_categories:
            col_name = f'profession_{category}'
            if col_name not in data.columns:
                data[col_name] = (data['profession_category'] == category).astype(float)
                logger.info(f"Added missing feature: {col_name}")
    
    # 3. Add season features if missing
    if 'date' in data.columns:
        # Extract month if needed
        if 'month' not in data.columns:
            data['month'] = pd.to_datetime(data['date']).dt.month
            logger.info("Added missing feature: month")
        
        # Add season if needed
        if 'season' not in data.columns:
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                else:  # 9, 10, 11
                    return 'Fall'
            
            data['season'] = data['month'].apply(get_season)
            logger.info("Added missing feature: season")
        
        # Add season one-hot encoding
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        for season in seasons:
            col_name = f'season_{season}'
            if col_name not in data.columns:
                data[col_name] = (data['season'] == season).astype(float)
                logger.info(f"Added missing feature: {col_name}")
    
    return data

def ensure_numeric_data(data_df, feature_columns):
    """Ensure all columns used for training contain only numeric data."""
    modified_df = data_df.copy()
    problematic_columns = []
    
    for col in feature_columns:
        if col in modified_df.columns:
            # Check if column contains any non-numeric values
            non_numeric = False
            
            # For object dtype columns, check if they can be converted to numeric
            if modified_df[col].dtype == 'object':
                non_numeric = True
                print(f"Column {col} has object dtype. Attempting conversion...")
                
                # Try to convert to numeric
                try:
                    modified_df[col] = pd.to_numeric(modified_df[col], errors='coerce')
                    non_numeric = False
                    print(f"Successfully converted {col} to numeric")
                except Exception as e:
                    print(f"Failed to convert {col}: {str(e)}")
                    problematic_columns.append(col)
                    continue
            
            # Even for numeric dtypes, check for NaN or inf values
            if modified_df[col].isna().any() or (np.isinf(modified_df[col]).any() if np.issubdtype(modified_df[col].dtype, np.number) else False):
                print(f"Column {col} has NaN or inf values. Filling with zeros...")
                modified_df[col] = modified_df[col].fillna(0)
                # Replace inf with large values
                modified_df[col] = modified_df[col].replace([np.inf, -np.inf], 0)
    
    # Drop columns that couldn't be converted to numeric
    if problematic_columns:
        print(f"Dropping problematic columns: {problematic_columns}")
        modified_df = modified_df.drop(columns=problematic_columns)
    
    # Print a sample of the data to verify
    for col in modified_df.columns:
        if col in feature_columns:
            print(f"Column {col} now has dtype: {modified_df[col].dtype}")
            if modified_df[col].dtype == 'object':
                print(f"WARNING: Column {col} is still object type")
    
    return modified_df

def main():
    """Main function to train models."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    create_output_directories(args.output_dir)
    
    # Load data
    users_df, sleep_data_df, wearable_data_df, external_factors_df = load_data(args.data_dir)

    # Ensure all dataframes have a 'user_id' column set as string
    users_df['user_id'] = users_df['user_id'].astype(str)
    sleep_data_df['user_id'] = sleep_data_df['user_id'].astype(str)
    if wearable_data_df is not None:
        wearable_data_df['user_id'] = wearable_data_df['user_id'].astype(str)
    
    # Check for required columns
    required_cols = ['user_id', 'date']
    for col in required_cols:
        if col not in sleep_data_df.columns:
            logger.error(f"Required column '{col}' missing from sleep_data_df")
            
            # If 'date' is missing but there's a similar column, rename it
            if col == 'date':
                date_like_cols = [c for c in sleep_data_df.columns if 'date' in c.lower() or 'day' in c.lower()]
                if date_like_cols:
                    logger.info(f"Found potential date column: {date_like_cols[0]}. Renaming to 'date'.")
                    sleep_data_df['date'] = sleep_data_df[date_like_cols[0]]
                else:
                    # Create a synthetic date column if no date column exists
                    logger.warning("Creating synthetic date column")
                    base_date = datetime(2024, 1, 1)
                    user_dates = {}
                    
                    for user_id in sleep_data_df['user_id'].unique():
                        user_rows = sleep_data_df[sleep_data_df['user_id'] == user_id]
                        user_dates[user_id] = [base_date + timedelta(days=i) for i in range(len(user_rows))]
                    
                    sleep_data_df['date'] = sleep_data_df.apply(
                        lambda row: user_dates[row['user_id']].pop(0), axis=1
                    )
            else:
                raise ValueError(f"Missing required column: {col}")
    
    # Ensure date is in datetime format
    try:
        sleep_data_df['date'] = pd.to_datetime(sleep_data_df['date'])
    except Exception as e:
        logger.error(f"Error converting date column to datetime: {str(e)}")
        # Try different formats if the standard conversion fails
        for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']:
            try:
                sleep_data_df['date'] = pd.to_datetime(sleep_data_df['date'], format=date_format)
                logger.info(f"Successfully converted date using format: {date_format}")
                break
            except:
                continue
    
    # Log original user counts for debugging
    original_user_count = sleep_data_df['user_id'].nunique()
    logger.info(f"Original data: {len(sleep_data_df)} records, {original_user_count} unique users")
    
    # Convert date columns in other dataframes if they exist
    if wearable_data_df is not None and 'date' in wearable_data_df.columns:
        wearable_data_df['date'] = pd.to_datetime(wearable_data_df['date'], errors='coerce')
    
    if external_factors_df is not None and 'date' in external_factors_df.columns:
        external_factors_df['date'] = pd.to_datetime(external_factors_df['date'], errors='coerce')
    
    # Add age_normalized to users_df before preprocessing
    if 'age' in users_df.columns and 'age_normalized' not in users_df.columns:
        users_df['age_normalized'] = users_df['age'] / 100.0
        logger.info("Added age_normalized column to users_df")
    
    # Pre-merge relevant data to avoid explosion in preprocessor
    logger.info("Performing controlled merge to avoid data explosion")
    
    # Start with sleep data as the base
    merged_df = sleep_data_df.copy()
    
    # Add user fields
    user_fields = ['user_id', 'age', 'age_normalized', 'gender', 'profession', 'region', 
                  'profession_category', 'region_category', 'sleep_pattern']
    available_user_fields = [field for field in user_fields if field in users_df.columns]
    
    # Merge with users selectively
    merged_df = pd.merge(
        merged_df, 
        users_df[available_user_fields], 
        on='user_id', 
        how='left'
    )
    
    # Add wearable data selectively if available
    if wearable_data_df is not None:
        # Skip huge array columns
        skip_cols = ['heart_rate_data', 'movement_data', 'sleep_stage_data']
        wearable_cols = [col for col in wearable_data_df.columns if col not in skip_cols]
        
        # Merge on user_id and date
        merged_df = pd.merge(
            merged_df, 
            wearable_data_df[wearable_cols], 
            on=['user_id', 'date'], 
            how='left',
            suffixes=('', '_wearable')
        )
        
        # Remove duplicate columns with _wearable suffix
        wearable_suffix_cols = [col for col in merged_df.columns if col.endswith('_wearable')]
        merged_df = merged_df.drop(columns=wearable_suffix_cols)
    
    # Add basic features that might be missing
    merged_df = add_missing_features(merged_df)
    
    # Verify no data explosion occurred
    if len(merged_df) > len(sleep_data_df) * 1.1:
        logger.error(f"Data explosion detected: {len(merged_df)} records vs original {len(sleep_data_df)} records")
        # Cap to original record count
        merged_df = merged_df.iloc[:len(sleep_data_df)]
        logger.info(f"Capped merged data to original record count: {len(merged_df)} records")
    
    logger.info(f"Controlled merge complete: {len(merged_df)} records, {merged_df['user_id'].nunique()} unique users")
    
    # Skip the regular preprocessor and use our merged data directly
    processed_data = merged_df
    
    # Handle missing values
    for col in processed_data.columns:
        if processed_data[col].isna().any():
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                processed_data[col] = processed_data[col].fillna(0)
            else:
                processed_data[col] = processed_data[col].fillna('unknown')
    
    # Convert date columns to datetime
    date_cols = ['date', 'bedtime', 'sleep_onset_time', 'wake_time', 'out_bed_time']
    for col in date_cols:
        if col in processed_data.columns and not pd.api.types.is_datetime64_any_dtype(processed_data[col]):
            processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
    
    # Add calculated sleep metrics if missing
    if 'sleep_efficiency' not in processed_data.columns and 'sleep_duration_hours' in processed_data.columns and 'time_in_bed_hours' in processed_data.columns:
        processed_data['sleep_efficiency'] = processed_data['sleep_duration_hours'] / processed_data['time_in_bed_hours']
        processed_data['sleep_efficiency'] = processed_data['sleep_efficiency'].clip(0, 1)  # Bound within valid range
        logger.info("Calculated sleep_efficiency from duration and time in bed")
    
    logger.info(f"Processed data ready: {len(processed_data)} records, {processed_data['user_id'].nunique()} unique users")
    
    # Engineer features
    logger.info("Starting feature engineering")
    feature_engineer = FeatureEngineering(config.get('feature_engineering', {}))
    features_df, targets_df = feature_engineer.create_features(processed_data)
    
    # Ensure user_id is present in features_df and targets_df
    if 'user_id' not in features_df.columns and 'user_id' in processed_data.columns:
        logger.info("Adding user_id to features_df from processed_data")
        features_df['user_id'] = processed_data['user_id'].values[:len(features_df)]
    
    if 'user_id' not in targets_df.columns and 'user_id' in processed_data.columns:
        logger.info("Adding user_id to targets_df from processed_data")
        targets_df['user_id'] = processed_data['user_id'].values[:len(targets_df)]
    
    # Log feature engineering results
    logger.info(f"Features: {len(features_df)} records, {features_df['user_id'].nunique() if 'user_id' in features_df.columns else 0} unique users")
    logger.info(f"Targets: {len(targets_df)} records, {targets_df['user_id'].nunique() if 'user_id' in targets_df.columns else 0} unique users")
    
    # Ensure date is present in features_df
    if 'date' not in features_df.columns and 'date' in processed_data.columns:
        features_df['date'] = processed_data['date'].values[:len(features_df)]
    
    # Create a combined dataframe for model training
    combined_data = pd.DataFrame()
    
    # First add user_id and date - these are essential for the model
    if 'user_id' in features_df.columns:
        combined_data['user_id'] = features_df['user_id']
    else:
        logger.error("user_id missing from features - using index as user_id")
        combined_data['user_id'] = [f"user_{i}" for i in range(len(features_df))]
    
    if 'date' in features_df.columns:
        combined_data['date'] = features_df['date']
    else:
        logger.warning("date column missing - creating synthetic dates")
        base_date = datetime(2024, 1, 1)
        user_dates = {}
        for i, user_id in enumerate(combined_data['user_id'].unique()):
            mask = combined_data['user_id'] == user_id
            count = mask.sum()
            dates = [base_date + timedelta(days=i) for i in range(count)]
            user_dates[user_id] = dates
        
        combined_data['date'] = combined_data.apply(
            lambda row: user_dates[row['user_id']].pop(0), axis=1
        )
    
    # Add all feature columns (except user_id and date)
    feature_cols = [col for col in features_df.columns if col not in ['user_id', 'date']]
    for col in feature_cols:
        combined_data[col] = features_df[col]
    
    # Add target column as sleep_efficiency (which is what SleepQualityModel expects)
    if 'sleep_quality' in targets_df.columns:
        combined_data['sleep_efficiency'] = targets_df['sleep_quality']
    elif 'sleep_efficiency' in processed_data.columns:
        # Get sleep_efficiency directly from processed_data if available
        combined_data['sleep_efficiency'] = processed_data['sleep_efficiency'].values[:len(combined_data)]
    
    # Add a column to track synthetic data
    combined_data['is_synthetic'] = False
    
    # Verify age_normalized exists in the combined data
    if 'age_normalized' not in combined_data.columns:
        if 'age' in combined_data.columns:
            combined_data['age_normalized'] = combined_data['age'] / 100.0
            logger.info("Added missing age_normalized to combined_data from age")
        else:
            # Use a default value
            combined_data['age_normalized'] = 0.35  # Default age 35
            logger.warning("Added default age_normalized as age column is missing")
    
    # Check all features required by the model
    # From the config model features list - include all the ones that might be required
    model_features = config.get('sleep_quality_model', {}).get('features', [
        'sleep_duration_hours', 'sleep_efficiency', 'awakenings_count', 'total_awake_minutes',
        'deep_sleep_percentage', 'rem_sleep_percentage', 'sleep_onset_latency_minutes',
        'heart_rate_variability', 'average_heart_rate', 'age_normalized',
        'profession_healthcare', 'profession_tech', 'profession_service', 'profession_education',
        'profession_office', 'profession_other', 'season_Winter', 'season_Spring', 'season_Summer', 'season_Fall'
    ])
    
    # Ensure all required features exist
    for feature in model_features:
        if feature not in combined_data.columns:
            logger.warning(f"Required feature '{feature}' missing. Adding with default value.")
            if feature.startswith('profession_'):
                # For profession features, set to 0 (not this profession)
                combined_data[feature] = 0
            elif feature.startswith('season_'):
                # For season features, set based on current date or default to Spring
                if 'date' in combined_data.columns:
                    season = combined_data['date'].dt.month.apply(lambda m: 'Winter' if m in [12, 1, 2] else
                                                                'Spring' if m in [3, 4, 5] else
                                                                'Summer' if m in [6, 7, 8] else 'Fall')
                    combined_data[feature] = (season == feature.replace('season_', '')).astype(float)
                else:
                    # Default to Spring
                    combined_data[feature] = 1.0 if feature == 'season_Spring' else 0.0
            elif feature in ['deep_sleep_percentage', 'rem_sleep_percentage']:
                # Default reasonable values for sleep stages
                combined_data[feature] = 0.2  # 20% for each sleep stage
            elif feature in ['sleep_duration_hours']:
                combined_data[feature] = 7.0  # Default 7 hours
            elif feature in ['sleep_efficiency']:
                combined_data[feature] = 0.85  # Default 85% efficiency
            else:
                # Default to 0 for other features
                combined_data[feature] = 0.0
    
    # Split data into train/validation (80/20)
    unique_users = combined_data['user_id'].unique()
    train_users, val_users = train_test_split(
        unique_users, 
        test_size=args.test_size, 
        random_state=args.seed
    )

    train_data = combined_data[combined_data['user_id'].isin(train_users)].reset_index(drop=True)
    val_data = combined_data[combined_data['user_id'].isin(val_users)].reset_index(drop=True)

    logger.info(f"Split data into {len(train_data)} training samples and {len(val_data)} validation samples")

    # Add this new code here
    logger.info("Ensuring all data is numeric before training...")
    feature_columns = [col for col in train_data.columns if col not in ['user_id', 'date', 'is_synthetic']]
    train_data = ensure_numeric_data(train_data, feature_columns)
    val_data = ensure_numeric_data(val_data, feature_columns)
    
    logger.info(f"Split data into {len(train_data)} training samples and {len(val_data)} validation samples")
    
    # Create and train sleep quality model
    sleep_quality_config = config.get('sleep_quality_model', {})
    sleep_quality_model = SleepQualityModel(args.config)
    
    logger.info("Training sleep quality model...")
    
    # Train the model (with fallback to handle limited data)
    try:
        training_history = sleep_quality_model.train(train_data, val_data)
        logger.info("Sleep quality model training completed successfully")
    except Exception as e:
        logger.warning(f"Error during sequence-based training: {str(e)}")
        logger.info("Falling back to limited data training method")
        training_history = sleep_quality_model.train_with_limited_data(combined_data)
        logger.info("Fallback training completed successfully")
    
    # Extract metrics from training history
    metrics = {
        'final_train_loss': training_history['train_losses'][-1],
        'final_val_loss': training_history.get('val_losses', [training_history['train_losses'][-1]])[-1],
        'features_used': training_history['features']
    }
    
    # Save the model
    model_path = os.path.join(args.output_dir, 'sleep_quality_model')
    sleep_quality_model.save(model_path)
    logger.info(f"Saved sleep quality model to {model_path}")
    
    # Update config with training details for model card
    sleep_quality_config.update({
        'num_users': len(users_df),
        'days_per_user': len(sleep_data_df) // len(users_df) if len(users_df) > 0 else 0,
        'input_features': training_history['features']
    })
    
    # Generate model card using the method in SleepQualityModel
    model_card_path = os.path.join(args.output_dir, 'model_card', 'sleep_quality_model.json')
    performance_metrics = {
        'mse': metrics['final_val_loss'],
        'rmse': np.sqrt(metrics['final_val_loss']),
        'features': metrics['features_used']
    }
    
    try:
        sleep_quality_model.generate_model_card(
            model_card_path,
            performance_metrics=performance_metrics,
            training_data_description={
                'num_users': sleep_quality_config['num_users'],
                'num_records': len(sleep_data_df),
                'date_range': f"{sleep_data_df['date'].min()} to {sleep_data_df['date'].max()}"
            }
        )
        logger.info(f"Generated model card at {model_card_path}")
    except Exception as e:
        logger.error(f"Error generating model card: {str(e)}")
    
    # Train transfer learning model if configured
    if 'transfer_learning' in config and config.get('transfer_learning', {}).get('enabled', False):
        transfer_learning_config = config['transfer_learning']
        
        logger.info("Training transfer learning model...")
        transfer_model = TransferLearning(
            config_path=args.config
        )
        
        # Load the base model
        transfer_model.load_base_model(model_path)
        
        # For demonstration, we'll use a subset of users for transfer learning
        transfer_users = np.random.choice(
            users_df['user_id'],
            size=min(transfer_learning_config.get('min_user_samples', 5), len(users_df)),
            replace=False
        ).tolist()
        
        # Adapt model to each user
        for user_id in transfer_users:
            user_data = combined_data[combined_data['user_id'] == user_id]
            if len(user_data) >= transfer_learning_config['hyperparameters'].get('min_user_samples', 5):
                logger.info(f"Adapting model for user: {user_id}")
                try:
                    history = transfer_model.adapt_to_user(user_id, user_data)
                    
                    # Save user model
                    transfer_model.save_user_model(user_id, os.path.join(args.output_dir, 'user_models/model'))
                except Exception as e:
                    logger.error(f"Error adapting model for user {user_id}: {str(e)}")
            else:
                logger.warning(f"Not enough data for user {user_id} to perform transfer learning")
    
    logger.info("Model training complete!")

if __name__ == "__main__":
    main()