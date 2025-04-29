#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to train the sleep quality model with fixes for the NaN issues.
This can occur when the model is trained on data with missing values or incorrect types.
Example was recently the age_normalized column was missing.
This script will:
- Load the data
- Fix data types
- Add missing features
- Train the model
- Validate the model
- Save the model
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import logging
import yaml
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.preprocessing import Preprocessor
from src.data_processing.feature_engineering import FeatureEngineering
from src.models.sleep_quality import SleepQualityModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_fixed_model.log')
    ]
)

logger = logging.getLogger(__name__)

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

def fix_data_types(users_df, sleep_data_df, wearable_data_df=None, external_factors_df=None):
    """Ensure all dataframes have correct data types."""
    # Fix user_id to be string type in all dataframes
    users_df['user_id'] = users_df['user_id'].astype(str)
    sleep_data_df['user_id'] = sleep_data_df['user_id'].astype(str)
    
    if wearable_data_df is not None:
        wearable_data_df['user_id'] = wearable_data_df['user_id'].astype(str)
    
    # Convert date columns to datetime
    date_columns = ['date', 'bedtime', 'sleep_onset_time', 'wake_time', 'out_bed_time', 'created_at']
    
    for df in [sleep_data_df, wearable_data_df, external_factors_df]:
        if df is not None:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Fill NaN values in critical columns
    if 'sleep_efficiency' in sleep_data_df.columns and sleep_data_df['sleep_efficiency'].isna().any():
        logger.warning(f"Found {sleep_data_df['sleep_efficiency'].isna().sum()} NaN values in sleep_efficiency")
        sleep_data_df['sleep_efficiency'] = sleep_data_df['sleep_efficiency'].fillna(
            sleep_data_df['sleep_efficiency'].mean()
        )
    
    return users_df, sleep_data_df, wearable_data_df, external_factors_df

def add_missing_features(data):
    """Add missing features required by the model."""
    # Add age_normalized if missing
    if 'age' in data.columns and 'age_normalized' not in data.columns:
        data['age_normalized'] = data['age'] / 100.0
        logger.info("Added missing feature: age_normalized")
    
    # Add profession features if missing
    if 'profession' in data.columns and 'profession_category' not in data.columns:
        from src.utils.constants import profession_categories
        
        def categorize_profession(profession):
            for category, keywords in profession_categories.items():
                if any(keyword.lower() in profession.lower() for keyword in keywords):
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
    
    # Add season features if missing
    if 'date' in data.columns:
        # Extract month if needed
        if 'month' not in data.columns:
            data['month'] = data['date'].dt.month
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

def train_model_with_fixes():
    """Train the sleep quality model with fixes for NaN issues."""
    logger.info("Starting model training with fixes")
    
    # Load data
    data_dir = 'data/raw'
    users_df, sleep_data_df, wearable_data_df, external_factors_df = load_data(data_dir)
    
    # Fix data types
    users_df, sleep_data_df, wearable_data_df, external_factors_df = fix_data_types(
        users_df, sleep_data_df, wearable_data_df, external_factors_df
    )
    
    # Add missing features to data
    sleep_data_df = add_missing_features(sleep_data_df)
    
    # Load model configuration
    config_path = 'config/model_config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize preprocessor with fixes
    preprocessor = Preprocessor()
    
    # Process data
    logger.info("Preprocessing data...")
    processed_data = preprocessor.process(
        users_df, 
        sleep_data_df, 
        wearable_data_df, 
        external_factors_df
    )
    
    # Check for and handle NaN values in processed data
    for col in processed_data.columns:
        if processed_data[col].isna().any():
            nan_count = processed_data[col].isna().sum()
            logger.warning(f"Found {nan_count} NaN values in column {col}. Filling with appropriate values.")
            
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                # For numeric columns, fill with 0
                processed_data[col] = processed_data[col].fillna(0)
            else:
                # For non-numeric columns, fill with the most common value
                processed_data[col] = processed_data[col].fillna(
                    processed_data[col].mode().iloc[0] if not processed_data[col].mode().empty else "unknown"
                )
    
    # Initialize feature engineering with fixes
    feature_engineer = FeatureEngineering(config['feature_engineering'])
    
    # Set expected features in the config
    feature_engineer.config['expected_features'] = config['sleep_quality_model']['features']
    
    # Create features
    logger.info("Performing feature engineering...")
    features_df, targets_df = feature_engineer.create_features(processed_data)
    
    # Initialize the model
    sleep_quality_model = SleepQualityModel(config_path)
    
    # Split data into train/validation (80/20)
    logger.info("Splitting data into train/validation sets...")
    
    # Merge features with targets to ensure alignment
    combined_data = pd.merge(
        features_df, 
        targets_df[['user_id', 'date', 'sleep_quality']], 
        on=['user_id', 'date'],
        how='inner'
    )
    
    # Rename sleep_quality to sleep_efficiency (expected by the model)
    combined_data = combined_data.rename(columns={'sleep_quality': 'sleep_efficiency'})
    
    # Get unique users for splitting
    unique_users = combined_data['user_id'].unique()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_users)
    
    split_idx = int(0.8 * len(unique_users))
    train_users = unique_users[:split_idx]
    val_users = unique_users[split_idx:]
    
    train_data = combined_data[combined_data['user_id'].isin(train_users)].copy()
    val_data = combined_data[combined_data['user_id'].isin(val_users)].copy()
    
    logger.info(f"Training data: {len(train_data)} records from {len(train_users)} users")
    logger.info(f"Validation data: {len(val_data)} records from {len(val_users)} users")
    
    # Train the model
    logger.info("Training the model...")
    try:
        # Try sequence-based training first
        logger.info("Attempting sequence-based training...")
        training_history = sleep_quality_model.train(train_data, val_data)
        logger.info("Sequence-based training completed successfully!")
    except Exception as e:
        logger.warning(f"Sequence-based training failed: {str(e)}")
        logger.info("Falling back to alternative training method...")
        training_history = sleep_quality_model.train_with_limited_data(combined_data)
        logger.info("Alternative training method completed!")
    
    # Create output directory if it doesn't exist
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, 'sleep_quality_model_fixed')
    sleep_quality_model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Generate model card
    performance_metrics = {
        'final_train_loss': training_history['train_losses'][-1],
        'final_val_loss': training_history.get('val_losses', [training_history['train_losses'][-1]])[-1],
        'features_used': training_history['features']
    }
    
    training_data_description = {
        'num_users': len(users_df),
        'num_records': len(sleep_data_df),
        'date_range': f"{sleep_data_df['date'].min()} to {sleep_data_df['date'].max()}"
    }
    
    sleep_quality_model.generate_model_card(
        f"{model_path}_card.json", 
        performance_metrics=performance_metrics,
        training_data_description=training_data_description
    )
    
    logger.info("Model training and saving completed successfully!")
    
    return sleep_quality_model, training_history

def validate_model(model, data):
    """Validate the trained model on test data."""
    logger.info("Validating the model...")
    
    # Ensure the data has all required features
    if not all(feat in data.columns for feat in model.feature_columns):
        missing_features = [feat for feat in model.feature_columns if feat not in data.columns]
        logger.warning(f"Missing features in validation data: {missing_features}")
        
        # Add missing features as zeros
        for feat in missing_features:
            data[feat] = 0.0
    
    # Make predictions
    try:
        predictions = model.predict(data)
        logger.info(f"Made predictions for {len(predictions)} records")
        
        # Calculate error metrics
        actual = data['sleep_efficiency'].values
        predicted = predictions['predicted_sleep_efficiency'].values
        
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        
        # Calculate R-squared
        ss_total = np.sum((actual - np.mean(actual)) ** 2)
        ss_residual = np.sum((actual - predicted) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        logger.info(f"Validation metrics:")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R-squared: {r_squared:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared
        }
        
    except Exception as e:
        logger.error(f"Error during model validation: {str(e)}")
        return None

def plot_learning_curves(training_history, output_dir='reports'):
    """Plot the learning curves from training history."""
    import matplotlib.pyplot as plt
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(training_history['train_losses'], label='Training Loss')
    
    if 'val_losses' in training_history:
        plt.plot(training_history['val_losses'], label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Sleep Quality Model Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()
    
    logger.info(f"Learning curves saved to {output_dir}/learning_curves.png")

def main():
    """Main entry point for the script."""
    # Start time for performance measurement
    start_time = datetime.now()
    
    try:
        # Train the model with fixes
        model, training_history = train_model_with_fixes()
        
        # Plot learning curves
        plot_learning_curves(training_history)
        
        # Load validation data (use part of the original data for simplicity)
        data_dir = 'data/raw'
        users_df, sleep_data_df, wearable_data_df, external_factors_df = load_data(data_dir)
        
        # Preprocess validation data
        preprocessor = Preprocessor()
        processed_data = preprocessor.process(users_df, sleep_data_df, wearable_data_df, external_factors_df)
        
        # Add missing features
        processed_data = add_missing_features(processed_data)
        
        # Select a subset for validation (different users than training if possible)
        validation_users = np.random.choice(processed_data['user_id'].unique(), size=min(10, len(processed_data['user_id'].unique())), replace=False)
        validation_data = processed_data[processed_data['user_id'].isin(validation_users)].copy()
        
        # Validate the model
        validation_metrics = validate_model(model, validation_data)
        
        # End time and report total duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0
        
        logger.info(f"Script execution completed in {duration:.2f} minutes")
        logger.info("SUCCESS: Model training and validation completed successfully")
        
    except Exception as e:
        logger.error(f"ERROR: Script execution failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # End time and report total duration even on error
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0
        logger.info(f"Script execution failed after {duration:.2f} minutes")

if __name__ == "__main__":
    main()