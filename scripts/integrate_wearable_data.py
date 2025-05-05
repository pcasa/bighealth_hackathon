# scripts/integrate_wearable_data.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to process wearable data and integrate it with sleep data for model training.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.data_processing.preprocessing import Preprocessor
from src.core.data_processing.feature_engineering import FeatureEngineering
from src.core.models.sleep_quality import SleepQualityModel
from src.core.wearables.wearable_tranformer_manager import WearableTransformerManager
from src.utils.data_validation_fix import ensure_sleep_data_format, validate_dataframe_for_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/wearable_integration.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Integrate wearable data with sleep data for model training')
    
    parser.add_argument(
        '--sleep-data', 
        type=str, 
        default='data/enhanced_demo/data/sleep_data.csv',
        help='Path to sleep data CSV file'
    )
    
    parser.add_argument(
        '--users-data', 
        type=str, 
        default='data/enhanced_demo/data/users.csv',
        help='Path to user profiles CSV file'
    )
    
    parser.add_argument(
        '--wearable-data', 
        type=str, 
        default='data/enhanced_demo/data/wearable_data.csv',
        help='Path to wearable data CSV file'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/processed',
        help='Directory to save processed data'
    )
    
    parser.add_argument(
        '--model-dir', 
        type=str, 
        default='models',
        help='Directory to save trained model'
    )
    
    parser.add_argument(
        '--train-model', 
        action='store_true',
        help='Train the model after processing data'
    )
    
    return parser.parse_args()

def main():
    """Main function to integrate wearable data."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading sleep data from {args.sleep_data}")
    sleep_data = pd.read_csv(args.sleep_data)
    
    logger.info(f"Loading user profiles from {args.users_data}")
    users_data = pd.read_csv(args.users_data)
    
    # Check if wearable data exists
    if os.path.exists(args.wearable_data):
        logger.info(f"Loading wearable data from {args.wearable_data}")
        wearable_data = pd.read_csv(args.wearable_data)
    else:
        logger.warning(f"Wearable data file not found: {args.wearable_data}")
        wearable_data = None
    
    # Initialize components
    preprocessor = Preprocessor()
    feature_engineering = FeatureEngineering()
    wearable_manager = WearableTransformerManager()
    
    # Process wearable data if available
    if wearable_data is not None:
        if 'device_type' in wearable_data.columns:
            logger.info("Processing wearable data by device type")
            
            # Split by device type
            wearable_by_device = {}
            for device_type, group in wearable_data.groupby('device_type'):
                wearable_by_device[device_type] = group
            
            # Transform each device type
            all_transformed = []
            for device_type, device_data in wearable_by_device.items():
                logger.info(f"Transforming {len(device_data)} records for device type: {device_type}")
                transformed = wearable_manager.transform_data(device_data, device_type, users_data)
                all_transformed.append(transformed)
            
            # Combine all transformed data
            if all_transformed:
                standardized_wearable = pd.concat(all_transformed, ignore_index=True)
                logger.info(f"Created {len(standardized_wearable)} standardized wearable records")
            else:
                standardized_wearable = None
        else:
            # Assume data is already in standardized format
            logger.info("Assuming wearable data is already in standardized format")
            standardized_wearable = wearable_data
    else:
        standardized_wearable = None
    
    # Preprocess sleep data with wearable data
    logger.info("Preprocessing sleep data")
    processed_data = preprocessor.preprocess_sleep_data(sleep_data, standardized_wearable)
    
    # Basic validation
    processed_data = ensure_sleep_data_format(processed_data)
    processed_data = validate_dataframe_for_model(processed_data)
    
    # Save processed data
    processed_file = os.path.join(args.output_dir, 'processed_sleep_data.csv')
    processed_data.to_csv(processed_file, index=False)
    logger.info(f"Saved processed data to {processed_file}")
    
    # Feature engineering
    logger.info("Performing feature engineering")
    features_df, targets_df = feature_engineering.create_features(processed_data)
    
    # Save engineered data
    features_file = os.path.join(args.output_dir, 'engineered_features.csv')
    targets_file = os.path.join(args.output_dir, 'engineered_targets.csv')
    
    features_df.to_csv(features_file, index=False)
    targets_df.to_csv(targets_file, index=False)
    
    logger.info(f"Saved engineered features to {features_file}")
    logger.info(f"Saved engineered targets to {targets_file}")
    
    # Train model if requested
    if args.train_model:
        logger.info("Training sleep quality model")
        
        # Split into train/validation sets
        train_idx = int(len(features_df) * 0.8)
        
        train_features = features_df.iloc[:train_idx]
        train_targets = targets_df.iloc[:train_idx]
        
        val_features = features_df.iloc[train_idx:]
        val_targets = targets_df.iloc[train_idx:]
        
        # Combine features and targets for training
        train_data = pd.concat([train_features, train_targets[['sleep_quality']]], axis=1)
        train_data = train_data.rename(columns={'sleep_quality': 'sleep_efficiency'})
        
        val_data = pd.concat([val_features, val_targets[['sleep_quality']]], axis=1)
        val_data = val_data.rename(columns={'sleep_quality': 'sleep_efficiency'})
        
        # Initialize and train model
        model = SleepQualityModel()
        try:
            training_history = model.train(train_data, val_data)
            logger.info("Model training completed successfully")
            
            # Save model
            model_path = os.path.join(args.model_dir, 'sleep_quality_model')
            model.save(model_path)
            logger.info(f"Saved model to {model_path}")
            
            # Generate model card
            model_card_path = os.path.join(args.model_dir, 'model_cards', 'sleep_quality_model.json')
            os.makedirs(os.path.dirname(model_card_path), exist_ok=True)
            
            performance_metrics = {
                'mse': training_history['val_losses'][-1],
                'rmse': np.sqrt(training_history['val_losses'][-1]),
                'features': training_history['features']
            }
            
            # Add wearable features info if available
            if 'wearable_features' in training_history and training_history['wearable_features']:
                performance_metrics['wearable_features'] = training_history['wearable_features']
            
            model.generate_model_card(
                model_card_path,
                performance_metrics=performance_metrics,
                training_data_description={
                    'num_users': len(users_data),
                    'num_records': len(processed_data),
                    'date_range': f"{processed_data['date'].min()} to {processed_data['date'].max()}",
                    'wearable_data': 'Yes' if standardized_wearable is not None else 'No'
                }
            )
            logger.info(f"Generated model card at {model_card_path}")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            logger.info("Falling back to limited data training method")
            
            try:
                training_history = model.train_with_limited_data(processed_data)
                logger.info("Fallback training completed successfully")
                
                # Save model
                model_path = os.path.join(args.model_dir, 'sleep_quality_model')
                model.save(model_path)
                logger.info(f"Saved model to {model_path}")
            except Exception as e:
                logger.error(f"Error during fallback training: {str(e)}")
                return
    
    logger.info("Wearable data integration complete!")

if __name__ == "__main__":
    main()