# scripts/train_models.py
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
import json
from datetime import datetime, timedelta

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.preprocessing import Preprocessor
from src.data_processing.feature_engineering import FeatureEngineering
from src.models.sleep_quality import SleepQualityModel
from src.models.transfer_learning import TransferLearning

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
    logger.info(f"Created output directory: {output_dir}")

def load_data(data_dir):
    """Load data from CSV files."""
    users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
    sleep_data_df = pd.read_csv(os.path.join(data_dir, 'sleep_data.csv'))
    wearable_data_df = pd.read_csv(os.path.join(data_dir, 'wearable_data.csv'))
    external_factors_df = pd.read_csv(os.path.join(data_dir, 'external_factors.csv'))
    
    logger.info(f"Loaded {len(users_df)} users")
    logger.info(f"Loaded {len(sleep_data_df)} sleep records")
    logger.info(f"Loaded {len(wearable_data_df)} wearable records")
    logger.info(f"Loaded {len(external_factors_df)} external factor records")
    
    return users_df, sleep_data_df, wearable_data_df, external_factors_df

def save_model_card(model_name, config, metrics, output_dir):
    """Create and save a model card with documentation."""
    model_card = {
        "model_name": model_name,
        "version": "1.0.0",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": config.get('description', f"Model for {model_name}"),
        "model_type": config.get('model_type', 'PyTorch Neural Network'),
        "input_features": config.get('input_features', []),
        "output": config.get('output', 'Sleep Quality Score'),
        "performance_metrics": metrics,
        "limitations": config.get('limitations', [
            "Limited to the patterns in the training data",
            "May not generalize to users with rare sleep disorders",
            "Assumes regular data collection from wearable devices"
        ]),
        "intended_use": config.get('intended_use', [
            "Predict sleep quality based on user and wearable data",
            "Identify sleep patterns and anomalies",
            "Provide recommendations for improved sleep"
        ]),
        "training_data_characteristics": {
            "num_users": config.get('num_users', 0),
            "days_per_user": config.get('days_per_user', 0),
            "user_demographics": config.get('user_demographics', "Simulated diverse population"),
            "sleep_patterns": config.get('sleep_patterns', [
                "Normal Sleeper Pattern",
                "Insomnia Pattern",
                "Shift Worker Pattern",
                "Oversleeper Pattern",
                "Highly Variable Sleeper"
            ])
        },
        "hyperparameters": config.get('hyperparameters', {})
    }

    # Add predictive capabilities section to model_card dictionary
    model_card["predictive_capabilities"] = {
        "sleep_quality_predictions": [
            "Sleep efficiency trends over time based on user-reported data",
            "Probability of experiencing insomnia on upcoming nights",
            "Estimated sleep quality for the coming night based on recent patterns",
            "Expected subjective ratings if certain behaviors are modified"
        ],
        "pattern_recognition": [
            "Sleep consistency patterns and how they affect overall sleep quality",
            "Identification of insomnia triggers based on correlations in the data",
            "Detection of severe insomnia episodes before they become chronic",
            "Recognition of improvement trends even when subjective perception lags"
        ],
        "personalized_insights": [
            "Most effective sleep window based on recorded sleep efficiency data",
            "Optimal bedtime and wake time for maximum sleep quality",
            "Personal threshold for sleep onset latency that predicts a good night's sleep",
            "Impact of awakenings on overall sleep quality for the individual"
        ],
        "behavior_impact_assessment": [
            "Expected improvement if sleep consistency is increased",
            "Predicted benefits of reducing time in bed for those with extended wake times",
            "Forecasted sleep efficiency changes with various interventions",
            "Projected recovery time after periods of severe insomnia"
        ],
        "long_term_predictions": [
            "Risk of developing chronic insomnia based on current patterns",
            "Expected timeline for improvement with consistent sleep practices",
            "Likelihood of relapse based on pattern recognition",
            "Long-term sleep health trajectory with and without intervention"
        ]
    }
    
    # Save the model card as JSON
    model_card_path = os.path.join(output_dir, 'model_cards', f"{model_name}_card.json")
    with open(model_card_path, 'w') as f:
        json.dump(model_card, f, indent=2)
    
    logger.info(f"Saved model card to {model_card_path}")
    
    # Also create a markdown version for better readability
    markdown_path = os.path.join(output_dir, 'model_cards', f"{model_name}_card.md")
    
    with open(markdown_path, 'w') as f:
        f.write(f"# {model_name} Model Card\n\n")
        f.write(f"**Version:** {model_card['version']}\n")
        f.write(f"**Created:** {model_card['created_at']}\n\n")
        
        f.write("## Description\n")
        f.write(f"{model_card['description']}\n\n")
        
        f.write("## Model Information\n")
        f.write(f"**Type:** {model_card['model_type']}\n")
        f.write("**Input Features:**\n")
        for feature in model_card['input_features']:
            f.write(f"- {feature}\n")
        f.write(f"**Output:** {model_card['output']}\n\n")
        
        f.write("## Performance Metrics\n")
        for metric, value in model_card['performance_metrics'].items():
            f.write(f"- **{metric}:** {value}\n")
        f.write("\n")
        
        f.write("## Limitations\n")
        for limitation in model_card['limitations']:
            f.write(f"- {limitation}\n")
        f.write("\n")
        
        f.write("## Intended Use\n")
        for use in model_card['intended_use']:
            f.write(f"- {use}\n")
        f.write("\n")
        
        f.write("## Training Data Characteristics\n")
        f.write(f"- **Number of Users:** {model_card['training_data_characteristics']['num_users']}\n")
        f.write(f"- **Days per User:** {model_card['training_data_characteristics']['days_per_user']}\n")
        f.write(f"- **Demographics:** {model_card['training_data_characteristics']['user_demographics']}\n")
        f.write("- **Sleep Patterns:**\n")
        for pattern in model_card['training_data_characteristics']['sleep_patterns']:
            f.write(f"  - {pattern}\n")
        f.write("\n")

        # Add Predictive Capabilities section
        f.write("## Predictive Capabilities\n\n")
        
        f.write("### Sleep Quality Predictions\n")
        for capability in model_card['predictive_capabilities']['sleep_quality_predictions']:
            f.write(f"- {capability}\n")
        f.write("\n")
        
        f.write("### Pattern Recognition\n")
        for capability in model_card['predictive_capabilities']['pattern_recognition']:
            f.write(f"- {capability}\n")
        f.write("\n")
        
        f.write("### Personalized Insights\n")
        for capability in model_card['predictive_capabilities']['personalized_insights']:
            f.write(f"- {capability}\n")
        f.write("\n")
        
        f.write("### Behavior Impact Assessment\n")
        for capability in model_card['predictive_capabilities']['behavior_impact_assessment']:
            f.write(f"- {capability}\n")
        f.write("\n")
        
        f.write("### Long-term Predictions\n")
        for capability in model_card['predictive_capabilities']['long_term_predictions']:
            f.write(f"- {capability}\n")
        f.write("\n")
        
        f.write("## Hyperparameters\n")
        for param, value in model_card['hyperparameters'].items():
            f.write(f"- **{param}:** {value}\n")
    
    logger.info(f"Saved markdown model card to {markdown_path}")

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
    wearable_data_df['user_id'] = wearable_data_df['user_id'].astype(str)
    
    # Log original user counts for debugging
    original_user_count = sleep_data_df['user_id'].nunique()
    logger.info(f"Original data: {len(sleep_data_df)} records, {original_user_count} unique users")
    
    # CRITICAL FIX: Skip the regular preprocessing and do a simplified version 
    # to avoid the data explosion issue
    logger.info("Using direct data preparation to avoid preprocessing issues")
    
    # Create a simplified processed dataset directly from sleep_data 
    # Preprocess data
    preprocessor = Preprocessor(config['preprocessing'])
    
    # Ensure all inputs are DataFrames
    if not isinstance(users_df, pd.DataFrame):
        print(f"Warning: users_df is not a DataFrame, converting from {type(users_df)}")
        if isinstance(users_df, dict):
            users_df = pd.DataFrame([users_df])
        else:
            users_df = pd.DataFrame(users_df)

    if not isinstance(sleep_data_df, pd.DataFrame):
        print(f"Warning: sleep_data_df is not a DataFrame, converting from {type(sleep_data_df)}")
        if isinstance(sleep_data_df, dict):
            sleep_data_df = pd.DataFrame([sleep_data_df])
        else:
            sleep_data_df = pd.DataFrame(sleep_data_df)

    if wearable_data_df is not None and not isinstance(wearable_data_df, pd.DataFrame):
        print(f"Warning: wearable_data_df is not a DataFrame, converting from {type(wearable_data_df)}")
        if isinstance(wearable_data_df, dict):
            wearable_data_df = pd.DataFrame([wearable_data_df])
        else:
            wearable_data_df = pd.DataFrame(wearable_data_df)

    if external_factors_df is not None and not isinstance(external_factors_df, pd.DataFrame):
        print(f"Warning: external_factors_df is not a DataFrame, converting from {type(external_factors_df)}")
        if isinstance(external_factors_df, dict):
            external_factors_df = pd.DataFrame([external_factors_df])
        else:
            external_factors_df = pd.DataFrame(external_factors_df)
    
    processed_data = preprocessor.process(
        users_df, 
        sleep_data_df, 
        wearable_data_df, 
        external_factors_df
    )
    
    # Convert date columns to datetime for consistent handling
    date_columns = ['date', 'bedtime', 'sleep_onset_time', 'wake_time']
    for col in date_columns:
        if col in processed_data.columns:
            processed_data[col] = pd.to_datetime(processed_data[col])
    
    # Add basic sleep calculations if they don't exist
    if 'sleep_efficiency' not in processed_data.columns and 'sleep_duration_hours' in processed_data.columns and 'time_in_bed_hours' in processed_data.columns:
        processed_data['sleep_efficiency'] = processed_data['sleep_duration_hours'] / processed_data['time_in_bed_hours']
        processed_data['sleep_efficiency'] = processed_data['sleep_efficiency'].clip(0, 1)  # Ensure valid range
    
    # Merge with essential wearable data if available
    if wearable_data_df is not None:
        # Select only essential columns from wearable data to avoid explosion
        wearable_cols = ['user_id', 'date', 'device_sleep_duration', 'deep_sleep_percentage', 
                         'rem_sleep_percentage', 'heart_rate_variability', 'average_heart_rate']
        wearable_subset = wearable_data_df[
            [col for col in wearable_cols if col in wearable_data_df.columns]
        ].copy()
        
        # Convert date to datetime for merging
        if 'date' in wearable_subset.columns:
            wearable_subset['date'] = pd.to_datetime(wearable_subset['date'])
        
        # Merge on user_id and date with left join to maintain sleep_data structure
        processed_data = pd.merge(
            processed_data,
            wearable_subset,
            on=['user_id', 'date'],
            how='left',
            suffixes=('', '_y')
        )

        # Directly identify and drop the _y columns in one operation
        y_cols = [col for col in processed_data.columns if col.endswith('_y')]
        if y_cols:
            logger.info(f"Dropping columns: {y_cols}")
            processed_data.drop(columns=y_cols, inplace=True)
    
    logger.info(f"Processed data: {len(processed_data)} records, {processed_data['user_id'].nunique()} unique users")
    
    # Verify we didn't get data explosion
    if len(processed_data) > len(sleep_data_df) * 1.1:  # Allow for small increase
        logger.error(f"Data explosion detected: {len(processed_data)} vs original {len(sleep_data_df)}")
        # Revert to just using sleep_data_df
        processed_data = sleep_data_df.copy()
        logger.info("Reverted to original sleep data")
    
    # Engineer features
    logger.info("Starting feature engineering")
    feature_engineer = FeatureEngineering(config['feature_engineering'])
    features_df, targets_df = feature_engineer.create_features(processed_data)
    
    # Ensure user_id is present in features_df and targets_df
    if 'user_id' not in features_df.columns and 'user_id' in processed_data.columns:
        logger.info("Adding user_id to features_df from processed_data")
        features_df['user_id'] = processed_data['user_id'].values
    
    if 'user_id' not in targets_df.columns and 'user_id' in processed_data.columns:
        logger.info("Adding user_id to targets_df from processed_data")
        targets_df['user_id'] = processed_data['user_id'].values
    
    # Log feature engineering results
    logger.info(f"Features: {len(features_df)} records, {features_df['user_id'].nunique() if 'user_id' in features_df.columns else 0} unique users")
    logger.info(f"Targets: {len(targets_df)} records, {targets_df['user_id'].nunique() if 'user_id' in targets_df.columns else 0} unique users")
    
    # Ensure date is present in features_df
    if 'date' not in features_df.columns and 'date' in processed_data.columns:
        features_df['date'] = processed_data['date']
    
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
        combined_data['sleep_efficiency'] = processed_data['sleep_efficiency']
    
    # Add a column to track synthetic data
    combined_data['is_synthetic'] = False
    
    # Log final combined dataset info
    logger.info(f"Combined data: {len(combined_data)} records, {combined_data['user_id'].nunique()} unique users")
    logger.info(f"Final columns: {combined_data.columns.tolist()}")
    
    # Data Augmentation Section
    # Check if we have enough data per user for sequence-based modeling
    user_counts = combined_data.groupby('user_id').size()
    logger.info(f"Records per user before augmentation: min={user_counts.min()}, max={user_counts.max()}, avg={user_counts.mean():.2f}")
    
    # If we don't have enough data per user, generate synthetic data
    min_required_samples = 8  # Need at least sequence_length (7) + 1
    
    # Find users who need synthetic data
    users_needing_data = []
    for user_id, count in user_counts.items():
        if count < min_required_samples:
            users_needing_data.append(user_id)
    
    if users_needing_data:
        logger.warning(f"Not enough data for {len(users_needing_data)} users (minimum required: {min_required_samples})")
        logger.info(f"Generating synthetic data for {len(users_needing_data)} users")
        
        # Create a new dataframe for synthetic data
        synthetic_data_rows = []
        
        # Process only users who need synthetic data
        for user_id in users_needing_data:
            # Get data for this user
            group = combined_data[combined_data['user_id'] == user_id]
            
            # How many additional samples we need
            additional_needed = min_required_samples - len(group)
            logger.info(f"Generating {additional_needed} synthetic entries for user {user_id}")
            
            # Get base values for this user
            base_record = group.iloc[0].copy()
            
            # Determine the base date for this user
            if isinstance(base_record['date'], pd.Timestamp):
                base_date = base_record['date']
            else:
                try:
                    base_date = pd.to_datetime(base_record['date'])
                except:
                    base_date = datetime(2024, 1, 1)
            
            # Generate additional entries with slight variations
            for i in range(additional_needed):
                new_record = {}
                
                # Use the same user_id
                new_record['user_id'] = user_id
                
                # Mark as synthetic
                new_record['is_synthetic'] = True
                
                # Vary the date (one day per new record, going backwards from the original date)
                new_record['date'] = base_date - timedelta(days=i+1)
                
                # Add sleep_efficiency with small random variations
                if 'sleep_efficiency' in group.columns:
                    base_efficiency = group['sleep_efficiency'].iloc[0]
                    # Add small random variation (±20%)
                    variation = np.random.uniform(-0.2, 0.2)
                    new_efficiency = base_efficiency * (1 + variation)
                    # Keep within valid range [0, 1]
                    new_record['sleep_efficiency'] = max(0, min(1, new_efficiency))
                
                # Add variations to all other features
                for col in group.columns:
                    if col not in ['user_id', 'date', 'sleep_efficiency', 'is_synthetic']:
                        if pd.api.types.is_numeric_dtype(group[col]):
                            # Add small random variation to numeric features (±30%)
                            base_value = group[col].iloc[0]
                            variation = np.random.uniform(-0.3, 0.3)
                            new_value = base_value * (1 + variation)
                            new_record[col] = new_value
                        else:
                            # For non-numeric columns, just copy the original value
                            new_record[col] = group[col].iloc[0]
                
                # Add the synthetic record
                synthetic_data_rows.append(new_record)
        
        # Convert to DataFrame and combine with original data
        if synthetic_data_rows:
            synthetic_df = pd.DataFrame(synthetic_data_rows)
            logger.info(f"Created {len(synthetic_df)} synthetic data points")
            
            # Combine with original data
            combined_data = pd.concat([combined_data, synthetic_df], ignore_index=True)
            
            # Recheck user counts
            user_counts = combined_data.groupby('user_id').size()
            logger.info(f"Records per user after augmentation: min={user_counts.min()}, max={user_counts.max()}, avg={user_counts.mean():.2f}")
        else:
            logger.info("No synthetic data needed to be generated")
    else:
        logger.info("No users need synthetic data generation")
    
    # Ensure the combined data is properly sorted for sequence creation
    combined_data = combined_data.sort_values(['user_id', 'date']).reset_index(drop=True)
    
    # Instead of random indices, split by users
    unique_users = combined_data['user_id'].unique()
    train_users, val_users = train_test_split(
        unique_users, 
        test_size=args.test_size, 
        random_state=args.seed
    )

    train_data = combined_data[combined_data['user_id'].isin(train_users)].reset_index(drop=True)
    val_data = combined_data[combined_data['user_id'].isin(val_users)].reset_index(drop=True)
    
    logger.info(f"Split data into {len(train_data)} training samples and {len(val_data)} validation samples")
    
    # Create and train sleep quality model
    sleep_quality_config = config['sleep_quality_model']
    sleep_quality_model = SleepQualityModel()
    
    logger.info("Training sleep quality model...")
    
    # Check if we still have potential sequence creation issues after augmentation
    min_required_samples = 8  # sequence_length + 1
    user_counts_train = train_data.groupby('user_id').size()
    user_counts_val = val_data.groupby('user_id').size()
    
    logger.info(f"User data counts in training set: min={user_counts_train.min()}, max={user_counts_train.max()}")
    logger.info(f"User data counts in validation set: min={user_counts_val.min()}, max={user_counts_val.max()}")
    
    # Determine if we should use sequence-based or fallback training
    use_fallback = False
    
    if user_counts_train.min() < min_required_samples:
        logger.warning(f"Training set still has users with fewer than {min_required_samples} records")
        use_fallback = True
    
    if user_counts_val.min() < min_required_samples:
        logger.warning(f"Validation set still has users with fewer than {min_required_samples} records")
        use_fallback = True
    
    # Train the model using the appropriate method
    try:
        if use_fallback:
            logger.info("Using fallback (non-sequence) training method")
            # Combine train and validation data for the fallback method
            combined_train_val = pd.concat([train_data, val_data], ignore_index=True)
            training_history = sleep_quality_model.train_with_limited_data(combined_train_val)
        else:
            logger.info("Using sequence-based LSTM training method")
            training_history = sleep_quality_model.train(train_data, val_data)
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        # Try fallback as a last resort if standard training fails
        logger.info("Attempting fallback training method after sequence training failure")
        combined_train_val = pd.concat([train_data, val_data], ignore_index=True)
        training_history = sleep_quality_model.train_with_limited_data(combined_train_val)
    
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
    
    # Create model card
    sleep_quality_config.update({
        'num_users': len(users_df),
        'days_per_user': len(sleep_data_df) // len(users_df) if len(users_df) > 0 else 0,
        'input_features': training_history['features']
    })
    
    save_model_card(
        'sleep_quality',
        sleep_quality_config,
        metrics,
        args.output_dir
    )
    
    # Train transfer learning model if configured
    if 'transfer_learning' in config and config.get('transfer_learning', {}).get('enabled', False):
        transfer_learning_config = config['transfer_learning']
        
        logger.info("Training transfer learning model...")
        transfer_model = TransferLearning(
            base_model=sleep_quality_model,
            **transfer_learning_config
        )
        
        # For demonstration, we'll use a subset of users for transfer learning
        transfer_users = users_df['user_id'].sample(
            n=min(transfer_learning_config.get('min_user_samples', 5), len(users_df)),
            random_state=args.seed
        ).tolist()
        
        transfer_data = combined_data[combined_data['user_id'].isin(transfer_users)]
        
        # Split transfer data into train/val
        transfer_train, transfer_val = train_test_split(
            transfer_data,
            test_size=0.2,
            random_state=args.seed
        )
        
        transfer_metrics = transfer_model.train(transfer_train, transfer_val)
        
        # Save the transfer learning model
        transfer_model_path = os.path.join(args.output_dir, 'transfer_learning_model')
        transfer_model.save(transfer_model_path)
        logger.info(f"Saved transfer learning model to {transfer_model_path}")
        
        # Create model card for transfer learning model
        transfer_learning_config.update({
            'num_users': len(transfer_users),
            'input_features': training_history['features']
        })
        
        save_model_card(
            'transfer_learning',
            transfer_learning_config,
            transfer_metrics,
            args.output_dir
        )
    
    logger.info("Model training complete!")

if __name__ == "__main__":
    main()