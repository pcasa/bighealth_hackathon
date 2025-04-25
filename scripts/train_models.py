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
from datetime import datetime

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
    
    # Preprocess data
    preprocessor = Preprocessor(config['preprocessing'])
    
    processed_data = preprocessor.process(
        users_df, 
        sleep_data_df, 
        wearable_data_df, 
        external_factors_df
    )
    
    # Engineer features
    feature_engineer = FeatureEngineering(config['feature_engineering'])
    features_df, targets_df = feature_engineer.create_features(processed_data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, 
        targets_df, 
        test_size=args.test_size, 
        random_state=args.seed
    )
    
    logger.info(f"Split data into {len(X_train)} training samples and {len(X_test)} testing samples")
    
    # Create and train sleep quality model
    sleep_quality_config = config['sleep_quality_model']
    sleep_quality_model = SleepQualityModel()
    
    logger.info("Training sleep quality model...")
    metrics = sleep_quality_model.train(
        X_train, 
        y_train, 
        X_test, 
        y_test,
        verbose=True
    )
    
    # Save the model
    model_path = os.path.join(args.output_dir, 'sleep_quality_model.pt')
    sleep_quality_model.save(model_path)
    logger.info(f"Saved sleep quality model to {model_path}")
    
    # Create model card
    sleep_quality_config.update({
        'num_users': len(users_df),
        'days_per_user': config['sleep_data_generation']['days_per_user'],
        'input_features': features_df.columns.tolist()
    })
    
    save_model_card(
        'sleep_quality',
        sleep_quality_config,
        metrics,
        args.output_dir
    )
    
    # Train transfer learning model if configured
    if config['models'].get('transfer_learning', {}).get('enabled', False):
        transfer_learning_config = config['models']['transfer_learning']
        
        logger.info("Training transfer learning model...")
        transfer_model = TransferLearning(
            base_model=sleep_quality_model,
            **transfer_learning_config
        )
        
        # For demonstration, we'll use a subset of users for transfer learning
        # In a real scenario, you might have specific users for this
        transfer_users = users_df['user_id'].sample(
            n=transfer_learning_config.get('num_transfer_users', 5),
            random_state=args.seed
        ).tolist()
        
        transfer_features = features_df[features_df['user_id'].isin(transfer_users)]
        transfer_targets = targets_df[targets_df['user_id'].isin(transfer_users)]
        
        transfer_metrics = transfer_model.train(
            transfer_features,
            transfer_targets['sleep_quality']
        )
        
        # Save the transfer learning model
        transfer_model_path = os.path.join(args.output_dir, 'transfer_learning_model.pt')
        transfer_model.save(transfer_model_path)
        logger.info(f"Saved transfer learning model to {transfer_model_path}")
        
        # Create model card for transfer learning model
        transfer_learning_config.update({
            'num_users': len(transfer_users),
            'input_features': features_df.columns.tolist()
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