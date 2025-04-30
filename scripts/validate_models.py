# scripts/validate_models.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to validate trained models for the Sleep Insights App.
This evaluates model performance on test data and generates validation reports.
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.preprocessing import Preprocessor
from src.data_processing.feature_engineering import FeatureEngineering
from src.models.sleep_quality import SleepQualityModel
from src.models.transfer_learning import TransferLearning
from src.monitoring.data_drift import DataDriftDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('validate_models.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate trained models for Sleep Insights App')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='src/config/model_config.yaml',
        help='Path to model configuration file'
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data/raw',
        help='Directory containing the test data'
    )
    
    parser.add_argument(
        '--models-dir', 
        type=str, 
        default='models',
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='reports',
        help='Directory to save validation reports'
    )
    
    parser.add_argument(
        '--test-only', 
        action='store_true',
        help='Only use test data (no validation on training data)'
    )
    
    parser.add_argument(
        '--drift-detection', 
        action='store_true',
        help='Run data drift detection'
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
    Path(os.path.join(output_dir, 'figures')).mkdir(parents=True, exist_ok=True)
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

def load_models(models_dir):
    """Load trained models."""
    models = {}
    
    # Load sleep quality model
    sleep_quality_path = os.path.join(models_dir, 'sleep_quality_model.pt')
    if os.path.exists(sleep_quality_path):
        models['sleep_quality'] = SleepQualityModel.load(sleep_quality_path)
        logger.info(f"Loaded sleep quality model from {sleep_quality_path}")
    else:
        logger.warning(f"Sleep quality model not found at {sleep_quality_path}")
    
    # Load transfer learning model if available
    transfer_learning_path = os.path.join(models_dir, 'transfer_learning_model.pt')
    if os.path.exists(transfer_learning_path):
        models['transfer_learning'] = TransferLearning.load(transfer_learning_path)
        logger.info(f"Loaded transfer learning model from {transfer_learning_path}")
    
    return models

def generate_performance_plots(y_true, y_pred, output_dir, model_name):
    """Generate performance plots for regression models."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - True vs Predicted Values')
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, 'figures', f'{model_name}_true_vs_pred.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved plot to {plot_path}")
    
    # Plot error distribution
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'{model_name} - Error Distribution')
    plt.grid(True)
    
    error_path = os.path.join(output_dir, 'figures', f'{model_name}_error_dist.png')
    plt.savefig(error_path)
    plt.close()
    logger.info(f"Saved plot to {error_path}")
    
    return [plot_path, error_path]

def generate_confusion_matrix(y_true, y_pred, output_dir, model_name):
    """Generate confusion matrix for classification models."""
    # Convert to labels if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    
    plot_path = os.path.join(output_dir, 'figures', f'{model_name}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved confusion matrix to {plot_path}")
    
    return plot_path

def generate_validation_report(model_name, metrics, plots, output_dir):
    """Generate validation report in markdown format."""
    report_path = os.path.join(output_dir, f'{model_name}_validation_report.md')
    
    with open(report_path, 'w') as f:
        f.write(f"# {model_name} Model Validation Report\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        for metric, value in metrics.items():
            f.write(f"| {metric} | {value:.4f} |\n")
        
        f.write("\n## Visualizations\n\n")
        
        for plot in plots:
            plot_name = os.path.basename(plot)
            f.write(f"![{plot_name}](figures/{plot_name})\n\n")
    
    logger.info(f"Saved validation report to {report_path}")
    return report_path

def run_drift_detection(reference_data, current_data, config, output_dir):
    """Run data drift detection and generate reports."""
    drift_detector = DataDriftDetector(config.get('drift_detection', {}))
    
    drift_results = drift_detector.detect_drift(reference_data, current_data)
    
    # Create drift report
    report_path = os.path.join(output_dir, 'data_drift_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Data Drift Detection Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Drift Detected:** {drift_results['drift_detected']}\n")
        f.write(f"- **Drift Score:** {drift_results['drift_score']:.4f}\n")
        f.write(f"- **Threshold:** {drift_results['threshold']:.4f}\n\n")
        
        f.write("## Feature-level Drift\n\n")
        f.write("| Feature | Drift Score | Drift Detected |\n")
        f.write("|---------|-------------|----------------|\n")
        
        for feature, score in drift_results['feature_drift_scores'].items():
            is_drifted = score > drift_results['threshold']
            f.write(f"| {feature} | {score:.4f} | {'Yes' if is_drifted else 'No'} |\n")
    
    logger.info(f"Saved drift detection report to {report_path}")
    
    # Generate drift visualization
    # Generate drift visualization
    plt.figure(figsize=(12, 8))
    
    features = list(drift_results['feature_drift_scores'].keys())
    scores = list(drift_results['feature_drift_scores'].values())
    
    # Sort features by drift score for better visualization
    sorted_indices = np.argsort(scores)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    
    # Color code based on threshold
    colors = ['red' if score > drift_results['threshold'] else 'blue' for score in sorted_scores]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_features[:20], sorted_scores[:20], color=colors[:20])
    plt.axvline(x=drift_results['threshold'], color='red', linestyle='--', label='Threshold')
    plt.xlabel('Drift Score')
    plt.ylabel('Features')
    plt.title('Feature Drift Scores (Top 20 Features)')
    plt.legend()
    plt.tight_layout()
    
    drift_plot_path = os.path.join(output_dir, 'figures', 'feature_drift_scores.png')
    plt.savefig(drift_plot_path)
    plt.close()
    logger.info(f"Saved drift visualization to {drift_plot_path}")
    
    return report_path, drift_plot_path

def main():
    """Main function to validate models."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
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
    features_df, targets_df = feature_engineer.engineer_features(processed_data)
    
    # Load models
    models = load_models(args.models_dir)
    
    # Validate each model
    for model_name, model in models.items():
        logger.info(f"Validating {model_name} model...")
        
        # Make predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(features_df)
        else:
            predictions = model(torch.tensor(features_df.values, dtype=torch.float32)).detach().numpy()
        
        # Calculate metrics based on model type
        if model_name == 'sleep_quality':
            target = targets_df['sleep_quality']
            # Regression metrics
            metrics = {
                'mse': mean_squared_error(target, predictions),
                'rmse': np.sqrt(mean_squared_error(target, predictions)),
                'mae': mean_absolute_error(target, predictions),
                'r2': r2_score(target, predictions)
            }
            
            # Generate plots
            plots = generate_performance_plots(target, predictions, args.output_dir, model_name)
            
        else:
            # Classification metrics (assuming multiclass)
            if isinstance(predictions, np.ndarray) and len(predictions.shape) > 1 and predictions.shape[1] > 1:
                pred_classes = np.argmax(predictions, axis=1)
            else:
                pred_classes = predictions
                
            if isinstance(targets_df['sleep_quality'], np.ndarray) and len(targets_df['sleep_quality'].shape) > 1:
                true_classes = np.argmax(targets_df['sleep_quality'], axis=1)
            else:
                true_classes = targets_df['sleep_quality']
            
            metrics = {
                'accuracy': accuracy_score(true_classes, pred_classes),
                'precision_macro': precision_score(true_classes, pred_classes, average='macro'),
                'recall_macro': recall_score(true_classes, pred_classes, average='macro'),
                'f1_macro': f1_score(true_classes, pred_classes, average='macro')
            }
            
            # Generate confusion matrix
            plots = [generate_confusion_matrix(true_classes, pred_classes, args.output_dir, model_name)]
        
        # Generate validation report
        generate_validation_report(model_name, metrics, plots, args.output_dir)
    
    # Run drift detection if requested
    if args.drift_detection:
        logger.info("Running data drift detection...")
        
        # For demonstration, we'll use half the data as reference and half as current
        # In a real scenario, you might have specific reference and current datasets
        ref_idx = np.random.choice(len(features_df), size=len(features_df) // 2, replace=False)
        current_idx = np.array(list(set(range(len(features_df))) - set(ref_idx)))
        
        reference_data = features_df.iloc[ref_idx]
        current_data = features_df.iloc[current_idx]
        
        run_drift_detection(reference_data, current_data, config, args.output_dir)
    
    logger.info("Model validation complete!")

if __name__ == "__main__":
    main()