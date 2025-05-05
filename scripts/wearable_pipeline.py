# scripts/wearable_pipeline.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end pipeline for wearable data processing, model training, and visualization.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import necessary modules
from scripts.integrate_wearable_data import main as integrate_main
from scripts.visualize_wearable_data import main as visualize_main

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run end-to-end wearable data pipeline')
    
    parser.add_argument(
        '--sleep-data', 
        type=str, 
        default='data/raw/sleep_data.csv',
        help='Path to sleep data CSV file'
    )
    
    parser.add_argument(
        '--users-data', 
        type=str, 
        default='data/raw/users.csv',
        help='Path to user profiles CSV file'
    )
    
    parser.add_argument(
        '--wearable-data', 
        type=str, 
        default='data/raw/wearable_data.csv',
        help='Path to wearable data CSV file'
    )
    
    parser.add_argument(
        '--external-data', 
        type=str, 
        default=None,
        help='Path to external factors data CSV file'
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
        '--viz-dir', 
        type=str, 
        default='data/visualizations',
        help='Directory to save visualizations'
    )
    
    parser.add_argument(
        '--skip-integration', 
        action='store_true',
        help='Skip wearable data integration step'
    )
    
    parser.add_argument(
        '--skip-training', 
        action='store_true',
        help='Skip model training step'
    )
    
    parser.add_argument(
        '--skip-visualization', 
        action='store_true',
        help='Skip visualization step'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the full pipeline."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'data/logs/wearable_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    logger = logging.getLogger('WearablePipeline')
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs('data/logs', exist_ok=True)
    
    logger.info("Starting wearable data processing pipeline")
    
    # Step 1: Integrate wearable data
    if not args.skip_integration:
        logger.info("Step 1: Integrating wearable data")
        
        # Set up arguments for integrate_wearable_data.py
        sys.argv = [
            'integrate_wearable_data.py',
            f'--sleep-data={args.sleep_data}',
            f'--users-data={args.users_data}',
            f'--wearable-data={args.wearable_data}',
            f'--output-dir={args.output_dir}',
            f'--model-dir={args.model_dir}'
        ]
        
        if not args.skip_training:
            sys.argv.append('--train-model')
        
        try:
            integrate_main()
            logger.info("Wearable data integration completed successfully")
        except Exception as e:
            logger.error(f"Error during wearable data integration: {str(e)}")
            return
    else:
        logger.info("Skipping wearable data integration step")
    
    # Step 2: Create visualizations
    if not args.skip_visualization:
        logger.info("Step 2: Creating visualizations")
        
        # Set up arguments for visualize_wearable_data.py
        sys.argv = [
            'visualize_wearable_data.py',
            f'--data={os.path.join(args.output_dir, "processed_sleep_data.csv")}',
            f'--output-dir={args.viz_dir}'
        ]
        
        try:
            visualize_main()
            logger.info("Visualization creation completed successfully")
        except Exception as e:
            logger.error(f"Error during visualization creation: {str(e)}")
            return
    else:
        logger.info("Skipping visualization step")
    
    logger.info("Wearable data processing pipeline completed successfully")

if __name__ == "__main__":
    main()