#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to generate personalized sleep insights and visualizations for a specific user.
Takes into account profession and region information to provide tailored recommendations.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.analysis.sleep_metrics import calculate_sleep_metrics
from src.core.analysis.sleep_visualization import generate_sleep_visualizations
from src.core.recommendation.recommendation_generator import generate_personalized_recommendations
from src.core.reporting.report_generator import create_user_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate personalized sleep insights')
    
    parser.add_argument(
        '--user-id', 
        type=str, 
        required=True,
        help='User ID to generate insights for'
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data/enhanced_demo/data',
        help='Directory containing the sleep data'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='reports/user_insights',
        help='Directory to save insights'
    )
    
    return parser.parse_args()


def load_user_data(user_id, data_dir):
    """Load data for a specific user."""
    # Load all data files
    users_file = os.path.join(data_dir, 'users.csv')
    sleep_data_file = os.path.join(data_dir, 'sleep_data.csv')
    wearable_data_file = os.path.join(data_dir, 'wearable_data.csv')
    
    if not os.path.exists(users_file) or not os.path.exists(sleep_data_file):
        raise FileNotFoundError(f"Required data files not found in {data_dir}")
    
    # Load user profile
    users_df = pd.read_csv(users_file)
    user_profile = users_df[users_df['user_id'] == user_id]
    
    if len(user_profile) == 0:
        raise ValueError(f"User with ID {user_id} not found")
    
    # Load sleep data for user
    sleep_data_df = pd.read_csv(sleep_data_file)
    user_sleep_data = sleep_data_df[sleep_data_df['user_id'] == user_id].copy()
    
    if len(user_sleep_data) == 0:
        raise ValueError(f"No sleep data found for user with ID {user_id}")
    
    # Ensure datetime columns are properly formatted
    date_columns = ['date', 'bedtime', 'sleep_onset_time', 'wake_time', 'out_bed_time']
    for col in date_columns:
        if col in user_sleep_data.columns:
            user_sleep_data[col] = pd.to_datetime(user_sleep_data[col])
    
    # Load wearable data if available
    user_wearable_data = None
    if os.path.exists(wearable_data_file):
        wearable_data_df = pd.read_csv(wearable_data_file)
        user_wearable_data = wearable_data_df[wearable_data_df['user_id'] == user_id].copy()
        
        # If wearable data exists, ensure dates are properly formatted
        if len(user_wearable_data) > 0 and 'date' in user_wearable_data.columns:
            user_wearable_data['date'] = pd.to_datetime(user_wearable_data['date'])
    
    # Load recommendations if available
    user_recommendations = None
    recommendations_file = os.path.join(data_dir, 'recommendations.csv')
    if os.path.exists(recommendations_file):
        recommendations_df = pd.read_csv(recommendations_file)
        user_recommendations = recommendations_df[recommendations_df['user_id'] == user_id].copy()
        
        # Ensure timestamp is properly formatted
        if len(user_recommendations) > 0 and 'timestamp' in user_recommendations.columns:
            user_recommendations['timestamp'] = pd.to_datetime(user_recommendations['timestamp'])
    
    print(f"Loaded data for user {user_id}")
    print(f"- Profile information: {len(user_profile)} record")
    print(f"- Sleep data: {len(user_sleep_data)} records")
    if user_wearable_data is not None:
        print(f"- Wearable data: {len(user_wearable_data)} records")
    if user_recommendations is not None:
        print(f"- Recommendations: {len(user_recommendations)} records")
    
    return user_profile.iloc[0], user_sleep_data, user_wearable_data, user_recommendations


def main():
    """Main function to generate user insights."""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Load user data
        user_profile, sleep_data, wearable_data, recommendations = load_user_data(
            args.user_id, args.data_dir
        )
        
        # Calculate sleep metrics
        sleep_metrics = calculate_sleep_metrics(sleep_data)
        
        # Generate visualizations
        visualization_dir = generate_sleep_visualizations(
            args.user_id, sleep_data, wearable_data, sleep_metrics, args.output_dir
        )
        
        # Generate personalized recommendations
        personalized_recommendations = generate_personalized_recommendations(
            user_profile, sleep_metrics, recommendations
        )
        
        # Create user report
        report_path = create_user_report(
            user_profile, sleep_metrics, personalized_recommendations, 
            visualization_dir, args.output_dir
        )
        
        print(f"Generated insights for user {args.user_id}")
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error generating insights: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())