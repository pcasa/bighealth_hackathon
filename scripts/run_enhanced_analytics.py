#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to analyze sleep score trends across demographic dimensions.
This script demonstrates the usage of the enhanced analytics functionality.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.enhanced_sleep_score_analytics import EnhancedSleepScoreAnalytics
from src.data_processing.preprocessing import Preprocessor

def main():
    """Run the enhanced sleep score analytics on sleep data"""
    print("Starting Enhanced Sleep Score Analytics...")
    
    # Load data
    sleep_data_path = 'data/raw/sleep_data.csv'
    users_data_path = 'data/raw/users.csv'
    
    # Check if files exist
    if not os.path.exists(sleep_data_path) or not os.path.exists(users_data_path):
        print(f"Error: Required data files not found.")
        return 1
    
    # Initialize preprocessor and analytics
    preprocessor = Preprocessor()
    analytics = EnhancedSleepScoreAnalytics()
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        sleep_data = pd.read_csv(sleep_data_path)
        users_data = pd.read_csv(users_data_path)
        
        # Merge user profiles with sleep data
        merged_data = pd.merge(sleep_data, users_data, on='user_id')
        
        # Preprocess the data
        processed_data = preprocessor.preprocess_sleep_data(merged_data)
        
        # Ensure date is in datetime format
        processed_data['date'] = pd.to_datetime(processed_data['date'])
        
        # Add required dimension columns
        # Add age range column
        processed_data['age_range'] = processed_data['age'].apply(analytics._get_age_range)
        
        # Add season column
        processed_data['month'] = processed_data['date'].dt.month
        processed_data['season'] = processed_data['month'].apply(analytics._get_season)
        
        # Categorize professions if not already done
        if 'profession_category' not in processed_data.columns:
            processed_data['profession_category'] = processed_data['profession'].apply(analytics._categorize_profession)
        
        # Categorize regions if not already done
        if 'region_category' not in processed_data.columns:
            processed_data['region_category'] = processed_data['region'].apply(analytics._categorize_region)
        
        # Calculate sleep scores
        print("Calculating sleep scores...")
        scored_data = analytics.calculate_sleep_scores(processed_data)
        
        # Set output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"reports/sleep_analysis_{timestamp}"
        
        # Run the full analysis (both static and trend)
        print("Running full analysis...")
        results = analytics.run_full_analysis(scored_data, output_dir)
        
        print(f"Analysis complete! Results saved to {output_dir}")
        print(f"Static analysis: {output_dir}/static_analysis/")
        print(f"Trend analysis: {output_dir}/trend_analysis/")
        
        return 0
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())