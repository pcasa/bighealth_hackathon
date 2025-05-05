# scripts/visualize_wearable_data.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to create visualizations for wearable sleep data.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create visualizations for wearable sleep data')
    
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Path to processed sleep data with wearable info'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/visualizations',
        help='Directory to save visualizations'
    )
    
    parser.add_argument(
        '--user-id', 
        type=str, 
        default=None,
        help='Filter visualizations to a specific user'
    )
    
    return parser.parse_args()

def create_visualizations(data, output_dir, user_id=None):
    """Create visualizations for wearable sleep data."""
    # Filter to user if specified
    if user_id:
        data = data[data['user_id'] == user_id]
        if len(data) == 0:
            print(f"No data found for user {user_id}")
            return
        
        # Create user-specific output directory
        output_dir = os.path.join(output_dir, user_id)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure date is datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort by date
    data = data.sort_values('date')
    
    # 1. Sleep Stage Distribution
    if all(col in data.columns for col in ['deep_sleep_percentage', 'light_sleep_percentage', 'rem_sleep_percentage']):
        plt.figure(figsize=(10, 6))
        
        # Prepare data for stacked bar chart
        stage_data = data[['date', 'deep_sleep_percentage', 'light_sleep_percentage', 'rem_sleep_percentage', 'awake_percentage']]
        stage_data = stage_data.set_index('date')
        
        # Plot stacked bars
        ax = stage_data.plot(kind='bar', stacked=True, 
                           color=['#2c3e50', '#3498db', '#9b59b6', '#e74c3c'])
        
        plt.title('Sleep Stage Distribution by Night')
        plt.xlabel('Date')
        plt.ylabel('Proportion')
        plt.legend(['Deep Sleep', 'Light Sleep', 'REM Sleep', 'Awake'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sleep_stage_distribution.png'))
        plt.close()
    
    # 2. Heart Rate Variability Trend
    if 'heart_rate_variability' in data.columns:
        plt.figure(figsize=(12, 6))
        
        plt.plot(data['date'], data['heart_rate_variability'], marker='o', linestyle='-', color='#2980b9')
        
        # Add a smoothed trend line
        try:
            from scipy.signal import savgol_filter
            hrv_smooth = savgol_filter(data['heart_rate_variability'], 
                                      min(5, len(data) - 2 if len(data) > 2 else 1), # Window size
                                      1) # Polynomial order
            plt.plot(data['date'], hrv_smooth, linestyle='-', color='#e74c3c', alpha=0.7)
        except:
            # If smoothing fails, just skip it
            pass
        
        plt.title('Heart Rate Variability Trend')
        plt.xlabel('Date')
        plt.ylabel('HRV (ms)')
        plt.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Good HRV')
        plt.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Fair HRV')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hrv_trend.png'))
        plt.close()
    
    # 3. Sleep Quality Correlation with Heart Rate
    if all(col in data.columns for col in ['sleep_efficiency', 'average_heart_rate']):
        plt.figure(figsize=(10, 6))
        
        sns.scatterplot(x='average_heart_rate', y='sleep_efficiency', data=data, alpha=0.7)
        
        # Add trendline
        sns.regplot(x='average_heart_rate', y='sleep_efficiency', data=data, 
                   scatter=False, line_kws={"color": "red"})
        
        plt.title('Sleep Efficiency vs. Average Heart Rate')
        plt.xlabel('Average Heart Rate (bpm)')
        plt.ylabel('Sleep Efficiency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sleep_efficiency_vs_heart_rate.png'))
        plt.close()
    
    # 4. Sleep Stage vs. Heart Rate Variability
    if all(col in data.columns for col in ['deep_sleep_percentage', 'heart_rate_variability']):
        plt.figure(figsize=(10, 6))
        
        sns.scatterplot(x='heart_rate_variability', y='deep_sleep_percentage', data=data, 
                       color='blue', alpha=0.7)
        
        # Add trendline
        sns.regplot(x='heart_rate_variability', y='deep_sleep_percentage', data=data, 
                   scatter=False, line_kws={"color": "blue"})
        
        if 'rem_sleep_percentage' in data.columns:
            sns.scatterplot(x='heart_rate_variability', y='rem_sleep_percentage', data=data, 
                           color='purple', alpha=0.7)
            
            # Add trendline
            sns.regplot(x='heart_rate_variability', y='rem_sleep_percentage', data=data, 
                       scatter=False, line_kws={"color": "purple"})
        
        plt.title('Sleep Stages vs. Heart Rate Variability')
        plt.xlabel('Heart Rate Variability (ms)')
        plt.ylabel('Percentage of Total Sleep')
        plt.legend(['Deep Sleep', 'Deep Sleep Trend', 'REM Sleep', 'REM Sleep Trend'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sleep_stages_vs_hrv.png'))
        plt.close()
    
    # 5. Sleep Quality Metrics Dashboard
    plt.figure(figsize=(15, 10))
    
    # Create a 2x2 grid of subplots for key metrics
    plt.subplot(2, 2, 1)
    plt.plot(data['date'], data['sleep_efficiency'], 'o-', color='#3498db')
    plt.title('Sleep Efficiency')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(data['date'], data['sleep_duration_hours'], 'o-', color='#2ecc71')
    plt.axhline(y=7, color='green', linestyle='--', alpha=0.5)
    plt.axhline(y=9, color='green', linestyle='--', alpha=0.5)
    plt.title('Sleep Duration (hours)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    if 'deep_sleep_percentage' in data.columns:
        plt.plot(data['date'], data['deep_sleep_percentage'], 'o-', color='#2c3e50')
        plt.axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
        plt.title('Deep Sleep %')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    if 'heart_rate_variability' in data.columns:
        plt.plot(data['date'], data['heart_rate_variability'], 'o-', color='#e74c3c')
        plt.axhline(y=50, color='green', linestyle='--', alpha=0.5)
        plt.title('Heart Rate Variability (ms)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    elif 'rem_sleep_percentage' in data.columns:
        plt.plot(data['date'], data['rem_sleep_percentage'], 'o-', color='#9b59b6')
        plt.axhline(y=0.25, color='green', linestyle='--', alpha=0.5)
        plt.title('REM Sleep %')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_quality_dashboard.png'))
    plt.close()
    
    # 6. Correlation Matrix
    numeric_cols = ['sleep_efficiency', 'sleep_duration_hours']
    if 'deep_sleep_percentage' in data.columns:
        numeric_cols.append('deep_sleep_percentage')
    if 'rem_sleep_percentage' in data.columns:
        numeric_cols.append('rem_sleep_percentage')
    if 'light_sleep_percentage' in data.columns:
        numeric_cols.append('light_sleep_percentage')
    if 'heart_rate_variability' in data.columns:
        numeric_cols.append('heart_rate_variability')
    if 'average_heart_rate' in data.columns:
        numeric_cols.append('average_heart_rate')
    if 'blood_oxygen' in data.columns:
        numeric_cols.append('blood_oxygen')
    
    if len(numeric_cols) > 3:  # Only create if we have enough metrics
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        
        plt.title('Correlation Matrix of Sleep Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()
    
    # If user-specific, create a summary card
    if user_id:
        create_user_summary(data, output_dir, user_id)

def create_user_summary(data, output_dir, user_id):
    """Create a summary card for a specific user."""
    # Calculate key metrics
    avg_efficiency = data['sleep_efficiency'].mean()
    avg_duration = data['sleep_duration_hours'].mean()
    
    # Stage averages
    deep_pct = data['deep_sleep_percentage'].mean() if 'deep_sleep_percentage' in data.columns else None
    rem_pct = data['rem_sleep_percentage'].mean() if 'rem_sleep_percentage' in data.columns else None
    light_pct = data['light_sleep_percentage'].mean() if 'light_sleep_percentage' in data.columns else None
    
    # HRV
    avg_hrv = data['heart_rate_variability'].mean() if 'heart_rate_variability' in data.columns else None
    
    # Heart rate
    avg_hr = data['average_heart_rate'].mean() if 'average_heart_rate' in data.columns else None
    min_hr = data['min_heart_rate'].mean() if 'min_heart_rate' in data.columns else None
    
    # Create a summary card with matplotlib
    plt.figure(figsize=(12, 8))
    
    # Title
    plt.text(0.5, 0.95, f"Sleep Summary for User {user_id}", 
             horizontalalignment='center', fontsize=20, fontweight='bold')
    
    # Date range
    plt.text(0.5, 0.9, f"Data Range: {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}", 
             horizontalalignment='center', fontsize=14)
    
    # Key metrics
    plt.text(0.1, 0.80, "Sleep Efficiency:", fontsize=14, fontweight='bold')
    plt.text(0.4, 0.80, f"{avg_efficiency:.1%}", fontsize=14)
    
    plt.text(0.1, 0.75, "Sleep Duration:", fontsize=14, fontweight='bold')
    plt.text(0.4, 0.75, f"{avg_duration:.1f} hours", fontsize=14)
    
    if deep_pct is not None:
        plt.text(0.1, 0.70, "Deep Sleep:", fontsize=14, fontweight='bold')
        plt.text(0.4, 0.70, f"{deep_pct:.1%}", fontsize=14)
    
    if rem_pct is not None:
        plt.text(0.1, 0.65, "REM Sleep:", fontsize=14, fontweight='bold')
        plt.text(0.4, 0.65, f"{rem_pct:.1%}", fontsize=14)
    
    if light_pct is not None:
        plt.text(0.1, 0.60, "Light Sleep:", fontsize=14, fontweight='bold')
        plt.text(0.4, 0.60, f"{light_pct:.1%}", fontsize=14)
    
    if avg_hrv is not None:
        plt.text(0.1, 0.55, "Heart Rate Variability:", fontsize=14, fontweight='bold')
        plt.text(0.4, 0.55, f"{avg_hrv:.1f} ms", fontsize=14)
    
    if avg_hr is not None:
        plt.text(0.1, 0.50, "Average Heart Rate:", fontsize=14, fontweight='bold')
        plt.text(0.4, 0.50, f"{avg_hr:.1f} bpm", fontsize=14)
    
    # Sleep quality assessment
    quality_score = 0
    quality_factors = []
    
    if avg_efficiency >= 0.85:
        quality_score += 1
        quality_factors.append("Good sleep efficiency")
    
    if 7 <= avg_duration <= 9:
        quality_score += 1
        quality_factors.append("Optimal sleep duration")
    
    if deep_pct is not None and deep_pct >= 0.2:
        quality_score += 1
        quality_factors.append("Good deep sleep percentage")
    
    if rem_pct is not None and rem_pct >= 0.2:
        quality_score += 1
        quality_factors.append("Good REM sleep percentage")
    
    if avg_hrv is not None and avg_hrv >= 40:
        quality_score += 1
        quality_factors.append("Good heart rate variability")
    
    # Overall assessment
    assessment = ""
    if quality_score >= 4:
        assessment = "Excellent Sleep Quality"
    elif quality_score >= 3:
        assessment = "Good Sleep Quality"
    elif quality_score >= 2:
        assessment = "Fair Sleep Quality"
    else:
        assessment = "Sleep Quality Needs Improvement"
    
    plt.text(0.1, 0.40, "Overall Assessment:", fontsize=16, fontweight='bold')
    plt.text(0.1, 0.35, assessment, fontsize=16, color='blue')
    
    # Key factors
    plt.text(0.1, 0.30, "Key Factors:", fontsize=14, fontweight='bold')
    
    for i, factor in enumerate(quality_factors):
        plt.text(0.1, 0.25 - (i * 0.05), f"• {factor}", fontsize=12)
    
    # Hide axes
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_summary.png'))
    plt.close()

def main():
    """Main function to create visualizations."""
    args = parse_args()
    
    # Load data
    print(f"Loading data from {args.data}")
    data = pd.read_csv(args.data)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(data, args.output_dir, args.user_id)
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()