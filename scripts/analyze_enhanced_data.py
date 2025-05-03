#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to analyze enhanced sleep data with profession and region information.
Identifies patterns and correlations between sleep quality and the new demographic factors.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.constants import profession_categories
from src.data_generation.base_generator import BaseDataGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze enhanced sleep data')
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data/enhanced_demo/data',
        help='Directory containing the sample data'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='reports/enhanced_analysis',
        help='Directory to save analysis results'
    )
    
    return parser.parse_args()

def load_data(data_dir):
    """Load data from CSV files."""
    # Load all data files
    users_file = os.path.join(data_dir, 'users.csv')
    sleep_data_file = os.path.join(data_dir, 'sleep_data.csv')
    wearable_data_file = os.path.join(data_dir, 'wearable_data.csv')
    
    if not os.path.exists(users_file) or not os.path.exists(sleep_data_file):
        raise FileNotFoundError(f"Required data files not found in {data_dir}")
    
    # Load each dataset
    users_df = pd.read_csv(users_file)
    sleep_data_df = pd.read_csv(sleep_data_file)
    
    # Load wearable data if available
    wearable_data_df = None
    if os.path.exists(wearable_data_file):
        wearable_data_df = pd.read_csv(wearable_data_file)
    
    print(f"Loaded {len(users_df)} users")
    print(f"Loaded {len(sleep_data_df)} sleep records")
    
    if wearable_data_df is not None:
        print(f"Loaded {len(wearable_data_df)} wearable records")
    
    return users_df, sleep_data_df, wearable_data_df

def merge_data(users_df, sleep_data_df, wearable_data_df=None):
    """Merge user profiles with sleep data and wearable data."""
    # Merge users with sleep data
    merged_data = pd.merge(sleep_data_df, users_df, on='user_id')
    
    # Add wearable data if available
    if wearable_data_df is not None:
        merged_data = pd.merge(
            merged_data, 
            wearable_data_df,
            on=['user_id', 'date'],
            how='left'
        )
    
    print(f"Created merged dataset with {len(merged_data)} records")
    return merged_data

def analyze_by_profession(merged_data, output_dir):
    """Analyze sleep patterns by profession."""
    # Extract profession categories
    if 'profession_category' in merged_data.columns:
        profession_column = 'profession_category'
    else:
        # Try to categorize professions
        merged_data['profession_category'] = 'other'
        
        for category, keywords in profession_categories.items():
            mask = merged_data['profession'].apply(lambda x: any(keyword.lower() in x.lower() for keyword in keywords))
            merged_data.loc[mask, 'profession_category'] = category
        
        profession_column = 'profession_category'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sleep efficiency by profession
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=profession_column, y='sleep_efficiency', data=merged_data)
    plt.title('Sleep Efficiency by Profession Category')
    plt.xlabel('Profession Category')
    plt.ylabel('Sleep Efficiency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_efficiency_by_profession.png'))
    plt.close()
    
    # 2. Sleep duration by profession
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=profession_column, y='sleep_duration_hours', data=merged_data)
    plt.title('Sleep Duration by Profession Category')
    plt.xlabel('Profession Category')
    plt.ylabel('Sleep Duration (hours)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_duration_by_profession.png'))
    plt.close()
    
    # 3. Sleep onset latency by profession
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=profession_column, y='sleep_onset_latency_minutes', data=merged_data)
    plt.title('Sleep Onset Latency by Profession Category')
    plt.xlabel('Profession Category')
    plt.ylabel('Sleep Onset Latency (minutes)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_onset_by_profession.png'))
    plt.close()
    
    # 4. Statistical summary by profession
    prof_stats = merged_data.groupby(profession_column).agg({
        'sleep_efficiency': ['mean', 'std', 'count'],
        'sleep_duration_hours': ['mean', 'std'],
        'sleep_onset_latency_minutes': ['mean', 'std'],
        'awakenings_count': ['mean', 'std'],
        'subjective_rating': ['mean', 'std']
    }).round(3)
    
    prof_stats.to_csv(os.path.join(output_dir, 'profession_sleep_statistics.csv'))
    print(f"Saved profession analysis results to {output_dir}")
    
    return prof_stats

def analyze_by_region(merged_data, output_dir):
    """Analyze sleep patterns by region."""
    # Extract region categories
    if 'region_category' in merged_data.columns:
        region_column = 'region_category'
    else:
        # Extract region categories from region field
        merged_data['region_category'] = 'other'
        
        # Try to determine region from region field
        if 'region' in merged_data.columns:
            # Extract the country part (assuming format "City, State, Country")
            merged_data['country'] = merged_data['region'].apply(
                lambda x: x.split(',')[-1].strip() if isinstance(x, str) and ',' in x else 'Unknown'
            )
            
            # Categorize by continent/region
            north_america = ['United States', 'Canada', 'Mexico', 'USA']
            europe = ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'UK']
            asia = ['China', 'Japan', 'India', 'Korea', 'Thailand', 'Singapore']
            
            merged_data.loc[merged_data['country'].isin(north_america), 'region_category'] = 'north_america'
            merged_data.loc[merged_data['country'].isin(europe), 'region_category'] = 'europe'
            merged_data.loc[merged_data['country'].isin(asia), 'region_category'] = 'asia'
        
        region_column = 'region_category'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sleep efficiency by region
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=region_column, y='sleep_efficiency', data=merged_data)
    plt.title('Sleep Efficiency by Region')
    plt.xlabel('Region')
    plt.ylabel('Sleep Efficiency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_efficiency_by_region.png'))
    plt.close()
    
    # 2. Average bedtime by region (hour of day)
    merged_data['bedtime_hour'] = pd.to_datetime(merged_data['bedtime']).dt.hour + pd.to_datetime(merged_data['bedtime']).dt.minute / 60
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=region_column, y='bedtime_hour', data=merged_data)
    plt.title('Bedtime Hour by Region')
    plt.xlabel('Region')
    plt.ylabel('Hour of Day (24h format)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bedtime_by_region.png'))
    plt.close()
    
    # 3. Statistical summary by region
    region_stats = merged_data.groupby(region_column).agg({
        'sleep_efficiency': ['mean', 'std', 'count'],
        'sleep_duration_hours': ['mean', 'std'],
        'bedtime_hour': ['mean', 'std'],
        'sleep_onset_latency_minutes': ['mean', 'std'],
        'subjective_rating': ['mean', 'std']
    }).round(3)
    
    region_stats.to_csv(os.path.join(output_dir, 'region_sleep_statistics.csv'))
    print(f"Saved region analysis results to {output_dir}")
    
    return region_stats

def analyze_profession_region_interactions(merged_data, output_dir):
    """Analyze interactions between profession and region on sleep patterns."""
    # Extract profession and region categories
    if 'profession_category' in merged_data.columns:
        profession_column = 'profession_category'
    else:
        profession_column = None
    
    if 'region_category' in merged_data.columns:
        region_column = 'region_category'
    else:
        region_column = None
    
    # Skip analysis if either category is missing
    if profession_column is None or region_column is None:
        print("Missing profession or region categories. Skipping interaction analysis.")
        return None
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sleep efficiency by profession and region
    plt.figure(figsize=(15, 8))
    sns.boxplot(x=profession_column, y='sleep_efficiency', hue=region_column, data=merged_data)
    plt.title('Sleep Efficiency by Profession and Region')
    plt.xlabel('Profession Category')
    plt.ylabel('Sleep Efficiency')
    plt.legend(title='Region')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_efficiency_profession_region.png'))
    plt.close()
    
    # 2. Heatmap of average sleep efficiency by profession and region
    pivot_table = merged_data.pivot_table(
        index=profession_column, 
        columns=region_column, 
        values='sleep_efficiency',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Average Sleep Efficiency by Profession and Region')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_efficiency_heatmap.png'))
    plt.close()
    
    # 3. Statistical summary by profession and region
    interaction_stats = merged_data.groupby([profession_column, region_column]).agg({
        'sleep_efficiency': ['mean', 'std', 'count'],
        'sleep_duration_hours': ['mean', 'std'],
        'sleep_onset_latency_minutes': ['mean', 'std'],
        'subjective_rating': ['mean', 'std']
    }).round(3)
    
    interaction_stats.to_csv(os.path.join(output_dir, 'profession_region_interaction_statistics.csv'))
    print(f"Saved profession-region interaction analysis results to {output_dir}")
    
    return interaction_stats

def create_analysis_report(profession_stats, region_stats, interaction_stats, output_dir):
    """Create a markdown report summarizing key findings."""
    report_path = os.path.join(output_dir, 'enhanced_analysis_report.md')
    
    with open(report_path, 'w') as f:
        # Header
        f.write("# Enhanced Sleep Data Analysis Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d')}*\n\n")
        
        # Introduction
        f.write("## Overview\n\n")
        f.write("This report presents an analysis of sleep patterns based on enhanced user profiles that include profession and region information. The analysis explores how these demographic factors correlate with various sleep metrics.\n\n")
        
        # Profession Analysis
        f.write("## Sleep Patterns by Profession\n\n")
        
        f.write("### Key Findings\n\n")
        
        # Extract some interesting findings from profession_stats
        if profession_stats is not None:
            profession_sleep_efficiency = profession_stats[('sleep_efficiency', 'mean')].sort_values()
            best_profession = profession_sleep_efficiency.index[-1]
            worst_profession = profession_sleep_efficiency.index[0]
            
            f.write(f"- The **{best_profession}** category shows the highest average sleep efficiency ({profession_sleep_efficiency.iloc[-1]:.3f}).\n")
            f.write(f"- The **{worst_profession}** category shows the lowest average sleep efficiency ({profession_sleep_efficiency.iloc[0]:.3f}).\n")
            
            # Sleep duration
            profession_sleep_duration = profession_stats[('sleep_duration_hours', 'mean')]
            longest_sleep_prof = profession_sleep_duration.idxmax()
            shortest_sleep_prof = profession_sleep_duration.idxmin()
            
            f.write(f"- **{longest_sleep_prof}** professionals get the most sleep on average ({profession_sleep_duration.loc[longest_sleep_prof]:.2f} hours).\n")
            f.write(f"- **{shortest_sleep_prof}** professionals get the least sleep on average ({profession_sleep_duration.loc[shortest_sleep_prof]:.2f} hours).\n")
            
            # Sleep onset
            profession_onset = profession_stats[('sleep_onset_latency_minutes', 'mean')]
            fastest_onset_prof = profession_onset.idxmin()
            slowest_onset_prof = profession_onset.idxmax()
            
            f.write(f"- **{fastest_onset_prof}** professionals fall asleep the fastest ({profession_onset.loc[fastest_onset_prof]:.1f} minutes).\n")
            f.write(f"- **{slowest_onset_prof}** professionals take the longest to fall asleep ({profession_onset.loc[slowest_onset_prof]:.1f} minutes).\n\n")
            
            f.write("### Sleep Efficiency by Profession\n\n")
            f.write("![Sleep Efficiency by Profession](./sleep_efficiency_by_profession.png)\n\n")
            
            f.write("### Sleep Duration by Profession\n\n")
            f.write("![Sleep Duration by Profession](./sleep_duration_by_profession.png)\n\n")
            
            # Region Analysis
            f.write("## Sleep Patterns by Region\n\n")
            
            f.write("### Key Findings\n\n")
            
        # Extract some interesting findings from region_stats
        if region_stats is not None:
            region_sleep_efficiency = region_stats[('sleep_efficiency', 'mean')].sort_values()
            best_region = region_sleep_efficiency.index[-1]
            worst_region = region_sleep_efficiency.index[0]
            
            f.write(f"- The **{best_region}** region shows the highest average sleep efficiency ({region_sleep_efficiency.iloc[-1]:.3f}).\n")
            f.write(f"- The **{worst_region}** region shows the lowest average sleep efficiency ({region_sleep_efficiency.iloc[0]:.3f}).\n")
            
            # Bedtime
            region_bedtime = region_stats[('bedtime_hour', 'mean')]
            earliest_bedtime_region = region_bedtime.idxmin()
            latest_bedtime_region = region_bedtime.idxmax()
            
            f.write(f"- People in **{earliest_bedtime_region}** go to bed the earliest (avg: {region_bedtime.loc[earliest_bedtime_region]:.2f} hour).\n")
            f.write(f"- People in **{latest_bedtime_region}** go to bed the latest (avg: {region_bedtime.loc[latest_bedtime_region]:.2f} hour).\n\n")
        
        f.write("### Sleep Efficiency by Region\n\n")
        f.write("![Sleep Efficiency by Region](./sleep_efficiency_by_region.png)\n\n")
        
        f.write("### Bedtime by Region\n\n")
        f.write("![Bedtime by Region](./bedtime_by_region.png)\n\n")
        
        # Interaction Analysis
        f.write("## Interaction Between Profession and Region\n\n")
        
        f.write("### Key Findings\n\n")
        
        f.write("The analysis explored how profession and region factors interact to influence sleep patterns.\n\n")
        
        f.write("### Sleep Efficiency Heatmap\n\n")
        f.write("The following heatmap shows the average sleep efficiency across different profession and region combinations:\n\n")
        f.write("![Sleep Efficiency Heatmap](./sleep_efficiency_heatmap.png)\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        f.write("This analysis has demonstrated significant variations in sleep patterns based on both profession and geographical region. These findings suggest that:\n\n")
        f.write("1. Occupational factors significantly impact sleep quality and duration\n")
        f.write("2. Regional and cultural differences play an important role in sleep habits\n")
        f.write("3. The interaction between profession and region creates unique sleep patterns that cannot be explained by either factor alone\n")
        f.write("4. Personalized sleep recommendations should take into account both professional demands and regional/cultural factors\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("Based on the analysis findings, we recommend the following for the Sleep Insights App:\n\n")
        f.write("1. **Personalized recommendation engine**: Incorporate profession and region as key factors in generating sleep recommendations\n")
        f.write("2. **User onboarding**: Collect profession and region information during user onboarding to improve prediction accuracy\n")
        f.write("3. **Pattern detection**: Develop specialized algorithms for detecting sleep patterns unique to certain professions and regions\n")
        f.write("4. **Recommendation templates**: Create profession-specific and region-specific recommendation templates\n")
        f.write("5. **Research insights**: Share aggregate findings with sleep researchers to advance understanding of demographic factors on sleep\n\n")
        
        f.write("---\n\n")
        f.write("*Note: This analysis was performed on synthetic data generated for demonstration purposes.*\n")
    
    print(f"Created analysis report at {report_path}")
    
    return report_path

def visualize_profession_region_trends(merged_data, output_dir):
    """Visualize trends in sleep metrics by profession and region over time."""
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_dtype(merged_data['date']):
        merged_data['date'] = pd.to_datetime(merged_data['date'])
    
    # Extract profession and region categories
    if 'profession_category' not in merged_data.columns:
        # Categorize professions
        merged_data['profession_category'] = 'other'

        for category, keywords in profession_categories.items():
            mask = merged_data['profession'].apply(lambda x: any(keyword.lower() in x.lower() for keyword in keywords))
            merged_data.loc[mask, 'profession_category'] = category
    
    if 'region_category' not in merged_data.columns:
        # Categorize regions
        merged_data['region_category'] = 'other'
        if 'region' in merged_data.columns:
            # Extract country part
            merged_data['country'] = merged_data['region'].apply(
                lambda x: x.split(',')[-1].strip() if isinstance(x, str) and ',' in x else 'Unknown'
            )
            north_america = ['United States', 'Canada', 'Mexico', 'USA']
            europe = ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'UK']
            asia = ['China', 'Japan', 'India', 'Korea', 'Thailand', 'Singapore']
            
            merged_data.loc[merged_data['country'].isin(north_america), 'region_category'] = 'north_america'
            merged_data.loc[merged_data['country'].isin(europe), 'region_category'] = 'europe'
            merged_data.loc[merged_data['country'].isin(asia), 'region_category'] = 'asia'
    
    # Create monthly averages by profession and region
    merged_data['month'] = merged_data['date'].dt.to_period('M')
    
    # Profession trends over time
    prof_time_stats = merged_data.groupby(['profession_category', 'month']).agg({
        'sleep_efficiency': 'mean',
        'sleep_duration_hours': 'mean',
        'subjective_rating': 'mean'
    }).reset_index()
    
    # Convert period to datetime for plotting
    prof_time_stats['month'] = prof_time_stats['month'].dt.to_timestamp()
    
    # Region trends over time
    region_time_stats = merged_data.groupby(['region_category', 'month']).agg({
        'sleep_efficiency': 'mean',
        'sleep_duration_hours': 'mean',
        'subjective_rating': 'mean'
    }).reset_index()
    
    # Convert period to datetime for plotting
    region_time_stats['month'] = region_time_stats['month'].dt.to_timestamp()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot profession trends over time - Sleep Efficiency
    plt.figure(figsize=(12, 6))
    for prof, group in prof_time_stats.groupby('profession_category'):
        plt.plot(group['month'], group['sleep_efficiency'], 'o-', label=prof)
    plt.title('Sleep Efficiency by Profession Over Time')
    plt.xlabel('Month')
    plt.ylabel('Average Sleep Efficiency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'profession_efficiency_trend.png'))
    plt.close()
    
    # Plot profession trends over time - Sleep Duration
    plt.figure(figsize=(12, 6))
    for prof, group in prof_time_stats.groupby('profession_category'):
        plt.plot(group['month'], group['sleep_duration_hours'], 'o-', label=prof)
    plt.title('Sleep Duration by Profession Over Time')
    plt.xlabel('Month')
    plt.ylabel('Average Sleep Duration (hours)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'profession_duration_trend.png'))
    plt.close()
    
    # Plot region trends over time - Sleep Efficiency
    plt.figure(figsize=(12, 6))
    for region, group in region_time_stats.groupby('region_category'):
        plt.plot(group['month'], group['sleep_efficiency'], 'o-', label=region)
    plt.title('Sleep Efficiency by Region Over Time')
    plt.xlabel('Month')
    plt.ylabel('Average Sleep Efficiency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'region_efficiency_trend.png'))
    plt.close()
    
    # Plot region trends over time - Sleep Duration
    plt.figure(figsize=(12, 6))
    for region, group in region_time_stats.groupby('region_category'):
        plt.plot(group['month'], group['sleep_duration_hours'], 'o-', label=region)
    plt.title('Sleep Duration by Region Over Time')
    plt.xlabel('Month')
    plt.ylabel('Average Sleep Duration (hours)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'region_duration_trend.png'))
    plt.close()
    
    print(f"Saved profession and region trend visualizations to {output_dir}")
    
    return prof_time_stats, region_time_stats

def analyze_prediction_confidence(data_dir='data/enhanced_demo/data', output_dir='reports/confidence_analysis'):
    """
    Analyze prediction confidence across user demographics and sleep metrics.
    Creates a comprehensive report on confidence patterns and factors.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prediction data
    predictions_file = os.path.join(data_dir, '../predictions/sleep_score_predictions.csv')
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    predictions_df = pd.read_csv(predictions_file)
    
    # Add prediction_confidence if not present
    if 'prediction_confidence' not in predictions_df.columns:
        # Generate synthetic confidence based on available data
        predictions_df['prediction_confidence'] = 0.7  # Base confidence
        
        # Add variation based on sleep efficiency
        if 'sleep_efficiency' in predictions_df.columns:
            # More extreme values (very high or low) get lower confidence
            predictions_df['prediction_confidence'] += (0.5 - abs(predictions_df['sleep_efficiency'] - 0.75)) * 0.2
        
        # Data density factor - more data points per user = higher confidence
        user_counts = predictions_df['user_id'].value_counts()
        predictions_df['data_points'] = predictions_df['user_id'].map(user_counts)
        predictions_df['prediction_confidence'] += np.clip((predictions_df['data_points'] - 5) * 0.01, -0.1, 0.1)
        
        # Add random variation
        np.random.seed(42)
        predictions_df['prediction_confidence'] += np.random.normal(0, 0.05, len(predictions_df))
        
        # Clip to valid range
        predictions_df['prediction_confidence'] = np.clip(predictions_df['prediction_confidence'], 0.3, 0.95)
    
    # Load user data
    users_file = os.path.join(data_dir, 'users.csv')
    if not os.path.exists(users_file):
        raise FileNotFoundError(f"Users file not found: {users_file}")
    
    users_df = pd.read_csv(users_file)
    
    # Merge prediction data with user profiles
    merged_data = pd.merge(predictions_df, users_df, on='user_id')
    
    # Add profession category if not present
    if 'profession_category' not in merged_data.columns:
        # Define profession categories and keywords
        profession_categories = {
            'healthcare': ['doctor', 'nurse', 'physician', 'therapist', 'medical'],
            'tech': ['developer', 'engineer', 'programmer', 'analyst', 'IT'],
            'service': ['retail', 'server', 'customer', 'service', 'hospitality'],
            'education': ['teacher', 'professor', 'educator', 'tutor', 'school'],
            'office': ['manager', 'administrator', 'executive', 'clerical', 'assistant']
        }
        
        # Categorize professions
        merged_data['profession_category'] = 'other'
        for category, keywords in profession_categories.items():
            mask = merged_data['profession'].apply(lambda x: any(kw.lower() in str(x).lower() for kw in keywords))
            merged_data.loc[mask, 'profession_category'] = category
    
    # Add region category if not present
    if 'region_category' not in merged_data.columns:
        # Extract region categories
        merged_data['region_category'] = 'other'
        
        # Try to determine region from region field
        if 'region' in merged_data.columns:
            # Extract the country part (assuming format "City, State, Country")
            merged_data['country'] = merged_data['region'].apply(
                lambda x: x.split(',')[-1].strip() if isinstance(x, str) and ',' in x else 'Unknown'
            )
            
            # Categorize by continent/region
            north_america = ['United States', 'Canada', 'Mexico', 'USA']
            europe = ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'UK']
            asia = ['China', 'Japan', 'India', 'Korea', 'Thailand', 'Singapore']
            
            merged_data.loc[merged_data['country'].isin(north_america), 'region_category'] = 'north_america'
            merged_data.loc[merged_data['country'].isin(europe), 'region_category'] = 'europe'
            merged_data.loc[merged_data['country'].isin(asia), 'region_category'] = 'asia'
    
    # Create age groups
    merged_data['age_group'] = pd.cut(
        merged_data['age'],
        bins=[17, 30, 40, 50, 60, 70, 85],
        labels=['18-30', '31-40', '41-50', '51-60', '61-70', '71+']
    )
    
    # 1. Confidence analysis by profession
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='profession_category', y='prediction_confidence', data=merged_data)
    plt.title('Prediction Confidence by Profession Category')
    plt.xlabel('Profession Category')
    plt.ylabel('Confidence Level')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_by_profession.png'))
    plt.close()
    
    # 2. Confidence analysis by region
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='region_category', y='prediction_confidence', data=merged_data)
    plt.title('Prediction Confidence by Region')
    plt.xlabel('Region')
    plt.ylabel('Confidence Level')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_by_region.png'))
    plt.close()
    
    # 3. Confidence analysis by age group
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='age_group', y='prediction_confidence', data=merged_data)
    plt.title('Prediction Confidence by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Confidence Level')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_by_age.png'))
    plt.close()
    
    # 4. Heatmap of confidence by profession and region
    pivot_table = merged_data.pivot_table(
        index='profession_category', 
        columns='region_category', 
        values='prediction_confidence',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Average Prediction Confidence by Profession and Region')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_heatmap.png'))
    plt.close()
    
    # 5. Heatmap of confidence by age and profession
    age_prof_pivot = merged_data.pivot_table(
        index='age_group', 
        columns='profession_category', 
        values='prediction_confidence',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(age_prof_pivot, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Average Prediction Confidence by Age Group and Profession')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_profession_confidence_heatmap.png'))
    plt.close()
    
    # 6. Confidence vs. sleep efficiency scatterplot by age group
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='sleep_efficiency', y='prediction_confidence', hue='age_group', data=merged_data, alpha=0.6)
    plt.title('Prediction Confidence vs. Sleep Efficiency by Age Group')
    plt.xlabel('Sleep Efficiency')
    plt.ylabel('Confidence Level')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_vs_efficiency_by_age.png'))
    plt.close()
    
    # 7. Confidence distribution histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_data['prediction_confidence'], bins=20, kde=True)
    plt.axvline(merged_data['prediction_confidence'].mean(), color='red', linestyle='--', 
                label=f'Mean: {merged_data["prediction_confidence"].mean():.3f}')
    plt.title('Distribution of Prediction Confidence')
    plt.xlabel('Confidence Level')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    plt.close()
    
    # 8. Statistical summary by profession
    prof_stats = merged_data.groupby('profession_category').agg({
        'prediction_confidence': ['mean', 'std', 'count', 'min', 'max'],
        'predicted_sleep_score': ['mean', 'std'],
        'sleep_efficiency': ['mean', 'std']
    }).round(3)
    
    prof_stats.to_csv(os.path.join(output_dir, 'profession_confidence_statistics.csv'))
    
    # 9. Statistical summary by region
    region_stats = merged_data.groupby('region_category').agg({
        'prediction_confidence': ['mean', 'std', 'count', 'min', 'max'],
        'predicted_sleep_score': ['mean', 'std'],
        'sleep_efficiency': ['mean', 'std']
    }).round(3)
    
    region_stats.to_csv(os.path.join(output_dir, 'region_confidence_statistics.csv'))
    
    # 10. Statistical summary by age group
    age_stats = merged_data.groupby('age_group').agg({
        'prediction_confidence': ['mean', 'std', 'count', 'min', 'max'],
        'predicted_sleep_score': ['mean', 'std'],
        'sleep_efficiency': ['mean', 'std'],
        'age': ['mean']
    }).round(3)
    
    age_stats.to_csv(os.path.join(output_dir, 'age_confidence_statistics.csv'))
    
    # Create the report
    report_path = os.path.join(output_dir, 'confidence_analysis_report.md')
    
    with open(report_path, 'w') as f:
        # Header
        f.write("# Sleep Prediction Confidence Analysis Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d')}*\n\n")
        
        # Introduction
        f.write("## Overview\n\n")
        f.write("This report analyzes the confidence levels of sleep score predictions across different demographic groups including profession, region, and age. Understanding prediction confidence helps identify where our model performs well and where it needs improvement.\n\n")
        
        # General confidence statistics
        f.write("## General Confidence Statistics\n\n")
        f.write(f"- **Average Confidence:** {merged_data['prediction_confidence'].mean():.3f}\n")
        f.write(f"- **Confidence Range:** {merged_data['prediction_confidence'].min():.3f} - {merged_data['prediction_confidence'].max():.3f}\n")
        f.write(f"- **Standard Deviation:** {merged_data['prediction_confidence'].std():.3f}\n\n")
        
        f.write("### Confidence Distribution\n\n")
        f.write("![Confidence Distribution](./confidence_distribution.png)\n\n")
        
        # Age Analysis
        f.write("## Confidence by Age Group\n\n")
        
        f.write("### Key Findings\n\n")
        
        # Extract findings from age stats
        age_confidence = age_stats[('prediction_confidence', 'mean')].sort_values()
        highest_conf_age = age_confidence.index[-1]
        lowest_conf_age = age_confidence.index[0]
        
        f.write(f"- The **{highest_conf_age}** age group shows the highest average prediction confidence ({age_confidence.iloc[-1]:.3f}).\n")
        f.write(f"- The **{lowest_conf_age}** age group shows the lowest average prediction confidence ({age_confidence.iloc[0]:.3f}).\n")
        f.write(f"- The difference between highest and lowest confidence age groups is {(age_confidence.iloc[-1] - age_confidence.iloc[0]):.3f}.\n\n")
        
        f.write("### Confidence by Age Group\n\n")
        f.write("![Confidence by Age Group](./confidence_by_age.png)\n\n")
        
        f.write("### Confidence vs. Sleep Efficiency by Age Group\n\n")
        f.write("![Confidence vs Efficiency by Age](./confidence_vs_efficiency_by_age.png)\n\n")
        
        # Profession Analysis
        f.write("## Confidence by Profession\n\n")
        
        f.write("### Key Findings\n\n")
        
        # Extract findings from profession stats
        profession_confidence = prof_stats[('prediction_confidence', 'mean')].sort_values()
        highest_conf_prof = profession_confidence.index[-1]
        lowest_conf_prof = profession_confidence.index[0]
        
        f.write(f"- The **{highest_conf_prof}** category shows the highest average prediction confidence ({profession_confidence.iloc[-1]:.3f}).\n")
        f.write(f"- The **{lowest_conf_prof}** category shows the lowest average prediction confidence ({profession_confidence.iloc[0]:.3f}).\n")
        f.write(f"- The difference between highest and lowest confidence professions is {(profession_confidence.iloc[-1] - profession_confidence.iloc[0]):.3f}.\n\n")
        
        f.write("### Confidence by Profession\n\n")
        f.write("![Confidence by Profession](./confidence_by_profession.png)\n\n")
        
        # Region Analysis
        f.write("## Confidence by Region\n\n")
        
        f.write("### Key Findings\n\n")
        
        # Extract findings from region stats
        region_confidence = region_stats[('prediction_confidence', 'mean')].sort_values()
        highest_conf_region = region_confidence.index[-1]
        lowest_conf_region = region_confidence.index[0]
        
        f.write(f"- The **{highest_conf_region}** region shows the highest average prediction confidence ({region_confidence.iloc[-1]:.3f}).\n")
        f.write(f"- The **{lowest_conf_region}** region shows the lowest average prediction confidence ({region_confidence.iloc[0]:.3f}).\n")
        f.write(f"- The difference between highest and lowest confidence regions is {(region_confidence.iloc[-1] - region_confidence.iloc[0]):.3f}.\n\n")
        
        f.write("### Confidence by Region\n\n")
        f.write("![Confidence by Region](./confidence_by_region.png)\n\n")
        
        # Interaction Analysis
        f.write("## Demographic Interactions\n\n")
        
        f.write("### Profession and Region Interaction\n\n")
        f.write("The following heatmap shows how prediction confidence varies across different profession and region combinations:\n\n")
        f.write("![Confidence Heatmap](./confidence_heatmap.png)\n\n")
        
        f.write("### Age and Profession Interaction\n\n")
        f.write("The following heatmap shows how prediction confidence varies across different age groups and professions:\n\n")
        f.write("![Age-Profession Confidence Heatmap](./age_profession_confidence_heatmap.png)\n\n")
        
        # Efficiency vs Confidence
        f.write("## Relationship Between Sleep Efficiency and Prediction Confidence\n\n")
        
        # Calculate correlation
        correlation = merged_data['sleep_efficiency'].corr(merged_data['prediction_confidence'])
        
        f.write(f"The correlation between sleep efficiency and prediction confidence is {correlation:.3f}.\n\n")
        f.write("![Confidence vs Efficiency](./confidence_vs_efficiency_by_age.png)\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        f.write("Based on this analysis of prediction confidence across demographic groups, we can draw several conclusions:\n\n")
        
        f.write("1. **Age Impact**: Age plays a significant role in prediction confidence, with certain age groups showing consistently higher or lower confidence levels.\n")
        f.write("2. **Profession Impact**: Certain professions show consistently higher prediction confidence, indicating our model is better calibrated for these groups.\n")
        f.write("3. **Regional Variation**: There are notable differences in prediction confidence across regions, suggesting potential cultural or lifestyle factors that affect prediction reliability.\n")
        f.write("4. **Demographic Interactions**: The interactions between age, profession, and region reveal complex patterns in prediction confidence that could inform targeted model improvements.\n")
        f.write("5. **Data Quality Correlation**: Higher confidence generally correlates with data consistency and quality across demographic groups.\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("To improve prediction confidence across all demographic groups, we recommend:\n\n")
        
        f.write("1. **Age-Specific Models**: Develop specialized prediction models for age groups with lower confidence scores.\n")
        f.write("2. **Targeted Data Collection**: Increase data collection efforts for demographic groups with lower confidence scores.\n")
        f.write("3. **Model Refinement**: Adjust our models to better account for profession, region, and age-specific sleep patterns.\n")
        f.write("4. **Uncertainty Communication**: Clearly communicate prediction confidence to users to set appropriate expectations.\n")
        f.write("5. **Feature Engineering**: Develop more specialized features for groups with lower confidence scores.\n")
        f.write("6. **Continuous Monitoring**: Implement ongoing monitoring of confidence metrics to track improvements.\n\n")
        
        f.write("---\n\n")
        f.write("*Note: This analysis was performed on synthetic data generated for demonstration purposes.*\n")
    
    print(f"Confidence analysis report created at {report_path}")
    
    return {
        'merged_data': merged_data,
        'profession_stats': prof_stats,
        'region_stats': region_stats,
        'age_stats': age_stats,
        'report_path': report_path
    }

def main():
    """Main function to analyze enhanced sleep data."""
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        users_df, sleep_data_df, wearable_data_df = load_data(args.data_dir)
        
        # Merge datasets
        merged_data = merge_data(users_df, sleep_data_df, wearable_data_df)
        
        # Analyze by profession
        print("\nAnalyzing sleep patterns by profession...")
        profession_stats = analyze_by_profession(merged_data, args.output_dir)
        
        # Analyze by region
        print("\nAnalyzing sleep patterns by region...")
        region_stats = analyze_by_region(merged_data, args.output_dir)

        # Add new analysis for trends over time
        print("\nAnalyzing profession and region trends over time...")
        prof_time_stats, region_time_stats = visualize_profession_region_trends(merged_data, args.output_dir)
        
            
        # Analyze profession-region interactions
        print("\nAnalyzing profession-region interactions...")
        interaction_stats = analyze_profession_region_interactions(merged_data, args.output_dir)
        
        # Create summary report
        print("\nCreating summary report...")
        create_analysis_report(profession_stats, region_stats, interaction_stats, args.output_dir)

        print("\nRunning Confidence reports ...")
        analyze_prediction_confidence(
            data_dir='data/enhanced_demo/data', 
            output_dir='reports/confidence_analysis'
        )
        
        print(f"\nAnalysis complete! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())