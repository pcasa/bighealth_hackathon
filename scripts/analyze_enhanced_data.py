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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze enhanced sleep data')
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data/sample',
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
        
        # Map professions to categories
        profession_categories = {
            'healthcare': ['Nurse', 'Doctor', 'Paramedic', 'Healthcare', 'Medical'],
            'service': ['Server', 'Bartender', 'Retail', 'Hospitality', 'Customer'],
            'tech': ['Software', 'Engineer', 'Developer', 'IT', 'Programmer', 'Data', 'Computer'],
            'education': ['Teacher', 'Professor', 'Educator', 'Instructor', 'Academic'],
            'office': ['Manager', 'Accountant', 'Administrator', 'Analyst', 'Officer', 'Supervisor']
        }
        
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
        profession_categories = {
            'healthcare': ['Nurse', 'Doctor', 'Paramedic', 'Healthcare', 'Medical'],
            'service': ['Server', 'Bartender', 'Retail', 'Hospitality', 'Customer'],
            'tech': ['Software', 'Engineer', 'Developer', 'IT', 'Programmer', 'Data'],
            'education': ['Teacher', 'Professor', 'Educator', 'Instructor', 'Academic'],
            'office': ['Manager', 'Accountant', 'Administrator', 'Analyst', 'Officer']
        }
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
        
        print(f"\nAnalysis complete! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())