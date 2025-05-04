"""
Module for generating sleep data visualizations.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def generate_sleep_visualizations(user_id, sleep_data, wearable_data, metrics, output_dir):
    """
    Generate visualizations of sleep patterns for the user.
    
    Args:
        user_id: User identifier
        sleep_data: DataFrame containing sleep records
        wearable_data: DataFrame containing wearable device data (optional)
        metrics: Dictionary of calculated sleep metrics
        output_dir: Directory to save visualizations
        
    Returns:
        str: Path to the user's visualization directory
    """
    # Create user-specific output directory
    user_dir = os.path.join(output_dir, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # Generate visualizations
    _generate_efficiency_trend(sleep_data, user_dir)
    _generate_duration_trend(sleep_data, user_dir)
    
    if 'is_weekend' in sleep_data.columns:
        _generate_weekday_weekend_comparison(sleep_data, user_dir)
    
    _generate_correlation_heatmap(sleep_data, wearable_data, user_dir)
    _generate_quality_distribution(sleep_data, metrics, user_dir)
    
    return user_dir


def _generate_efficiency_trend(sleep_data, output_dir):
    """Generate sleep efficiency trend visualization."""
    plt.figure(figsize=(12, 6))
    sleep_data_sorted = sleep_data.sort_values('date')
    plt.plot(sleep_data_sorted['date'], sleep_data_sorted['sleep_efficiency'] * 100, 'o-', color='blue')
    plt.axhline(y=85, color='green', linestyle='--', alpha=0.7, label='Good (85%)')
    plt.axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='Fair (70%)')
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Poor (50%)')
    plt.title('Sleep Efficiency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sleep Efficiency (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_efficiency_trend.png'))
    plt.close()


def _generate_duration_trend(sleep_data, output_dir):
    """Generate sleep duration trend visualization."""
    plt.figure(figsize=(12, 6))
    sleep_data_sorted = sleep_data.sort_values('date')
    plt.plot(sleep_data_sorted['date'], sleep_data_sorted['sleep_duration_hours'], 'o-', color='purple')
    plt.axhline(y=8, color='green', linestyle='--', alpha=0.7, label='Ideal (8h)')
    plt.axhline(y=7, color='blue', linestyle='--', alpha=0.7, label='Good (7h)')
    plt.axhline(y=6, color='orange', linestyle='--', alpha=0.7, label='Minimum (6h)')
    plt.title('Sleep Duration Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sleep Duration (hours)')
    plt.ylim(0, 10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_duration_trend.png'))
    plt.close()


def _generate_weekday_weekend_comparison(sleep_data, output_dir):
    """Generate weekday vs weekend comparison visualizations."""
    # Duration comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_weekend', y='sleep_duration_hours', data=sleep_data)
    plt.title('Sleep Duration: Weekday vs Weekend')
    plt.xlabel('Weekend?')
    plt.ylabel('Sleep Duration (hours)')
    plt.xticks([0, 1], ['Weekday', 'Weekend'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weekday_weekend_duration.png'))
    plt.close()
    
    # Bedtime comparison if available
    if 'bedtime' in sleep_data.columns:
        sleep_data['bedtime_hour'] = sleep_data['bedtime'].dt.hour + sleep_data['bedtime'].dt.minute / 60
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='is_weekend', y='bedtime_hour', data=sleep_data)
        plt.title('Bedtime: Weekday vs Weekend')
        plt.xlabel('Weekend?')
        plt.ylabel('Hour of Day (24h format)')
        plt.xticks([0, 1], ['Weekday', 'Weekend'])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'weekday_weekend_bedtime.png'))
        plt.close()


def _generate_correlation_heatmap(sleep_data, wearable_data, output_dir):
    """Generate correlation heatmap of sleep metrics."""
    plt.figure(figsize=(12, 10))
    
    # Select relevant columns for correlation analysis
    corr_columns = ['sleep_efficiency', 'sleep_duration_hours', 'time_in_bed_hours', 
                    'sleep_onset_latency_minutes', 'awakenings_count', 'subjective_rating']
    
    # Add wearable data correlations if available
    if wearable_data is not None and len(wearable_data) > 0:
        # Merge sleep and wearable data on date
        merged_data = pd.merge(sleep_data, wearable_data, on=['user_id', 'date'], 
                               how='inner', suffixes=('', '_wearable'))
        
        if len(merged_data) > 0:
            wearable_columns = ['deep_sleep_percentage', 'rem_sleep_percentage', 
                               'heart_rate_variability', 'average_heart_rate']
            available_columns = [col for col in wearable_columns if col in merged_data.columns]
            
            if available_columns:
                corr_columns.extend(available_columns)
                
                # Use merged data for correlation
                corr_data = merged_data[corr_columns].corr()
            else:
                corr_data = sleep_data[corr_columns].corr()
        else:
            corr_data = sleep_data[corr_columns].corr()
    else:
        corr_data = sleep_data[corr_columns].corr()
    
    # Generate heatmap
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Between Sleep Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_metrics_correlation.png'))
    plt.close()


def _generate_quality_distribution(sleep_data, metrics, output_dir):
    """Generate sleep quality distribution visualization."""
    plt.figure(figsize=(10, 6))
    plt.hist(sleep_data['subjective_rating'], bins=10, alpha=0.7, color='teal')
    plt.axvline(x=metrics['avg_sleep_rating'], color='red', linestyle='--', 
                label=f'Average ({metrics["avg_sleep_rating"]:.1f})')
    plt.title('Distribution of Subjective Sleep Quality Ratings')
    plt.xlabel('Sleep Quality Rating (1-10)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_quality_distribution.png'))
    plt.close()