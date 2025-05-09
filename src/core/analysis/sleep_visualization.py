"""
Module for generating sleep data visualizations.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_sleep_visualizations(user_id, sleep_data, wearable_data, metrics, output_dir):
    """Generate visualizations of sleep patterns for the user."""
    # Create user-specific output directory
    user_dir = os.path.join(output_dir, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # Ensure date is in datetime format
    if 'date' in sleep_data.columns and not pd.api.types.is_datetime64_dtype(sleep_data['date']):
        sleep_data['date'] = pd.to_datetime(sleep_data['date'])
    
    # Sort data by date
    sleep_data_sorted = sleep_data.sort_values('date')
    
    # 1. Sleep efficiency over time
    plt.figure(figsize=(12, 6))
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
    plt.savefig(os.path.join(user_dir, 'sleep_efficiency_trend.png'))
    plt.close()
    
    # 2. Sleep duration over time
    plt.figure(figsize=(12, 6))
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
    plt.savefig(os.path.join(user_dir, 'sleep_duration_trend.png'))
    plt.close()
    
    # 3. Weekday vs Weekend patterns
    if 'is_weekend' in sleep_data.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='is_weekend', y='sleep_duration_hours', data=sleep_data)
        plt.title('Sleep Duration: Weekday vs Weekend')
        plt.xlabel('Weekend?')
        plt.ylabel('Sleep Duration (hours)')
        plt.xticks([0, 1], ['Weekday', 'Weekend'])
        plt.tight_layout()
        plt.savefig(os.path.join(user_dir, 'weekday_weekend_duration.png'))
        plt.close()
        
        # Bedtime consistency
        if 'bedtime' in sleep_data.columns:
            # Convert bedtime to hours since midnight for plotting
            sleep_data['bedtime_hour'] = sleep_data['bedtime'].dt.hour + sleep_data['bedtime'].dt.minute / 60
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='is_weekend', y='bedtime_hour', data=sleep_data)
            plt.title('Bedtime: Weekday vs Weekend')
            plt.xlabel('Weekend?')
            plt.ylabel('Hour of Day (24h format)')
            plt.xticks([0, 1], ['Weekday', 'Weekend'])
            plt.tight_layout()
            plt.savefig(os.path.join(user_dir, 'weekday_weekend_bedtime.png'))
            plt.close()
    
    # 4. Sleep metrics correlation
    plt.figure(figsize=(12, 10))

    # Select only numeric columns for correlation
    sleep_numeric = sleep_data.select_dtypes(include=['number']).columns.tolist()
    corr_columns = [col for col in ['sleep_efficiency', 'sleep_duration_hours', 
                    'time_in_bed_hours', 'sleep_onset_latency_minutes', 
                    'awakenings_count', 'subjective_rating'] 
                    if col in sleep_numeric]

    # Add wearable data correlations if available
    if wearable_data is not None and len(wearable_data) > 0:
        try:
            # Create copies of the dataframes for merging
            sleep_copy = sleep_data.copy()
            wearable_copy = wearable_data.copy()
            
            # Make sure date columns are datetime
            if 'date' in sleep_copy.columns:
                sleep_copy['date'] = pd.to_datetime(sleep_copy['date'])
            if 'date' in wearable_copy.columns:
                wearable_copy['date'] = pd.to_datetime(wearable_copy['date'])
            
            # Make sure user_id is string type
            sleep_copy['user_id'] = sleep_copy['user_id'].astype(str)
            wearable_copy['user_id'] = wearable_copy['user_id'].astype(str)
            
            # Merge data
            merged_data = pd.merge(sleep_copy, wearable_copy, on=['user_id', 'date'], 
                                how='inner', suffixes=('', '_wearable'))
            
            # Ensure all numeric columns are float type
            for col in corr_columns:
                if col in merged_data.columns:
                    merged_data[col] = merged_data[col].astype(float)

            # After merging data but before correlation calculation
            # Convert all numeric columns to float for consistency
            for col in corr_columns + ['heart_rate_variability']:
                if col in merged_data.columns:
                    merged_data[col] = merged_data[col].astype(float)

            # Add variation to any column with single value
            for col in corr_columns + ['heart_rate_variability']:
                if col in merged_data.columns and merged_data[col].nunique() <= 1:
                    print(f"Adding variation to column {col} with {merged_data[col].nunique()} unique values")
                    merged_data[col] = merged_data[col] + np.random.normal(0, 0.01, size=len(merged_data))
            
            if len(merged_data) > 0:
                # Add available wearable columns to correlation list
                wearable_columns = ['deep_sleep_percentage', 'rem_sleep_percentage', 
                                'heart_rate_variability', 'average_heart_rate']
                
                for col in wearable_columns:
                    if col in merged_data.columns and pd.api.types.is_numeric_dtype(merged_data[col]):
                        corr_columns.append(col)
                        merged_data[col] = merged_data[col].astype(float)
                
                # Create correlation matrix from merged data
                corr_data = merged_data[corr_columns].corr()
            else:
                corr_data = sleep_data[corr_columns].corr()
        except Exception as e:
            corr_data = sleep_data[corr_columns].corr()
    else:
        corr_data = sleep_data[corr_columns].corr()

    # Generate heatmap
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Between Sleep Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(user_dir, 'sleep_metrics_correlation.png'))
    plt.close()


    # 5. Sleep quality distribution
    plt.figure(figsize=(10, 6))
    plt.hist(sleep_data['subjective_rating'], bins=10, alpha=0.7, color='teal')
    plt.axvline(x=metrics['avg_sleep_rating'], color='red', linestyle='--', label=f'Average ({metrics["avg_sleep_rating"]:.1f})')
    plt.title('Distribution of Subjective Sleep Quality Ratings')
    plt.xlabel('Sleep Quality Rating (1-10)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(user_dir, 'sleep_quality_distribution.png'))
    plt.close()
    
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
    """Generate correlation heatmap with explicit type checking and conversion."""
    plt.figure(figsize=(12, 10))
    
    # Create a working copy to avoid modifying originals
    sleep_data_copy = sleep_data.copy()
    
    # Convert timestamps to datetime if needed
    for col in sleep_data_copy.columns:
        if pd.api.types.is_object_dtype(sleep_data_copy[col]):
            try:
                # Check if column contains timestamps
                pd.to_datetime(sleep_data_copy[col], errors='raise')
                # If it does, drop it from correlation analysis
                sleep_data_copy.drop(columns=[col], inplace=True)
            except:
                pass
    
    # Explicitly select only numeric columns for correlation
    numeric_columns = sleep_data_copy.select_dtypes(include=['number']).columns.tolist()
    corr_columns = [col for col in ['sleep_efficiency', 'sleep_duration_hours', 
                                   'time_in_bed_hours', 'sleep_onset_latency_minutes', 
                                   'awakenings_count', 'subjective_rating'] 
                   if col in numeric_columns]
    
    # Add wearable metrics if available
    if wearable_data is not None and len(wearable_data) > 0:
        # Create safe merged dataset
        try:
            # Convert dates to datetime for joining
            sleep_date = pd.to_datetime(sleep_data['date']) if 'date' in sleep_data.columns else None
            wearable_date = pd.to_datetime(wearable_data['date']) if 'date' in wearable_data.columns else None
            
            if sleep_date is not None and wearable_date is not None:
                sleep_data_copy['date'] = sleep_date
                wearable_data_copy = wearable_data.copy()
                wearable_data_copy['date'] = wearable_date
                
                # Merge
                merged = pd.merge(sleep_data_copy, wearable_data_copy, on=['user_id', 'date'], 
                                 how='inner', suffixes=('', '_wearable'))
                
                # Get additional numeric columns
                wearable_numeric = [col for col in ['deep_sleep_percentage', 'rem_sleep_percentage', 
                                                  'heart_rate_variability', 'average_heart_rate'] 
                                  if col in merged.columns and pd.api.types.is_numeric_dtype(merged[col])]
                
                if wearable_numeric:
                    corr_columns.extend(wearable_numeric)
                    # Ensure all numeric
                    corr_data = merged[corr_columns].astype(float).corr()
                else:
                    corr_data = sleep_data_copy[corr_columns].astype(float).corr()
            else:
                corr_data = sleep_data_copy[corr_columns].astype(float).corr()
        except Exception as e:
            print(f"Error merging data: {str(e)}")
            corr_data = sleep_data_copy[corr_columns].astype(float).corr()
    else:
        corr_data = sleep_data_copy[corr_columns].astype(float).corr()
    
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