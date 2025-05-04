"""
Module for calculating sleep metrics and statistics for user analysis.
"""

import numpy as np
import pandas as pd


def calculate_sleep_metrics(sleep_data):
    """
    Calculate key sleep metrics for the user.
    
    Args:
        sleep_data: DataFrame containing sleep records for a user
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    # Make a copy to avoid modifying the original
    data = sleep_data.copy()
    
    # Ensure numeric columns are properly converted
    numeric_columns = [
        'sleep_efficiency', 'sleep_duration_hours', 'time_in_bed_hours',
        'sleep_onset_latency_minutes', 'awakenings_count', 'subjective_rating'
    ]
    
    for col in numeric_columns:
        if col in data.columns:
            # Convert to numeric, coercing errors to NaN
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    metrics = {}
    
    # Overall averages
    metrics['avg_sleep_efficiency'] = data['sleep_efficiency'].mean()
    metrics['avg_sleep_duration'] = data['sleep_duration_hours'].mean()
    metrics['avg_time_in_bed'] = data['time_in_bed_hours'].mean()
    metrics['avg_sleep_onset'] = data['sleep_onset_latency_minutes'].mean()
    metrics['avg_awakenings'] = data['awakenings_count'].mean()
    metrics['avg_sleep_rating'] = data['subjective_rating'].mean()

    # Recent trend (last 7 days vs previous 7 days)
    if len(data) >= 14:
        # Sort by date and get recent and previous data
        sorted_data = data.sort_values('date')
        recent_data = sorted_data.tail(7)
        previous_data = sorted_data.iloc[-14:-7]
        
        # Calculate efficiency metrics
        metrics['recent_efficiency'] = recent_data['sleep_efficiency'].mean()
        metrics['previous_efficiency'] = previous_data['sleep_efficiency'].mean()
        metrics['efficiency_change'] = metrics['recent_efficiency'] - metrics['previous_efficiency']
        
        # Calculate duration metrics
        metrics['recent_duration'] = recent_data['sleep_duration_hours'].mean()
        metrics['previous_duration'] = previous_data['sleep_duration_hours'].mean()
        metrics['duration_change'] = metrics['recent_duration'] - metrics['previous_duration']
    else:
        metrics['efficiency_change'] = 0
        metrics['duration_change'] = 0
    
    # Weekday vs weekend patterns
    if 'is_weekend' in data.columns:
        # Convert is_weekend to boolean if it's not already
        if data['is_weekend'].dtype != 'bool':
            data['is_weekend'] = data['is_weekend'].astype(bool)
        
        weekday_data = data[~data['is_weekend']]
        weekend_data = data[data['is_weekend']]
        
        if len(weekday_data) > 0 and len(weekend_data) > 0:
            metrics['weekday_efficiency'] = weekday_data['sleep_efficiency'].mean()
            metrics['weekend_efficiency'] = weekend_data['sleep_efficiency'].mean()
            metrics['weekday_duration'] = weekday_data['sleep_duration_hours'].mean()
            metrics['weekend_duration'] = weekend_data['sleep_duration_hours'].mean()
            
            # Bedtime consistency - ensure datetime first
            if 'bedtime' in data.columns and pd.api.types.is_datetime64_dtype(data['bedtime']):
                data['bedtime_hour'] = data['bedtime'].dt.hour + data['bedtime'].dt.minute / 60
                metrics['weekday_bedtime'] = weekday_data['bedtime_hour'].mean() if 'bedtime_hour' in weekday_data.columns else None
                metrics['weekend_bedtime'] = weekend_data['bedtime_hour'].mean() if 'bedtime_hour' in weekend_data.columns else None
                
                if metrics['weekday_bedtime'] is not None and metrics['weekend_bedtime'] is not None:
                    metrics['bedtime_difference'] = abs(metrics['weekend_bedtime'] - metrics['weekday_bedtime'])

    # Best and worst days - handle safely
    try:
        if len(data) > 0 and not data['sleep_efficiency'].isna().all():
            # Get index of max/min efficiency
            best_idx = data['sleep_efficiency'].idxmax()
            worst_idx = data['sleep_efficiency'].idxmin()
            
            # Get date and efficiency values
            if best_idx is not None and best_idx in data.index:
                metrics['best_day'] = data.loc[best_idx, 'date']
                metrics['best_efficiency'] = data.loc[best_idx, 'sleep_efficiency']
            else:
                metrics['best_day'] = None
                metrics['best_efficiency'] = data['sleep_efficiency'].max()
                
            if worst_idx is not None and worst_idx in data.index:
                metrics['worst_day'] = data.loc[worst_idx, 'date']
                metrics['worst_efficiency'] = data.loc[worst_idx, 'sleep_efficiency']
            else:
                metrics['worst_day'] = None
                metrics['worst_efficiency'] = data['sleep_efficiency'].min()
        else:
            # No valid data
            metrics['best_day'] = None
            metrics['worst_day'] = None
            metrics['best_efficiency'] = None
            metrics['worst_efficiency'] = None
    except Exception as e:
        # Fallback if there's an error
        print(f"Error calculating best/worst days: {e}")
        metrics['best_day'] = None
        metrics['worst_day'] = None
        metrics['best_efficiency'] = data['sleep_efficiency'].max() if 'sleep_efficiency' in data else None
        metrics['worst_efficiency'] = data['sleep_efficiency'].min() if 'sleep_efficiency' in data else None
    
    return metrics


def calculate_weekly_trends(sleep_data, metric='sleep_efficiency', weeks=4):
    """
    Calculate weekly trends for a given sleep metric.
    
    Args:
        sleep_data: DataFrame containing sleep records
        metric: The metric to calculate trends for
        weeks: Number of weeks to analyze
        
    Returns:
        dict: Weekly averages and trends
    """
    if len(sleep_data) < 7:
        return {'status': 'insufficient_data'}
    
    # Make sure date is datetime
    sleep_data = sleep_data.copy()
    sleep_data['date'] = pd.to_datetime(sleep_data['date'])
    
    # Ensure metric column is numeric
    sleep_data[metric] = pd.to_numeric(sleep_data[metric], errors='coerce')
    
    # Sort by date
    sleep_data = sleep_data.sort_values('date')
    
    # Get the most recent date
    latest_date = sleep_data['date'].max()
    
    # Create weekly buckets
    weekly_data = []
    for i in range(weeks):
        start_date = latest_date - pd.Timedelta(days=7*(i+1))
        end_date = latest_date - pd.Timedelta(days=7*i)
        
        week_data = sleep_data[(sleep_data['date'] > start_date) & 
                              (sleep_data['date'] <= end_date)]
        
        if len(week_data) > 0:
            avg_value = week_data[metric].mean()
            weekly_data.append({
                'week': f"Week {weeks-i}",
                'average': avg_value,
                'records': len(week_data)
            })
    
    # Calculate trend
    if len(weekly_data) >= 2:
        trend = weekly_data[0]['average'] - weekly_data[-1]['average']
    else:
        trend = 0
    
    return {
        'status': 'success',
        'weekly_data': weekly_data,
        'overall_trend': trend
    }