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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

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
        default='data/sample',
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

def calculate_sleep_metrics(sleep_data):
    """Calculate key sleep metrics for the user."""
    metrics = {}
    
    # Overall averages
    metrics['avg_sleep_efficiency'] = sleep_data['sleep_efficiency'].mean()
    metrics['avg_sleep_duration'] = sleep_data['sleep_duration_hours'].mean()
    metrics['avg_time_in_bed'] = sleep_data['time_in_bed_hours'].mean()
    metrics['avg_sleep_onset'] = sleep_data['sleep_onset_latency_minutes'].mean()
    metrics['avg_awakenings'] = sleep_data['awakenings_count'].mean()
    metrics['avg_sleep_rating'] = sleep_data['subjective_rating'].mean()
    
    # Recent trend (last 7 days vs previous 7 days)
    if len(sleep_data) >= 14:
        recent_data = sleep_data.sort_values('date').tail(7)
        previous_data = sleep_data.sort_values('date').iloc[-14:-7]
        
        metrics['recent_efficiency'] = recent_data['sleep_efficiency'].mean()
        metrics['previous_efficiency'] = previous_data['sleep_efficiency'].mean()
        metrics['efficiency_change'] = metrics['recent_efficiency'] - metrics['previous_efficiency']
        
        metrics['recent_duration'] = recent_data['sleep_duration_hours'].mean()
        metrics['previous_duration'] = previous_data['sleep_duration_hours'].mean()
        metrics['duration_change'] = metrics['recent_duration'] - metrics['previous_duration']
    else:
        metrics['efficiency_change'] = 0
        metrics['duration_change'] = 0
    
    # Weekday vs weekend patterns
    if 'is_weekend' in sleep_data.columns:
        weekday_data = sleep_data[~sleep_data['is_weekend']]
        weekend_data = sleep_data[sleep_data['is_weekend']]
        
        if len(weekday_data) > 0 and len(weekend_data) > 0:
            metrics['weekday_efficiency'] = weekday_data['sleep_efficiency'].mean()
            metrics['weekend_efficiency'] = weekend_data['sleep_efficiency'].mean()
            metrics['weekday_duration'] = weekday_data['sleep_duration_hours'].mean()
            metrics['weekend_duration'] = weekend_data['sleep_duration_hours'].mean()
            
            # Bedtime consistency
            if 'bedtime' in sleep_data.columns:
                sleep_data['bedtime_hour'] = sleep_data['bedtime'].dt.hour + sleep_data['bedtime'].dt.minute / 60
                metrics['weekday_bedtime'] = weekday_data['bedtime_hour'].mean()
                metrics['weekend_bedtime'] = weekend_data['bedtime_hour'].mean()
                metrics['bedtime_difference'] = abs(metrics['weekend_bedtime'] - metrics['weekday_bedtime'])
    
    # Best and worst days
    metrics['best_day'] = sleep_data.loc[sleep_data['sleep_efficiency'].idxmax()]['date']
    metrics['worst_day'] = sleep_data.loc[sleep_data['sleep_efficiency'].idxmin()]['date']
    metrics['best_efficiency'] = sleep_data['sleep_efficiency'].max()
    metrics['worst_efficiency'] = sleep_data['sleep_efficiency'].min()
    
    return metrics

def generate_sleep_visualizations(user_id, sleep_data, wearable_data, metrics, output_dir):
    """Generate visualizations of sleep patterns for the user."""
    # Create user-specific output directory
    user_dir = os.path.join(output_dir, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # 1. Sleep efficiency over time
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
    
    # Select relevant columns for correlation analysis
    corr_columns = ['sleep_efficiency', 'sleep_duration_hours', 'time_in_bed_hours', 
                    'sleep_onset_latency_minutes', 'awakenings_count', 'subjective_rating']
    
    # Add wearable data correlations if available
    if wearable_data is not None and len(wearable_data) > 0:
        # Merge sleep and wearable data on date
        merged_data = pd.merge(sleep_data, wearable_data, on=['user_id', 'date'], how='inner', suffixes=('', '_wearable'))
        
        if len(merged_data) > 0:
            wearable_columns = ['deep_sleep_percentage', 'rem_sleep_percentage', 'heart_rate_variability', 'average_heart_rate']
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

def generate_personalized_recommendations(user_profile, sleep_metrics, recommendations):
    """Generate personalized recommendations based on user profile and sleep data."""
    personalized_recommendations = []
    
    # Extract key information
    profession = user_profile['profession']
    age = user_profile['age']
    sleep_pattern = user_profile['sleep_pattern']
    
    # Extract region information (assuming format "City, State, Country")
    region = "Unknown"
    country = "Unknown"
    
    if 'region' in user_profile and isinstance(user_profile['region'], str):
        region = user_profile['region']
        if ',' in region:
            parts = region.split(',')
            if len(parts) >= 3:
                country = parts[-1].strip()
    
    # Determine profession category
    profession_category = 'other'
    profession_categories = {
        'healthcare': ['Nurse', 'Doctor', 'Paramedic', 'Healthcare', 'Medical'],
        'service': ['Server', 'Bartender', 'Retail', 'Hospitality', 'Customer'],
        'tech': ['Software', 'Engineer', 'Developer', 'IT', 'Programmer', 'Data', 'Computer'],
        'education': ['Teacher', 'Professor', 'Educator', 'Instructor', 'Academic'],
        'office': ['Manager', 'Accountant', 'Administrator', 'Analyst', 'Officer', 'Supervisor']
    }
    
    for category, keywords in profession_categories.items():
        if any(keyword.lower() in profession.lower() for keyword in keywords):
            profession_category = category
            break
    
    # Determine region category
    region_category = 'other'
    north_america = ['United States', 'Canada', 'Mexico', 'USA']
    europe = ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'UK']
    asia = ['China', 'Japan', 'India', 'Korea', 'Thailand', 'Singapore']
    
    if country in north_america:
        region_category = 'north_america'
    elif country in europe:
        region_category = 'europe'
    elif country in asia:
        region_category = 'asia'
    
    # Generate general sleep recommendations
    general_rec = {
        'category': 'general',
        'title': 'General Sleep Recommendation',
        'content': f"Based on your sleep pattern ({sleep_pattern}), aim for consistent sleep and wake times, even on weekends. This helps regulate your body's internal clock."
    }
    personalized_recommendations.append(general_rec)
    
    # Generate profession-specific recommendations
    if profession_category == 'healthcare':
        prof_rec = {
            'category': 'profession',
            'title': 'Healthcare Professional Sleep Tips',
            'content': f"As a {profession}, your shifting work schedule can impact sleep quality. Try using blackout curtains and white noise machines to create a consistent sleep environment regardless of when you sleep."
        }
    elif profession_category == 'tech':
        prof_rec = {
            'category': 'profession',
            'title': 'Tech Professional Sleep Tips',
            'content': f"Your profession likely involves significant screen time. Consider using blue light filters on all devices, especially in the evening, and try to disconnect from screens at least 1 hour before bedtime."
        }
    elif profession_category == 'service':
        prof_rec = {
            'category': 'profession',
            'title': 'Service Industry Sleep Tips',
            'content': f"Service positions like yours often involve variable schedules and potentially stressful interactions. Try a 10-minute decompression ritual after work to mentally separate work stress from sleep time."
        }
    elif profession_category == 'education':
        prof_rec = {
            'category': 'profession',
            'title': 'Educator Sleep Tips',
            'content': f"As an educator, work stress and take-home work can affect sleep. Set clear boundaries between work and personal time. Try to finish grading or preparation at least 2 hours before bedtime."
        }
    else:
        prof_rec = {
            'category': 'profession',
            'title': f'Sleep Tips for {profession}',
            'content': f"Consider how your work as a {profession} impacts your sleep schedule and stress levels. Create a transition routine between work and sleep to help your mind unwind."
        }
    
    personalized_recommendations.append(prof_rec)
    
    # Generate region-specific recommendations
    if region_category == 'north_america':
        region_rec = {
            'category': 'region',
            'title': 'Regional Sleep Consideration',
            'content': f"In {country}, many people struggle with work-life balance. Consider setting clear boundaries on work hours and notifications to protect your sleep time."
        }
    elif region_category == 'europe':
        region_rec = {
            'category': 'region',
            'title': 'Regional Sleep Consideration',
            'content': f"In many European countries like {country}, later dinner times can impact sleep quality. Try to eat your last meal at least 3 hours before bedtime for better sleep quality."
        }
    elif region_category == 'asia':
        region_rec = {
            'category': 'region',
            'title': 'Regional Sleep Consideration',
            'content': f"In {country}, urban light pollution and population density can affect sleep quality. Consider using room-darkening curtains and white noise to create an optimal sleep environment."
        }
    else:
        region_rec = {
            'category': 'region',
            'title': 'Regional Sleep Consideration',
            'content': f"Consider how the cultural norms and environment in {region} might be affecting your sleep, including meal times, social expectations, and climate factors."
        }
    
    personalized_recommendations.append(region_rec)
    
    # Add data-driven recommendations based on sleep metrics
    if sleep_metrics['avg_sleep_onset'] > 30:
        onset_rec = {
            'category': 'data',
            'title': 'Improve Sleep Onset Time',
            'content': f"You're taking an average of {sleep_metrics['avg_sleep_onset']:.1f} minutes to fall asleep. Try a relaxation technique like deep breathing or progressive muscle relaxation to reduce sleep onset time."
        }
        personalized_recommendations.append(onset_rec)
    
    if 'bedtime_difference' in sleep_metrics and sleep_metrics['bedtime_difference'] > 1.5:
        consistency_rec = {
            'category': 'data',
            'title': 'Improve Bedtime Consistency',
            'content': f"Your weekend bedtime differs from weekday by {sleep_metrics['bedtime_difference']:.1f} hours. Try to keep this difference under 1 hour to prevent 'social jet lag' and improve overall sleep quality."
        }
        personalized_recommendations.append(consistency_rec)
    
    if sleep_metrics['avg_awakenings'] > 2:
        awakening_rec = {
            'category': 'data',
            'title': 'Reduce Night Awakenings',
            'content': f"You're experiencing about {sleep_metrics['avg_awakenings']:.1f} awakenings per night. Consider factors like room temperature, noise, or light that might be disrupting your sleep. A cooler, darker, and quieter environment often helps reduce awakenings."
        }
        personalized_recommendations.append(awakening_rec)
    
    # Include recent pattern recommendation
    if 'efficiency_change' in sleep_metrics:
        if sleep_metrics['efficiency_change'] > 0.05:
            trend_rec = {
                'category': 'trend',
                'title': 'Recent Improvement',
                'content': f"Your sleep efficiency has improved by {sleep_metrics['efficiency_change']*100:.1f}% in the last week compared to the previous week. Whatever changes you've made recently appear to be working well!"
            }
        elif sleep_metrics['efficiency_change'] < -0.05:
            trend_rec = {
                'category': 'trend',
                'title': 'Recent Decline',
                'content': f"Your sleep efficiency has declined by {abs(sleep_metrics['efficiency_change']*100):.1f}% in the last week. Consider what factors might have changed recently and try to address them."
            }
        else:
            trend_rec = {
                'category': 'trend',
                'title': 'Stable Patterns',
                'content': f"Your sleep patterns have been relatively stable recently. To further improve, try introducing one new sleep hygiene practice this week."
            }
        
        personalized_recommendations.append(trend_rec)
    
    return personalized_recommendations

def create_user_report(user_profile, sleep_metrics, recommendations, visualization_dir, output_dir):
    """Create a comprehensive HTML report for the user."""
    user_id = user_profile['user_id']
    report_path = os.path.join(output_dir, f"{user_id}_sleep_insights.html")
    
    # Format metrics for display
    formatted_metrics = {
        'avg_sleep_efficiency': f"{sleep_metrics['avg_sleep_efficiency']*100:.1f}%",
        'avg_sleep_duration': f"{sleep_metrics['avg_sleep_duration']:.1f} hours",
        'avg_time_in_bed': f"{sleep_metrics['avg_time_in_bed']:.1f} hours",
        'avg_sleep_onset': f"{sleep_metrics['avg_sleep_onset']:.1f} minutes",
        'avg_awakenings': f"{sleep_metrics['avg_awakenings']:.1f}",
        'avg_sleep_rating': f"{sleep_metrics['avg_sleep_rating']:.1f}/10"
    }
    
    # Add trend metrics if available
    if 'efficiency_change' in sleep_metrics:
        efficiency_trend = "↑" if sleep_metrics['efficiency_change'] > 0 else "↓" if sleep_metrics['efficiency_change'] < 0 else "→"
        formatted_metrics['efficiency_trend'] = f"{efficiency_trend} {abs(sleep_metrics['efficiency_change']*100):.1f}%"
        
        duration_trend = "↑" if sleep_metrics['duration_change'] > 0 else "↓" if sleep_metrics['duration_change'] < 0 else "→"
        formatted_metrics['duration_trend'] = f"{duration_trend} {abs(sleep_metrics['duration_change']):.1f} hours"
    
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sleep Insights for User {user_id}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #3a86ff;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .section {{
                margin-bottom: 30px;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
            }}
            .metrics-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
            }}
            .metric-card {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                min-width: 180px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric-title {{
                font-size: 0.9rem;
                color: #666;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 1.4rem;
                font-weight: bold;
                color: #333;
            }}
            .recommendation {{
                background-color: #f0f7ff;
                border-left: 5px solid #3a86ff;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 0 5px 5px 0;
            }}
            .recommendation-title {{
                font-weight: bold;
                margin-bottom: 5px;
                color: #2b5797;
            }}
            .chart-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: space-between;