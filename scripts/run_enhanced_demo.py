#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced demo script that simulates a larger, more realistic dataset with:
- Users created throughout the year
- Sleep data for up to 4 months after user creation
- Small percentage of users with wearable data
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import yaml

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation.user_generator import UserGenerator
from src.data_generation.sleep_data_generator import SleepDataGenerator
from src.data_generation.wearable_data_generator import WearableDataGenerator
from src.core.data_processing.preprocessing import Preprocessor
from src.core.recommendation.recommendation_engine import SleepRecommendationEngine
from src.core.models.sleep_quality import SleepQualityModel
from src.data_generation.base_generator import BaseDataGenerator
from src.data_generation.sleep_data_generator import SleepDataGenerator
from src.data_generation.wearable_data_generator import WearableDataGenerator
from src.core.models.improved_sleep_score import ImprovedSleepScoreCalculator
from src.utils.data_validation_fix import ensure_sleep_data_format


def load_config(config_path='src/config/data_generation_config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_enhanced_demo_users(config, count=500, seed=42):
    """
    Create users distributed throughout the year with realistic attributes
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize user generator with config
    user_generator = UserGenerator(config_path='src/config/data_generation_config.yaml')
    
    # Start with base users
    all_users = user_generator.generate_users(count=count)
    
    # Add created_at dates distributed throughout the year
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 10, 20)  # End before our cutoff to allow for data generation
    
    # Generate timestamps - more users in recent months (exponential distribution)
    days_range = (end_date - start_date).days
    # Use exponential distribution to favor recent months
    days_distribution = np.random.exponential(scale=days_range/3, size=count)
    # Cap to our date range and invert (so most recent dates are more frequent)
    days_distribution = np.clip(days_distribution, 0, days_range)
    days_distribution = days_range - days_distribution
    
    # Convert to actual dates
    created_dates = [start_date + timedelta(days=int(days)) for days in days_distribution]
    created_dates.sort()  # Sort chronologically
    
    # Add created_at timestamps to users dataframe
    all_users['created_at'] = [date.strftime('%Y-%m-%d %H:%M:%S') for date in created_dates]
    
    print(f"Generated {len(all_users)} users distributed throughout the year")
    
    return all_users

def generate_enhanced_sleep_data(users_df, config, seed=42):
    """
    Generate sleep data for each user starting from their creation date
    up to 4 months later (or current date, whichever is sooner)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize sleep data generator
    sleep_data_generator = SleepDataGenerator()
    
    all_sleep_data = []
    
    # Process each user individually
    for _, user in users_df.iterrows():
        # Parse creation date
        created_at = datetime.strptime(user['created_at'], '%Y-%m-%d %H:%M:%S')
        
        # Define data generation period (4 months or until current date)
        start_date = created_at  # Use datetime object directly, not .date()
        end_date = min(
            created_at + timedelta(days=120),  # 4 months (120 days)
            datetime(2024, 10, 30)  # Cap at training data cutoff
        )
        
        # Create a single-user dataframe for the generator
        user_df = pd.DataFrame([user.to_dict()])
        
        # Generate sleep data for this user and date range
        user_sleep_data = sleep_data_generator.generate_sleep_data(
            user_df,
            start_date=start_date,
            end_date=end_date
        )
        
        all_sleep_data.append(user_sleep_data)
    
    # Combine all user data
    combined_sleep_data = pd.concat(all_sleep_data)
    
    print(f"Generated {len(combined_sleep_data)} sleep records across {len(users_df)} users")
    
    return combined_sleep_data

def generate_enhanced_wearable_data(sleep_data_df, users_df, wearable_percentage=20, seed=42):
    """
    Generate wearable data for a small percentage of users
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Select subset of users who have wearable devices
    unique_users = sleep_data_df['user_id'].unique()
    num_wearable_users = int(len(unique_users) * wearable_percentage / 100)
    wearable_users = np.random.choice(unique_users, size=num_wearable_users, replace=False)
    
    # Filter sleep data to only users with wearables
    wearable_sleep_data = sleep_data_df[sleep_data_df['user_id'].isin(wearable_users)]
    
    # Initialize wearable data generator
    wearable_generator = WearableDataGenerator(
        config_path='src/config/data_generation_config.yaml',
        device_config_path='src/config/device_profiles.yaml'
    )
    
    # Generate wearable data for the subset
    wearable_data = wearable_generator.generate_wearable_data(wearable_sleep_data, users_df)
    
    print(f"Generated wearable data for {num_wearable_users} users ({wearable_percentage}% of total)")
    print(f"Total wearable records: {len(wearable_data)}")
    
    return wearable_data

def analyze_dataset(users_df, sleep_data_df, wearable_data_df=None):
    """
    Generate basic analytics about the dataset
    """
    # Analyze user distribution
    print("\nUser Demographics:")
    print(f"Total users: {len(users_df)}")
    print(f"Gender distribution: {users_df['gender'].value_counts(normalize=True).multiply(100).round(1).to_dict()}")
    print(f"Age statistics: min={users_df['age'].min()}, max={users_df['age'].max()}, avg={users_df['age'].mean():.1f}")
    print(f"Sleep pattern distribution: {users_df['sleep_pattern'].value_counts(normalize=True).multiply(100).round(1).to_dict()}")
    
    # Analyze sleep data
    print("\nSleep Data Statistics:")
    print(f"Total sleep records: {len(sleep_data_df)}")
    print(f"Average records per user: {len(sleep_data_df) / len(users_df):.1f}")
    
    # Calculate sleep metrics if available
    if 'sleep_efficiency' in sleep_data_df.columns:
        print(f"Average sleep efficiency: {sleep_data_df['sleep_efficiency'].mean():.2f}")
    if 'sleep_duration_hours' in sleep_data_df.columns:
        print(f"Average sleep duration: {sleep_data_df['sleep_duration_hours'].mean():.2f} hours")
    if 'subjective_rating' in sleep_data_df.columns:
        print(f"Average subjective rating: {sleep_data_df['subjective_rating'].mean():.2f}/10")
    
    # Analyze wearable data if available
    if wearable_data_df is not None:
        print("\nWearable Data Statistics:")
        print(f"Total wearable records: {len(wearable_data_df)}")
        print(f"Users with wearable data: {wearable_data_df['user_id'].nunique()}")
        
        if 'device_type' in users_df.columns:
            wearable_users = users_df[users_df['user_id'].isin(wearable_data_df['user_id'].unique())]
            print(f"Device distribution: {wearable_users['device_type'].value_counts(normalize=True).multiply(100).round(1).to_dict()}")

def generate_visualizations(users_df, sleep_data_df, output_dir):
    """
    Generate visualizations of the dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")

    # Ensure date fields are datetime objects
    if 'date' in sleep_data_df.columns and not pd.api.types.is_datetime64_dtype(sleep_data_df['date']):
        sleep_data_df['date'] = pd.to_datetime(sleep_data_df['date'])
        
    if 'created_at' in users_df.columns and not pd.api.types.is_datetime64_dtype(users_df['created_at']):
        users_df['created_at'] = pd.to_datetime(users_df['created_at'])

    # Check if 'created_at' exists, add it if missing
    if 'created_at' not in users_df.columns:
        print("Warning: 'created_at' field missing, generating placeholder values")
        # Generate placeholders for created_at
        start_date = datetime(2024, 1, 1)
        days = np.random.randint(0, 120, size=len(users_df))
        users_df['created_at'] = [start_date + timedelta(days=d) for d in days]
        users_df['created_at'] = users_df['created_at'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    
    
    # 1. User creation timeline
    plt.figure(figsize=(12, 6))
    created_at = pd.to_datetime(users_df['created_at'])
    created_at_counts = created_at.dt.to_period('M').value_counts().sort_index()
    created_at_counts.plot(kind='bar')
    plt.title('User Signups by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of New Users')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'user_signups_by_month.png'))
    plt.close()
    
    # 2. Sleep efficiency distribution
    if 'sleep_efficiency' in sleep_data_df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(sleep_data_df['sleep_efficiency'], bins=30, kde=True)
        plt.title('Distribution of Sleep Efficiency')
        plt.xlabel('Sleep Efficiency')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sleep_efficiency_distribution.png'))
        plt.close()
    
    # 3. Sleep pattern distribution
    plt.figure(figsize=(12, 6))
    sleep_patterns = users_df['sleep_pattern'].value_counts()
    sleep_patterns.plot(kind='pie', autopct='%1.1f%%', textprops={'fontsize': 12})
    plt.title('Distribution of Sleep Patterns')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_pattern_distribution.png'))
    plt.close()
    
    # 4. Average sleep duration by age group
    plt.figure(figsize=(12, 6))
    users_with_sleep = pd.merge(sleep_data_df, users_df[['user_id', 'age']], on='user_id')
    users_with_sleep['age_group'] = pd.cut(users_with_sleep['age'], 
                                           bins=[17, 30, 40, 50, 60, 70, 85], 
                                           labels=['18-30', '31-40', '41-50', '51-60', '61-70', '71+'])
    avg_duration_by_age = users_with_sleep.groupby('age_group')['sleep_duration_hours'].mean()
    avg_duration_by_age.plot(kind='bar')
    plt.title('Average Sleep Duration by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Average Sleep Duration (hours)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_duration_by_age.png'))
    plt.close()
    
    # 5. Sleep data volume over time
    plt.figure(figsize=(12, 6))
    sleep_data_df['date'] = pd.to_datetime(sleep_data_df['date'])
    sleep_data_counts = sleep_data_df.groupby(sleep_data_df['date'].dt.to_period('M')).size()
    sleep_data_counts.plot(kind='line', marker='o')
    plt.title('Sleep Data Volume by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Sleep Records')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_data_volume_by_month.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def run_enhanced_demo(output_dir='data/enhanced_demo', user_count=500, wearable_percentage=20, seed=42):
    """
    Run the enhanced demo with more realistic data generation
    """
    print(f"Starting Enhanced Sleep Insights Demo with {user_count} users...")
    
    # Load configuration
    config = load_config()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'recommendations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Step 1: Generate users distributed throughout the year
    # Import the function from data_generation_script
    from data_generation_script import generate_user_profiles
    users_df = generate_user_profiles(user_count, config_path='src/config/data_generation_config.yaml')
    users_df.to_csv(os.path.join(output_dir, 'data', 'users.csv'), index=False)
    
    # Step 2: Generate sleep data for each user from creation date
    # Import the function from data_generation_script
    from data_generation_script import SleepDataGenerator
    sleep_data_gen = SleepDataGenerator(config_path='src/config/data_generation_config.yaml')
    sleep_data_df = sleep_data_gen.generate_sleep_data(users_df)
    sleep_data_df.to_csv(os.path.join(output_dir, 'data', 'sleep_data.csv'), index=False)
    
    # Step 3: Generate wearable data for a percentage of users
    print(f"\nGenerating wearable data for {wearable_percentage}% of users...")
    from data_generation_script import WearableDataGenerator
    wearable_gen = WearableDataGenerator(config_path='src/config/data_generation_config.yaml', 
                                       device_config_path='src/config/device_profiles.yaml')
    wearable_data_df = wearable_gen.generate_wearable_data(sleep_data_df, users_df)
    wearable_data_df.to_csv(os.path.join(output_dir, 'data', 'wearable_data.csv'), index=False)
    
    # Step 4: Analyze dataset
    sleep_data_df = ensure_sleep_data_format(sleep_data_df)
    analyze_dataset(users_df, sleep_data_df, wearable_data_df)
    
    # Step 5: Generate visualizations
    generate_visualizations(users_df, sleep_data_df, os.path.join(output_dir, 'visualizations'))
    
    # Step 6: Generate sample recommendations for select users
    print("\nGenerating sample recommendations...")
    # Initialize preprocessor and recommendation engine
    preprocessor = Preprocessor()
    recommendation_engine = SleepRecommendationEngine()
    
    # Select a sample of users with enough data for recommendations
    user_data_counts = sleep_data_df['user_id'].value_counts()
    qualified_users = user_data_counts[user_data_counts >= 14].index.tolist()
    
    if len(qualified_users) > 20:
        sample_users = np.random.choice(qualified_users, 20, replace=False)
    else:
        sample_users = qualified_users
    
    # Process each sample user
    all_recommendations = []
    
    for user_id in sample_users:
        # Get user sleep data
        user_data = sleep_data_df[sleep_data_df['user_id'] == user_id].copy()
        
        # Preprocess data
        processed_data = preprocessor.preprocess_sleep_data(user_data)
        
        # Get user profile for profession and region
        user_profile = users_df[users_df['user_id'] == user_id].iloc[0]
        profession = user_profile['profession']
        region = user_profile['region']
        
        # Analyze progress
        progress_data = recommendation_engine.analyze_progress(user_id, processed_data)
        
        # Generate recommendation with profession and region context
        recommendation = recommendation_engine.generate_recommendation(
            user_id, progress_data, profession, region
        )
        
        # Store recommendation
        rec_data = {
            'user_id': user_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': recommendation['message'],
            'confidence': recommendation.get('confidence', 0.5),
            'category': recommendation.get('category', 'unknown')
        }
        all_recommendations.append(rec_data)
        
    # Save all recommendations
    recommendations_df = pd.DataFrame(all_recommendations)
    recommendations_df.to_csv(os.path.join(output_dir, 'recommendations', 'sample_recommendations.csv'), index=False)
    
    print(f"\nEnhanced demo completed successfully!")
    print(f"Results saved to the '{output_dir}' directory:")
    print(f"  - User data: {output_dir}/data/users.csv")
    print(f"  - Sleep data: {output_dir}/data/sleep_data.csv")
    print(f"  - Wearable data: {output_dir}/data/wearable_data.csv")
    print(f"  - Visualizations: {output_dir}/visualizations/")
    print(f"  - Recommendations: {output_dir}/recommendations/")

    # After Step 5 but before Step 6 in run_enhanced_demo function:

    # Generate sleep score predictions for each user
    print("\nGenerating sleep score predictions...")
    sleep_quality_model = SleepQualityModel()
    sleep_quality_model.sleep_score_calculator = ImprovedSleepScoreCalculator()

    # Try to load a pre-trained model if available
    model_path = 'models/sleep_quality_model'
    try:
        sleep_quality_model.load(model_path)
        print("Loaded pre-trained sleep quality model")
    except Exception as e:
        print(f"Could not load pre-trained model: {e}")
        print("Will use default scoring method")

    # Process each user and generate predictions
    all_predictions = []

    # DEBUGGING: Limit to a single user for testing
    # Only process one user for debugging
    debug_user = list(sleep_data_df['user_id'].unique())[0]
    user_data = sleep_data_df[sleep_data_df['user_id'] == debug_user].copy()

    # Process just one row for debugging
    for idx, row in user_data.iloc[:3].iterrows():
        try:
            # Create a full data dictionary
            sleep_data = {
                'sleep_efficiency': row.get('sleep_efficiency', 0.8),
                'sleep_duration_hours': row.get('sleep_duration_hours', 7.0),
                'sleep_onset_latency_minutes': row.get('sleep_onset_latency_minutes', 15),
                'awakenings_count': row.get('awakenings_count', 2),
                'total_awake_minutes': row.get('total_awake_minutes', 20),
                'subjective_rating': row.get('subjective_rating', 7)
            }
            
            # Print the input values for Debugging
            # print(f"Sleep data for user {debug_user}:")
            # for k, v in sleep_data.items():
            #     print(f"  {k}: {v}")
            
            # Debug: Calculate each component score separately
            calculator = sleep_quality_model.sleep_score_calculator
            
            # Use SleepScoreInput to validate the data
            from src.core.models.improved_sleep_score import SleepScoreInput
            validated_data = SleepScoreInput(**sleep_data)
            
            duration_score = calculator._score_duration(validated_data)
            efficiency_score = calculator._score_efficiency(validated_data)
            onset_score = calculator._score_onset(validated_data)
            continuity_score = calculator._score_continuity(validated_data)
            subjective_score = calculator._score_subjective(validated_data)
            
            print(f"Component scores:")
            print(f"  Duration: {duration_score}")
            print(f"  Efficiency: {efficiency_score}")
            print(f"  Onset: {onset_score}")
            print(f"  Continuity: {continuity_score}")
            print(f"  Subjective: {subjective_score}")
            
            # Use SleepQualityModel's method which might return an object or a simple score
            score_result = sleep_quality_model.calculate_sleep_score(
                sleep_data.get('sleep_efficiency', 0.8),
                sleep_data.get('subjective_rating', 7),
                {
                    'deep_sleep_percentage': sleep_data.get('deep_sleep_percentage', 0.2),
                    'rem_sleep_percentage': sleep_data.get('rem_sleep_percentage', 0.25),
                    'sleep_onset_latency_minutes': sleep_data.get('sleep_onset_latency_minutes', 15),
                    'awakenings_count': sleep_data.get('awakenings_count', 2)
                }
            )

            # Handle the result appropriately based on the SleepQualityModel's return type
            if isinstance(score_result, dict) and 'total_score' in score_result:
                sleep_score = score_result['total_score']
            elif hasattr(score_result, 'total_score'):
                sleep_score = score_result.total_score
            else:
                sleep_score = score_result
            print(f"Final score: {sleep_score}")
            
        except Exception as e:
            print(f"Error: {e}")
    # DEBUGGING END: Limit to a single user for testing

    for user_id in sleep_data_df['user_id'].unique():
        # Get user sleep data
        user_data = sleep_data_df[sleep_data_df['user_id'] == user_id].copy()
        
        # Skip users with very little data
        if len(user_data) < 3:
            continue
        
        # Calculate sleep scores
        for idx, row in user_data.iterrows():
            try:
                # Use the sleep_score_calculator from the model
                sleep_score_result = sleep_quality_model.calculate_sleep_score_with_confidence(
                    row.get('sleep_efficiency', 0.8),
                    row.get('subjective_rating', 7),
                    {
                        'deep_sleep_percentage': row.get('deep_sleep_percentage', 0.2),
                        'rem_sleep_percentage': row.get('rem_sleep_percentage', 0.25),
                        'sleep_onset_latency_minutes': row.get('sleep_onset_latency_minutes', 15),
                        'awakenings_count': row.get('awakenings_count', 2)
                    }
                )

                # Store prediction
                prediction = {
                    'user_id': user_id,
                    'date': row['date'],
                    'sleep_efficiency': row.get('sleep_efficiency', 0.8),
                    'sleep_duration_hours': row.get('sleep_duration_hours', 7.0),
                    'predicted_sleep_score': sleep_score_result['score'],
                    'prediction_confidence': sleep_score_result['confidence']
                }
                all_predictions.append(prediction)
            except Exception as e:
                print(f"Error calculating sleep score for user {user_id}: {e}")
                continue

    # Save all predictions
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv(os.path.join(output_dir, 'predictions', 'sleep_score_predictions.csv'), index=False)
        print(f"Generated sleep score predictions for {len(predictions_df)} sleep records")
    else:
        print("No sleep score predictions were generated")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced sleep insights demo')
    parser.add_argument('--output-dir', type=str, default='data/enhanced_demo',
                        help='Directory to save demo output')
    parser.add_argument('--user-count', type=int, default=100,
                        help='Number of users to generate')
    parser.add_argument('--wearable-percentage', type=int, default=20,
                        help='Percentage of users with wearable devices')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    run_enhanced_demo(
        output_dir=args.output_dir,
        user_count=args.user_count,
        wearable_percentage=args.wearable_percentage,
        seed=args.seed
    )