# scripts/run_demo.py

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation.user_generator import UserGenerator
from src.data_generation.sleep_data_generator import SleepDataGenerator
from src.data_processing.preprocessing import Preprocessor
from src.models.sleep_quality import SleepQualityModel
from src.models.recommendation_engine import SleepRecommendationEngine

def create_demo_users():
    """Create sample users for the demo"""
    # Create one user of each sleep pattern type
    users = [
        {
            'user_id': 'user_normal',
            'age': 35,
            'gender': 'female',
            'sleep_pattern': 'normal',
            'device_type': 'apple_watch',
            'data_consistency': 0.95,
            'sleep_consistency': 0.9
        },
        {
            'user_id': 'user_insomnia',
            'age': 42,
            'gender': 'male',
            'sleep_pattern': 'insomnia',
            'device_type': 'fitbit',
            'data_consistency': 0.85,
            'sleep_consistency': 0.6
        },
        {
            'user_id': 'user_shift_worker',
            'age': 29,
            'gender': 'non-binary',
            'sleep_pattern': 'shift_worker',
            'device_type': 'google_watch',
            'data_consistency': 0.8,
            'sleep_consistency': 0.5
        },
        {
            'user_id': 'user_improving',
            'age': 31,
            'gender': 'female',
            'sleep_pattern': 'insomnia',  # Starts with insomnia but will improve
            'device_type': 'samsung_watch',
            'data_consistency': 0.9,
            'sleep_consistency': 0.7
        }
    ]
    
    return pd.DataFrame(users)

def generate_sleep_data(users_df, days=30):
    """Generate sleep data for demo users"""
    sleep_data_generator = SleepDataGenerator()
    
    # Generate standard sleep data
    all_sleep_data = []
    
    for _, user in users_df.iterrows():
        user_sleep_data = []
        
        # Get pattern parameters
        pattern = user['sleep_pattern']
        
        # Set base bedtime and wake time
        if pattern == 'shift_worker':
            base_bedtime = datetime.combine(datetime.today(), datetime.strptime('08:00', '%H:%M').time())
            base_waketime = datetime.combine(datetime.today(), datetime.strptime('15:00', '%H:%M').time())
        else:
            base_bedtime = datetime.combine(datetime.today(), datetime.strptime('22:30', '%H:%M').time())
            base_waketime = datetime.combine(datetime.today(), datetime.strptime('06:30', '%H:%M').time())
        
        # Generate data for each day
        for day in range(days):
            current_date = datetime.now() - timedelta(days=days-day-1)
            
            # For the 'improving' user, gradually improve sleep metrics
            if user['user_id'] == 'user_improving' and day > 10:
                improvement_factor = min(1.0, (day - 10) / 20)  # Gradually improve
                
                # Adjust sleep efficiency
                sleep_efficiency = 0.65 + (0.25 * improvement_factor)
                
                # Adjust sleep duration
                sleep_duration = 5 + (2 * improvement_factor)
                
                # Adjust awakenings
                awakenings = max(1, int(5 - (3 * improvement_factor)))
                
                # Adjust subjective rating
                rating = 4 + int(5 * improvement_factor)
                
            else:
                # Use pattern-specific values
                if pattern == 'normal':
                    sleep_efficiency = np.random.uniform(0.85, 0.95)
                    sleep_duration = np.random.uniform(6.5, 8.5)
                    awakenings = np.random.randint(0, 3)
                    rating = np.random.randint(7, 10)
                    
                elif pattern == 'insomnia':
                    sleep_efficiency = np.random.uniform(0.6, 0.75)
                    sleep_duration = np.random.uniform(4, 6)
                    awakenings = np.random.randint(3, 7)
                    rating = np.random.randint(2, 6)
                    
                elif pattern == 'shift_worker':
                    sleep_efficiency = np.random.uniform(0.7, 0.85)
                    sleep_duration = np.random.uniform(5, 7)
                    awakenings = np.random.randint(2, 5)
                    rating = np.random.randint(4, 8)
                    
                else:  # Default
                    sleep_efficiency = np.random.uniform(0.7, 0.9)
                    sleep_duration = np.random.uniform(5.5, 7.5)
                    awakenings = np.random.randint(1, 4)
                    rating = np.random.randint(4, 9)
            
            # Skip some days based on consistency
            if np.random.random() > user['data_consistency']:
                continue
                
            # Calculate precise times
            bedtime = base_bedtime - timedelta(days=days-day-1)
            
            # Add variance to bedtime
            bedtime_variance = np.random.randint(-30, 31)  # ±30 minutes
            bedtime += timedelta(minutes=bedtime_variance)
            
            # Sleep onset time (time to fall asleep)
            if pattern == 'insomnia':
                sleep_onset_minutes = np.random.randint(30, 90)
            else:
                sleep_onset_minutes = np.random.randint(5, 20)
                
            sleep_onset_time = bedtime + timedelta(minutes=sleep_onset_minutes)
            
            # Wake time based on sleep duration
            wake_time = sleep_onset_time + timedelta(hours=sleep_duration)
            
            # Create sleep record
            sleep_record = {
                'user_id': user['user_id'],
                'date': current_date.strftime('%Y-%m-%d'),
                'bedtime': bedtime.strftime('%Y-%m-%d %H:%M:%S'),
                'sleep_onset_time': sleep_onset_time.strftime('%Y-%m-%d %H:%M:%S'),
                'wake_time': wake_time.strftime('%Y-%m-%d %H:%M:%S'),
                'time_in_bed_hours': (wake_time - bedtime).total_seconds() / 3600,
                'sleep_duration_hours': sleep_duration,
                'sleep_onset_latency_minutes': sleep_onset_minutes,
                'awakenings_count': awakenings,
                'total_awake_minutes': awakenings * np.random.randint(5, 15),
                'sleep_efficiency': sleep_efficiency,
                'subjective_rating': rating
            }
            
            user_sleep_data.append(sleep_record)
        
        all_sleep_data.extend(user_sleep_data)
    
    return pd.DataFrame(all_sleep_data)

def visualize_user_progress(user_id, sleep_data, recommendations):
    """Create visualizations of user progress and recommendations"""
    # Filter data for this user
    user_data = sleep_data[sleep_data['user_id'] == user_id].copy()
    user_data['date'] = pd.to_datetime(user_data['date'])
    user_data = user_data.sort_values('date')
    
    # Get user recommendations
    user_recs = recommendations[recommendations['user_id'] == user_id].copy()
    if not user_recs.empty:
        user_recs['timestamp'] = pd.to_datetime(user_recs['timestamp'])
        user_recs = user_recs.sort_values('timestamp')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot sleep efficiency
    ax1.plot(user_data['date'], user_data['sleep_efficiency'] * 100, 'o-', label='Sleep Efficiency (%)')
    ax1.set_ylabel('Sleep Efficiency (%)')
    ax1.set_title(f'Sleep Progress for {user_id}')
    ax1.grid(True)
    ax1.legend()
    
    # Plot sleep duration
    ax2.plot(user_data['date'], user_data['sleep_duration_hours'], 'o-', color='green', label='Sleep Duration (hours)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sleep Duration (hours)')
    ax2.grid(True)
    ax2.legend()
    
    # Add recommendation markers
    if not user_recs.empty:
        for _, rec in user_recs.iterrows():
            timestamp = rec['timestamp']
            # Find nearest date in sleep data
            nearest_idx = (user_data['date'] - timestamp).abs().idxmin()
            nearest_date = user_data.loc[nearest_idx, 'date']
            
            # Add marker to both plots
            efficiency = user_data.loc[nearest_idx, 'sleep_efficiency'] * 100
            duration = user_data.loc[nearest_idx, 'sleep_duration_hours']
            
            ax1.annotate('R', (nearest_date, efficiency), 
                        xytext=(0, 15), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'),
                        color='red', fontsize=12, fontweight='bold')
            
            ax2.annotate('R', (nearest_date, duration), 
                        xytext=(0, -15), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'),
                        color='red', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('demo/visualizations', exist_ok=True)
    fig.savefig(f'demo/visualizations/{user_id}_progress.png')
    plt.close(fig)

def run_demo():
    """Run the complete demo pipeline"""
    print("Starting Sleep Insights App Demo...")
    
    # Create output directories
    os.makedirs('demo', exist_ok=True)
    os.makedirs('demo/data', exist_ok=True)
    os.makedirs('demo/recommendations', exist_ok=True)
    
    # Step 1: Create demo users
    print("Creating demo users...")
    users_df = create_demo_users()
    users_df.to_csv('demo/data/demo_users.csv', index=False)
    
    # Step 2: Generate sleep data
    print("Generating sleep data...")
    sleep_data_df = generate_sleep_data(users_df)
    sleep_data_df.to_csv('demo/data/demo_sleep_data.csv', index=False)
    
    # Step 3: Preprocess data
    print("Preprocessing sleep data...")
    preprocessor = Preprocessor()
    processed_data = preprocessor.preprocess_sleep_data(sleep_data_df)
    
    # Step 4: Load models
    print("Loading sleep quality model...")
    sleep_quality_model = SleepQualityModel()
    try:
        sleep_quality_model.load('models/sleep_quality_model')
    except:
        print("Warning: Could not load sleep quality model. Using dummy predictions.")
    
    # Step 5: Initialize recommendation engine
    print("Initializing recommendation engine...")
    recommendation_engine = SleepRecommendationEngine()
    
    # Step 6: Process each user
    print("Processing users and generating recommendations...")
    all_recommendations = []
    
    for user_id in users_df['user_id']:
        print(f"  Processing user {user_id}...")
        
        # Get user sleep data
        user_data = processed_data[processed_data['user_id'] == user_id].copy()
        
        # Calculate sleep scores
        sleep_scores = []
        for _, row in user_data.iterrows():
            # Simple score calculation if model not loaded
            if hasattr(sleep_quality_model, 'model') and sleep_quality_model.model is not None:
                additional_metrics = {
                    'deep_sleep_percentage': row.get('deep_sleep_percentage', 0.2),
                    'rem_sleep_percentage': row.get('rem_sleep_percentage', 0.25),
                    'sleep_onset_latency_minutes': row.get('sleep_onset_latency_minutes', 15),
                    'awakenings_count': row.get('awakenings_count', 2)
                }
                score = sleep_quality_model.calculate_sleep_score(
                    row['sleep_efficiency'], 
                    row.get('subjective_rating'),
                    additional_metrics
                )
            else:
                # Fallback if model not loaded
                score = int(row['sleep_efficiency'] * 80 + row.get('subjective_rating', 5) * 2)
                
            sleep_scores.append(score)
        
        user_data['sleep_score'] = sleep_scores
        
        # Analyze progress
        progress_data = recommendation_engine.analyze_progress(user_id, user_data)
        
        # Generate recommendation
        message = recommendation_engine.generate_recommendation(user_id, progress_data)
        
        # Store recommendation
        recommendation = {
            'user_id': user_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': message
        }
        all_recommendations.append(recommendation)
        
        # Generate visualization
        visualize_user_progress(user_id, sleep_data_df, pd.DataFrame(all_recommendations))
    
    # Save all recommendations
    recommendations_df = pd.DataFrame(all_recommendations)
    recommendations_df.to_csv('demo/recommendations/demo_recommendations.csv', index=False)
    
    print("\nDemo completed successfully!")
    print("Results saved to the 'demo' directory:")
    print("  - User data: demo/data/demo_users.csv")
    print("  - Sleep data: demo/data/demo_sleep_data.csv")
    print("  - Recommendations: demo/recommendations/demo_recommendations.csv")
    print("  - Visualizations: demo/visualizations/")

if __name__ == "__main__":
    run_demo()