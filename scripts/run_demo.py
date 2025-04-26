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

from src.data_processing.preprocessing import Preprocessor
from src.models.recommendation_engine import SleepRecommendationEngine

def create_demo_users():
    """Create sample users for the demo"""
    # Create users with different sleep patterns
    users = [
        {
            'user_id': 'user_normal',
            'age': 35,
            'gender': 'female',
            'sleep_pattern': 'normal',
            'data_consistency': 0.95,
            'sleep_consistency': 0.9
        },
        {
            'user_id': 'user_insomnia',
            'age': 42,
            'gender': 'male',
            'sleep_pattern': 'insomnia',
            'data_consistency': 0.85,
            'sleep_consistency': 0.6
        },
        {
            'user_id': 'user_shift_worker',
            'age': 29,
            'gender': 'non-binary',
            'sleep_pattern': 'shift_worker',
            'data_consistency': 0.8,
            'sleep_consistency': 0.5
        },
        {
            'user_id': 'user_improving',
            'age': 31,
            'gender': 'female',
            'sleep_pattern': 'insomnia',  # Starts with insomnia but will improve
            'data_consistency': 0.9,
            'sleep_consistency': 0.7
        },
        {
            'user_id': 'user_no_sleep',
            'age': 45,
            'gender': 'male',
            'sleep_pattern': 'severe_insomnia',
            'data_consistency': 0.95,
            'sleep_consistency': 0.3
        }
    ]
    
    return pd.DataFrame(users)

def generate_form_submissions(users_df, days=30):
    """Generate sleep form data for demo users"""
    all_sleep_data = []
    
    for _, user in users_df.iterrows():
        user_sleep_data = []
        
        # Get pattern parameters
        pattern = user['sleep_pattern']
        
        # Generate data for each day
        for day in range(days):
            current_date = datetime.now() - timedelta(days=days-day-1)
            
            # Skip some days based on consistency
            if np.random.random() > user['data_consistency']:
                continue
            
            # For user with severe insomnia, include some nights with no sleep
            if user['user_id'] == 'user_no_sleep' and np.random.random() < 0.3:
                # Record with no sleep
                sleep_record = {
                    'user_id': user['user_id'],
                    'date': current_date.strftime('%Y-%m-%d'),
                    'bedtime': (current_date.replace(hour=22, minute=30)).strftime('%Y-%m-%d %H:%M:%S'),
                    'out_bed_time': ((current_date + timedelta(days=1)).replace(hour=7, minute=0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'no_sleep': True,
                    'subjective_rating': 1
                }
                user_sleep_data.append(sleep_record)
                continue
            
            # For the 'improving' user, gradually improve sleep metrics
            if user['user_id'] == 'user_improving' and day > 10:
                improvement_factor = min(1.0, (day - 10) / 20)  # Gradually improve
                
                # Generate form data with improvement
                bedtime_hour = (22 + np.random.randint(-1, 2)) % 24
                bedtime_minute = np.random.randint(0, 60)
                bedtime = current_date.replace(hour=bedtime_hour, minute=bedtime_minute)
                
                # Time to fall asleep improves gradually
                sleep_onset_minutes = max(10, int(45 - (30 * improvement_factor)))
                sleep_time = bedtime + timedelta(minutes=sleep_onset_minutes)
                
                # Awakenings improve gradually
                awakenings_count = max(1, int(5 - (3 * improvement_factor)))
                time_awake_minutes = max(5, int(40 - (30 * improvement_factor)))
                
                # Total sleep duration improves
                sleep_hours = 5 + (2 * improvement_factor)
                wake_time = sleep_time + timedelta(hours=sleep_hours) - timedelta(minutes=time_awake_minutes)
                
                # Out of bed time
                out_bed_minute = np.random.randint(0, 30)
                out_bed_time = wake_time + timedelta(minutes=out_bed_minute)
                
                # Subjective rating improves
                rating = 4 + int(5 * improvement_factor)
                
            else:
                # Generate pattern-specific form data
                if pattern == 'normal':
                    # Typical good sleeper
                    bedtime_hour = (22 + np.random.randint(-1, 2)) % 24
                    bedtime_minute = np.random.randint(0, 60)
                    bedtime = current_date.replace(hour=bedtime_hour, minute=bedtime_minute)
                    
                    sleep_onset_minutes = np.random.randint(5, 20)
                    sleep_time = bedtime + timedelta(minutes=sleep_onset_minutes)
                    
                    awakenings_count = np.random.randint(0, 3)
                    time_awake_minutes = np.random.randint(0, 20) if awakenings_count > 0 else 0
                    
                    sleep_hours = np.random.uniform(6.5, 8.5)
                    wake_time = sleep_time + timedelta(hours=sleep_hours) - timedelta(minutes=time_awake_minutes)
                    
                    out_bed_minute = np.random.randint(0, 30)
                    out_bed_time = wake_time + timedelta(minutes=out_bed_minute)
                    
                    rating = np.random.randint(7, 11)  # 7-10 rating
                    
                elif pattern == 'insomnia':
                    # Insomnia pattern
                    bedtime_hour = (22 + np.random.randint(-1, 2)) % 24
                    bedtime_minute = np.random.randint(0, 60)
                    bedtime = current_date.replace(hour=bedtime_hour, minute=bedtime_minute)
                    
                    sleep_onset_minutes = np.random.randint(30, 90)  # Long time to fall asleep
                    sleep_time = bedtime + timedelta(minutes=sleep_onset_minutes)
                    
                    awakenings_count = np.random.randint(3, 7)  # Many awakenings
                    time_awake_minutes = np.random.randint(30, 90)  # Long time awake
                    
                    sleep_hours = np.random.uniform(4, 6)  # Short sleep
                    wake_time = sleep_time + timedelta(hours=sleep_hours) - timedelta(minutes=time_awake_minutes)
                    
                    out_bed_minute = np.random.randint(0, 30)
                    out_bed_time = wake_time + timedelta(minutes=out_bed_minute)
                    
                    rating = np.random.randint(2, 7)  # 2-6 rating
                    
                elif pattern == 'shift_worker':
                    # Shift worker (daytime sleeper)
                    if np.random.random() < 0.7:  # Day sleep
                        bedtime_hour = (8 + np.random.randint(-1, 2)) % 24
                        bedtime = current_date.replace(hour=bedtime_hour, minute=np.random.randint(0, 60))
                        
                        sleep_onset_minutes = np.random.randint(10, 40)
                        sleep_time = bedtime + timedelta(minutes=sleep_onset_minutes)
                        
                        awakenings_count = np.random.randint(2, 5)
                        time_awake_minutes = np.random.randint(15, 45)
                        
                        sleep_hours = np.random.uniform(5, 7)
                        wake_time = sleep_time + timedelta(hours=sleep_hours) - timedelta(minutes=time_awake_minutes)
                        
                        out_bed_time = wake_time + timedelta(minutes=np.random.randint(5, 25))
                    else:  # Night sleep
                        bedtime_hour = (22 + np.random.randint(-1, 2)) % 24
                        bedtime = current_date.replace(hour=bedtime_hour, minute=np.random.randint(0, 60))
                        
                        sleep_onset_minutes = np.random.randint(10, 30)
                        sleep_time = bedtime + timedelta(minutes=sleep_onset_minutes)
                        
                        awakenings_count = np.random.randint(1, 4)
                        time_awake_minutes = np.random.randint(10, 30)
                        
                        sleep_hours = np.random.uniform(5.5, 7.5)
                        wake_time = sleep_time + timedelta(hours=sleep_hours) - timedelta(minutes=time_awake_minutes)
                        
                        out_bed_time = wake_time + timedelta(minutes=np.random.randint(5, 25))
                    
                    rating = np.random.randint(4, 9)  # 4-8 rating
                    
                else:  # Default/other patterns
                    bedtime_hour = (22 + np.random.randint(-2, 3)) % 24  # More variable
                    bedtime_minute = np.random.randint(0, 60)
                    bedtime = current_date.replace(hour=bedtime_hour, minute=bedtime_minute)
                    
                    sleep_onset_minutes = np.random.randint(15, 45)
                    sleep_time = bedtime + timedelta(minutes=sleep_onset_minutes)
                    
                    awakenings_count = np.random.randint(1, 5)
                    time_awake_minutes = np.random.randint(10, 60)
                    
                    sleep_hours = np.random.uniform(5, 8)
                    wake_time = sleep_time + timedelta(hours=sleep_hours) - timedelta(minutes=time_awake_minutes)
                    
                    out_bed_minute = np.random.randint(0, 45)  # More variable
                    out_bed_time = wake_time + timedelta(minutes=out_bed_minute)
                    
                    rating = np.random.randint(3, 9)  # 3-8 rating
            
            # Create sleep record from form data
            sleep_record = {
                'user_id': user['user_id'],
                'date': current_date.strftime('%Y-%m-%d'),
                'bedtime': bedtime.strftime('%Y-%m-%d %H:%M:%S'),
                'sleep_onset_time': sleep_time.strftime('%Y-%m-%d %H:%M:%S'),
                'wake_time': wake_time.strftime('%Y-%m-%d %H:%M:%S'),
                'out_bed_time': out_bed_time.strftime('%Y-%m-%d %H:%M:%S'),
                'awakenings_count': awakenings_count,
                'time_awake_minutes': time_awake_minutes,
                'subjective_rating': rating,
                'no_sleep': False
            }
            
            # Calculate derived metrics
            time_in_bed_hours = (datetime.strptime(sleep_record['out_bed_time'], '%Y-%m-%d %H:%M:%S') - 
                                 datetime.strptime(sleep_record['bedtime'], '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600
            
            sleep_duration_hours = ((datetime.strptime(sleep_record['wake_time'], '%Y-%m-%d %H:%M:%S') - 
                       datetime.strptime(sleep_record['sleep_onset_time'], '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600) - (time_awake_minutes / 60)
            
            sleep_efficiency = max(0, min(1, sleep_duration_hours / time_in_bed_hours if time_in_bed_hours > 0 else 0))
            
            # Add calculated fields
            sleep_record['time_in_bed_hours'] = time_in_bed_hours
            sleep_record['sleep_duration_hours'] = sleep_duration_hours
            sleep_record['sleep_efficiency'] = sleep_efficiency
            
            user_sleep_data.append(sleep_record)
        
        all_sleep_data.extend(user_sleep_data)
    
    return pd.DataFrame(all_sleep_data)

def visualize_user_progress(user_id, sleep_data, recommendations):
    """Create visualizations of user progress and recommendations"""
    # Filter data for this user
    user_data = sleep_data[sleep_data['user_id'] == user_id].copy()
    
    # Convert date to datetime if needed
    if 'date' in user_data.columns and not pd.api.types.is_datetime64_dtype(user_data['date']):
        user_data['date'] = pd.to_datetime(user_data['date'])
    
    user_data = user_data.sort_values('date')
    
    # Handle potential no-sleep days
    if 'no_sleep' in user_data.columns:
        # Mask for days with sleep data
        sleep_days = ~user_data['no_sleep']
        has_no_sleep_days = not sleep_days.all()
    else:
        sleep_days = pd.Series([True] * len(user_data))
        has_no_sleep_days = False
    
    # Get user recommendations
    user_recs = recommendations[recommendations['user_id'] == user_id].copy() if recommendations is not None else pd.DataFrame()
    if not user_recs.empty:
        user_recs['timestamp'] = pd.to_datetime(user_recs['timestamp'])
        user_recs = user_recs.sort_values('timestamp')
    
    # Create figure with the appropriate number of subplots
    if has_no_sleep_days:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot sleep efficiency for days with sleep
    if 'sleep_efficiency' in user_data.columns:
        sleep_efficiency = user_data.loc[sleep_days, 'sleep_efficiency'] * 100 if has_no_sleep_days else user_data['sleep_efficiency'] * 100
        dates = user_data.loc[sleep_days, 'date'] if has_no_sleep_days else user_data['date']
        
        ax1.plot(dates, sleep_efficiency, 'o-', label='Sleep Efficiency (%)')
        ax1.set_ylabel('Sleep Efficiency (%)')
        ax1.set_title(f'Sleep Progress for {user_id}')
        ax1.grid(True)
        ax1.legend()
    
    # Plot subjective ratings
    if 'subjective_rating' in user_data.columns:
        ax2.plot(user_data['date'], user_data['subjective_rating'], 'o-', color='green', label='Subjective Rating (1-10)')
        ax2.set_ylabel('Subjective Rating (1-10)')
        ax2.grid(True)
        ax2.legend()
    
    # Plot no-sleep indicator if applicable
    if has_no_sleep_days:
        # Create a binary indicator for no-sleep nights
        no_sleep_indicator = user_data['no_sleep'].astype(int)
        ax3.bar(user_data['date'], no_sleep_indicator, color='red', label='No Sleep Night')
        ax3.set_ylabel('No Sleep')
        ax3.set_ylim(0, 1.5)  # Set y-limit to make bars visible
        ax3.set_xlabel('Date')
        ax3.legend()
    else:
        ax2.set_xlabel('Date')
    
    # Add recommendation markers
    if not user_recs.empty:
        for _, rec in user_recs.iterrows():
            timestamp = rec['timestamp']
            # Find nearest date in sleep data
            nearest_idx = (user_data['date'] - timestamp).abs().idxmin()
            nearest_date = user_data.loc[nearest_idx, 'date']
            
            # Add marker to plots
            if 'sleep_efficiency' in user_data.columns and nearest_idx in user_data.index:
                if user_data.loc[nearest_idx, 'no_sleep'] if 'no_sleep' in user_data.columns else False:
                    # Don't mark efficiency for no-sleep nights
                    pass
                else:
                    efficiency = user_data.loc[nearest_idx, 'sleep_efficiency'] * 100
                    ax1.annotate('R', (nearest_date, efficiency), 
                                xytext=(0, 15), textcoords='offset points',
                                arrowprops=dict(arrowstyle='->', color='red'),
                                color='red', fontsize=12, fontweight='bold')
            
            if 'subjective_rating' in user_data.columns and nearest_idx in user_data.index:
                rating = user_data.loc[nearest_idx, 'subjective_rating']
                ax2.annotate('R', (nearest_date, rating), 
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
    
    # Step 2: Generate form data
    print("Generating sleep form data...")
    form_data_df = generate_form_submissions(users_df)
    form_data_df.to_csv('demo/data/demo_form_data.csv', index=False)
    
    # Step 3: Preprocess data
    print("Preprocessing sleep data...")
    preprocessor = Preprocessor()
    processed_data = preprocessor.preprocess_sleep_data(form_data_df)
    
    # Step 4: Initialize recommendation engine
    print("Initializing recommendation engine...")
    recommendation_engine = SleepRecommendationEngine()
    
    # Step 5: Process each user
    print("Processing users and generating recommendations...")
    all_recommendations = []
    
    for user_id in users_df['user_id']:
        print(f"  Processing user {user_id}...")
        
        # Get user sleep data
        user_data = processed_data[processed_data['user_id'] == user_id].copy()
        
        try:
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
            
            print(f"  Recommendation: {message[:50]}...")
            
        except Exception as e:
            print(f"  Error processing user {user_id}: {str(e)}")
    
    # Save all recommendations
    recommendations_df = pd.DataFrame(all_recommendations)
    recommendations_df.to_csv('demo/recommendations/demo_recommendations.csv', index=False)
    
    # Generate visualizations
    print("Generating visualizations...")
    for user_id in users_df['user_id']:
        try:
            visualize_user_progress(user_id, form_data_df, recommendations_df)
            print(f"  Created visualization for {user_id}")
        except Exception as e:
            print(f"  Error creating visualization for {user_id}: {str(e)}")
    
    print("\nDemo completed successfully!")
    print("Results saved to the 'demo' directory:")
    print("  - User data: demo/data/demo_users.csv")
    print("  - Form data: demo/data/demo_form_data.csv")
    print("  - Recommendations: demo/recommendations/demo_recommendations.csv")
    print("  - Visualizations: demo/visualizations/")

if __name__ == "__main__":
    run_demo()