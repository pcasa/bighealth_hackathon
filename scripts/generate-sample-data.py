#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to generate a sample sleep dataset with enhanced user profiles.
This script creates a small sample of users with professions and regions.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our updated generators
from src.data_generation.user_generator import UserGenerator
from src.data_generation.sleep_data_generator import SleepDataGenerator
from src.data_generation.wearable_data_generator import WearableDataGenerator
from src.data_generation.external_factors_generator import ExternalFactorsGenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate sample sleep data with enhanced profiles')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='src/config/data_generation_config.yaml',
        help='Path to data generation configuration file'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/sample',
        help='Directory to save generated data'
    )
    
    parser.add_argument(
        '--num-users', 
        type=int, 
        default=50,
        help='Number of users to generate'
    )
    
    parser.add_argument(
        '--days', 
        type=int, 
        default=30,
        help='Number of days of data'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()

def main():
    """Generate sample sleep data with enhanced user profiles."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_users} user profiles...")
    
    # Initialize generators
    try:
        user_generator = UserGenerator(config_path=args.config)
        sleep_data_generator = SleepDataGenerator(config_path=args.config)
        wearable_data_generator = WearableDataGenerator(config_path=args.config)
        external_factors_generator = ExternalFactorsGenerator()
    except Exception as e:
        print(f"Error initializing generators: {str(e)}")
        return
    
    # Generate user profiles
    users_df = user_generator.generate_users(count=args.num_users)
    
    # Save user profiles
    users_file = os.path.join(args.output_dir, 'users.csv')
    users_df.to_csv(users_file, index=False)
    print(f"✓ Generated {len(users_df)} user profiles saved to {users_file}")
    
    # Create date range
    tmp_start_date = datetime.date(year=2025, month=1, day=1)
    end_date = fake.date_between(start_date=tmp_start_date, end_date='-1y')
    start_date = end_date - timedelta(days=args.days)
    
    # Generate sleep data
    print("Generating sleep data...")
    sleep_data_df = sleep_data_generator.generate_sleep_data(
        users_df, 
        start_date=start_date,
        end_date=end_date
    )
    
    # Save sleep data
    sleep_data_file = os.path.join(args.output_dir, 'sleep_data.csv')
    sleep_data_df.to_csv(sleep_data_file, index=False)
    print(f"✓ Generated {len(sleep_data_df)} sleep records saved to {sleep_data_file}")
    
    # Generate wearable data
    print("Generating wearable device data...")
    wearable_data_df = wearable_data_generator.generate_wearable_data(sleep_data_df, users_df)
    
    # Save wearable data
    wearable_data_file = os.path.join(args.output_dir, 'wearable_data.csv')
    wearable_data_df.to_csv(wearable_data_file, index=False)
    print(f"✓ Generated {len(wearable_data_df)} wearable records saved to {wearable_data_file}")
    
    # Generate external factors data
    print("Generating external factors data...")
    weather_data = external_factors_generator.generate_weather_data(
        start_date, 
        end_date
    )
    
    activity_data = external_factors_generator.generate_activity_data(
        sleep_data_df, 
        users_df
    )
    
    # Combine external data
    external_factors_df = pd.merge(
        weather_data,
        activity_data,
        on='date',
        how='outer'
    )
    
    # Save external factors data
    external_factors_file = os.path.join(args.output_dir, 'external_factors.csv')
    external_factors_df.to_csv(external_factors_file, index=False)
    print(f"✓ Generated external factors data saved to {external_factors_file}")
    
    # Generate sample recommendations based on data
    print("Generating sample recommendations...")
    sample_recommendations = generate_sample_recommendations(users_df, sleep_data_df)
    
    # Save recommendations
    recommendations_file = os.path.join(args.output_dir, 'recommendations.csv')
    sample_recommendations.to_csv(recommendations_file, index=False)
    print(f"✓ Generated {len(sample_recommendations)} recommendations saved to {recommendations_file}")
    
    print("\nSample data generation complete!")
    print(f"Summary:")
    print(f"- {len(users_df)} users with professions and regions")
    print(f"- {len(sleep_data_df)} sleep records")
    print(f"- {len(wearable_data_df)} wearable device records")
    print(f"- {len(external_factors_df)} external factor records")
    print(f"- {len(sample_recommendations)} sample recommendations")
    
def generate_sample_recommendations(users_df, sleep_data_df):
    """Generate sample recommendations based on user sleep data"""
    recommendations = []
    
    # Recommendation templates by sleep pattern
    templates = {
        'normal': [
            "Your sleep has been consistently good. Continue maintaining your regular sleep schedule.",
            "Your sleep efficiency is excellent. Consider adding a brief relaxation routine before bed to maintain this pattern.",
            "You're getting good quality sleep. Remember that consistency is key to maintaining healthy sleep patterns."
        ],
        'insomnia': [
            "Try implementing a 15-minute meditation before bedtime to reduce sleep onset time.",
            "Consider limiting screen time in the hour before bed to improve your sleep quality.",
            "Your sleep data shows frequent awakenings. A cooler bedroom temperature (65-68°F) might help you stay asleep longer.",
            "Try to establish a consistent sleep schedule, even on weekends, to help regulate your body's internal clock."
        ],
        'shift_worker': [
            "As a shift worker, consider using blackout curtains to improve your daytime sleep quality.",
            "Try using a white noise machine to mask daytime sounds during your sleep periods.",
            "Given your profession, it's important to prioritize sleep hygiene. Consider using eye masks and earplugs for uninterrupted sleep.",
            "Your changing schedule may benefit from exposure to bright light when you need to be alert and darkness when you need to sleep."
        ],
        'oversleeper': [
            "While you're getting plenty of sleep, the quality could be improved. Consider reducing your time in bed to 8-9 hours.",
            "Your sleep data suggests you may be spending too much time in bed. Try setting a consistent wake-up time, even on weekends.",
            "Consider adding more physical activity to your day, which might help improve your sleep quality rather than quantity."
        ],
        'variable': [
            "Your inconsistent sleep schedule may be affecting your overall sleep quality. Try to establish a more regular sleep routine.",
            "Your sleep data shows high variability. Setting a consistent bedtime and wake time could improve your sleep quality.",
            "Consider creating a bedtime routine that signals to your body it's time to sleep, which may help with your irregular sleep patterns."
        ]
    }
    
    # Additional templates based on profession
    profession_templates = {
        'healthcare': [
            "As a healthcare professional, your shift work can impact sleep. Try to maintain the same sleep routine even on days off.",
            "Healthcare workers often struggle with stress-related sleep issues. Consider a mindfulness practice to help unwind after work."
        ],
        'tech': [
            "Working with screens can affect your sleep quality. Consider using blue light filters in the evening hours.",
            "Tech professionals often have irregular work hours. Try to establish clear boundaries between work and sleep time."
        ]
    }
    
    # Additional templates based on region
    region_templates = {
        'europe': [
            "In your region, longer summer daylight hours can affect sleep. Consider using room-darkening curtains during summer months.",
            "Cultural norms in your region may include later dinners. Try to eat at least 3 hours before bedtime for better sleep quality."
        ],
        'asia': [
            "Your region's sleep patterns tend toward earlier bedtimes. Maintaining this cultural pattern can support your circadian rhythm.",
            "Urban light pollution in many Asian cities can affect sleep quality. Consider using an eye mask if external light is an issue."
        ]
    }
    
    # Generate 2-4 recommendations per user
    for _, user in users_df.iterrows():
        user_id = user['user_id']
        pattern = user['sleep_pattern']
        profession = user.get('profession', '')
        region = user.get('region', '')
        
        # Extract profession category
        profession_category = None
        if 'Nurse' in profession or 'Doctor' in profession or 'Healthcare' in profession:
            profession_category = 'healthcare'
        elif 'Software' in profession or 'Developer' in profession or 'Engineer' in profession:
            profession_category = 'tech'
        
        # Extract region category
        region_category = None
        if region and ',' in region:
            country = region.split(',')[-1].strip()
            if country in ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain']:
                region_category = 'europe'
            elif country in ['China', 'Japan', 'India', 'Korea', 'Thailand']:
                region_category = 'asia'
        
        # Get user sleep data
        user_sleep_data = sleep_data_df[sleep_data_df['user_id'] == user_id]
        
        if len(user_sleep_data) == 0:
            continue
        
        # Determine how many recommendations to generate
        num_recommendations = np.random.randint(2, 5)
        
        # Get pattern-specific templates
        pattern_specific = templates.get(pattern, templates['normal'])
        
        # Get profession-specific templates if available
        prof_specific = []
        if profession_category and profession_category in profession_templates:
            prof_specific = profession_templates[profession_category]
        
        # Get region-specific templates if available
        region_specific = []
        if region_category and region_category in region_templates:
            region_specific = region_templates[region_category]
        
        # Combine all relevant templates
        all_templates = pattern_specific + prof_specific + region_specific
        
        # Ensure we have enough unique templates
        if len(all_templates) < num_recommendations:
            all_templates.extend(templates['normal'])
        
        # Select random templates without replacement
        selected_templates = np.random.choice(all_templates, size=min(num_recommendations, len(all_templates)), replace=False)
        
        # Generate a date for each recommendation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        for i, template in enumerate(selected_templates):
            # Generate a random date within the last 30 days
            days_ago = np.random.randint(0, 30)
            rec_date = end_date - timedelta(days=days_ago)
            
            recommendations.append({
                'user_id': user_id,
                'timestamp': rec_date.strftime('%Y-%m-%d %H:%M:%S'),
                'message': template
            })
    
    return pd.DataFrame(recommendations)

if __name__ == "__main__":
    main()