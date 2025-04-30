#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data generation script for Sleep Insights App.
This demonstrates how to use the BaseDataGenerator, SleepDataGenerator, and WearableDataGenerator
to create synthetic data for testing and development.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.data_generation.base_generator import BaseDataGenerator
from src.data_generation.sleep_data_generator import SleepDataGenerator
from src.data_generation.wearable_data_generator import WearableDataGenerator
from src.utils.constants import profession_categories

def generate_user_profiles(num_users=100, config_path='src/config/data_generation_config.yaml'):
    """Generate synthetic user profiles for data generation."""
    base_gen = BaseDataGenerator(config_path)
    user_profiles = []
    
    # Get configuration
    config = base_gen.config
    profile_config = config['user_profiles']
    time_settings = config['time_settings']

    # Parse the start and end dates from configuration
    start_date = datetime.strptime(time_settings['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(time_settings['end_date'], '%Y-%m-%d')
    
    # Calculate the date range in days
    date_range_days = (end_date - start_date).days
    if date_range_days <= 0:
        # Fallback if date range is invalid
        date_range_days = 365
    
    # Generate user profiles
    for i in range(num_users):
        user_id = f"user_{i:04d}"
        
        # Generate basic demographics
        age = np.random.randint(profile_config['age_range'][0], profile_config['age_range'][1])
        gender = np.random.choice(profile_config['genders'])
        
        # Assign sleep pattern based on configuration distribution
        sleep_pattern = np.random.choice(
            list(profile_config['sleep_patterns'].keys()),
            p=list(profile_config['sleep_patterns'].values())
        )
        
        # Assign device type based on configuration distribution
        device_type = np.random.choice(
            list(profile_config['device_distribution'].keys()),
            p=list(profile_config['device_distribution'].values())
        )
        
        # Assign region based on configuration distribution
        region_category = np.random.choice(
            list(profile_config['region_distribution'].keys()),
            p=list(profile_config['region_distribution'].values())
        )
        
        # Generate a more detailed region based on the category
        region = generate_region(region_category)
        
        # Assign profession based on configuration distribution
        profession_category = np.random.choice(
            list(profile_config['profession_categories'].keys()),
            p=list(profile_config['profession_categories'].values())
        )
        
        # Generate a specific profession based on the category
        profession = generate_profession(profession_category)
        
        # Generate consistency scores (how consistently they track sleep)
        data_consistency = np.random.uniform(0.5, 0.95)  # How often they log data
        sleep_consistency = np.random.uniform(0.3, 0.9)  # How consistent their sleep schedule is
        
        # Less consistency for certain patterns
        if sleep_pattern == 'variable':
            sleep_consistency *= 0.6
        elif sleep_pattern == 'shift_worker':
            sleep_consistency *= 0.7


        # Generate random created_at date between start_date and end_date
        random_days = np.random.randint(0, date_range_days + 1)
        created_at = (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d %H:%M:%S')
        
        
        user_profile = {
            'user_id': user_id,
            'age': age,
            'gender': gender,
            'region': region,
            'region_category': region_category,
            'profession': profession,
            'profession_category': profession_category,
            'sleep_pattern': sleep_pattern,
            'device_type': device_type,
            'data_consistency': data_consistency,
            'sleep_consistency': sleep_consistency,
            'created_at': created_at 
        }
        
        user_profiles.append(user_profile)
    
    return pd.DataFrame(user_profiles)

def generate_region(region_category):
    """Generate a specific region based on the category."""
    regions = {
        'north_america': [
            'New York, NY, United States',
            'Los Angeles, CA, United States',
            'Chicago, IL, United States',
            'Toronto, ON, Canada',
            'Mexico City, Mexico'
        ],
        'europe': [
            'London, England, United Kingdom',
            'Paris, France',
            'Berlin, Germany',
            'Madrid, Spain',
            'Rome, Italy'
        ],
        'asia': [
            'Tokyo, Japan',
            'Shanghai, China',
            'Mumbai, India',
            'Seoul, South Korea',
            'Bangkok, Thailand'
        ],
        'other': [
            'Sydney, NSW, Australia',
            'Rio de Janeiro, Brazil',
            'Cape Town, South Africa',
            'Dubai, UAE',
            'Auckland, New Zealand'
        ]
    }
    
    return np.random.choice(regions[region_category])

def generate_profession(profession_category):
    """Generate a specific profession based on the category."""
    professions = {
        'healthcare': [
            'Nurse', 'Doctor', 'Physical Therapist', 'Pharmacist', 'Medical Technician',
            'Dentist', 'Veterinarian', 'Radiologist', 'Hospital Administrator'
        ],
        'tech': [
            'Software Engineer', 'Data Scientist', 'Product Manager', 'UX Designer',
            'System Administrator', 'IT Support', 'Database Administrator', 'QA Tester'
        ],
        'service': [
            'Retail Associate', 'Server', 'Barista', 'Flight Attendant', 'Hotel Staff',
            'Customer Service Representative', 'Hairstylist', 'Chef', 'Cashier'
        ],
        'education': [
            'Teacher', 'Professor', 'School Administrator', 'Guidance Counselor',
            'Librarian', 'Research Assistant', 'Tutor', 'Education Consultant'
        ],
        'industrial': [
            'Factory Worker', 'Construction Worker', 'Electrician', 'Plumber', 
            'Machinist', 'Carpenter', 'Welder', 'Mechanic', 'Warehouse Staff'
        ],
        'office': [
            'Accountant', 'Marketing Specialist', 'HR Manager', 'Administrative Assistant',
            'Sales Representative', 'Financial Analyst', 'Office Manager', 'Executive'
        ]
    }
    
    return np.random.choice(professions[profession_category])

def main():
    """Generate and save synthetic data for the Sleep Insights App."""
    # Set paths
    config_path = 'src/config/data_generation_config.yaml'
    device_config_path = 'src/config/device_profiles.yaml'
    output_dir = 'data/raw'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set parameters
    num_users = 100
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)  # 3 months of data
    
    print(f"Generating data for {num_users} users from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Generate user profiles
    print("Generating user profiles...")
    users_df = generate_user_profiles(num_users, config_path)
    
    # Debug: Check if user_id and device_type are in users_df
    print(f"Users DataFrame columns: {users_df.columns.tolist()}")
    if 'user_id' not in users_df.columns:
        print("ERROR: 'user_id' column missing from users_df")
    if 'device_type' not in users_df.columns:
        print("ERROR: 'device_type' column missing from users_df")
    
    # Initialize generators
    sleep_data_gen = SleepDataGenerator(config_path)
    wearable_data_gen = WearableDataGenerator(config_path, device_config_path)
    
    # Generate sleep data
    print("Generating sleep data...")
    sleep_data_df = sleep_data_gen.generate_sleep_data(users_df, start_date, end_date)
    
    # Debug: Check if sleep_data_df has user_id
    print(f"Sleep DataFrame columns: {sleep_data_df.columns.tolist()}")
    if 'user_id' not in sleep_data_df.columns:
        print("ERROR: 'user_id' column missing from sleep_data_df")
        # Add user_id column if missing (emergency fix)
        if len(sleep_data_df) > 0:
            print("Attempting to fix sleep_data_df by adding user_id column...")
            # This is a fallback - the proper fix would be in SleepDataGenerator
            sleep_data_df['user_id'] = [f"user_{i:04d}" for i in range(len(sleep_data_df))]
    
    # Generate wearable data
    print("Generating wearable data...")
    wearable_data_df = wearable_data_gen.generate_wearable_data(sleep_data_df, users_df)
    
    # Save generated data
    print("Saving generated data...")
    users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False)
    sleep_data_df.to_csv(os.path.join(output_dir, 'sleep_data.csv'), index=False)
    
    # Remove the large sequence/data arrays before saving to CSV
    wearable_save_df = wearable_data_df.copy()
    sequence_cols = ['heart_rate_data', 'movement_data', 'sleep_stage_data']
    for col in sequence_cols:
        if col in wearable_save_df.columns:
            wearable_save_df[col] = wearable_save_df[col].apply(lambda x: str(x)[:100] + '...' if isinstance(x, (list, np.ndarray)) else x)
    
    wearable_save_df.to_csv(os.path.join(output_dir, 'wearable_data.csv'), index=False)
    
    # Generate summary statistics
    user_count = len(users_df)
    sleep_record_count = len(sleep_data_df)
    wearable_record_count = len(wearable_data_df)
    avg_records_per_user = sleep_record_count / user_count if user_count > 0 else 0
    
    print("\nData Generation Summary:")
    print(f"Total users: {user_count}")
    print(f"Total sleep records: {sleep_record_count}")
    print(f"Total wearable records: {wearable_record_count}")
    print(f"Average records per user: {avg_records_per_user:.1f}")
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    main()