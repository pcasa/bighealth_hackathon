import numpy as np
import pandas as pd
import yaml
import os
from datetime import datetime, timedelta, time

class SleepDataGenerator:
    def __init__(self, config_path='config/data_generation_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.sleep_patterns = self.config['sleep_patterns']
        self.time_settings = self.config['time_settings']
        
    def generate_sleep_data(self, users_df, start_date=None, end_date=None):
        """Generate sleep data for a set of users over a time period"""
        if start_date is None:
            start_date = datetime.strptime(self.time_settings['start_date'], '%Y-%m-%d')
        if end_date is None:
            end_date = datetime.strptime(self.time_settings['end_date'], '%Y-%m-%d')
        
        all_sleep_data = []
        
        for _, user in users_df.iterrows():
            user_sleep_data = self._generate_user_sleep_data(user, start_date, end_date)
            all_sleep_data.extend(user_sleep_data)
        
        return pd.DataFrame(all_sleep_data)
    
    def _generate_user_sleep_data(self, user, start_date, end_date):
        """Generate sleep data for a single user over a time period"""
        user_data = []
        pattern = user['sleep_pattern']
        pattern_params = self.sleep_patterns[pattern]
        
        # Set base bedtime and wake time based on pattern
        if pattern == 'shift_worker':
            base_bedtime = time(hour=8, minute=0)  # 8:00 AM for day sleepers
            base_waketime = time(hour=15, minute=0)  # 3:00 PM
            if np.random.random() > pattern_params['day_sleep_probability']:
                base_bedtime = time(hour=22, minute=0)  # 10:00 PM for night workers
                base_waketime = time(hour=5, minute=0)  # 5:00 AM
        else:
            base_bedtime = time(hour=22, minute=30)  # 10:30 PM
            base_waketime = time(hour=6, minute=30)  # 6:30 AM
        
        current_date = start_date
        while current_date <= end_date:
            # Check if the user skipped logging this day
            missing_prob = self.time_settings['missing_data_probability']
            if current_date.weekday() >= 5:  # Weekend
                missing_prob = self.time_settings['weekend_missing_probability']
                
            # Skip if user's consistency level says they miss this day
            if np.random.random() > user['data_consistency'] or np.random.random() < missing_prob:
                current_date += timedelta(days=1)
                continue
            
            # Generate sleep data for this day
            sleep_data = self._generate_daily_sleep(
                user, current_date, pattern, pattern_params, base_bedtime, base_waketime
            )
            user_data.append(sleep_data)
            
            current_date += timedelta(days=1)
        
        return user_data
    
    def _generate_daily_sleep(self, user, date, pattern, pattern_params, base_bedtime, base_waketime):
        """Generate sleep data for a single day"""
        
        # Add variability based on pattern and user consistency
        bedtime_variance_minutes = int(30 * (1 - user['sleep_consistency']))
        waketime_variance_minutes = int(30 * (1 - user['sleep_consistency']))
        
        if pattern == 'variable':
            var_hours = np.random.uniform(
                pattern_params['bedtime_variance_hours'][0], 
                pattern_params['bedtime_variance_hours'][1]
            )
            bedtime_variance_minutes = int(var_hours * 60)
            waketime_variance_minutes = int(var_hours * 60)
        
        # Apply variance to base times
        bedtime_delta = np.random.randint(-bedtime_variance_minutes, bedtime_variance_minutes)
        waketime_delta = np.random.randint(-waketime_variance_minutes, waketime_variance_minutes)
        
        bedtime_date = date - timedelta(days=1)  # Previous day for bedtime
        bed_hour, bed_minute = base_bedtime.hour, base_bedtime.minute
        bed_datetime = datetime.combine(bedtime_date.date(), time(hour=bed_hour, minute=bed_minute))
        bed_datetime += timedelta(minutes=bedtime_delta)
        
        wake_hour, wake_minute = base_waketime.hour, base_waketime.minute
        wake_datetime = datetime.combine(date.date(), time(hour=wake_hour, minute=wake_minute))
        wake_datetime += timedelta(minutes=waketime_delta)
        
        # Handle pattern-specific parameters
        if pattern == 'insomnia':
            sleep_onset_minutes = np.random.randint(
                pattern_params['sleep_onset_minutes'][0],
                pattern_params['sleep_onset_minutes'][1]
            )
        else:
            sleep_onset_minutes = np.random.randint(5, 20)
        
        # Calculate sleep onset time
        sleep_onset_datetime = bed_datetime + timedelta(minutes=sleep_onset_minutes)
        
        # Generate sleep duration based on pattern
        if 'sleep_duration_hours' in pattern_params:
            sleep_hours = np.random.uniform(
                pattern_params['sleep_duration_hours'][0],
                pattern_params['sleep_duration_hours'][1]
            )
        else:
            sleep_hours = np.random.uniform(6, 8)  # Default
        
        # Generate awakenings
        if 'awakenings_count' in pattern_params:
            awakenings = np.random.randint(
                pattern_params['awakenings_count'][0],
                pattern_params['awakenings_count'][1] + 1
            )
        else:
            awakenings = np.random.randint(0, 3)  # Default
        
        # Calculate total time awake during night
        if 'awakening_duration_minutes' in pattern_params:
            avg_awakening_minutes = np.random.randint(
                pattern_params['awakening_duration_minutes'][0],
                pattern_params['awakening_duration_minutes'][1]
            )
        else:
            avg_awakening_minutes = np.random.randint(3, 10)  # Default
            
        total_awake_minutes = awakenings * avg_awakening_minutes
        
        # Calculate sleep efficiency
        time_in_bed = (wake_datetime - bed_datetime).total_seconds() / 3600  # hours
        sleep_efficiency = sleep_hours / time_in_bed
        
        # Adjust for realistic limits
        if 'sleep_efficiency' in pattern_params:
            min_efficiency, max_efficiency = pattern_params['sleep_efficiency']
            sleep_efficiency = max(min_efficiency, min(sleep_efficiency, max_efficiency))
        
        # Generate subjective rating
        if 'subjective_rating_range' in pattern_params:
            rating_min, rating_max = pattern_params['subjective_rating_range']
        else:
            rating_min, rating_max = 3, 9  # Default
            
        subjective_rating = np.random.randint(rating_min, rating_max + 1)
        
        # Create sleep record
        return {
            'user_id': user['user_id'],
            'date': date.strftime('%Y-%m-%d'),
            'bedtime': bed_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'sleep_onset_time': sleep_onset_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'wake_time': wake_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'time_in_bed_hours': time_in_bed,
            'sleep_duration_hours': sleep_hours,
            'sleep_onset_latency_minutes': sleep_onset_minutes,
            'awakenings_count': awakenings,
            'total_awake_minutes': total_awake_minutes,
            'sleep_efficiency': sleep_efficiency,
            'subjective_rating': subjective_rating
        }