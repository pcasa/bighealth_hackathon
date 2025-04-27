import numpy as np
import pandas as pd
import yaml
import os
import json
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
        
        # Extract profession categories for special handling
        # (In a real implementation, we would parse these from profession strings)
        profession_categories = {
            'healthcare': ['Nurse', 'Doctor', 'Paramedic', 'Healthcare'],
            'service': ['Server', 'Bartender', 'Retail', 'Hospitality'],
            'tech': ['Software', 'Engineer', 'Developer', 'IT'],
            'transit': ['Driver', 'Pilot', 'Operator', 'Transit']
        }
        
        for _, user in users_df.iterrows():
            # Determine profession category for this user
            user_profession_category = None
            for category, keywords in profession_categories.items():
                if any(keyword.lower() in user['profession'].lower() for keyword in keywords):
                    user_profession_category = category
                    break
            
            # Extract region components (simplistic approach for demo)
            region_parts = user['region'].split(', ')
            country = region_parts[-1] if len(region_parts) >= 3 else None
            
            # Determine region category
            user_region_category = None
            if country:
                if country in ['United States', 'Canada', 'Mexico']:
                    user_region_category = 'north_america'
                elif country in ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain']:
                    user_region_category = 'europe'
                elif country in ['China', 'Japan', 'India', 'Korea', 'Thailand']:
                    user_region_category = 'asia'
                else:
                    user_region_category = 'other'
            
            # Generate sleep data with these additional factors
            user_sleep_data = self._generate_user_sleep_data(
                user, 
                start_date, 
                end_date, 
                profession_category=user_profession_category,
                region_category=user_region_category
            )
            
            all_sleep_data.extend(user_sleep_data)
        
        return pd.DataFrame(all_sleep_data)
    
    def _generate_user_sleep_data(self, user, start_date, end_date, profession_category=None, region_category=None):
        """Generate sleep data for a single user over a time period"""
        user_data = []
        pattern = user['sleep_pattern']
        pattern_params = self.sleep_patterns[pattern].copy()  # Make a copy to avoid modifying the original
        
        # Apply profession-specific modifiers if available
        if profession_category and 'profession_modifiers' in pattern_params:
            if profession_category in pattern_params['profession_modifiers']:
                modifiers = pattern_params['profession_modifiers'][profession_category]
                for key, value in modifiers.items():
                    if isinstance(value, list):
                        # For list parameters like ranges
                        pattern_params[key] = value
                    else:
                        # For numeric adjustments
                        if key in pattern_params and isinstance(pattern_params[key], list) and len(pattern_params[key]) == 2:
                            # Adjust range values
                            pattern_params[key] = [val + value for val in pattern_params[key]]
                        elif key in pattern_params:
                            # Direct replacement
                            pattern_params[key] = value
        
        # Apply region-specific modifiers if available
        if region_category and 'region_modifiers' in pattern_params:
            if region_category in pattern_params['region_modifiers']:
                modifiers = pattern_params['region_modifiers'][region_category]
                for key, value in modifiers.items():
                    pattern_params[key] = value
            # Set base bedtime and wake time based on pattern, profession, and region
            if pattern == 'shift_worker':
                # Adjust shift work schedule based on profession
                if profession_category == 'healthcare':
                    # More likely to be night shifts
                    day_sleep_prob = 0.8
                elif profession_category == 'service':
                    # More variable shifts
                    day_sleep_prob = 0.5
                else:
                    # Use default from pattern params
                    day_sleep_prob = pattern_params.get('day_sleep_probability', 0.7)
                    
                if np.random.random() > day_sleep_prob:
                    # Night worker - sleeps during day
                    base_bedtime = time(hour=8, minute=0)  # 8:00 AM
                    base_waketime = time(hour=15, minute=0)  # 3:00 PM
                else:
                    # Day worker - sleeps at night
                    base_bedtime = time(hour=22, minute=0)  # 10:00 PM
                    base_waketime = time(hour=5, minute=0)  # 5:00 AM
            else:
                # Adjust based on region
                if region_category == 'europe':
                    # European countries tend to have later bedtimes
                    base_bedtime = time(hour=23, minute=30)  # 11:30 PM
                    base_waketime = time(hour=7, minute=30)  # 7:30 AM
                elif region_category == 'asia':
                    # Some Asian countries have earlier bedtimes
                    base_bedtime = time(hour=22, minute=0)  # 10:00 PM
                    base_waketime = time(hour=6, minute=0)  # 6:00 AM
                else:
                    # Default North American times
                    base_bedtime = time(hour=22, minute=30)  # 10:30 PM
                    base_waketime = time(hour=6, minute=30)  # 6:30 AM
            
            current_date = start_date
            while current_date <= end_date:
                # Check if the user skipped logging this day
                missing_prob = self.time_settings['missing_data_probability']
                
                # Adjust missing data probability based on profession
                if profession_category and 'profession_missing_data' in self.time_settings:
                    if profession_category in self.time_settings['profession_missing_data']:
                        missing_prob = self.time_settings['profession_missing_data'][profession_category]
                
                # Weekend adjustment
                if current_date.weekday() >= 5:  # Weekend
                    missing_prob = self.time_settings['weekend_missing_probability']
                    
                # Skip if user's consistency level says they miss this day
                if np.random.random() > user['data_consistency'] or np.random.random() < missing_prob:
                    current_date += timedelta(days=1)
                    continue
                
                # Generate sleep data for this day with profession and region context
                sleep_data = self._generate_daily_sleep(
                    user, current_date, pattern, pattern_params, 
                    base_bedtime, base_waketime,
                    profession_category, region_category
                )
                user_data.append(sleep_data)
                
                current_date += timedelta(days=1)
            
            return user_data
    
    def _generate_daily_sleep(self, user, date, pattern, pattern_params, base_bedtime, base_waketime, 
                             profession_category=None, region_category=None):
        """Generate sleep data for a single day"""
        
        # Add variability based on pattern, user consistency, and weekday/weekend
        is_weekend = date.weekday() >= 5
        
        # Weekend adjustment - people often sleep later on weekends
        weekend_bedtime_adjustment = 0
        weekend_waketime_adjustment = 0
        
        if is_weekend:
            weekend_bedtime_adjustment = np.random.randint(0, 90)  # Up to 1.5 hours later
            weekend_waketime_adjustment = np.random.randint(30, 120)  # 0.5 to 2 hours later
        
        # Base variability from user consistency
        bedtime_variance_minutes = max(1, int(30 * (1 - user['sleep_consistency'])))
        waketime_variance_minutes = max(1, int(30 * (1 - user['sleep_consistency'])))
        
        # Special handling for variable sleepers
        if pattern == 'variable':
            var_hours = np.random.uniform(
                pattern_params['bedtime_variance_hours'][0], 
                pattern_params['bedtime_variance_hours'][1]
            )
            bedtime_variance_minutes = max(1, int(var_hours * 60))
            waketime_variance_minutes = max(1, int(var_hours * 60))
        
        # Apply variance to base times
        bedtime_delta = np.random.randint(-bedtime_variance_minutes, bedtime_variance_minutes + 1)
        waketime_delta = np.random.randint(-waketime_variance_minutes, waketime_variance_minutes + 1)
        
        # Apply weekend adjustments
        bedtime_delta += weekend_bedtime_adjustment
        waketime_delta += weekend_waketime_adjustment
        
        bedtime_date = date - timedelta(days=1)  # Previous day for bedtime
        bed_hour, bed_minute = base_bedtime.hour, base_bedtime.minute
        bed_datetime = datetime.combine(bedtime_date.date(), time(hour=bed_hour, minute=bed_minute))
        bed_datetime += timedelta(minutes=bedtime_delta)
        
        wake_hour, wake_minute = base_waketime.hour, base_waketime.minute
        wake_datetime = datetime.combine(date.date(), time(hour=wake_hour, minute=wake_minute))
        wake_datetime += timedelta(minutes=waketime_delta)
        
        # Handle pattern-specific parameters for sleep onset
        if pattern == 'insomnia':
            # Insomnia pattern has longer sleep onset
            sleep_onset_minutes = np.random.randint(
                pattern_params['sleep_onset_minutes'][0],
                pattern_params['sleep_onset_minutes'][1]
            )
            
            # Age factor - older people with insomnia often have longer sleep onset
            if user['age'] > 60:
                sleep_onset_minutes += np.random.randint(0, 15)
        else:
            sleep_onset_minutes = np.random.randint(5, 20)
            
            # Add profession impact - high stress jobs may have longer sleep onset
            if profession_category in ['healthcare', 'tech'] and np.random.random() < 0.3:
                sleep_onset_minutes += np.random.randint(5, 15)
        
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
        
        # Weekend adjustment for sleep duration
        if is_weekend and pattern != 'shift_worker':
            sleep_hours += np.random.uniform(0, 1)  # Up to 1 hour extra sleep on weekends
        
        # Generate awakenings
        if 'awakenings_count' in pattern_params:
            awakenings = np.random.randint(
                pattern_params['awakenings_count'][0],
                pattern_params['awakenings_count'][1] + 1
            )
        else:
            awakenings = np.random.randint(0, 3)  # Default
        
        # Age impacts awakenings - older people tend to wake more
        if user['age'] > 60:
            awakenings += np.random.randint(0, 2)
        
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
        
        # Create complete sleep record
        return {
            'user_id': user['user_id'],
            'date': date.strftime('%Y-%m-%d'),
            'bedtime': bed_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'sleep_onset_time': sleep_onset_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'wake_time': wake_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'out_bed_time': (wake_datetime + timedelta(minutes=np.random.randint(5, 30))).strftime('%Y-%m-%d %H:%M:%S'),
            'time_in_bed_hours': time_in_bed,
            'sleep_duration_hours': sleep_hours,
            'sleep_onset_latency_minutes': sleep_onset_minutes,
            'awakenings_count': awakenings,
            'total_awake_minutes': total_awake_minutes,
            'sleep_efficiency': sleep_efficiency,
            'subjective_rating': subjective_rating,
            'is_weekend': is_weekend,
            'profession_category': profession_category,
            'region_category': region_category
        }