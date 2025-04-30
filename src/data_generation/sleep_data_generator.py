import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time

from src.data_generation.base_generator import BaseDataGenerator
from src.core.models.data_models import SleepEntry, UserProfile

class SleepDataGenerator(BaseDataGenerator):
    """
    Sleep data generator that inherits from the BaseDataGenerator.
    Generates synthetic sleep data for users based on their profile and sleep pattern.
    """
    def __init__(self, config_path='src/config/data_generation_config.yaml'):
        # Initialize the base generator
        super().__init__(config_path)
        
        # Store specific settings for sleep data generation
        self.sleep_patterns = self.config['sleep_patterns']
        self.time_settings = self.config['time_settings']
        
    def generate_sleep_data(self, users_df, start_date=None, end_date=None):
        """Generate sleep data for a set of users over a time period"""
        if start_date is None:
            start_date = datetime.strptime(self.time_settings['start_date'], '%Y-%m-%d')
        if end_date is None:
            end_date = datetime.strptime(self.time_settings['end_date'], '%Y-%m-%d')
        
        print(f"Generating sleep data from {start_date} to {end_date}")
        
        # Verify users_df has the required columns
        required_cols = ['user_id', 'sleep_pattern', 'profession', 'region']
        missing_cols = [col for col in required_cols if col not in users_df.columns]
        if missing_cols:
            print(f"WARNING: users_df is missing required columns: {missing_cols}")
            print(f"Available columns: {users_df.columns.tolist()}")
        
        all_sleep_data = []
        
        for _, user_row in users_df.iterrows():
            # Ensure user_row has 'user_id'
            if 'user_id' not in user_row:
                print(f"WARNING: user_row is missing 'user_id', skipping")
                continue
                
            # Print user_id for debugging
            # print(f"Generating sleep data for user: {user_row['user_id']}")
            
            # Convert row to dict
            user_dict = user_row.to_dict()
            
            # Extract profession category
            user_profession_category = self.get_category_from_keywords(
                user_dict.get('profession', ''), 
                {'healthcare': ['doctor', 'nurse', 'medical', 'healthcare', 'hospital'],
                'tech': ['engineer', 'developer', 'programmer', 'IT', 'tech'],
                'service': ['retail', 'server', 'customer', 'service', 'hospitality'],
                'education': ['teacher', 'professor', 'educator', 'tutor', 'school'],
                'industrial': ['factory', 'plant', 'construction', 'manufacturing', 'worker'],
                'office': ['clerk', 'manager', 'administrative', 'office', 'executive']}
            )
            
            # Extract region category
            user_region_category = self.extract_region_category(user_dict.get('region', ''))
            
            # Generate sleep data with these additional factors
            try:
                user_sleep_data = self._generate_user_sleep_data(
                    user_dict, 
                    start_date, 
                    end_date, 
                    profession_category=user_profession_category,
                    region_category=user_region_category
                )
                
                all_sleep_data.extend(user_sleep_data)
            except Exception as e:
                print(f"Error processing user {user_dict.get('user_id', 'unknown')}: {e}")
                continue
        
        # Check if we generated any data
        if not all_sleep_data:
            print("WARNING: No sleep data was generated!")
            return pd.DataFrame()
        
        # Convert list of dictionaries to DataFrame
        df_data = []
        for entry in all_sleep_data:
            if isinstance(entry, dict):
                df_data.append(entry)
            elif hasattr(entry, 'dict'):
                df_data.append(entry.dict())
            else:
                print(f"WARNING: Unexpected entry type: {type(entry)}")
                
        result_df = pd.DataFrame(df_data)
        
        # Ensure user_id is in the result
        if 'user_id' not in result_df.columns:
            print("ERROR: 'user_id' column missing from generated sleep data")
            if len(result_df) > 0:
                print("Attempting to fix by adding user_ids...")
                # This is a last resort - add user_ids sequentially
                unique_users = users_df['user_id'].unique()
                if len(unique_users) > 0:
                    # Distribute user_ids proportionally
                    result_df['user_id'] = [unique_users[i % len(unique_users)] for i in range(len(result_df))]
        else:
            print(f"Successfully generated sleep data with user_id column. Sample values: {result_df['user_id'].head(5).tolist()}")
        
        return result_df
    
    def _generate_user_sleep_data(self, user, start_date, end_date, profession_category=None, region_category=None):
        """Generate sleep data for a single user over a time period"""
        user_data = []
        
        # Extract user details
        user_id = user.get('user_id')
        if not user_id:
            raise ValueError("User is missing user_id")
            
        pattern = user.get('sleep_pattern')
        if not pattern or pattern not in self.sleep_patterns:
            print(f"WARNING: Invalid sleep pattern '{pattern}' for user {user_id}, using 'normal'")
            pattern = 'normal'
            
        pattern_params = self.sleep_patterns[pattern].copy()  # Make a copy to avoid modifying the original
        
        # Apply profession-specific modifiers
        if profession_category and 'profession_modifiers' in pattern_params:
            if profession_category in pattern_params['profession_modifiers']:
                modifiers = pattern_params['profession_modifiers'][profession_category]
                pattern_params = self.apply_modifiers(pattern_params, modifiers)
        
        # Apply region-specific modifiers
        if region_category and 'region_modifiers' in pattern_params:
            if region_category in pattern_params['region_modifiers']:
                modifiers = pattern_params['region_modifiers'][region_category]
                pattern_params = self.apply_modifiers(pattern_params, modifiers)
        
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
        
        # Generate sleep data for each day
        current_date = start_date
        while current_date <= end_date:
            # Check if user skipped logging this day based on consistency
            data_consistency = user.get('data_consistency', 0.8)  # Default if missing
            if data_consistency < 1.0:
                # Higher probability to skip weekends
                is_weekend = current_date.weekday() >= 5
                skip_probability = (1 - data_consistency)
                
                # Adjust skip probability for weekends if configured
                if is_weekend and 'weekend_missing_probability' in self.time_settings:
                    skip_probability = max(skip_probability, self.time_settings['weekend_missing_probability'])
                    
                # Check if this day should be skipped
                if np.random.random() < skip_probability:
                    current_date += timedelta(days=1)
                    continue
            
            # Generate sleep data for this day
            try:
                sleep_data = self._generate_daily_sleep(
                    user, current_date, pattern, pattern_params, 
                    base_bedtime, base_waketime,
                    profession_category, region_category
                )
                
                # CRITICAL: Explicitly ensure user_id is included
                sleep_data['user_id'] = user_id
                
                user_data.append(sleep_data)
                
            except Exception as e:
                print(f"Error generating sleep data for user {user_id} on day {current_date}: {e}")
            
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
        sleep_consistency = user.get('sleep_consistency', 0.7)  # Default if missing
        bedtime_variance_minutes = max(1, int(30 * (1 - sleep_consistency)))
        waketime_variance_minutes = max(1, int(30 * (1 - sleep_consistency)))
        
        # Special handling for variable sleepers
        if pattern == 'variable':
            # Use safe dictionary access with default values
            bedtime_variance = pattern_params.get('bedtime_variance_hours', [1, 2])
            var_hours = np.random.uniform(bedtime_variance[0], bedtime_variance[1])
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
        
        # Handle pattern-specific parameters for sleep onset - safe dictionary access
        if pattern == 'insomnia':
            # Safe access with defaults
            onset_range = pattern_params.get('sleep_onset_minutes', [30, 60])
            sleep_onset_minutes = np.random.randint(onset_range[0], onset_range[1])
            
            # Age factor - older people with insomnia often have longer sleep onset
            if user.get('age', 30) > 60:
                sleep_onset_minutes += np.random.randint(0, 15)
        else:
            sleep_onset_minutes = np.random.randint(5, 20)
            
            # Add profession impact - high stress jobs may have longer sleep onset
            if profession_category in ['healthcare', 'tech'] and np.random.random() < 0.3:
                sleep_onset_minutes += np.random.randint(5, 15)
        
        # Calculate sleep onset time
        sleep_onset_datetime = bed_datetime + timedelta(minutes=sleep_onset_minutes)
        
        # Generate sleep duration based on pattern - safe dictionary access
        sleep_duration_range = pattern_params.get('sleep_duration_hours', [6, 8])
        sleep_hours = np.random.uniform(sleep_duration_range[0], sleep_duration_range[1])
        
        # Weekend adjustment for sleep duration
        if is_weekend and pattern != 'shift_worker':
            sleep_hours += np.random.uniform(0, 1)  # Up to 1 hour extra sleep on weekends
        
        # Generate awakenings - safe dictionary access
        awakening_range = pattern_params.get('awakenings_count', [0, 2])
        awakenings = int(np.random.randint(awakening_range[0], awakening_range[1] + 1))
        
        # Age impacts awakenings - older people tend to wake more
        if user.get('age', 30) > 60:
            awakenings += np.random.randint(0, 2)
        
        # Calculate total time awake during night - safe dictionary access
        awakening_duration_range = pattern_params.get('awakening_duration_minutes', [3, 10])
        avg_awakening_minutes = np.random.randint(
            awakening_duration_range[0], 
            awakening_duration_range[1]
        )
        
        total_awake_minutes = awakenings * avg_awakening_minutes
        
        # Calculate sleep efficiency
        time_in_bed = (wake_datetime - bed_datetime).total_seconds() / 3600  # hours
        sleep_efficiency = sleep_hours / time_in_bed
        
        # Adjust for realistic limits - safe dictionary access
        if 'sleep_efficiency' in pattern_params:
            min_efficiency, max_efficiency = pattern_params['sleep_efficiency']
            sleep_efficiency = max(min_efficiency, min(sleep_efficiency, max_efficiency))
        else:
            # Default efficiency bounds
            sleep_efficiency = max(0.7, min(sleep_efficiency, 0.95))
        
        # Generate subjective rating - safe dictionary access
        rating_range = pattern_params.get('subjective_rating_range', [3, 9])
        subjective_rating = np.random.randint(rating_range[0], rating_range[1] + 1)
        
        # Get season using the base class method
        season = self.get_season_from_date(date)
        
        # CRITICAL: Ensure out of bed time is after wake time
        out_bed_time = wake_datetime + timedelta(minutes=np.random.randint(5, 30))
        
        # Create complete sleep record
        sleep_record = {
            'user_id': user.get('user_id'),  # CRITICAL: Include user_id!
            'date': date.strftime('%Y-%m-%d'),
            'bedtime': bed_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'sleep_onset_time': sleep_onset_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'wake_time': wake_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'out_bed_time': out_bed_time.strftime('%Y-%m-%d %H:%M:%S'),
            'time_in_bed_hours': float(time_in_bed),
            'sleep_duration_hours': float(sleep_hours),
            'sleep_onset_latency_minutes': int(sleep_onset_minutes),
            'awakenings_count': int(awakenings),
            'total_awake_minutes': int(total_awake_minutes),
            'sleep_efficiency': float(sleep_efficiency),
            'subjective_rating': int(subjective_rating),
            'is_weekend': bool(is_weekend),
            'profession_category': profession_category,
            'region_category': region_category,
            'season': season
        }
        
        return sleep_record