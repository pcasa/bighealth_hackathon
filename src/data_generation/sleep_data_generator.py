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
        """Generate sleep data for a set of users over a time period with improved error handling"""
        if start_date is None:
            try:
                start_date = datetime.strptime(self.time_settings['start_date'], '%Y-%m-%d')
            except (ValueError, KeyError):
                # Fallback if start_date is invalid or missing
                start_date = datetime(2024, 1, 1)
                print(f"Using fallback start_date: {start_date}")
                
        if end_date is None:
            try:
                end_date = datetime.strptime(self.time_settings['end_date'], '%Y-%m-%d')
            except (ValueError, KeyError):
                # Fallback if end_date is invalid or missing
                end_date = start_date + timedelta(days=90)  # Default to 90 days of data
                print(f"Using fallback end_date: {end_date}")
        
        # Ensure end_date is after start_date
        if end_date <= start_date:
            end_date = start_date + timedelta(days=90)
            print(f"Adjusted end_date to ensure it's after start_date: {end_date}")
        
        print(f"Generating sleep data from {start_date} to {end_date}")
        
        # Verify users_df has the required columns
        required_cols = ['user_id', 'sleep_pattern']
        missing_cols = [col for col in required_cols if col not in users_df.columns]
        if missing_cols:
            print(f"WARNING: users_df is missing required columns: {missing_cols}")
            print(f"Available columns: {users_df.columns.tolist()}")
            
            # Add missing required columns with default values
            if 'user_id' not in users_df.columns:
                users_df['user_id'] = [f"user_{i:04d}" for i in range(len(users_df))]
                print("Added default user_id column")
                
            if 'sleep_pattern' not in users_df.columns:
                # Default to 70% normal, 20% insomnia, 10% variable
                patterns = np.random.choice(
                    ['normal', 'insomnia', 'variable'],
                    size=len(users_df),
                    p=[0.7, 0.2, 0.1]
                )
                users_df['sleep_pattern'] = patterns
                print("Added default sleep_pattern column")
        
        all_sleep_data = []
        
        # Create a copy of users_df to avoid modifying the original
        users_df_copy = users_df.copy()
        
        # Ensure all string columns are actually strings, not numeric
        for col in users_df_copy.columns:
            if col in ['user_id', 'sleep_pattern', 'profession', 'region']:
                users_df_copy[col] = users_df_copy[col].astype(str)
        
        for _, user_row in users_df_copy.iterrows():
            # Ensure user_row has 'user_id'
            if 'user_id' not in user_row:
                print(f"WARNING: user_row is missing 'user_id', generating a new one")
                user_row['user_id'] = f"user_{np.random.randint(10000):04d}"
                    
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
                print(f"Error processing user {user_dict.get('user_id', 'unknown')}: {str(e)}")
                # Continue with next user instead of failing
                continue
        
        # Check if we generated any data
        if not all_sleep_data:
            print("WARNING: No sleep data was generated!")
            # Return an empty DataFrame with the expected columns
            return pd.DataFrame(columns=[
                'user_id', 'date', 'bedtime', 'sleep_onset_time', 'wake_time', 'out_bed_time',
                'time_in_bed_hours', 'sleep_duration_hours', 'sleep_onset_latency_minutes',
                'awakenings_count', 'total_awake_minutes', 'sleep_efficiency',
                'subjective_rating', 'is_weekend', 'profession_category', 'region_category',
                'season', 'no_sleep'
            ])
        
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
                unique_users = users_df_copy['user_id'].unique()
                if len(unique_users) > 0:
                    # Distribute user_ids proportionally
                    result_df['user_id'] = [unique_users[i % len(unique_users)] for i in range(len(result_df))]
        else:
            unique_user_ids = result_df['user_id'].unique()[:5].tolist()
            print(f"Successfully generated sleep data with user_id column. Sample unique users: {unique_user_ids}")
            
        # Ensure all required columns are present, with defaults if missing
        required_columns = [
            'user_id', 'date', 'bedtime', 'sleep_onset_time', 'wake_time', 'out_bed_time',
            'time_in_bed_hours', 'sleep_duration_hours', 'sleep_onset_latency_minutes',
            'awakenings_count', 'total_awake_minutes', 'sleep_efficiency',
            'subjective_rating', 'is_weekend'
        ]
        
        for col in required_columns:
            if col not in result_df.columns:
                print(f"WARNING: Required column '{col}' missing. Adding with default values.")
                if col == 'user_id':
                    result_df[col] = [f"user_{i:04d}" for i in range(len(result_df))]
                elif col == 'date':
                    result_df[col] = start_date + timedelta(days=1)
                elif col in ['bedtime', 'sleep_onset_time', 'wake_time', 'out_bed_time']:
                    # Generate sequential timestamps for each record
                    base_time = datetime(2024, 1, 1, 22, 0) # 10:00 PM base time
                    timestamps = []
                    for i in range(len(result_df)):
                        if col == 'bedtime':
                            ts = base_time + timedelta(days=i)
                        elif col == 'sleep_onset_time':
                            ts = base_time + timedelta(days=i, minutes=30)
                        elif col == 'wake_time':
                            ts = base_time + timedelta(days=i+1, hours=8)
                        else:  # out_bed_time
                            ts = base_time + timedelta(days=i+1, hours=8, minutes=15)
                        timestamps.append(ts.strftime('%Y-%m-%d %H:%M:%S'))
                    result_df[col] = timestamps
                elif col == 'time_in_bed_hours':
                    if 'bedtime' in result_df.columns and 'out_bed_time' in result_df.columns:
                        # Try to calculate from bedtime and out_bed_time
                        try:
                            bedtimes = pd.to_datetime(result_df['bedtime'])
                            out_bed_times = pd.to_datetime(result_df['out_bed_time'])
                            result_df[col] = ((out_bed_times - bedtimes).dt.total_seconds() / 3600).clip(lower=0.1)
                        except:
                            # Default to 8 hours if calculation fails
                            result_df[col] = 8.0
                    else:
                        # Default value
                        result_df[col] = 8.0
                elif col == 'sleep_duration_hours':
                    if 'sleep_efficiency' in result_df.columns and 'time_in_bed_hours' in result_df.columns:
                        # Calculate from efficiency and time in bed
                        result_df[col] = (result_df['sleep_efficiency'] * result_df['time_in_bed_hours']).clip(lower=0.1)
                    else:
                        # Default value
                        result_df[col] = 7.0
                elif col == 'sleep_onset_latency_minutes':
                    if 'bedtime' in result_df.columns and 'sleep_onset_time' in result_df.columns:
                        # Try to calculate from bedtime and sleep_onset_time
                        try:
                            bedtimes = pd.to_datetime(result_df['bedtime'])
                            onset_times = pd.to_datetime(result_df['sleep_onset_time'])
                            result_df[col] = ((onset_times - bedtimes).dt.total_seconds() / 60).clip(lower=1)
                        except:
                            # Default to 15 minutes if calculation fails
                            result_df[col] = 15
                    else:
                        # Default value
                        result_df[col] = 15
                elif col == 'awakenings_count':
                    # Default value
                    result_df[col] = 2
                elif col == 'total_awake_minutes':
                    # Default value
                    result_df[col] = 20
                elif col == 'sleep_efficiency':
                    if 'sleep_duration_hours' in result_df.columns and 'time_in_bed_hours' in result_df.columns:
                        # Calculate from duration and time in bed
                        time_in_bed = result_df['time_in_bed_hours'].clip(lower=0.1)  # Ensure > 0
                        result_df[col] = (result_df['sleep_duration_hours'] / time_in_bed).clip(lower=0, upper=1)
                    else:
                        # Default value
                        result_df[col] = 0.85
                elif col == 'subjective_rating':
                    # Default value
                    result_df[col] = 7
                elif col == 'is_weekend':
                    # If date column exists, calculate from date
                    if 'date' in result_df.columns:
                        try:
                            dates = pd.to_datetime(result_df['date'])
                            result_df[col] = dates.dt.dayofweek >= 5  # 5 = Saturday, 6 = Sunday
                        except:
                            # Default to False if calculation fails
                            result_df[col] = False
                    else:
                        # Default value
                        result_df[col] = False
        
        # Fill any remaining NaN values with reasonable defaults
        for col in result_df.columns:
            if result_df[col].isna().any():
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    # For numeric columns, fill with mean or a reasonable default
                    if result_df[col].count() > 0:
                        result_df[col] = result_df[col].fillna(result_df[col].mean())
                    else:
                        if 'efficiency' in col:
                            result_df[col] = result_df[col].fillna(0.85)
                        elif 'hours' in col:
                            result_df[col] = result_df[col].fillna(7.0)
                        elif 'minutes' in col:
                            result_df[col] = result_df[col].fillna(15)
                        elif 'count' in col:
                            result_df[col] = result_df[col].fillna(2)
                        else:
                            result_df[col] = result_df[col].fillna(0)
                else:
                    # For non-numeric columns, fill with most common value or a default
                    if result_df[col].count() > 0:
                        most_common = result_df[col].mode().iloc[0]
                        result_df[col] = result_df[col].fillna(most_common)
                    else:
                        if col == 'season':
                            result_df[col] = result_df[col].fillna('spring')
                        elif col == 'profession_category' or col == 'region_category':
                            result_df[col] = result_df[col].fillna('other')
                        else:
                            result_df[col] = result_df[col].fillna('unknown')
        
        return result_df
    
    # Updates to SleepDataGenerator in sleep_data_generator.py

    # Updated parameter list in the function signature:
    def _generate_daily_sleep(self, user, date, pattern, pattern_params, base_bedtime, base_waketime, 
                        profession_category=None, region_category=None, temporal_quality_modifier=0):
        """Generate sleep data for a single day with enhanced variability and temporal patterns"""
        
        # Ensure date is a datetime object
        if isinstance(date, str):
            date = pd.to_datetime(date)

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

        # Ensure wake_datetime is after bed_datetime
        if wake_datetime <= bed_datetime:
            # Add at least 4 hours to wake_datetime if it's not after bed_datetime
            wake_datetime = bed_datetime + timedelta(hours=4)
        # Ensure time in bed doesn't exceed 24 hours
        elif (wake_datetime - bed_datetime).total_seconds() > 24 * 3600:
            # Cap the maximum time in bed to 24 hours
            wake_datetime = bed_datetime + timedelta(hours=24)
        
        # ===== MODIFIED: "Bad night" probability calculation influenced by temporal patterns =====
        # Base probability of having a particularly bad night
        bad_night_probability = 0.05  # 5% chance of a bad night for most sleepers
        
        # Increase for insomnia pattern
        if pattern == 'insomnia':
            bad_night_probability = 0.25  # 25% chance of a particularly bad night
        elif pattern == 'variable':
            bad_night_probability = 0.15  # 15% chance for variable sleepers
        elif pattern == 'shift_worker':
            bad_night_probability = 0.12  # 12% chance for shift workers
        
        # Adjust based on temporal quality modifier
        # Negative modifier increases bad night probability
        if temporal_quality_modifier < 0:
            # Increase bad night probability when in a negative cycle
            # The -0.3 to 0.3 modifier can change probability by up to 30%
            bad_night_probability -= temporal_quality_modifier  # Subtract negative number = add
        
        # Cap the probability at 0.8 (80%)
        bad_night_probability = min(0.8, max(0.01, bad_night_probability))
        
        # Determine if this is a "bad night"
        is_bad_night = np.random.random() < bad_night_probability
        # ===== END MODIFIED =====
        
        # Handle pattern-specific parameters for sleep onset - safe dictionary access
        if pattern == 'insomnia':
            # Safe access with defaults
            onset_range = pattern_params.get('sleep_onset_minutes', [30, 60])
            
            # For insomnia, modulate onset based on whether it's a "bad night"
            if is_bad_night:
                # Much longer onset during bad nights (45-90 minutes)
                sleep_onset_minutes = np.random.randint(45, 90)
            else:
                sleep_onset_minutes = np.random.randint(onset_range[0], onset_range[1])
                
            # ===== NEW: Apply temporal modifier to sleep onset latency =====
            # Negative modifier increases onset time, positive decreases
            # We apply a larger effect for insomnia pattern
            temporal_onset_adjustment = int(-temporal_quality_modifier * 30)
            sleep_onset_minutes += temporal_onset_adjustment
            sleep_onset_minutes = max(5, sleep_onset_minutes)  # Ensure minimum of 5 minutes
            # ===== END NEW =====
            
            # Age factor - older people with insomnia often have longer sleep onset
            if user.get('age', 30) > 60:
                sleep_onset_minutes += np.random.randint(0, 15)
        else:
            # For non-insomnia patterns
            if is_bad_night:
                # Even normal sleepers have occasional nights with longer onset
                sleep_onset_minutes = np.random.randint(20, 45)
            else:
                sleep_onset_minutes = np.random.randint(5, 20)
                
            # ===== NEW: Apply temporal modifier to sleep onset latency =====
            # Smaller effect for non-insomnia patterns
            temporal_onset_adjustment = int(-temporal_quality_modifier * 15)
            sleep_onset_minutes += temporal_onset_adjustment
            sleep_onset_minutes = max(3, sleep_onset_minutes)  # Ensure minimum of 3 minutes
            # ===== END NEW =====
                
            # Add profession impact - high stress jobs may have longer sleep onset
            if profession_category in ['healthcare', 'tech'] and np.random.random() < 0.3:
                sleep_onset_minutes += np.random.randint(5, 15)
        
        # Calculate sleep onset time
        sleep_onset_datetime = bed_datetime + timedelta(minutes=sleep_onset_minutes)
        
        # Generate sleep duration based on pattern - safe dictionary access
        sleep_duration_range = pattern_params.get('sleep_duration_hours', [6, 8])
        
        # ===== MODIFIED: Adjust sleep duration based on temporal patterns =====
        # Apply temporal quality modifier to the duration range
        # Positive modifier increases sleep duration, negative decreases
        duration_adjustment = temporal_quality_modifier * 1.5  # Scale by 1.5 hours for significant effect
        
        if is_bad_night:
            # Substantially shorter sleep on bad nights (1-3 hours below minimum)
            min_duration = max(2, sleep_duration_range[0] - np.random.uniform(1, 3) + duration_adjustment)
            max_duration = max(min_duration + 1, sleep_duration_range[0] - 0.5 + duration_adjustment)
            sleep_hours = np.random.uniform(min_duration, max_duration)
        else:
            adjusted_min = sleep_duration_range[0] + duration_adjustment
            adjusted_max = sleep_duration_range[1] + duration_adjustment
            sleep_hours = np.random.uniform(adjusted_min, adjusted_max)
        
        # Ensure reasonable limits
        sleep_hours = max(0, min(14, sleep_hours))  # Cap between 0-14 hours
        # ===== END MODIFIED =====
        
        # Weekend adjustment for sleep duration
        if is_weekend and pattern != 'shift_worker' and not is_bad_night:
            sleep_hours += np.random.uniform(0, 1)  # Up to 1 hour extra sleep on weekends
        
        # ===== MODIFIED: Adjust awakenings based on temporal patterns =====
        # Generate awakenings - safe dictionary access
        awakening_range = pattern_params.get('awakenings_count', [0, 2])
        
        # Adjust awakenings for bad nights and temporal patterns
        if is_bad_night:
            # More frequent awakenings on bad nights
            min_awakenings = awakening_range[1]
            max_awakenings = min_awakenings + np.random.randint(2, 5)
            
            # Additional effect from temporal patterns (negative = more awakenings)
            awakening_adjustment = int(-temporal_quality_modifier * 2)
            max_awakenings += awakening_adjustment
            
            awakenings = int(np.random.randint(min_awakenings, max_awakenings + 1))
        else:
            # Base awakenings
            base_min = awakening_range[0]
            base_max = awakening_range[1]
            
            # Apply temporal modifier
            awakening_adjustment = int(-temporal_quality_modifier * 1.5)
            min_awakenings = max(0, base_min + awakening_adjustment)
            max_awakenings = max(min_awakenings + 1, base_max + awakening_adjustment)
            
            awakenings = int(np.random.randint(min_awakenings, max_awakenings + 1))
        # ===== END MODIFIED =====
        
        # Age impacts awakenings - older people tend to wake more
        if user.get('age', 30) > 60:
            awakenings += np.random.randint(0, 2)
        
        # Calculate total time awake during night - safe dictionary access
        awakening_duration_range = pattern_params.get('awakening_duration_minutes', [3, 10])
        
        # ===== MODIFIED: Adjust awakening duration based on temporal patterns =====
        # Adjust awakening duration for bad nights
        if is_bad_night:
            # Longer periods awake on bad nights
            min_awakening = awakening_duration_range[1]
            max_awakening = min_awakening + np.random.randint(10, 30)
            
            # Apply temporal effect (negative modifier = longer awakenings)
            awakening_adjustment = int(-temporal_quality_modifier * 10)
            max_awakening += awakening_adjustment
            
            avg_awakening_minutes = np.random.randint(min_awakening, max_awakening)
        else:
            base_min = awakening_duration_range[0]
            base_max = awakening_duration_range[1]
            
            # Apply temporal modifier
            awakening_adjustment = int(-temporal_quality_modifier * 5)
            min_awakening = max(1, base_min + awakening_adjustment)
            max_awakening = max(min_awakening + 2, base_max + awakening_adjustment)
            
            avg_awakening_minutes = np.random.randint(min_awakening, max_awakening)
        # ===== END MODIFIED =====
        
        total_awake_minutes = awakenings * avg_awakening_minutes
        
        # ===== MODIFIED: Special case for severe insomnia - affected by temporal patterns =====
        no_sleep_night = False
        
        # Chance of complete sleepless night for insomnia
        # Significantly higher chance during negative temporal periods
        if pattern == 'insomnia' and is_bad_night:
            no_sleep_probability = 0.15  # Base probability
            
            # Adjust based on temporal modifier
            if temporal_quality_modifier < 0:
                no_sleep_probability -= temporal_quality_modifier * 2  # Double the effect
                no_sleep_probability = min(0.6, no_sleep_probability)  # Cap at 60%
            
            if np.random.random() < no_sleep_probability:
                no_sleep_night = True
                sleep_hours = 0
                awakenings = 0  # Can't have awakenings if no sleep
                sleep_onset_minutes = (wake_datetime - bed_datetime).total_seconds() / 60  # Never fell asleep
                total_awake_minutes = (wake_datetime - bed_datetime).total_seconds() / 60  # Awake all night
        # ===== END MODIFIED =====
        
        # Calculate sleep efficiency
        time_in_bed = (wake_datetime - bed_datetime).total_seconds() / 3600  # hours
        # Add after calculating time_in_bed
        if time_in_bed > 24.0:
            # Cap to 24 hours and adjust wake_datetime
            time_in_bed = 24.0
            wake_datetime = bed_datetime + timedelta(hours=24)
        sleep_efficiency = 0 if time_in_bed == 0 else sleep_hours / time_in_bed  # Avoid division by zero
        
        # ===== MODIFIED: Apply direct temporal effect to sleep efficiency =====
        # Adjust for realistic limits - safe dictionary access
        if 'sleep_efficiency' in pattern_params:
            min_efficiency, max_efficiency = pattern_params['sleep_efficiency']
            # For bad nights, allow efficiency to go below the pattern minimum
            if is_bad_night:
                min_efficiency = max(0.15, min_efficiency - 0.25)  # Much lower minimum on bad nights
                max_efficiency = min(max_efficiency, min_efficiency + 0.15)  # Lower maximum too
            
            # Apply temporal effect directly to efficiency
            efficiency_adjustment = temporal_quality_modifier * 0.15  # Scale the effect (±15% maximum)
            min_efficiency += efficiency_adjustment
            max_efficiency += efficiency_adjustment
            
            # Ensure valid ranges
            min_efficiency = max(0.0, min(0.9, min_efficiency))
            max_efficiency = max(min_efficiency + 0.05, min(0.95, max_efficiency))
            
            # For no sleep nights, efficiency is 0
            if no_sleep_night:
                sleep_efficiency = 0
            else:
                sleep_efficiency = max(min_efficiency, min(sleep_efficiency, max_efficiency))
        else:
            # Default efficiency bounds with bad night and temporal adjustments
            if is_bad_night:
                min_bound = 0.4  # Lower minimum for bad nights
                max_bound = 0.7  # Lower maximum for bad nights
            else:
                min_bound = 0.7
                max_bound = 0.95
                
            # Apply temporal effect
            efficiency_adjustment = temporal_quality_modifier * 0.15
            min_bound += efficiency_adjustment
            max_bound += efficiency_adjustment
            
            # Ensure valid ranges
            min_bound = max(0.0, min(0.9, min_bound))
            max_bound = max(min_bound + 0.05, min(0.95, max_bound))
                
            # For no sleep nights, efficiency is 0
            if no_sleep_night:
                sleep_efficiency = 0
            else:
                sleep_efficiency = max(min_bound, min(sleep_efficiency, max_bound))
        # ===== END MODIFIED =====
        
        # ===== MODIFIED: Adjust subjective rating based on temporal patterns =====
        # Generate subjective rating
        if no_sleep_night:
            # For no sleep nights, rating is very low
            subjective_rating = np.random.randint(1, 3)
        elif is_bad_night:
            # For bad nights, use lower range with temporal influence
            base_max_rating = 5  # Base maximum for bad nights
            
            # Apply temporal effect (negative = lower ratings)
            rating_adjustment = int(temporal_quality_modifier * 2)
            max_rating = max(3, min(7, base_max_rating + rating_adjustment))
            
            subjective_rating = np.random.randint(1, max_rating)
        else:
            # Normal nights
            rating_range = pattern_params.get('subjective_rating_range', [3, 9])
            
            # Apply temporal effect
            rating_adjustment = int(temporal_quality_modifier * 2)
            min_rating = max(1, rating_range[0] + rating_adjustment)
            max_rating = max(min_rating + 1, min(10, rating_range[1] + rating_adjustment))
            
            subjective_rating = np.random.randint(min_rating, max_rating + 1)
        # ===== END MODIFIED =====
        
        # Get season using the base class method
        season = self.get_season_from_date(date)
        
        # CRITICAL: Ensure out of bed time is after wake time
        out_bed_time = wake_datetime + timedelta(minutes=np.random.randint(5, 30))
        
        # Create complete sleep record with temporal information
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
            'season': season,
            'no_sleep': no_sleep_night,  # Track complete sleepless nights
            'is_bad_night': is_bad_night,  # Flag for bad nights for debugging
            'temporal_modifier': temporal_quality_modifier  # Store for analysis
        }
        
        return sleep_record
    # Add this method to SleepDataGenerator class in sleep_data_generator.py

    # Add this method to SleepDataGenerator class in sleep_data_generator.py
    def _generate_user_sleep_data(self, user, start_date, end_date, profession_category=None, region_category=None):
        """Generate sleep data for a single user over a time period with temporal patterns"""
        user_data = []

        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
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
        
        # ===== NEW: TEMPORAL PATTERN VARIABLES =====
        # Sleep quality cycle length (days) - different for each user
        if pattern == 'variable':
            cycle_length = np.random.randint(3, 8)  # Shorter cycles for variable sleepers
        else:
            cycle_length = np.random.randint(10, 30)  # 10-30 day cycles for most people
        
        # Amplitude of cycle (how much it affects sleep) - different by pattern
        if pattern == 'insomnia':
            cycle_amplitude = np.random.uniform(0.15, 0.3)  # Stronger cycles for insomnia
        elif pattern == 'variable':
            cycle_amplitude = np.random.uniform(0.2, 0.35)  # Strongest cycles for variable sleepers
        else:
            cycle_amplitude = np.random.uniform(0.05, 0.15)  # Milder cycles for normal sleepers
        
        # Random phase offset for the cycle
        cycle_phase = np.random.uniform(0, 2 * np.pi)
        
        # Stress events - random periods of worse sleep
        stress_events = []
        # Number of stress events depends on the date range
        date_range_days = (end_date - start_date).days
        avg_events_per_year = 3
        expected_events = max(1, int(date_range_days / 365 * avg_events_per_year))
        
        # Increase events for high-stress professions
        if profession_category in ['healthcare', 'tech']:
            expected_events = int(expected_events * 1.5)
        
        # Generate random stress events
        for _ in range(np.random.poisson(expected_events)):
            # Random start within the date range
            event_start_day = np.random.randint(0, date_range_days)
            event_start = start_date + timedelta(days=event_start_day)
            # Duration between 3-14 days
            event_duration = np.random.randint(3, 15)
            # Severity between 0.2-0.5 (will reduce sleep quality)
            event_severity = np.random.uniform(0.2, 0.5)
            
            stress_events.append({
                'start': event_start,
                'end': event_start + timedelta(days=event_duration),
                'severity': event_severity
            })
        
        # Seasonal pattern - some regions have seasonal sleep effects
        seasonal_effect = 0.0
        if region_category == 'north_america':
            seasonal_effect = 0.1  # Moderate seasonal effect
        elif region_category == 'europe':
            seasonal_effect = 0.15  # Stronger seasonal effect for European regions
        
        # Progressive improvement/decline trends
        # Some users might be gradually improving or declining
        trend_direction = np.random.choice(['improving', 'declining', 'stable'], 
                                        p=[0.3, 0.2, 0.5])  # 30% improving, 20% declining, 50% stable
        trend_strength = np.random.uniform(0.001, 0.003)  # Small daily change
        
        # Sleep habits consistency improves over time for some users
        consistency_improves = np.random.random() < 0.4  # 40% chance of consistency improvement
        starting_consistency = user.get('sleep_consistency', 0.7)
        max_consistency_improvement = 0.2  # Maximum 20% improvement
        # ===== END NEW =====
        
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
        day_index = 0  # Counter for days since start
        
        while current_date <= end_date:
            # ===== NEW: Apply temporal patterns to adjust sleep quality =====
            # Calculate current consistency - may improve over time
            current_consistency = starting_consistency
            if consistency_improves:
                # Consistency improves gradually over time
                days_fraction = min(1.0, day_index / date_range_days)
                consistency_improvement = days_fraction * max_consistency_improvement
                current_consistency = min(0.95, starting_consistency + consistency_improvement)
            
            # Calculate cyclic pattern effect (sine wave)
            cycle_effect = cycle_amplitude * np.sin(2 * np.pi * day_index / cycle_length + cycle_phase)
            
            # Calculate seasonal effect
            if seasonal_effect > 0:
                month = current_date.month
                # Seasonal effect - worse sleep in winter for northern hemisphere
                # Sinusoidal pattern with worst in January (month 1)
                month_in_year_fraction = (month - 1) / 12  # 0-1 scale where 0 is January
                seasonal_position = np.sin(2 * np.pi * month_in_year_fraction + np.pi)  # +pi shifts so winter is worst
                seasonal_modifier = -seasonal_effect * seasonal_position  # Negative in winter, positive in summer
            else:
                seasonal_modifier = 0

            # Calculate trend effect (improving or declining over time)
            if trend_direction == 'improving':
                trend_modifier = trend_strength * day_index  # Positive trend over time
            elif trend_direction == 'declining':
                trend_modifier = -trend_strength * day_index  # Negative trend over time
            else:
                trend_modifier = 0  # No trend for stable users
            
            # Check for stress events
            stress_modifier = 0
            for event in stress_events:
                if event['start'] <= current_date <= event['end']:
                    stress_modifier = -event['severity']  # Negative effect during stress events
                    break
            
            # Combine all temporal effects
            temporal_quality_modifier = cycle_effect + seasonal_modifier + trend_modifier + stress_modifier
            # ===== END NEW =====
            
            # Check if user skipped logging this day based on consistency
            data_consistency = current_consistency  # Use the potentially improved consistency
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
                    day_index += 1
                    continue
            
            # Generate sleep data for this day, passing the temporal modifier
            try:
                sleep_data = self._generate_daily_sleep(
                    user, current_date, pattern, pattern_params, 
                    base_bedtime, base_waketime,
                    profession_category, region_category,
                    temporal_quality_modifier  # NEW: Pass temporal modifier
                )
                
                # CRITICAL: Explicitly ensure user_id is included
                sleep_data['user_id'] = user_id
                
                user_data.append(sleep_data)
                
            except Exception as e:
                print(f"Error generating sleep data for user {user_id} on day {current_date}: {e}")
            
            current_date += timedelta(days=1)
            day_index += 1
        
        return user_data