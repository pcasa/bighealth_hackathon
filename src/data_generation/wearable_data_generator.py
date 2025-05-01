import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml

from src.data_generation.base_generator import BaseDataGenerator

class WearableDataGenerator(BaseDataGenerator):
    """
    Wearable data generator that inherits from the BaseDataGenerator.
    Generates synthetic wearable device data based on sleep patterns and device specifications.
    """
    def __init__(self, config_path='src/config/data_generation_config.yaml', device_config_path='src/config/device_profiles.yaml'):
        # Initialize the base generator
        super().__init__(config_path)
        
        # Load device-specific configuration
        with open(device_config_path, 'r') as file:
            self.device_config = yaml.safe_load(file)
        
        self.device_params = self.config['device_parameters']
    
    def generate_wearable_data(self, sleep_data_df, users_df):
        """Generate wearable device data based on sleep data"""
        all_wearable_data = []
        
        # Ensure both DataFrames have the user_id column
        if 'user_id' not in sleep_data_df.columns:
            print("Error: 'user_id' column missing from sleep_data_df")
            return pd.DataFrame()
            
        if 'user_id' not in users_df.columns:
            print("Error: 'user_id' column missing from users_df")
            return pd.DataFrame()
        
        # Ensure users_df has the device_type column
        if 'device_type' not in users_df.columns:
            print("Error: 'device_type' column missing from users_df")
            return pd.DataFrame()
        
        # Print column names for debugging
        print(f"Sleep data columns: {sleep_data_df.columns.tolist()}")
        print(f"Users data columns: {users_df.columns.tolist()}")
        
        try:
            # Join sleep data with user data to get device type
            merged_data = pd.merge(sleep_data_df, users_df[['user_id', 'device_type']], on='user_id')
            
            print(f"Successfully merged data with {len(merged_data)} rows")
            
            for _, record in merged_data.iterrows():
                try:
                    wearable_data = self._generate_record_wearable_data(record)
                    all_wearable_data.append(wearable_data)
                except Exception as e:
                    print(f"Error generating wearable data for record: {e}")
                    continue
            
            return pd.DataFrame(all_wearable_data)
        except Exception as e:
            print(f"Error in generate_wearable_data: {e}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()
    
    # Fix for the WearableDataGenerator's _generate_record_wearable_data method

    def _generate_record_wearable_data(self, record):
        """Generate wearable data for a single sleep record with division by zero protection"""
        device_type = record['device_type']
        device_params = self.device_params[device_type]
        device_profile = self.device_config[device_type]
        
        # Convert string times to datetime objects
        try:
            bedtime = datetime.strptime(record['bedtime'], '%Y-%m-%d %H:%M:%S')
            sleep_onset = datetime.strptime(record['sleep_onset_time'], '%Y-%m-%d %H:%M:%S')
            wake_time = datetime.strptime(record['wake_time'], '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError) as e:
            print(f"Error parsing datetime: {e}")
            # Provide default values if datetime parsing fails
            current_date = datetime.now()
            bedtime = current_date - timedelta(hours=8)
            sleep_onset = bedtime + timedelta(minutes=20)
            wake_time = current_date
        
        # Add device measurement variance with the base class noise generation
        bedtime_variance_minutes = self.generate_time_based_noise(0, 0.5)  # ~5 minutes standard deviation
        sleep_onset_variance_minutes = self.generate_time_based_noise(0, 1.0)  # ~10 minutes standard deviation
        wake_time_variance_minutes = self.generate_time_based_noise(0, 0.5)  # ~5 minutes standard deviation
        
        # Convert to integer minutes
        bedtime_variance = int(bedtime_variance_minutes)
        sleep_onset_variance = int(sleep_onset_variance_minutes)
        wake_time_variance = int(wake_time_variance_minutes)
        
        device_bedtime = bedtime + timedelta(minutes=bedtime_variance)
        device_sleep_onset = sleep_onset + timedelta(minutes=sleep_onset_variance)
        device_wake_time = wake_time + timedelta(minutes=wake_time_variance)
        
        # CRITICAL FIX: Ensure device_sleep_onset is before device_wake_time
        if device_sleep_onset >= device_wake_time:
            # If times are invalid, adjust wake time to be after sleep onset
            device_wake_time = device_sleep_onset + timedelta(minutes=30)
        
        # Calculate sleep duration with division by zero protection
        device_sleep_duration = max(0.1, (device_wake_time - device_sleep_onset).total_seconds() / 3600)
        
        # CRITICAL FIX: Handle special case for no_sleep nights
        if 'no_sleep' in record and record['no_sleep']:
            # For no sleep nights, generate special sleep stage data
            sleep_stages = {
                'deep': 0.0,
                'light': 0.1,  # Some light sleep even on "no sleep" nights
                'rem': 0.0,
                'awake': 0.9,  # Mostly awake
                'cycles': 0,
                'sequence': ['awake'] * int((device_wake_time - device_sleep_onset).total_seconds() / 60)
            }
        else:
            # Normal sleep stage generation
            sleep_stages = self._generate_sleep_stages(
                device_type, device_sleep_onset, device_wake_time, device_sleep_duration, 
                record.get('profession_category', None)  # Pass profession if available
            )
        
        # Generate heart rate data
        heart_rate_data = self._generate_heart_rate_data(
            device_type, device_profile, device_bedtime, device_wake_time, 
            # CRITICAL FIX: Handle missing or zero sleep_efficiency
            record.get('sleep_efficiency', 0.7) or 0.7,  # Default to 0.7 if missing or zero
            record.get('awakenings_count', 0) or 0  # Default to 0 if missing or None
        )
        
        # Generate movement data
        movement_data = self._generate_movement_data(
            device_type, device_profile, device_sleep_onset, device_wake_time,
            record.get('awakenings_count', 0) or 0,  # Default to 0 if missing or None
            record.get('total_awake_minutes', 0) or 0  # Default to 0 if missing or None
        )
        
        # Calculate device-specific metrics with division by zero protection
        # CRITICAL FIX: Add safety checks for heart_rate_data
        if heart_rate_data and len(heart_rate_data) > 1:
            hrv = self._calculate_hrv(device_type, heart_rate_data, 
                                    record.get('sleep_efficiency', 0.7) or 0.7)
        else:
            # Default HRV if heart rate data is invalid
            hrv = 45.0
        
        # Blood oxygen if device supports it
        blood_oxygen = None
        if 'blood_oxygen' in device_profile['data_fields']:
            blood_oxygen = self._generate_blood_oxygen(
                record.get('sleep_efficiency', 0.7) or 0.7)
        
        # Get season using base class method (if date available)
        season = None
        if 'date' in record:
            try:
                date_obj = datetime.strptime(record['date'], '%Y-%m-%d')
                season = self.get_season_from_date(date_obj)
            except (ValueError, TypeError):
                # If date parsing fails, leave season as None
                pass
        
        return {
            'user_id': record['user_id'],
            'date': record.get('date', datetime.now().strftime('%Y-%m-%d')),
            'device_type': device_type,
            'device_bedtime': device_bedtime.strftime('%Y-%m-%d %H:%M:%S'),
            'device_sleep_onset': device_sleep_onset.strftime('%Y-%m-%d %H:%M:%S'),
            'device_wake_time': device_wake_time.strftime('%Y-%m-%d %H:%M:%S'),
            'device_sleep_duration': device_sleep_duration,  # Now guaranteed to be > 0
            'deep_sleep_percentage': sleep_stages['deep'],
            'light_sleep_percentage': sleep_stages['light'],
            'rem_sleep_percentage': sleep_stages['rem'],
            'awake_percentage': sleep_stages['awake'],
            'sleep_cycles': sleep_stages['cycles'],
            'average_heart_rate': np.mean(heart_rate_data) if heart_rate_data else 60,
            'min_heart_rate': np.min(heart_rate_data) if heart_rate_data else 50,
            'max_heart_rate': np.max(heart_rate_data) if heart_rate_data else 70,
            'heart_rate_variability': hrv,
            'movement_intensity': np.mean(movement_data) if movement_data else 0.1,
            'blood_oxygen': blood_oxygen,
            'heart_rate_data': heart_rate_data or [60, 60],  # Default if missing
            'movement_data': movement_data or [0.1, 0.1],    # Default if missing
            'sleep_stage_data': sleep_stages['sequence'],
            'season': season,
            'profession_category': record.get('profession_category', None),
            'region_category': record.get('region_category', None)
        }
    
    def _generate_sleep_stages(self, device_type, sleep_onset, wake_time, sleep_duration, profession=None):
        """Generate sleep stage percentages and sequence with profession-specific adjustments and division by zero protection"""
        # CRITICAL: Ensure sleep_duration is positive
        sleep_duration = max(0.1, sleep_duration)  # Minimum of 0.1 hour to prevent division by zero
        
        # CRITICAL: Ensure wake_time is after sleep_onset
        if wake_time <= sleep_onset:
            wake_time = sleep_onset + timedelta(minutes=30)  # Ensure at least 30 minutes of sleep
        
        # Base percentages for a normal sleeper
        base_deep = 0.20
        base_light = 0.50
        base_rem = 0.25
        base_awake = 0.05
        
        # Adjust based on sleep duration
        if sleep_duration < 6:
            # Less deep and REM for short sleepers
            deep_adj = -0.05
            rem_adj = -0.05
            light_adj = 0.05
            awake_adj = 0.05
        elif sleep_duration > 8:
            # More REM for long sleepers
            deep_adj = 0.0
            rem_adj = 0.05
            light_adj = -0.02
            awake_adj = -0.03
        else:
            deep_adj = 0.0
            rem_adj = 0.0
            light_adj = 0.0
            awake_adj = 0.0
        
        # Apply profession-specific adjustments
        if profession == 'healthcare':
            # Healthcare workers often have disrupted sleep architecture
            deep_adj -= 0.03
            awake_adj += 0.03
        elif profession == 'tech':
            # Tech workers often have screen time close to bedtime, affecting REM
            rem_adj -= 0.02
        elif profession == 'shift_worker':
            # Shift workers have generally lower sleep quality
            deep_adj -= 0.04
            awake_adj += 0.04
        
        # Add random variance using the base generator's noise function
        deep = base_deep + deep_adj + self.generate_time_based_noise(0, 0.03)
        rem = base_rem + rem_adj + self.generate_time_based_noise(0, 0.03)
        awake = base_awake + awake_adj + self.generate_time_based_noise(0, 0.01)
        
        # Ensure light makes up the remainder
        light = 1.0 - deep - rem - awake
        
        # Adjust for valid ranges
        deep = max(0.05, min(deep, 0.30))
        rem = max(0.10, min(rem, 0.35))
        awake = max(0.01, min(awake, 0.15))
        light = max(0.30, min(light, 0.70))
        
        # Normalize to ensure they sum to 1
        total = deep + light + rem + awake
        
        # CRITICAL: Prevent division by zero
        if total == 0:
            # If for some reason all values are zero, use default proportions
            deep = 0.20
            light = 0.50
            rem = 0.25
            awake = 0.05
            total = 1.0
        
        # Now we can safely divide
        deep /= total
        light /= total
        rem /= total
        awake /= total
        
        # Calculate approximate number of sleep cycles (90-110 minutes each)
        cycle_length_hours = np.random.uniform(1.5, 1.8)  # 90-108 minutes
        cycles = sleep_duration / cycle_length_hours
        
        # Generate a simplified sleep stage sequence
        # CRITICAL: Ensure total_minutes is positive
        total_minutes = max(1, int((wake_time - sleep_onset).total_seconds() / 60))
        
        try:
            sequence = self._generate_stage_sequence(
                total_minutes, deep, light, rem, awake, cycles
            )
        except Exception as e:
            # Fallback if sequence generation fails
            print(f"Error generating sleep stage sequence: {e}. Using fallback.")
            # Create a simple fallback sequence
            sequence = ['light'] * min(30, total_minutes)
            if total_minutes > 30:
                sequence.extend(['deep'] * min(60, total_minutes - 30))
            if total_minutes > 90:
                sequence.extend(['rem'] * min(30, total_minutes - 90))
            if total_minutes > 120:
                # Fill the rest with a pattern
                remaining = total_minutes - 120
                pattern = ['light', 'deep', 'rem', 'light'] * ((remaining // 4) + 1)
                sequence.extend(pattern[:remaining])
        
        return {
            'deep': deep,
            'light': light,
            'rem': rem,
            'awake': awake,
            'cycles': max(0, np.floor(cycles)),  # Ensure non-negative
            'sequence': sequence
        }

    def _generate_stage_sequence(self, total_minutes, deep_pct, light_pct, rem_pct, awake_pct, cycles):
        """Generate a minute-by-minute sequence of sleep stages with error handling"""
        # CRITICAL: Safety checks for inputs
        total_minutes = max(1, total_minutes)  # Ensure positive
        cycles = max(0.1, cycles)  # Ensure positive
        
        # Ensure percentages are valid (0-1 range)
        deep_pct = max(0.01, min(0.99, deep_pct))
        light_pct = max(0.01, min(0.99, light_pct))
        rem_pct = max(0.01, min(0.99, rem_pct))
        awake_pct = max(0.01, min(0.99, awake_pct))
        
        # Renormalize to ensure they sum to 1
        total = deep_pct + light_pct + rem_pct + awake_pct
        deep_pct /= total
        light_pct /= total
        rem_pct /= total
        awake_pct /= total
        
        # Simplified model: Start with light sleep, then cycles of deep->light->REM
        # with occasional awakenings
        stages = []
        cycle_minutes = total_minutes / cycles
        
        # First cycle typically has more deep sleep and less REM
        first_cycle_deep = deep_pct * 1.3
        first_cycle_rem = rem_pct * 0.5
        
        # Last cycles typically have more REM and less deep sleep
        last_cycle_deep = deep_pct * 0.7
        last_cycle_rem = rem_pct * 1.5
        
        # CRITICAL: Ensure valid cycle count to avoid infinite loops
        cycle_count = int(np.ceil(cycles))
        cycle_count = max(1, min(10, cycle_count))  # Limit to 1-10 cycles for safety
        
        for cycle in range(cycle_count):
            cycle_fraction = cycle / cycle_count
            
            # Adjust deep sleep percentage based on cycle number
            if cycle == 0:
                cycle_deep_pct = first_cycle_deep
                cycle_rem_pct = first_cycle_rem
            elif cycle >= cycle_count - 1:
                cycle_deep_pct = last_cycle_deep
                cycle_rem_pct = last_cycle_rem
            else:
                # Gradual transition
                cycle_deep_pct = first_cycle_deep - cycle_fraction * (first_cycle_deep - last_cycle_deep)
                cycle_rem_pct = first_cycle_rem + cycle_fraction * (last_cycle_rem - first_cycle_rem)
            
            # Adjust light to maintain proportions
            cycle_light_pct = 1.0 - cycle_deep_pct - cycle_rem_pct - awake_pct
            cycle_light_pct = max(0.1, cycle_light_pct)  # Ensure minimum of 10% light sleep
            
            # Determine minutes for each stage in this cycle
            if cycle == cycle_count - 1:
                # Last cycle might be shorter
                remaining_minutes = total_minutes - len(stages)
                remaining_minutes = max(1, remaining_minutes)  # Ensure positive
                cycle_minutes = remaining_minutes
            
            # CRITICAL: Ensure all minute calculations are positive
            light_start = max(1, int(cycle_minutes * 0.1))
            deep_minutes = max(1, int(cycle_minutes * cycle_deep_pct))
            light_minutes = max(1, int(cycle_minutes * cycle_light_pct) - light_start)
            rem_minutes = max(1, int(cycle_minutes * cycle_rem_pct))
            awake_minutes = max(0, int(cycle_minutes * awake_pct))  # Awake can be 0
            
            # CRITICAL: Check if adding all these minutes exceeds the total we need
            # If so, proportionately scale down to fit
            cycle_total = light_start + deep_minutes + light_minutes + rem_minutes + awake_minutes
            if cycle_total > cycle_minutes:
                scale_factor = cycle_minutes / cycle_total
                light_start = max(1, int(light_start * scale_factor))
                deep_minutes = max(1, int(deep_minutes * scale_factor))
                light_minutes = max(1, int(light_minutes * scale_factor))
                rem_minutes = max(1, int(rem_minutes * scale_factor))
                awake_minutes = max(0, int(awake_minutes * scale_factor))
            
            # Add initial light sleep
            stages.extend(['light'] * light_start)
            
            # Add deep sleep
            stages.extend(['deep'] * deep_minutes)
            
            # Add light sleep
            stages.extend(['light'] * light_minutes)
            
            # Add REM sleep
            stages.extend(['rem'] * rem_minutes)
            
            # Add brief awakening if not the first cycle
            if cycle > 0 and awake_minutes > 0:
                # Place awakening at end of cycle
                stages.extend(['awake'] * awake_minutes)
            
            # CRITICAL: Check if we've already exceeded the total minutes
            if len(stages) >= total_minutes:
                # Truncate to the exact length needed
                stages = stages[:total_minutes]
                break
        
        # Ensure we have exactly the right number of minutes
        if len(stages) < total_minutes:
            # Add light sleep to make up difference
            stages.extend(['light'] * (total_minutes - len(stages)))
        elif len(stages) > total_minutes:
            # Trim excess
            stages = stages[:total_minutes]
        
        return stages
    
    def _generate_heart_rate_data(self, device_type, device_profile, bedtime, wake_time, sleep_efficiency, awakenings):
        """Generate heart rate data for the sleep period using base class noise generation"""
        # Ensure bedtime is before wake_time
        if bedtime >= wake_time:
            # Handle error case - generate a minimal valid dataset
            return [60, 60]  # Return minimal valid heart rate data
        
        # Determine sampling rate from device
        sampling_minutes = device_profile['heart_rate']['sampling_rate_minutes']
        
        # Calculate number of samples (ensure at least 1)
        total_minutes = max(1, int((wake_time - bedtime).total_seconds() / 60))
        num_samples = max(1, total_minutes // sampling_minutes + 1)
        
        # Base heart rate pattern: higher at start, lower during deep sleep, increases during REM
        base_hr = np.random.randint(58, 68)  # Resting heart rate
        
        # Generate heart rate time series
        heart_rates = []
        
        # Initial higher heart rate during sleep onset
        initial_hr = base_hr + np.random.randint(5, 15)
        
        # Sleep efficiency affects overall heart rate variability (with bounds)
        sleep_efficiency = max(0.01, min(1.0, sleep_efficiency))  # Bound between 0.01 and 1.0
        variability = 2.0 + (1.0 - sleep_efficiency) * 10
        
        for i in range(num_samples):
            time_fraction = i / max(1, num_samples - 1)  # Avoid division by zero
            
            # Heart rate typically decreases in first third, then has periods of increase during REM
            if time_fraction < 0.2:
                # Initial settling down period (with safe calculation)
                hr = initial_hr - time_fraction * 5  # Simplified calculation
            elif time_fraction < 0.4:
                # Deepest sleep often in first half
                hr = base_hr - 5
            else:
                # More REM in later cycles, heart rate increases
                hr = base_hr + np.sin(time_fraction * 15) * 4
            
            # Add noise using the base generator method
            hr += self.generate_time_based_noise(0, variability/10)  # scale appropriately
            
            # Add spikes for awakenings (with safe probability)
            spike_probability = min(0.95, (awakenings / max(1, num_samples)) * 3)  # Cap at 0.95
            if awakenings > 0 and np.random.random() < spike_probability:
                hr += np.random.randint(10, 25)
            
            heart_rates.append(max(45, int(hr)))
        
        return heart_rates
    
    def _generate_movement_data(self, device_type, device_profile, sleep_onset, wake_time, awakenings, awake_minutes):
        """Generate movement intensity data for the sleep period"""
        # Movement data is typically less granular than heart rate
        sampling_minutes = 5  # Standard 5-minute intervals for movement
        
        # Calculate number of samples
        total_minutes = int((wake_time - sleep_onset).total_seconds() / 60)
        num_samples = total_minutes // sampling_minutes + 1
        
        # Movement sensitivity from device profile
        sensitivity = device_profile['movement']['sensitivity']
        noise_factor = device_profile['movement']['noise_factor']
        
        # Generate movement data
        movement_data = []
        
        # Calculate average movement per sample during awakenings
        awakening_movement = min(1.0, awake_minutes / (awakenings * 5) if awakenings > 0 else 0)
        
        for i in range(num_samples):
            time_fraction = i / num_samples
            
            # Base movement is low during sleep
            base_movement = 0.05 + np.random.random() * 0.1
            
            # Add some structured variation
            if np.random.random() < (awakenings / 20):
                # Occasional movement spike for turning over
                movement = np.random.uniform(0.3, 0.7)
            elif np.random.random() < awakening_movement:
                # Higher movement during awakening periods
                movement = np.random.uniform(0.5, 0.9)
            else:
                # Normal low sleep movement
                movement = base_movement
            
            # Add noise using base generator method
            movement += self.generate_time_based_noise(0, noise_factor)
            
            # Apply device sensitivity
            movement *= sensitivity
            
            # Ensure valid range
            movement = max(0, min(movement, 1.0))
            
            movement_data.append(round(movement, 3))
        
        return movement_data
    
    def _calculate_hrv(self, device_type, heart_rate_data, sleep_efficiency):
        """Calculate heart rate variability from heart rate data with enhanced error handling"""
        # First, ensure heart_rate_data is valid
        if heart_rate_data is None or not isinstance(heart_rate_data, (list, np.ndarray)) or len(heart_rate_data) < 2:
            # Not enough data to calculate HRV
            base_hrv = np.random.uniform(35, 65)  # Generate a reasonable random HRV
            device_factor = 1.0  # Default device factor
            
            # Adjust based on device type
            if device_type == 'apple_watch':
                device_factor = 1.1  # Higher quality detection
            elif device_type == 'fitbit':
                device_factor = 0.9  # Slightly lower precision
                
            return round(base_hrv * device_factor, 1)
        
        try:
            # Convert to numpy array if it's not already
            hr_array = np.array(heart_rate_data, dtype=float)
            
            # Handle infinite or NaN values
            hr_array = np.nan_to_num(hr_array, nan=60, posinf=120, neginf=40)
            
            # Calculate time differences between consecutive heart rates
            diffs = np.diff(hr_array)
            
            # If all differences are zero or the array is empty
            if len(diffs) == 0 or np.all(diffs == 0):
                base_hrv = 25.0 + np.random.uniform(-5, 5)  # Low HRV (rigid heart rate)
            else:
                # Calculate RMSSD (root mean square of successive differences)
                # This is a common HRV measure
                squared_diffs = np.square(diffs)
                mean_squared_diff = np.mean(squared_diffs)
                rmssd = np.sqrt(max(0.01, mean_squared_diff))  # Ensure positive value
                
                # Ensure sleep_efficiency is in valid range
                sleep_efficiency = max(0.01, min(1.0, sleep_efficiency if sleep_efficiency is not None else 0.8))
                
                # Base HRV calculation - higher efficiency typically means higher HRV
                base_hrv = rmssd * (0.8 + sleep_efficiency * 0.4)
            
            # Apply device-specific factors
            device_factor = 1.0  # Default
            if device_type == 'apple_watch':
                device_factor = 1.1  # Higher quality detection
            elif device_type == 'fitbit':
                device_factor = 0.9  # Slightly lower precision
                
            # Add natural variation
            variation = self.generate_time_based_noise(0.1, 0.05)
            
            # Calculate final HRV with bounds
            hrv = base_hrv * device_factor * (0.9 + variation)
            
            # Ensure within realistic limits (20-100 ms is typical range for adults)
            hrv = max(20, min(100, hrv))
            
            return round(hrv, 1)
            
        except Exception as e:
            # If anything goes wrong, provide a fallback value
            print(f"Error calculating HRV: {e}")
            return np.random.uniform(35, 65)  # Random reasonable value
    
    def _generate_blood_oxygen(self, sleep_efficiency):
        """Generate blood oxygen levels"""
        # Base SpO2 is typically 95-99%
        base_spo2 = 97.0
        
        # Sleep efficiency affects SpO2 stability
        if sleep_efficiency < 0.7:
            # Lower and more variable for poor sleepers
            base_spo2 -= np.random.uniform(1, 3)
            variability = 1.5
        else:
            variability = 0.8
        
        # Add noise using base generator method
        spo2 = base_spo2 + self.generate_time_based_noise(0, variability/3)
        
        # Ensure valid range
        spo2 = max(90, min(spo2, 100))
        
        return round(spo2, 1)