import numpy as np
import pandas as pd
import yaml
import os
from datetime import datetime, timedelta

class WearableDataGenerator:
    def __init__(self, config_path='config/data_generation_config.yaml', device_config_path='config/device_profiles.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        with open(device_config_path, 'r') as file:
            self.device_config = yaml.safe_load(file)
        
        self.device_params = self.config['device_parameters']
    
    def generate_wearable_data(self, sleep_data_df, users_df):
        """Generate wearable device data based on sleep data"""
        all_wearable_data = []
        
        # Join sleep data with user data to get device type
        merged_data = pd.merge(sleep_data_df, users_df[['user_id', 'device_type']], on='user_id')
        
        for _, record in merged_data.iterrows():
            wearable_data = self._generate_record_wearable_data(record)
            all_wearable_data.append(wearable_data)
        
        return pd.DataFrame(all_wearable_data)
    
    def _generate_record_wearable_data(self, record):
        """Generate wearable data for a single sleep record"""
        device_type = record['device_type']
        device_params = self.device_params[device_type]
        device_profile = self.device_config[device_type]
        
        # Convert string times to datetime objects
        bedtime = datetime.strptime(record['bedtime'], '%Y-%m-%d %H:%M:%S')
        sleep_onset = datetime.strptime(record['sleep_onset_time'], '%Y-%m-%d %H:%M:%S')
        wake_time = datetime.strptime(record['wake_time'], '%Y-%m-%d %H:%M:%S')
        
        # Add device measurement variance
        bedtime_variance = int(np.random.normal(0, 5))  # 5 minute standard deviation
        sleep_onset_variance = int(np.random.normal(0, 10))  # 10 minute standard deviation
        wake_time_variance = int(np.random.normal(0, 5))  # 5 minute standard deviation
        
        device_bedtime = bedtime + timedelta(minutes=bedtime_variance)
        device_sleep_onset = sleep_onset + timedelta(minutes=sleep_onset_variance)
        device_wake_time = wake_time + timedelta(minutes=wake_time_variance)
        
        # Generate sleep stages data
        sleep_stages = self._generate_sleep_stages(
            device_type, device_sleep_onset, device_wake_time, record['sleep_duration_hours']
        )
        
        # Generate heart rate data
        heart_rate_data = self._generate_heart_rate_data(
            device_type, device_profile, device_bedtime, device_wake_time, 
            record['sleep_efficiency'], record['awakenings_count']
        )
        
        # Generate movement data
        movement_data = self._generate_movement_data(
            device_type, device_profile, device_sleep_onset, device_wake_time,
            record['awakenings_count'], record['total_awake_minutes']
        )
        
        # Calculate device-specific metrics
        hrv = self._calculate_hrv(device_type, heart_rate_data, record['sleep_efficiency'])
        
        # Blood oxygen if device supports it
        blood_oxygen = None
        if 'blood_oxygen' in device_profile['data_fields']:
            blood_oxygen = self._generate_blood_oxygen(record['sleep_efficiency'])
        
        return {
            'user_id': record['user_id'],
            'date': record['date'],
            'device_type': device_type,
            'device_bedtime': device_bedtime.strftime('%Y-%m-%d %H:%M:%S'),
            'device_sleep_onset': device_sleep_onset.strftime('%Y-%m-%d %H:%M:%S'),
            'device_wake_time': device_wake_time.strftime('%Y-%m-%d %H:%M:%S'),
            'device_sleep_duration': (device_wake_time - device_sleep_onset).total_seconds() / 3600,
            'deep_sleep_percentage': sleep_stages['deep'],
            'light_sleep_percentage': sleep_stages['light'],
            'rem_sleep_percentage': sleep_stages['rem'],
            'awake_percentage': sleep_stages['awake'],
            'sleep_cycles': sleep_stages['cycles'],
            'average_heart_rate': np.mean(heart_rate_data),
            'min_heart_rate': np.min(heart_rate_data),
            'max_heart_rate': np.max(heart_rate_data),
            'heart_rate_variability': hrv,
            'movement_intensity': np.mean(movement_data),
            'blood_oxygen': blood_oxygen,
            'heart_rate_data': heart_rate_data,
            'movement_data': movement_data,
            'sleep_stage_data': sleep_stages['sequence']
        }
    
    def _generate_sleep_stages(self, device_type, sleep_onset, wake_time, sleep_duration):
        """Generate sleep stage percentages and sequence"""
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
        
        # Add random variance
        deep = base_deep + deep_adj + np.random.normal(0, 0.03)
        rem = base_rem + rem_adj + np.random.normal(0, 0.03)
        awake = base_awake + awake_adj + np.random.normal(0, 0.01)
        
        # Ensure light makes up the remainder
        light = 1.0 - deep - rem - awake
        
        # Adjust for valid ranges
        deep = max(0.05, min(deep, 0.30))
        rem = max(0.10, min(rem, 0.35))
        awake = max(0.01, min(awake, 0.15))
        light = max(0.30, min(light, 0.70))
        
        # Normalize to ensure they sum to 1
        total = deep + light + rem + awake
        deep /= total
        light /= total
        rem /= total
        awake /= total
        
        # Calculate approximate number of sleep cycles (90-110 minutes each)
        cycle_length_hours = np.random.uniform(1.5, 1.8)  # 90-108 minutes
        cycles = sleep_duration / cycle_length_hours
        
        # Generate a simplified sleep stage sequence
        total_minutes = int((wake_time - sleep_onset).total_seconds() / 60)
        sequence = self._generate_stage_sequence(
            total_minutes, deep, light, rem, awake, cycles
        )
        
        return {
            'deep': deep,
            'light': light,
            'rem': rem,
            'awake': awake,
            'cycles': np.floor(cycles),
            'sequence': sequence
        }
    
    def _generate_stage_sequence(self, total_minutes, deep_pct, light_pct, rem_pct, awake_pct, cycles):
        """Generate a minute-by-minute sequence of sleep stages"""
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
        
        for cycle in range(int(np.ceil(cycles))):
            cycle_fraction = cycle / int(np.ceil(cycles))
            
            # Adjust deep sleep percentage based on cycle number
            if cycle == 0:
                cycle_deep_pct = first_cycle_deep
                cycle_rem_pct = first_cycle_rem
            elif cycle >= int(cycles) - 1:
                cycle_deep_pct = last_cycle_deep
                cycle_rem_pct = last_cycle_rem
            else:
                # Gradual transition
                cycle_deep_pct = first_cycle_deep - cycle_fraction * (first_cycle_deep - last_cycle_deep)
                cycle_rem_pct = first_cycle_rem + cycle_fraction * (last_cycle_rem - first_cycle_rem)
            
            # Adjust light to maintain proportions
            cycle_light_pct = 1.0 - cycle_deep_pct - cycle_rem_pct - awake_pct
            
            # Determine minutes for each stage in this cycle
            if cycle == int(np.ceil(cycles)) - 1:
                # Last cycle might be shorter
                remaining_minutes = total_minutes - len(stages)
                cycle_minutes = remaining_minutes
            
            # Simplified cycle pattern
            light_start = int(cycle_minutes * 0.1)
            deep_minutes = int(cycle_minutes * cycle_deep_pct)
            light_minutes = int(cycle_minutes * cycle_light_pct) - light_start
            rem_minutes = int(cycle_minutes * cycle_rem_pct)
            awake_minutes = int(cycle_minutes * awake_pct)
            
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
        
        # Ensure we have exactly the right number of minutes
        if len(stages) < total_minutes:
            # Add light sleep to make up difference
            stages.extend(['light'] * (total_minutes - len(stages)))
        elif len(stages) > total_minutes:
            # Trim excess
            stages = stages[:total_minutes]
        
        return stages
    
    def _generate_heart_rate_data(self, device_type, device_profile, bedtime, wake_time, sleep_efficiency, awakenings):
        """Generate heart rate data for the sleep period"""
        # Determine sampling rate from device
        sampling_minutes = device_profile['heart_rate']['sampling_rate_minutes']
        
        # Calculate number of samples
        total_minutes = int((wake_time - bedtime).total_seconds() / 60)
        num_samples = total_minutes // sampling_minutes + 1
        
        # Base heart rate pattern: higher at start, lower during deep sleep, increases during REM
        base_hr = np.random.randint(58, 68)  # Resting heart rate
        
        # Generate heart rate time series
        heart_rates = []
        
        # Initial higher heart rate during sleep onset
        initial_hr = base_hr + np.random.randint(5, 15)
        
        # Sleep efficiency affects overall heart rate variability
        variability = 2.0 + (1.0 - sleep_efficiency) * 10
        
        for i in range(num_samples):
            time_fraction = i / num_samples
            
            # Heart rate typically decreases in first third, then has periods of increase during REM
            if time_fraction < 0.2:
                # Initial settling down period
                hr = initial_hr - time_fraction * 5 * initial_hr / base_hr
            elif time_fraction < 0.4:
                # Deepest sleep often in first half
                hr = base_hr - 5
            else:
                # More REM in later cycles, heart rate increases
                hr = base_hr + np.sin(time_fraction * 15) * 4
            
            # Add noise
            hr += np.random.normal(0, variability)
            
            # Add spikes for awakenings
            if awakenings > 0 and np.random.random() < (awakenings / num_samples) * 3:
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
            
            # Add noise based on device sensitivity
            movement += np.random.normal(0, noise_factor)
            
            # Apply device sensitivity
            movement *= sensitivity
            
            # Ensure valid range
            movement = max(0, min(movement, 1.0))
            
            movement_data.append(round(movement, 3))
        
        return movement_data
    
    def _calculate_hrv(self, device_type, heart_rate_data, sleep_efficiency):
        """Calculate heart rate variability from heart rate data"""
        # Simple RMSSD calculation (simplified)
        diffs = np.diff(heart_rate_data)
        rmssd = np.sqrt(np.mean(np.square(diffs)))
        
        # Adjust based on sleep efficiency - generally higher HRV is better
        base_hrv = rmssd * (0.8 + sleep_efficiency * 0.4)
        
        # Add device-specific variance
        hrv = base_hrv * (0.9 + np.random.random() * 0.2)
        
        return round(hrv, 1)
    
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
        
        # Add noise
        spo2 = base_spo2 + np.random.normal(0, variability)
        
        # Ensure valid range
        spo2 = max(90, min(spo2, 100))
        
        return round(spo2, 1)