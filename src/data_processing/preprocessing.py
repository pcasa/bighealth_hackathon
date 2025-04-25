import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataPreprocessor:
    def __init__(self):
        """Initialize the data preprocessor"""
        self.processed_data = None
    
    def preprocess_sleep_data(self, sleep_data, wearable_data=None, external_data=None):
        """Preprocess and merge sleep data with wearable and external data"""
        # Convert string dates to datetime for easier processing
        sleep_data['date'] = pd.to_datetime(sleep_data['date'])
        sleep_data['bedtime'] = pd.to_datetime(sleep_data['bedtime'])
        sleep_data['sleep_onset_time'] = pd.to_datetime(sleep_data['sleep_onset_time'])
        sleep_data['wake_time'] = pd.to_datetime(sleep_data['wake_time'])
        
        # Create processed dataframe for just sleep data initially
        processed_data = sleep_data.copy()
        
        # Add calculated sleep features
        processed_data = self._add_sleep_features(processed_data)
        
        # Merge with wearable data if provided
        if wearable_data is not None:
            wearable_data['date'] = pd.to_datetime(wearable_data['date'])
            wearable_data['device_bedtime'] = pd.to_datetime(wearable_data['device_bedtime'])
            wearable_data['device_sleep_onset'] = pd.to_datetime(wearable_data['device_sleep_onset'])
            wearable_data['device_wake_time'] = pd.to_datetime(wearable_data['device_wake_time'])
            
            # Merge on user_id and date
            processed_data = pd.merge(
                processed_data, 
                wearable_data.drop(['heart_rate_data', 'movement_data', 'sleep_stage_data'], axis=1), 
                on=['user_id', 'date'], 
                how='left'
            )
            
            # Add calculated wearable features
            processed_data = self._add_wearable_features(processed_data)
        
        # Merge with external data if provided
        if external_data is not None:
            if 'date' in external_data.columns:
                external_data['date'] = pd.to_datetime(external_data['date'])
                processed_data = pd.merge(
                    processed_data,
                    external_data,
                    on='date',
                    how='left'
                )
        
        # Handle missing values
        processed_data = self._handle_missing_values(processed_data)
        
        self.processed_data = processed_data
        return processed_data
    
    def _add_sleep_features(self, data):
        """Add calculated features from sleep data"""
        # Extract day of week
        data['day_of_week'] = data['date'].dt.dayofweek
        data['is_weekend'] = data['day_of_week'] >= 5
        
        # Calculate time to bed (how long before sleep onset)
        data['time_to_sleep_minutes'] = (
            (data['sleep_onset_time'] - data['bedtime']).dt.total_seconds() / 60
        )
        
        # Calculate consistency metrics
        data['user_sleep_time'] = data['sleep_onset_time'].dt.time
        data['user_wake_time'] = data['wake_time'].dt.time
        
        # Group by user to calculate consistency
        user_groups = data.groupby('user_id')
        
        # Initialize consistency columns
        data['bedtime_consistency'] = np.nan
        data['waketime_consistency'] = np.nan
        data['duration_consistency'] = np.nan
        
        for user_id, group in user_groups:
            # Calculate standard deviation of sleep times
            bedtime_std = group['bedtime'].dt.hour * 60 + group['bedtime'].dt.minute
            bedtime_std = min(bedtime_std.std(), 180) / 180  # Normalize to 0-1
            
            waketime_std = group['wake_time'].dt.hour * 60 + group['wake_time'].dt.minute
            waketime_std = min(waketime_std.std(), 180) / 180  # Normalize to 0-1
            
            duration_std = min(group['sleep_duration_hours'].std(), 3) / 3  # Normalize to 0-1
            
            # Convert to consistency (1 - normalized_std)
            data.loc[data['user_id'] == user_id, 'bedtime_consistency'] = 1 - bedtime_std
            data.loc[data['user_id'] == user_id, 'waketime_consistency'] = 1 - waketime_std
            data.loc[data['user_id'] == user_id, 'duration_consistency'] = 1 - duration_std
        
        return data
    
    def _add_wearable_features(self, data):
        """Add calculated features from wearable data"""
        # Calculate discrepancies between reported and measured sleep
        data['bedtime_discrepancy_minutes'] = np.nan
        data['sleep_onset_discrepancy_minutes'] = np.nan
        data['wake_time_discrepancy_minutes'] = np.nan
        data['duration_discrepancy_hours'] = np.nan
        
        # Only calculate for rows with both user and device data
        has_both = (~data['device_bedtime'].isna()) & (~data['bedtime'].isna())
        
        if has_both.any():
            data.loc[has_both, 'bedtime_discrepancy_minutes'] = (
                (data.loc[has_both, 'device_bedtime'] - data.loc[has_both, 'bedtime'])
                .dt.total_seconds() / 60
            )
            
            data.loc[has_both, 'sleep_onset_discrepancy_minutes'] = (
                (data.loc[has_both, 'device_sleep_onset'] - data.loc[has_both, 'sleep_onset_time'])
                .dt.total_seconds() / 60
            )
            
            data.loc[has_both, 'wake_time_discrepancy_minutes'] = (
                (data.loc[has_both, 'device_wake_time'] - data.loc[has_both, 'wake_time'])
                .dt.total_seconds() / 60
            )
            
            data.loc[has_both, 'duration_discrepancy_hours'] = (
                data.loc[has_both, 'device_sleep_duration'] - data.loc[has_both, 'sleep_duration_hours']
            )
        
        return data
    
    def _handle_missing_values(self, data):
        """Handle missing values in the processed data"""
        # For wearable data columns, fill with user averages where possible
        wearable_columns = [
            'device_sleep_duration', 'deep_sleep_percentage', 'light_sleep_percentage',
            'rem_sleep_percentage', 'awake_percentage', 'average_heart_rate',
            'heart_rate_variability'
        ]
        
        for col in wearable_columns:
            if col in data.columns and data[col].isna().any():
                # Fill with user average if available
                user_means = data.groupby('user_id')[col].transform('mean')
                data[col] = data[col].fillna(user_means)
                
                # For remaining NaNs, fill with global average
                if data[col].isna().any():
                    data[col] = data[col].fillna(data[col].mean())
        
        # For calculated metrics with NaN, use reasonable defaults
        consistency_columns = ['bedtime_consistency', 'waketime_consistency', 'duration_consistency']
        for col in consistency_columns:
            if col in data.columns:
                data[col] = data[col].fillna(0.5)  # Moderate consistency as default
        
        # Return the cleaned data
        return data