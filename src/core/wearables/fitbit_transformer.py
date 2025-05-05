"""
Fitbit data transformer that converts Fitbit JSON/CSV exports to standardized sleep format.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Union

from src.core.wearables.wearable_base_transformer import BaseWearableTransformer

logger = logging.getLogger(__name__)

class FitbitTransformer(BaseWearableTransformer):
    """Transform Fitbit sleep data to standardized format"""
    
    def __init__(self):
        super().__init__()
        self.device_type = "fitbit"
    
    def _transform_data(self, data_df: pd.DataFrame, users_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Transform Fitbit data to standardized sleep format
        
        Args:
            data_df: DataFrame with Fitbit data (or paths to JSON files)
            users_df: Optional DataFrame with user information
            
        Returns:
            DataFrame with standardized sleep data
        """
        # Check if this is raw Fitbit JSON export or pre-processed data
        if 'json_data' in data_df.columns or 'file_path' in data_df.columns:
            logger.info("Processing raw Fitbit JSON export")
            return self._process_fitbit_export(data_df, users_df)
        
        # For pre-processed Fitbit data
        transformed_data = []
        
        for _, row in data_df.iterrows():
            try:
                # Create a new record with standardized fields
                record = {
                    'user_id': row.get('user_id'),
                    'date': pd.to_datetime(row.get('date', row.get('start_time'))).strftime('%Y-%m-%d'),
                    'device_type': self.device_type
                }
                
                # Handle time fields
                if 'start_time' in row:
                    record['device_bedtime'] = pd.to_datetime(row['start_time']).strftime('%Y-%m-%d %H:%M:%S')
                elif 'bedtime' in row:
                    record['device_bedtime'] = pd.to_datetime(row['bedtime']).strftime('%Y-%m-%d %H:%M:%S')
                
                if 'end_time' in row:
                    record['device_wake_time'] = pd.to_datetime(row['end_time']).strftime('%Y-%m-%d %H:%M:%S')
                elif 'wake_time' in row:
                    record['device_wake_time'] = pd.to_datetime(row['wake_time']).strftime('%Y-%m-%d %H:%M:%S')
                
                # Calculate sleep onset time (typically 15-30 min after bedtime for Fitbit)
                if 'device_bedtime' in record:
                    bedtime = pd.to_datetime(record['device_bedtime'])
                    # Fitbit often records "trying to sleep" as part of bedtime
                    onset_minutes = 15  # Default 15 min to fall asleep with Fitbit
                    record['device_sleep_onset'] = (bedtime + timedelta(minutes=onset_minutes)).strftime('%Y-%m-%d %H:%M:%S')
                
                # Handle duration
                if 'duration_minutes' in row:
                    record['device_sleep_duration'] = row['duration_minutes'] / 60  # Convert to hours
                elif 'sleep_duration' in row:
                    record['device_sleep_duration'] = row['sleep_duration']
                elif 'minutes_asleep' in row:
                    record['device_sleep_duration'] = row['minutes_asleep'] / 60  # Convert to hours
                elif 'device_bedtime' in record and 'device_wake_time' in record:
                    # Calculate from timestamps
                    bedtime = pd.to_datetime(record['device_bedtime'])
                    wake_time = pd.to_datetime(record['device_wake_time'])
                    duration_hours = (wake_time - bedtime).total_seconds() / 3600
                    record['device_sleep_duration'] = duration_hours
                
                # Handle Fitbit specific fields for time in bed
                if 'minutes_in_bed' in row:
                    record['time_in_bed_hours'] = row['minutes_in_bed'] / 60  # Convert to hours
                
                # Handle sleep stages
                sleep_stages = {}
                
                if 'deep_minutes' in row:
                    sleep_stages['deep_sleep'] = row['deep_minutes'] / 60  # Convert to hours
                elif 'deep_sleep' in row:
                    sleep_stages['deep_sleep'] = row['deep_sleep']
                    
                if 'light_minutes' in row:
                    sleep_stages['light_sleep'] = row['light_minutes'] / 60  # Convert to hours
                elif 'light_sleep' in row:
                    sleep_stages['light_sleep'] = row['light_sleep']
                    
                if 'rem_minutes' in row:
                    sleep_stages['rem_sleep'] = row['rem_minutes'] / 60  # Convert to hours
                elif 'rem_sleep' in row:
                    sleep_stages['rem_sleep'] = row['rem_sleep']
                    
                if 'awake_minutes' in row:
                    sleep_stages['awake_time'] = row['awake_minutes'] / 60  # Convert to hours
                elif 'wake_minutes' in row:
                    sleep_stages['awake_time'] = row['wake_minutes'] / 60
                elif 'awake_time' in row:
                    sleep_stages['awake_time'] = row['awake_time']
                
                # Fitbit specific: handle restless time
                if 'restless_minutes' in row:
                    restless_hours = row['restless_minutes'] / 60
                    # Fitbit counts "restless" as a light form of awakening
                    # Add a portion of restless time to awake time
                    if 'awake_time' in sleep_stages:
                        sleep_stages['awake_time'] += restless_hours * 0.3  # Count 30% of restless as awake
                    else:
                        sleep_stages['awake_time'] = restless_hours * 0.3
                
                # Update the record with sleep stages
                record.update(sleep_stages)
                
                # Calculate sleep stage percentages
                if any(k in record for k in ['deep_sleep', 'light_sleep', 'rem_sleep', 'awake_time']):
                    stage_percentages = self._calculate_sleep_stages(record)
                    record.update(stage_percentages)
                
                # Heart rate metrics
                if 'average_hr' in row:
                    record['average_heart_rate'] = row['average_hr']
                elif 'average_heart_rate' in row:
                    record['average_heart_rate'] = row['average_heart_rate']
                    
                if 'lowest_hr' in row:
                    record['min_heart_rate'] = row['lowest_hr']
                elif 'min_heart_rate' in row:
                    record['min_heart_rate'] = row['min_heart_rate']
                    
                if 'highest_hr' in row:
                    record['max_heart_rate'] = row['highest_hr']
                elif 'max_heart_rate' in row:
                    record['max_heart_rate'] = row['max_heart_rate']
                
                # Heart rate variability
                if 'hr_variability' in row:
                    record['heart_rate_variability'] = row['hr_variability']
                elif 'heart_rate_variability' in row:
                    record['heart_rate_variability'] = row['heart_rate_variability']
                
                # Calculate sleep efficiency if not already set
                if 'sleep_efficiency' not in record:
                    if 'time_in_bed_hours' in record and 'device_sleep_duration' in record and record['time_in_bed_hours'] > 0:
                        record['sleep_efficiency'] = record['device_sleep_duration'] / record['time_in_bed_hours']
                    else:
                        record['sleep_efficiency'] = self._estimate_sleep_efficiency(record)
                
                # Calculate awakenings count if we have it from Fitbit
                if 'awakenings_count' not in record:
                    if 'awakenings' in row:
                        record['awakenings_count'] = row['awakenings']
                    elif 'number_of_awakenings' in row:
                        record['awakenings_count'] = row['number_of_awakenings']
                    else:
                        # Estimate from awake time - approximately one awakening every 10 minutes
                        if 'awake_time' in sleep_stages:
                            record['awakenings_count'] = max(1, int(sleep_stages['awake_time'] * 60 / 10))
                        else:
                            record['awakenings_count'] = 2  # Default value
                
                transformed_data.append(record)
                
            except Exception as e:
                logger.error(f"Error processing Fitbit record: {str(e)}")
                continue
        
        return pd.DataFrame(transformed_data)
    
    def _process_fitbit_export(self, data_df: pd.DataFrame, users_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Process raw Fitbit JSON export
        
        Args:
            data_df: DataFrame containing JSON data or file paths
            users_df: Optional DataFrame with user information
            
        Returns:
            DataFrame with standardized sleep data
        """
        sleep_records = []
        user_id = users_df['user_id'].iloc[0] if users_df is not None and len(users_df) > 0 else "unknown"
        
        for _, row in data_df.iterrows():
            try:
                # Extract JSON data
                if 'json_data' in row:
                    if isinstance(row['json_data'], str):
                        json_data = json.loads(row['json_data'])
                    else:
                        json_data = row['json_data']
                elif 'file_path' in row:
                    # Read JSON from file
                    with open(row['file_path'], 'r') as f:
                        json_data = json.load(f)
                else:
                    logger.warning("No JSON data found in row")
                    continue
                
                # Check if this is a Fitbit sleep export
                if 'sleep' not in json_data:
                    logger.warning("No sleep data found in Fitbit JSON")
                    continue
                
                # Process each sleep record
                for sleep_entry in json_data['sleep']:
                    try:
                        # Extract date
                        date_str = sleep_entry.get('dateOfSleep', sleep_entry.get('startTime', '').split('T')[0])
                        
                        # Extract start and end times
                        start_time = sleep_entry.get('startTime')
                        end_time = sleep_entry.get('endTime')
                        
                        if not start_time or not end_time:
                            continue
                        
                        # Create base record
                        record = {
                            'user_id': user_id,
                            'date': date_str,
                            'device_type': self.device_type,
                            'device_bedtime': start_time.replace('T', ' ').split('+')[0],
                            'device_wake_time': end_time.replace('T', ' ').split('+')[0]
                        }
                        
                        # Add minutes in bed
                        if 'timeInBed' in sleep_entry:
                            record['time_in_bed_hours'] = sleep_entry['timeInBed'] / 60
                        
                        # Add minutes asleep
                        if 'minutesAsleep' in sleep_entry:
                            record['device_sleep_duration'] = sleep_entry['minutesAsleep'] / 60
                        
                        # Calculate sleep efficiency
                        if 'timeInBed' in sleep_entry and 'minutesAsleep' in sleep_entry and sleep_entry['timeInBed'] > 0:
                            record['sleep_efficiency'] = sleep_entry['minutesAsleep'] / sleep_entry['timeInBed']
                        
                        # Add sleep stages if available
                        if 'levels' in sleep_entry and 'summary' in sleep_entry['levels']:
                            summary = sleep_entry['levels']['summary']
                            
                            if 'deep' in summary:
                                record['deep_sleep'] = summary['deep']['minutes'] / 60
                            if 'light' in summary:
                                record['light_sleep'] = summary['light']['minutes'] / 60
                            if 'rem' in summary:
                                record['rem_sleep'] = summary['rem']['minutes'] / 60
                            if 'wake' in summary:
                                record['awake_time'] = summary['wake']['minutes'] / 60
                        
                        # Add awakenings count
                        if 'awakeCount' in sleep_entry:
                            record['awakenings_count'] = sleep_entry['awakeCount']
                        elif 'awakensCount' in sleep_entry:
                            record['awakenings_count'] = sleep_entry['awakensCount']
                        
                        # Calculate sleep stage percentages
                        if any(k in record for k in ['deep_sleep', 'light_sleep', 'rem_sleep', 'awake_time']):
                            stage_percentages = self._calculate_sleep_stages(record)
                            record.update(stage_percentages)
                        
                        # Calculate sleep onset (typically 15 min after bedtime for Fitbit)
                        if 'device_bedtime' in record:
                            bedtime = pd.to_datetime(record['device_bedtime'])
                            record['device_sleep_onset'] = (bedtime + timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
                        
                        sleep_records.append(record)
                    except Exception as e:
                        logger.error(f"Error processing sleep entry: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error processing Fitbit JSON: {str(e)}")
                continue
        
        # Convert to DataFrame
        sleep_df = pd.DataFrame(sleep_records)
        
        if sleep_df.empty:
            logger.warning("No sleep records found in Fitbit data")
            return sleep_df
        
        return sleep_df
                    #