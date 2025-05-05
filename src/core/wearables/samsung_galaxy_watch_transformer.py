"""
Samsung Galaxy Watch data transformer that converts Samsung Health data to standardized sleep format.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import csv
import os
from typing import Dict, List, Optional, Union

from src.core.wearables.wearable_base_transformer import BaseWearableTransformer

logger = logging.getLogger(__name__)

class SamsungWatchTransformer(BaseWearableTransformer):
    """Transform Samsung Galaxy Watch/Samsung Health sleep data to standardized format"""
    
    def __init__(self):
        super().__init__()
        self.device_type = "samsung_watch"
    
    def _transform_data(self, data_df: pd.DataFrame, users_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Transform Samsung Health data to standardized sleep format
        
        Args:
            data_df: DataFrame with Samsung Health data (or paths to CSV/JSON files)
            users_df: Optional DataFrame with user information
            
        Returns:
            DataFrame with standardized sleep data
        """
        # Check if this is raw Samsung Health export or pre-processed data
        if 'csv_data' in data_df.columns or 'file_path' in data_df.columns:
            logger.info("Processing raw Samsung Health export")
            return self._process_samsung_export(data_df, users_df)
        
        # For pre-processed Samsung data
        transformed_data = []
        
        for _, row in data_df.iterrows():
            try:
                # Create a new record with standardized fields
                record = {
                    'user_id': row.get('user_id'),
                    'date': pd.to_datetime(row.get('date', row.get('sleep_start'))).strftime('%Y-%m-%d'),
                    'device_type': self.device_type
                }
                
                # Handle time fields
                if 'sleep_start' in row:
                    record['device_bedtime'] = pd.to_datetime(row['sleep_start']).strftime('%Y-%m-%d %H:%M:%S')
                elif 'bedtime' in row:
                    record['device_bedtime'] = pd.to_datetime(row['bedtime']).strftime('%Y-%m-%d %H:%M:%S')
                
                if 'sleep_end' in row:
                    record['device_wake_time'] = pd.to_datetime(row['sleep_end']).strftime('%Y-%m-%d %H:%M:%S')
                elif 'wake_time' in row:
                    record['device_wake_time'] = pd.to_datetime(row['wake_time']).strftime('%Y-%m-%d %H:%M:%S')
                
                # Calculate sleep onset time (Samsung typically records actual sleep onset directly)
                if 'sleep_onset' in row:
                    record['device_sleep_onset'] = pd.to_datetime(row['sleep_onset']).strftime('%Y-%m-%d %H:%M:%S')
                elif 'device_bedtime' in record:
                    # Estimate sleep onset as 10 minutes after bedtime for Samsung
                    bedtime = pd.to_datetime(record['device_bedtime'])
                    record['device_sleep_onset'] = (bedtime + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
                
                # Handle duration
                if 'total_sleep_time' in row:
                    # Samsung typically reports in seconds
                    if isinstance(row['total_sleep_time'], (int, float)) and row['total_sleep_time'] > 1000:
                        record['device_sleep_duration'] = row['total_sleep_time'] / 3600  # Convert seconds to hours
                    else:
                        record['device_sleep_duration'] = row['total_sleep_time']  # Already in hours
                elif 'sleep_duration' in row:
                    record['device_sleep_duration'] = row['sleep_duration']
                elif 'device_bedtime' in record and 'device_wake_time' in record:
                    # Calculate from timestamps
                    bedtime = pd.to_datetime(record['device_bedtime'])
                    wake_time = pd.to_datetime(record['device_wake_time'])
                    duration_hours = (wake_time - bedtime).total_seconds() / 3600
                    record['device_sleep_duration'] = duration_hours
                
                # Handle sleep stages
                sleep_stages = {}
                
                for stage in ['deep_sleep', 'light_sleep', 'rem_sleep', 'awake_time']:
                    if stage in row:
                        # Samsung sometimes gives percentages, sometimes minutes, sometimes seconds
                        value = row[stage]
                        
                        # If very large, assume seconds and convert to hours
                        if isinstance(value, (int, float)) and value > 1000:
                            sleep_stages[stage] = value / 3600
                        # If moderate, assume minutes and convert to hours
                        elif isinstance(value, (int, float)) and value > 60:
                            sleep_stages[stage] = value / 60
                        # If small, assume hours or decimal percentage
                        elif isinstance(value, (int, float)) and value < 1:
                            # If it's a percentage, convert to hours
                            if 'device_sleep_duration' in record and value < 1:
                                sleep_stages[stage] = record['device_sleep_duration'] * value
                            else:
                                sleep_stages[stage] = value  # Already in hours
                        else:
                            sleep_stages[stage] = value  # Assume hours
                
                # Update the record with sleep stages
                record.update(sleep_stages)
                
                # Calculate sleep stage percentages
                if any(k in record for k in ['deep_sleep', 'light_sleep', 'rem_sleep', 'awake_time']):
                    stage_percentages = self._calculate_sleep_stages(record)
                    record.update(stage_percentages)
                
                # Heart rate metrics
                if 'avg_hr' in row:
                    record['average_heart_rate'] = row['avg_hr']
                elif 'average_heart_rate' in row:
                    record['average_heart_rate'] = row['average_heart_rate']
                    
                if 'min_hr' in row:
                    record['min_heart_rate'] = row['min_hr']
                elif 'min_heart_rate' in row:
                    record['min_heart_rate'] = row['min_heart_rate']
                    
                if 'max_hr' in row:
                    record['max_heart_rate'] = row['max_hr']
                elif 'max_heart_rate' in row:
                    record['max_heart_rate'] = row['max_heart_rate']
                
                # Heart rate variability
                if 'hrv' in row:
                    record['heart_rate_variability'] = row['hrv']
                elif 'heart_rate_variability' in row:
                    record['heart_rate_variability'] = row['heart_rate_variability']
                
                # Blood oxygen
                if 'spo2_avg' in row:
                    record['blood_oxygen'] = row['spo2_avg']
                elif 'blood_oxygen' in row:
                    record['blood_oxygen'] = row['blood_oxygen']
                
                # Calculate time in bed (typically the total time from bedtime to wake time)
                if 'device_bedtime' in record and 'device_wake_time' in record:
                    bedtime = pd.to_datetime(record['device_bedtime'])
                    wake_time = pd.to_datetime(record['device_wake_time'])
                    record['time_in_bed_hours'] = (wake_time - bedtime).total_seconds() / 3600
                
                # Calculate sleep efficiency
                if 'sleep_efficiency' not in record:
                    if 'time_in_bed_hours' in record and 'device_sleep_duration' in record and record['time_in_bed_hours'] > 0:
                        record['sleep_efficiency'] = record['device_sleep_duration'] / record['time_in_bed_hours']
                    else:
                        record['sleep_efficiency'] = self._estimate_sleep_efficiency(record)
                
                # Calculate awakenings count
                if 'awakenings_count' not in record:
                    if 'awakenings' in row:
                        record['awakenings_count'] = row['awakenings']
                    elif 'awake_count' in row:
                        record['awakenings_count'] = row['awake_count']
                    else:
                        # Estimate from awake time
                        if 'awake_time' in sleep_stages:
                            record['awakenings_count'] = max(1, int(sleep_stages['awake_time'] * 60 / 7))  # Approx 7 min per awakening
                        else:
                            record['awakenings_count'] = 2  # Default value
                
                # Convert any percentages to proper decimals
                for key, value in record.items():
                    if key.endswith('_percentage') and isinstance(value, (int, float)) and value > 1:
                        record[key] = value / 100
                
                transformed_data.append(record)
                
            except Exception as e:
                logger.error(f"Error processing Samsung record: {str(e)}")
                continue
        
        return pd.DataFrame(transformed_data)
    
    def _process_samsung_export(self, data_df: pd.DataFrame, users_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Process raw Samsung Health export files
        
        Args:
            data_df: DataFrame containing paths to CSV/JSON files
            users_df: Optional DataFrame with user information
            
        Returns:
            DataFrame with standardized sleep data
        """
        sleep_records = []
        user_id = users_df['user_id'].iloc[0] if users_df is not None and len(users_df) > 0 else "unknown"
        
        for _, row in data_df.iterrows():
            try:
                # Extract file path or data
                if 'file_path' in row:
                    file_path = row['file_path']
                    
                    # Check if it's a CSV or JSON file
                    if file_path.endswith('.csv'):
                        # Read Samsung Health CSV export
                        try:
                            sleep_data = pd.read_csv(file_path)
                            
                            # Check if this is a sleep data CSV
                            if 'sleep_id' in sleep_data.columns or 'start_time' in sleep_data.columns:
                                # Process each sleep record
                                for _, sleep_entry in sleep_data.iterrows():
                                    sleep_record = self._parse_samsung_sleep_entry(sleep_entry, user_id)
                                    if sleep_record:
                                        sleep_records.append(sleep_record)
                        except Exception as e:
                            logger.error(f"Error reading Samsung Health CSV file: {str(e)}")
                            continue
                    
                    elif file_path.endswith('.json'):
                        # Read Samsung Health JSON export
                        try:
                            with open(file_path, 'r') as f:
                                sleep_data = json.load(f)
                            
                            # Process JSON data
                            if isinstance(sleep_data, list):
                                for sleep_entry in sleep_data:
                                    sleep_record = self._parse_samsung_sleep_json(sleep_entry, user_id)
                                    if sleep_record:
                                        sleep_records.append(sleep_record)
                            elif isinstance(sleep_data, dict):
                                sleep_record = self._parse_samsung_sleep_json(sleep_data, user_id)
                                if sleep_record:
                                    sleep_records.append(sleep_record)
                        except Exception as e:
                            logger.error(f"Error reading Samsung Health JSON file: {str(e)}")
                            continue
                elif 'csv_data' in row:
                    # Process embedded CSV data
                    try:
                        if isinstance(row['csv_data'], str):
                            # Convert string CSV to DataFrame
                            import io
                            sleep_data = pd.read_csv(io.StringIO(row['csv_data']))
                        else:
                            sleep_data = row['csv_data']
                        
                        # Process each sleep record
                        for _, sleep_entry in sleep_data.iterrows():
                            sleep_record = self._parse_samsung_sleep_entry(sleep_entry, user_id)
                            if sleep_record:
                                sleep_records.append(sleep_record)
                    except Exception as e:
                        logger.error(f"Error processing Samsung Health CSV data: {str(e)}")
                        continue
                else:
                    logger.warning("No file path or data found in row")
                    continue
            
            except Exception as e:
                logger.error(f"Error processing Samsung Health export: {str(e)}")
                continue
        
        # Convert to DataFrame
        sleep_df = pd.DataFrame(sleep_records)
        
        if sleep_df.empty:
            logger.warning("No sleep records found in Samsung Health data")
            return sleep_df
        
        return sleep_df
    
    def _parse_samsung_sleep_entry(self, entry, user_id):
        """Parse a Samsung Health CSV entry into a standardized sleep record"""
        try:
            # Extract date and times
            if 'start_time' in entry:
                start_time = pd.to_datetime(entry['start_time'])
            elif 'start_date' in entry:
                start_time = pd.to_datetime(entry['start_date'])
            else:
                return None
                
            if 'end_time' in entry:
                end_time = pd.to_datetime(entry['end_time'])
            elif 'end_date' in entry:
                end_time = pd.to_datetime(entry['end_date'])
            else:
                return None
            
            # Create base record
            record = {
                'user_id': user_id,
                'date': start_time.strftime('%Y-%m-%d'),
                'device_type': self.device_type,
                'device_bedtime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'device_wake_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add sleep duration
            if 'duration' in entry:
                # In Samsung Health, duration is typically in milliseconds
                duration_value = entry['duration']
                if isinstance(duration_value, (int, float)) and duration_value > 1000:
                    # Convert milliseconds to hours
                    record['device_sleep_duration'] = duration_value / (1000 * 60 * 60)
                else:
                    record['device_sleep_duration'] = duration_value
            
            # Add sleep stages if available
            for stage in ['deep', 'light', 'rem', 'awake']:
                stage_key = f'{stage}_sleep_time' if stage != 'awake' else 'awake_time'
                if stage_key in entry:
                    # Convert to hours - Samsung typically uses milliseconds
                    value = entry[stage_key]
                    record_key = f'{stage}_sleep' if stage != 'awake' else 'awake_time'
                    
                    if isinstance(value, (int, float)) and value > 1000:
                        record[record_key] = value / (1000 * 60 * 60)  # Convert to hours
                    else:
                        record[record_key] = value
            
            # Add movement count as proxy for awakenings if available
            if 'movement_count' in entry:
                record['awakenings_count'] = entry['movement_count']
            
            # Calculate sleep stage percentages
            if any(k in record for k in ['deep_sleep', 'light_sleep', 'rem_sleep', 'awake_time']):
                stage_percentages = self._calculate_sleep_stages(record)
                record.update(stage_percentages)
            
            # Calculate time in bed
            if 'device_bedtime' in record and 'device_wake_time' in record:
                bedtime = pd.to_datetime(record['device_bedtime'])
                wake_time = pd.to_datetime(record['device_wake_time'])
                record['time_in_bed_hours'] = (wake_time - bedtime).total_seconds() / 3600
            
            # Calculate sleep efficiency
            if 'time_in_bed_hours' in record and 'device_sleep_duration' in record and record['time_in_bed_hours'] > 0:
                record['sleep_efficiency'] = record['device_sleep_duration'] / record['time_in_bed_hours']
            else:
                record['sleep_efficiency'] = self._estimate_sleep_efficiency(record)
            
            return record
            
        except Exception as e:
            logger.error(f"Error parsing Samsung Health sleep entry: {str(e)}")
            return None
    
    def _parse_samsung_sleep_json(self, entry, user_id):
        """Parse a Samsung Health JSON entry into a standardized sleep record"""
        try:
            # Extract date and times - Samsung JSON can use different formats
            start_time = None
            end_time = None
            
            # Try different possible field names
            for start_field in ['start_time', 'startTime', 'bed_time', 'bedTime', 'start']:
                if start_field in entry:
                    start_time = pd.to_datetime(entry[start_field])
                    break
            
            for end_field in ['end_time', 'endTime', 'wake_time', 'wakeTime', 'end']:
                if end_field in entry:
                    end_time = pd.to_datetime(entry[end_field])
                    break
            
            if not start_time or not end_time:
                return None
            
            # Create base record
            record = {
                'user_id': user_id,
                'date': start_time.strftime('%Y-%m-%d'),
                'device_type': self.device_type,
                'device_bedtime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'device_wake_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Extract sleep duration - try different field names
            for duration_field in ['duration', 'sleep_duration', 'sleepDuration', 'total_sleep_time']:
                if duration_field in entry:
                    duration_value = entry[duration_field]
                    
                    # Convert to hours based on magnitude
                    if isinstance(duration_value, (int, float)):
                        if duration_value > 1000000:  # If in milliseconds
                            record['device_sleep_duration'] = duration_value / (1000 * 60 * 60)
                        elif duration_value > 1000:  # If in seconds
                            record['device_sleep_duration'] = duration_value / 3600
                        elif duration_value > 100:  # If in minutes
                            record['device_sleep_duration'] = duration_value / 60
                        else:  # If already in hours
                            record['device_sleep_duration'] = duration_value
                    break
            
            # Extract sleep stages - search for multiple possible formats
            # Samsung uses various formats in the JSON export
            
            # Format 1: Direct stage durations
            for stage in ['deep', 'light', 'rem', 'awake']:
                for stage_key in [f'{stage}_sleep', f'{stage}Sleep', f'{stage}_sleep_time', f'{stage}SleepTime', 
                                 f'{stage}_time', f'{stage}Time', f'{stage}_duration', f'{stage}Duration']:
                    if stage_key in entry:
                        value = entry[stage_key]
                        record_key = f'{stage}_sleep' if stage != 'awake' else 'awake_time'
                        
                        # Convert to hours based on magnitude
                        if isinstance(value, (int, float)):
                            if value > 1000000:  # If in milliseconds
                                record[record_key] = value / (1000 * 60 * 60)
                            elif value > 1000:  # If in seconds
                                record[record_key] = value / 3600
                            elif value > 100:  # If in minutes
                                record[record_key] = value / 60
                            elif value < 1 and 'device_sleep_duration' in record:
                                # If it's a percentage (0-1), convert to hours
                                record[record_key] = record['device_sleep_duration'] * value
                            else:  # If already in hours
                                record[record_key] = value
                        break
            
            # Format 2: Stages in a nested structure
            if 'stages' in entry:
                stages = entry['stages']
                for stage in ['deep', 'light', 'rem', 'awake']:
                    if stage in stages:
                        value = stages[stage]
                        record_key = f'{stage}_sleep' if stage != 'awake' else 'awake_time'
                        
                        # Convert to hours based on magnitude
                        if isinstance(value, (int, float)):
                            if value > 1000000:  # If in milliseconds
                                record[record_key] = value / (1000 * 60 * 60)
                            elif value > 1000:  # If in seconds
                                record[record_key] = value / 3600
                            elif value > 100:  # If in minutes
                                record[record_key] = value / 60
                            elif value < 1 and 'device_sleep_duration' in record:
                                # If it's a percentage (0-1), convert to hours
                                record[record_key] = record['device_sleep_duration'] * value
                            else:  # If already in hours
                                record[record_key] = value
            
            # Calculate sleep stage percentages
            if any(k in record for k in ['deep_sleep', 'light_sleep', 'rem_sleep', 'awake_time']):
                stage_percentages = self._calculate_sleep_stages(record)
                record.update(stage_percentages)
            
            # Add heart rate data if available
            for hr_field in ['avg_heart_rate', 'average_heart_rate', 'avgHeartRate', 'average_hr', 'avg_hr']:
                if hr_field in entry:
                    record['average_heart_rate'] = entry[hr_field]
                    break
                    
            for hr_field in ['min_heart_rate', 'minimum_heart_rate', 'minHeartRate', 'min_hr']:
                if hr_field in entry:
                    record['min_heart_rate'] = entry[hr_field]
                    break
                    
            for hr_field in ['max_heart_rate', 'maximum_heart_rate', 'maxHeartRate', 'max_hr']:
                if hr_field in entry:
                    record['max_heart_rate'] = entry[hr_field]
                    break
            
            # Add blood oxygen data if available
            for spo2_field in ['blood_oxygen', 'bloodOxygen', 'spo2', 'sp02_avg', 'oxygen_saturation']:
                if spo2_field in entry:
                    record['blood_oxygen'] = entry[spo2_field]
                    break
            
            # Add HRV data if available
            for hrv_field in ['hrv', 'heart_rate_variability', 'heartRateVariability', 'hrv_avg']:
                if hrv_field in entry:
                    record['heart_rate_variability'] = entry[hrv_field]
                    break
            
            # Add awakenings count if available
            for awaken_field in ['awakenings', 'awakeCount', 'awakenings_count', 'wake_count']:
                if awaken_field in entry:
                    record['awakenings_count'] = entry[awaken_field]
                    break
            
            # If no awakenings count yet, estimate from awake time
            if 'awakenings_count' not in record and 'awake_time' in record:
                # Approx 7 min per awakening
                record['awakenings_count'] = max(1, int(record['awake_time'] * 60 / 7))
            
            # Calculate sleep efficiency
            if 'sleep_efficiency' not in record:
                if 'time_in_bed_hours' in record and 'device_sleep_duration' in record and record['time_in_bed_hours'] > 0:
                    record['sleep_efficiency'] = record['device_sleep_duration'] / record['time_in_bed_hours']
                else:
                    record['sleep_efficiency'] = self._estimate_sleep_efficiency(record)
            
            return record
            
        except Exception as e:
            logger.error(f"Error parsing Samsung Health JSON entry: {str(e)}")
            return None