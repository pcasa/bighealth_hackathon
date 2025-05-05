"""
Google Pixel Watch data transformer that converts Google Fit sleep data to standardized sleep format.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import os
from typing import Dict, List, Optional, Union

from src.core.wearables.wearable_base_transformer import BaseWearableTransformer



logger = logging.getLogger(__name__)

class GoogleWatchTransformer(BaseWearableTransformer):
    """Transform Google Pixel Watch/Google Fit sleep data to standardized format"""
    
    def __init__(self):
        super().__init__()
        self.device_type = "google_watch"
    
    def _transform_data(self, data_df: pd.DataFrame, users_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Transform Google Fit data to standardized sleep format
        
        Args:
            data_df: DataFrame with Google Fit data (or paths to JSON files)
            users_df: Optional DataFrame with user information
            
        Returns:
            DataFrame with standardized sleep data
        """
        # Check if this is raw Google Fit export or pre-processed data
        if 'json_data' in data_df.columns or 'file_path' in data_df.columns:
            logger.info("Processing raw Google Fit export")
            return self._process_google_export(data_df, users_df)
        
        # For pre-processed Google data
        transformed_data = []
        
        for _, row in data_df.iterrows():
            try:
                # Create a new record with standardized fields
                record = {
                    'user_id': row.get('user_id'),
                    'date': pd.to_datetime(row.get('date', row.get('sleep_start_time'))).strftime('%Y-%m-%d'),
                    'device_type': self.device_type
                }
                
                # Handle time fields
                if 'sleep_start_time' in row:
                    record['device_bedtime'] = pd.to_datetime(row['sleep_start_time']).strftime('%Y-%m-%d %H:%M:%S')
                elif 'bedtime' in row:
                    record['device_bedtime'] = pd.to_datetime(row['bedtime']).strftime('%Y-%m-%d %H:%M:%S')
                
                if 'sleep_end_time' in row:
                    record['device_wake_time'] = pd.to_datetime(row['sleep_end_time']).strftime('%Y-%m-%d %H:%M:%S')
                elif 'wake_time' in row:
                    record['device_wake_time'] = pd.to_datetime(row['wake_time']).strftime('%Y-%m-%d %H:%M:%S')
                
                # Calculate sleep onset time (roughly 10 minutes after bedtime for Google Fit)
                if 'device_bedtime' in record:
                    bedtime = pd.to_datetime(record['device_bedtime'])
                    record['device_sleep_onset'] = (bedtime + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
                
                # Handle duration
                if 'sleep_duration' in row:
                    # Google Fit typically reports in milliseconds
                    if isinstance(row['sleep_duration'], (int, float)) and row['sleep_duration'] > 1000000:
                        record['device_sleep_duration'] = row['sleep_duration'] / (1000 * 60 * 60)  # Convert ms to hours
                    elif isinstance(row['sleep_duration'], (int, float)) and row['sleep_duration'] > 1000:
                        record['device_sleep_duration'] = row['sleep_duration'] / 3600  # Convert seconds to hours
                    else:
                        record['device_sleep_duration'] = row['sleep_duration']  # Already in hours
                elif 'device_bedtime' in record and 'device_wake_time' in record:
                    # Calculate from timestamps
                    bedtime = pd.to_datetime(record['device_bedtime'])
                    wake_time = pd.to_datetime(record['device_wake_time'])
                    duration_hours = (wake_time - bedtime).total_seconds() / 3600
                    record['device_sleep_duration'] = duration_hours
                
                # Handle sleep stages
                sleep_stages = {}
                
                for stage in ['deep_sleep_duration', 'light_sleep_duration', 'rem_sleep_duration', 'awake_duration']:
                    short_name = stage.replace('_duration', '')  # Convert to our standard names
                    
                    if stage in row:
                        # Google Fit typically reports in milliseconds
                        value = row[stage]
                        
                        # If very large, assume milliseconds and convert to hours
                        if isinstance(value, (int, float)) and value > 1000000:
                            sleep_stages[short_name] = value / (1000 * 60 * 60)
                        # If large, assume seconds and convert to hours
                        elif isinstance(value, (int, float)) and value > 1000:
                            sleep_stages[short_name] = value / 3600
                        # If moderate, assume minutes and convert to hours
                        elif isinstance(value, (int, float)) and value > 60:
                            sleep_stages[short_name] = value / 60
                        else:
                            sleep_stages[short_name] = value  # Assume already in hours
                
                # Update the record with sleep stages
                record.update(sleep_stages)
                
                # Calculate sleep stage percentages
                if any(k in record for k in ['deep_sleep', 'light_sleep', 'rem_sleep', 'awake_time']):
                    stage_percentages = self._calculate_sleep_stages(record)
                    record.update(stage_percentages)
                
                # Heart rate metrics
                if 'average_heart_rate' in row:
                    record['average_heart_rate'] = row['average_heart_rate']
                
                if 'min_heart_rate' in row:
                    record['min_heart_rate'] = row['min_heart_rate']
                
                if 'max_heart_rate' in row:
                    record['max_heart_rate'] = row['max_heart_rate']
                
                # Heart rate variability
                if 'heart_rate_variability' in row:
                    record['heart_rate_variability'] = row['heart_rate_variability']
                
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
                    if 'awake_count' in row:
                        record['awakenings_count'] = row['awake_count']
                    elif 'awake_time' in record:
                        # Estimate from awake time - approximately one awakening every 8 minutes for Google Fit
                        record['awakenings_count'] = max(1, int(record['awake_time'] * 60 / 8))
                    else:
                        record['awakenings_count'] = 2  # Default value
                
                transformed_data.append(record)
                
            except Exception as e:
                logger.error(f"Error processing Google Fit record: {str(e)}")
                continue
        
        return pd.DataFrame(transformed_data)
    
    def _process_google_export(self, data_df: pd.DataFrame, users_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Process raw Google Fit export files
        
        Args:
            data_df: DataFrame containing paths to JSON files
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
                
                # Extract sleep sessions from Google Fit JSON
                if 'sessions' in json_data:
                    # Process each sleep session
                    for session in json_data['sessions']:
                        # Check if it's a sleep session
                        if 'activityType' in session and session['activityType'] == 72:  # 72 is sleep in Google Fit
                            record = self._parse_google_sleep_session(session, user_id)
                            if record:
                                sleep_records.append(record)
                
                # Alternative format - "Fitness Activity"
                elif 'bucket' in json_data:
                    for bucket in json_data['bucket']:
                        if 'dataset' in bucket:
                            for dataset in bucket['dataset']:
                                if 'point' in dataset:
                                    for point in dataset['point']:
                                        if 'sleepSegment' in point or 'sleepStage' in point:
                                            record = self._parse_google_sleep_point(point, bucket, user_id)
                                            if record:
                                                sleep_records.append(record)
                
            except Exception as e:
                logger.error(f"Error processing Google Fit JSON: {str(e)}")
                continue
        
        # Convert to DataFrame
        sleep_df = pd.DataFrame(sleep_records)
        
        if sleep_df.empty:
            logger.warning("No sleep records found in Google Fit data")
            return sleep_df
        
        # Group by date to combine records from the same night
        try:
            grouped_records = []
            
            for date, group in sleep_df.groupby('date'):
                # Combine sleep stage data
                combined_record = {
                    'user_id': user_id,
                    'date': date,
                    'device_type': self.device_type,
                    'deep_sleep': 0,
                    'light_sleep': 0,
                    'rem_sleep': 0,
                    'awake_time': 0
                }
                
                # Find earliest bedtime and latest wake time
                if 'device_bedtime' in group.columns:
                    combined_record['device_bedtime'] = group['device_bedtime'].min()
                
                if 'device_wake_time' in group.columns:
                    combined_record['device_wake_time'] = group['device_wake_time'].max()
                
                # Calculate sleep onset
                if 'device_sleep_onset' in group.columns:
                    combined_record['device_sleep_onset'] = group['device_sleep_onset'].min()
                elif 'device_bedtime' in combined_record:
                    bedtime = pd.to_datetime(combined_record['device_bedtime'])
                    combined_record['device_sleep_onset'] = (bedtime + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
                
                # Sum up sleep stages
                for stage in ['deep_sleep', 'light_sleep', 'rem_sleep', 'awake_time']:
                    if stage in group.columns:
                        combined_record[stage] = group[stage].sum()
                
                # Get heart rate data
                if 'average_heart_rate' in group.columns and not group['average_heart_rate'].isna().all():
                    combined_record['average_heart_rate'] = group['average_heart_rate'].mean()
                
                if 'min_heart_rate' in group.columns and not group['min_heart_rate'].isna().all():
                    combined_record['min_heart_rate'] = group['min_heart_rate'].min()
                
                if 'max_heart_rate' in group.columns and not group['max_heart_rate'].isna().all():
                    combined_record['max_heart_rate'] = group['max_heart_rate'].max()
                
                if 'heart_rate_variability' in group.columns and not group['heart_rate_variability'].isna().all():
                    combined_record['heart_rate_variability'] = group['heart_rate_variability'].mean()
                
                # Calculate time in bed
                if 'device_bedtime' in combined_record and 'device_wake_time' in combined_record:
                    bedtime = pd.to_datetime(combined_record['device_bedtime'])
                    wake_time = pd.to_datetime(combined_record['device_wake_time'])
                    combined_record['time_in_bed_hours'] = (wake_time - bedtime).total_seconds() / 3600
                
                # Calculate total sleep duration
                total_sleep = combined_record.get('deep_sleep', 0) + combined_record.get('light_sleep', 0) + combined_record.get('rem_sleep', 0)
                if total_sleep > 0:
                    combined_record['device_sleep_duration'] = total_sleep
                elif 'time_in_bed_hours' in combined_record:
                    # Estimate as 85% of time in bed
                    combined_record['device_sleep_duration'] = combined_record['time_in_bed_hours'] * 0.85
                
                # Calculate sleep stage percentages
                stage_percentages = self._calculate_sleep_stages(combined_record)
                combined_record.update(stage_percentages)
                
                # Calculate sleep efficiency
                if 'time_in_bed_hours' in combined_record and 'device_sleep_duration' in combined_record and combined_record['time_in_bed_hours'] > 0:
                    combined_record['sleep_efficiency'] = combined_record['device_sleep_duration'] / combined_record['time_in_bed_hours']
                else:
                    combined_record['sleep_efficiency'] = self._estimate_sleep_efficiency(combined_record)
                
                # Estimate awakenings count
                if 'awakenings_count' not in combined_record and 'awake_time' in combined_record:
                    combined_record['awakenings_count'] = max(1, int(combined_record['awake_time'] * 60 / 8))
                
                grouped_records.append(combined_record)
            
            return pd.DataFrame(grouped_records)
            
        except Exception as e:
            logger.error(f"Error grouping Google Fit sleep records: {str(e)}")
            return sleep_df
    
    def _parse_google_sleep_session(self, session, user_id):
        """Parse a Google Fit sleep session into a standardized sleep record"""
        try:
            # Extract start and end times
            start_time = pd.to_datetime(int(session['startTimeMillis']), unit='ms')
            end_time = pd.to_datetime(int(session['endTimeMillis']), unit='ms')
            
            # Create base record
            record = {
                'user_id': user_id,
                'date': start_time.strftime('%Y-%m-%d'),
                'device_type': self.device_type,
                'device_bedtime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'device_wake_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Calculate sleep duration
            duration_ms = int(session['endTimeMillis']) - int(session['startTimeMillis'])
            record['device_sleep_duration'] = duration_ms / (1000 * 60 * 60)  # Convert to hours
            
            # Extract sleep stage data if available
            if 'sleepStages' in session:
                stages = session['sleepStages']
                
                # Initialize sleep stages
                record['deep_sleep'] = 0
                record['light_sleep'] = 0
                record['rem_sleep'] = 0
                record['awake_time'] = 0
                
                for stage in stages:
                    stage_type = stage.get('sleepStageType', '')
                    duration_millis = int(stage.get('endTimeMillis', 0)) - int(stage.get('startTimeMillis', 0))
                    duration_hours = duration_millis / (1000 * 60 * 60)
                    
                    if 'deep' in stage_type.lower():
                        record['deep_sleep'] += duration_hours
                    elif 'light' in stage_type.lower():
                        record['light_sleep'] += duration_hours
                    elif 'rem' in stage_type.lower():
                        record['rem_sleep'] += duration_hours
                    elif 'awake' in stage_type.lower():
                        record['awake_time'] += duration_hours
            
            # Calculate time in bed
            record['time_in_bed_hours'] = (end_time - start_time).total_seconds() / 3600
            
            # Calculate sleep efficiency
            if 'device_sleep_duration' in record and 'time_in_bed_hours' in record and record['time_in_bed_hours'] > 0:
                # Sleep duration minus awake time
                if 'awake_time' in record:
                    sleep_time = record['device_sleep_duration'] - record['awake_time']
                else:
                    sleep_time = record['device_sleep_duration'] * 0.9  # Estimate 90% of total
                    
                record['sleep_efficiency'] = sleep_time / record['time_in_bed_hours']
            else:
                record['sleep_efficiency'] = self._estimate_sleep_efficiency(record)
            
            # Calculate sleep stage percentages
            if any(k in record for k in ['deep_sleep', 'light_sleep', 'rem_sleep', 'awake_time']):
                stage_percentages = self._calculate_sleep_stages(record)
                record.update(stage_percentages)
            
            return record
            
        except Exception as e:
            logger.error(f"Error parsing Google Fit sleep session: {str(e)}")
            return None
    
    def _parse_google_sleep_point(self, point, bucket, user_id):
        """Parse a Google Fit sleep point into a standardized sleep record"""
        try:
            # Check what type of sleep data this is
            if 'sleepSegment' in point:
                # Extract segment info
                segment = point['sleepSegment']
                
                # Extract start and end times
                start_time = pd.to_datetime(int(segment['startTimeMillis']), unit='ms')
                end_time = pd.to_datetime(int(segment['endTimeMillis']), unit='ms')
                
                # Create record
                record = {
                    'user_id': user_id,
                    'date': start_time.strftime('%Y-%m-%d'),
                    'device_type': self.device_type,
                    'device_bedtime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'device_wake_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Calculate sleep duration
                duration_ms = int(segment['endTimeMillis']) - int(segment['startTimeMillis'])
                record['device_sleep_duration'] = duration_ms / (1000 * 60 * 60)  # Convert to hours
                
                return record
                
            elif 'sleepStage' in point:
                # Extract stage info
                stage_data = point['sleepStage']
                stage_type = stage_data.get('sleepStageType', '')
                
                # Extract start and end times from the bucket
                start_time = pd.to_datetime(int(bucket['startTimeMillis']), unit='ms')
                end_time = pd.to_datetime(int(bucket['endTimeMillis']), unit='ms')
                
                # Calculate duration
                duration_ms = int(bucket['endTimeMillis']) - int(bucket['startTimeMillis'])
                duration_hours = duration_ms / (1000 * 60 * 60)
                
                # Create record
                record = {
                    'user_id': user_id,
                    'date': start_time.strftime('%Y-%m-%d'),
                    'device_type': self.device_type,
                    'device_bedtime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'device_wake_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Set appropriate sleep stage
                if 'deep' in stage_type.lower():
                    record['deep_sleep'] = duration_hours
                elif 'light' in stage_type.lower():
                    record['light_sleep'] = duration_hours
                elif 'rem' in stage_type.lower():
                    record['rem_sleep'] = duration_hours
                elif 'awake' in stage_type.lower():
                    record['awake_time'] = duration_hours
                else:
                    # Unknown stage - just use as sleep duration
                    record['device_sleep_duration'] = duration_hours
                
                return record
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing Google Fit sleep point: {str(e)}")
            return None