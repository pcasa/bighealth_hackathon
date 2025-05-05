"""
Apple Watch data transformer that converts Health Export data to standardized sleep format.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Union
import xml.etree.ElementTree as ET
import zipfile
import io

from src.core.wearables.wearable_base_transformer import BaseWearableTransformer

logger = logging.getLogger(__name__)

class AppleWatchTransformer(BaseWearableTransformer):
    """Transform Apple Watch/Apple Health sleep data to standardized format"""
    
    def __init__(self):
        super().__init__()
        self.device_type = "apple_watch"
    
    def _transform_data(self, data_df: pd.DataFrame, users_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Transform Apple Watch data to standardized sleep format
        
        Args:
            data_df: DataFrame with Apple Watch data
            users_df: Optional DataFrame with user information
            
        Returns:
            DataFrame with standardized sleep data
        """
        # Check if this is raw Apple Health export (XML) or pre-processed data
        if 'xml_data' in data_df.columns or (len(data_df.columns) == 1 and 'data' in data_df.columns):
            logger.info("Processing raw Apple Health export XML")
            return self._process_health_export(data_df, users_df)
        
        # For pre-processed Apple Watch data
        transformed_data = []
        
        for _, row in data_df.iterrows():
            try:
                # Create a new record with standardized fields
                record = {
                    'user_id': row.get('user_id'),
                    'date': pd.to_datetime(row.get('date', row.get('start_date'))).strftime('%Y-%m-%d'),
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
                
                # Calculate sleep onset time (typically 5-15 min after bedtime)
                if 'device_bedtime' in record:
                    bedtime = pd.to_datetime(record['device_bedtime'])
                    onset_minutes = row.get('sleep_onset_minutes', 10)  # Default 10 min to fall asleep
                    record['device_sleep_onset'] = (bedtime + timedelta(minutes=onset_minutes)).strftime('%Y-%m-%d %H:%M:%S')
                
                # Handle duration
                if 'sleep_duration_seconds' in row:
                    record['device_sleep_duration'] = row['sleep_duration_seconds'] / 3600  # Convert to hours
                elif 'sleep_duration_minutes' in row:
                    record['device_sleep_duration'] = row['sleep_duration_minutes'] / 60  # Convert to hours
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
                
                if 'deep_sleep_seconds' in row:
                    sleep_stages['deep_sleep'] = row['deep_sleep_seconds'] / 3600  # Convert to hours
                elif 'deep_sleep_minutes' in row:
                    sleep_stages['deep_sleep'] = row['deep_sleep_minutes'] / 60  # Convert to hours
                elif 'deep_sleep' in row:
                    sleep_stages['deep_sleep'] = row['deep_sleep']
                    
                if 'core_sleep_seconds' in row or 'light_sleep_seconds' in row:
                    key = 'core_sleep_seconds' if 'core_sleep_seconds' in row else 'light_sleep_seconds'
                    sleep_stages['light_sleep'] = row[key] / 3600  # Convert to hours
                elif 'light_sleep_minutes' in row:
                    sleep_stages['light_sleep'] = row['light_sleep_minutes'] / 60  # Convert to hours
                elif 'light_sleep' in row:
                    sleep_stages['light_sleep'] = row['light_sleep']
                    
                if 'rem_sleep_seconds' in row:
                    sleep_stages['rem_sleep'] = row['rem_sleep_seconds'] / 3600  # Convert to hours
                elif 'rem_sleep_minutes' in row:
                    sleep_stages['rem_sleep'] = row['rem_sleep_minutes'] / 60  # Convert to hours
                elif 'rem_sleep' in row:
                    sleep_stages['rem_sleep'] = row['rem_sleep']
                    
                if 'awake_time_seconds' in row:
                    sleep_stages['awake_time'] = row['awake_time_seconds'] / 3600  # Convert to hours
                elif 'awake_minutes' in row:
                    sleep_stages['awake_time'] = row['awake_minutes'] / 60  # Convert to hours
                elif 'awake_time' in row:
                    sleep_stages['awake_time'] = row['awake_time']
                
                # Calculate sleep stage percentages
                row_with_stages = {**row, **sleep_stages}
                stage_percentages = self._calculate_sleep_stages(row_with_stages)
                record.update(stage_percentages)
                
                # Heart rate metrics
                if 'avg_heart_rate' in row:
                    record['average_heart_rate'] = row['avg_heart_rate']
                elif 'average_heart_rate' in row:
                    record['average_heart_rate'] = row['average_heart_rate']
                    
                if 'min_heart_rate' in row:
                    record['min_heart_rate'] = row['min_heart_rate']
                elif 'minimum_heart_rate' in row:
                    record['min_heart_rate'] = row['minimum_heart_rate']
                    
                if 'max_heart_rate' in row:
                    record['max_heart_rate'] = row['max_heart_rate']
                elif 'maximum_heart_rate' in row:
                    record['max_heart_rate'] = row['maximum_heart_rate']
                
                # Heart rate variability
                if 'hrv_ms' in row:
                    record['heart_rate_variability'] = row['hrv_ms']
                elif 'hrv' in row:
                    record['heart_rate_variability'] = row['hrv']
                elif 'heart_rate_variability' in row:
                    record['heart_rate_variability'] = row['heart_rate_variability']
                
                # Blood oxygen
                if 'oxygen_saturation' in row:
                    record['blood_oxygen'] = row['oxygen_saturation']
                elif 'blood_oxygen' in row:
                    record['blood_oxygen'] = row['blood_oxygen']
                
                # Add respiratory rate if available
                if 'respiratory_rate' in row:
                    record['respiratory_rate'] = row['respiratory_rate']
                
                # Use record to calculate any missing fields
                if 'device_sleep_duration' in record and 'device_bedtime' in record and 'device_wake_time' in record:
                    # Calculate time in bed
                    bedtime = pd.to_datetime(record['device_bedtime'])
                    wake_time = pd.to_datetime(record['device_wake_time'])
                    record['time_in_bed_hours'] = (wake_time - bedtime).total_seconds() / 3600
                    
                    # Estimate sleep efficiency
                    if 'device_sleep_duration' in record and record['time_in_bed_hours'] > 0:
                        record['sleep_efficiency'] = record['device_sleep_duration'] / record['time_in_bed_hours']
                    else:
                        record['sleep_efficiency'] = self._estimate_sleep_efficiency(record)
                
                transformed_data.append(record)
                
            except Exception as e:
                logger.error(f"Error processing Apple Watch record: {str(e)}")
                continue
        
        return pd.DataFrame(transformed_data)
    
    def _process_health_export(self, data_df: pd.DataFrame, users_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Process raw Apple Health Export XML data
        
        Args:
            data_df: DataFrame containing XML data or file paths
            users_df: Optional DataFrame with user information
            
        Returns:
            DataFrame with standardized sleep data
        """
        sleep_records = []
        user_id = users_df['user_id'].iloc[0] if users_df is not None and len(users_df) > 0 else "unknown"
        
        for _, row in data_df.iterrows():
            try:
                # Extract XML data
                if 'xml_data' in row:
                    xml_data = row['xml_data']
                elif 'data' in row and isinstance(row['data'], str):
                    xml_data = row['data']
                elif 'file_path' in row:
                    # Read XML from file
                    with open(row['file_path'], 'r') as f:
                        xml_data = f.read()
                else:
                    logger.warning("No XML data found in row")
                    continue
                
                # Parse XML
                root = ET.fromstring(xml_data)
                
                # Find sleep analysis records
                sleep_analyses = root.findall(".//Record[@type='HKCategoryTypeIdentifierSleepAnalysis']")
                
                # Process each sleep record
                for sleep_entry in sleep_analyses:
                    # Check if it's an in-bed or asleep record
                    value = sleep_entry.get('value')
                    
                    # Skip records that aren't sleep data
                    if value not in ['HKCategoryValueSleepAnalysisInBed', 'HKCategoryValueSleepAnalysisAsleep']:
                        continue
                    
                    # Extract start and end times
                    start_date = sleep_entry.get('startDate')
                    end_date = sleep_entry.get('endDate')
                    
                    if not start_date or not end_date:
                        continue
                    
                    # Convert to datetime
                    start_datetime = pd.to_datetime(start_date)
                    end_datetime = pd.to_datetime(end_date)
                    
                    # Calculate duration in hours
                    duration_hours = (end_datetime - start_datetime).total_seconds() / 3600
                    
                    # Create record
                    if value == 'HKCategoryValueSleepAnalysisInBed':
                        # This is an "in bed" record
                        record = {
                            'user_id': user_id,
                            'date': start_datetime.strftime('%Y-%m-%d'),
                            'device_type': self.device_type,
                            'device_bedtime': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                            'device_wake_time': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                            'time_in_bed_hours': duration_hours
                        }
                        sleep_records.append(record)
                    elif value == 'HKCategoryValueSleepAnalysisAsleep':
                        # This is an "asleep" record
                        record = {
                            'user_id': user_id,
                            'date': start_datetime.strftime('%Y-%m-%d'),
                            'device_type': self.device_type,
                            'device_sleep_onset': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                            'device_wake_time': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                            'device_sleep_duration': duration_hours
                        }
                        sleep_records.append(record)
                
                # Find heart rate records and process them
                hr_records = root.findall(".//Record[@type='HKQuantityTypeIdentifierHeartRate']")
                
                # Process heart rate records
                # (Code for this would be similar to sleep records processing)
                
                # Find HRV records
                hrv_records = root.findall(".//Record[@type='HKQuantityTypeIdentifierHeartRateVariabilitySDNN']")
                
                # Process HRV records
                # (Code for this would be similar to sleep records processing)
                
            except Exception as e:
                logger.error(f"Error processing Apple Health XML: {str(e)}")
                continue
        
        # Convert to DataFrame
        sleep_df = pd.DataFrame(sleep_records)
        
        if sleep_df.empty:
            logger.warning("No sleep records found in Apple Health data")
            return sleep_df
        
        # Process the records to combine in-bed and asleep records
        try:
            # Group by date to combine records from the same night
            processed_records = []
            
            for date, group in sleep_df.groupby('date'):
                # Get in-bed records
                in_bed = group[group['time_in_bed_hours'].notnull()]
                # Get asleep records
                asleep = group[group['device_sleep_duration'].notnull()]
                
                # If we have both types of records, combine them
                if not in_bed.empty and not asleep.empty:
                    # Use the in-bed record as base
                    combined = in_bed.iloc[0].to_dict()
                    
                    # Add sleep duration from asleep record
                    if 'device_sleep_duration' not in combined:
                        combined['device_sleep_duration'] = asleep['device_sleep_duration'].sum()
                    
                    # Calculate sleep efficiency
                    if combined['time_in_bed_hours'] > 0:
                        combined['sleep_efficiency'] = combined['device_sleep_duration'] / combined['time_in_bed_hours']
                    
                    # Estimate sleep stages based on sleep duration
                    combined.update(self._estimate_sleep_stages(combined))
                    
                    processed_records.append(combined)
                elif not in_bed.empty:
                    # Only have in-bed record
                    record = in_bed.iloc[0].to_dict()
                    # Estimate sleep duration (typically 85% of time in bed)
                    record['device_sleep_duration'] = record['time_in_bed_hours'] * 0.85
                    record['sleep_efficiency'] = 0.85
                    record.update(self._estimate_sleep_stages(record))
                    processed_records.append(record)
                elif not asleep.empty:
                    # Only have asleep record
                    record = asleep.iloc[0].to_dict()
                    # Estimate time in bed (typically sleep duration + 15%)
                    record['time_in_bed_hours'] = record['device_sleep_duration'] * 1.15
                    record['sleep_efficiency'] = 0.87
                    record.update(self._estimate_sleep_stages(record))
                    processed_records.append(record)
            
            return pd.DataFrame(processed_records)
            
        except Exception as e:
            logger.error(f"Error processing Apple Health sleep records: {str(e)}")
            # Return the original unprocessed records if there's an error
            return sleep_df
    
    def _estimate_sleep_stages(self, record):
        """Estimate sleep stages based on sleep duration and efficiency"""
        sleep_hours = record.get('device_sleep_duration', 7.0)
        efficiency = record.get('sleep_efficiency', 0.85)
        
        # Default percentages
        deep_pct = 0.20
        rem_pct = 0.25
        light_pct = 0.55
        
        # Adjust based on sleep duration
        if sleep_hours < 6:
            # Less REM and deep sleep for short sleepers
            deep_pct = 0.15
            rem_pct = 0.20
            light_pct = 0.65
        elif sleep_hours > 9:
            # More REM and deep sleep for long sleepers
            deep_pct = 0.22
            rem_pct = 0.28
            light_pct = 0.50
        
        # Calculate hours in each stage
        deep_hours = sleep_hours * deep_pct
        rem_hours = sleep_hours * rem_pct
        light_hours = sleep_hours * light_pct
        
        # Calculate awake time from efficiency
        time_in_bed = record.get('time_in_bed_hours', sleep_hours / efficiency if efficiency > 0 else sleep_hours * 1.15)
        awake_hours = time_in_bed - sleep_hours
        
        # Calculate percentages relative to time in bed
        total_hours = time_in_bed
        
        return {
            'deep_sleep_percentage': deep_hours / total_hours,
            'light_sleep_percentage': light_hours / total_hours,
            'rem_sleep_percentage': rem_hours / total_hours,
            'awake_percentage': awake_hours / total_hours
        }