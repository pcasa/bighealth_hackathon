"""
Base transformer class for wearable sleep data.
All device-specific transformers inherit from this class.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class BaseWearableTransformer:
    """Base class for wearable data transformers"""
    
    def __init__(self):
        """Initialize the transformer with a default device type"""
        self.device_type = "generic"
    
    def transform(self, 
                 data: Union[pd.DataFrame, List[Dict], Dict], 
                 users_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transform wearable data to standardized sleep format
        
        Args:
            data: Wearable data as DataFrame, list of dicts, or single dict
            users_df: Optional DataFrame with user information
            
        Returns:
            DataFrame with standardized sleep data
        """
        # Convert input to DataFrame if it's not already
        if isinstance(data, dict):
            data_df = pd.DataFrame([data])
        elif isinstance(data, list):
            data_df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            data_df = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Apply the device-specific transformation
        try:
            transformed_df = self._transform_data(data_df, users_df)
            
            # Ensure required columns exist
            required_columns = ['user_id', 'date', 'device_type']
            missing_columns = [col for col in required_columns if col not in transformed_df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns in transformed data: {missing_columns}")
                # Add missing columns with defaults
                for col in missing_columns:
                    if col == 'device_type':
                        transformed_df[col] = self.device_type
                    else:
                        transformed_df[col] = None
            
            return transformed_df
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def _transform_data(self, data_df: pd.DataFrame, users_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Device-specific transformation logic to be implemented by subclasses
        
        Args:
            data_df: DataFrame with device-specific data
            users_df: Optional DataFrame with user information
            
        Returns:
            DataFrame with standardized sleep data
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement _transform_data")
    
    def _calculate_sleep_stages(self, record: Dict) -> Dict:
        """
        Calculate sleep stage percentages from hour values
        
        Args:
            record: Dictionary containing sleep stage durations
            
        Returns:
            Dictionary with sleep stage percentages
        """
        # Extract sleep stage durations
        deep_sleep = record.get('deep_sleep', 0)
        light_sleep = record.get('light_sleep', 0)  
        rem_sleep = record.get('rem_sleep', 0)
        awake_time = record.get('awake_time', 0)
        
        # Calculate total time
        total_time = deep_sleep + light_sleep + rem_sleep + awake_time
        
        # Calculate percentages (avoid division by zero)
        if total_time > 0:
            deep_percentage = deep_sleep / total_time
            light_percentage = light_sleep / total_time
            rem_percentage = rem_sleep / total_time
            awake_percentage = awake_time / total_time
        else:
            # Default percentages if no sleep stage data
            deep_percentage = 0.2
            light_percentage = 0.5
            rem_percentage = 0.25
            awake_percentage = 0.05
        
        return {
            'deep_sleep_percentage': deep_percentage,
            'light_sleep_percentage': light_percentage,
            'rem_sleep_percentage': rem_percentage,
            'awake_percentage': awake_percentage
        }
    
    def _estimate_sleep_efficiency(self, record: Dict) -> float:
        """
        Estimate sleep efficiency when not directly provided
        
        Args:
            record: Dictionary containing sleep data
            
        Returns:
            Estimated sleep efficiency (0-1)
        """
        # If we have sleep duration and time in bed
        if 'device_sleep_duration' in record and 'time_in_bed_hours' in record and record['time_in_bed_hours'] > 0:
            return record['device_sleep_duration'] / record['time_in_bed_hours']
        
        # If we have sleep stage percentages
        elif 'deep_sleep_percentage' in record and 'rem_sleep_percentage' in record and 'light_sleep_percentage' in record:
            # Sleep efficiency is roughly the sum of all sleep stages (excluding awake)
            return record['deep_sleep_percentage'] + record['light_sleep_percentage'] + record['rem_sleep_percentage']
        
        # If we have subjective rating (assuming 1-10 scale)
        elif 'subjective_rating' in record:
            rating = record['subjective_rating']
            # Convert to 0-1 scale with some scaling to match typical efficiency patterns
            # Ratings below 5 have more dramatic effect on efficiency
            if rating <= 5:
                return 0.5 + (rating / 10)
            else:
                return 0.75 + (rating - 5) / 20
        
        # Default value based on average sleep efficiency
        else:
            return 0.85