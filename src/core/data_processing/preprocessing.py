import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pydantic import ValidationError

from src.core.models.data_models import SleepEntry, WearableData, UserProfile
from src.core.models.improved_sleep_score import ImprovedSleepScoreCalculator
from src.utils.data_validation_fix import ensure_sleep_data_format
from src.utils.ensure_valid_numeric_fields import ensure_valid_numeric_fields
from src.core.wearables.wearable_transformer_manager import WearableTransformerManager



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/preprocessor.log')
    ]
)

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, config=None):
        """Initialize the data preprocessor"""
        self.sleep_score_calculator = ImprovedSleepScoreCalculator()
        self.processed_data = None
        self.config = config or {}
        self.wearable_manager = WearableTransformerManager()
        
        # Set defaults for missing config values
        self.config.setdefault('convert_dates', True)
        self.config.setdefault('calculate_sleep_features', True)
        self.config.setdefault('calculate_wearable_features', True)
        self.config.setdefault('max_bedtime_std_minutes', 180)
        self.config.setdefault('max_waketime_std_minutes', 180)
        self.config.setdefault('max_duration_std_hours', 3)
        self.config.setdefault('default_consistency', 0.5)
        self.config.setdefault('missing_value_strategy', 'user_mean')
        self.config.setdefault('fallback_strategy', 'global_mean')

    def process(self, users_df, sleep_data_df, wearable_data_df=None, external_factors_df=None):
        """Process all data sources together"""
        # Validate inputs
        if users_df is None or sleep_data_df is None:
            raise ValueError("Users and sleep data are required")
        
        # IMPORTANT: Save original user_ids
        original_user_ids = sleep_data_df['user_id'].copy()
        original_row_count = len(sleep_data_df)

        logger.info(f"Processing {len(sleep_data_df)} rows of sleep data")
        
        # Preprocess sleep data
        processed_data = self.preprocess_sleep_data(sleep_data_df, wearable_data_df, external_factors_df)
        
        # CRITICAL FIX: Check for data explosion
        if len(processed_data) > original_row_count * 2:  # If more than double
            logger.info(f"WARNING: Data explosion detected - {len(processed_data)} rows vs original {original_row_count}")
            logger.info("Rebuilding processed_data with original structure...")
            
            # Create a new dataframe with the same structure as sleep_data_df
            fixed_data = sleep_data_df.copy()
            
            # Only keep the core rows that match the original sleep_data_df
            # And add any new calculated columns from processed_data
            for col in processed_data.columns:
                if col not in fixed_data.columns and col != 'user_id':
                    # For numerical columns, use the mean per user_id
                    if pd.api.types.is_numeric_dtype(processed_data[col]):
                        # Group by user_id and get mean
                        col_means = processed_data.groupby('user_id')[col].mean()
                        
                        # Map these means to the original data structure
                        fixed_data[col] = fixed_data['user_id'].map(col_means)
                    else:
                        # For non-numeric columns, use the most common value per user_id
                        col_modes = processed_data.groupby('user_id')[col].agg(
                            lambda x: x.mode().iloc[0] if not x.mode().empty else None
                        )
                        fixed_data[col] = fixed_data['user_id'].map(col_modes)
            
            # Use the fixed data
            processed_data = fixed_data
            logger.info(f"Fixed data has {len(processed_data)} rows with {processed_data.columns.size} columns")
        
        # Ensure user_id is in processed_data
        if 'user_id' not in processed_data.columns:
            processed_data['user_id'] = original_user_ids.values[:len(processed_data)]
            logger.info("Added user_id from original data")
        
        # Validate time_in_bed_hours before returning 
        if 'time_in_bed_hours' in processed_data.columns:
            processed_data['time_in_bed_hours'] = processed_data['time_in_bed_hours'].clip(upper=24.0)

        processed_data = ensure_valid_numeric_fields(processed_data)

        return processed_data
    
    def _get_season(self, month):
        """Determine season from month (Northern Hemisphere)"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring' 
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # 9, 10, 11
            return 'Fall'
    
    def process_wearable_data(self, wearable_data_df, device_type, users_df=None):
        """Process wearable data using the appropriate transformer"""
        try:
            transformed_data = self.wearable_manager.transform_data(
                wearable_data_df, 
                device_type,
                users_df
            )
            return transformed_data
        except Exception as e:
            print(f"Error processing wearable data: {e}")
            return pd.DataFrame()

    def preprocess_sleep_data(self, sleep_data, wearable_data=None, external_data=None):
        """Preprocess and merge sleep data with wearable and external data"""
        # Existing preprocessing code...
        processed_data = sleep_data.copy()
        
        # If wearable data is provided as a dictionary of {device_type: data_df}
        if isinstance(wearable_data, dict):
            all_wearable_data = []
            for device_type, data_df in wearable_data.items():
                transformed = self.process_wearable_data(data_df, device_type)
                all_wearable_data.append(transformed)
            
            if all_wearable_data:
                combined_wearable = pd.concat(all_wearable_data, ignore_index=True)
                # Merge with sleep data
                processed_data = self._merge_sleep_and_wearable_data(processed_data, combined_wearable)
        
        # If wearable data is provided as a single DataFrame with device_type column
        elif wearable_data is not None:
            if 'device_type' in wearable_data.columns:
                # Split by device type and transform each
                all_wearable_data = []
                for device_type, group in wearable_data.groupby('device_type'):
                    transformed = self.process_wearable_data(group, device_type)
                    all_wearable_data.append(transformed)
                
                if all_wearable_data:
                    combined_wearable = pd.concat(all_wearable_data, ignore_index=True)
                    # Merge with sleep data
                    processed_data = self._merge_sleep_and_wearable_data(processed_data, combined_wearable)
            else:
                # Assume data is already in standardized format
                processed_data = self._merge_sleep_and_wearable_data(processed_data, wearable_data)
        
        # Rest of the preprocessing code...
        
        return processed_data
    
    def _merge_sleep_and_wearable_data(self, sleep_data, wearable_data):
        """Merge sleep and wearable data with smart handling of conflicting fields"""
        # Identify columns to merge on
        merge_columns = ['user_id', 'date']
        
        # Columns that should be taken from wearable data if available
        wearable_priority_columns = [
            'deep_sleep_percentage', 'light_sleep_percentage', 'rem_sleep_percentage',
            'heart_rate_variability', 'average_heart_rate', 'min_heart_rate', 'max_heart_rate',
            'blood_oxygen', 'awakenings_count'
        ]
        
        # Merge the dataframes
        merged_data = pd.merge(
            sleep_data,
            wearable_data,
            on=merge_columns,
            how='left',
            suffixes=('', '_wearable')
        )
        
        # For priority columns, prefer wearable data where available
        for col in wearable_priority_columns:
            wearable_col = f"{col}_wearable"
            if wearable_col in merged_data.columns:
                # Where wearable data exists, use it
                mask = ~merged_data[wearable_col].isna()
                if col in merged_data.columns:
                    merged_data.loc[mask, col] = merged_data.loc[mask, wearable_col]
                else:
                    # If column doesn't exist in sleep data, create it
                    merged_data[col] = merged_data[wearable_col]
                
                # Drop the _wearable column
                merged_data = merged_data.drop(columns=[wearable_col])
        
        # Drop any remaining _wearable columns
        wearable_cols = [col for col in merged_data.columns if col.endswith('_wearable')]
        if wearable_cols:
            merged_data = merged_data.drop(columns=wearable_cols)
        
        return merged_data
    
    def transform_wearable_data(self, wearable_data):
        """Transform wearable device data to standard sleep data format"""
        # Create a copy of the incoming DataFrame to avoid modifying the original
        transformed_data = wearable_data.copy()
        
        # If sleep_score_calculator is initialized, use its transformation method
        if hasattr(self, 'sleep_score_calculator'):
            # Get the original transformation as a dictionary
            transformed_dict = self.sleep_score_calculator.transform_wearable_data(wearable_data.to_dict('records')[0])
            
            # Convert the dictionary to a DataFrame with the same index as the original
            dict_df = pd.DataFrame([transformed_dict], index=wearable_data.index[:1])
            
            # Update the transformed data with the new columns
            for col in dict_df.columns:
                if col not in transformed_data.columns:
                    transformed_data[col] = dict_df[col].values[0]
        
        return transformed_data
    
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

        # Handle alternative field names
        if 'total_awake_minutes' in data.columns and 'time_awake_minutes' not in data.columns:
            data['time_awake_minutes'] = data['total_awake_minutes']
        elif 'time_awake_minutes' in data.columns and 'total_awake_minutes' not in data.columns:
            data['total_awake_minutes'] = data['time_awake_minutes']
        
        # Return the cleaned data
        return data