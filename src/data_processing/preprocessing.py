import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models.improved_sleep_score import ImprovedSleepScoreCalculator


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessor.log')
    ]
)

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, config=None):
        """Initialize the data preprocessor"""
        self.sleep_score_calculator = ImprovedSleepScoreCalculator()
        self.processed_data = None
        self.config = config or {}
        
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
        
        return processed_data
    
    def preprocess_sleep_data(self, sleep_data, wearable_data=None, external_data=None):
        """Preprocess and merge sleep data with wearable and external data"""
        # IMPORTANT: Make a copy of sleep_data to avoid modifying the original
        processed_data = sleep_data.copy()
        
        # Convert string dates to datetime for easier processing
        if self.config.get('convert_dates', True):
            if 'date' in processed_data.columns:
                processed_data['date'] = pd.to_datetime(processed_data['date'])
            if 'bedtime' in processed_data.columns:
                processed_data['bedtime'] = pd.to_datetime(processed_data['bedtime'])
            if 'sleep_onset_time' in processed_data.columns:
                processed_data['sleep_onset_time'] = pd.to_datetime(processed_data['sleep_onset_time'])
            if 'wake_time' in processed_data.columns:
                processed_data['wake_time'] = pd.to_datetime(processed_data['wake_time'])
        
        # Add calculated sleep features
        processed_data = self._add_sleep_features(processed_data)


        # Merge with wearable data if provided
        if wearable_data is not None:
            # Convert dates in wearable data
            if self.config.get('convert_dates', True):
                if 'date' in wearable_data.columns:
                    wearable_data['date'] = pd.to_datetime(wearable_data['date'])
                if 'device_bedtime' in wearable_data.columns:
                    wearable_data['device_bedtime'] = pd.to_datetime(wearable_data['device_bedtime'])
                if 'device_sleep_onset' in wearable_data.columns:
                    wearable_data['device_sleep_onset'] = pd.to_datetime(wearable_data['device_sleep_onset'])
                if 'device_wake_time' in wearable_data.columns:
                    wearable_data['device_wake_time'] = pd.to_datetime(wearable_data['device_wake_time'])
            
            # Columns to drop from wearable data
            columns_to_drop = self.config.get('wearable_drop_columns', 
                                            ['heart_rate_data', 'movement_data', 'sleep_stage_data'])
            
            # Drop specified columns if they exist
            wearable_cols_to_drop = [col for col in columns_to_drop if col in wearable_data.columns]
            wearable_data_cleaned = wearable_data.drop(wearable_cols_to_drop, axis=1, errors='ignore')

            transformed_wearable_data = self.transform_wearable_data(wearable_data_cleaned)
            
            # IMPORTANT: Ensure user_id is preserved in wearable data
            if 'user_id' not in processed_data.columns and 'user_id' in wearable_data.columns:
                # If user_id is missing in processed_data but exists in wearable_data, something is wrong
                # Let's keep track of where it happens
                logger.info("WARNING: user_id missing in processed_data but present in wearable_data")
            
            # Get merge columns from config or use defaults
            merge_columns = self.config.get('wearable_merge_columns', ['user_id', 'date'])
            
            # Verify merge columns exist in both dataframes
            for col in merge_columns:
                if col not in processed_data.columns:
                    logger.info(f"WARNING: Merge column {col} not in processed_data")
                if col not in wearable_data_cleaned.columns:
                    logger.info(f"WARNING: Merge column {col} not in wearable_data")
            
            # Only include merge columns that exist in both dataframes
            valid_merge_columns = [col for col in merge_columns 
                                if col in processed_data.columns and col in wearable_data_cleaned.columns]
            
            if not valid_merge_columns:
                logger.info("ERROR: No valid merge columns. Skipping wearable data merge.")
            else:
                def column_is_type(df, t):
                    return df.transform(lambda x: x.apply(type).eq(t)).all()
                
                # Merge on valid columns
                processed_data = pd.merge(
                    processed_data, 
                    transformed_wearable_data, 
                    on=valid_merge_columns, 
                    how='left',
                    suffixes=('', '_y')
                )
            
            # Add calculated wearable features
            if self.config.get('calculate_wearable_features', True):
                processed_data = self._add_wearable_features(processed_data)
        
        # Merge with external data if provided
        if external_data is not None:
            if 'date' in external_data.columns:
                if self.config.get('convert_dates', True):
                    external_data['date'] = pd.to_datetime(external_data['date'])
                                
                processed_data = pd.merge(
                    processed_data,
                    external_data,
                    on='date',
                    how='left',
                    suffixes=('', '_y')
                )

                # Directly identify and drop the _y columns in one operation
                y_cols = [col for col in processed_data.columns if col.endswith('_y')]
                if y_cols:
                    logger.info(f"Dropping columns: {y_cols}")
                    processed_data.drop(columns=y_cols, inplace=True)
        
        # Handle missing values
        processed_data = self._handle_missing_values(processed_data)

        # Check for essential columns before returning
        essential_columns = ['user_id', 'date', 'sleep_efficiency']
        missing_columns = [col for col in essential_columns if col not in processed_data.columns]
        
        if missing_columns:
            logger.info(f"Warning: Essential columns are missing after preprocessing: {missing_columns}")
        
        self.processed_data = processed_data
        
        # Check for essential columns before returning
        essential_columns = ['user_id', 'date', 'sleep_efficiency']
        missing_columns = [col for col in essential_columns if col not in processed_data.columns]
        
        if missing_columns:
            print(f"Warning: Essential columns are missing after preprocessing: {missing_columns}")
        
        return processed_data
    
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
        
        # Return the cleaned data
        return data