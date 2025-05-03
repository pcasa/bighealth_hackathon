from datetime import datetime
import logging
import shutil

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def patch_file(file_path):
    """Patch the feature_engineering.py file with validation improvements"""
    # First create a backup of the original file
    backup_path = file_path + '.bak'
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    # Read the original file content
    with open(file_path, 'r') as file:
        content = file.read()

def fix_feature_ranges(df, feature_set_constraints=None):
    """
    Fix feature values to ensure they meet FeatureSet constraints.
    This function explicitly forces values into valid ranges.
    """
    # Define constraints based on FeatureSet if not provided
    if feature_set_constraints is None:
        feature_set_constraints = {
            # Non-negative fields
            'non_negative': [
                'sleep_duration_hours', 'time_in_bed_hours', 'sleep_onset_latency_minutes',
                'awakenings_count', 'total_awake_minutes', 'subjective_rating',
                'sleep_efficiency', 'heart_rate_variability', 'average_heart_rate'
            ],
            # Bounded between 0.0 and 1.0
            'zero_to_one': [
                'sleep_efficiency', 'deep_sleep_percentage', 'rem_sleep_percentage', 
                'light_sleep_percentage', 'awake_percentage', 'age_normalized',
                'profession_healthcare', 'profession_tech', 'profession_service', 
                'profession_education', 'profession_office', 'profession_other',
                'season_Winter', 'season_Spring', 'season_Summer', 'season_Fall'
            ]
        }
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Fix non-negative fields
    for field in feature_set_constraints['non_negative']:
        if field in df_copy.columns:
            df_copy[field] = df_copy[field].apply(
                lambda x: max(0, float(x)) if pd.notnull(x) and isinstance(x, (int, float)) else x
            )
    
    # Fix zero_to_one bounded fields
    for field in feature_set_constraints['zero_to_one']:
        if field in df_copy.columns:
            df_copy[field] = df_copy[field].apply(
                lambda x: min(max(0, float(x)), 1.0) if pd.notnull(x) and isinstance(x, (int, float)) else x
            )
    
    # Convert integer fields appropriately
    integer_fields = ['awakenings_count']
    for field in integer_fields:
        if field in df_copy.columns:
            df_copy[field] = df_copy[field].apply(
                lambda x: int(max(0, float(x))) if pd.notnull(x) and isinstance(x, (int, float)) else x
            )
    
    return df_copy

def fix_scaling_issues(df, feature_columns=None):
    """
    Fix issues related to scaling that may produce negative values.
    This is specifically to handle StandardScaler outputs that may produce
    negative values for features that should be non-negative.
    
    Args:
        df: DataFrame with scaled features
        feature_columns: List of feature columns to check, if None check all
        
    Returns:
        DataFrame with fixed scaled values
    """
    import pandas as pd
    import logging
    
    logger = logging.getLogger(__name__)
    
    if feature_columns is None:
        feature_columns = df.columns.tolist()
    
    df_copy = df.copy()
    
    # Feature constraints that must be preserved after scaling
    non_negative_prefixes = [
        'sleep_duration', 'time_in_bed', 'sleep_onset_latency',
        'awakenings_count', 'total_awake', 'heart_rate'
    ]
    
    zero_to_one_prefixes = [
        'sleep_efficiency', 'deep_sleep', 'rem_sleep', 
        'light_sleep', 'awake_percentage', 'age_normalized',
        'profession_', 'season_'
    ]
    
    # Identify columns that need fixing after scaling
    for col in feature_columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            continue
            
        # Check if column should be non-negative
        is_non_negative = any(col.startswith(prefix) for prefix in non_negative_prefixes)
        
        # Check if column should be between 0 and 1
        is_zero_to_one = any(col.startswith(prefix) for prefix in zero_to_one_prefixes)
        
        # Apply fixes if needed
        if is_non_negative and col in df_copy.columns:
            try:
                negative_values = (df_copy[col] < 0).sum()
                if negative_values > 0:
                    logger.info(f"Fixing {negative_values} negative scaled values in {col}")
                    df_copy[col] = df_copy[col].apply(lambda x: max(0, x) if pd.notnull(x) and isinstance(x, (int, float)) else x)
            except TypeError:
                logger.warning(f"Column {col} contains non-numeric values that cannot be compared with < operator")
        
        if is_zero_to_one and col in df_copy.columns:
            try:
                out_of_bounds = ((df_copy[col] < 0) | (df_copy[col] > 1)).sum()
                if out_of_bounds > 0:
                    logger.info(f"Fixing {out_of_bounds} out-of-bounds scaled values in {col}")
                    df_copy[col] = df_copy[col].apply(lambda x: min(max(0, x), 1.0) if pd.notnull(x) and isinstance(x, (int, float)) else x)
            except TypeError:
                logger.warning(f"Column {col} contains non-numeric values that cannot be compared with < or > operators")
    
    return df_copy

def validate_dataframe_for_model(df, feature_model=None):
    """
    Perform comprehensive validation on a DataFrame to ensure it's ready for model training.
    This combines all validation steps into one function.
    
    Args:
        df: DataFrame to validate
        feature_model: Pydantic model to use for validation (optional)
        
    Returns:
        DataFrame with fixed values
    """
    # First ensure user_ids are clean strings
    if 'user_id' in df.columns:
        df['user_id'] = df['user_id'].astype(str)
        # Remove any fractional parts
        df['user_id'] = df['user_id'].apply(lambda x: x.split('.')[0] if '.' in x else x)
    
    # Fix numeric features to be within valid ranges
    df = fix_feature_ranges(df)
    
    # Fix scaling issues that might create invalid values
    df = fix_scaling_issues(df)
    
    # Clean any NaN/inf values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # If a feature model is provided, validate against it
    if feature_model is not None:
        valid_indices = []
        for i, row in df.iterrows():
            try:
                # Extract fields from row based on feature model fields
                model_fields = set(feature_model.__annotations__.keys())
                row_fields = {k: v for k, v in row.items() if k in model_fields}
                
                # Validate with model
                feature_model(**row_fields)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Row {i} validation error: {str(e)}")
        
        # Keep only valid rows
        if len(valid_indices) < len(df):
            logger.warning(f"Removed {len(df) - len(valid_indices)} invalid rows during validation")
            df = df.loc[valid_indices]
    
    return df


def fix_feature_ranges(df, feature_set_constraints=None):
    """
    Fix feature values to ensure they meet FeatureSet constraints.
    This function explicitly forces values into valid ranges.
    """
    # Define constraints based on FeatureSet if not provided
    if feature_set_constraints is None:
        feature_set_constraints = {
            # Non-negative fields
            'non_negative': [
                'sleep_duration_hours', 'time_in_bed_hours', 'sleep_onset_latency_minutes',
                'awakenings_count', 'total_awake_minutes', 'subjective_rating',
                'sleep_efficiency', 'heart_rate_variability', 'average_heart_rate'
            ],
            # Bounded between 0.0 and 1.0
            'zero_to_one': [
                'sleep_efficiency', 'deep_sleep_percentage', 'rem_sleep_percentage', 
                'light_sleep_percentage', 'awake_percentage', 'age_normalized',
                'profession_healthcare', 'profession_tech', 'profession_service', 
                'profession_education', 'profession_office', 'profession_other',
                'season_Winter', 'season_Spring', 'season_Summer', 'season_Fall'
            ]
        }
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Fix non-negative fields
    for field in feature_set_constraints['non_negative']:
        if field in df_copy.columns:
            df_copy[field] = df_copy[field].apply(
                lambda x: max(0, float(x)) if pd.notnull(x) and isinstance(x, (int, float)) else x
            )
    
    # Fix zero_to_one bounded fields
    for field in feature_set_constraints['zero_to_one']:
        if field in df_copy.columns:
            df_copy[field] = df_copy[field].apply(
                lambda x: min(max(0, float(x)), 1.0) if pd.notnull(x) and isinstance(x, (int, float)) else x
            )
    
    # Convert integer fields appropriately
    integer_fields = ['awakenings_count']
    for field in integer_fields:
        if field in df_copy.columns:
            df_copy[field] = df_copy[field].apply(
                lambda x: int(max(0, float(x))) if pd.notnull(x) and isinstance(x, (int, float)) else x
            )
    
    return df_copy

def fix_feature_types(df):
    """Fix data types to ensure compatibility with Pydantic models"""
    # Fields that should be integers
    integer_fields = ['awakenings_count', 'age']
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Convert integer fields
    for field in integer_fields:
        if field in df_copy.columns:
            # First ensure non-negative
            df_copy[field] = df_copy[field].apply(
                lambda x: max(0, float(x)) if pd.notnull(x) and isinstance(x, (int, float)) else x
            )
            # Then convert to integer
            df_copy[field] = df_copy[field].apply(
                lambda x: int(float(x)) if pd.notnull(x) and isinstance(x, (int, float)) else x
            )
    
    return df_copy

def patch_feature_engineering(feature_engineering_instance):
    """
    Patch a FeatureEngineering instance to use our improved validation.
    
    Args:
        feature_engineering_instance: The instance to patch
        
    Returns:
        Patched instance
    """
    original_scale_features = feature_engineering_instance._scale_features
    
    def patched_scale_features(data, feature_columns):
        # Call original scaling function
        scaled_data = original_scale_features(data, feature_columns)
        
        # Apply our fix to ensure values remain in valid ranges
        fixed_data = fix_scaling_issues(scaled_data, feature_columns)
        
        return fixed_data
    
    # Replace the method with our patched version
    feature_engineering_instance._scale_features = patched_scale_features
    
    return feature_engineering_instance

def ensure_sleep_data_format(data):
    """Ensure sleep data has consistent field names and formats"""
    # Handle alternative field names
    if 'total_awake_minutes' in data.columns and 'time_awake_minutes' not in data.columns:
        data['time_awake_minutes'] = data['total_awake_minutes']
    elif 'time_awake_minutes' in data.columns and 'total_awake_minutes' not in data.columns:
        data['total_awake_minutes'] = data['time_awake_minutes']
    
    # Ensure date fields are strings
    date_fields = ['date', 'bedtime', 'sleep_onset_time', 'wake_time', 'out_bed_time']
    for field in date_fields:
        if field in data.columns and pd.api.types.is_datetime64_dtype(data[field]):
            data[field] = data[field].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return data

def ensure_datetime(date_value):
    """Safely convert a value to datetime, handling both strings and existing timestamps"""
    if date_value is None:
        return None
    if isinstance(date_value, (pd.Timestamp, datetime)):
        return date_value
    try:
        return pd.to_datetime(date_value)
    except:
        return None
    
def safe_parse_datetime(value):
    """Safely parse a datetime value that could be a string or already a datetime"""
    if value is None:
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value
    try:
        return pd.to_datetime(value)
    except:
        return None