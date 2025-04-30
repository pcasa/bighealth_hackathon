

import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ensure_valid_numeric_fields.log')
    ]
)

logger = logging.getLogger(__name__)

def ensure_valid_numeric_fields(df):
    """
    Ensure all numeric fields in the dataframe meet validation requirements:
    - Convert NumPy float types to Python float types
    - Ensure values are within valid ranges
    - Fix problematic values that would cause Pydantic validation errors
    """
    # Fields that must be non-negative (>= 0)
    non_negative_fields = [
        'sleep_duration_hours', 'time_in_bed_hours', 'sleep_onset_latency_minutes',
        'awakenings_count', 'total_awake_minutes', 'subjective_rating',
        'sleep_efficiency', 'deep_sleep_percentage', 'rem_sleep_percentage', 
        'light_sleep_percentage', 'awake_percentage', 'heart_rate_variability',
        'average_heart_rate', 'age', 'age_normalized'
    ]
    
    # Bounded fields with specific ranges
    bounded_fields = {
        'sleep_efficiency': (0.0, 1.0),
        'deep_sleep_percentage': (0.0, 1.0),
        'rem_sleep_percentage': (0.0, 1.0),
        'light_sleep_percentage': (0.0, 1.0),
        'awake_percentage': (0.0, 1.0),
        'age_normalized': (0.0, 1.0),
        'subjective_rating': (1, 10)
    }
    
    # Count handling
    integer_fields = [
        'awakenings_count', 
        'age'
    ]
    
    logger.info("Ensuring numeric fields meet validation requirements")
    
    # First handle NumPy type conversion for all numeric columns
    for col in df.columns:
        if col in non_negative_fields or col in bounded_fields or col in integer_fields or \
           (pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype.kind in 'fcmiu'):
            # Convert NumPy types to Python native types
            try:
                # For non-null values, explicitly convert NumPy types to Python types
                df[col] = df[col].apply(lambda x: float(x) if pd.notnull(x) and hasattr(x, 'item') else x)
            except Exception as e:
                logger.warning(f"Error converting column {col}: {str(e)}")
    
    # Handle non-negative fields
    for field in non_negative_fields:
        if field in df.columns:
            # Count invalid values
            invalid_count = (df[field] < 0).sum() if pd.api.types.is_numeric_dtype(df[field]) else 0
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} negative values in {field}, setting to 0")
                # Replace negative values with 0
                df[field] = df[field].apply(lambda x: max(0, x) if pd.notnull(x) and isinstance(x, (int, float)) else x)
    
    # Handle bounded fields
    for field, (min_val, max_val) in bounded_fields.items():
        if field in df.columns:
            # Count out-of-bounds values
            below_min = (df[field] < min_val).sum() if pd.api.types.is_numeric_dtype(df[field]) else 0
            above_max = (df[field] > max_val).sum() if pd.api.types.is_numeric_dtype(df[field]) else 0
            
            if below_min > 0 or above_max > 0:
                logger.warning(f"Field {field}: {below_min} values below {min_val}, {above_max} values above {max_val}")
                
                # Clip values to valid range
                df[field] = df[field].apply(
                    lambda x: min(max(x, min_val), max_val) if pd.notnull(x) and isinstance(x, (int, float)) else x
                )
    
    # Handle integer fields
    for field in integer_fields:
        if field in df.columns:
            # Count non-integer values
            non_integer_count = 0
            if pd.api.types.is_numeric_dtype(df[field]):
                non_integer_count = df[field].apply(
                    lambda x: x != int(x) if pd.notnull(x) and isinstance(x, (int, float)) else False
                ).sum()
            
            if non_integer_count > 0:
                logger.warning(f"Found {non_integer_count} non-integer values in {field}, converting to integers")
                # Convert to integers
                df[field] = df[field].apply(lambda x: int(x) if pd.notnull(x) and isinstance(x, (int, float)) else x)
    
    # Handle NaN values - fill with valid defaults
    for field in non_negative_fields:
        if field in df.columns and df[field].isna().any():
            # Use appropriate defaults
            if field in bounded_fields:
                min_val, max_val = bounded_fields[field]
                default_val = min_val  # Use minimum valid value as default
            else:
                default_val = 0  # Use 0 as default for unbounded non-negative fields
                
            na_count = df[field].isna().sum()
            logger.warning(f"Filling {na_count} NaN values in {field} with {default_val}")
            df[field] = df[field].fillna(default_val)
    
    return df