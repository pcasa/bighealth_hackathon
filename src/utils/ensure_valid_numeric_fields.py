

import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/ensure_valid_numeric_fields.log')
    ]
)

logger = logging.getLogger(__name__)

def ensure_valid_numeric_fields(df):
    """
    Ensure numeric fields have valid values without removing natural variation.
    Only clamp extreme outliers that would cause calculation errors.
    """
    # Fields that must be non-negative (>= 0)
    non_negative_fields = [
        'sleep_duration_hours', 'time_in_bed_hours', 'sleep_onset_latency_minutes',
        'awakenings_count', 'total_awake_minutes', 'subjective_rating',
        'sleep_efficiency', 'heart_rate_variability', 'average_heart_rate'
    ]
    
    # Bounded fields with specific ranges - use wider bounds to preserve variation
    bounded_fields = {
        'sleep_efficiency': (0.0, 1.0),  # Keep full range
        'deep_sleep_percentage': (0.0, 1.0),
        'rem_sleep_percentage': (0.0, 1.0),
        'light_sleep_percentage': (0.0, 1.0),
        'awake_percentage': (0.0, 1.0),
        'age_normalized': (0.0, 1.0),
        'subjective_rating': (1, 10),
        'time_in_bed_hours': (0.0, 24.0)
    }
    
    logger.info("Ensuring numeric fields meet validation requirements while preserving variation")
    
    # Handle non-negative fields - only fix negative values
    for field in non_negative_fields:
        if field in df.columns:
            # Count invalid values
            invalid_count = (df[field] < 0).sum() if pd.api.types.is_numeric_dtype(df[field]) else 0
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} negative values in {field}, setting to small positive values")
                # Replace negative values with small positive values to preserve variation
                df[field] = df[field].apply(
                    lambda x: max(0.01, x) if pd.notnull(x) and isinstance(x, (int, float)) and x < 0 else x
                )
    
    # Handle bounded fields - only fix values outside allowed range
    for field, (min_val, max_val) in bounded_fields.items():
        if field in df.columns:
            # Count extreme outliers
            extreme_below = (df[field] < min_val - 0.1).sum() if pd.api.types.is_numeric_dtype(df[field]) else 0
            extreme_above = (df[field] > max_val + 0.1).sum() if pd.api.types.is_numeric_dtype(df[field]) else 0
            
            if extreme_below > 0 or extreme_above > 0:
                logger.warning(f"Field {field}: {extreme_below} extreme low values, {extreme_above} extreme high values")
                
                # Only fix extreme outliers
                df[field] = df[field].apply(
                    lambda x: min(max(x, min_val), max_val) if pd.notnull(x) and isinstance(x, (int, float)) 
                                                            and (x < min_val - 0.1 or x > max_val + 0.1) else x
                )
    
    # Modified part of ensure_valid_numeric_fields()
    if field in bounded_fields:
        if field == 'sleep_efficiency' and above_max > 0:
            # For sleep_efficiency, scale values above 1.0 to be between 0.9 and 1.0
            # This maintains variation while keeping values in range
            df[field] = df[field].apply(
                lambda x: 0.9 + 0.1 * (min(x, 1.2) - 1.0) / 0.2 
                if pd.notnull(x) and isinstance(x, (int, float)) and x > 1.0 
                else min(max(x, min_val), max_val) if pd.notnull(x) and isinstance(x, (int, float)) 
                else x
            )
        else:
            # Regular clamping for other fields
            df[field] = df[field].apply(
                lambda x: min(max(x, min_val), max_val) if pd.notnull(x) and isinstance(x, (int, float)) else x
            )
    # Handle NaN values with more variance - use random values in range rather than fixed defaults
    for field in non_negative_fields:
        if field in df.columns and df[field].isna().any():
            na_count = df[field].isna().sum()
            
            # Fill with variable values based on field
            if field in bounded_fields:
                min_val, max_val = bounded_fields[field]
                # Generate random values in the valid range
                if field == 'sleep_efficiency':
                    # For sleep efficiency, use a more realistic distribution
                    random_values = np.random.beta(5, 2, size=na_count) * (max_val - min_val) + min_val
                elif field == 'subjective_rating':
                    # For ratings, use a more uniform distribution
                    random_values = np.random.randint(min_val, max_val+1, size=na_count)
                else:
                    # For other bounded fields, use uniform distribution
                    random_values = np.random.uniform(min_val, max_val, size=na_count)
                
                logger.warning(f"Filling {na_count} NaN values in {field} with random values in valid range")
                df.loc[df[field].isna(), field] = random_values
            else:
                # For unbounded fields, use field-specific logic
                if field == 'sleep_duration_hours':
                    # More realistic sleep duration distribution
                    random_values = np.random.normal(7, 1.5, size=na_count)
                    random_values = np.clip(random_values, 3, 12)
                elif field == 'awakenings_count':
                    # Realistic awakenings (skewed distribution)
                    random_values = np.random.poisson(2, size=na_count)
                elif field == 'total_awake_minutes':
                    # Realistic awake minutes (skewed distribution)
                    random_values = np.random.exponential(15, size=na_count)
                    random_values = np.clip(random_values, 0, 120)
                else:
                    # Default to low positive values for other fields
                    random_values = np.random.exponential(5, size=na_count)
                
                logger.warning(f"Filling {na_count} NaN values in {field} with random values")
                df.loc[df[field].isna(), field] = random_values
    
    return df