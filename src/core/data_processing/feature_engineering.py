import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Dict, Optional
from pydantic import BaseModel, Field

from src.utils.data_validation_fix import ensure_sleep_data_format

class FeatureSet(BaseModel):
    """Model to validate engineered features"""
    # User identification
    user_id: str
    
    # Temporal features
    date: Optional[str] = None
    hour_sin: Optional[float] = None
    hour_cos: Optional[float] = None
    month_sin: Optional[float] = None
    month_cos: Optional[float] = None
    day_of_week_sin: Optional[float] = None
    day_of_week_cos: Optional[float] = None
    is_weekend: Optional[bool] = None
    
    # Sleep metrics
    sleep_efficiency: Optional[float] = Field(None, ge=0.0, le=1.0)
    sleep_duration_hours: Optional[float] = Field(None, ge=0.0, le=24.0)
    sleep_onset_latency_minutes: Optional[float] = Field(None, ge=0.0)
    awakenings_count: Optional[int] = Field(None, ge=0)
    total_awake_minutes: Optional[float] = Field(None, ge=0.0)
    
    # Demographic features
    age_normalized: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # One-hot encoded professions
    profession_healthcare: Optional[float] = Field(None, ge=0.0, le=1.0)
    profession_tech: Optional[float] = Field(None, ge=0.0, le=1.0)
    profession_service: Optional[float] = Field(None, ge=0.0, le=1.0)
    profession_education: Optional[float] = Field(None, ge=0.0, le=1.0)
    profession_office: Optional[float] = Field(None, ge=0.0, le=1.0)
    profession_other: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # One-hot encoded seasons
    season_Winter: Optional[float] = Field(None, ge=0.0, le=1.0)
    season_Spring: Optional[float] = Field(None, ge=0.0, le=1.0)
    season_Summer: Optional[float] = Field(None, ge=0.0, le=1.0)
    season_Fall: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Wearable features
    deep_sleep_percentage: Optional[float] = Field(None, ge=0.0, le=1.0)
    rem_sleep_percentage: Optional[float] = Field(None, ge=0.0, le=1.0)
    light_sleep_percentage: Optional[float] = Field(None, ge=0.0, le=1.0)
    awake_percentage: Optional[float] = Field(None, ge=0.0, le=1.0)
    heart_rate_variability: Optional[float] = None
    average_heart_rate: Optional[float] = None
    
    # Weather and external factors
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    precipitation: Optional[float] = None

class FeatureEngineering:
    def __init__(self, config=None):
        """Initialize the feature engineering class"""
        self.config = config or {}
        
        # Set defaults for missing config values
        self.config.setdefault('scaling_method', 'standard')
        self.config.setdefault('include_time_features', True)
        self.config.setdefault('include_wearable_features', True)
        self.config.setdefault('include_external_features', True)
        self.config.setdefault('categorical_encoding', 'one-hot')
        self.config.setdefault('handle_outliers', True)
        
        # Initialize scalers based on config
        if self.config['scaling_method'] == 'standard':
            self.scaler = StandardScaler()
        elif self.config['scaling_method'] == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()  # Default to standard
            
        self.categorical_encoders = {}
        self.feature_columns = []


    def _fix_feature_types(self, feature_dict):
        """Fix data types to ensure compatibility with Pydantic models"""
        # Make a copy to avoid modifying the original
        fixed_dict = feature_dict.copy()
        
        # Convert awakenings_count from float to int
        if 'awakenings_count' in fixed_dict and isinstance(fixed_dict['awakenings_count'], float):
            fixed_dict['awakenings_count'] = int(fixed_dict['awakenings_count'])

        # Convert is_weekend from float to bool
        if 'is_weekend' in fixed_dict and not isinstance(fixed_dict['is_weekend'], bool):
            # If it's a number, consider positive values as True, 0 or negative as False
            if isinstance(fixed_dict['is_weekend'], (int, float, np.number)):
                fixed_dict['is_weekend'] = bool(fixed_dict['is_weekend'] > 0)
            # If it's a string, use string conversion
            elif isinstance(fixed_dict['is_weekend'], str):
                fixed_dict['is_weekend'] = fixed_dict['is_weekend'].lower() in ('true', 't', 'yes', 'y', '1')
        
        
        # Convert any other fields that need type conversion here
        
        return fixed_dict
    
    # Update the create_features method in the FeatureEngineering class
    def create_features(self, data, include_wearable=None, include_external=None):
        """Create features for model training with improvements for handling missing features"""
        feature_data = data.copy()
        
        # Override defaults with provided parameters if given
        include_wearable = include_wearable if include_wearable is not None else self.config['include_wearable_features']
        include_external = include_external if include_external is not None else self.config['include_external_features']
        
        # Basic sleep features
        basic_features = [
            'sleep_duration_hours', 'sleep_efficiency', 'time_in_bed_hours',
            'sleep_onset_latency_minutes', 'awakenings_count', 'total_awake_minutes',
            'subjective_rating'
        ]
        
        # Consistency features
        consistency_features = [
            'bedtime_consistency', 'waketime_consistency', 'duration_consistency',
        ]
        
        # Handle demographic features
        demographic_features = []
        
        # Age normalization
        if 'age' in feature_data.columns and 'age_normalized' not in feature_data.columns:
            feature_data['age_normalized'] = feature_data['age'] / 100.0
            demographic_features.append('age_normalized')
        elif 'age_normalized' in feature_data.columns:
            demographic_features.append('age_normalized')
        
        # Profession one-hot encoding
        if 'profession_category' in feature_data.columns:
            profession_categories = ['healthcare', 'tech', 'service', 'education', 'office', 'other']
            for category in profession_categories:
                feature_name = f'profession_{category}'
                if feature_name not in feature_data.columns:
                    feature_data[feature_name] = (feature_data['profession_category'] == category).astype(float)
                demographic_features.append(feature_name)
        
        # Time features
        if self.config['include_time_features']:
            feature_data = self._add_time_features(feature_data)
            time_features = [
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'day_of_week_sin', 'day_of_week_cos', 'is_weekend'
            ]
        else:
            time_features = []
        
        # Season features
        if 'date' in feature_data.columns:
            if 'month' not in feature_data.columns:
                feature_data['month'] = feature_data['date'].dt.month
            
            if 'season' not in feature_data.columns:
                feature_data['season'] = feature_data['month'].apply(self._get_season)
            
            # Add seasonal one-hot encoding
            seasons = ['Winter', 'Spring', 'Summer', 'Fall']
            for season in seasons:
                feature_name = f'season_{season}'
                if feature_name not in feature_data.columns:
                    feature_data[feature_name] = (feature_data['season'] == season).astype(float)
                demographic_features.append(feature_name)
        
        # Wearable features
        wearable_features = []
        if include_wearable:
            # Check for common wearable data fields
            possible_wearable_features = [
                'deep_sleep_percentage', 'light_sleep_percentage', 'rem_sleep_percentage', 
                'awake_percentage', 'heart_rate_variability', 'average_heart_rate',
                'min_heart_rate', 'max_heart_rate', 'blood_oxygen', 'respiratory_rate'
            ]
            
            # Use only features actually present in the data
            wearable_features = [f for f in possible_wearable_features if f in feature_data.columns]
            
            # Add derived wearable features
            if len(wearable_features) > 0:
                derived_features = self._create_wearable_derived_features(feature_data)
                for feat in derived_features.columns:
                    if feat not in feature_data.columns:
                        feature_data[feat] = derived_features[feat]
                        wearable_features.append(feat)
        
        # External factor features
        external_features = []
        if include_external:
            # Weather features if available
            if 'temperature' in feature_data.columns:
                weather_features = ['temperature', 'humidity', 'pressure', 'precipitation']
                external_features.extend([f for f in weather_features if f in feature_data.columns])
            
            # Activity features if available
            if 'steps' in feature_data.columns:
                activity_features = ['steps', 'active_minutes', 'stress_level']
                external_features.extend
        
        # Combine all feature columns
        feature_columns = basic_features + consistency_features + time_features
        feature_columns += demographic_features + wearable_features + external_features
        feature_columns = [col for col in feature_columns if col in feature_data.columns]

        # Handle missing features that we expect to have
        expected_features = self.config.get('expected_features', [])
        for col in expected_features:
            if col not in feature_columns and col not in feature_data.columns:
                print(f"Warning: Expected feature '{col}' is missing. Creating dummy feature.")
                if 'normalized' in col:
                    feature_data[col] = 0.5  # Default normalized value
                elif col.startswith('profession_'):
                    feature_data[col] = 0.0  # Default for one-hot encoding
                elif col.startswith('season_'):
                    feature_data[col] = 0.0  # Default for one-hot encoding
                else:
                    feature_data[col] = 0.0  # Default value
                
                feature_columns.append(col)
        
        feature_data = ensure_sleep_data_format(feature_data)
        
        # Scale numerical features
        feature_data = self._scale_features(feature_data, feature_columns)
        
        # Handle NaN values
        for col in feature_columns:
            if feature_data[col].isna().any():
                print(f"Filling {feature_data[col].isna().sum()} NaN values in column {col}")
                if feature_data[col].dtype == float or feature_data[col].dtype == int:
                    feature_data[col] = feature_data[col].fillna(0.0)
                else:
                    feature_data[col] = feature_data[col].fillna(feature_data[col].mode().iloc[0] if not feature_data[col].mode().empty else "unknown")
        
        # Store feature columns for later use
        self.feature_columns = feature_columns

        # Validate features using Pydantic
        valid_features = []
        invalid_indices = []
        
        for i, row in feature_data.iterrows():
            try:
                # Extract features
                feature_dict = {col: row[col] for col in self.feature_columns if col in row}
                
                # Fix data types before validation
                fixed_feature_dict = self._fix_feature_types(feature_dict)
                
                # Validate features
                FeatureSet(user_id=row['user_id'], **fixed_feature_dict)
                valid_features.append(i)
            except Exception as e:
                print(f"Invalid feature set at index {i}: {e}")
                invalid_indices.append(i)
        
        # Keep only valid features
        if invalid_indices:
            print(f"Removing {len(invalid_indices)} invalid feature sets")
            feature_data = feature_data.loc[valid_features]
        
        # Separate targets from features
        targets_df = pd.DataFrame()
        
        # Check if we have the necessary columns for targets
        id_columns = []
        
        # Check which ID columns are available
        if 'user_id' in feature_data.columns:
            id_columns.append('user_id')
        
        if 'date' in feature_data.columns:
            id_columns.append('date')
        
        # Create targets if we have the necessary data
        if 'sleep_efficiency' in feature_data.columns:
            # Initialize targets with available ID columns
            targets_dict = {}
            for col in id_columns:
                targets_dict[col] = feature_data[col]
            
            # Add target value
            targets_dict['sleep_quality'] = feature_data['sleep_efficiency']
            
            # Create targets dataframe
            targets_df = pd.DataFrame(targets_dict)
            
            # Remove target columns from features
            feature_data = feature_data.drop(['sleep_efficiency'], axis=1, errors='ignore')
        
        return feature_data, targets_df
    
    def _create_wearable_derived_features(self, data):
        """Create derived features from wearable data"""
        derived_features = pd.DataFrame(index=data.index)
        
        # HRV quality score (scaled 0-1)
        if 'heart_rate_variability' in data.columns:
            hrv = data['heart_rate_variability']
            # HRV varies by age, but generally 20-60ms is normal range
            derived_features['hrv_quality'] = hrv.apply(
                lambda x: min(1.0, max(0.0, (x - 15) / 45)) if not pd.isna(x) else np.nan
            )
        
        # Sleep stage balance - measure of how balanced sleep stages are
        if all(col in data.columns for col in ['deep_sleep_percentage', 'rem_sleep_percentage']):
            # Ideal: ~20% deep sleep, ~25% REM
            deep = data['deep_sleep_percentage']
            rem = data['rem_sleep_percentage']
            
            ideal_deep = 0.20
            ideal_rem = 0.25
            
            deep_diff = (deep - ideal_deep).abs()
            rem_diff = (rem - ideal_rem).abs()
            
            # Combine into a single score (0-1, higher is better)
            derived_features['sleep_stage_balance'] = 1.0 - (deep_diff + rem_diff)
            derived_features['sleep_stage_balance'] = derived_features['sleep_stage_balance'].clip(0, 1)
        
        # Heart rate dip (how much lower night HR is compared to typical daytime)
        if 'average_heart_rate' in data.columns:
            # Assume typical daytime HR is around 70-80 bpm
            typical_day_hr = 75
            night_hr = data['average_heart_rate']
            
            # Calculate dip percentage (normally 10-20% is healthy)
            derived_features['hr_dip_percentage'] = (typical_day_hr - night_hr) / typical_day_hr
            derived_features['hr_dip_percentage'] = derived_features['hr_dip_percentage'].clip(0, 0.5)
        
        # Blood oxygen quality (if available)
        if 'blood_oxygen' in data.columns:
            spo2 = data['blood_oxygen']
            # Normal SpO2 is 95-100%
            derived_features['blood_oxygen_quality'] = spo2.apply(
                lambda x: min(1.0, max(0.0, (x - 90) / 10)) if not pd.isna(x) else np.nan
            )
        
        return derived_features

    def _get_season(self, month):
        """Determine season from month (Northern Hemisphere)"""
        if isinstance(month, pd.Series):
            return month.apply(self._get_season)
        
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # 9, 10, 11
            return 'Fall'
    
    def _add_time_features(self, data):
        """Add cyclical time features"""
        # Hour of day from bedtime (cyclical)
        if 'bedtime' in data.columns:
            data['hour'] = data['bedtime'].dt.hour + data['bedtime'].dt.minute / 60
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        # Month of year (cyclical)
        if 'date' in data.columns:
            data['month'] = data['date'].dt.month
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            
            # Day of week (cyclical)
            data['day_of_week'] = data['date'].dt.dayofweek
            data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        return data
    
    def _scale_features(self, data, feature_columns):
        """Scale numerical features"""
        numerical_features = [col for col in feature_columns if col in data.columns]
        
        if numerical_features and len(data) > 0:
            # Save original values
            data_orig = data.copy()
            
            # Handle outliers if configured
            if self.config.get('handle_outliers', True):
                for col in numerical_features:
                    if col in data.columns:
                        # Simple outlier capping at 3 standard deviations
                        mean = data[col].mean()
                        std = data[col].std()
                        if not pd.isna(std) and std > 0:
                            lower_bound = mean - 3 * std
                            upper_bound = mean + 3 * std
                            data[col] = data[col].clip(lower_bound, upper_bound)
            
            # Fit scaler on data
            self.scaler.fit(data[numerical_features])
            
            # Transform data
            data[numerical_features] = self.scaler.transform(data[numerical_features])
            
            # Handle infinities that might have been created
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaNs with original means
            for col in numerical_features:
                if data[col].isna().any():
                    mean_val = data_orig[col].mean()
                    data[col] = data[col].fillna(mean_val)
        
        return data
    
    def _encode_categorical(self, data, column):
        """One-hot encode a categorical column"""
        if column in data.columns:
            # Create one-hot encoding
            dummies = pd.get_dummies(data[column], prefix=column)
            
            # Store categories for future use
            self.categorical_encoders[column] = dummies.columns.tolist()
            
            # Join with original data
            data = pd.concat([data, dummies], axis=1)
        
        return data
    
    def transform_new_data(self, new_data):
        """Transform new data using the same transformations as training data"""
        transformed_data = new_data.copy()
        
        # Add time features if configured
        if self.config['include_time_features']:
            transformed_data = self._add_time_features(transformed_data)
        
        # Scale numerical features using fitted scaler
        numerical_features = [col for col in self.feature_columns 
                            if col in transformed_data.columns and 
                                not col.startswith('sleep_pattern_') and 
                                not col.startswith('device_type_')]
        
        if numerical_features:
            transformed_data[numerical_features] = self.scaler.transform(transformed_data[numerical_features])
        
        # Handle categorical features
        for cat_col, encoder_cols in self.categorical_encoders.items():
            if cat_col in transformed_data.columns:
                # Create one-hot encoding
                dummies = pd.get_dummies(transformed_data[cat_col], prefix=cat_col)
                
                # Make sure all expected columns are present
                for col in encoder_cols:
                    if col not in dummies.columns:
                        dummies[col] = 0
                
                # Remove any extra columns
                dummies = dummies[encoder_cols]
                
                # Join with transformed data
                transformed_data = pd.concat([transformed_data, dummies], axis=1)
        
        return transformed_data