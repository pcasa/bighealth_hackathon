import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
    
    def create_features(self, data, include_wearable=None, include_external=None):
        """Create features for model training"""
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
        
        # Time features
        if self.config['include_time_features']:
            feature_data = self._add_time_features(feature_data)
            time_features = [
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'day_of_week_sin', 'day_of_week_cos', 'is_weekend'
            ]
        else:
            time_features = []
        
        # Wearable features
        wearable_features = []
        if include_wearable and 'device_sleep_duration' in feature_data.columns:
            wearable_features = [
                'device_sleep_duration', 'deep_sleep_percentage', 'light_sleep_percentage',
                'rem_sleep_percentage', 'awake_percentage', 'average_heart_rate',
                'min_heart_rate', 'max_heart_rate', 'heart_rate_variability',
                'sleep_cycles', 'movement_intensity'
            ]
            
            # Add discrepancy features only if they exist
            if 'bedtime_discrepancy_minutes' in feature_data.columns:
                discrepancy_features = [
                    'bedtime_discrepancy_minutes', 'sleep_onset_discrepancy_minutes',
                    'wake_time_discrepancy_minutes', 'duration_discrepancy_hours'
                ]
                wearable_features.extend(discrepancy_features)
        
        # External factor features
        external_features = []
        if include_external:
            # Weather features if available
            if 'temperature' in feature_data.columns:
                weather_features = ['temperature', 'humidity', 'pressure', 'precipitation']
                external_features.extend(weather_features)
            
            # Activity features if available
            if 'steps' in feature_data.columns:
                activity_features = ['steps', 'active_minutes', 'stress_level']
                external_features.extend(activity_features)
        
        # Combine all feature columns
        feature_columns = basic_features + consistency_features + time_features + wearable_features + external_features
        feature_columns = [col for col in feature_columns if col in feature_data.columns]
        
        # Check for and handle missing columns
        for col in feature_columns:
            if col not in feature_data.columns:
                feature_columns.remove(col)
        
        # Scale numerical features
        feature_data = self._scale_features(feature_data, feature_columns)
        
        # Add categorical features
        categorical_method = self.config['categorical_encoding']
        if categorical_method == 'one-hot':
            if 'sleep_pattern' in feature_data.columns:
                feature_data = self._encode_categorical(feature_data, 'sleep_pattern')
                cat_cols = [col for col in feature_data.columns if col.startswith('sleep_pattern_')]
                feature_columns.extend(cat_cols)
            
            if 'device_type' in feature_data.columns:
                feature_data = self._encode_categorical(feature_data, 'device_type')
                cat_cols = [col for col in feature_data.columns if col.startswith('device_type_')]
                feature_columns.extend(cat_cols)
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        # Separate targets from features
        
        targets_df = pd.DataFrame()
    
        # Check if we have the necessary columns for targets
        required_columns = ['sleep_efficiency']
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