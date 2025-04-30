#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to update the feature_engineering.py file with improved validation
"""

import os
import shutil
import re
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('patch_feature_engineering.log')
    ]
)

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
    
    # Add import for the data validation fix
    import_pattern = r'from pydantic import BaseModel, Field'
    import_replacement = 'from pydantic import BaseModel, Field\nfrom src.utils.data_validation_fix import fix_feature_ranges, fix_scaling_issues'
    
    content = re.sub(import_pattern, import_replacement, content)
    
    # Update the _scale_features method to include validation
    scale_pattern = r'def _scale_features\(self, data, feature_columns\):.*?return data'
    scale_replacement = '''def _scale_features(self, data, feature_columns):
        """Scale numerical features with improved validation"""
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
            
            # *** NEW VALIDATION STEP ***
            # Fix any negative values created by scaling
            data = fix_scaling_issues(data, numerical_features)
        
        return data'''
    
    content = re.sub(scale_pattern, scale_replacement, content, flags=re.DOTALL)
    
    # Update the create_features method to add validation
    create_pattern = r'# Validate features using Pydantic.*?return feature_data, targets_df'
    create_replacement = '''# Validate features using data validation fix
        feature_data = fix_feature_ranges(feature_data)
        
        # Validate features using Pydantic
        valid_features = []
        invalid_indices = []
        
        for i, row in feature_data.iterrows():
            try:
                feature_dict = {col: row[col] for col in self.feature_columns if col in row}
                # Validate features
                FeatureSet(user_id=row['user_id'], **feature_dict)
                valid_features.append(i)
            except Exception as e:
                print(f"Invalid feature set at index {i}: {e}")
                invalid_indices.append(i)
        
        # Keep only valid features
        if invalid_indices:
            print(f"Removing {len(invalid_indices)} invalid feature sets")
            feature_data = feature_data.loc[valid_features]
            # If we removed too many rows, log a warning
            if len(feature_data) < len(feature_data) * 0.1:  # If we lost more than 90%
                print(f"WARNING: Removed {len(invalid_indices)} of {len(feature_data) + len(invalid_indices)} rows due to validation failures")
        
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
            
            # Validate targets
            targets_df = fix_feature_ranges(targets_df)
            
            # Remove target columns from features
            feature_data = feature_data.drop(['sleep_efficiency'], axis=1, errors='ignore')
        
        return feature_data, targets_df'''
    
    content = re.sub(create_pattern, create_replacement, content, flags=re.DOTALL)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    logger.info(f"Updated {file_path} with validation improvements")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python patch_feature_engineering.py path/to/feature_engineering.py")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        sys.exit(1)
    
    success = patch_file(file_path)
    
    if success:
        print(f"Successfully patched {file_path}")
        print("The original file has been backed up with .bak extension")
    else:
        print(f"Failed to patch {file_path}")