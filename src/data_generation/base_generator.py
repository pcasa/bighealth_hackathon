# src/data_generation/base_generator.py

import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class BaseDataGenerator:
    """Base class for all data generators with common functionality"""
    
    def __init__(self, config_path='config/data_generation_config.yaml'):
        """Initialize the base generator with configuration"""
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def generate_time_based_noise(self, base_value, variance_pct=0.1):
        """Generate noisy values with specified variance percentage"""
        noise = np.random.normal(0, base_value * variance_pct)
        return base_value + noise
    
    def get_season_from_date(self, date):
        """Determine season from a date (Northern Hemisphere)"""
        month = date.month
        if 3 <= month <= 5:
            return 'spring'
        elif 6 <= month <= 8:
            return 'summer'
        elif 9 <= month <= 11:
            return 'fall'
        else:
            return 'winter'
            
    def apply_modifiers(self, base_params, modifiers):
        """Apply modifiers to base parameters"""
        if not modifiers:
            return base_params
            
        modified_params = base_params.copy()
        
        for key, value in modifiers.items():
            if isinstance(value, list) and key in modified_params and isinstance(modified_params[key], list):
                # For list parameters like ranges
                modified_params[key] = value
            elif isinstance(value, (int, float)) and key in modified_params:
                if isinstance(modified_params[key], list) and len(modified_params[key]) == 2:
                    # Adjust range values
                    modified_params[key] = [val + value for val in modified_params[key]]
                else:
                    # Direct replacement
                    modified_params[key] = value
        
        return modified_params
        
    def get_category_from_keywords(self, text, category_dict):
        """Extract category from text based on keywords"""
        if not isinstance(text, str):
            return "other"
            
        for category, keywords in category_dict.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                return category
        
        return "other"
    
    def extract_region_category(self, region):
        """Extract region category from region string"""
        if not isinstance(region, str) or ',' not in region:
            return "other"
            
        parts = region.split(',')
        country = parts[-1].strip()
        
        north_america = ['United States', 'Canada', 'Mexico', 'USA']
        europe = ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'UK']
        asia = ['China', 'Japan', 'India', 'Korea', 'Thailand', 'Singapore']
        
        if country in north_america:
            return "north_america"
        elif country in europe:
            return "europe"
        elif country in asia:
            return "asia"
        else:
            return "other"
    
    def create_date_range(self, start_date, end_date):
        """Create a range of dates between start and end"""
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        return pd.date_range(start=start_date, end=end_date)