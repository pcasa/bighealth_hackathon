# Update src/core/data/repository.py
from datetime import datetime, timedelta
import os

import pandas as pd


class DataRepository:
    """Data access layer for sleep data"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.cache = {}
        # Set data directory to enhanced demo data
        self.data_dir = 'data/enhanced_demo/data'
    
    def get_user_data(self, user_id=None):
        """Get user data, optionally filtered by user_id"""
        # Implement data loading from CSV files in data/enhanced_demo/data directory
        try:
            users_file = 'data/enhanced_demo/data/users.csv'
            if os.path.exists(users_file):
                data = pd.read_csv(users_file)
                if user_id is not None:
                    data = data[data['user_id'] == user_id]
                return data
            else:
                return pd.DataFrame()  # Return empty DataFrame if file doesn't exist
        except Exception as e:
            print(f"Error loading user data: {str(e)}")
            return pd.DataFrame()  # Ret
    
    def get_sleep_data(self, user_id=None, days=None):
        """Get sleep data, optionally filtered by user_id and recent days"""
        # Implementation for sleep data
        sleep_file = os.path.join(self.data_dir, 'sleep_data.csv')
        if not os.path.exists(sleep_file):
            return pd.DataFrame()
            
        sleep_df = pd.read_csv(sleep_file)
        
        # Convert date column to datetime
        if 'date' in sleep_df.columns:
            sleep_df['date'] = pd.to_datetime(sleep_df['date'])
        
        # Filter by user_id if provided
        if user_id:
            sleep_df = sleep_df[sleep_df['user_id'] == user_id]
        
        # Filter by recent days if provided
        if days is not None:
            cutoff_date = datetime.now() - timedelta(days=days)
            sleep_df = sleep_df[sleep_df['date'] >= cutoff_date]
        
        return sleep_df
    
    # Add other methods as needed (save_user, save_sleep_entry, etc.)

    def get_user_profile(self, user_id):
        """Get user profile data by user_id"""
        # Check if in cache
        cache_key = f"user_profile_{user_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get all user data
        user_data = self.get_user_data()
        
        # Filter to specific user
        if user_data is not None and len(user_data) > 0:
            user_profile = user_data[user_data['user_id'] == user_id]
            if len(user_profile) > 0:
                result = user_profile.iloc[0].to_dict()
                # Cache result
                self.cache[cache_key] = result
                return result
        
        # Return empty dict if user not found
        return {}