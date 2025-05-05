# src/core/repositories/data_repository.py
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Optional

class DataRepository:
    """Data access layer for sleep and user data"""
    
    def __init__(self, data_dir='data/enhanced_demo/data'):
        self.data_dir = data_dir
        self.cache = {}
        os.makedirs(data_dir, exist_ok=True)
    
    def get_user_data(self, user_id=None):
        """Get user data, optionally filtered by user_id"""
        users_file = os.path.join(self.data_dir, 'users.csv')
        
        if not os.path.exists(users_file):
            columns = ['user_id', 'age', 'gender', 'profession', 'region', 
                     'sleep_pattern', 'device_type', 'data_consistency', 
                     'sleep_consistency', 'created_at']
            return pd.DataFrame(columns=columns)
        
        users_df = pd.read_csv(users_file)
        
        if user_id:
            return users_df[users_df['user_id'] == user_id]
        
        return users_df
    
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
    
    def get_sleep_data(self, user_id=None, days=None):
        """Get sleep data, optionally filtered by user_id and recent days"""
        sleep_file = os.path.join(self.data_dir, 'sleep_data.csv')
        
        if not os.path.exists(sleep_file):
            return pd.DataFrame()
        
        sleep_df = pd.read_csv(sleep_file)
        
        if user_id:
            sleep_df = sleep_df[sleep_df['user_id'] == user_id]
        
        if days:
            sleep_df['date'] = pd.to_datetime(sleep_df['date'])
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            sleep_df = sleep_df[sleep_df['date'] >= cutoff_date]
            
        return sleep_df
    
    def save_user(self, user_data):
        """Save user data"""
        users_file = os.path.join(self.data_dir, 'users.csv')
        users_df = self.get_user_data()
        
        if 'user_id' in user_data and user_data['user_id'] in users_df['user_id'].values:
            # Update existing user
            users_df.loc[users_df['user_id'] == user_data['user_id']] = user_data
        else:
            # Add new user
            users_df = pd.concat([users_df, pd.DataFrame([user_data])], ignore_index=True)
        
        users_df.to_csv(users_file, index=False)
        return user_data
    
    def delete_user(self, user_id):
        """Delete user"""
        users_file = os.path.join(self.data_dir, 'users.csv')
        users_df = self.get_user_data()
        
        if user_id not in users_df['user_id'].values:
            return False
        
        users_df = users_df[users_df['user_id'] != user_id]
        users_df.to_csv(users_file, index=False)
        return True
    
    def save_sleep_entry(self, sleep_entry):
        """Save sleep entry"""
        sleep_file = os.path.join(self.data_dir, 'sleep_data.csv')
        sleep_df = self.get_sleep_data()
        
        # Check for existing entry (same user and date)
        if not sleep_df.empty:
            mask = (sleep_df['user_id'] == sleep_entry['user_id']) & (sleep_df['date'] == sleep_entry['date'])
            if mask.any():
                # Update existing entry
                for key, value in sleep_entry.items():
                    sleep_df.loc[mask, key] = value
            else:
                # Append new entry
                sleep_df = pd.concat([sleep_df, pd.DataFrame([sleep_entry])], ignore_index=True)
        else:
            # Create new DataFrame with this entry
            sleep_df = pd.DataFrame([sleep_entry])
        
        sleep_df.to_csv(sleep_file, index=False)
        return sleep_entry
    
    def save_recommendation(self, user_id, recommendation, timestamp=None):
        """Save recommendation"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        rec_dir = os.path.join(self.data_dir, '../recommendations')
        os.makedirs(rec_dir, exist_ok=True)
        
        rec_data = {
            'user_id': user_id,
            'timestamp': timestamp,
            'message': recommendation['message'] if isinstance(recommendation, dict) else recommendation,
            'confidence': recommendation.get('confidence', 0.5) if isinstance(recommendation, dict) else 0.5
        }
        
        # Save to user-specific file
        user_rec_file = os.path.join(rec_dir, f"{user_id}_recommendations.csv")
        
        if os.path.exists(user_rec_file):
            rec_df = pd.read_csv(user_rec_file)
            rec_df = pd.concat([rec_df, pd.DataFrame([rec_data])], ignore_index=True)
        else:
            rec_df = pd.DataFrame([rec_data])
            
        rec_df.to_csv(user_rec_file, index=False)
        
        # Also save to combined file
        all_rec_file = os.path.join(rec_dir, "all_recommendations.csv")
        
        if os.path.exists(all_rec_file):
            all_rec_df = pd.read_csv(all_rec_file)
            all_rec_df = pd.concat([all_rec_df, pd.DataFrame([rec_data])], ignore_index=True)
        else:
            all_rec_df = pd.DataFrame([rec_data])
            
        all_rec_df.to_csv(all_rec_file, index=False)
        
        return rec_data