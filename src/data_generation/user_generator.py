import numpy as np
import pandas as pd
import yaml
import os
import uuid
from datetime import datetime, timedelta

class UserGenerator:
    def __init__(self, config_path='config/data_generation_config.yaml'):
        print(config_path)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.user_config = self.config['user_profiles']
        self.sleep_patterns = self.config['sleep_patterns']
        
    def generate_users(self, count=None):
        """Generate a specified number of user profiles"""
        if count is None:
            count = self.user_config['count']
        
        users = []
        
        for _ in range(count):
            user = self._generate_single_user()
            users.append(user)
        
        return pd.DataFrame(users)
    
    def _generate_single_user(self):
        """Generate a single user profile"""
        age_range = self.user_config['age_range']
        genders = self.user_config['genders']
        
        # Generate pattern based on distribution
        sleep_pattern_dist = self.user_config['sleep_patterns']
        patterns = list(sleep_pattern_dist.keys())
        weights = list(sleep_pattern_dist.values())
        sleep_pattern = np.random.choice(patterns, p=weights)
        
        # Generate device based on distribution
        device_dist = self.user_config['device_distribution']
        devices = list(device_dist.keys())
        device_weights = list(device_dist.values())
        device = np.random.choice(devices, p=device_weights)
        
        # Generate data consistency level (how often they might skip logging)
        data_consistency = np.random.beta(12, 2)  # Skewed toward higher consistency
        
        # Generate sleep consistency (regularity of sleep patterns)
        if sleep_pattern == 'variable':
            sleep_consistency = np.random.beta(2, 5)  # Lower consistency
        else:
            sleep_consistency = np.random.beta(5, 2)  # Higher consistency
        
        return {
            'user_id': str(uuid.uuid4())[:8],
            'age': np.random.randint(age_range[0], age_range[1] + 1),
            'gender': np.random.choice(genders),
            'sleep_pattern': sleep_pattern,
            'device_type': device,
            'data_consistency': data_consistency,
            'sleep_consistency': sleep_consistency,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }