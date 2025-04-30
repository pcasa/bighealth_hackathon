# src/data_generation/external_factors_generator.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.data_generation.base_generator import BaseDataGenerator

class ExternalFactorsGenerator(BaseDataGenerator):
    def __init__(self, config_path='src/config/data_generation_config.yaml'):
        """Initialize the external factors generator"""
        super().__init__(config_path)
        
        # Weather patterns (temperature in Fahrenheit, humidity, pressure)
        self.weather_patterns = {
            'winter': {'temp_range': [20, 50], 'humidity_range': [30, 70], 'pressure_range': [990, 1020]},
            'spring': {'temp_range': [40, 70], 'humidity_range': [40, 80], 'pressure_range': [1000, 1025]},
            'summer': {'temp_range': [60, 95], 'humidity_range': [50, 90], 'pressure_range': [1005, 1020]},
            'fall': {'temp_range': [35, 75], 'humidity_range': [40, 75], 'pressure_range': [995, 1015]}
        }
    
    def generate_weather_data(self, start_date, end_date):
        """Generate daily weather data for a time period"""
        date_range = self.create_date_range(start_date, end_date)
        weather_data = []
        
        for date in date_range:
            # Determine season
            season = self.get_season_from_date(date)
            
            # Get patterns for this season
            patterns = self.weather_patterns[season]
            
            # Generate weather values
            temp = np.random.uniform(patterns['temp_range'][0], patterns['temp_range'][1])
            humidity = np.random.uniform(patterns['humidity_range'][0], patterns['humidity_range'][1])
            pressure = np.random.uniform(patterns['pressure_range'][0], patterns['pressure_range'][1])
            
            # Add some continuity with previous day (if available)
            if weather_data:
                prev_temp = weather_data[-1]['temperature']
                prev_humidity = weather_data[-1]['humidity']
                prev_pressure = weather_data[-1]['pressure']
                
                # Blend with previous day (70% new, 30% continuation)
                temp = 0.7 * temp + 0.3 * prev_temp
                humidity = 0.7 * humidity + 0.3 * prev_humidity
                pressure = 0.7 * pressure + 0.3 * prev_pressure
            
            # Random weather events
            precip_prob = 0.3 if humidity > 70 else 0.1
            precipitation = np.random.choice([0, np.random.uniform(0.1, 25.0)], p=[1-precip_prob, precip_prob])
            
            weather_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'temperature': round(temp, 1),
                'humidity': round(humidity, 1),
                'pressure': round(pressure, 1),
                'precipitation': round(precipitation, 1),
                'season': season
            })
        
        return pd.DataFrame(weather_data)
    
    def generate_activity_data(self, sleep_data_df, users_df):
        """Generate daily activity data that correlates with sleep patterns"""
        activity_data = []
        
        # Merge sleep data with users to get user profiles
        merged_data = pd.merge(sleep_data_df, users_df[['user_id', 'sleep_pattern']], on='user_id')
        
        for _, record in merged_data.iterrows():
            # Base steps on sleep pattern and quality
            sleep_pattern = record['sleep_pattern']
            sleep_quality = record['sleep_efficiency']
            
            # Generate activity metrics
            steps, active_minutes, stress_level = self._generate_activity_metrics(
                sleep_pattern, sleep_quality, record['subjective_rating']
            )
            
            activity_data.append({
                'user_id': record['user_id'],
                'date': record['date'],
                'steps': steps,
                'active_minutes': active_minutes,
                'stress_level': stress_level,
                'correlated_sleep_efficiency': sleep_quality,
                'correlated_sleep_rating': record['subjective_rating']
            })
        
        return pd.DataFrame(activity_data)
    
    def _generate_activity_metrics(self, sleep_pattern, sleep_efficiency, sleep_rating):
        """Generate activity metrics based on sleep pattern and quality"""
        # Base step ranges by pattern
        if sleep_pattern == 'insomnia':
            base_steps_range = [2000, 8000]  # Less active
        elif sleep_pattern == 'oversleeper':
            base_steps_range = [3000, 7000]  # Less active
        elif sleep_pattern == 'shift_worker':
            base_steps_range = [4000, 10000]  # Moderate activity
        else:
            base_steps_range = [5000, 12000]  # Normal activity range
        
        # Adjust based on sleep quality
        quality_factor = 0.7 + sleep_efficiency * 0.6  # 0.7-1.3 range
        steps_min = base_steps_range[0] * quality_factor
        steps_max = base_steps_range[1] * quality_factor
        
        # Generate steps with randomness
        steps = int(np.random.uniform(steps_min, steps_max))
        
        # Active minutes typically 20-60 for every 10K steps
        active_minutes = int(steps / 10000 * np.random.uniform(200, 600))
        
        # Stress level (1-10) inversely related to sleep quality
        base_stress = 10 - sleep_rating  # Invert sleep rating (1-10)
        stress_noise = np.random.normal(0, 1)
        stress_level = max(1, min(10, round(base_stress + stress_noise)))
        
        return steps, active_minutes, stress_level