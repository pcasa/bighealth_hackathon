# src/models/recommendation_engine.py

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os

class SleepRecommendationEngine:
    def __init__(self, config_path='config/recommendations_config.yaml'):
        """Initialize the recommendation engine with templates and selection logic"""
        self.config = self._load_config(config_path)
        self.message_templates = self._load_message_templates()
        self.user_message_history = {}  # Track messages sent to each user
        
    def _load_config(self, config_path):
        """Load recommendation configuration"""
        import yaml
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _load_message_templates(self):
        """Load message templates from configuration"""
        templates_path = self.config.get('templates_path', 'config/message_templates.json')
        with open(templates_path, 'r') as file:
            return json.load(file)
    
    def analyze_progress(self, user_id, sleep_data, window=14):
        """Analyze user's sleep progress over a specified window of days"""
        if len(sleep_data) < 3:  # Need at least 3 data points for trend analysis
            return {
                'trend': 'insufficient_data',
                'consistency': 0,
                'improvement_rate': 0,
                'key_metrics': {}
            }
            
        # Sort by date
        sorted_data = sleep_data.sort_values('date')
        
        # Limit to window size
        recent_data = sorted_data.tail(window)
        
        # Handle non-sleep nights
        has_sleep_data = recent_data[~recent_data.get('no_sleep', False)]
        
        if len(has_sleep_data) < 2:
            return {
                'trend': 'insufficient_sleep_data',
                'consistency': 0,
                'improvement_rate': 0,
                'key_metrics': {
                    'no_sleep_count': len(recent_data) - len(has_sleep_data)
                }
            }
        
        # Calculate sleep efficiency if not present
        if 'sleep_efficiency' not in has_sleep_data.columns:
            # Calculate from time in bed and estimated sleep time
            has_sleep_data['sleep_efficiency'] = self._calculate_sleep_efficiency(has_sleep_data)
        
        # Calculate key metrics
        sleep_efficiency = has_sleep_data['sleep_efficiency'].values
        
        # Calculate time in bed if not present
        if 'time_in_bed_hours' not in has_sleep_data.columns:
            has_sleep_data['time_in_bed_hours'] = self._calculate_time_in_bed(has_sleep_data)
        
        time_in_bed = has_sleep_data['time_in_bed_hours'].values
        
        # Use awakenings and time awake
        awakenings = has_sleep_data['awakenings_count'].values if 'awakenings_count' in has_sleep_data.columns else None
        time_awake = has_sleep_data['time_awake_minutes'].values if 'time_awake_minutes' in has_sleep_data.columns else None
        
        # Calculate subjective ratings
        ratings = has_sleep_data['subjective_rating'].values if 'subjective_rating' in has_sleep_data.columns else None
        
        # Calculate trends (simple linear regression)
        efficiency_trend = self._calculate_trend(sleep_efficiency)
        time_in_bed_trend = self._calculate_trend(time_in_bed)
        ratings_trend = self._calculate_trend(ratings) if ratings is not None else 0
        
        # Calculate consistency
        days_logged = len(recent_data)
        expected_days = min(window, (sorted_data['date'].max() - sorted_data['date'].min()).days + 1)
        consistency = days_logged / expected_days if expected_days > 0 else 0
        
        # Determine overall trend
        if efficiency_trend > 0.01 and ratings_trend > 0.1:
            trend = 'strong_improvement'
        elif efficiency_trend > 0 and ratings_trend > 0:
            trend = 'improvement'
        elif efficiency_trend < -0.01 and ratings_trend < -0.1:
            trend = 'regression'
        elif efficiency_trend < 0 or ratings_trend < 0:
            trend = 'slight_regression'
        else:
            trend = 'stable'
            
        # Key metrics to reference in recommendations
        key_metrics = {
            'avg_efficiency': np.mean(sleep_efficiency),
            'avg_time_in_bed': np.mean(time_in_bed),
            'recent_efficiency': sleep_efficiency[-1] if len(sleep_efficiency) > 0 else None,
            'recent_time_in_bed': time_in_bed[-1] if len(time_in_bed) > 0 else None,
            'best_efficiency': np.max(sleep_efficiency) if len(sleep_efficiency) > 0 else None,
            'avg_awakenings': np.mean(awakenings) if awakenings is not None else None,
            'avg_time_awake': np.mean(time_awake) if time_awake is not None else None,
            'recent_rating': ratings[-1] if ratings is not None and len(ratings) > 0 else None,
            'consistent_bedtime': self._check_bedtime_consistency(has_sleep_data)
        }
        
        # Check for no-sleep occurrences
        no_sleep_count = len(recent_data) - len(has_sleep_data)
        if no_sleep_count > 0:
            key_metrics['no_sleep_count'] = no_sleep_count
            if no_sleep_count >= 3:
                trend = 'severe_insomnia'
            elif no_sleep_count >= 2:
                trend = 'moderate_insomnia'
        
        return {
            'trend': trend,
            'consistency': consistency,
            'improvement_rate': efficiency_trend,
            'key_metrics': key_metrics
        }
    
    def _calculate_sleep_efficiency(self, sleep_data):
        """Calculate sleep efficiency from available metrics"""
        # If we have sleep duration and time in bed
        if 'sleep_duration_hours' in sleep_data.columns and 'time_in_bed_hours' in sleep_data.columns:
            return sleep_data['sleep_duration_hours'] / sleep_data['time_in_bed_hours']
        
        # If we have bedtime, sleep time, wake time and out of bed time
        elif all(col in sleep_data.columns for col in ['bedtime', 'sleep_time', 'wake_time', 'out_bed_time']):
            # Convert to datetime if needed
            for col in ['bedtime', 'sleep_time', 'wake_time', 'out_bed_time']:
                if not pd.api.types.is_datetime64_dtype(sleep_data[col]):
                    sleep_data[col] = pd.to_datetime(sleep_data[col])
            
            # Calculate time in bed and estimated sleep time
            sleep_data['time_in_bed'] = (sleep_data['out_bed_time'] - sleep_data['bedtime']).dt.total_seconds() / 3600
            sleep_data['sleep_time_est'] = (sleep_data['wake_time'] - sleep_data['sleep_time']).dt.total_seconds() / 3600
            
            # Subtract time awake during night if available
            if 'time_awake_minutes' in sleep_data.columns:
                sleep_data['sleep_time_est'] -= sleep_data['time_awake_minutes'] / 60
            
            return sleep_data['sleep_time_est'] / sleep_data['time_in_bed']
        
        # Fallback - estimate from subjective rating
        elif 'subjective_rating' in sleep_data.columns:
            # Simple estimation - assumes rating is 1-10 scale
            return sleep_data['subjective_rating'] / 10
        
        # No data available
        return np.ones(len(sleep_data)) * 0.7  # Default to moderate efficiency
    
    def _calculate_time_in_bed(self, sleep_data):
        """Calculate time in bed from timestamps"""
        # If we already have it
        if 'time_in_bed_hours' in sleep_data.columns:
            return sleep_data['time_in_bed_hours']
        
        # Calculate from bedtime and out of bed time
        if 'bedtime' in sleep_data.columns and 'out_bed_time' in sleep_data.columns:
            for col in ['bedtime', 'out_bed_time']:
                if not pd.api.types.is_datetime64_dtype(sleep_data[col]):
                    sleep_data[col] = pd.to_datetime(sleep_data[col])
            
            return (sleep_data['out_bed_time'] - sleep_data['bedtime']).dt.total_seconds() / 3600
        
        # Default fallback
        return np.ones(len(sleep_data)) * 8  # Default to 8 hours
    
    def _calculate_trend(self, values):
        """Calculate the slope of a trend line for the given values"""
        if values is None or len(values) < 2:
            return 0
            
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0]
    
    def _check_bedtime_consistency(self, sleep_data):
        """Check if bedtime is consistent"""
        if 'bedtime' not in sleep_data.columns:
            return False
            
        if not pd.api.types.is_datetime64_dtype(sleep_data['bedtime']):
            sleep_data['bedtime'] = pd.to_datetime(sleep_data['bedtime'])
            
        bedtime_hours = sleep_data['bedtime'].dt.hour + sleep_data['bedtime'].dt.minute / 60
        return np.std(bedtime_hours) < 1.0  # Less than 1 hour standard deviation
    
    def generate_recommendation(self, user_id, progress_data):
        """Generate a personalized recommendation based on progress analysis"""
        trend = progress_data['trend']
        consistency = progress_data['consistency']
        key_metrics = progress_data['key_metrics']
        
        # Get appropriate message category
        if trend == 'insufficient_data':
            category = 'onboarding'
        elif trend == 'severe_insomnia':
            category = 'severe_insomnia'
        elif trend == 'moderate_insomnia':
            category = 'moderate_insomnia'
        elif trend == 'strong_improvement':
            category = 'strong_encouragement'
        elif trend == 'improvement':
            category = 'encouragement'
        elif trend == 'regression':
            category = 'support'
        elif trend == 'slight_regression':
            category = 'gentle_reminder'
        else:  # stable
            category = 'maintenance'
            
        # Adjust for consistency
        if consistency < 0.7 and trend not in ['insufficient_data', 'severe_insomnia', 'moderate_insomnia']:
            category = 'consistency_reminder'
        
        # Get previously sent messages for this user
        user_history = self.user_message_history.get(user_id, [])
        
        # Find eligible templates that haven't been sent recently
        eligible_templates = self._get_eligible_templates(category, user_history)
        
        if not eligible_templates:
            # Fallback to any template in this category
            eligible_templates = self.message_templates.get(category, [])
            
        if not eligible_templates:
            return "Keep tracking your sleep for personalized recommendations!"
        
        # Select a random template from eligible ones
        template = random.choice(eligible_templates)
        
        # Personalize the message
        message = self._personalize_message(template, key_metrics)
        
        # Update user history
        self._update_user_history(user_id, category, template)
        
        return message
    
    def _get_eligible_templates(self, category, user_history, recency=7):
        """Get templates that haven't been used recently"""
        all_templates = self.message_templates.get(category, [])
        
        if not user_history:
            return all_templates
            
        # Filter out recently used templates
        recent_templates = [h['template'] for h in user_history[-recency:]]
        return [t for t in all_templates if t not in recent_templates]
    
    def _personalize_message(self, template, metrics):
        """Fill in template with personalized metrics"""
        message = template
        
        # Replace tokens with actual values
        replacements = {
            '{avg_efficiency}': f"{metrics.get('avg_efficiency', 0)*100:.0f}%",
            '{avg_time_in_bed}': f"{metrics.get('avg_time_in_bed', 0):.1f} hours",
            '{recent_efficiency}': f"{metrics.get('recent_efficiency', 0)*100:.0f}%",
            '{recent_time_in_bed}': f"{metrics.get('recent_time_in_bed', 0):.1f} hours",
            '{best_efficiency}': f"{metrics.get('best_efficiency', 0)*100:.0f}%",
            '{avg_awakenings}': f"{metrics.get('avg_awakenings', 0):.1f}",
            '{avg_time_awake}': f"{metrics.get('avg_time_awake', 0):.0f} minutes",
            '{recent_rating}': f"{metrics.get('recent_rating', 0):.0f}/10",
            '{no_sleep_count}': f"{metrics.get('no_sleep_count', 0)}"
        }
        
        for token, value in replacements.items():
            message = message.replace(token, value)
            
        return message
    
    def _update_user_history(self, user_id, category, template):
        """Update the history of messages sent to this user"""
        if user_id not in self.user_message_history:
            self.user_message_history[user_id] = []
            
        self.user_message_history[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'template': template
        })
        
        # Trim history if needed
        if len(self.user_message_history[user_id]) > 30:
            self.user_message_history[user_id] = self.user_message_history[user_id][-30:]
    
    def save_history(self, filepath):
        """Save message history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.user_message_history, f)
    
    def load_history(self, filepath):
        """Load message history from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.user_message_history = json.load(f)