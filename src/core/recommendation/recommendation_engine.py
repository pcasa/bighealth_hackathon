# src/models/recommendation_engine.py

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Union

from src.utils.constants import profession_categories
from src.core.models.data_models import ProgressAnalysis, SleepMetrics, Recommendation



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
        
        # Handle non-sleep nights - CORRECTED CODE HERE
        if 'no_sleep' in recent_data.columns:
            has_sleep_data = recent_data[recent_data['no_sleep'] != True].copy()
            no_sleep_count = len(recent_data) - len(has_sleep_data)
        else:
            has_sleep_data = recent_data.copy()
            no_sleep_count = 0
        
        if len(has_sleep_data) < 2:
            return {
                'trend': 'insufficient_sleep_data',
                'consistency': 0,
                'improvement_rate': 0,
                'key_metrics': {
                    'no_sleep_count': no_sleep_count
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
        key_metrics = SleepMetrics(
            avg_efficiency=np.mean(sleep_efficiency),
            avg_time_in_bed=np.mean(time_in_bed),
            recent_efficiency=sleep_efficiency[-1] if len(sleep_efficiency) > 0 else None,
            recent_time_in_bed=time_in_bed[-1] if len(time_in_bed) > 0 else None,
            best_efficiency=np.max(sleep_efficiency) if len(sleep_efficiency) > 0 else None,
            avg_awakenings=np.mean(awakenings) if awakenings is not None else None,
            avg_time_awake=np.mean(time_awake) if time_awake is not None else None,
            recent_rating=ratings[-1] if ratings is not None and len(ratings) > 0 else None,
            consistent_bedtime=self._check_bedtime_consistency(has_sleep_data)
        )
        
        # Check for no-sleep occurrences
        no_sleep_count = len(recent_data) - len(has_sleep_data)
        if no_sleep_count > 0:
            key_metrics.no_sleep_count = no_sleep_count
            if no_sleep_count >= 3:
                trend = 'severe_insomnia'
            elif no_sleep_count >= 2:
                trend = 'moderate_insomnia'
        
        return ProgressAnalysis(
            trend=trend,
            consistency=consistency,
            improvement_rate=efficiency_trend,
            key_metrics=key_metrics
        )
    
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
    
    def generate_recommendation(self, user_id, progress_data, profession=None, region=None):
        """Generate a personalized recommendation based on progress analysis with profession and region context"""
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

        # Get appropriate message category
        if trend == 'insufficient_data':
            category = 'onboarding'
        
        # Get previously sent messages for this user
        user_history = self.user_message_history.get(user_id, [])
        
        # Find eligible templates that haven't been sent recently
        eligible_templates = self._get_eligible_templates(category, user_history)
        
        if not eligible_templates:
            # Fallback to any template in this category
            eligible_templates = self.message_templates.get(category, [])
            
        if not eligible_templates:
            return {"message": "Keep tracking your sleep for personalized recommendations!", "confidence": 0.3}
        
        # Select a random template from eligible ones
        template = random.choice(eligible_templates)
        
        # Personalize the message
        message = self._personalize_message(template, key_metrics)
        
        # Add profession-specific advice if available
        if profession:
            profession_advice = self._get_profession_advice(profession, trend)
            if profession_advice:
                message += f" {profession_advice}"
        
        # Add region-specific advice if available
        if region:
            region_advice = self._get_region_advice(region, trend)
            if region_advice:
                message += f" {region_advice}"
        
        # Update user history
        self._update_user_history(user_id, category, template)
        
        # Calculate confidence for this recommendation
        # Calculate confidence
        confidence = self.calculate_recommendation_confidence(user_id, progress_data)
        
        return Recommendation(
            message=message,
            confidence=confidence,
            category=category
        )
    

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
        
        # Replace tokens with actual values, handling None values
        replacements = {
            '{avg_efficiency}': f"{metrics.get('avg_efficiency', 0)*100:.0f}%" if metrics.get('avg_efficiency') is not None else "N/A",
            '{avg_time_in_bed}': f"{metrics.get('avg_time_in_bed', 0):.1f} hours" if metrics.get('avg_time_in_bed') is not None else "N/A",
            '{recent_efficiency}': f"{metrics.get('recent_efficiency', 0)*100:.0f}%" if metrics.get('recent_efficiency') is not None else "N/A",
            '{recent_time_in_bed}': f"{metrics.get('recent_time_in_bed', 0):.1f} hours" if metrics.get('recent_time_in_bed') is not None else "N/A",
            '{best_efficiency}': f"{metrics.get('best_efficiency', 0)*100:.0f}%" if metrics.get('best_efficiency') is not None else "N/A",
            '{avg_awakenings}': f"{metrics.get('avg_awakenings', 0):.1f}" if metrics.get('avg_awakenings') is not None else "N/A",
            '{avg_time_awake}': f"{metrics.get('avg_time_awake', 0):.0f} minutes" if metrics.get('avg_time_awake') is not None else "N/A",
            '{recent_rating}': f"{metrics.get('recent_rating', 0):.0f}/10" if metrics.get('recent_rating') is not None else "N/A",
            '{no_sleep_count}': f"{metrics.get('no_sleep_count', 0)}" if metrics.get('no_sleep_count') is not None else "0"
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


    def _get_profession_advice(self, profession, trend):
        """Get profession-specific sleep advice"""
        # Extract profession category from profession string
        profession_category = self._extract_profession_category(profession)
        
        # Define profession-specific advice by category and trend
        profession_advice = {
            'healthcare': {
                'strong_improvement': "Your profession often involves irregular schedules, so maintaining this progress is a significant achievement.",
                'improvement': "As a healthcare professional, consider using blackout curtains to improve sleep during day sleep periods.",
                'regression': "Healthcare work can be stressful. Consider a 10-minute mindfulness practice before bed to help transition to sleep.",
                'stable': "Given the demands of healthcare work, maintaining stable sleep is an accomplishment. Continue your successful routines."
            },
            'tech': {
                'strong_improvement': "You've made excellent progress despite the screen time common in tech roles. Keep limiting blue light before bed.",
                'improvement': "For tech professionals, try using blue light filters on devices after 8pm to help maintain your sleep improvements.",
                'regression': "Tech work often means significant screen time. Try disconnecting from devices at least 1 hour before bedtime.",
                'stable': "For people in tech, balancing screen time with good sleep is key. Consider physical activity to offset sedentary work."
            },
            'service': {
                'strong_improvement': "Service industry roles often have variable schedules. You've done well adapting your sleep routine despite this challenge.",
                'improvement': "Your service role may involve irregular hours. Try to maintain a consistent wind-down routine regardless of when your shift ends.",
                'regression': "Service work can involve stressful interactions. Try a 10-minute decompression ritual after work to mentally separate work stress from sleep time.",
                'stable': "In service roles, maintaining consistent sleep despite variable schedules is commendable. Continue prioritizing your sleep routine."
            },
            'education': {
                'strong_improvement': "You've improved despite the mental demands of educational work. Well done on creating boundaries for better sleep.",
                'improvement': "As an educator, try to finish grading or preparation at least 2 hours before bedtime to give your mind time to wind down.",
                'regression': "Educational work often comes home with you. Set clearer boundaries between work and sleep time to improve your rest.",
                'stable': "You're maintaining good sleep while balancing the demands of education work. Remember to keep work materials out of your sleep space."
            },
            'office': {
                'strong_improvement': "You've made great progress despite the sedentary nature of office work. Physical activity is clearly helping your sleep.",
                'improvement': "Office work can be mentally draining but physically inactive. Consider adding moderate exercise to help deepen your sleep.",
                'regression': "Screen exposure and mental stress from office work may be affecting your sleep. Create a clear transition between work and rest.",
                'stable': "You've found a good balance with your office work schedule. Continue to maintain separation between work and sleep environments."
            },
            'other': {
                'strong_improvement': "You've made excellent progress in your sleep quality. Your work-life balance seems to be improving.",
                'improvement': "Consider how your work schedule affects your sleep timing and create consistent boundaries to protect your rest time.",
                'regression': "Work-related factors might be impacting your sleep lately. Assess if you can create better transitions between work and rest.",
                'stable': "You're maintaining consistent sleep patterns alongside your work schedule. Continue your successful sleep routines."
            }
        }
        
        # Return advice if available for this profession and trend
        if profession_category in profession_advice and trend in profession_advice[profession_category]:
            return profession_advice[profession_category][trend]
        
        # Default to 'other' category if specific advice not available
        if 'other' in profession_advice and trend in profession_advice['other']:
            return profession_advice['other'][trend]
        
        return ""

    def _get_region_advice(self, region, trend):
        """Get region-specific sleep advice"""
        # Extract region category
        region_category = self._extract_region_category(region)
        
        # Define region-specific advice
        region_advice = {
            'north_america': {
                'strong_improvement': "Your sleep has improved well despite the common work-focused culture in your region.",
                'improvement': "In North America, many struggle with work-life balance. Creating clear boundaries around work hours helps protect sleep time.",
                'regression': "In your region, it's common to prioritize work over sleep. Try to resist this cultural pressure for better sleep quality.",
                'stable': "In North American culture, maintaining good sleep boundaries can be challenging but important for long-term health."
            },
            'europe': {
                'strong_improvement': "Your sleep has improved significantly. The later dinner times common in your region don't seem to be affecting you.",
                'improvement': "In many European countries, later dinner times can impact sleep. Try to eat your evening meal at least 3 hours before bedtime.",
                'regression': "The dining culture in your region may be affecting your sleep. Consider adjusting evening meal timing for better sleep.",
                'stable': "You're maintaining consistent sleep despite regional factors. Continue your good sleep habits."
            },
            'asia': {
                'strong_improvement': "You've made significant progress despite urban light pollution common in many Asian cities.",
                'improvement': "In Asian urban areas, consider using room-darkening curtains and white noise to create an optimal sleep environment.",
                'regression': "The high population density in your region might be affecting your sleep. Consider tools to block out noise and light.",
                'stable': "You've adapted well to your regional sleep challenges. Maintaining this pattern is excellent for your health."
            },
            'other': {
                'strong_improvement': "You've made excellent progress adapting your sleep to your regional environment.",
                'improvement': "Consider regional factors like light, noise, and climate that might impact your sleep and address them for better rest.",
                'regression': "Regional environmental factors might be affecting your sleep lately. Consider adjustments to your sleep environment.",
                'stable': "You've established a good sleep routine that works well in your region. Continue your successful practices."
            }
        }
        
        # Return advice if available for this region and trend
        if region_category in region_advice and trend in region_advice[region_category]:
            return region_advice[region_category][trend]
        
        # Default to 'other' category if specific advice not available
        if 'other' in region_advice and trend in region_advice['other']:
            return region_advice['other'][trend]
        
        return ""
    
    def _extract_profession_category(self, profession):
        
        for category, keywords in profession_categories.items():
            if any(keyword.lower() in profession.lower() for keyword in keywords):
                return category
                
        return "other"

    def _extract_region_category(self, region):
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
    
    def calculate_recommendation_confidence(self, user_id, progress_data, user_sleep_data=None):
        """Calculate confidence level for recommendations based on data quality and user factors"""
        # Start with a base confidence
        confidence = 0.6  # Base confidence level (60%)
        
        # Factor 1: Amount of data (more data = higher confidence)
        if user_sleep_data is not None:
            data_points = len(user_sleep_data)
        else:
            # Get an estimate from the user history
            user_history = self.user_message_history.get(user_id, [])
            data_points = len(user_history) * 2  # Rough estimate
        
        if data_points >= 30:
            confidence += 0.15  # High confidence for 30+ days of data
        elif data_points >= 14:
            confidence += 0.1   # Medium confidence for 14-29 days
        elif data_points < 7:
            confidence -= 0.2   # Low confidence for less than 7 days
        
        # Factor 2: Data consistency (how regularly user logs data)
        if 'consistency' in progress_data:
            tracking_consistency = progress_data['consistency']
            # Scale from -0.1 (very inconsistent) to +0.1 (very consistent)
            consistency_factor = (tracking_consistency - 0.5) * 0.2
            confidence += consistency_factor
        
        # Factor 3: Sleep pattern (some patterns are more predictable than others)
        if 'trend' in progress_data:
            if progress_data['trend'] in ['severe_insomnia', 'moderate_insomnia']:
                confidence -= 0.15  # Insomnia patterns are less predictable
            elif progress_data['trend'] in ['strong_improvement', 'improvement']:
                confidence += 0.05  # Clear improvement trends are more reliable
            elif progress_data['trend'] == 'stable':
                confidence += 0.1   # Stable patterns are most predictable
        
        # Factor 4: Key metrics completeness
        if 'key_metrics' in progress_data:
            metrics = progress_data['key_metrics']
            expected_metrics = ['avg_efficiency', 'recent_efficiency', 'avg_awakenings']
            missing_metrics = sum(1 for m in expected_metrics if m not in metrics or metrics[m] is None)
            confidence -= missing_metrics * 0.05  # Reduce confidence for each missing key metric
        
        # Ensure confidence is within valid range [0.2, 0.95]
        confidence = max(0.2, min(0.95, confidence))
        
        # Round to 2 decimal places
        return round(confidence, 2)

    def save_history(self, filepath):
        """Save message history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.user_message_history, f)
    
    def load_history(self, filepath):
        """Load message history from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.user_message_history = json.load(f)