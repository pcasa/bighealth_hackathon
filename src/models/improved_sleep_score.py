class ImprovedSleepScoreCalculator:
    """
    An improved sleep score calculator that uses all basic sleep metrics and
    provides a more balanced scoring system.
    """
    
    def __init__(self):
        """Initialize the sleep score calculator with default parameters"""
        # Define ideal ranges for sleep metrics
        self.ideal_ranges = {
            'sleep_duration_hours': (7.0, 9.0),  # Recommended by sleep experts
            'sleep_efficiency': (0.85, 0.95),    # Percentage of time in bed actually sleeping
            'sleep_onset_latency_minutes': (5, 20),  # Time to fall asleep
            'awakenings_count': (0, 2),          # Number of awakenings
            'total_awake_minutes': (0, 20),      # Total time awake during night
            'bedtime_hour': (21, 23),            # Ideal bedtime (9pm-11pm)
            'waketime_hour': (6, 8)              # Ideal wake time (6am-8am)
        }
        
        # Define component weights (must sum to 1.0)
        self.component_weights = {
            'duration': 0.20,             # Sleep duration
            'efficiency': 0.15,           # Sleep efficiency
            'onset': 0.15,                # Sleep onset latency
            'continuity': 0.20,           # Sleep continuity (awakenings + time awake)
            'timing': 0.10,               # Sleep timing (bedtime and waketime)
            'subjective': 0.20            # Subjective rating
        }
    
    def calculate_score(self, sleep_data, include_details=False):
        """
        Calculate a comprehensive sleep score based on a variety of metrics.
        
        Args:
            sleep_data: Dict containing sleep metrics
            include_details: If True, return component scores along with total
            
        Returns:
            int or dict: Sleep score (0-100) or dict with scores and components
        """
        component_scores = {}
        
        # 1. Duration component (0-100)
        component_scores['duration'] = self._score_duration(sleep_data)
        
        # 2. Efficiency component (0-100)
        component_scores['efficiency'] = self._score_efficiency(sleep_data)
        
        # 3. Onset component (0-100)
        component_scores['onset'] = self._score_onset(sleep_data)
        
        # 4. Continuity component (0-100)
        component_scores['continuity'] = self._score_continuity(sleep_data)
        
        # 5. Timing component (0-100)
        component_scores['timing'] = self._score_timing(sleep_data)
        
        # 6. Subjective component (0-100)
        component_scores['subjective'] = self._score_subjective(sleep_data)
        
        # Handle special cases (no sleep)
        if sleep_data.get('no_sleep', False):
            # If no sleep, score is 0
            final_score = 0
        else:
            # Calculate weighted average
            final_score = self._calculate_weighted_score(component_scores)
        
        # Round to integer
        final_score = int(round(final_score))
        
        if include_details:
            return {
                'total_score': final_score,
                'component_scores': component_scores
            }
        else:
            return final_score
    
    def _calculate_weighted_score(self, component_scores):
        """Calculate weighted average of component scores"""
        # Initialize weighted score
        weighted_score = 0
        total_weight = 0
        
        # Add each available component
        for component, weight in self.component_weights.items():
            if component in component_scores and component_scores[component] is not None:
                weighted_score += component_scores[component] * weight
                total_weight += weight
        
        # Normalize if not all components are available
        if total_weight > 0:
            weighted_score = weighted_score / total_weight * 100
        else:
            weighted_score = 50  # Default score if no components available
        
        return max(0, min(100, weighted_score))
    
    def _score_duration(self, sleep_data):
        """Score sleep duration component"""
        # Check if duration is available
        if 'sleep_duration_hours' not in sleep_data:
            return None
            
        duration = sleep_data['sleep_duration_hours']
        min_ideal, max_ideal = self.ideal_ranges['sleep_duration_hours']
        
        # Score based on how close to ideal range
        if duration < min_ideal:
            # Too little sleep - penalize more severely
            # Score decreases from 70 (at min_ideal) down to 0
            return max(0, 70 * duration / min_ideal)
        elif duration <= max_ideal:
            # Ideal range - full score
            return 100
        else:
            # Too much sleep - gentle penalty
            # Score decreases from 100 to 70 over next 2 hours
            return max(70, 100 - 15 * (duration - max_ideal))
    
    def _score_efficiency(self, sleep_data):
        """Score sleep efficiency component"""
        # Check if efficiency is available
        if 'sleep_efficiency' not in sleep_data:
            # Try to calculate it if we have duration and time in bed
            if 'sleep_duration_hours' in sleep_data and 'time_in_bed_hours' in sleep_data:
                efficiency = sleep_data['sleep_duration_hours'] / sleep_data['time_in_bed_hours']
            else:
                return None
        else:
            efficiency = sleep_data['sleep_efficiency']
        
        min_ideal, max_ideal = self.ideal_ranges['sleep_efficiency']
        
        # Ensure value is between 0 and 1
        efficiency = max(0, min(1, efficiency))
        
        # Score based on how close to ideal range
        if efficiency < min_ideal:
            # Below ideal - score decreases linearly to 0
            return 100 * efficiency / min_ideal
        elif efficiency <= max_ideal:
            # Ideal range - full score
            return 100
        else:
            # Above ideal - small penalty (might indicate too little time in bed)
            return max(80, 100 - 200 * (efficiency - max_ideal))
    
    def _score_onset(self, sleep_data):
        """Score sleep onset component"""
        # Check if onset latency is available
        if 'sleep_onset_latency_minutes' not in sleep_data:
            return None
            
        latency = sleep_data['sleep_onset_latency_minutes']
        min_ideal, max_ideal = self.ideal_ranges['sleep_onset_latency_minutes']
        
        # Score based on how close to ideal range
        if latency < min_ideal:
            # Too quick - may indicate exhaustion
            return 80
        elif latency <= max_ideal:
            # Ideal range - full score
            return 100
        elif latency <= 30:
            # Slightly delayed - mild penalty
            return 90 - (latency - max_ideal) * 1
        elif latency <= 60:
            # Moderately delayed - steeper penalty
            return 80 - (latency - 30) * 1.5
        else:
            # Severely delayed - largest penalty
            return max(0, 35 - (latency - 60) * 0.35)
    
    def _score_continuity(self, sleep_data):
        """Score sleep continuity component (awakenings and time awake)"""
        awakenings_score = None
        awake_time_score = None
        
        # Check if awakenings count is available
        if 'awakenings_count' in sleep_data:
            awakenings = sleep_data['awakenings_count']
            min_ideal, max_ideal = self.ideal_ranges['awakenings_count']
            
            # Score based on awakenings
            if awakenings <= max_ideal:
                # Ideal range - full score
                awakenings_score = 100
            else:
                # Penalize each additional awakening
                awakenings_score = max(0, 100 - 10 * (awakenings - max_ideal))
        
        # Check if time awake is available
        if 'total_awake_minutes' in sleep_data:
            awake_time = sleep_data['total_awake_minutes']
            min_ideal, max_ideal = self.ideal_ranges['total_awake_minutes']
            
            # Score based on time awake
            if awake_time <= max_ideal:
                # Ideal range - full score
                awake_time_score = 100
            elif awake_time <= 45:
                # Mild disruption
                awake_time_score = 90 - (awake_time - max_ideal) * 0.8
            else:
                # Severe disruption
                awake_time_score = max(0, 70 - (awake_time - 45) * 0.7)
        
        # Combine scores if both are available, otherwise use the available one
        if awakenings_score is not None and awake_time_score is not None:
            # Weight time awake more heavily than count
            return 0.4 * awakenings_score + 0.6 * awake_time_score
        elif awakenings_score is not None:
            return awakenings_score
        elif awake_time_score is not None:
            return awake_time_score
        else:
            return None
    
    def _score_timing(self, sleep_data):
        """Score sleep timing component (bedtime and waketime)"""
        bedtime_score = None
        waketime_score = None
        
        # Check if bedtime is available
        if 'bedtime' in sleep_data:
            # Extract hour from datetime
            if hasattr(sleep_data['bedtime'], 'hour'):
                bedtime_hour = sleep_data['bedtime'].hour
            else:
                # Try to parse from string
                try:
                    from datetime import datetime
                    dt = datetime.strptime(sleep_data['bedtime'], '%Y-%m-%d %H:%M:%S')
                    bedtime_hour = dt.hour
                except:
                    bedtime_hour = None
            
            if bedtime_hour is not None:
                # Handle after midnight (early hours are actually late)
                if bedtime_hour < 5:
                    bedtime_hour += 24
                
                min_ideal, max_ideal = self.ideal_ranges['bedtime_hour']
                
                # Score based on how close to ideal range
                if bedtime_hour < min_ideal:
                    # Too early
                    bedtime_score = 70 + 15 * (bedtime_hour / min_ideal)
                elif bedtime_hour <= max_ideal:
                    # Ideal range
                    bedtime_score = 100
                elif bedtime_hour <= max_ideal + 2:
                    # Slightly late
                    bedtime_score = 100 - 15 * (bedtime_hour - max_ideal) / 2
                else:
                    # Very late
                    bedtime_score = max(40, 85 - 15 * (bedtime_hour - (max_ideal + 2)))
        
        # Check if wake time is available
        if 'wake_time' in sleep_data:
            # Extract hour from datetime
            if hasattr(sleep_data['wake_time'], 'hour'):
                waketime_hour = sleep_data['wake_time'].hour
            else:
                # Try to parse from string
                try:
                    from datetime import datetime
                    dt = datetime.strptime(sleep_data['wake_time'], '%Y-%m-%d %H:%M:%S')
                    waketime_hour = dt.hour
                except:
                    waketime_hour = None
            
            if waketime_hour is not None:
                min_ideal, max_ideal = self.ideal_ranges['waketime_hour']
                
                # Score based on how close to ideal range
                if waketime_hour < min_ideal - 1:
                    # Very early
                    waketime_score = max(40, 70 * waketime_hour / (min_ideal - 1))
                elif waketime_hour < min_ideal:
                    # Slightly early
                    waketime_score = 80 + 20 * (waketime_hour - (min_ideal - 1))
                elif waketime_hour <= max_ideal:
                    # Ideal range
                    waketime_score = 100
                elif waketime_hour <= max_ideal + 2:
                    # Slightly late
                    waketime_score = 100 - 15 * (waketime_hour - max_ideal) / 2
                else:
                    # Very late
                    waketime_score = max(40, 85 - 15 * (waketime_hour - (max_ideal + 2)))
        
        # Combine scores if both are available, otherwise use the available one
        if bedtime_score is not None and waketime_score is not None:
            return (bedtime_score + waketime_score) / 2
        elif bedtime_score is not None:
            return bedtime_score
        elif waketime_score is not None:
            return waketime_score
        else:
            return None
    
    def _score_subjective(self, sleep_data):
        """Score subjective component based on user rating"""
        # Check if subjective rating is available
        if 'subjective_rating' not in sleep_data:
            return None
            
        rating = sleep_data['subjective_rating']
        
        # Convert rating to 0-100 scale 
        # Assuming rating is on 1-10 scale
        if 1 <= rating <= 10:
            return (rating - 1) / 9 * 100
        # Assuming rating is already on 0-100 scale
        elif 0 <= rating <= 100:
            return rating
        else:
            return None
    
    def transform_wearable_data(self, wearable_data):
        """
        Transform wearable device data to standard format for sleep score calculation.
        
        Args:
            wearable_data: Dict containing wearable metrics
            
        Returns:
            dict: Standardized sleep data
        """
        sleep_data = {}
        
        # Copy over user_id and date if available
        if 'user_id' in wearable_data:
            sleep_data['user_id'] = wearable_data['user_id']
        if 'date' in wearable_data:
            sleep_data['date'] = wearable_data['date']
        
        # Map standard fields
        field_mappings = {
            'device_sleep_duration': 'sleep_duration_hours',
            'sleep_efficiency': 'sleep_efficiency',
            'device_bedtime': 'bedtime',
            'device_sleep_onset': 'sleep_onset_time',
            'device_wake_time': 'wake_time',
            'awakenings_count': 'awakenings_count'
        }
        
        for source, target in field_mappings.items():
            if source in wearable_data and wearable_data[source] is not None:
                sleep_data[target] = wearable_data[source]
        
        # Calculate total awake time if not directly available
        if 'total_awake_minutes' not in sleep_data:
            if 'deep_sleep_percentage' in wearable_data and 'rem_sleep_percentage' in wearable_data:
                # Calculate from sleep stages
                deep = wearable_data['deep_sleep_percentage']
                rem = wearable_data['rem_sleep_percentage']
                light = wearable_data.get('light_sleep_percentage', 1 - deep - rem)
                awake = wearable_data.get('awake_percentage', 0)
                
                if 'device_sleep_duration' in wearable_data:
                    # Convert percentage to minutes
                    sleep_data['total_awake_minutes'] = awake * wearable_data['device_sleep_duration'] * 60
            
            elif 'awakenings_count' in sleep_data:
                # Estimate based on typical awakening duration
                sleep_data['total_awake_minutes'] = sleep_data['awakenings_count'] * 5
        
        # Calculate sleep onset latency if not directly available
        if 'sleep_onset_latency_minutes' not in sleep_data and 'bedtime' in sleep_data and 'sleep_onset_time' in sleep_data:
            # Extract times from datetime objects or strings
            try:
                from datetime import datetime
                if isinstance(sleep_data['bedtime'], str):
                    bedtime = datetime.strptime(sleep_data['bedtime'], '%Y-%m-%d %H:%M:%S')
                else:
                    bedtime = sleep_data['bedtime']
                    
                if isinstance(sleep_data['sleep_onset_time'], str):
                    sleep_onset = datetime.strptime(sleep_data['sleep_onset_time'], '%Y-%m-%d %H:%M:%S')
                else:
                    sleep_onset = sleep_data['sleep_onset_time']
                
                # Calculate latency in minutes
                latency = (sleep_onset - bedtime).total_seconds() / 60
                sleep_data['sleep_onset_latency_minutes'] = max(0, latency)
            except:
                pass
        
        # Include additional metrics that might be useful
        additional_metrics = ['deep_sleep_percentage', 'rem_sleep_percentage', 
                             'light_sleep_percentage', 'heart_rate_variability']
        
        for metric in additional_metrics:
            if metric in wearable_data and wearable_data[metric] is not None:
                sleep_data[metric] = wearable_data[metric]
        
        return sleep_data

# Usage example:
if __name__ == "__main__":
    # Sample user-reported sleep data
    user_data = {
        'user_id': 'user123',
        'date': '2025-04-27',
        'bedtime': '2025-04-26 23:00:00',
        'sleep_onset_time': '2025-04-26 23:25:00',
        'wake_time': '2025-04-27 07:00:00',
        'out_bed_time': '2025-04-27 07:15:00',
        'sleep_duration_hours': 7.5,
        'time_in_bed_hours': 8.25,
        'sleep_efficiency': 0.91,
        'sleep_onset_latency_minutes': 25,
        'awakenings_count': 2,
        'total_awake_minutes': 10,
        'subjective_rating': 8
    }
    
    # Sample wearable data
    wearable_data = {
        'user_id': 'user123',
        'date': '2025-04-27',
        'device_type': 'apple_watch',
        'device_bedtime': '2025-04-26 23:05:00',
        'device_sleep_onset': '2025-04-26 23:30:00',
        'device_wake_time': '2025-04-27 06:55:00',
        'device_sleep_duration': 7.4,
        'deep_sleep_percentage': 0.22,
        'rem_sleep_percentage': 0.23,
        'light_sleep_percentage': 0.5,
        'awake_percentage': 0.05,
        'awakenings_count': 3,
        'heart_rate_variability': 45
    }
    
    # Initialize calculator
    calculator = ImprovedSleepScoreCalculator()
    
    # Calculate score for user-reported data
    user_score_details = calculator.calculate_score(user_data, include_details=True)
    print(f"User-reported sleep score: {user_score_details['total_score']}")
    print("Component scores:")
    for component, score in user_score_details['component_scores'].items():
        print(f"  {component}: {score}")
    
    # Transform wearable data and calculate score
    transformed_wearable_data = calculator.transform_wearable_data(wearable_data)
    wearable_score = calculator.calculate_score(transformed_wearable_data)
    print(f"\nWearable-based sleep score: {wearable_score}")