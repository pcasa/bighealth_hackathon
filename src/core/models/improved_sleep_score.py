import pandas as pd
from pydantic import BaseModel, Field, model_validator, validator
from typing import Dict, Optional, List, Union
from datetime import datetime

class SleepScoreInput(BaseModel):
    """Input model for sleep score calculation"""
    # Basic metrics
    sleep_efficiency: Optional[float] = Field(None, ge=0.0, le=1.0)
    sleep_duration_hours: Optional[float] = Field(None, ge=0.0, le=24.0) 
    time_in_bed_hours: Optional[float] = Field(None, ge=0.0, le=24.0)
    
    # Sleep quality metrics
    sleep_onset_latency_minutes: Optional[float] = Field(None, ge=0.0)
    awakenings_count: Optional[int] = Field(None, ge=0)
    total_awake_minutes: Optional[float] = Field(None, ge=0.0)
    time_awake_minutes: Optional[float] = Field(None, ge=0.0)  # Alternative name
    subjective_rating: Optional[float] = Field(None, ge=0.0, le=10.0)
    
    # Timestamps
    bedtime: Optional[Union[str, datetime]] = None
    sleep_onset_time: Optional[Union[str, datetime]] = None
    wake_time: Optional[Union[str, datetime]] = None
    out_bed_time: Optional[Union[str, datetime]] = None
    
    # Sleep stage data
    deep_sleep_percentage: Optional[float] = Field(None, ge=0.0, le=1.0)
    rem_sleep_percentage: Optional[float] = Field(None, ge=0.0, le=1.0)
    light_sleep_percentage: Optional[float] = Field(None, ge=0.0, le=1.0)
    awake_percentage: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Special cases
    no_sleep: bool = False
    
    # Demographic data for adjustments
    age: Optional[int] = Field(None, ge=0)
    profession_category: Optional[str] = None
    
    @model_validator(mode='before')
    def calculate_missing_fields(cls, values):
        """Calculate missing fields where possible"""
        # If we have sleep_duration_hours and time_in_bed_hours but not efficiency
        if isinstance(values, dict):  # Ensure values is a dict
            if values.get('sleep_efficiency') is None and values.get('sleep_duration_hours') is not None and values.get('time_in_bed_hours') is not None:
                if values['time_in_bed_hours'] > 0:
                    values['sleep_efficiency'] = min(1.0, values['sleep_duration_hours'] / values['time_in_bed_hours'])
            
            # Handle alternative names
            if values.get('total_awake_minutes') is None and values.get('time_awake_minutes') is not None:
                values['total_awake_minutes'] = values['time_awake_minutes']
        
        return values


class ComponentScore(BaseModel):
    """Individual component score"""
    score: float = Field(..., ge=0.0, le=100.0)
    description: Optional[str] = None
    raw_value: Optional[float] = None


class SleepScoreOutput(BaseModel):
    """Output model for sleep score calculation"""
    total_score: int = Field(..., ge=0, le=100)
    base_score: int = Field(..., ge=0, le=100)
    component_scores: Dict[str, ComponentScore]
    demographic_adjustment: float
    adjustment_reasons: List[str] = []

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
        # Validate input data with the Pydantic model
        try:
            # If dict, validate directly
            if isinstance(sleep_data, dict):
                input_data = SleepScoreInput(**sleep_data)
            # If DataFrame row or similar, convert to dict first
            elif hasattr(sleep_data, 'to_dict'):
                input_data = SleepScoreInput(**sleep_data.to_dict())
            else:
                # Try direct conversion
                input_data = SleepScoreInput(**sleep_data)
        except Exception as e:
            print(f"Error validating sleep data: {e}")
            # Return a default score or raise
            if include_details:
                return SleepScoreOutput(
                    total_score=50,
                    base_score=50,
                    component_scores={},
                    demographic_adjustment=0,
                    adjustment_reasons=["Data validation failed"]
                )
            return 50
        
        component_scores = {}
        
        # 1. Duration component (0-100)
        component_scores['duration'] = self._score_duration(input_data)
        
        # 2. Efficiency component (0-100)
        component_scores['efficiency'] = self._score_efficiency(input_data)
        
        # 3. Onset component (0-100)
        component_scores['onset'] = self._score_onset(input_data)
        
        # 4. Continuity component (0-100)
        component_scores['continuity'] = self._score_continuity(input_data)
        
        # 5. Timing component (0-100)
        component_scores['timing'] = self._score_timing(input_data)
        
        # 6. Subjective component (0-100)
        component_scores['subjective'] = self._score_subjective(input_data)
        
        # Handle special cases (no sleep)
        if input_data.no_sleep:
            # If no sleep, score is 0
            final_score = 0
        else:
            # Calculate weighted average
            final_score = self._calculate_weighted_score(component_scores)


        # Convert component scores to ComponentScore objects
        detailed_components = {}
        for component, score in component_scores.items():
            if score is not None:
                detailed_components[component] = ComponentScore(
                    score=score,
                    description=self._get_component_description(component, score)
                )
        
        # Round to integer
        final_score = int(round(final_score))
        
        if include_details:
            # Create output model
            output = SleepScoreOutput(
                total_score=final_score,
                base_score=final_score,  # Before demographic adjustments
                component_scores=detailed_components,
                demographic_adjustment=0,  # Default, could be adjusted later
                adjustment_reasons=[]
            )
            return output
        else:
            return final_score
        
    def _get_component_description(self, component, score):
        """Get a description of the component score"""
        descriptions = {
            'duration': {
                'excellent': "Optimal sleep duration of 7-9 hours",
                'good': "Good sleep duration slightly outside optimal range",
                'fair': "Fair sleep duration - significantly shorter or longer than optimal",
                'poor': "Poor sleep duration - far from recommended range"
            },
            'efficiency': {
                'excellent': "Excellent sleep efficiency (time asleep / time in bed)",
                'good': "Good sleep efficiency with minimal time awake",
                'fair': "Fair sleep efficiency with moderate time awake",
                'poor': "Poor sleep efficiency with significant time awake"
            },
            'onset': {
                'excellent': "Fall asleep quickly within optimal range",
                'good': "Good sleep onset time, slightly delayed",
                'fair': "Moderately delayed sleep onset",
                'poor': "Significantly delayed sleep onset"
            },
            'continuity': {
                'excellent': "Minimal awakenings and disruptions",
                'good': "Few brief awakenings",
                'fair': "Several awakenings or extended wake time",
                'poor': "Highly fragmented sleep with many awakenings"
            },
            'timing': {
                'excellent': "Optimal sleep timing aligned with circadian rhythm",
                'good': "Good sleep timing slightly off optimal window",
                'fair': "Sleep timing moderately misaligned",
                'poor': "Sleep timing significantly misaligned with natural rhythm"
            },
            'subjective': {
                'excellent': "Excellent subjective sleep experience",
                'good': "Good subjective sleep quality",
                'fair': "Fair subjective sleep quality",
                'poor': "Poor subjective sleep experience"
            }
        }
        
        # Determine category based on score
        if score >= 90:
            category = 'excellent'
        elif score >= 70:
            category = 'good'
        elif score >= 50:
            category = 'fair'
        else:
            category = 'poor'
        
        return descriptions[component][category]

    def _calculate_weighted_score(self, component_scores):
        """Calculate weighted average of component scores"""
        weighted_score = 0
        total_weight = 0
        
        for component, weight in self.component_weights.items():
            if component in component_scores and component_scores[component] is not None:
                component_contribution = component_scores[component] * weight
                weighted_score += component_contribution
                total_weight += weight
        
        
        # Normalize if not all components are available
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 50  # Default score
        
        return max(0, min(100, final_score))
    
    def _score_duration(self, sleep_data):
        """Score sleep duration component"""
        # Check if duration is available
        if sleep_data.sleep_duration_hours is None:
            return None
            
        duration = sleep_data.sleep_duration_hours
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
        if sleep_data.sleep_efficiency is None:
            # Try to calculate it if we have duration and time in bed
            if sleep_data.sleep_duration_hours is not None and sleep_data.time_in_bed_hours is not None:
                efficiency = sleep_data.sleep_duration_hours / sleep_data.time_in_bed_hours
            else:
                return None
        else:
            efficiency = sleep_data.sleep_efficiency
        
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
        # Debug
        latency = None
        
        if hasattr(sleep_data, 'sleep_onset_latency_minutes'):
            latency = sleep_data.sleep_onset_latency_minutes
        elif isinstance(sleep_data, dict) and 'sleep_onset_latency_minutes' in sleep_data:
            latency = sleep_data['sleep_onset_latency_minutes']
        else:
            print("DEBUG: No sleep_onset_latency_minutes found")


        """Score sleep onset component"""
        # Check if onset latency is available - fix the attribute access for dict input
        latency = getattr(sleep_data, 'sleep_onset_latency_minutes', None) if hasattr(sleep_data, 'sleep_onset_latency_minutes') else sleep_data.get('sleep_onset_latency_minutes')
        
        if latency is None:
            return None
            
        # Apply more rigorous scoring
        min_ideal, max_ideal = self.ideal_ranges['sleep_onset_latency_minutes']
        
        # More penalty for very short or long latency
        if latency < min_ideal:
            return 80  # Falling asleep too quickly might indicate sleep deprivation
        elif latency <= max_ideal:
            return 100
        elif latency <= 30:
            return 90 - (latency - max_ideal) * 2  # More significant penalty
        elif latency <= 60:
            return 80 - (latency - 30) * 1.5  # Steeper decline for delays
        else:
            return max(0, 35 - (latency - 60) * 0.5)  # More severe penalty for very long delays
    
    def _score_continuity(self, sleep_data):
        """Score sleep continuity component (awakenings and time awake)"""
        awakenings_score = None
        awake_time_score = None
        
        # Process awakenings count if available
        if sleep_data.awakenings_count is not None:
            awakenings = sleep_data.awakenings_count
            min_ideal, max_ideal = self.ideal_ranges['awakenings_count']
            
            # Score based on awakenings
            if awakenings <= max_ideal:
                awakenings_score = 100
            else:
                awakenings_score = max(0, 100 - 10 * (awakenings - max_ideal))
        
        # Determine awake time - use available data or estimate
        awake_time = None
        if sleep_data.total_awake_minutes is not None:
            awake_time = sleep_data.total_awake_minutes
        elif sleep_data.awakenings_count is not None and sleep_data.awakenings_count > 0:
            # Estimate based on pattern
            if hasattr(sleep_data, 'sleep_pattern') and sleep_data.sleep_pattern == 'insomnia':
                awake_time = sleep_data.awakenings_count * 30  # 30 min for insomnia
            else:
                awake_time = sleep_data.awakenings_count * 10  # 10 min normally
        
        # Score awake time if we have it
        if awake_time is not None:
            # Extended scale for insomnia
            if awake_time <= 20:  # Ideal range
                awake_time_score = 100
            elif awake_time <= 45:  # Mild disruption
                awake_time_score = 90 - (awake_time - 20) * 0.8
            elif awake_time <= 120:  # Up to 2 hours
                awake_time_score = 70 - (awake_time - 45) * 0.25
            else:  # More than 2 hours
                awake_time_score = max(0, 50 - (awake_time - 120) * 0.1)
        
        # Combine scores with more weight on time awake
        if awakenings_score is not None and awake_time_score is not None:
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
        if sleep_data.bedtime is not None:
            # Extract hour from datetime
            if hasattr(sleep_data.bedtime, 'hour'):
                bedtime_hour = sleep_data.bedtime.hour
            else:
                # Try to parse from string
                try:
                    if isinstance(sleep_data.bedtime, pd.Timestamp):
                        bedtime_hour = sleep_data.bedtime.hour
                    else:
                        dt = pd.to_datetime(sleep_data.bedtime)
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
        if sleep_data.wake_time is not None:
            # Extract hour from datetime
            if hasattr(sleep_data.wake_time, 'hour'):
                waketime_hour = sleep_data.wake_time.hour
            else:
                # Try to parse from string
                try:
                    from datetime import datetime
                    dt = datetime.strptime(sleep_data.wake_time, '%Y-%m-%d %H:%M:%S')
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
        if sleep_data.subjective_rating is None:
            return None
            
        rating = sleep_data.subjective_rating
        
        # Convert rating to 0-100 scale, handling floats
        if 0 <= rating <= 10:
            return rating * 10  # Scale linearly from 0-10 to 0-100
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
    print(f"User-reported sleep score: {user_score_details.total_score}")
    print("Component scores:")
    for component, score in user_score_details.component_scores.items():
        print(f"  {component}: {score.score}")
    
    # Transform wearable data and calculate score
    transformed_wearable_data = calculator.transform_wearable_data(wearable_data)
    wearable_score = calculator.calculate_score(transformed_wearable_data)
    print(f"\nWearable-based sleep score: {wearable_score}")