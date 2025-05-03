import numpy as np
import logging
from typing import Dict, Optional, Any, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

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
    subjective_rating: Optional[int] = Field(None, ge=0, le=10)
    
    # Demographics
    age: Optional[int] = Field(None, ge=0)
    profession_category: Optional[str] = None
    region_category: Optional[str] = None
    
    # Additional data
    is_weekend: Optional[bool] = False
    no_sleep: bool = False


class SleepScoreOutput(BaseModel):
    """Output model for sleep score calculation"""
    total_score: int = Field(..., ge=0, le=100)
    component_scores: Dict[str, float] = {}
    demographic_adjustments: Dict[str, float] = {}
    metadata: Dict[str, Any] = {}


class SleepScoreCalculator:
    """
    Enhanced sleep score calculator with greater variability and sensitivity.
    All sleep score calculations should use this class.
    """
    
    def __init__(self):
        """Initialize the sleep score calculator with ideal ranges and weights"""
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
        
        logger.info("Improved sleep score calculator initialized")
    
    def calculate_score(self, sleep_data: Union[Dict, SleepScoreInput]) -> Union[int, SleepScoreOutput]:
        """
        Calculate a comprehensive sleep score based on sleep metrics.
        
        Args:
            sleep_data: Either a dictionary with sleep metrics or a SleepScoreInput model
            
        Returns:
            int or SleepScoreOutput: Sleep score (0-100) or detailed output
        """
        # Convert dict to SleepScoreInput if needed
        try:
            if isinstance(sleep_data, dict):
                input_data = SleepScoreInput(**sleep_data)
            else:
                input_data = sleep_data
        except Exception as e:
            logger.error(f"Error validating sleep data: {e}")
            return 50  # Default score on error
            
        # Print debug info
        user_id = sleep_data.get('user_id', 'unknown') if isinstance(sleep_data, dict) else 'unknown'
        logger.debug(f"Calculating sleep score for user {user_id}")
        
        # Calculate component scores
        component_scores = {}
        
        # Duration component
        duration_score = self._score_duration(input_data)
        if duration_score is not None:
            component_scores['duration'] = duration_score
            logger.debug(f"  Duration score: {duration_score}")
        
        # Efficiency component
        efficiency_score = self._score_efficiency(input_data)
        if efficiency_score is not None:
            component_scores['efficiency'] = efficiency_score
            logger.debug(f"  Efficiency score: {efficiency_score}")
        
        # Onset component
        onset_score = self._score_onset(input_data)
        if onset_score is not None:
            component_scores['onset'] = onset_score
            logger.debug(f"  Onset score: {onset_score}")
        
        # Continuity component
        continuity_score = self._score_continuity(input_data)
        if continuity_score is not None:
            component_scores['continuity'] = continuity_score
            logger.debug(f"  Continuity score: {continuity_score}")
        
        # Subjective component
        subjective_score = self._score_subjective(input_data)
        if subjective_score is not None:
            component_scores['subjective'] = subjective_score
            logger.debug(f"  Subjective score: {subjective_score}")
        
        # Handle special cases
        if input_data.no_sleep:
            base_score = 0
            logger.debug("  No sleep reported, score set to 0")
        else:
            # Calculate weighted average
            base_score = self._calculate_weighted_score(component_scores)
            logger.debug(f"  Base score: {base_score}")
        
        # Apply demographic adjustments
        demographic_adjustments = {}
        
        # Age-based adjustments with more variation
        if input_data.age is not None:
            age_adjustment = self._get_age_adjustment(input_data.age, component_scores)
            demographic_adjustments['age'] = age_adjustment
            logger.debug(f"  Age adjustment: {age_adjustment}")
        
        # Profession-based adjustments
        if input_data.profession_category is not None:
            prof_adjustment = self._get_profession_adjustment(
                input_data.profession_category, component_scores
            )
            demographic_adjustments['profession'] = prof_adjustment
            logger.debug(f"  Profession adjustment: {prof_adjustment}")
        
        # Region-based adjustments
        if input_data.region_category is not None:
            region_adjustment = self._get_region_adjustment(
                input_data.region_category, component_scores
            )
            demographic_adjustments['region'] = region_adjustment
            logger.debug(f"  Region adjustment: {region_adjustment}")
        
        # Apply adjustments - with significant variation
        total_adjustment = sum(demographic_adjustments.values())
        adjusted_score = base_score + total_adjustment
        
        # Apply randomization factor for more natural distribution (1-5% variation)
        random_factor = np.random.uniform(0.95, 1.05)
        final_score = int(round(min(100, max(0, adjusted_score * random_factor))))
        
        logger.debug(f"  Final score: {final_score}")
        
        # Create detailed output
        output = SleepScoreOutput(
            total_score=final_score,
            component_scores={k: float(v) for k, v in component_scores.items()},
            demographic_adjustments=demographic_adjustments,
            metadata={
                'base_score': base_score,
                'adjusted_score': adjusted_score,
                'random_factor': random_factor
            }
        )
        
        return output
    
    def _score_continuity(self, sleep_data):
        """Score sleep continuity component (awakenings and time awake)"""
        awakenings_score = None
        awake_time_score = None
        
        # Check if awakenings count is available
        if sleep_data.awakenings_count is not None:
            awakenings = sleep_data.awakenings_count
            min_ideal, max_ideal = self.ideal_ranges['awakenings_count']
            
            # Score based on awakenings
            if awakenings <= max_ideal:
                # Ideal range - full score
                awakenings_score = 100
            else:
                # Penalize each additional awakening
                awakenings_score = max(0, 100 - 10 * (awakenings - max_ideal))
        
        # Check if time awake is available
        if sleep_data.total_awake_minutes is not None:
            awake_time = sleep_data.total_awake_minutes
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
    
    def _score_duration(self, sleep_data):
        """Score sleep duration component with a more realistic distribution"""
        if sleep_data.sleep_duration_hours is None:
            return None
            
        duration = sleep_data.sleep_duration_hours
        min_ideal, max_ideal = self.ideal_ranges['sleep_duration_hours']
        
        # Create a more granular, bell-curve style scoring function
        if duration < 3:  # Severely deprived
            # Very severe penalty for extremely short sleep
            return max(0, duration * 10)  # 0-30 score for 0-3 hours
        elif duration < min_ideal:  # Too little sleep
            # More gradual increase as duration approaches ideal
            range_fraction = (duration - 3) / (min_ideal - 3)
            return 30 + range_fraction * 60  # 30-90 score for 3-7 hours
        elif duration <= max_ideal:  # Ideal range
            # Bell curve within ideal range - perfect at the center
            center = (min_ideal + max_ideal) / 2
            distance_from_center = abs(duration - center)
            max_distance = (max_ideal - min_ideal) / 2
            return 100 - 10 * (distance_from_center / max_distance)  # 90-100 range
        elif duration <= 10:  # Somewhat too much sleep
            # Moderate penalty for oversleeping
            excess = duration - max_ideal
            return max(70, 90 - excess * 20)  # 90 down to 70 for 9-10 hours
        else:  # Excessive sleep
            # Steeper penalty for excessive sleep
            return max(30, 70 - (duration - 10) * 10)  # Below 70 for >10 hours
    
    def _score_efficiency(self, sleep_data):
        """Score sleep efficiency component with more realistic distribution"""
        # Check if efficiency is available
        if sleep_data.sleep_efficiency is None:
            # Try to calculate it if we have duration and time in bed
            if sleep_data.sleep_duration_hours is not None and sleep_data.time_in_bed_hours is not None:
                if sleep_data.time_in_bed_hours == 0:
                    return None  # Avoid division by zero
                efficiency = sleep_data.sleep_duration_hours / sleep_data.time_in_bed_hours
            else:
                return None
        else:
            efficiency = sleep_data.sleep_efficiency
        
        min_ideal, max_ideal = self.ideal_ranges['sleep_efficiency']
        
        # Ensure value is between 0 and 1
        efficiency = max(0, min(1, efficiency))
        
        # Create a more gradual scoring curve with significant penalties for poor efficiency
        if efficiency < 0.5:  # Very poor efficiency
            # More significant penalty for very poor efficiency
            # Linear from 0-40 for efficiency 0-0.5
            return efficiency * 80
        elif efficiency < min_ideal:  # Below ideal but not terrible
            # More gradual increase from 40-90 as efficiency increases from 0.5-0.85
            range_fraction = (efficiency - 0.5) / (min_ideal - 0.5)
            return 40 + range_fraction * 50
        elif efficiency <= max_ideal:  # Ideal range
            # Even in the ideal range, provide some gradation (90-100)
            range_position = (efficiency - min_ideal) / (max_ideal - min_ideal)
            return 90 + range_position * 10
        else:  # Above ideal (might indicate unrealistic data)
            # Steeper penalty for unrealistically high efficiency
            return max(70, 100 - 300 * (efficiency - max_ideal))
    
    def _score_onset(self, sleep_data):
        """Score sleep onset component with more realistic penalties"""
        if sleep_data.sleep_onset_latency_minutes is None:
            return None
            
        latency = sleep_data.sleep_onset_latency_minutes
        min_ideal, max_ideal = self.ideal_ranges['sleep_onset_latency_minutes']
        
        # Add more gradation and steeper penalties
        if latency < min_ideal - 3:  # Too quick - may indicate exhaustion
            return max(40, 60 + (latency / min_ideal) * 30)  # 60-90 range
        elif latency < min_ideal:  # Slightly quick
            return 90 + (latency - (min_ideal - 3)) * 3  # 90-100 range
        elif latency <= max_ideal:  # Ideal range
            return 100  # Perfect score in ideal range
        elif latency <= 30:  # Slightly delayed
            return max(70, 100 - (latency - max_ideal) * 2)  # 100-80 range
        elif latency <= 60:  # Moderately delayed
            return max(40, 70 - (latency - 30) * 1)  # 70-40 range
        else:  # Severely delayed
            return max(0, 40 - (latency - 60) * 0.5)  # 40-0 range