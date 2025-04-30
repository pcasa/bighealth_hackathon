# src/core/models/data_models.py

from pydantic import BaseModel, Field, field_validator, model_validator, validator
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from enum import Enum


# Enum types for better validation
class SleepPattern(str, Enum):
    NORMAL = "normal"
    INSOMNIA = "insomnia"
    SHIFT_WORKER = "shift_worker"
    OVERSLEEPER = "oversleeper"
    VARIABLE = "variable"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non-binary"
    OTHER = "other"


class DeviceType(str, Enum):
    APPLE_WATCH = "apple_watch"
    GOOGLE_WATCH = "google_watch"
    FITBIT = "fitbit"
    SAMSUNG_WATCH = "samsung_watch"


# User Models
class UserProfile(BaseModel):
    user_id: str
    age: int = Field(..., ge=18, le=100)
    gender: str
    profession: str
    region: str
    sleep_pattern: str
    device_type: str
    data_consistency: float = Field(..., ge=0.0, le=1.0)
    sleep_consistency: float = Field(..., ge=0.0, le=1.0)
    created_at: str
    
    @validator('sleep_pattern')
    def validate_sleep_pattern(cls, v):
        valid_patterns = ['normal', 'insomnia', 'shift_worker', 'oversleeper', 'variable']
        if v not in valid_patterns:
            raise ValueError(f'Invalid sleep pattern. Must be one of: {", ".join(valid_patterns)}')
        return v
    
    @validator('device_type')
    def validate_device_type(cls, v):
        valid_devices = ['apple_watch', 'google_watch', 'fitbit', 'samsung_watch']
        if v not in valid_devices:
            raise ValueError(f'Invalid device type. Must be one of: {", ".join(valid_devices)}')
        return v


# Sleep Data Models
class SleepEntry(BaseModel):
    user_id: str
    date: str
    bedtime: str
    sleep_onset_time: str
    wake_time: str
    out_bed_time: str
    time_in_bed_hours: Optional[float] = None
    sleep_duration_hours: Optional[float] = None
    sleep_onset_latency_minutes: Optional[int] = None
    awakenings_count: int = Field(..., ge=0)
    total_awake_minutes: int
    sleep_efficiency: Optional[float] = None
    subjective_rating: int = Field(..., ge=1, le=10)
    is_weekend: Optional[bool] = None
    no_sleep: bool = False
    
    @field_validator('sleep_efficiency')
    @classmethod
    def validate_sleep_efficiency(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Sleep efficiency must be between 0 and 1')
        return v
    
    @model_validator(mode='after')
    def calculate_derived_fields(self):
        # Calculate sleep onset latency if not provided
        if self.sleep_onset_latency_minutes is None and self.bedtime and self.sleep_onset_time:
            try:
                bedtime = datetime.strptime(self.bedtime, '%Y-%m-%d %H:%M:%S')
                sleep_onset = datetime.strptime(self.sleep_onset_time, '%Y-%m-%d %H:%M:%S')
                latency = (sleep_onset - bedtime).total_seconds() / 60
                self.sleep_onset_latency_minutes = max(0, int(latency))
            except:
                pass
        
        # Calculate time in bed if not provided
        if self.time_in_bed_hours is None and self.bedtime and self.out_bed_time:
            try:
                bedtime = datetime.strptime(self.bedtime, '%Y-%m-%d %H:%M:%S')
                out_bed = datetime.strptime(self.out_bed_time, '%Y-%m-%d %H:%M:%S')
                time_in_bed = (out_bed - bedtime).total_seconds() / 3600
                self.time_in_bed_hours = max(0, time_in_bed)
            except:
                pass
        
        # Calculate sleep duration if not provided
        if self.sleep_duration_hours is None and self.sleep_onset_time and self.wake_time:
            try:
                sleep_onset = datetime.strptime(self.sleep_onset_time, '%Y-%m-%d %H:%M:%S')
                wake_time = datetime.strptime(self.wake_time, '%Y-%m-%d %H:%M:%S')
                total_time = (wake_time - sleep_onset).total_seconds() / 3600
                # Subtract awake time if available
                if self.total_awake_minutes:
                    total_time -= self.total_awake_minutes / 60
                self.sleep_duration_hours = max(0, total_time)
            except:
                pass
        
        # Calculate sleep efficiency if not provided
        if self.sleep_efficiency is None and self.sleep_duration_hours and self.time_in_bed_hours:
            if self.time_in_bed_hours > 0:
                efficiency = self.sleep_duration_hours / self.time_in_bed_hours
                self.sleep_efficiency = max(0, min(1, efficiency))
        
        # Check for is_weekend if not provided
        if self.is_weekend is None and self.date:
            try:
                date_obj = datetime.strptime(self.date, '%Y-%m-%d')
                self.is_weekend = date_obj.weekday() >= 5
            except:
                pass
        
        return self


# Wearable Data Models
class WearableData(BaseModel):
    user_id: str
    date: str
    device_type: str
    device_bedtime: str
    device_sleep_onset: str
    device_wake_time: str
    device_sleep_duration: float
    deep_sleep_percentage: float = Field(..., ge=0.0, le=1.0)
    light_sleep_percentage: float = Field(..., ge=0.0, le=1.0)
    rem_sleep_percentage: float = Field(..., ge=0.0, le=1.0)
    awake_percentage: float = Field(..., ge=0.0, le=1.0)
    sleep_cycles: float
    average_heart_rate: float
    min_heart_rate: float
    max_heart_rate: float
    heart_rate_variability: float
    movement_intensity: float
    blood_oxygen: Optional[float] = None
    
    @model_validator(mode='after')
    def validate_sleep_percentages(self):
        # Ensure sleep stage percentages sum to approximately 1
        sleep_stages = [
            self.deep_sleep_percentage,
            self.light_sleep_percentage,
            self.rem_sleep_percentage,
            self.awake_percentage
        ]
        total = sum(sleep_stages)
        if total > 0 and abs(total - 1.0) > 0.01:  # Allow small rounding errors
            # Normalize to sum to 1.0
            self.deep_sleep_percentage = self.deep_sleep_percentage / total
            self.light_sleep_percentage = self.light_sleep_percentage / total
            self.rem_sleep_percentage = self.rem_sleep_percentage / total
            self.awake_percentage = self.awake_percentage / total
        return self


# Analysis and Recommendation Models
class SleepMetrics(BaseModel):
    avg_efficiency: Optional[float] = None
    avg_time_in_bed: Optional[float] = None
    recent_efficiency: Optional[float] = None
    recent_time_in_bed: Optional[float] = None
    best_efficiency: Optional[float] = None
    avg_awakenings: Optional[float] = None
    avg_time_awake: Optional[float] = None
    recent_rating: Optional[float] = None
    consistent_bedtime: Optional[bool] = None
    no_sleep_count: Optional[int] = None
    
    @field_validator('avg_efficiency', 'recent_efficiency', 'best_efficiency')
    @classmethod
    def validate_efficiency(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Efficiency must be between 0 and 1')
        return v


class ProgressAnalysis(BaseModel):
    trend: str
    consistency: float
    improvement_rate: float
    key_metrics: SleepMetrics
    
    @field_validator('trend')
    @classmethod
    def validate_trend(cls, v):
        valid_trends = [
            'insufficient_data', 'insufficient_sleep_data', 'strong_improvement', 
            'improvement', 'regression', 'slight_regression', 'stable',
            'severe_insomnia', 'moderate_insomnia'
        ]
        if v not in valid_trends:
            raise ValueError(f'Invalid trend. Must be one of: {", ".join(valid_trends)}')
        return v
    
    @field_validator('consistency')
    @classmethod
    def validate_consistency(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Consistency must be between 0 and 1')
        return v


class Recommendation(BaseModel):
    message: str
    confidence: float
    category: str
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        valid_categories = [
            'general', 'profession', 'region', 'data', 'trend', 'consistency'
        ]
        if v not in valid_categories:
            raise ValueError(f'Invalid category. Must be one of: {", ".join(valid_categories)}')
        return v


# Models for data generation
class GenerationConfig(BaseModel):
    user_count: int = 100
    start_date: str
    end_date: str
    wearable_percentage: int = Field(20, ge=0, le=100)
    
    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Invalid date format. Use YYYY-MM-DD')
        
    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v, info):
        if 'start_date' in info.data:
            start = datetime.strptime(info.data['start_date'], '%Y-%m-%d')
            end = datetime.strptime(v, '%Y-%m-%d')
            if end <= start:
                raise ValueError('End date must be after start date')
        return v