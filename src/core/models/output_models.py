# src/core/models/output_models.py

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime


class SleepPrediction(BaseModel):
    """Standardized sleep prediction output"""
    user_id: str
    date: str
    predicted_sleep_efficiency: float = Field(..., ge=0.0, le=1.0)
    prediction_confidence: float = Field(..., ge=0.0, le=1.0)
    sleep_score: Optional[int] = Field(None, ge=0, le=100)


class UserCentricResult(BaseModel):
    """Base model for user-centric results"""
    user_id: str
    timestamp: datetime
    

class SleepInsight(UserCentricResult):
    """Sleep insight for user reporting"""
    insight_type: str
    message: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    metrics: Dict[str, Any] = {}
    source_data_points: int = 0
    

class SleepRecommendation(UserCentricResult):
    """Sleep recommendation for user reporting"""
    message: str
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    profession_impact: Optional[Dict[str, Any]] = None
    region_impact: Optional[Dict[str, Any]] = None


class AnalysisReport(BaseModel):
    """Complete analysis report for user or admin dashboards"""
    generated_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None  # None for aggregate reports
    period_start: datetime
    period_end: datetime
    sleep_quality_avg: float = Field(..., ge=0.0, le=1.0)
    sleep_score_avg: int = Field(..., ge=0, le=100)
    trend: str
    consistency: float = Field(..., ge=0.0, le=1.0)
    key_metrics: Dict[str, Any]
    insights: List[SleepInsight] = []
    recommendations: List[SleepRecommendation] = []
    visualizations: Dict[str, str] = {}  # path to visualization files