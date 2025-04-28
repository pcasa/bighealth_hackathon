# src/api/main.py

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import sys
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta

from scripts.sleep_advisor import get_user_sleep_data

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recommendation_engine import SleepRecommendationEngine
from models.sleep_quality import SleepQualityModel
from data_processing.preprocessing import Preprocessor

# Import our new user profile routes
from api.user_profile_routes import router as user_router
from src.utils.constants import profession_categories

# Initialize application
app = FastAPI(
    title="Sleep Insights API",
    description="API for analyzing sleep data and generating recommendations for insomnia",
    version="0.2.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
recommendation_engine = SleepRecommendationEngine()
sleep_quality_model = SleepQualityModel()
preprocessor = Preprocessor()

# Try to load the sleep quality model
try:
    sleep_quality_model.load('models/sleep_quality_model')
except Exception as e:
    print(f"Warning: Could not load sleep quality model: {str(e)}")

# Data models
class ConfidenceScore(BaseModel):
    value: Optional[float] = None
    confidence: float

class SleepEntryResponse(BaseModel):
    status: str
    message: str
    entry_date: str
    sleep_score: ConfidenceScore
    trend: str
    consistency: float
    key_metrics: Dict
    profession_impact: Dict
    region_impact: Dict
    recommendation: str
    predictions: Dict[str, ConfidenceScore]

class AnalysisResultWithConfidence(BaseModel):
    trend: str
    consistency: float
    improvement_rate: float
    key_metrics: Dict[str, ConfidenceScore]
    recommendations: ConfidenceScore
    profession_impact: Dict
    region_impact: Dict
    overall_confidence: float

class RecommendationWithConfidence(BaseModel):
    user_id: str
    recommendation: ConfidenceScore
    timestamp: str
    profession_impact: Dict
    region_impact: Dict
    overall_confidence: float
    data_points_used: int
    
class SleepEntry(BaseModel):
    user_id: str
    date: str
    bedtime: str
    sleep_onset_time: str
    wake_time: str
    out_bed_time: str
    awakenings_count: int
    time_awake_minutes: int
    subjective_rating: int
    no_sleep: bool = False

class UserHistory(BaseModel):
    user_id: str
    days: int = 30

class AnalysisResult(BaseModel):
    trend: str
    consistency: float
    improvement_rate: float
    key_metrics: Dict
    recommendations: Optional[str] = None

class EnhancedSleepEntry(BaseModel):
    user_id: str
    date: str
    bedtime: str
    sleep_onset_time: str
    wake_time: str
    out_bed_time: str
    awakenings_count: int
    time_awake_minutes: int
    subjective_rating: int
    no_sleep: bool = False
    profession_impact: Optional[Dict[str, float]] = None
    region_impact: Optional[Dict[str, float]] = None

# Include user profile routes
app.include_router(user_router)

# API routes
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Enhanced Sleep Insights API",
        "version": "0.2.0",
        "documentation": "/docs",
        "new_features": [
            "Enhanced user profiles with profession and region information",
            "Profession-specific sleep recommendations",
            "Region-specific sleep insights",
            "Statistical analysis by profession and region categories"
        ]
    }

@app.post("/sleep/log", status_code=201)
async def log_sleep(entry: EnhancedSleepEntry):
    """Log a new sleep entry with enhanced profession and region analysis"""
    try:
        # Convert to dictionary for easier handling
        entry_dict = entry.dict(exclude={"profession_impact", "region_impact"})
        
        # Create a dataframe from the entry
        entry_df = pd.DataFrame([entry_dict])
        
        # Load user profile to get profession and region data
        user_id = entry.user_id
        users_df = load_user_data()
        user_profile = users_df[users_df['user_id'] == user_id]
        
        if len(user_profile) > 0:
            # Extract profession and region
            profession = user_profile.iloc[0].get('profession', '')
            region = user_profile.iloc[0].get('region', '')
            
            # Calculate profession impact
            profession_impact = calculate_profession_impact(profession, entry_dict)
            
            # Calculate region impact
            region_impact = calculate_region_impact(region, entry_dict)
        else:
            profession = ""
            region = ""
            profession_impact = {}
            region_impact = {}
        
        # Save to the appropriate data store
        historical_path = 'data/raw/sleep_data.csv'
        
        # Append to existing file or create new one
        historical_df = None
        if os.path.exists(historical_path):
            historical_df = pd.read_csv(historical_path)
            
            # Check for duplicate entry (same user and date)
            mask = (historical_df['user_id'] == entry.user_id) & (historical_df['date'] == entry.date)
            if mask.any():
                # Update existing entry
                historical_df.loc[mask] = entry_dict
            else:
                # Append new entry
                historical_df = pd.concat([historical_df, entry_df], ignore_index=True)
        else:
            # Create new file with this entry
            historical_df = entry_df
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(historical_path), exist_ok=True)
        
        # Save updated data
        historical_df.to_csv(historical_path, index=False)
        
        # Now generate all available predictions and insights
        
        # 1. Filter data for this user
        user_data = historical_df[historical_df['user_id'] == entry.user_id].copy()
        
        # Ensure dates are in datetime format
        user_data['date'] = pd.to_datetime(user_data['date'])
        user_data = user_data.sort_values('date')
        
        # Check if we have enough data for analysis
        if len(user_data) < 3:
            return {
                "status": "success", 
                "message": "Sleep entry logged successfully",
                "note": "Not enough historical data for comprehensive analysis. Continue logging sleep for more insights.",
                "profession_impact": profession_impact,
                "region_impact": region_impact
            }
        
        # 2. Preprocess data
        processed_data = preprocessor.preprocess_sleep_data(user_data)
        
        # 3. Calculate sleep score for this entry
        row = processed_data[processed_data['date'] == pd.to_datetime(entry.date)].iloc[0]
        
        sleep_score = None
        if entry.no_sleep:
            sleep_score = 0
        else:
            # Create additional metrics dictionary
            additional_metrics = {}
            for col in ['deep_sleep_percentage', 'rem_sleep_percentage', 'sleep_onset_latency_minutes', 'awakenings_count']:
                if col in row:
                    additional_metrics[col] = row[col]
            
            # Calculate score
            try:
                sleep_score = sleep_quality_model.calculate_sleep_score(
                    row['sleep_efficiency'], 
                    row.get('subjective_rating'),
                    additional_metrics
                )
            except Exception as e:
                # Fallback if model fails
                print(f"Warning: Model-based score calculation failed: {str(e)}")
                # Simple calculation
                sleep_score = int(row['sleep_efficiency'] * 80)
                if 'subjective_rating' in row:
                    sleep_score += int(row['subjective_rating'] * 2)
                sleep_score = min(100, max(0, sleep_score))
        
        # 4. Analyze sleep trends
        progress_data = recommendation_engine.analyze_progress(entry.user_id, processed_data)
        
        # 5. Generate profession and region-aware recommendation
        recommendation = generate_enhanced_recommendation(
            entry.user_id, 
            progress_data, 
            profession, 
            region
        )
        
        # 6. Store recommendation
        rec_dir = 'data/recommendations'
        os.makedirs(rec_dir, exist_ok=True)
        
        user_rec_file = os.path.join(rec_dir, f"{entry.user_id}_recommendations.csv")
        rec_data = {
            'user_id': entry.user_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'message': recommendation
        }
        
        # Append to existing file or create new one
        rec_df = pd.DataFrame([rec_data])
        if os.path.exists(user_rec_file):
            existing_recs = pd.read_csv(user_rec_file)
            updated_recs = pd.concat([existing_recs, rec_df], ignore_index=True)
            updated_recs.to_csv(user_rec_file, index=False)
        else:
            rec_df.to_csv(user_rec_file, index=False)
        
        # 7. Make predictions about future sleep with confidence scores
        # Calculate averages for recent data (last 7 days)
        recent_data = processed_data.tail(7)
        avg_efficiency = recent_data['sleep_efficiency'].mean() if 'sleep_efficiency' in recent_data else None
        efficiency_trend = progress_data.get('improvement_rate', 0)

        # Get prediction confidence
        prediction_confidence = calculate_prediction_confidence(processed_data, progress_data)

        # Predict tomorrow's efficiency based on trend with confidence
        predicted_efficiency = None
        if avg_efficiency is not None:
            predicted_efficiency = min(1.0, max(0, avg_efficiency + efficiency_trend))
            
        # Calculate score confidence for sleep_score
        sleep_score_confidence = min(0.95, prediction_confidence + 0.1)  # Slightly higher confidence for current data

        # Predict optimal bedtime and wake time
        optimal_bedtime = None
        optimal_waketime = None
        bedtime_confidence = None
        waketime_confidence = None

        if 'bedtime' in recent_data.columns and 'wake_time' in recent_data.columns:
            # Find the day with best sleep efficiency
            best_day_idx = recent_data['sleep_efficiency'].idxmax() if 'sleep_efficiency' in recent_data else None
            
            if best_day_idx is not None and best_day_idx in recent_data.index:
                best_day = recent_data.loc[best_day_idx]
                
                if isinstance(best_day['bedtime'], str):
                    best_bedtime = datetime.strptime(best_day['bedtime'], '%Y-%m-%d %H:%M:%S')
                else:
                    best_bedtime = best_day['bedtime']
                
                if isinstance(best_day['wake_time'], str):
                    best_waketime = datetime.strptime(best_day['wake_time'], '%Y-%m-%d %H:%M:%S')
                else:
                    best_waketime = best_day['wake_time']
                
                # Extract just the time component
                optimal_bedtime = best_bedtime.strftime('%H:%M')
                optimal_waketime = best_waketime.strftime('%H:%M')
                
                # Calculate confidence based on sleep consistency
                if 'key_metrics' in progress_data and 'bedtime_consistency' in progress_data['key_metrics']:
                    bedtime_consistency = progress_data['key_metrics']['bedtime_consistency']
                    bedtime_confidence = min(0.9, bedtime_consistency + 0.2)  # Scale up slightly
                    waketime_confidence = min(0.9, bedtime_consistency + 0.1)  # Usually less confident about wake time
                else:
                    bedtime_confidence = prediction_confidence - 0.05
                    waketime_confidence = prediction_confidence - 0.1

        # 8. Return comprehensive response with enhanced profession and region insights and confidence scores
        return {
            "status": "success",
            "message": "Sleep entry logged successfully",
            
            # Entry details
            "entry_date": entry.date,
            "sleep_score": {
                "value": sleep_score,
                "confidence": sleep_score_confidence
            },
            
            # Analysis
            "trend": progress_data.get('trend'),
            "consistency": progress_data.get('consistency'),
            "key_metrics": progress_data.get('key_metrics'),
            
            # Enhanced insights
            "profession_impact": profession_impact,
            "region_impact": region_impact,
            
            # Recommendation
            "recommendation": recommendation,
            
            # Predictions with confidence scores
            "predictions": {
                "estimated_next_efficiency": {
                    "value": round(predicted_efficiency * 100) if predicted_efficiency else None,
                    "confidence": prediction_confidence
                },
                "optimal_bedtime": {
                    "value": optimal_bedtime,
                    "confidence": bedtime_confidence
                },
                "optimal_waketime": {
                    "value": optimal_waketime,
                    "confidence": waketime_confidence
                },
                "expected_improvement": {
                    "value": "increasing" if efficiency_trend > 0.005 else 
                            "decreasing" if efficiency_trend < -0.005 else "stable",
                    "confidence": prediction_confidence - 0.05  # Slightly lower confidence for trend prediction
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging sleep entry: {str(e)}")


@app.get("/sleep/analysis/{user_id}")
async def analyze_sleep(user_id: str, days: int = 30):
    """Analyze sleep data for a user with enhanced profession and region insights"""
    try:
        # Load data
        data_path = 'data/raw/sleep_data.csv'
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="No sleep data found")
        
        # Read data
        sleep_data = pd.read_csv(data_path)
        
        # Filter for this user
        user_data = sleep_data[sleep_data['user_id'] == user_id].copy()
        
        if len(user_data) == 0:
            raise HTTPException(status_code=404, detail=f"No sleep data found for user {user_id}")
        
        # Load user profile to get profession and region data
        users_df = load_user_data()
        user_profile = users_df[users_df['user_id'] == user_id]
        
        profession = ""
        region = ""
        if len(user_profile) > 0:
            profession = user_profile.iloc[0].get('profession', '')
            region = user_profile.iloc[0].get('region', '')
        
        # Ensure dates are in datetime format
        user_data['date'] = pd.to_datetime(user_data['date'])
        
        # Filter to requested days
        start_date = datetime.now() - timedelta(days=days)
        user_data = user_data[user_data['date'] >= start_date]
        
        if len(user_data) == 0:
            raise HTTPException(status_code=404, detail=f"No recent sleep data found for user {user_id}")
        
        # Preprocess data
        processed_data = preprocessor.preprocess_sleep_data(user_data)
        
        # Analyze progress
        progress_data = recommendation_engine.analyze_progress(user_id, processed_data)
        
        # Generate enhanced recommendation
        recommendation = generate_enhanced_recommendation(
            user_id, 
            progress_data, 
            profession, 
            region
        )
        
        # Add profession and region impact analysis
        profession_impact = analyze_profession_impact(profession, progress_data)
        region_impact = analyze_region_impact(region, progress_data)
        
        # Calculate prediction confidence based on data quality
        prediction_confidence = calculate_prediction_confidence(processed_data, progress_data)
        
        # Add confidence scores to metrics
        metrics_with_confidence = {}
        if 'key_metrics' in progress_data:
            for metric, value in progress_data['key_metrics'].items():
                if value is not None:
                    # Calculate different confidence scores for different metrics
                    if metric in ['avg_efficiency', 'recent_efficiency', 'best_efficiency']:
                        metric_confidence = min(0.95, prediction_confidence + 0.1)
                    elif metric in ['avg_awakenings', 'avg_time_awake']:
                        metric_confidence = prediction_confidence
                    elif metric in ['consistent_bedtime']:
                        metric_confidence = min(0.9, prediction_confidence + 0.05)
                    else:
                        metric_confidence = prediction_confidence - 0.05
                    
                    # Store value and confidence
                    metrics_with_confidence[metric] = {
                        "value": value,
                        "confidence": round(metric_confidence, 2)
                    }
                else:
                    metrics_with_confidence[metric] = {
                        "value": None,
                        "confidence": 0
                    }
        
        # Add recommendation to result with confidence score
        result = {
            "trend": progress_data.get('trend'),
            "consistency": progress_data.get('consistency'),
            "improvement_rate": progress_data.get('improvement_rate'),
            "key_metrics": metrics_with_confidence,
            "recommendations": {
                "text": recommendation,
                "confidence": min(0.9, prediction_confidence + 0.05)  # Slightly higher confidence for recommendations
            },
            "profession_impact": profession_impact,
            "region_impact": region_impact,
            "overall_confidence": prediction_confidence
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sleep data: {str(e)}")


@app.get("/sleep/recommendation/{user_id}")
async def get_recommendation(user_id: str, days: int = 30):
    """Get a personalized profession and region-aware recommendation for a user"""
    try:
        # Load data
        data_path = 'data/raw/sleep_data.csv'
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="No sleep data found")
        
        # Read data
        sleep_data = pd.read_csv(data_path)
        
        # Filter for this user
        user_data = sleep_data[sleep_data['user_id'] == user_id].copy()
        
        if len(user_data) == 0:
            raise HTTPException(status_code=404, detail=f"No sleep data found for user {user_id}")
        
        # Load user profile to get profession and region data
        users_df = load_user_data()
        user_profile = users_df[users_df['user_id'] == user_id]
        
        profession = ""
        region = ""
        if len(user_profile) > 0:
            profession = user_profile.iloc[0].get('profession', '')
            region = user_profile.iloc[0].get('region', '')
        
        # Ensure dates are in datetime format
        user_data['date'] = pd.to_datetime(user_data['date'])
        
        # Filter to requested days
        start_date = datetime.now() - timedelta(days=days)
        user_data = user_data[user_data['date'] >= start_date]
        
        if len(user_data) == 0:
            raise HTTPException(status_code=404, detail=f"No recent sleep data found for user {user_id}")
        
        # Preprocess data
        processed_data = preprocessor.preprocess_sleep_data(user_data)
        
        # Analyze progress
        progress_data = recommendation_engine.analyze_progress(user_id, processed_data)
        
        # Generate enhanced recommendation
        recommendation = generate_enhanced_recommendation(
            user_id, 
            progress_data, 
            profession, 
            region
        )
        
        # Calculate prediction confidence
        prediction_confidence = calculate_prediction_confidence(processed_data, progress_data)
        
        # Calculate recommendation confidence based on data quality
        recommendation_confidence = min(0.9, prediction_confidence + 0.05)
        
        # Store recommendation
        rec_dir = 'data/recommendations'
        os.makedirs(rec_dir, exist_ok=True)
        
        user_rec_file = os.path.join(rec_dir, f"{user_id}_recommendations.csv")
        rec_data = {
            'user_id': user_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'message': recommendation,
            'confidence': recommendation_confidence  # Store confidence with recommendation
        }
        
        # Append to existing file or create new one
        rec_df = pd.DataFrame([rec_data])
        if os.path.exists(user_rec_file):
            existing_recs = pd.read_csv(user_rec_file)
            updated_recs = pd.concat([existing_recs, rec_df], ignore_index=True)
            updated_recs.to_csv(user_rec_file, index=False)
        else:
            rec_df.to_csv(user_rec_file, index=False)
        
        # Analyze profession and region impact with confidence
        profession_impact = analyze_profession_impact(profession, progress_data)
        profession_impact['confidence'] = min(0.85, prediction_confidence)  # Slightly lower confidence for impact analysis
        
        region_impact = analyze_region_impact(region, progress_data)
        region_impact['confidence'] = min(0.85, prediction_confidence)  # Slightly lower confidence for impact analysis
        
        return {
            "user_id": user_id,
            "recommendation": {
                "text": recommendation,
                "confidence": recommendation_confidence
            },
            "timestamp": rec_data['timestamp'],
            "profession_impact": profession_impact,
            "region_impact": region_impact,
            "overall_confidence": prediction_confidence,
            "data_points_used": len(processed_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

@app.get("/sleep/detailed-score/{user_id}")
async def get_detailed_sleep_score(user_id: str, date: str = None):
    """Get a detailed sleep score breakdown for a user's sleep record"""
    # Get user sleep data for specified date or most recent
    user_data = get_user_sleep_data(user_id, date)
    
    if user_data is None:
        raise HTTPException(status_code=404, detail="No sleep data found")
    
    # Calculate detailed score
    detailed_score = sleep_quality_model.calculate_comprehensive_sleep_score(user_data, include_details=True)
    
    return detailed_score

# Helper functions
def load_user_data():
    """Load user data from CSV file"""
    data_path = 'data/raw/users.csv'
    if not os.path.exists(data_path):
        # If file doesn't exist, create an empty dataframe with the right columns
        return pd.DataFrame(columns=[
            'user_id', 'age', 'gender', 'profession', 'region', 
            'sleep_pattern', 'device_type', 'data_consistency', 
            'sleep_consistency', 'created_at'
        ])
    
    return pd.read_csv(data_path)

def calculate_profession_impact(profession, sleep_entry):
    """Calculate how profession impacts sleep metrics"""
    # Extract profession category
    profession_category = "other"

    for category, keywords in profession_categories.items():
        if any(keyword.lower() in profession.lower() for keyword in keywords):
            profession_category = category
            break
    
    # Define typical profession impacts on sleep
    profession_impacts = {
        'healthcare': {
            'sleep_onset': 0.2,  # Harder to fall asleep after shift work
            'sleep_efficiency': -0.1,  # Lower efficiency due to irregular schedule
            'awakenings': 0.15,  # More awakenings due to on-call mentality
            'circadian_disruption': 0.3  # High disruption from shift work
        },
        'service': {
            'sleep_onset': 0.15,  # Moderate difficulty falling asleep
            'sleep_efficiency': -0.05,  # Slightly lower efficiency
            'awakenings': 0.1,  # Some increased awakenings
            'circadian_disruption': 0.2  # Moderate disruption from variable shifts
        },
        'tech': {
            'sleep_onset': 0.25,  # Higher difficulty due to evening screen time
            'sleep_efficiency': -0.05,  # Slightly lower efficiency
            'awakenings': 0.05,  # Minimal increase in awakenings
            'circadian_disruption': 0.15  # Moderate disruption from blue light
        },
        'education': {
            'sleep_onset': 0.1,  # Slight difficulty falling asleep
            'sleep_efficiency': -0.03,  # Minor impact on efficiency
            'awakenings': 0.05,  # Minimal increase in awakenings
            'circadian_disruption': 0.05  # Low disruption
        },
        'office': {
            'sleep_onset': 0.05,  # Minor difficulty falling asleep
            'sleep_efficiency': -0.02,  # Very minor impact on efficiency
            'awakenings': 0.05,  # Minimal increase in awakenings
            'circadian_disruption': 0.05  # Low disruption
        },
        'other': {
            'sleep_onset': 0.05,  # Default values
            'sleep_efficiency': -0.02,
            'awakenings': 0.05,
            'circadian_disruption': 0.05
        }
    }
    
    return profession_impacts.get(profession_category, profession_impacts['other'])

def calculate_region_impact(region, sleep_entry):
    """Calculate how region impacts sleep metrics"""
    # Extract region category from address
    region_category = "other"
    if region and isinstance(region, str) and ',' in region:
        parts = region.split(',')
        country = parts[-1].strip()
        
        north_america = ['United States', 'Canada', 'Mexico', 'USA']
        europe = ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'UK']
        asia = ['China', 'Japan', 'India', 'Korea', 'Thailand', 'Singapore']
        
        if country in north_america:
            region_category = "north_america"
        elif country in europe:
            region_category = "europe"
        elif country in asia:
            region_category = "asia"
    
    # Define typical region impacts on sleep
    region_impacts = {
        'north_america': {
            'sleep_schedule': -0.05,  # Earlier bedtimes but not optimal
            'sleep_duration': -0.1,  # Shorter sleep duration than recommended
            'weekend_variation': 0.2,  # High variation between weekday/weekend
            'cultural_factors': 0.1  # Some cultural impact
        },
        'europe': {
            'sleep_schedule': 0.05,  # Better aligned with circadian rhythm
            'sleep_duration': 0.05,  # Slightly longer duration
            'weekend_variation': 0.1,  # Moderate variation
            'cultural_factors': 0.15  # Moderate cultural impact (late dinners, etc.)
        },
        'asia': {
            'sleep_schedule': -0.1,  # Often later bedtimes in urban areas
            'sleep_duration': -0.15,  # Often shorter sleep duration
            'weekend_variation': 0.05,  # Less variation between weekday/weekend
            'cultural_factors': 0.2  # Higher cultural impact
        },
        'other': {
            'sleep_schedule': 0,  # Default values
            'sleep_duration': 0,
            'weekend_variation': 0.1,
            'cultural_factors': 0.1
        }
    }
    
    return region_impacts.get(region_category, region_impacts['other'])

def analyze_profession_impact(profession, progress_data):
    """Generate more detailed analysis of how profession impacts sleep patterns"""
    # Extract profession category
    profession_category = "other"
    
    for category, keywords in profession_categories.items():
        if any(keyword.lower() in profession.lower() for keyword in keywords):
            profession_category = category
            break
    
    # Define analysis based on profession category
    profession_analysis = {
        'healthcare': {
            'impact_level': 'high',
            'key_challenges': [
                'Irregular shift schedules disrupting circadian rhythm',
                'High stress levels affecting sleep quality',
                'Difficulty unwinding after emotional work experiences'
            ],
            'sleep_efficiency_impact': -15,  # Percentage points
            'sleep_consistency_impact': -20,
            'recommendation_priority': 'Establish consistent sleep routines despite shift work'
        },
        'service': {
            'impact_level': 'moderate',
            'key_challenges': [
                'Variable work schedules affecting sleep consistency',
                'Evening shifts can delay natural sleep onset',
                'Physical demands causing body fatigue but mental alertness'
            ],
            'sleep_efficiency_impact': -10,
            'sleep_consistency_impact': -15,
            'recommendation_priority': 'Create wind-down routine to transition from work to sleep'
        },
        'tech': {
            'impact_level': 'moderate',
            'key_challenges': [
                'Blue light exposure from screens delaying melatonin production',
                'High mental stimulation making it harder to "turn off" the brain',
                'Sedentary work reducing physical tiredness needed for good sleep'
            ],
            'sleep_efficiency_impact': -10,
            'sleep_consistency_impact': -5,
            'recommendation_priority': 'Reduce screen time before bed and increase physical activity'
        },
        'education': {
            'impact_level': 'moderate-low',
            'key_challenges': [
                'Take-home work encroaching on evening relaxation time',
                'Stress from student/parent interactions persisting into evening',
                'Seasonal workload variations causing inconsistent sleep patterns'
            ],
            'sleep_efficiency_impact': -7,
            'sleep_consistency_impact': -10,
            'recommendation_priority': 'Set clear boundaries between work and personal time'
        },
        'office': {
            'impact_level': 'low',
            'key_challenges': [
                'Sedentary work patterns reducing sleep pressure',
                'Work stress carrying over into sleep time',
                'Consistent schedule may lead to monotony and poor sleep hygiene'
            ],
            'sleep_efficiency_impact': -5,
            'sleep_consistency_impact': -3,
            'recommendation_priority': 'Increase physical activity and create clear work-home boundaries'
        },
        'other': {
            'impact_level': 'varied',
            'key_challenges': [
                'Professional stress may impact sleep quality',
                'Work schedules may affect consistent sleep timing',
                'Work-related physical or mental demands may impact sleep'
            ],
            'sleep_efficiency_impact': -5,
            'sleep_consistency_impact': -5,
            'recommendation_priority': 'Assess specific work-related sleep disruptors'
        }
    }
    
    return profession_analysis.get(profession_category, profession_analysis['other'])

def analyze_region_impact(region, progress_data):
    """Generate more detailed analysis of how region impacts sleep patterns"""
    # Extract region category
    region_category = "other"
    if region and isinstance(region, str) and ',' in region:
        parts = region.split(',')
        country = parts[-1].strip()
        
        north_america = ['United States', 'Canada', 'Mexico', 'USA']
        europe = ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'UK']
        asia = ['China', 'Japan', 'India', 'Korea', 'Thailand', 'Singapore']
        
        if country in north_america:
            region_category = "north_america"
        elif country in europe:
            region_category = "europe"
        elif country in asia:
            region_category = "asia"
    
    # Define analysis based on region category
    region_analysis = {
        'north_america': {
            'impact_level': 'moderate',
            'key_factors': [
                'Early work start times often misaligned with natural circadian rhythms',
                'High work-focus culture often prioritizes productivity over rest',
                'High prevalence of artificial lighting affecting melatonin production'
            ],
            'cultural_sleep_practices': [
                'Emphasis on individual sleeping arrangements',
                'Relatively early dinner times (beneficial for sleep)',
                'High caffeine consumption affecting sleep onset'
            ],
            'recommendation_priority': 'Adjust schedule to better match natural circadian rhythm'
        },
        'europe': {
            'impact_level': 'moderate-low',
            'key_factors': [
                'Later work start times generally better aligned with natural rhythms',
                'Later dinner times in Southern Europe can affect sleep quality',
                'Better work-life balance culture in many countries'
            ],
            'cultural_sleep_practices': [
                'Afternoon rest periods in some countries (siesta culture)',
                'Greater emphasis on sleep as part of overall wellness',
                'Less dependence on sleep medication than North America'
            ],
            'recommendation_priority': 'Adjust evening meal timing to improve sleep quality'
        },
        'asia': {
            'impact_level': 'high',
            'key_factors': [
                'Longer work hours in many countries reducing available sleep time',
                'High population density and light pollution in urban areas',
                'Different cultural attitudes toward sleep and productivity'
            ],
            'cultural_sleep_practices': [
                'Co-sleeping arrangements more common in some countries',
                'Acceptance of napping in some work cultures',
                'Different bedding traditions affecting sleep comfort'
            ],
            'recommendation_priority': 'Find balance between cultural expectations and sleep needs'
        },
        'other': {
            'impact_level': 'varied',
            'key_factors': [
                'Local daylight patterns may impact circadian rhythm',
                'Cultural attitudes toward sleep and rest vary widely',
                'Work schedules and commute times affect available sleep time'
            ],
            'cultural_sleep_practices': [
                'Regional sleep practices vary based on climate and culture',
                'Different attitudes toward sleep medication and aids',
                'Various traditions around sleeping environment'
            ],
            'recommendation_priority': 'Identify specific regional factors affecting your sleep'
        }
    }
    
    return region_analysis.get(region_category, region_analysis['other'])

def generate_enhanced_recommendation(user_id, progress_data, profession, region):
    """Generate personalized sleep recommendation considering profession and region"""
    # First get the standard recommendation
    base_recommendation = recommendation_engine.generate_recommendation(user_id, progress_data)

    # Extract profession category
    profession_category = _extract_profession_category(profession)

    # Extract region category
    region_category = _extract_region_category(region)
    
    # Extract profession category
    profession_category = "other"
    
    for category, keywords in profession_categories.items():
        if any(keyword.lower() in profession.lower() for keyword in keywords):
            profession_category = category
            break
    
    # Extract region category
    region_category = "other"
    if region and isinstance(region, str) and ',' in region:
        parts = region.split(',')
        country = parts[-1].strip()
        
        north_america = ['United States', 'Canada', 'Mexico', 'USA']
        europe = ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'UK']
        asia = ['China', 'Japan', 'India', 'Korea', 'Thailand', 'Singapore']
        
        if country in north_america:
            region_category = "north_america"
        elif country in europe:
            region_category = "europe"
        elif country in asia:
            region_category = "asia"
    
    # Get trend information
    trend = progress_data.get('trend', 'stable')
    
    # Add profession-specific advice
    profession_advice = ""
    if profession_category == 'healthcare':
        profession_advice = " As a healthcare professional, consider using blackout curtains and white noise to improve sleep quality during day sleep periods. Use light exposure therapy after night shifts to reset your circadian rhythm."
    elif profession_category == 'tech':
        profession_advice = " Working in tech often means significant screen time. Try using blue light filters in the evening and disconnect from screens at least 1 hour before bedtime. Also consider increasing physical activity to balance sedentary work."
    elif profession_category == 'service':
        profession_advice = " In the service industry, varying shifts can disrupt sleep patterns. Try to maintain a consistent wind-down routine regardless of when your shift ends to signal to your body it's time to sleep."
    elif profession_category == 'education':
        profession_advice = " As an educator, try to set boundaries on when you stop grading or preparation work in the evening to give your mind time to wind down before sleep."
    elif profession_category == 'office':
        profession_advice = " Office work can be mentally draining but physically inactive. Consider adding exercise to your routine and establishing clear boundaries between work and relaxation time."
    
    # Add region-specific advice
    region_advice = ""
    if region_category == 'north_america':
        region_advice = " In your region, many struggle with early work start times. If possible, try to align your sleep schedule with your natural circadian rhythm rather than social expectations."
    elif region_category == 'europe':
        region_advice = " In many European countries, later dinner times can impact sleep. Try to eat your evening meal at least 3 hours before bedtime for better sleep quality."
    elif region_category == 'asia':
        region_advice = " In many Asian urban areas, high population density and light pollution can affect sleep. Consider using room-darkening curtains and white noise to create an optimal sleep environment."
    
    # Combine recommendations
    enhanced_recommendation = base_recommendation + profession_advice + region_advice
    
    return enhanced_recommendation

def _extract_region_category(region):
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

def _extract_profession_category(profession):
    """Extract profession category from profession string"""
    
    for category, keywords in profession_categories.items():
        if any(keyword.lower() in profession.lower() for keyword in keywords):
            return category
            
    return "other"

# Add a new helper function to calculate prediction confidence
def calculate_prediction_confidence(user_data, progress_data):
    """Calculate confidence level for predictions based on data quality"""
    confidence = 0.5  # Base confidence level (50%)
    
    # Factor 1: Amount of data
    data_points = len(user_data)
    if data_points >= 30:
        confidence += 0.2  # High confidence for 30+ days of data
    elif data_points >= 14:
        confidence += 0.1  # Medium confidence for 14-29 days
    elif data_points < 7:
        confidence -= 0.2  # Low confidence for less than 7 days
        
    # Factor 2: Data consistency (how regularly user logs data)
    if 'consistency' in progress_data:
        tracking_consistency = progress_data['consistency']
        # Scale from -0.1 (very inconsistent) to +0.1 (very consistent)
        consistency_factor = (tracking_consistency - 0.5) * 0.2
        confidence += consistency_factor
    
    # Factor 3: Data variability (stable patterns are more predictable)
    if 'key_metrics' in progress_data and 'bedtime_consistency' in progress_data['key_metrics']:
        bedtime_consistency = progress_data['key_metrics']['bedtime_consistency']
        if bedtime_consistency > 0.8:
            confidence += 0.1  # High consistency leads to better predictions
        elif bedtime_consistency < 0.4:
            confidence -= 0.1  # Low consistency reduces prediction confidence
    
    # Factor 4: Recent changes/trends affect prediction confidence
    if 'improvement_rate' in progress_data:
        abs_rate = abs(progress_data['improvement_rate'])
        if abs_rate > 0.02:  # Significant change recently
            confidence -= 0.1  # Rapid changes reduce prediction confidence
    
    # Factor 5: Account for insomnia pattern which is harder to predict
    if 'trend' in progress_data:
        if progress_data['trend'] in ['severe_insomnia', 'moderate_insomnia']:
            confidence -= 0.15  # Insomnia patterns are less predictable
    
    # Ensure confidence is within valid range [0.1, 0.95]
    confidence = max(0.1, min(0.95, confidence))
    
    # Round to 2 decimal places
    return round(confidence, 2)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)