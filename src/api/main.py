# src/api/main.py

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import sys
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recommendation_engine import SleepRecommendationEngine
from models.sleep_quality import SleepQualityModel
from data_processing.preprocessing import Preprocessor

# Initialize application
app = FastAPI(
    title="Sleep Insights API",
    description="API for analyzing sleep data and generating recommendations for insomnia",
    version="0.1.0"
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

# API routes
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Sleep Insights API",
        "version": "0.1.0",
        "documentation": "/docs"
    }

@app.post("/sleep/log", status_code=201)
async def log_sleep(entry: SleepEntry):
    """Log a new sleep entry for a user and return all available predictions"""
    try:
        # Convert to dictionary for easier handling
        entry_dict = entry.dict()
        
        # Create a dataframe from the entry
        entry_df = pd.DataFrame([entry_dict])
        
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
                "note": "Not enough historical data for comprehensive analysis. Continue logging sleep for more insights."
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
        
        # 5. Generate recommendation
        recommendation = recommendation_engine.generate_recommendation(entry.user_id, progress_data)
        
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
        
        # 7. Make predictions about future sleep
        # Calculate averages for recent data (last 7 days)
        recent_data = processed_data.tail(7)
        avg_efficiency = recent_data['sleep_efficiency'].mean() if 'sleep_efficiency' in recent_data else None
        efficiency_trend = progress_data.get('improvement_rate', 0)
        
        # Predict tomorrow's efficiency based on trend
        predicted_efficiency = None
        if avg_efficiency is not None:
            predicted_efficiency = min(1.0, max(0, avg_efficiency + efficiency_trend))
        
        # Predict optimal bedtime and wake time
        optimal_bedtime = None
        optimal_waketime = None
        
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
        
        # 8. Return comprehensive response
        return {
            "status": "success",
            "message": "Sleep entry logged successfully",
            
            # Entry details
            "entry_date": entry.date,
            "sleep_score": sleep_score,
            
            # Analysis
            "trend": progress_data.get('trend'),
            "consistency": progress_data.get('consistency'),
            "key_metrics": progress_data.get('key_metrics'),
            
            # Recommendation
            "recommendation": recommendation,
            
            # Predictions
            "predictions": {
                "estimated_next_efficiency": round(predicted_efficiency * 100) if predicted_efficiency else None,
                "optimal_bedtime": optimal_bedtime,
                "optimal_waketime": optimal_waketime,
                "expected_improvement": "increasing" if efficiency_trend > 0.005 else 
                                       "decreasing" if efficiency_trend < -0.005 else "stable"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging sleep entry: {str(e)}")

@app.get("/sleep/analysis/{user_id}")
async def analyze_sleep(user_id: str, days: int = 30):
    """Analyze sleep data for a user"""
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
        
        # Generate recommendation
        recommendation = recommendation_engine.generate_recommendation(user_id, progress_data)
        
        # Add recommendation to result
        result = progress_data.copy()
        result['recommendations'] = recommendation
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sleep data: {str(e)}")

@app.get("/sleep/recommendation/{user_id}")
async def get_recommendation(user_id: str, days: int = 30):
    """Get a personalized recommendation for a user"""
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
        
        # Generate recommendation
        recommendation = recommendation_engine.generate_recommendation(user_id, progress_data)
        
        # Store recommendation
        rec_dir = 'data/recommendations'
        os.makedirs(rec_dir, exist_ok=True)
        
        user_rec_file = os.path.join(rec_dir, f"{user_id}_recommendations.csv")
        rec_data = {
            'user_id': user_id,
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
        
        return {
            "user_id": user_id,
            "recommendation": recommendation,
            "timestamp": rec_data['timestamp']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

@app.get("/sleep/score/{user_id}")
async def calculate_sleep_score(user_id: str, date: Optional[str] = None):
    """Calculate sleep score for a specific user and date"""
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
        
        # Filter by date if provided
        if date:
            user_data = user_data[user_data['date'] == date]
            
            if len(user_data) == 0:
                raise HTTPException(status_code=404, detail=f"No sleep data found for user {user_id} on {date}")
        else:
            # Use most recent date
            user_data['date'] = pd.to_datetime(user_data['date'])
            user_data = user_data.sort_values('date', ascending=False).head(1)
        
        # Check for no_sleep flag
        if 'no_sleep' in user_data.columns and user_data['no_sleep'].iloc[0]:
            return {
                "user_id": user_id,
                "date": user_data['date'].iloc[0],
                "sleep_score": 0,
                "note": "No sleep recorded for this night"
            }
        
        # Preprocess data
        processed_data = preprocessor.preprocess_sleep_data(user_data)
        
        # Prepare data for score calculation
        row = processed_data.iloc[0]
        
        # Create additional metrics dictionary
        additional_metrics = {}
        for col in ['deep_sleep_percentage', 'rem_sleep_percentage', 'sleep_onset_latency_minutes', 'awakenings_count']:
            if col in row:
                additional_metrics[col] = row[col]
        
        # Calculate score
        try:
            score = sleep_quality_model.calculate_sleep_score(
                row['sleep_efficiency'], 
                row.get('subjective_rating'),
                additional_metrics
            )
        except Exception as e:
            # Fallback if model fails
            print(f"Warning: Model-based score calculation failed: {str(e)}")
            # Simple calculation
            score = int(row['sleep_efficiency'] * 80)
            if 'subjective_rating' in row:
                score += int(row['subjective_rating'] * 2)
            score = min(100, max(0, score))
        
        return {
            "user_id": user_id,
            "date": user_data['date'].iloc[0] if isinstance(user_data['date'].iloc[0], str) else user_data['date'].iloc[0].strftime("%Y-%m-%d"),
            "sleep_score": score,
            "sleep_efficiency": float(row['sleep_efficiency']),
            "subjective_rating": int(row['subjective_rating']) if 'subjective_rating' in row else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating sleep score: {str(e)}")

@app.get("/sleep/history/{user_id}")
async def get_sleep_history(user_id: str, days: int = 30):
    """Get sleep history for a user"""
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
        
        # Ensure dates are in datetime format
        user_data['date'] = pd.to_datetime(user_data['date'])
        
        # Filter to requested days
        start_date = datetime.now() - timedelta(days=days)
        user_data = user_data[user_data['date'] >= start_date]
        
        # Sort by date
        user_data = user_data.sort_values('date')
        
        # Calculate sleep scores
        sleep_scores = []
        processed_data = preprocessor.preprocess_sleep_data(user_data)
        
        for _, row in processed_data.iterrows():
            # Skip no-sleep nights
            if 'no_sleep' in row and row['no_sleep']:
                sleep_scores.append(0)
                continue
                
            # Create additional metrics dictionary
            additional_metrics = {}
            for col in ['deep_sleep_percentage', 'rem_sleep_percentage', 'sleep_onset_latency_minutes', 'awakenings_count']:
                if col in row:
                    additional_metrics[col] = row[col]
            
            # Calculate score
            try:
                score = sleep_quality_model.calculate_sleep_score(
                    row['sleep_efficiency'], 
                    row.get('subjective_rating'),
                    additional_metrics
                )
            except Exception as e:
                # Fallback if model fails
                score = int(row['sleep_efficiency'] * 80)
                if 'subjective_rating' in row:
                    score += int(row['subjective_rating'] * 2)
                score = min(100, max(0, score))
            
            sleep_scores.append(score)
        
        # Add scores to data
        processed_data['sleep_score'] = sleep_scores
        
        # Convert to dictionary format for API response
        history = []
        for _, row in processed_data.iterrows():
            entry = {
                "date": row['date'].strftime("%Y-%m-%d"),
                "sleep_score": row['sleep_score'],
                "sleep_efficiency": float(row['sleep_efficiency']) if 'sleep_efficiency' in row else None,
                "subjective_rating": int(row['subjective_rating']) if 'subjective_rating' in row else None,
                "time_in_bed_hours": float(row['time_in_bed_hours']) if 'time_in_bed_hours' in row else None,
                "sleep_duration_hours": float(row['sleep_duration_hours']) if 'sleep_duration_hours' in row else None,
                "awakenings_count": int(row['awakenings_count']) if 'awakenings_count' in row else None,
                "time_awake_minutes": int(row['time_awake_minutes']) if 'time_awake_minutes' in row else None,
                "no_sleep": bool(row['no_sleep']) if 'no_sleep' in row else False
            }
            history.append(entry)
        
        return {
            "user_id": user_id,
            "days": days,
            "entries": history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sleep history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)