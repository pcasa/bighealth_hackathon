# src/api/routes/sleep_routes.py
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List

from src.core.models.data_models import SleepEntry
from src.core.services.sleep_service import SleepService
from src.core.repositories.data_repository import DataRepository
from src.core.models.sleep_quality import SleepQualityModel
from src.core.recommendation.recommendation_engine import SleepRecommendationEngine
from src.core.data_processing.preprocessing import Preprocessor

# Dependency
def get_sleep_service():
   repository = DataRepository()
   sleep_quality_model = SleepQualityModel()
   recommendation_engine = SleepRecommendationEngine()
   preprocessor = Preprocessor()
   
   # Try to load sleep quality model
   try:
       sleep_quality_model.load('models/sleep_quality_model')
   except Exception as e:
       print(f"Warning: Could not load sleep quality model: {str(e)}")
       
   return SleepService(repository, sleep_quality_model, recommendation_engine, preprocessor)

router = APIRouter(
   prefix="/sleep",
   tags=["Sleep"],
   responses={404: {"description": "Not found"}}
)

@router.post("/log", response_model=Dict, status_code=201)
async def log_sleep(entry: SleepEntry, service: SleepService = Depends(get_sleep_service)):
   """Log a sleep entry and get analysis"""
   try:
       result = await service.log_sleep_entry(entry.dict())
       return result
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/{user_id}", response_model=Dict)
async def analyze_sleep(user_id: str, days: int = 30, service: SleepService = Depends(get_sleep_service)):
   """Analyze sleep data for a user"""
   try:
       result = await service.analyze_sleep(user_id, days)
       if "status" in result and result["status"] == "error":
           raise HTTPException(status_code=404, detail=result["message"])
       return result
   except HTTPException:
       raise
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendation/{user_id}", response_model=Dict)
async def get_recommendation(user_id: str, days: int = 30, service: SleepService = Depends(get_sleep_service)):
   """Get personalized sleep recommendation for a user"""
   try:
       result = await service.analyze_sleep(user_id, days)
       if "status" in result and result["status"] == "error":
           raise HTTPException(status_code=404, detail=result["message"])
       
       # Extract just the recommendation portion
       return {
           "user_id": user_id,
           "recommendation": result["recommendations"],
           "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "overall_confidence": result["overall_confidence"]
       }
   except HTTPException:
       raise
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))

@router.get("/detailed-score/{user_id}", response_model=Dict)
async def get_detailed_sleep_score(user_id: str, days: int = 200, service: SleepService = Depends(get_sleep_service)):
   """Get a detailed sleep score breakdown for a user's sleep record"""
   try:
       return await service.analyze_sleep(user_id, days)
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))