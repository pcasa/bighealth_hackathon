# Update to src/api/routes/wearable_routes.py
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from typing import Dict, List, Optional
import pandas as pd
import io
import json

from src.core.wearables.wearable_transformer_manager import WearableTransformerManager
from src.core.services.sleep_service import SleepService
from src.core.repositories.data_repository import DataRepository
from src.core.models.sleep_quality import SleepQualityModel
from src.core.recommendation.recommendation_engine import SleepRecommendationEngine
from src.core.data_processing.preprocessing import Preprocessor

# Dependency
def get_wearable_service():
    repository = DataRepository()
    wearable_manager = WearableTransformerManager()
    preprocessor = Preprocessor()
    sleep_quality_model = SleepQualityModel()
    recommendation_engine = SleepRecommendationEngine()
    
    # Try to load sleep quality model
    try:
        sleep_quality_model.load('models/sleep_quality_model')
    except Exception as e:
        print(f"Warning: Could not load sleep quality model: {str(e)}")
    
    sleep_service = SleepService(repository, sleep_quality_model, recommendation_engine, preprocessor)
    
    return wearable_manager, sleep_service, repository

router = APIRouter(
    prefix="/wearable",
    tags=["Wearable Data"],
    responses={404: {"description": "Not found"}}
)

@router.post("/upload", response_model=Dict, status_code=201)
async def upload_wearable_data(
    file: UploadFile = File(...),
    device_type: str = Form(...),
    user_id: str = Form(...),
    services: tuple = Depends(get_wearable_service)
):
    """Upload and process wearable data file"""
    wearable_manager, sleep_service, repository = services
    
    try:
        # Read file content
        content = await file.read()
        
        # Process based on file type
        if file.filename.endswith('.csv'):
            # Parse CSV
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            # Parse JSON
            df = pd.DataFrame(json.loads(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or JSON.")
        
        # Add user_id if not present
        if 'user_id' not in df.columns:
            df['user_id'] = user_id
        
        # Get user profile
        user_df = repository.get_user_data(user_id)
        
        if len(user_df) == 0:
            raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
        
        # Transform wearable data
        transformed_data = wearable_manager.transform_data(df, device_type, user_df)
        
        if len(transformed_data) == 0:
            raise HTTPException(status_code=400, detail="Could not transform wearable data")
        
        # Save transformed data
        for _, row in transformed_data.iterrows():
            # Create sleep entry
            sleep_entry = {
                'user_id': row['user_id'],
                'date': row['date'],
                'bedtime': row.get('device_bedtime', None),
                'sleep_onset_time': row.get('device_sleep_onset', None),
                'wake_time': row.get('device_wake_time', None),
                'sleep_duration_hours': row.get('device_sleep_duration', None),
                'sleep_efficiency': row.get('sleep_efficiency', None),
                'deep_sleep_percentage': row.get('deep_sleep_percentage', None),
                'rem_sleep_percentage': row.get('rem_sleep_percentage', None),
                'light_sleep_percentage': row.get('light_sleep_percentage', None),
                'heart_rate_variability': row.get('heart_rate_variability', None),
                'average_heart_rate': row.get('average_heart_rate', None)
            }
            
            # Filter out None values
            sleep_entry = {k: v for k, v in sleep_entry.items() if v is not None}
            
            # Check if entry already exists
            existing_entries = repository.get_sleep_data(user_id=user_id)
            existing_entries = existing_entries[existing_entries['date'] == sleep_entry['date']]
            
            if len(existing_entries) > 0:
                # Update existing entry
                for key, value in sleep_entry.items():
                    existing_entries.loc[existing_entries.index[0], key] = value
                repository.save_sleep_entry(existing_entries.iloc[0].to_dict())
            else:
                # Save new entry
                repository.save_sleep_entry(sleep_entry)
        
        # Analyze sleep data
        analysis_result = await sleep_service.analyze_sleep(user_id)
        
        return {
            "status": "success",
            "message": f"Successfully processed {len(transformed_data)} records",
            "device_type": device_type,
            "analysis": analysis_result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing wearable data: {str(e)}")

@router.get("/devices", response_model=List[str])
async def get_supported_devices(services: tuple = Depends(get_wearable_service)):
    """Get list of supported wearable devices"""
    wearable_manager, _, _ = services
    
    return wearable_manager.get_supported_devices()

@router.get("/{user_id}", response_model=Dict)
async def get_wearable_data(user_id: str, services: tuple = Depends(get_wearable_service)):
    """Get all wearable data for a user"""
    _, _, repository = services
    
    # Get user profile
    user_df = repository.get_user_data(user_id)
    
    if len(user_df) == 0:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
    
    # Get sleep data for user
    sleep_data = repository.get_sleep_data(user_id=user_id)
    
    if len(sleep_data) == 0:
        raise HTTPException(status_code=404, detail=f"No sleep data found for user with ID {user_id}")
    
    # Filter to records with wearable data
    wearable_columns = [
        'deep_sleep_percentage', 'rem_sleep_percentage', 'light_sleep_percentage',
        'heart_rate_variability', 'average_heart_rate'
    ]
    
    has_wearable = False
    for col in wearable_columns:
        if col in sleep_data.columns:
            has_wearable = True
            break
    
    if not has_wearable:
        return {
            "user_id": user_id,
            "wearable_data": False,
            "message": "No wearable data found for this user"
        }
    
    # Extract wearable data from sleep records
    wearable_data = []
    
    for _, row in sleep_data.iterrows():
        record = {
            'date': row['date']
        }
        
        # Copy wearable-specific columns
        for col in wearable_columns:
            if col in row and pd.notna(row[col]):
                record[col] = row[col]
        
        # Only add if it has some wearable data
        if len(record) > 1:  # More than just the date
            wearable_data.append(record)
    
    return {
        "user_id": user_id,
        "wearable_data": True,
        "data_count": len(wearable_data),
        "records": wearable_data
    }

@router.get("/{user_id}/summary", response_model=Dict)
async def get_wearable_summary(user_id: str, services: tuple = Depends(get_wearable_service)):
    """Get summary statistics of wearable data for a user"""
    _, sleep_service, repository = services
    
    # Get user sleep data
    sleep_data = repository.get_sleep_data(user_id=user_id)
    
    if len(sleep_data) == 0:
        raise HTTPException(status_code=404, detail=f"No sleep data found for user with ID {user_id}")
    
    # Check for wearable data
    wearable_columns = [
        'deep_sleep_percentage', 'rem_sleep_percentage', 'light_sleep_percentage',
        'heart_rate_variability', 'average_heart_rate'
    ]
    
    has_wearable = False
    for col in wearable_columns:
        if col in sleep_data.columns and not sleep_data[col].isna().all():
            has_wearable = True
            break
    
    if not has_wearable:
        return {
            "user_id": user_id,
            "wearable_data": False,
            "message": "No wearable data found for this user"
        }
    
    # Calculate summary statistics
    summary = {
        "user_id": user_id,
        "wearable_data": True,
        "record_count": len(sleep_data),
        "stats": {}
    }
    
    for col in wearable_columns:
        if col in sleep_data.columns and not sleep_data[col].isna().all():
            summary["stats"][col] = {
                "avg": sleep_data[col].mean(),
                "min": sleep_data[col].min(),
                "max": sleep_data[col].max(),
                "count": sleep_data[col].count()
            }
    
    # Get latest date with wearable data
    latest_date = None
    for col in wearable_columns:
        if col in sleep_data.columns:
            # Get most recent date with non-null value for this column
            date_with_data = sleep_data[~sleep_data[col].isna()]['date'].max()
            if date_with_data and (latest_date is None or date_with_data > latest_date):
                latest_date = date_with_data
    
    summary["latest_data_date"] = latest_date
    
    return summary