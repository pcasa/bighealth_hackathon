# src/api/user_profile_routes.py

from fastapi import APIRouter, HTTPException, Depends, Body, Query
from typing import List, Dict, Optional
from pydantic import BaseModel
import pandas as pd
import os
from datetime import datetime
from src.utils.constants import profession_categories

router = APIRouter(
    prefix="/user",
    tags=["User Profiles"],
    responses={404: {"description": "Not found"}},
)

# Define data models
class UserProfile(BaseModel):
    user_id: str
    age: int
    gender: str
    profession: str
    region: str
    sleep_pattern: str
    device_type: str
    data_consistency: float
    sleep_consistency: float
    created_at: str
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "abc12345",
                "age": 34,
                "gender": "female",
                "profession": "Software Engineer",
                "region": "Seattle, Washington, United States",
                "sleep_pattern": "variable",
                "device_type": "apple_watch",
                "data_consistency": 0.85,
                "sleep_consistency": 0.72,
                "created_at": "2025-04-27 10:15:30"
            }
        }

class UserProfileCreate(BaseModel):
    age: int
    gender: str
    profession: str
    region: str
    sleep_pattern: str
    device_type: str
    
    class Config:
        schema_extra = {
            "example": {
                "age": 34,
                "gender": "female",
                "profession": "Software Engineer",
                "region": "Seattle, Washington, United States",
                "sleep_pattern": "variable",
                "device_type": "apple_watch"
            }
        }

class UserProfileUpdate(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    profession: Optional[str] = None
    region: Optional[str] = None
    sleep_pattern: Optional[str] = None
    device_type: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "profession": "Senior Software Engineer",
                "region": "Bellevue, Washington, United States"
            }
        }

class ProfessionStats(BaseModel):
    profession: str
    count: int
    avg_sleep_efficiency: float
    avg_sleep_duration: float
    avg_subjective_rating: float

class RegionStats(BaseModel):
    region: str
    count: int
    avg_sleep_efficiency: float
    avg_sleep_duration: float
    avg_subjective_rating: float

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

def load_sleep_data():
    """Load sleep data from CSV file"""
    data_path = 'data/raw/sleep_data.csv'
    if not os.path.exists(data_path):
        return pd.DataFrame()
    
    return pd.read_csv(data_path)

def save_user_data(users_df):
    """Save user data to CSV file"""
    data_path = 'data/raw/users.csv'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    users_df.to_csv(data_path, index=False)

# Routes
@router.get("/", response_model=List[UserProfile])
async def get_all_users(limit: int = Query(100, ge=1, le=1000)):
    """Get all user profiles"""
    users_df = load_user_data()
    
    # Limit the number of users returned
    if len(users_df) > limit:
        users_df = users_df.head(limit)
    
    # Convert to list of dictionaries
    users = users_df.to_dict('records')
    
    return users

@router.get("/{user_id}", response_model=UserProfile)
async def get_user(user_id: str):
    """Get a specific user profile by ID"""
    users_df = load_user_data()
    
    # Find the user
    user = users_df[users_df['user_id'] == user_id]
    
    if len(user) == 0:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    return user.iloc[0].to_dict()

@router.post("/", response_model=UserProfile, status_code=201)
async def create_user(user: UserProfileCreate):
    """Create a new user profile"""
    users_df = load_user_data()
    
    # Generate a new user ID
    import uuid
    user_id = str(uuid.uuid4())[:8]
    
    # Check if user ID already exists (unlikely but possible)
    while user_id in users_df['user_id'].values:
        user_id = str(uuid.uuid4())[:8]
    
    # Create new user record
    new_user = user.dict()
    new_user['user_id'] = user_id
    new_user['data_consistency'] = 0.85  # Default value
    new_user['sleep_consistency'] = 0.75  # Default value
    new_user['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Add to dataframe
    users_df = pd.concat([users_df, pd.DataFrame([new_user])], ignore_index=True)
    
    # Save updated data
    save_user_data(users_df)
    
    return new_user

@router.put("/{user_id}", response_model=UserProfile)
async def update_user(user_id: str, user_update: UserProfileUpdate):
    """Update a user profile"""
    users_df = load_user_data()
    
    # Find the user
    user_mask = users_df['user_id'] == user_id
    
    if not user_mask.any():
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    # Update user fields
    update_data = user_update.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        users_df.loc[user_mask, field] = value
    
    # Save updated data
    save_user_data(users_df)
    
    # Return updated user
    return users_df[user_mask].iloc[0].to_dict()

@router.delete("/{user_id}", status_code=204)
async def delete_user(user_id: str):
    """Delete a user profile"""
    users_df = load_user_data()
    
    # Find the user
    user_mask = users_df['user_id'] == user_id
    
    if not user_mask.any():
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    # Remove user
    users_df = users_df[~user_mask]
    
    # Save updated data
    save_user_data(users_df)
    
    return None

@router.get("/stats/profession", response_model=List[ProfessionStats])
async def get_profession_stats():
    """Get sleep statistics by profession category"""
    users_df = load_user_data()
    sleep_df = load_sleep_data()
    
    if len(users_df) == 0 or len(sleep_df) == 0:
        return []
    
    # Categorize professions
    def categorize_profession(profession):
        for category, keywords in profession_categories.items():
            if any(keyword.lower() in profession.lower() for keyword in keywords):
                return category
        return "other"
    
    users_df['profession_category'] = users_df['profession'].apply(categorize_profession)
    
    # Merge user and sleep data
    merged_df = pd.merge(sleep_df, users_df[['user_id', 'profession_category']], on='user_id')
    
    # Calculate statistics by profession
    prof_stats = merged_df.groupby('profession_category').agg({
        'sleep_efficiency': 'mean',
        'sleep_duration_hours': 'mean',
        'subjective_rating': 'mean',
        'user_id': 'count'
    }).reset_index()
    
    # Format for API response
    result = []
    for _, row in prof_stats.iterrows():
        result.append({
            "profession": row['profession_category'],
            "count": int(row['user_id']),
            "avg_sleep_efficiency": float(row['sleep_efficiency']),
            "avg_sleep_duration": float(row['sleep_duration_hours']),
            "avg_subjective_rating": float(row['subjective_rating'])
        })
    
    return result

@router.get("/stats/region", response_model=List[RegionStats])
async def get_region_stats():
    """Get sleep statistics by region category"""
    users_df = load_user_data()
    sleep_df = load_sleep_data()
    
    if len(users_df) == 0 or len(sleep_df) == 0:
        return []
    
    # Extract region from full address (City, State, Country)
    def extract_region_category(region):
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
    
    users_df['region_category'] = users_df['region'].apply(extract_region_category)
    
    # Merge user and sleep data
    merged_df = pd.merge(sleep_df, users_df[['user_id', 'region_category']], on='user_id')
    
    # Calculate statistics by region
    region_stats = merged_df.groupby('region_category').agg({
        'sleep_efficiency': 'mean',
        'sleep_duration_hours': 'mean',
        'subjective_rating': 'mean',
        'user_id': 'count'
    }).reset_index()
    
    # Format for API response
    result = []
    for _, row in region_stats.iterrows():
        result.append({
            "region": row['region_category'],
            "count": int(row['user_id']),
            "avg_sleep_efficiency": float(row['sleep_efficiency']),
            "avg_sleep_duration": float(row['sleep_duration_hours']),
            "avg_subjective_rating": float(row['subjective_rating'])
        })
    
    return result