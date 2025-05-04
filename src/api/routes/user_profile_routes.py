# src/api/routes/user_profile_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Optional

# from src.api.models.user_models import UserProfileCreate, UserProfileUpdate
from src.core.models.data_models import UserProfile
from src.core.services.user_service import UserService
from src.core.repositories.data_repository import DataRepository

# Dependency
def get_user_service():
    repository = DataRepository()
    return UserService(repository)

router = APIRouter(
    prefix="/user",
    tags=["User Profiles"],
    responses={404: {"description": "Not found"}}
)

@router.get("/", response_model=List[Dict])
async def get_all_users(
    limit: int = Query(100, ge=1, le=1000),
    service: UserService = Depends(get_user_service)
):
    """Get all user profiles"""
    return await service.get_all_users(limit)

@router.get("/{user_id}", response_model=Dict)
async def get_user(user_id: str, service: UserService = Depends(get_user_service)):
    """Get a specific user profile by ID"""
    user = await service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user

# @router.post("/", response_model=Dict, status_code=201)
# async def create_user(user: UserProfileCreate, service: UserService = Depends(get_user_service)):
#     """Create a new user profile"""
#     return await service.create_user(user.dict())

# @router.put("/{user_id}", response_model=Dict)
# async def update_user(
#     user_id: str, 
#    user_update: UserProfileUpdate, 
#     service: UserService = Depends(get_user_service)
# ):
#     """Update a user profile"""
#     user = await service.update_user(user_id, user_update.dict(exclude_unset=True))
#     if not user:
#         raise HTTPException(status_code=404, detail=f"User {user_id} not found")
#     return user

@router.delete("/{user_id}", status_code=204)
async def delete_user(user_id: str, service: UserService = Depends(get_user_service)):
    """Delete a user profile"""
    success = await service.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return None

@router.get("/stats/profession", response_model=List[Dict])
async def get_profession_stats(service: UserService = Depends(get_user_service)):
    """Get sleep statistics by profession category"""
    return await service.get_profession_stats()