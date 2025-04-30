# src/api/sleep_routes.py
from fastapi import APIRouter, Depends, HTTPException
from src.core.services import SleepService

router = APIRouter(prefix="/sleep", tags=["Sleep"])

@router.post("/log")
async def log_sleep(entry: SleepEntryModel, service: SleepService = Depends(get_sleep_service)):
    """Log sleep entry with optimized service handling"""
    try:
        result = await service.log_sleep_entry(entry)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))