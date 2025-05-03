# src/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import sleep_routes, user_profile_routes

app = FastAPI(
    title="Sleep Insights API",
    description="API for analyzing sleep data and generating recommendations",
    version="0.3.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sleep_routes.router)
app.include_router(user_profile_routes.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Sleep Insights API",
        "version": "0.3.0",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)