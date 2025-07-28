from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import settings
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create app
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="OMHA Exhibit Tagging Service using AWS Bedrock"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    logging.info(f"Starting {settings.app_name}")
    logging.info(f"Using model: {settings.bedrock_model_id}")

@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "model": settings.bedrock_model_id
    }