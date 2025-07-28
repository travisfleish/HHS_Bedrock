from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import settings
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('omha_tagger.log', mode='a')
    ]
)

# Set specific loggers to debug level for more detail
logging.getLogger("app.services.exhibit_tagger").setLevel(logging.DEBUG)
logging.getLogger("app.services.bedrock_service").setLevel(logging.INFO)

# Create app
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="OMHA Exhibit Tagging Service"
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
    if settings.use_openai:
        logging.info(f"Using OpenAI with model: {settings.openai_model}")
    else:
        logging.info(f"Using AWS Bedrock with model: {settings.bedrock_model_id}")


@app.get("/")
async def root():
    service_type = "OpenAI" if settings.use_openai else "AWS Bedrock"
    model = settings.openai_model if settings.use_openai else settings.bedrock_model_id

    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "ai_service": service_type,
        "model": model
    }