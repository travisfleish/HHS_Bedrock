# OMHA Exhibit Tagger

An AI-powered service for classifying Medicare appeal exhibits using AWS Bedrock.

## Features

- Document classification using AWS Bedrock (Claude 3)
- Support for PDF, DOCX, and text files
- RESTful API with FastAPI
- Configurable taxonomy
- Docker support

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and add your AWS credentials
3. Install dependencies: `pip install -r requirements.txt`
4. Run the server: `uvicorn app.main:app --reload`

## API Endpoints

- `POST /api/v1/tag-exhibits` - Classify multiple documents
- `GET /api/v1/health` - Health check
- `GET /api/v1/taxonomy` - Get current taxonomy

## Docker

Run with Docker Compose:

    docker-compose up

## Testing

    pytest tests/