#!/usr/bin/env python3
"""
Create OMHA Exhibit Tagger project structure with all files and content
"""

import os
import textwrap


def create_file(path, content):
    """Create a file with the given content"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(textwrap.dedent(content).strip())
    print(f"Created: {path}")


# Project root
PROJECT_ROOT = "omha-exhibit-tagger"

# File contents
files = {
    "app/__init__.py": "",

    "app/config.py": '''
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # AWS Settings
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # Bedrock Settings
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    max_tokens: int = 4096
    temperature: float = 0.1

    # App Settings
    app_name: str = "OMHA Exhibit Tagger"
    debug: bool = False

    # Taxonomy Version
    taxonomy_version: str = "2.1"

    class Config:
        env_file = ".env"

settings = Settings()
''',

    "app/models.py": '''
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    # Legal Documents
    RECONSIDERATION_DECISION = "reconsideration_decision"
    ALJ_APPEAL_REQUEST = "alj_appeal_request"
    ALJ_DECISION = "alj_decision"

    # Medical Evidence
    PHYSICIAN_ORDER = "physician_order"
    DIAGNOSTIC_REPORT = "diagnostic_report"
    CLINICAL_NOTES = "clinical_notes"
    DISCHARGE_SUMMARY = "discharge_summary"

    # Administrative
    REMITTANCE_ADVICE = "remittance_advice"
    CORRESPONDENCE = "correspondence"

    # Unknown
    UNCLASSIFIED = "unclassified"

class ExhibitClassification(BaseModel):
    exhibit_id: str
    document_type: DocumentType
    confidence: float = Field(ge=0, le=1)
    key_indicators: List[str]
    requires_review: bool
    extracted_data: Optional[Dict] = None

class DocumentUpload(BaseModel):
    filename: str
    content: str  # Base64 encoded
    mime_type: str

class TaggingRequest(BaseModel):
    case_id: str
    documents: List[DocumentUpload]
    extract_detailed_data: bool = False

class TaggingResponse(BaseModel):
    case_id: str
    processing_time: float
    total_exhibits: int
    classifications: List[ExhibitClassification]
    taxonomy_version: str
    timestamp: datetime
''',

    "app/main.py": '''
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
''',

    "app/services/__init__.py": "",

    "app/services/bedrock_service.py": '''
import boto3
import json
from typing import Dict, Any
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class BedrockService:
    def __init__(self):
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
        self.model_id = settings.bedrock_model_id

    async def invoke_model(self, prompt: str, system_prompt: str) -> str:
        """Invoke Bedrock model with prompt"""
        try:
            # Format for Claude 3
            if "anthropic" in self.model_id:
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": settings.max_tokens,
                    "temperature": settings.temperature,
                    "system": system_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            # Add other model formats as needed
            else:
                raise ValueError(f"Unsupported model: {self.model_id}")

            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(request_body)
            )

            response_body = json.loads(response['body'].read())

            # Extract text based on model
            if "anthropic" in self.model_id:
                return response_body['content'][0]['text']

        except Exception as e:
            logger.error(f"Bedrock invocation failed: {str(e)}")
            raise
''',

    "app/services/document_processor.py": '''
from typing import List, Dict
import base64
import PyPDF2
from docx import Document
import io

class DocumentProcessor:
    """Process various document formats"""

    @staticmethod
    def extract_text_from_pdf(content: bytes) -> str:
        """Extract text from PDF content"""
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_parts = []

        for page in pdf_reader.pages:
            text_parts.append(page.extract_text())

        return '\\n'.join(text_parts)

    @staticmethod
    def extract_text_from_docx(content: bytes) -> str:
        """Extract text from DOCX content"""
        doc_file = io.BytesIO(content)
        doc = Document(doc_file)

        paragraphs = [p.text for p in doc.paragraphs]
        return '\\n'.join(paragraphs)

    @staticmethod
    def process_document(content: str, mime_type: str) -> str:
        """Process document based on mime type"""
        # Decode base64 content
        content_bytes = base64.b64decode(content)

        if mime_type == "application/pdf":
            return DocumentProcessor.extract_text_from_pdf(content_bytes)
        elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                           "application/msword"]:
            return DocumentProcessor.extract_text_from_docx(content_bytes)
        elif mime_type == "text/plain":
            return content_bytes.decode('utf-8')
        else:
            # Try to decode as text
            return content_bytes.decode('utf-8')
''',

    "app/services/exhibit_tagger.py": '''
import json
from typing import List, Dict
from app.models import DocumentType, ExhibitClassification
from app.services.bedrock_service import BedrockService
from app.utils.taxonomy import get_taxonomy_prompt
import logging

logger = logging.getLogger(__name__)

class ExhibitTagger:
    def __init__(self, bedrock_service: BedrockService):
        self.bedrock = bedrock_service
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build system prompt with current taxonomy"""
        taxonomy = get_taxonomy_prompt()

        return f"""You are an OMHA exhibit classifier for Medicare appeals. 

{taxonomy}

CRITICAL RULES:
1. Classify each document using ONLY the terms from the taxonomy above
2. Output JSON with: document_type, confidence (0-1), key_indicators (list), requires_review (boolean)
3. Set requires_review=true if confidence < 0.7 or document has ambiguous classification
4. For key_indicators, list 3-5 specific text elements that led to your classification

Examples:
- "DISCHARGE SUMMARY" header → discharge_summary
- "Medicare Reconsideration Decision" → reconsideration_decision
- "CT THORAX WITH CONTRAST" → diagnostic_report
"""

    async def classify_document(self, document_text: str, exhibit_id: str) -> ExhibitClassification:
        """Classify a single document"""

        prompt = f"""Classify this medical document:

DOCUMENT:
{document_text[:8000]}  # Limit for context window

OUTPUT FORMAT:
{{
    "document_type": "exact_term_from_taxonomy",
    "confidence": 0.95,
    "key_indicators": ["indicator1", "indicator2", "indicator3"],
    "requires_review": false
}}
"""

        try:
            response = await self.bedrock.invoke_model(prompt, self.system_prompt)

            # Parse JSON response
            classification_data = json.loads(response)

            # Validate document type
            try:
                doc_type = DocumentType(classification_data['document_type'])
            except ValueError:
                doc_type = DocumentType.UNCLASSIFIED
                classification_data['requires_review'] = True

            return ExhibitClassification(
                exhibit_id=exhibit_id,
                document_type=doc_type,
                confidence=classification_data.get('confidence', 0.0),
                key_indicators=classification_data.get('key_indicators', []),
                requires_review=classification_data.get('requires_review', True)
            )

        except Exception as e:
            logger.error(f"Classification failed for exhibit {exhibit_id}: {str(e)}")
            return ExhibitClassification(
                exhibit_id=exhibit_id,
                document_type=DocumentType.UNCLASSIFIED,
                confidence=0.0,
                key_indicators=["Classification error"],
                requires_review=True
            )

    async def classify_batch(self, documents: List[Dict]) -> List[ExhibitClassification]:
        """Classify multiple documents"""
        classifications = []

        for doc in documents:
            classification = await self.classify_document(
                doc['content'], 
                doc['exhibit_id']
            )
            classifications.append(classification)

        return classifications
''',

    "app/api/__init__.py": "",

    "app/api/routes.py": '''
from fastapi import APIRouter, HTTPException, Depends
from typing import List
import time
import base64
from datetime import datetime

from app.models import TaggingRequest, TaggingResponse, ExhibitClassification
from app.services.bedrock_service import BedrockService
from app.services.exhibit_tagger import ExhibitTagger
from app.services.document_processor import DocumentProcessor
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["exhibit-tagger"])

# Dependency injection
def get_bedrock_service():
    return BedrockService()

def get_exhibit_tagger(bedrock: BedrockService = Depends(get_bedrock_service)):
    return ExhibitTagger(bedrock)

@router.post("/tag-exhibits", response_model=TaggingResponse)
async def tag_exhibits(
    request: TaggingRequest,
    tagger: ExhibitTagger = Depends(get_exhibit_tagger)
):
    """Tag multiple exhibit documents"""
    start_time = time.time()

    try:
        # Process documents
        documents = []
        for idx, doc in enumerate(request.documents):
            # Extract text based on mime type
            text_content = DocumentProcessor.process_document(doc.content, doc.mime_type)

            documents.append({
                'exhibit_id': f"{request.case_id}_exhibit_{idx+1}",
                'content': text_content,
                'filename': doc.filename
            })

        # Classify documents
        classifications = await tagger.classify_batch(documents)

        # Build response
        processing_time = time.time() - start_time

        return TaggingResponse(
            case_id=request.case_id,
            processing_time=processing_time,
            total_exhibits=len(classifications),
            classifications=classifications,
            taxonomy_version=settings.taxonomy_version,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": settings.bedrock_model_id}

@router.get("/taxonomy")
async def get_taxonomy():
    """Get current taxonomy"""
    from app.utils.taxonomy import get_full_taxonomy
    return get_full_taxonomy()
''',

    "app/utils/__init__.py": "",

    "app/utils/taxonomy.py": '''
import json
from typing import Dict

# This could be loaded from a database or config file
OMHA_TAXONOMY = {
    "version": "2.1",
    "categories": {
        "legal_documents": {
            "reconsideration_decision": {
                "description": "Medicare reconsideration decision letter",
                "indicators": ["reconsideration decision", "QIC", "unfavorable/favorable"]
            },
            "alj_appeal_request": {
                "description": "Request for ALJ hearing",
                "indicators": ["ALJ appeal", "administrative law judge", "hearing request"]
            },
            "alj_decision": {
                "description": "Administrative Law Judge decision",
                "indicators": ["ALJ decision", "hearing decision", "administrative law judge"]
            }
        },
        "medical_evidence": {
            "physician_order": {
                "description": "Prescription or order for equipment/service",
                "indicators": ["prescription", "order form", "physician signature", "Rx"]
            },
            "diagnostic_report": {
                "description": "Lab results, imaging reports",
                "indicators": ["CT", "MRI", "X-ray", "lab results", "findings", "impressions"]
            },
            "clinical_notes": {
                "description": "Progress notes, consultation notes, H&P",
                "indicators": ["progress note", "clinical note", "visit note", "consultation"]
            },
            "discharge_summary": {
                "description": "Hospital discharge summary",
                "indicators": ["discharge summary", "discharge instructions", "hospital course"]
            }
        },
        "administrative": {
            "remittance_advice": {
                "description": "Medicare payment/denial information",
                "indicators": ["remittance advice", "ERA", "claim status", "payment"]
            },
            "correspondence": {
                "description": "Letters, general correspondence",
                "indicators": ["letter", "correspondence", "RE:"]
            }
        }
    }
}

def get_taxonomy_prompt() -> str:
    """Convert taxonomy to prompt format"""
    prompt_lines = ["DOCUMENT CLASSIFICATION TAXONOMY:"]

    for category, types in OMHA_TAXONOMY['categories'].items():
        prompt_lines.append(f"\\n{category.upper()}:")
        for doc_type, details in types.items():
            prompt_lines.append(f"  - {doc_type}: {details['description']}")
            prompt_lines.append(f"    Indicators: {', '.join(details['indicators'])}")

    return "\\n".join(prompt_lines)

def get_full_taxonomy() -> Dict:
    """Get full taxonomy structure"""
    return OMHA_TAXONOMY
''',

    "requirements.txt": '''
fastapi==0.104.1
uvicorn[standard]==0.24.0
boto3==1.34.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
PyPDF2==3.0.1
python-docx==1.1.0
aiofiles==23.2.0
''',

    ".env": '''
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# App Configuration
APP_NAME=OMHA Exhibit Tagger
DEBUG=false
TAXONOMY_VERSION=2.1
''',

    ".gitignore": '''
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Environment variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Logs
*.log
logs/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Distribution
dist/
build/
*.egg-info/
''',

    "Dockerfile": '''
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ app/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
''',

    "docker-compose.yml": '''
version: '3.8'

services:
  omha-tagger:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AWS_REGION=${AWS_REGION}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID}
    volumes:
      - ./app:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
''',

    "README.md": """# OMHA Exhibit Tagger

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
""",

    "tests/__init__.py": "",

    "tests/test_exhibit_tagger.py": '''
import pytest
from app.services.exhibit_tagger import ExhibitTagger
from app.models import DocumentType

# Add your tests here
'''
}

# Create all files
for filepath, content in files.items():
    full_path = os.path.join(PROJECT_ROOT, filepath)
    create_file(full_path, content)

print(f"\n✅ Project created successfully in '{PROJECT_ROOT}' directory!")
print("\nNext steps:")
print("1. cd omha-exhibit-tagger")
print("2. python -m venv venv")
print("3. source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
print("4. pip install -r requirements.txt")
print("5. Update .env with your AWS credentials")
print("6. uvicorn app.main:app --reload")