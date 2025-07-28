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