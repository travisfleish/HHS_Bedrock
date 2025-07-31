from fastapi import APIRouter, HTTPException, Depends
from typing import List
import time
import base64
from datetime import datetime
import logging

from app.models import TaggingRequest, TaggingResponse, ExhibitClassification
from app.services.bedrock_service import BedrockService
from app.services.exhibit_tagger import ExhibitTagger
from app.services.document_processor import DocumentProcessor
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["exhibit-tagger"])
logger = logging.getLogger(__name__)


# Dependency injection
def get_bedrock_service():
    return BedrockService()


def get_document_processor():
    """Get document processor with OCR settings"""
    return DocumentProcessor(
        ocr_enabled=getattr(settings, 'ocr_enabled', True),
        ocr_dpi=getattr(settings, 'ocr_dpi', 300),
        min_text_threshold=getattr(settings, 'min_text_threshold', 100),
        parallel_ocr=getattr(settings, 'parallel_ocr', True),
        max_workers=getattr(settings, 'ocr_max_workers', 4)
    )


def get_exhibit_tagger(bedrock: BedrockService = Depends(get_bedrock_service)):
    return ExhibitTagger(bedrock)


@router.post("/tag-exhibits", response_model=TaggingResponse)
async def tag_exhibits(
        request: TaggingRequest,
        tagger: ExhibitTagger = Depends(get_exhibit_tagger),
        doc_processor: DocumentProcessor = Depends(get_document_processor)
):
    """Tag multiple exhibit documents with enhanced extraction"""
    start_time = time.time()

    # Track extraction metrics (for logging only)
    extraction_metrics = {
        'total_extraction_time': 0,
        'ocr_pages': 0,
        'low_confidence_docs': 0,
        'extraction_warnings': []
    }

    try:
        # Process documents
        documents = []
        for idx, doc in enumerate(request.documents):
            exhibit_id = f"{request.case_id}_exhibit_{idx + 1}"

            # Extract text - handle both old and new processor versions
            result = doc_processor.process_document(doc.content, doc.mime_type)

            # Check if enhanced processor returned tuple
            if isinstance(result, tuple):
                text_content, extraction_result = result

                # Log extraction results
                logger.info(f"Exhibit {exhibit_id}: {extraction_result.method} extraction, "
                            f"{len(text_content)} chars, confidence: {extraction_result.confidence:.2f}")

                # Track metrics for logging
                extraction_metrics['total_extraction_time'] += extraction_result.extraction_time
                if extraction_result.method in ['ocr', 'mixed']:
                    extraction_metrics['ocr_pages'] += extraction_result.pages_processed
                if extraction_result.confidence < 0.5:
                    extraction_metrics['low_confidence_docs'] += 1
                if extraction_result.warnings:
                    extraction_metrics['extraction_warnings'].extend(
                        [f"{exhibit_id}: {w}" for w in extraction_result.warnings]
                    )

                # Store extraction result for later use
                extraction_result_obj = extraction_result
            else:
                # Old processor returns just text
                text_content = result
                extraction_result_obj = None
                logger.info(f"Exhibit {exhibit_id}: extracted {len(text_content)} chars")

            # Add document for classification
            documents.append({
                'exhibit_id': exhibit_id,
                'content': text_content,
                'filename': doc.filename,
                'extraction_result': extraction_result_obj
            })

            # Warn if minimal text extracted
            if len(text_content.strip()) < 50:
                logger.warning(f"Minimal text extracted for {exhibit_id} ({doc.filename}): "
                               f"only {len(text_content)} characters")

        # Log extraction summary if we have metrics
        if extraction_metrics['total_extraction_time'] > 0:
            logger.info(f"Document extraction complete: {extraction_metrics['total_extraction_time']:.2f}s total, "
                        f"{extraction_metrics['ocr_pages']} OCR pages, "
                        f"{extraction_metrics['low_confidence_docs']} low confidence docs")

        if extraction_metrics['extraction_warnings']:
            logger.warning(f"Extraction warnings: {extraction_metrics['extraction_warnings']}")

        # Classify documents with detailed extraction if requested
        classifications = await tagger.classify_batch(
            documents,
            extract_details=request.extract_detailed_data
        )

        # Add extraction confidence to classifications if available
        for i, classification in enumerate(classifications):
            if i < len(documents) and documents[i].get('extraction_result'):
                extraction_result = documents[i]['extraction_result']
                # Adjust classification confidence based on extraction confidence
                classification.confidence *= extraction_result.confidence

                # Add extraction method to extracted_data if present
                if classification.extracted_data is None:
                    classification.extracted_data = {}
                classification.extracted_data['_extraction_method'] = extraction_result.method
                classification.extracted_data['_extraction_confidence'] = extraction_result.confidence

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
        logger.error(f"Tag exhibits failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint with system capabilities"""
    service_type = "OpenAI" if settings.use_openai else "AWS Bedrock"
    model = settings.openai_model if settings.use_openai else settings.bedrock_model_id

    # Check OCR availability
    try:
        from app.services.document_processor import OCR_AVAILABLE
        ocr_available = OCR_AVAILABLE
    except:
        ocr_available = False

    return {
        "status": "healthy",
        "service": service_type,
        "model": model,
        "ocr_available": ocr_available,
        "ocr_enabled": getattr(settings, 'ocr_enabled', False),
        "features": {
            "pdf_extraction": True,
            "docx_extraction": True,
            "ocr_extraction": ocr_available and getattr(settings, 'ocr_enabled', False),
            "parallel_ocr": getattr(settings, 'parallel_ocr', False),
            "detailed_extraction": True
        }
    }


@router.get("/taxonomy")
async def get_taxonomy():
    """Get current taxonomy"""
    from app.utils.taxonomy import get_full_taxonomy
    return get_full_taxonomy()


@router.post("/test-extraction")
async def test_extraction(
        file_content: str,
        mime_type: str,
        doc_processor: DocumentProcessor = Depends(get_document_processor)
):
    """Test endpoint for document extraction (development only)"""
    if not settings.debug:
        raise HTTPException(status_code=403, detail="Test endpoints only available in debug mode")

    try:
        result = doc_processor.process_document(file_content, mime_type)

        # Handle both old and new processor versions
        if isinstance(result, tuple):
            text, extraction_result = result
            return {
                "extracted_text_preview": text[:500] + "..." if len(text) > 500 else text,
                "total_length": len(text),
                "extraction_result": {
                    "method": extraction_result.method,
                    "pages_processed": extraction_result.pages_processed,
                    "extraction_time": extraction_result.extraction_time,
                    "confidence": extraction_result.confidence,
                    "warnings": extraction_result.warnings
                }
            }
        else:
            # Old processor
            text = result
            return {
                "extracted_text_preview": text[:500] + "..." if len(text) > 500 else text,
                "total_length": len(text),
                "extraction_result": {
                    "method": "unknown",
                    "pages_processed": 0,
                    "extraction_time": 0,
                    "confidence": 1.0,
                    "warnings": []
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))