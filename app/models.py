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