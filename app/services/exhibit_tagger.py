import json
from typing import List, Dict, Optional
from app.models import DocumentType, ExhibitClassification
from app.services.bedrock_service import BedrockService
from app.utils.taxonomy import get_taxonomy_prompt
import logging
import re
import time

logger = logging.getLogger(__name__)


class ExhibitTagger:
    def __init__(self, bedrock_service: BedrockService):
        self.bedrock = bedrock_service
        self.system_prompt = self._build_system_prompt()
        # Reduced context window to leave room for response
        self.max_document_length = 6000  # Reduced from 8000
        logger.info("ExhibitTagger initialized with max_document_length: %d", self.max_document_length)

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
5. IMPORTANT: Return ONLY valid JSON, no additional text or markdown formatting

Examples:
- "DISCHARGE SUMMARY" header → discharge_summary
- "Medicare Reconsideration Decision" → reconsideration_decision
- "CT THORAX WITH CONTRAST" → diagnostic_report
"""

    def _build_extraction_prompt(self, document_type: str) -> str:
        """Build specific extraction prompt based on document type"""

        extraction_prompts = {
            "reconsideration_decision": """Extract the following information:
- decision_date: Date of the decision
- appeal_number: Medicare appeal number or case number
- decision_outcome: FAVORABLE or UNFAVORABLE
- appellant_name: Name of the appellant
- beneficiary_name: Name of the beneficiary
- beneficiary_hic: Health Insurance Claim number
- service_dates: Date(s) of service in question
- denial_reasons: List of reasons for denial (if unfavorable)
- amount_in_dispute: Dollar amount in dispute
- next_appeal_deadline: Deadline for next level of appeal
- qic_contractor: Name of the QIC contractor""",

            "alj_appeal_request": """Extract the following information:
- request_date: Date of the appeal request
- appellant_name: Name of the appellant
- beneficiary_name: Name of the beneficiary
- beneficiary_hic: Health Insurance Claim number
- reconsideration_number: Previous reconsideration number
- service_dates: Date(s) of service being appealed
- provider_name: Name of provider/supplier
- reason_for_appeal: Main reason for the appeal
- requested_remedy: What the appellant is requesting""",

            "physician_order": """Extract the following information:
- order_date: Date of the order/prescription
- physician_name: Name of ordering physician
- physician_npi: NPI number if available
- patient_name: Patient name
- diagnosis_codes: ICD codes listed
- ordered_items: Specific items/services ordered
- duration: Duration of need if specified
- medical_necessity: Statement of medical necessity""",

            "diagnostic_report": """Extract the following information:
- report_date: Date of the report
- report_type: Type of diagnostic test (CT, MRI, X-ray, etc.)
- patient_name: Patient name
- ordering_physician: Name of ordering physician
- findings: Key findings or impressions
- diagnosis_codes: Any ICD codes mentioned
- recommendations: Any recommendations made""",

            "clinical_notes": """Extract the following information:
- visit_date: Date of visit
- patient_name: Patient name
- provider_name: Provider name
- chief_complaint: Chief complaint or reason for visit
- diagnoses: List of diagnoses
- treatment_plan: Treatment plan or recommendations
- medications: Medications prescribed or reviewed
- follow_up: Follow-up instructions""",

            "discharge_summary": """Extract the following information:
- discharge_date: Date of discharge
- admission_date: Date of admission
- patient_name: Patient name
- primary_diagnosis: Primary diagnosis
- secondary_diagnoses: List of secondary diagnoses
- procedures_performed: Any procedures performed during stay
- discharge_instructions: Discharge instructions
- follow_up_appointments: Scheduled follow-up appointments
- medications_on_discharge: Medications prescribed at discharge""",

            "remittance_advice": """Extract the following information:
- remittance_date: Date of remittance advice
- claim_numbers: Claim numbers (ICN)
- service_dates: Service dates
- billed_amount: Amount billed
- allowed_amount: Amount allowed
- paid_amount: Amount paid
- denial_codes: Denial or adjustment codes
- patient_responsibility: Patient responsibility amount"""
        }

        return extraction_prompts.get(document_type, """Extract any relevant information including:
- dates
- names (patient, provider, facility)
- dollar amounts
- important codes or numbers
- key medical findings or decisions""")

    def _extract_json_from_response(self, response: str) -> dict:
        """Extract JSON from response, handling various formats"""
        logger.debug("Attempting to extract JSON from response of length: %d", len(response))

        try:
            # First try direct JSON parsing
            return json.loads(response)
        except json.JSONDecodeError:
            logger.debug("Direct JSON parsing failed, trying to extract from markdown")

            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                logger.debug("Found JSON in markdown code block")
                return json.loads(json_match.group(1))

            # Try to find JSON object in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                logger.debug("Found JSON object in response")
                return json.loads(json_match.group(0))

            # If all else fails, raise the original error
            logger.error("Could not extract JSON from response: %s", response[:200])
            raise json.JSONDecodeError("Could not extract JSON from response", response, 0)

    def _truncate_document(self, text: str, max_length: int) -> str:
        """Intelligently truncate document to fit within token limits"""
        original_length = len(text)

        if original_length <= max_length:
            logger.debug("Document length (%d) within limits", original_length)
            return text

        logger.info("Truncating document from %d to %d characters", original_length, max_length)

        # Try to find a good breaking point
        truncated = text[:max_length]

        # Look for the last complete sentence
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')

        # Use the latest breaking point
        break_point = max(last_period, last_newline)
        if break_point > max_length * 0.8:  # Only use if it's not too far back
            truncated = truncated[:break_point + 1]
            logger.debug("Truncated at natural break point: %d", break_point)

        return truncated + "... [truncated]"

    async def extract_detailed_data(self, document_text: str, document_type: str) -> Optional[Dict]:
        """Extract detailed information based on document type"""
        logger.info("Starting detailed extraction for document type: %s", document_type)
        start_time = time.time()

        if document_type == "unclassified":
            logger.info("Skipping extraction for unclassified document")
            return None

        extraction_prompt = self._build_extraction_prompt(document_type)

        # Truncate document for extraction to ensure we don't exceed token limits
        truncated_text = self._truncate_document(document_text, self.max_document_length)

        prompt = f"""Extract specific information from this {document_type.replace('_', ' ')} document:

{extraction_prompt}

DOCUMENT:
{truncated_text}

Return ONLY a JSON object with the extracted information. Use null for any information not found.
"""

        try:
            extraction_system_prompt = """You are a medical document data extractor. 
Extract only factual information directly stated in the document. 
Return clean JSON with the requested fields. 
Use null for missing information.
Do not infer or assume information not explicitly stated."""

            logger.debug("Sending extraction request to model")
            response = await self.bedrock.invoke_model(prompt, extraction_system_prompt)
            logger.debug("Received extraction response")

            extracted_data = self._extract_json_from_response(response)
            logger.info("Successfully extracted %d fields", len(extracted_data))

            # Clean up the extracted data (remove None values for cleaner output)
            cleaned_data = {k: v for k, v in extracted_data.items() if v is not None}

            extraction_time = time.time() - start_time
            logger.info("Extraction completed in %.2f seconds", extraction_time)

            return cleaned_data

        except Exception as e:
            logger.error("Data extraction failed after %.2f seconds: %s",
                         time.time() - start_time, str(e), exc_info=True)
            # Return partial data instead of None
            return {"extraction_error": f"Failed to extract detailed data: {str(e)[:100]}"}

    async def classify_document(self, document_text: str, exhibit_id: str,
                                extract_details: bool = False) -> ExhibitClassification:
        """Classify a single document with optional detailed extraction"""
        logger.info("Starting classification for exhibit: %s (extract_details=%s)",
                    exhibit_id, extract_details)
        start_time = time.time()

        # Log document characteristics
        logger.debug("Document length: %d characters", len(document_text))

        # Truncate document for classification
        truncated_text = self._truncate_document(document_text, self.max_document_length)

        prompt = f"""Classify this medical document:

DOCUMENT:
{truncated_text}

OUTPUT FORMAT (return ONLY this JSON, no other text):
{{
    "document_type": "exact_term_from_taxonomy",
    "confidence": 0.95,
    "key_indicators": ["indicator1", "indicator2", "indicator3"],
    "requires_review": false
}}
"""

        try:
            logger.debug("Sending classification request to model")
            response = await self.bedrock.invoke_model(prompt, self.system_prompt)
            classification_time = time.time() - start_time
            logger.debug("Received classification response in %.2f seconds", classification_time)

            # Parse JSON response with improved error handling
            classification_data = self._extract_json_from_response(response)
            logger.info("Classified as: %s with confidence: %.2f",
                        classification_data.get('document_type', 'unknown'),
                        classification_data.get('confidence', 0))

            # Validate document type
            try:
                doc_type = DocumentType(classification_data['document_type'])
            except (ValueError, KeyError):
                logger.warning("Invalid document type: %s",
                               classification_data.get('document_type', 'missing'))
                doc_type = DocumentType.UNCLASSIFIED
                classification_data['requires_review'] = True

            # Extract detailed data if requested
            extracted_data = None
            if extract_details and doc_type != DocumentType.UNCLASSIFIED:
                try:
                    logger.info("Starting detailed extraction for %s", exhibit_id)
                    extracted_data = await self.extract_detailed_data(
                        document_text,
                        classification_data['document_type']
                    )
                except Exception as e:
                    logger.error("Extraction failed for %s: %s", exhibit_id, str(e))
                    extracted_data = {"extraction_error": "Failed to extract detailed data"}

            total_time = time.time() - start_time
            logger.info("Completed processing for %s in %.2f seconds", exhibit_id, total_time)

            return ExhibitClassification(
                exhibit_id=exhibit_id,
                document_type=doc_type,
                confidence=classification_data.get('confidence', 0.0),
                key_indicators=classification_data.get('key_indicators', []),
                requires_review=classification_data.get('requires_review', True),
                extracted_data=extracted_data
            )

        except json.JSONDecodeError as e:
            logger.error("JSON parsing failed for exhibit %s: %s", exhibit_id, str(e))
            logger.error("Response was: %s", response[:500] if 'response' in locals() else 'No response')
            return ExhibitClassification(
                exhibit_id=exhibit_id,
                document_type=DocumentType.UNCLASSIFIED,
                confidence=0.0,
                key_indicators=["JSON parsing error"],
                requires_review=True
            )
        except Exception as e:
            logger.error("Classification failed for exhibit %s: %s", exhibit_id, str(e), exc_info=True)
            return ExhibitClassification(
                exhibit_id=exhibit_id,
                document_type=DocumentType.UNCLASSIFIED,
                confidence=0.0,
                key_indicators=["Classification error"],
                requires_review=True
            )

    async def classify_batch(self, documents: List[Dict], extract_details: bool = False) -> List[ExhibitClassification]:
        """Classify multiple documents"""
        logger.info("Starting batch classification for %d documents (extract_details=%s)",
                    len(documents), extract_details)
        batch_start_time = time.time()
        classifications = []

        for idx, doc in enumerate(documents):
            logger.info("Processing document %d/%d: %s", idx + 1, len(documents), doc['exhibit_id'])

            try:
                classification = await self.classify_document(
                    doc['content'],
                    doc['exhibit_id'],
                    extract_details
                )
                classifications.append(classification)
            except Exception as e:
                logger.error("Batch classification failed for %s: %s",
                             doc['exhibit_id'], str(e), exc_info=True)
                # Add a failed classification instead of skipping
                classifications.append(ExhibitClassification(
                    exhibit_id=doc['exhibit_id'],
                    document_type=DocumentType.UNCLASSIFIED,
                    confidence=0.0,
                    key_indicators=["Processing error"],
                    requires_review=True,
                    extracted_data={"error": str(e)[:100]} if extract_details else None
                ))

        batch_total_time = time.time() - batch_start_time
        logger.info("Completed batch classification in %.2f seconds (%.2f sec/doc average)",
                    batch_total_time, batch_total_time / len(documents) if documents else 0)

        return classifications