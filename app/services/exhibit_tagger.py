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