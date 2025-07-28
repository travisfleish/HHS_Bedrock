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
        prompt_lines.append(f"\n{category.upper()}:")
        for doc_type, details in types.items():
            prompt_lines.append(f"  - {doc_type}: {details['description']}")
            prompt_lines.append(f"    Indicators: {', '.join(details['indicators'])}")

    return "\n".join(prompt_lines)

def get_full_taxonomy() -> Dict:
    """Get full taxonomy structure"""
    return OMHA_TAXONOMY