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

        return '\n'.join(text_parts)

    @staticmethod
    def extract_text_from_docx(content: bytes) -> str:
        """Extract text from DOCX content"""
        doc_file = io.BytesIO(content)
        doc = Document(doc_file)

        paragraphs = [p.text for p in doc.paragraphs]
        return '\n'.join(paragraphs)

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