from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # AWS Settings
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # Bedrock Settings
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"

    # OpenAI Settings
    use_openai: bool = True  # Set to True to use OpenAI instead of Bedrock
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"  # or "gpt-3.5-turbo" for cheaper option

    # Common Model Settings
    max_tokens: int = 4096
    temperature: float = 0.1

    # App Settings
    app_name: str = "OMHA Exhibit Tagger"
    debug: bool = False

    # Taxonomy Version
    taxonomy_version: str = "2.1"

    # OCR Settings
    ocr_enabled: bool = True
    ocr_dpi: int = 300  # Higher DPI for better quality
    min_text_threshold: int = 100  # Minimum chars to consider page as text-based
    parallel_ocr: bool = True  # Process OCR pages in parallel
    ocr_max_workers: int = 4  # Max parallel OCR workers

    # Document Processing Settings
    max_document_size_mb: int = 50  # Maximum document size in MB
    extraction_timeout: int = 300  # Timeout for document extraction in seconds

    # Performance Settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # Cache extraction results for 1 hour

    # Logging Settings
    log_level: str = "INFO"
    log_extraction_metrics: bool = True

    class Config:
        env_file = ".env"


settings = Settings()