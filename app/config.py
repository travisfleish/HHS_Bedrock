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

    class Config:
        env_file = ".env"


settings = Settings()