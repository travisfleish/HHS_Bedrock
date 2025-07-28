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