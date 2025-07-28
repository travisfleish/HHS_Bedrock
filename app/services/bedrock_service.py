import boto3
import json
from typing import Dict, Any
import logging
from app.config import settings
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


class BedrockService:
    def __init__(self):
        # Check if we're using OpenAI or Bedrock
        self.use_openai = settings.use_openai

        if self.use_openai:
            # Initialize OpenAI client
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
            self.model_id = settings.openai_model
        else:
            # Initialize Bedrock client
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
            self.model_id = settings.bedrock_model_id

    async def invoke_model(self, prompt: str, system_prompt: str) -> str:
        """Invoke either OpenAI or Bedrock model with prompt"""
        try:
            if self.use_openai:
                # Use OpenAI
                response = self.openai_client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens
                )
                return response.choices[0].message.content

            else:
                # Use Bedrock (existing code)
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
            logger.error(f"Model invocation failed: {str(e)}")
            raise