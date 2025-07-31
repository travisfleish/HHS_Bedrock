import boto3
import json
from typing import Dict, Any
import logging
from app.config import settings
import openai
from openai import OpenAI
import asyncio
import time
from botocore.exceptions import ClientError

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
        if self.use_openai:
            return await self._invoke_openai(prompt, system_prompt)
        else:
            return await self._invoke_bedrock_with_retry(prompt, system_prompt)

    async def _invoke_openai(self, prompt: str, system_prompt: str) -> str:
        """Use OpenAI"""
        try:
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
        except Exception as e:
            logger.error(f"OpenAI invocation failed: {str(e)}")
            raise

    async def _invoke_bedrock_with_retry(self, prompt: str, system_prompt: str, max_retries: int = 5) -> str:
        """Invoke Bedrock with exponential backoff retry logic"""
        for attempt in range(max_retries):
            try:
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

                # Add small delay between requests to avoid throttling
                if attempt > 0:
                    await asyncio.sleep(0.5)

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

            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ThrottlingException':
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2^attempt seconds (1, 2, 4, 8, 16...)
                        wait_time = min(2 ** attempt, 30)  # Cap at 30 seconds
                        logger.warning(f"Throttled on attempt {attempt + 1}/{max_retries}. Waiting {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded for Bedrock invocation")
                        raise
                else:
                    logger.error(f"Bedrock invocation failed: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in Bedrock invocation: {str(e)}")
                raise

        raise Exception("Failed to invoke Bedrock model after all retries")