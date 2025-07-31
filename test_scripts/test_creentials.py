#!/usr/bin/env python3
"""
Test Claude 3.5 Sonnet Access
"""

import boto3
import json

print("🎉 Testing your new Claude 3.5 Sonnet access...\n")

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

# Test Claude 3.5 Sonnet
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

try:
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": "Say 'Hello from Claude 3.5 Sonnet on Bedrock!' and tell me what model you are."
                }
            ],
            "max_tokens": 100,
            "temperature": 0.1
        }),
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response['body'].read())
    content = response_body['content'][0]['text']

    print("✅ SUCCESS! Claude 3.5 Sonnet is working!")
    print(f"\nResponse: {content}")

    print("\n" + "=" * 60)
    print("🎯 Your OMHA Tagger is ready to use!")
    print("\n📝 Update your .env file with:")
    print("   USE_OPENAI=false")
    print("   BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0")
    print("   AWS_REGION=us-east-1")

except Exception as e:
    print(f"❌ Error: {str(e)}")
    print("\nThe access might still be propagating. Wait 1-2 minutes and try again.")