#!/usr/bin/env python3
"""
Quick Bedrock Permission Test
"""

import boto3
import json
import time

# Wait a bit for permissions to propagate
print("Waiting 10 seconds for permissions to propagate...")
time.sleep(10)

REGION = "us-east-1"
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)

# Test with Claude Instant (usually the most accessible)
print("\n🧪 Testing Claude Instant v1...")
try:
    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-instant-v1",
        body=json.dumps({
            "prompt": "\n\nHuman: Say 'Permissions working!'\n\nAssistant:",
            "max_tokens_to_sample": 20
        }),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response['body'].read())
    print(f"✅ SUCCESS! Response: {result.get('completion', result)}")

except Exception as e:
    print(f"❌ Still failing: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Make sure you saved and created the policy")
    print("2. Check if the policy is attached to your user")
    print("3. Try logging out and back into AWS CLI: aws configure")