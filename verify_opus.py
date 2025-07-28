# Quick test script to verify model access
import boto3

client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

# List available models (if you have permissions)
bedrock = boto3.client('bedrock', region_name='us-east-1')
try:
    response = bedrock.list_foundation_models()
    for model in response['modelSummaries']:
        if 'claude' in model['modelId'].lower():
            print(f"Available: {model['modelId']}")
except Exception as e:
    print(f"Cannot list models: {e}")