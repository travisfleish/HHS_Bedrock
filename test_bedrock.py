import boto3
import json

# Replace with your actual model ID
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
REGION = "us-east-2"  # or the region you're using

# Initialize the Bedrock runtime client
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# Claude requires the prompt to be inside the 'messages' key if using Claude Messages API
payload = {
    "messages": [
        {
            "role": "user",
            "content": "What's the capital of France?"
        }
    ],
    "max_tokens": 200,
    "temperature": 0.5,
    "top_p": 1.0
}

try:
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    print("\n✅ Response:")
    print(json.dumps(result, indent=2))

except bedrock.exceptions.AccessDeniedException as e:
    print("\n❌ AccessDeniedException:")
    print(e)

except bedrock.exceptions.ValidationException as e:
    print("\n❌ ValidationException:")
    print(e)

except Exception as e:
    print("\n❌ Unexpected error:")
    print(e)
