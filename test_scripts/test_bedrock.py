#!/usr/bin/env python3
"""
Bedrock IAM and Model Access Diagnostic
Tests IAM permissions and finds which Claude models you can actually use
"""

import boto3
import json
from datetime import datetime
import time
from botocore.exceptions import ClientError

REGION = "us-east-1"


def check_iam_permissions():
    """Check IAM permissions for Bedrock"""
    print("\n" + "=" * 60)
    print("IAM PERMISSION CHECK")
    print("=" * 60)

    iam = boto3.client('iam')
    sts = boto3.client('sts')

    try:
        # Get current identity
        identity = sts.get_caller_identity()
        print(f"\n✅ Current Identity:")
        print(f"   Account: {identity['Account']}")
        print(f"   User/Role ARN: {identity['Arn']}")
        print(f"   User ID: {identity['UserId']}")

        # Check if using role or user
        if ":assumed-role/" in identity['Arn']:
            print(f"   Type: Assumed Role")
            role_name = identity['Arn'].split('/')[-2]
            print(f"   Role Name: {role_name}")
        elif ":user/" in identity['Arn']:
            print(f"   Type: IAM User")
            user_name = identity['Arn'].split('/')[-1]
            print(f"   User Name: {user_name}")

    except Exception as e:
        print(f"❌ Error getting identity: {str(e)}")
        return False

    # Check Bedrock permissions
    print("\n📋 Checking Bedrock permissions...")

    # Test basic Bedrock access
    bedrock = boto3.client('bedrock', region_name=REGION)
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION)

    permissions = {
        'bedrock:ListFoundationModels': False,
        'bedrock:GetFoundationModel': False,
        'bedrock:InvokeModel': False,
        'bedrock:InvokeModelWithResponseStream': False,
    }

    # Test ListFoundationModels
    try:
        bedrock.list_foundation_models()
        permissions['bedrock:ListFoundationModels'] = True
        print("   ✅ bedrock:ListFoundationModels - Allowed")
    except ClientError as e:
        if e.response['Error']['Code'] == 'AccessDeniedException':
            print("   ❌ bedrock:ListFoundationModels - Denied")
        else:
            print(f"   ⚠️  bedrock:ListFoundationModels - Error: {e.response['Error']['Code']}")

    # Test GetFoundationModel (try with a known model)
    try:
        bedrock.get_foundation_model(modelIdentifier="anthropic.claude-instant-v1")
        permissions['bedrock:GetFoundationModel'] = True
        print("   ✅ bedrock:GetFoundationModel - Allowed")
    except ClientError as e:
        if e.response['Error']['Code'] == 'AccessDeniedException':
            print("   ❌ bedrock:GetFoundationModel - Denied")
        else:
            print(f"   ⚠️  bedrock:GetFoundationModel - Error: {e.response['Error']['Code']}")

    # Test InvokeModel (will test with actual models later)
    print("   ℹ️  bedrock:InvokeModel - Will test with actual models")
    print("   ℹ️  bedrock:InvokeModelWithResponseStream - Will test with actual models")

    return permissions


def check_model_specific_permissions(bedrock_runtime):
    """Check permissions for specific model patterns"""
    print("\n📋 Checking model-specific permissions...")

    test_cases = [
        {
            "name": "Claude Instant v1",
            "model_id": "anthropic.claude-instant-v1",
            "payload": {
                "prompt": "\n\nHuman: Hi\n\nAssistant:",
                "max_tokens_to_sample": 10
            }
        },
        {
            "name": "Claude v2",
            "model_id": "anthropic.claude-v2",
            "payload": {
                "prompt": "\n\nHuman: Hi\n\nAssistant:",
                "max_tokens_to_sample": 10
            }
        },
        {
            "name": "Claude 3 Haiku",
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "payload": {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            }
        },
        {
            "name": "Claude 3 Sonnet",
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "payload": {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            }
        },
        {
            "name": "Claude 3.5 Sonnet v1",
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "payload": {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            }
        }
    ]

    results = []

    for test in test_cases:
        print(f"\n   Testing {test['name']} ({test['model_id']})...")

        try:
            response = bedrock_runtime.invoke_model(
                modelId=test['model_id'],
                body=json.dumps(test['payload']),
                contentType="application/json",
                accept="application/json"
            )

            # If we get here, it worked
            response_body = json.loads(response['body'].read())
            print(f"   ✅ SUCCESS - Model is accessible")
            results.append({
                "model": test['model_id'],
                "status": "SUCCESS",
                "name": test['name']
            })

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']

            if error_code == 'AccessDeniedException':
                print(f"   ❌ ACCESS DENIED - No permission for this model")
                # Check if it's a specific resource ARN issue
                if "arn:aws:bedrock" in error_msg:
                    print(f"      Details: {error_msg}")
            elif error_code == 'ResourceNotFoundException':
                print(f"   ⚠️  NOT FOUND - Model not available in this region")
            elif error_code == 'ValidationException':
                print(f"   ⚠️  VALIDATION ERROR - {error_msg}")
            else:
                print(f"   ❌ ERROR - {error_code}: {error_msg}")

            results.append({
                "model": test['model_id'],
                "status": error_code,
                "name": test['name'],
                "error": error_msg
            })

        except Exception as e:
            print(f"   ❌ UNEXPECTED ERROR - {type(e).__name__}: {str(e)}")
            results.append({
                "model": test['model_id'],
                "status": "ERROR",
                "name": test['name'],
                "error": str(e)
            })

    return results


def check_resource_based_permissions():
    """Check if there are resource-based permission restrictions"""
    print("\n📋 Checking for resource-based restrictions...")

    # This checks if your permissions are limited to specific model ARNs
    bedrock = boto3.client('bedrock', region_name=REGION)

    try:
        # Get all available models
        response = bedrock.list_foundation_models()
        all_models = response.get('modelSummaries', [])

        claude_models = [m for m in all_models if 'claude' in m['modelId'].lower()]

        print(f"\n   Total Claude models in region: {len(claude_models)}")

        # Group by access pattern
        instant_models = [m for m in claude_models if 'instant' in m['modelId']]
        v2_models = [m for m in claude_models if 'claude-v2' in m['modelId']]
        v3_models = [m for m in claude_models if 'claude-3' in m['modelId'] and 'claude-3-5' not in m['modelId']]
        v35_models = [m for m in claude_models if 'claude-3-5' in m['modelId']]

        print(f"   - Claude Instant models: {len(instant_models)}")
        print(f"   - Claude v2 models: {len(v2_models)}")
        print(f"   - Claude 3 models: {len(v3_models)}")
        print(f"   - Claude 3.5 models: {len(v35_models)}")

    except Exception as e:
        print(f"   ❌ Error checking models: {str(e)}")


def generate_iam_policy_recommendation(results):
    """Generate recommended IAM policy based on test results"""
    print("\n" + "=" * 60)
    print("IAM POLICY RECOMMENDATIONS")
    print("=" * 60)

    working_models = [r for r in results if r['status'] == 'SUCCESS']
    denied_models = [r for r in results if r['status'] == 'AccessDeniedException']

    if not working_models and denied_models:
        print("\n❌ No models are currently accessible due to IAM restrictions")
        print("\n📝 Recommended IAM Policy to add:")
        print("""
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:us-east-1:*:foundation-model/anthropic.claude-instant-v1*",
                "arn:aws:bedrock:us-east-1:*:foundation-model/anthropic.claude-v2*",
                "arn:aws:bedrock:us-east-1:*:foundation-model/anthropic.claude-3-haiku*",
                "arn:aws:bedrock:us-east-1:*:foundation-model/anthropic.claude-3-sonnet*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:ListFoundationModels",
                "bedrock:GetFoundationModel"
            ],
            "Resource": "*"
        }
    ]
}
""")
        print("\n💡 How to apply:")
        print("   1. Go to IAM Console")
        print("   2. Find your user/role")
        print("   3. Add this policy as an inline policy or attach a similar managed policy")
        print("   4. For Claude 3.5 models, you may need additional permissions")

    elif working_models:
        print(f"\n✅ You have access to {len(working_models)} model(s)")
        print("\n📝 Your current permissions allow:")
        for model in working_models:
            print(f"   - {model['name']}: {model['model']}")


def main():
    """Run all diagnostics"""
    print("=" * 60)
    print("BEDROCK IAM AND ACCESS DIAGNOSTIC")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Region: {REGION}")
    print(f"Boto3 Version: {boto3.__version__}")

    # Check IAM permissions
    iam_permissions = check_iam_permissions()

    # Initialize Bedrock runtime
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION)
        print("\n✅ Bedrock runtime client initialized")
    except Exception as e:
        print(f"\n❌ Failed to initialize Bedrock runtime: {str(e)}")
        return

    # Check model-specific permissions
    results = check_model_specific_permissions(bedrock_runtime)

    # Check resource-based restrictions
    check_resource_based_permissions()

    # Generate recommendations
    generate_iam_policy_recommendation(results)

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    working_models = [r for r in results if r['status'] == 'SUCCESS']

    if working_models:
        print(f"\n✅ WORKING MODELS ({len(working_models)}):")
        for model in working_models:
            print(f"\n   Model: {model['model']}")
            print(f"   Name: {model['name']}")
            print(f"   ✅ Ready to use in your application")

        print("\n📝 Update your app/config.py:")
        print(f'   bedrock_model_id = "{working_models[0]["model"]}"')
        print(f'   use_openai = False')
    else:
        print("\n❌ No working models found")
        print("\n🔧 Next steps:")
        print("   1. Check the IAM policy recommendations above")
        print("   2. Contact your AWS administrator to update permissions")
        print("   3. Consider using OpenAI as a fallback option")


if __name__ == "__main__":
    main()