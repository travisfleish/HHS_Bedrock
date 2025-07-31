import boto3

bedrock = boto3.client('bedrock', region_name='us-east-1')

response = bedrock.create_provisioned_model_throughput(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',  # or your chosen model
    provisionedModelName='my-provisioned-model',
    modelUnits=1,  # Number of model units
    commitmentDuration='OneMonth',  # or 'SixMonths' or 'None'
    tags=[
        {
            'key': 'Environment',
            'value': 'Production'
        }
    ]
)