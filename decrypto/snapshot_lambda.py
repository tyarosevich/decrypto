################## Example lambda to save a snapshot to S3. #################################


import json
import boto3
import uuid
import datetime
import os
def lambda_handler(event, context):
    snapshotarn = event['Records'][0]['Sns']['Message']
    snapshotarn=snapshotarn.split()
    snapshotarn=snapshotarn[10].replace('.','')
    print(snapshotarn)
    rds=boto3.client('rds')
    export=rds.start_export_task(
        ExportTaskIdentifier='export'+'-'+uuid.uuid4().hex,
        SourceArn=snapshotarn,
        S3BucketName=os.environ.get('S3_BUCKET_NAME'),
        IamRoleArn=os.environ.get('IAM_ROLE_FOR_EXPORT_TASK'),
        KmsKeyId=os.environ.get('KMS_KEY_ID'),
        S3Prefix=os.environ.get('S3_PREFIX'),
    )
    status={
        'ExportTaskIdentifier':export['ExportTaskIdentifier'],
        'S3Bucket':export['S3Bucket'],
        'S3Prefix':export['S3Prefix'],
        'Status':export['Status'],
        'ResponseMetadata':export['ResponseMetadata'],
    }
    print(status)
    # TODO implement
    return {
        'statusCode': 200,
        'body': status
    }