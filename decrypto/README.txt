The files in this folder were used to test AWS Lambda functions, their interactions with the database, and authorization
functionality OUTSIDE the VPC. These were then trivially changed to operate within the VPC, querying resources OUTSIDE
the VPC and then writing to the db. The complete Lambda app is found in ./lambda_deployment_package, which is ignored for
size, with the source code alone in ./lambda_src.