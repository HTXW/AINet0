import boto3

forecast = boto3.client('forecast')

def list_predictors():
    predictors = []
    response = forecast.list_predictors()
    predictors.extend(response['Predictors'])

    while 'NextToken' in response:
        response = forecast.list_predictors(NextToken=response['NextToken'])
        predictors.extend(response['Predictors'])

    return predictors

predictors = list_predictors()
for predictor in predictors:
    print(f"Predictor ARN: {predictor['PredictorArn']}, Name: {predictor['PredictorName']}")

"""# **Delete Predictors**"""

import boto3

forecast = boto3.client('forecast', region_name='us-west-2')

predictor_arns = [
    "arn:aws:forecast:us-west-2:158285887431:predictor/hourcpu_usage_forecast_deepar",
    "arn:aws:forecast:us-west-2:158285887431:predictor/hourcpu_usage_forecast_plus",
    "arn:aws:forecast:us-west-2:158285887431:predictor/hourcpu_usage_forecast_ARplus",
    "arn:aws:forecast:us-west-2:158285887431:predictor/hourcpu_usage_forecast_cnn",
    "arn:aws:forecast:us-west-2:158285887431:predictor/hourcpu_usage_forecast_deepARplus"
]

for predictor_arn in predictor_arns:
    response = forecast.delete_predictor(PredictorArn=predictor_arn)
    print(f"Deleted Predictor: {predictor_arn}")

"""# **List Dataset and Delete required Dataset**"""

def list_dataset_groups():
    dataset_groups = []
    response = forecast.list_dataset_groups()
    dataset_groups.extend(response['DatasetGroups'])

    while 'NextToken' in response:
        response = forecast.list_dataset_groups(NextToken=response['NextToken'])
        dataset_groups.extend(response['DatasetGroups'])

    return dataset_groups

dataset_groups = list_dataset_groups()
for dataset_group in dataset_groups:
    print(f"Dataset Group ARN: {dataset_group['DatasetGroupArn']}, Name: {dataset_group['DatasetGroupName']}")

def list_datasets():
    datasets = []
    response = forecast.list_datasets()
    datasets.extend(response['Datasets'])

    while 'NextToken' in response:
        response = forecast.list_datasets(NextToken=response['NextToken'])
        datasets.extend(response['Datasets'])

    return datasets

datasets = list_datasets()
for dataset in datasets:
    print(f"Dataset ARN: {dataset['DatasetArn']}, Name: {dataset['DatasetName']}")

import boto3

forecast = boto3.client('forecast')

def list_dataset_import_jobs():
    dataset_import_jobs = []
    response = forecast.list_dataset_import_jobs()
    dataset_import_jobs.extend(response['DatasetImportJobs'])

    while 'NextToken' in response:
        response = forecast.list_dataset_import_jobs(NextToken=response['NextToken'])
        dataset_import_jobs.extend(response['DatasetImportJobs'])

    return dataset_import_jobs

dataset_import_jobs = list_dataset_import_jobs()
for job in dataset_import_jobs:
    print(f"Dataset Import Job ARN: {job['DatasetImportJobArn']}, Name: {job['DatasetImportJobName']}")

dataset_group_arns = [
    "arn:aws:forecast:us-west-2:158285887431:dataset-group/hourcpu_usage_forecast_gp"
]

for dataset_group_arn in dataset_group_arns:
    try:
        forecast.delete_dataset_group(DatasetGroupArn=dataset_group_arn)
        print(f"Deleted Dataset Group: {dataset_group_arn}")
    except forecast.exceptions.ResourceInUseException as e:
        print(f"Could not delete Dataset Group: {dataset_group_arn} - {e}")