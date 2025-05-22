import sys
import os
import time
import boto3


sys.path.insert( 0, os.path.abspath("../../common") )
session = boto3.Session(region_name="us-west-2")
forecast = session.client(service_name='forecast')
forecastquery = session.client(service_name='forecastquery')
forecast.list_dataset_groups()

"""# **AWS Setup and Data Import**"""

import pandas as pd
df = pd.read_csv("s3://cpu-data1/preprocessed_data.csv", dtype = object)
df.head(3)

role_arn = 'arn:aws:iam::588738590847:role/role1'
DATASET_FREQUENCY = "1min"
TIMESTAMP_FORMAT = "yyyy-MM-dd hh:mm:ss"
project = "one_min_cpu_usage"
datasetName= project+'_ds'
datasetGroupName= project +'_gp'
bucketName = "cpu-data1"
s3DataPath = f"s3://{bucketName}/preprocessed_data.csv"

schema ={
   "Attributes":[
      {
         "AttributeName":"timestamp",
         "AttributeType":"timestamp"
      },
      {
         "AttributeName":"item_id",
         "AttributeType":"string"
      },
      {
         "AttributeName":"target_value",
         "AttributeType":"float"
      }

   ]
}

response=forecast.create_dataset(
                    Domain="CUSTOM",
                    DatasetType='TARGET_TIME_SERIES',
                    DatasetName=datasetName,
                    DataFrequency=DATASET_FREQUENCY,
                    Schema = schema
                   )
datasetArn = response['DatasetArn']

forecast.describe_dataset(DatasetArn=datasetArn)

# Create dataset group
create_dataset_group_response = forecast.create_dataset_group(DatasetGroupName=datasetGroupName,
                                                              Domain="CUSTOM",
                                                              DatasetArns= [datasetArn]
                                                             )
datasetGroupArn = create_dataset_group_response['DatasetGroupArn']

forecast.describe_dataset_group(DatasetGroupArn=datasetGroupArn)
datasetImportJobName = 'CP_DSIMPORT_JOB_TARGET'
ds_import_job_response=forecast.create_dataset_import_job(DatasetImportJobName=datasetImportJobName,
                                                          DatasetArn=datasetArn,
                                                          DataSource= {
                                                              "S3Config" : {
                                                                 "Path":s3DataPath,
                                                                 "RoleArn": role_arn
                                                              }
                                                          },
                                                          TimestampFormat=TIMESTAMP_FORMAT
                                                         )

ds_import_job_arn=ds_import_job_response['DatasetImportJobArn']
print(ds_import_job_arn)

"""# **Create Predictors**
Create predictors using the DeepAR+ and CNN-QR algorithms.
"""
predictorName= project+'_deepArplus_algo_new'
forecastHorizon = 30
algorithmArn = 'arn:aws:forecast:::algorithm/Deep_AR_Plus'

create_predictor_response=forecast.create_predictor(PredictorName=predictorName,
                                                  AlgorithmArn=algorithmArn,
                                                  ForecastHorizon=forecastHorizon,
                                                  PerformAutoML= False,
                                                  PerformHPO=False,
                                                  EvaluationParameters= {"NumberOfBacktestWindows": 1,
                                                                         "BackTestWindowOffset": 30},
                                                  InputDataConfig= {"DatasetGroupArn": datasetGroupArn},
                                                  FeaturizationConfig= {"ForecastFrequency": "1min",
                                                                        "Featurizations":
                                                                        [
                                                                          {"AttributeName": "target_value",
                                                                           "FeaturizationPipeline":
                                                                            [
                                                                              {"FeaturizationMethodName": "filling",
                                                                               "FeaturizationMethodParameters":
                                                                                {"frontfill": "none",
                                                                                 "middlefill": "zero",
                                                                                 "backfill": "zero"}
                                                                              }
                                                                            ]
                                                                          }
                                                                        ]
                                                                       }
                                                 )

predictorArn=create_predictor_response['PredictorArn']

"""## **CNN QR Predictor**"""

import boto3

forecast = boto3.client(service_name='forecast')
project = 'one_min_cpu_usage'
predictor_name_prop = project + '_cnn_algo'
forecast_horizon = 30
algorithm_arn = 'arn:aws:forecast:::algorithm/CNN-QR'
dataset_group_arn = 'arn:aws:forecast:us-west-2:158285887431:dataset-group/one_min_cpu_usage_gp'

create_predictor_response = forecast.create_predictor(
    PredictorName=predictor_name_prop,
    AlgorithmArn=algorithm_arn,
    ForecastHorizon=forecast_horizon,
    PerformAutoML=False,
    PerformHPO=False,
    EvaluationParameters={
        "NumberOfBacktestWindows": 1,
        "BackTestWindowOffset": 30
    },
    InputDataConfig={
        "DatasetGroupArn": dataset_group_arn
    },
    FeaturizationConfig={
        "ForecastFrequency": "1min",
        "Featurizations": [
            {
                "AttributeName": "target_value",
                "FeaturizationPipeline": [
                    {
                        "FeaturizationMethodName": "filling",
                        "FeaturizationMethodParameters": {
                            "frontfill": "none",
                            "middlefill": "zero",
                            "backfill": "zero"
                        }
                    }
                ]
            }
        ]
    }
)

predictor_arn_prop = create_predictor_response['PredictorArn']
print("Predictor ARN:", predictor_arn_prop)

"""## **List of Predictors**"""

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

"""# **Accuracy Metrics**"""

forecast.get_accuracy_metrics(PredictorArn=predictorArn)

predictorcnnArn = "arn:aws:forecast:us-west-2:158285887431:predictor/one_min_cpu_usage_cnn_algo"
PredictordeepARN = "arn:aws:forecast:us-west-2:158285887431:predictor/one_min_cpu_usage_deepArplus_algo_new"

forecast.get_accuracy_metrics(PredictorArn=PredictordeepARN)

forecast.get_accuracy_metrics(PredictorArn=predictorcnnArn)

"""# **Visualise the plot**"""

import matplotlib.pyplot as plt

metrics = {
    'Predictor': ['CNN-QR', 'DeepAR+'],
    'MAPE': [0.0648, 0.1300],
    'RMSE': [49.8639, 54.4374],
    'WAPE': [0.0744, 0.1015]
}

plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.bar(metrics['Predictor'], metrics['MAPE'], color=['blue', 'green'])
plt.title('MAPE Comparison')
plt.ylabel('MAPE')

plt.subplot(1, 3, 2)
plt.bar(metrics['Predictor'], metrics['RMSE'], color=['blue', 'green'])
plt.title('RMSE Comparison')
plt.ylabel('RMSE')

plt.subplot(1, 3, 3)
plt.bar(metrics['Predictor'], metrics['WAPE'], color=['blue', 'green'])
plt.title('WAPE Comparison')
plt.ylabel('WAPE')

plt.tight_layout()
plt.show()
plt.savefig('rmse_comparison_plot.png', dpi=300)

#create forecast
project = 'one_min_cpu_usage'
forecastName= project+'_cnn_algo_forecast'
predictorArn = 'arn:aws:forecast:us-west-2:158285887431:predictor/one_min_cpu_usage_cnn_algo'
create_forecast_response=forecast.create_forecast(ForecastName=forecastName,
                                                  PredictorArn=predictorArn)
forecastArn = create_forecast_response['ForecastArn']

#get forecast
print(forecastArn)
forecastResponse = forecastquery.query_forecast(
    ForecastArn=forecastArn,
    Filters={"item_id":"15"}
)
print(forecastResponse)

#get forecast
print(forecastArn)
forecastResponse = forecastquery.query_forecast(
    ForecastArn=forecastArn,
    Filters={"item_id":"1"}
)
print(forecastResponse)

#get forecast
print(forecastArn)
forecastResponse = forecastquery.query_forecast(
    ForecastArn=forecastArn,
    Filters={"item_id":"20"}
)
print(forecastResponse)

"""## **DEEP AR PLUS RESULTS**"""

#create forecast
project = 'one_min_cpu_usage'
forecastName= project+'_deepArplus_algo_forecast'
predictorArn = 'arn:aws:forecast:us-west-2:158285887431:predictor/one_min_cpu_usage_deepArplus_algo_new'
create_forecast_response=forecast.create_forecast(ForecastName=forecastName,
                                                  PredictorArn=predictorArn)
forecastArn = create_forecast_response['ForecastArn']

#get forecast
print(forecastArn)

forecastResponse = forecastquery.query_forecast(
    ForecastArn=forecastArn,
    Filters={"item_id":"20"}
)
print(forecastResponse)

forecastExportName= project+'forecast_export'
outputPath = f"s3://{bucketName}/output"

forecast_export_response = forecast.create_forecast_export_job(
                                                                ForecastExportJobName = forecastExportName,
                                                                ForecastArn=forecastArn,
                                                                Destination = {
                                                                   "S3Config" : {
                                                                       "Path":outputPath,
                                                                       "RoleArn": role_arn
                                                                   }
                                                                }
                                                              )

forecastExportJobArn = forecast_export_response['ForecastExportJobArn']
print(forecastExportJobArn)