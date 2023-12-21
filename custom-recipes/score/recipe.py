### SECTION 1 - Package Imports
# Dataiku Imports
import dataiku
from dataiku.customrecipe import get_input_names_for_role
from dataiku.customrecipe import get_output_names_for_role
from dataiku.customrecipe import get_recipe_config
from dataiku import pandasutils as pdu
from dataiku.snowpark import DkuSnowpark
from dataiku import customrecipe

# Other ML Imports
import pandas as pd, numpy as np
import json
import os
import pprint

# Snowpark Imports
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import snowflake.snowpark
from snowflake.snowpark.version import VERSION
from snowflake.snowpark.session import Session
import snowflake.snowpark.types as T
from snowflake.snowpark.functions import sproc, udf, when, col

# Snowpark-ML Imports
import snowflake.snowpark.functions as F
from snowflake.ml.registry import model_registry

### SECTION 2 - Recipe Inputs, Outputs, and User-Inputted Parameters
# Get input and output datasets
input_dataset_names = get_input_names_for_role('input_dataset_name')
input_dataset_name = input_dataset_names[0]
input_dataset = dataiku.Dataset(input_dataset_name) 

output_score_dataset_names = get_output_names_for_role('output_score_dataset_name')
output_score_dataset_name = output_score_dataset_names[0]
output_score_dataset = dataiku.Dataset(output_score_dataset_name)

# Get the input Dataiku Saved Model name
saved_model_names = get_input_names_for_role('saved_model_name')
saved_model_name = saved_model_names[0]
saved_model_id = saved_model_name.split(".")[1]

# Models trained using the plugin will be deployed to the Snowflake MODEL_REGISTRY database
snowflake_model_registry = "MODEL_REGISTRY"

# Get recipe user-inputted parameters and print to the logs
recipe_config = get_recipe_config()
print("-----------------------------")
print("Recipe Input Config")
pprint.pprint(recipe_config)
print("-----------------------------")

warehouse = recipe_config.get('warehouse', None)

client = dataiku.api_client()
project = client.get_default_project()

saved_model = project.get_saved_model(saved_model_id)
saved_model_name = saved_model.get_settings().settings['name']
active_model_version_id = saved_model.get_active_version()['id']
saved_model_version_details = saved_model.get_version_details(active_model_version_id)

prediction_type = saved_model_version_details.details['coreParams']['prediction_type']

# If the Saved Model is two-class classification, get the optimal model threshold
if prediction_type == 'BINARY_CLASSIFICATION':
    model_threshold = saved_model_version_details.details['perf']['usedThreshold']

# Get the Snowflake model name (standard name per the SnowparkML train plugin)
snowflake_model_name = project.project_key + "_" + saved_model_name

### SECTION 4 - Set up Snowpark
dku_snowpark = DkuSnowpark()

snowflake_connection_name = input_dataset.get_config()['params']['connection']

session = dku_snowpark.get_session(snowflake_connection_name)

# Change the Snowflake warehouse if user chose to override the connection's default warehouse
if warehouse:
    warehouse = f'"{warehouse}"'
    session.use_warehouse(warehouse)

# If the Snowflake connection doesn't have a default schema, pull the schema name from the input dataset settings
connection_schema = session.get_current_schema()
if not connection_schema:    
    input_dataset_info = input_dataset.get_location_info()
    input_dataset_schema = input_dataset_info['info']['schema']
    session.use_schema(input_dataset_schema)

# Get the Snowflake Model Registry
registry = model_registry.ModelRegistry(session = session, database_name = snowflake_model_registry)

# Get the Snowflake Model Registry model that matches the input Dataiku Saved Model active version
model = model_registry.ModelReference(registry = registry, 
                                      model_name = snowflake_model_name, 
                                      model_version = active_model_version_id)
loaded_model = model.load_model()

# Get the input Snowpark dataframe to score
input_dataset_snow_df = dku_snowpark.get_dataframe(input_dataset, session = session)

# Use the Snowflake Model Registry model to predict new records
# Two-class classification here
if prediction_type == 'BINARY_CLASSIFICATION':
    
    # If no 'SAMPLE_WEIGHTS' column in input dataset, create this column and populate with nulls
    if 'SAMPLE_WEIGHTS' not in input_dataset_snow_df.columns:
        input_dataset_snow_df = input_dataset_snow_df.withColumn('SAMPLE_WEIGHTS', F.lit(None).cast(T.StringType()))
    
    # Make predictions - custom implementation of 'PREDICTION' based on optimal threshold
    predictions = loaded_model.predict_proba(input_dataset_snow_df)
    print("HIHIHI")
    print(predictions.columns)
    target_col_value_cols = [col for col in predictions.columns if "predict_proba" in col]
    target_col_values = [col.replace('"','').replace('predict_proba_','') for col in target_col_value_cols]
    predictions = predictions.withColumn('PREDICTION', F.when(F.col(target_col_value_cols[-1]) > model_threshold, target_col_values[-1]).otherwise(target_col_values[0]))
    predictions = predictions.drop('SAMPLE_WEIGHTS')

# Make predictions for regression models
else:
    predictions = loaded_model.predict(input_dataset_snow_df)

# Write the predictions to output Snowflake dataset
dku_snowpark.write_with_schema(output_score_dataset, predictions)
