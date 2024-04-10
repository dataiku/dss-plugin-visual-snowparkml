### SECTION 1 - Package Imports
# Dataiku Imports
import dataiku
from dataiku.customrecipe import get_input_names_for_role
from dataiku.customrecipe import get_output_names_for_role
from dataiku.customrecipe import get_recipe_config
from dataiku import pandasutils as pdu
from dataiku.snowpark import DkuSnowpark
from dataiku import customrecipe

from visualsnowparkml.plugin_config_loading import load_score_config_snowpark_session

# Other ML Imports
import pandas as pd, numpy as np
import json
import os
import pprint
from traceback import format_tb

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
from snowflake.ml.registry import Registry


### SECTION 2 - Load User-Inputted Config, Inputs, and Outputs
params, session, input_dataset, saved_model_id, output_score_dataset = load_score_config_snowpark_session()

# Get recipe user-inputted parameters and print to the logs
print("-----------------------------")
print("Recipe Input Params")
attrs = dir(params)
for attr in attrs:
    if not attr.startswith('__'):
        print(str(attr) + ': ' + str(getattr(params, attr)))
print("-----------------------------")

# Models trained using the plugin will be deployed to the Snowflake MODEL_REGISTRY database
#snowflake_model_registry = "MODEL_REGISTRY"

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

# Get the Snowflake Model Registry
registry = Registry(session = session)

# Get the Snowflake Model Registry model that matches the input Dataiku Saved Model active version
try:
    model = registry.get_model(snowflake_model_name).version(active_model_version_id)

except KeyError as err:
    raise KeyError(format_tb(err.__traceback__)[0] + err.args[0] + "\nMake sure that your input model was trained using the Visual Snowpark ML train plugin recipe, and that the model was successfully deployed the model to the Snowpark ML registry.") from None
    
#loaded_model = model.load_model()

# Get the input Snowpark dataframe to score
input_dataset_snow_df = dku_snowpark.get_dataframe(input_dataset, session = session)

# Use the Snowflake Model Registry model to predict new records
# Two-class classification here
if prediction_type == 'BINARY_CLASSIFICATION':
    
    # If no 'SAMPLE_WEIGHTS' column in input dataset, create this column and populate with nulls
    if 'SAMPLE_WEIGHTS' not in input_dataset_snow_df.columns:
        input_dataset_snow_df = input_dataset_snow_df.withColumn('SAMPLE_WEIGHTS', F.lit(None).cast(T.StringType()))
    
    # Make predictions - custom implementation of 'PREDICTION' based on optimal threshold
    predictions = model.run(input_dataset_snow_df, function_name = "predict_proba")
    print("PATPAT")
    print(predictions.columns)
    predictions.show(5)
    target_col_value_cols = [col for col in predictions.columns if "predict_proba" in col]
    target_col_values = [col.replace('"','').replace('predict_proba_','') for col in target_col_value_cols]
    
    print(target_col_value_cols)
    print(target_col_values)
    predictions = predictions.withColumn('PREDICTION', F.when(F.col(target_col_value_cols[-1]) > model_threshold, target_col_values[-1]).otherwise(target_col_values[0]))
    predictions = predictions.drop('SAMPLE_WEIGHTS')

# Make predictions for regression models
else:
    predictions = model.run(input_dataset_snow_df, function_name = "predict")

# Write the predictions to output Snowflake dataset
dku_snowpark.write_with_schema(output_score_dataset, predictions)
