# SECTION 1 - Package Imports
# Dataiku Imports
import dataiku
from dataiku.snowpark import DkuSnowpark

from visualsnowparkml.plugin_config_loading import load_score_config_snowpark_session

# Other Imports
import pandas as pd
import numpy as np
from traceback import format_tb

# Snowpark Imports
import snowflake.snowpark.types as T

# Snowpark-ML Imports
import snowflake.snowpark.functions as F
from snowflake.ml.registry import Registry


# SECTION 2 - Load User-Inputted Config, Inputs, and Outputs
params, session, input_dataset, saved_model_id, output_score_dataset = load_score_config_snowpark_session()

# Get recipe user-inputted parameters and print to the logs
print("-----------------------------")
print("Recipe Input Params")
attrs = dir(params)
for attr in attrs:
    if not attr.startswith('__'):
        print(str(attr) + ': ' + str(getattr(params, attr)))
print("-----------------------------")

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

# SECTION 4 - Set up Snowpark
dku_snowpark = DkuSnowpark()

# Get the Snowflake Model Registry
registry = Registry(session=session)

# Get the Snowflake Model Registry model that matches the input Dataiku Saved Model active version
try:
    model = registry.get_model(snowflake_model_name).version(active_model_version_id)

except KeyError as err:
    raise KeyError(format_tb(err.__traceback__)[0] + err.args[0] + "\nMake sure that your input model was trained using the Visual Snowpark ML train plugin recipe, and that the model was successfully deployed the model to the Snowpark ML registry.") from None

# Get the input Snowpark dataframe to score
input_dataset_snow_df = dku_snowpark.get_dataframe(input_dataset, session=session)

# Use the Snowflake Model Registry model to predict new records
# Two-class classification here
if prediction_type == 'BINARY_CLASSIFICATION':

    # If no 'SAMPLE_WEIGHTS' column in input dataset, create this column and populate with nulls
    if 'SAMPLE_WEIGHTS' not in input_dataset_snow_df.columns:
        input_dataset_snow_df = input_dataset_snow_df.withColumn('SAMPLE_WEIGHTS', F.lit(None).cast(T.StringType()))

    # Make predictions - custom implementation of 'PREDICTION' based on optimal threshold
    predictions = model.run(input_dataset_snow_df, function_name="predict_proba")
    target_col_value_cols = [col for col in predictions.columns if "PREDICT_PROBA" in col]
    target_col_values = [col.replace('"', '').replace('PREDICT_PROBA_', '') for col in target_col_value_cols]
    predictions = predictions.withColumn('PREDICTION', F.when(F.col(target_col_value_cols[-1]) > model_threshold, target_col_values[-1]).otherwise(target_col_values[0]))
    for target_col_value_col in target_col_value_cols:
        predictions = predictions.withColumnRenamed(target_col_value_col, target_col_value_col.replace('"', ''))
    predictions = predictions.drop('SAMPLE_WEIGHTS')

elif prediction_type == 'MULTICLASS':
    # If no 'SAMPLE_WEIGHTS' column in input dataset, create this column and populate with nulls
    if 'SAMPLE_WEIGHTS' not in input_dataset_snow_df.columns:
        input_dataset_snow_df = input_dataset_snow_df.withColumn('SAMPLE_WEIGHTS', F.lit(None).cast(T.StringType()))

    # Make predictions - custom implementation of 'PREDICTION' based on optimal threshold
    predictions = model.run(input_dataset_snow_df, function_name="predict_proba")
    predictions = model.run(predictions, function_name="predict")
    target_col_value_cols = [col for col in predictions.columns if "PREDICT_PROBA" in col]
    target_col_values = [col.replace('"', '').replace('PREDICT_PROBA_', '') for col in target_col_value_cols]
    for target_col_value_col in target_col_value_cols:
        predictions = predictions.withColumnRenamed(target_col_value_col, target_col_value_col.replace('"', ''))
    predictions = predictions.drop('SAMPLE_WEIGHTS')

# Make predictions for regression models
else:
    predictions = model.run(input_dataset_snow_df, function_name="predict")

# Write the predictions to output Snowflake dataset
dku_snowpark.write_with_schema(output_score_dataset, predictions)
