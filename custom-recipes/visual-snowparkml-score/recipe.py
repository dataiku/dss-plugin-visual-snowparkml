# Package Imports
import dataiku
from dataiku.snowpark import DkuSnowpark
from visualsnowparkml.plugin_config_loading import load_score_config_snowpark_session
from traceback import format_tb
import snowflake.snowpark.types as T
import snowflake.snowpark.functions as F
from snowflake.ml.registry import Registry

import logging

logger = logging.getLogger("visualsnowflakemlplugin")
logging.basicConfig(
    level=logging.INFO,
    format='Visual Snowflake ML plugin %(levelname)s - %(message)s'
)


def log_params(params):
    """Log all non-dunder attributes of the params object."""
    logger.info("Recipe Input Params")
    for attr in dir(params):
        if not attr.startswith('__'):
            logger.info(f"{attr}: {getattr(params, attr)}")


def add_sample_weights_column_if_missing(snowpark_df):
    """Add a null SAMPLE_WEIGHTS column if it doesn't exist."""
    if 'SAMPLE_WEIGHTS' not in snowpark_df.columns:
        return snowpark_df.withColumn('SAMPLE_WEIGHTS', F.lit(None).cast(T.StringType()))
    return snowpark_df


def strip_quotes_from_proba_columns(snowpark_df, proba_columns):
    """Remove surrounding quotes from PREDICT_PROBA column names."""
    for col_name in proba_columns:
        if col_name.startswith('"') and col_name.endswith('"'):
            snowpark_df = snowpark_df.withColumnRenamed(
                col_name, col_name.replace('"', ''))
    return snowpark_df


def get_proba_columns_and_values(snowpark_df):
    """Extract PREDICT_PROBA columns and their corresponding class values."""
    proba_columns = [
        col for col in snowpark_df.columns if "PREDICT_PROBA" in col]
    class_values = [col.replace('"', '').replace(
        'PREDICT_PROBA_', '') for col in proba_columns]
    return proba_columns, class_values


# Load Configuration
params, session, input_dataset, saved_model_id, output_score_dataset = load_score_config_snowpark_session()
log_params(params)

client = dataiku.api_client()
project = client.get_default_project()

saved_model = project.get_saved_model(saved_model_id)
saved_model_name = saved_model.get_settings().settings['name']
active_model_version_id = saved_model.get_active_version()['id']
saved_model_version_details = saved_model.get_version_details(
    active_model_version_id)

prediction_type = saved_model_version_details.details['coreParams']['prediction_type']

# If the Saved Model is two-class classification, get the model threshold from the Dataiku MLflow model object
if prediction_type == 'BINARY_CLASSIFICATION':
    model_threshold = saved_model_version_details.details['userMeta']['activeClassifierThreshold']
    logger.info(f"Model threshold: {model_threshold}")

# Get the Snowflake model name (standard name per the SnowparkML train plugin)
snowflake_model_name = project.project_key + "_" + saved_model_name

# SECTION 4 - Set up Snowpark
dku_snowpark = DkuSnowpark()

# Get the Snowflake Model Registry
registry = Registry(session=session)

# Get the Snowflake Model Registry model that matches the input Dataiku Saved Model active version.
# Also change the default model version in Snowflake if it doesn't match the active version in Dataiku
try:
    parent_model = registry.get_model(snowflake_model_name)
    parent_model.default = active_model_version_id
    model_version = parent_model.version(active_model_version_id)

except KeyError as err:
    raise KeyError(format_tb(err.__traceback__)[
                   0] + err.args[0] + "\nMake sure that your input model was trained using the Visual Snowpark ML train plugin recipe, and that the model was successfully deployed the model to the Snowpark ML registry.") from None

# Get the input Snowpark dataframe to score
input_dataset_snow_df = dku_snowpark.get_dataframe(
    input_dataset, session=session)

# Generate predictions based on prediction type
logger.info(f"Model version functions: {model_version.show_functions()}")

if prediction_type == 'BINARY_CLASSIFICATION':
    input_dataset_snow_df = add_sample_weights_column_if_missing(
        input_dataset_snow_df)
    predictions_snow_df = model_version.run(
        input_dataset_snow_df, function_name='"predict_proba"')

    proba_columns, class_values = get_proba_columns_and_values(
        predictions_snow_df)

    # Calculate predicted class based on model threshold from Dataiku MLflow model UI
    predictions_snow_df = predictions_snow_df.withColumn(
        'PREDICTION',
        F.when(F.col(proba_columns[-1]) > model_threshold,
               class_values[-1]).otherwise(class_values[0])
    )
    predictions_snow_df = strip_quotes_from_proba_columns(
        predictions_snow_df, proba_columns)
    predictions_snow_df = predictions_snow_df.drop('SAMPLE_WEIGHTS')

elif prediction_type == 'MULTICLASS':
    input_dataset_snow_df = add_sample_weights_column_if_missing(
        input_dataset_snow_df)
    predictions_snow_df = model_version.run(
        input_dataset_snow_df, function_name='"predict_proba"')
    predictions_snow_df = model_version.run(
        predictions_snow_df, function_name='"predict"')
    proba_columns, _ = get_proba_columns_and_values(predictions_snow_df)
    predictions_snow_df = strip_quotes_from_proba_columns(
        predictions_snow_df, proba_columns)
    predictions_snow_df = predictions_snow_df.drop('SAMPLE_WEIGHTS')

else:  # Regression
    predictions_snow_df = model_version.run(
        input_dataset_snow_df, function_name='"predict"')

dku_snowpark.write_with_schema(output_score_dataset, predictions_snow_df)
