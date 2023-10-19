### SECTION 1 - Package Imports
# Dataiku Imports

import dataiku
from dataiku.customrecipe import get_input_names_for_role
from dataiku.customrecipe import get_output_names_for_role
from dataiku.customrecipe import get_recipe_config
from dataiku import pandasutils as pdu
from dataiku.snowpark import DkuSnowpark
from dataikuapi.dss.ml import DSSPredictionMLTaskSettings
from dataiku.core.flow import FLOW
from dataiku import customrecipe

# Other ML Imports
import pandas as pd, numpy as np
import mlflow
from mlflow.deployments import get_deploy_client
import json
import joblib
from itertools import chain
from scipy.stats import uniform, truncnorm, randint, loguniform
from datetime import datetime
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score, r2_score, mean_absolute_error, mean_squared_error, d2_absolute_error_score, d2_pinball_score
from sklearn.metrics import classification_report
import os
import re
from cloudpickle import dump, load
import sys
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
from snowflake.ml.utils import connection_params
from snowflake.ml.modeling.metrics.correlation import correlation
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.model_selection import RandomizedSearchCV, GridSearchCV
from snowflake.ml.modeling.compose import ColumnTransformer
from snowflake.ml.modeling.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from snowflake.ml.modeling.xgboost import XGBClassifier, XGBRegressor
from snowflake.ml.modeling.lightgbm import LGBMClassifier, LGBMRegressor
from snowflake.ml.modeling.tree import DecisionTreeClassifier, DecisionTreeRegressor
from snowflake.ml.modeling.linear_model import LogisticRegression, Lasso, PoissonRegressor, GammaRegressor, TweedieRegressor
from snowflake.ml.modeling.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from snowflake.ml.modeling.impute import SimpleImputer
#from snowflake.ml.modeling.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score, r2_score, mean_absolute_error, mean_squared_error, d2_absolute_error_score, d2_pinball_score
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

saved_model_names = get_input_names_for_role('saved_model_name')
saved_model_name = saved_model_names[0]
saved_model_id = saved_model_name.split(".")[1]
snowflake_model_registry = "MODEL_REGISTRY"

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
model_threshold = saved_model_version_details.details['perf']['usedThreshold']

snowflake_model_name = project.project_key + "_" + saved_model_name

### SECTION 4 - Set up Snowpark
dku_snowpark = DkuSnowpark()

snowflake_connection_name = input_dataset.get_config()['params']['connection']

session = dku_snowpark.get_session(snowflake_connection_name)

if warehouse:
    warehouse = f'"{warehouse}"'
    session.use_warehouse(warehouse)

connection_schema = session.get_current_schema()

if not connection_schema:    
    input_dataset_info = input_dataset.get_location_info()
    input_dataset_schema = input_dataset_info['info']['schema']
    session.use_schema(input_dataset_schema)

registry = model_registry.ModelRegistry(session = session, database_name = snowflake_model_registry)

model = model_registry.ModelReference(registry = registry, 
                                      model_name = snowflake_model_name, 
                                      model_version = active_model_version_id)
loaded_model = model.load_model()

input_dataset_snow_df = dku_snowpark.get_dataframe(input_dataset, session = session)
print("111")
print(input_dataset_snow_df.columns)
if prediction_type == 'BINARY_CLASSIFICATION':
    if 'SAMPLE_WEIGHTS' not in input_dataset_snow_df.columns:
        input_dataset_snow_df = input_dataset_snow_df.withColumn('SAMPLE_WEIGHTS', F.lit(None).cast(T.StringType()))
        print("222")
        print(input_dataset_snow_df.columns)
    predictions = loaded_model.predict_proba(input_dataset_snow_df)
    print("333")
    print(predictions.columns)
    predictions = predictions.withColumn('PREDICTION', F.when(F.col('PREDICT_PROBA_1') > model_threshold, 1).otherwise(0))
    predictions = predictions.drop('SAMPLE_WEIGHTS')
else:
    predictions = loaded_model.predict_proba(input_dataset_snow_df)

dku_snowpark.write_with_schema(output_score_dataset, predictions)
