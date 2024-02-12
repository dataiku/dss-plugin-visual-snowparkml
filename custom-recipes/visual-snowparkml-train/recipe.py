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
import joblib
from itertools import chain
from scipy.stats import uniform, truncnorm, randint, loguniform
from datetime import datetime
import os
import re
from cloudpickle import dump, load
import sys
import pprint
import time

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
from snowflake.ml.modeling.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score, r2_score, mean_absolute_error, mean_squared_error
import snowflake.snowpark.functions as F
from snowflake.ml.registry import model_registry

### SECTION 2 - Recipe Inputs, Outputs, and User-Inputted Parameters
# Get input and output datasets
input_dataset_names = get_input_names_for_role('input_dataset_name')
input_dataset_name = input_dataset_names[0]
input_dataset = dataiku.Dataset(input_dataset_name) 

output_train_dataset_names = get_output_names_for_role('output_train_dataset_name')
output_train_dataset = dataiku.Dataset(output_train_dataset_names[0])

output_test_dataset_names = get_output_names_for_role('output_test_dataset_name')
output_test_dataset = dataiku.Dataset(output_test_dataset_names[0])

model_experiment_tracking_folder_names = get_output_names_for_role('model_experiment_tracking_folder_name')
model_experiment_tracking_folder = dataiku.Folder(model_experiment_tracking_folder_names[0])
model_experiment_tracking_folder_id = model_experiment_tracking_folder.get_id()
    
# Get recipe user-inputted parameters and print to the logs
recipe_config = get_recipe_config()
print("-----------------------------")
print("Recipe Input Config")
pprint.pprint(recipe_config)
print("-----------------------------")

model_name = recipe_config.get('model_name', None)
col_label = recipe_config.get('col_label', None)
prediction_type = recipe_config.get('prediction_type', None)
disable_class_weights = recipe_config.get('disable_class_weights', None)
time_ordering = recipe_config.get('time_ordering', False)
time_ordering_variable = recipe_config.get('time_ordering_variable', None)

train_ratio = recipe_config.get('train_ratio', None)
random_seed = recipe_config.get('random_seed', None)
model_metric = recipe_config.get('model_metric', None)
warehouse = recipe_config.get('warehouse', None)
deploy_to_snowflake_model_registry = recipe_config.get('deploy_to_snowflake_model_registry', False)
snowflake_model_registry = 'MODEL_REGISTRY'

# Map metric name from dropdown to sklearn-compatible name
if model_metric == 'ROC AUC':
    scoring_metric = 'roc_auc'
elif model_metric == 'Accuracy':
    scoring_metric = 'accuracy'
elif model_metric == 'F1 Score':
    scoring_metric = 'f1'
elif model_metric == 'Precision':
    scoring_metric = 'precision'
elif model_metric == 'Recall':
    scoring_metric = 'recall'
elif model_metric == 'R2':
    scoring_metric = 'r2'
elif model_metric == 'MAE':
    scoring_metric = 'neg_mean_absolute_error'
elif model_metric == 'MSE':
    scoring_metric = 'neg_mean_squared_error'

inputDatasetColumns = recipe_config.get('inputDatasetColumns', None)
selectedInputColumns = recipe_config.get('selectedInputColumns', None)
selectedOption1 = recipe_config.get('selectedOption1', None)
selectedOption2 = recipe_config.get('selectedOption2', None)
selectedConstantImpute = recipe_config.get('selectedConstantImpute', None)

logistic_regression = recipe_config.get('logistic_regression', None)
logistic_regression_c_min = recipe_config.get('logistic_regression_c_min', None)
logistic_regression_c_max = recipe_config.get('logistic_regression_c_max', None)

random_forest_classification = recipe_config.get('random_forest_classification', None)
random_forest_classification_n_estimators_min = recipe_config.get('random_forest_classification_n_estimators_min', None)
random_forest_classification_n_estimators_max = recipe_config.get('random_forest_classification_n_estimators_max', None)
random_forest_classification_max_depth_min = recipe_config.get('random_forest_classification_max_depth_min', None)
random_forest_classification_max_depth_max = recipe_config.get('random_forest_classification_max_depth_max', None)
random_forest_classification_min_samples_leaf_min = recipe_config.get('random_forest_classification_min_samples_leaf_min', None)
random_forest_classification_min_samples_leaf_max = recipe_config.get('random_forest_classification_min_samples_leaf_max', None)

xgb_classification = recipe_config.get('xgb_classification', None)
xgb_classification_n_estimators_min = recipe_config.get('xgb_classification_n_estimators_min', None)
xgb_classification_n_estimators_max = recipe_config.get('xgb_classification_n_estimators_max', None)
xgb_classification_max_depth_min = recipe_config.get('xgb_classification_max_depth_min', None)
xgb_classification_max_depth_max = recipe_config.get('xgb_classification_max_depth_max', None)
xgb_classification_min_child_weight_min = recipe_config.get('xgb_classification_min_child_weight_min', None)
xgb_classification_min_child_weight_max = recipe_config.get('xgb_classification_min_child_weight_max', None)
xgb_classification_learning_rate_min = recipe_config.get('xgb_classification_learning_rate_min', None)
xgb_classification_learning_rate_max = recipe_config.get('xgb_classification_learning_rate_max', None)

lgbm_classification = recipe_config.get('lgbm_classification', None)
lgbm_classification_n_estimators_min = recipe_config.get('lgbm_classification_n_estimators_min', None)
lgbm_classification_n_estimators_max = recipe_config.get('lgbm_classification_n_estimators_max', None)
lgbm_classification_max_depth_min = recipe_config.get('lgbm_classification_max_depth_min', None)
lgbm_classification_max_depth_max = recipe_config.get('lgbm_classification_max_depth_max', None)
lgbm_classification_min_child_weight_min = recipe_config.get('lgbm_classification_min_child_weight_min', None)
lgbm_classification_min_child_weight_max = recipe_config.get('lgbm_classification_min_child_weight_max', None)
lgbm_classification_learning_rate_min = recipe_config.get('lgbm_classification_learning_rate_min', None)
lgbm_classification_learning_rate_max = recipe_config.get('lgbm_classification_learning_rate_max', None)

gb_classification = recipe_config.get('gb_classification', None)
gb_classification_n_estimators_min = recipe_config.get('gb_classification_n_estimators_min', None)
gb_classification_n_estimators_max = recipe_config.get('gb_classification_n_estimators_max', None)
gb_classification_max_depth_min = recipe_config.get('gb_classification_max_depth_min', None)
gb_classification_max_depth_max = recipe_config.get('gb_classification_max_depth_max', None)
gb_classification_min_samples_leaf_min = recipe_config.get('gb_classification_min_samples_leaf_min', None)
gb_classification_min_samples_leaf_max = recipe_config.get('gb_classification_min_samples_leaf_max', None)
gb_classification_learning_rate_min = recipe_config.get('gb_classification_learning_rate_min', None)
gb_classification_learning_rate_max = recipe_config.get('gb_classification_learning_rate_max', None)

decision_tree_classification = recipe_config.get('decision_tree_classification', None)
decision_tree_classification_max_depth_min = recipe_config.get('decision_tree_classification_max_depth_min', None)
decision_tree_classification_max_depth_max = recipe_config.get('decision_tree_classification_max_depth_max', None)
decision_tree_classification_min_samples_leaf_min = recipe_config.get('decision_tree_classification_min_samples_leaf_min', None)
decision_tree_classification_min_samples_leaf_max = recipe_config.get('decision_tree_classification_min_samples_leaf_max', None)

lasso_regression = recipe_config.get('lasso_regression', None)
lasso_regression_alpha_min = recipe_config.get('lasso_regression_alpha_min', None)
lasso_regression_alpha_max = recipe_config.get('lasso_regression_alpha_max', None)

random_forest_regression = recipe_config.get('random_forest_regression', None)
random_forest_regression_n_estimators_min = recipe_config.get('random_forest_regression_n_estimators_min', None)
random_forest_regression_n_estimators_max = recipe_config.get('random_forest_regression_n_estimators_max', None)
random_forest_regression_max_depth_min = recipe_config.get('random_forest_regression_max_depth_min', None)
random_forest_regression_max_depth_max = recipe_config.get('random_forest_regression_max_depth_max', None)
random_forest_regression_min_samples_leaf_min = recipe_config.get('random_forest_regression_min_samples_leaf_min', None)
random_forest_regression_min_samples_leaf_max = recipe_config.get('random_forest_regression_min_samples_leaf_max', None)

xgb_regression = recipe_config.get('xgb_regression', None)
xgb_regression_n_estimators_min = recipe_config.get('xgb_regression_n_estimators_min', None)
xgb_regression_n_estimators_max = recipe_config.get('xgb_regression_n_estimators_max', None)
xgb_regression_max_depth_min = recipe_config.get('xgb_regression_max_depth_min', None)
xgb_regression_max_depth_max = recipe_config.get('xgb_regression_max_depth_max', None)
xgb_regression_min_child_weight_min = recipe_config.get('xgb_regression_min_child_weight_min', None)
xgb_regression_min_child_weight_max = recipe_config.get('xgb_regression_min_child_weight_max', None)
xgb_regression_learning_rate_min = recipe_config.get('xgb_regression_learning_rate_min', None)
xgb_regression_learning_rate_max = recipe_config.get('xgb_regression_learning_rate_max', None)

lgbm_regression = recipe_config.get('lgbm_regression', None)
lgbm_regression_n_estimators_min = recipe_config.get('lgbm_regression_n_estimators_min', None)
lgbm_regression_n_estimators_max = recipe_config.get('lgbm_regression_n_estimators_max', None)
lgbm_regression_max_depth_min = recipe_config.get('lgbm_regression_max_depth_min', None)
lgbm_regression_max_depth_max = recipe_config.get('lgbm_regression_max_depth_max', None)
lgbm_regression_min_child_weight_min = recipe_config.get('lgbm_regression_min_child_weight_min', None)
lgbm_regression_min_child_weight_max = recipe_config.get('lgbm_regression_min_child_weight_max', None)
lgbm_regression_learning_rate_min = recipe_config.get('lgbm_regression_learning_rate_min', None)
lgbm_regression_learning_rate_max = recipe_config.get('lgbm_regression_learning_rate_max', None)

gb_regression = recipe_config.get('gb_regression', None)
gb_regression_n_estimators_min = recipe_config.get('gb_regression_n_estimators_min', None)
gb_regression_n_estimators_max = recipe_config.get('gb_regression_n_estimators_max', None)
gb_regression_max_depth_min = recipe_config.get('gb_regression_max_depth_min', None)
gb_regression_max_depth_max = recipe_config.get('gb_regression_max_depth_max', None)
gb_regression_min_samples_leaf_min = recipe_config.get('gb_regression_min_samples_leaf_min', None)
gb_regression_min_samples_leaf_max = recipe_config.get('gb_regression_min_samples_leaf_max', None)
gb_regression_learning_rate_min = recipe_config.get('gb_regression_learning_rate_min', None)
gb_regression_learning_rate_max = recipe_config.get('gb_regression_learning_rate_max', None)

decision_tree_regression = recipe_config.get('decision_tree_regression', None)
decision_tree_regression_max_depth_min = recipe_config.get('decision_tree_regression_max_depth_min', None)
decision_tree_regression_max_depth_max = recipe_config.get('decision_tree_regression_max_depth_max', None)
decision_tree_regression_min_samples_leaf_min = recipe_config.get('decision_tree_regression_min_samples_leaf_min', None)
decision_tree_regression_min_samples_leaf_max = recipe_config.get('decision_tree_regression_min_samples_leaf_max', None)

n_iter = recipe_config.get('n_iter', None)

### SECTION 3 - Set up MLflow Experiment Tracking
# MLFLOW Variables
MLFLOW_CODE_ENV_NAME = "py_38_snowpark"
MLFLOW_EXPERIMENT_NAME = model_name + "_exp"
SAVED_MODEL_NAME = model_name
MODEL_NAME = model_name

# Get a Dataiku API client and the current project
client = dataiku.api_client()
project = client.get_default_project()

# Set up the Dataiku MLflow extension and setup an experiment pointing to the output models folder
mlflow_extension = project.get_mlflow_extension()
mlflow_handle = project.setup_mlflow(managed_folder=model_experiment_tracking_folder)
mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
mlflow_experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

### SECTION 4 - Set up Snowpark
# Get a Snowpark session using the input dataset Snowflake connection
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

### SECTION 5 - Add a Target Class Weights Column if Two-Class Classification and do Train/Test Split
# Convert the input dataset into a Snowpark dataframe
input_snowpark_df = dku_snowpark.get_dataframe(input_dataset)

# Create a dictionary to store all dataset columns as read in by pandas vs. how they're stored on Snowflake. 
# E.g. {'feat_1':'"feat_1"', 'feat_2':'"feat_2"', 'FEAT_1':'FEAT_1'}
# We use this lookup dictionary later on to map column names to their actual Snowflake names, 
# where many have double quotes surrounding them to prevent Snowflake from auto-capitalizing
if disable_class_weights:
    features_quotes_lookup = {}
    sample_weight_col = None
else:
    features_quotes_lookup = {'SAMPLE_WEIGHTS': 'SAMPLE_WEIGHTS'}
    sample_weight_col = 'SAMPLE_WEIGHTS'

for snowflake_column in input_snowpark_df.columns:
    if snowflake_column.startswith('"') and snowflake_column.endswith('"'):
        features_quotes_lookup[snowflake_column.replace('"', '')] = snowflake_column
    else:
        features_quotes_lookup[snowflake_column] = snowflake_column
        
# This function will call the lookup dictionary and return the Snowflake column name
def sf_col_name(col_name):
    return features_quotes_lookup[col_name]

col_label_sf = sf_col_name(col_label)

if time_ordering_variable:
    time_ordering_variable = time_ordering_variable['name']
    time_ordering_variable_sf = sf_col_name(time_ordering_variable)

# Get a list of Target column values if two-class classification
if prediction_type == "two-class classification":
    col_label_values = list(input_snowpark_df.select(sf_col_name(col_label)).distinct().to_pandas()[col_label])
else:
    col_label_values = None

# Function to retrieve the Snowflake data type from the corresponding pandas data type 
def convert_snowpark_df_col_dtype(snowpark_df, col):

    col_label_dtype_mappings = {
        'binary': T.BinaryType(),
        'boolean': T.BooleanType(),
        'decimal': T.DecimalType(),
        'double': T.DoubleType(),
        'double precision': T.DoubleType(),
        'number': T.DecimalType(),
        'numeric': T.DecimalType(),
        'float': T.FloatType(),
        'float4': T.FloatType(),
        'float8': T.FloatType(),
        'real': T.FloatType(),
        'integer': T.IntegerType(),
        'bigint': T.LongType(),
        'int': T.IntegerType(),
        'tinyint': T.IntegerType(),
        'byteint': T.IntegerType(),
        'smallint': T.ShortType(),
        'varchar': T.StringType(),
        'char': T.StringType(),
        'character': T.StringType(),
        'string(4194304)': T.StringType(),
        'text': T.StringType()
    }

    for col_dtype in snowpark_df.dtypes:
        if col_dtype[0] == col:
            new_col_dtype = col_label_dtype_mappings[col_dtype[1]]
            
    return new_col_dtype

# Function to add sample weights (inverse proportion of target class column) to the Snowpark df 
def add_sample_weights_col_to_snowpark_df(snowpark_df, col):
    sf_col = sf_col_name(col)
    y_collect = snowpark_df.select(sf_col).groupBy(sf_col).count().collect()
    unique_y = [x[col] for x in y_collect]
    total_y = sum([x["COUNT"] for x in y_collect])
    unique_y_count = len(y_collect)
    bin_count = [x["COUNT"] for x in y_collect]

    class_weights = {i: ii for i, ii in zip(unique_y, total_y / (unique_y_count * np.array(bin_count)))}

    res = []
    for key, val in class_weights.items():
        res.append([key,val])

    col_label_dtype = convert_snowpark_df_col_dtype(snowpark_df, sf_col)

    schema = T.StructType([T.StructField(sf_col, col_label_dtype), T.StructField("SAMPLE_WEIGHTS", T.DoubleType())])
    df_to_join = session.create_dataframe(res,schema)

    snowpark_df = snowpark_df.join(df_to_join, [sf_col], 'left')

    return snowpark_df

# Add sample weights column if two-class classification
if prediction_type == "two-class classification" and not disable_class_weights:
    input_snowpark_df = add_sample_weights_col_to_snowpark_df(input_snowpark_df, col_label)

# If chosen by the user, split train/test sets based on the time ordering column
if time_ordering:
    time_ordering_variable_unix = time_ordering_variable_sf + '_UNIX'
    input_snowpark_df = input_snowpark_df.withColumn(time_ordering_variable_unix, F.unix_timestamp(input_snowpark_df[time_ordering_variable_sf]))
    
    split_percentile_value = input_snowpark_df.approx_quantile(time_ordering_variable_unix, [train_ratio])[0]
    
    train_snowpark_df = input_snowpark_df.filter(col(time_ordering_variable_unix) < split_percentile_value)
    test_snowpark_df = input_snowpark_df.filter(col(time_ordering_variable_unix) >= split_percentile_value)
    
    train_snowpark_df = train_snowpark_df.drop(time_ordering_variable_unix)
    test_snowpark_df = test_snowpark_df.drop(time_ordering_variable_unix)

    print("train set nrecords: " + str(train_snowpark_df.count()))
    print("test set nrecords: " + str(test_snowpark_df.count()))
    
    #cv = TimeSeriesSplit(n_splits=3)
    cv = 3
    
# Regular train/test split
else:
    test_ratio = 1 - train_ratio
    train_snowpark_df, test_snowpark_df = input_snowpark_df.random_split(weights = [train_ratio, test_ratio], seed = random_seed)
    cv = 3

### SECTION 6 - Write Train/Test Datasets to Output Tables
dku_snowpark.write_with_schema(output_train_dataset, train_snowpark_df)
dku_snowpark.write_with_schema(output_test_dataset, test_snowpark_df)

### SECTION 7 - Create a feature preprocessing Pipeline for all selected input columns and the encoding/rescaling + imputation methods chosen
# List of numeric and categorical dtypes in order to auto-select a reasonable encoding/rescaling and missingness imputation method based on the column
numeric_dtypes_list = ['number','decimal','numeric','int','integer','bigint','smallint','tinyint','byteint',
                       'float','float4','float8','double','double precision','real']

categorical_dtypes_list = ['varchar','char','character','string','text','binary','varbinary','boolean','date',
                           'datetime','time','timestamp','timestamp_ltz','timestamp_ntz','timestamp_tz',
                           'variant','object','array','geography','geometry']

# Create list of input features and the encoding/rescaling and missingness imputation method chosen
included_features_handling_list = []

for feature_column in inputDatasetColumns:
    col_name = feature_column['name']
    col_name_sf = sf_col_name(col_name)
    feature_column['name'] = col_name_sf

    if col_name in selectedInputColumns:
        if selectedInputColumns[col_name]:
            feature_column['include'] = True
            if col_name in selectedOption1:
                feature_column["encoding_rescaling"] = selectedOption1[col_name]
            elif feature_column['type'] in numeric_dtypes_list:
                feature_column["encoding_rescaling"] = 'Standard rescaling'
            else:
                feature_column["encoding_rescaling"] = 'Dummy encoding'
                
            if col_name in selectedOption2:
                feature_column["missingness_impute"] = selectedOption2[col_name]
                if selectedOption2[col_name] == 'Constant':
                    if col_name in selectedConstantImpute:
                        feature_column["constant_impute"] = selectedConstantImpute[col_name]
                        
            elif feature_column['type'] in numeric_dtypes_list:
                feature_column["missingness_impute"] = 'Median'
            else:
                feature_column["missingness_impute"] = 'Most frequent value'

            if feature_column["encoding_rescaling"] == 'Dummy encoding':
                feature_column["max_categories"] = 20
                
            included_features_handling_list.append(feature_column)

# List of just the input feature names
included_feature_names = [feature['name'] for feature in included_features_handling_list]

# Create a list of Pipelines for each feature, the encoding/rescaling method, and missingness imputation method
col_transformer_list = []

for feature in included_features_handling_list:
    feature_name = feature["name"]    
    transformer_name = feature_name[1:-1] + '_tform'

    feature_transformers = []

    if feature["missingness_impute"] == "Average":
        feature_transformers.append(('imputer', SimpleImputer(strategy = 'mean')))
    if feature["missingness_impute"] == "Median":
        feature_transformers.append(('imputer', SimpleImputer(strategy = 'median')))
    if feature["missingness_impute"] == "Constant":
        if "constant_impute" in feature:
            feature_transformers.append(('imputer', SimpleImputer(strategy = 'constant', fill_value = feature["constant_impute"])))
        else:
            feature_transformers.append(('imputer', SimpleImputer(strategy = 'constant')))
    if feature["missingness_impute"] == "Most frequent value":
        feature_transformers.append(('imputer', SimpleImputer(strategy = 'most_frequent')))
    if feature["encoding_rescaling"] == "Standard rescaling":
        feature_transformers.append(('enc', StandardScaler()))
    if feature["encoding_rescaling"] == "Min-max rescaling":
        feature_transformers.append(('enc', MinMaxScaler()))
    if feature["encoding_rescaling"] == "Dummy encoding":
        feature_transformers.append(('enc', OneHotEncoder(handle_unknown = 'infrequent_if_exist',
                                                          max_categories = 10)))
    if feature["encoding_rescaling"] == "Ordinal encoding":
        feature_transformers.append(('enc', OrdinalEncoder(handle_unknown = 'use_encoded_value',
                                                           unknown_value = -1,
                                                           encoded_missing_value = -1)))
    col_transformer_list.append((transformer_name, Pipeline(feature_transformers), [feature_name]))

preprocessor = ColumnTransformer(transformers = col_transformer_list)

### SECTION 9 - Initialize algorithms selected and hyperparameter spaces for the RandomSearch
algorithms = []

if prediction_type == "two-class classification":
    if random_forest_classification:
        algorithms.append({'algorithm': 'random_forest_classification',
                           'sklearn_obj': RandomForestClassifier(),
                           'gs_params': {'clf__n_estimators': randint(random_forest_classification_n_estimators_min,random_forest_classification_n_estimators_max),
                                         'clf__max_depth': randint(random_forest_classification_max_depth_min,random_forest_classification_max_depth_max),
                                         'clf__min_samples_leaf': randint(random_forest_classification_min_samples_leaf_min,random_forest_classification_min_samples_leaf_max)}})

    if logistic_regression:
        algorithms.append({'algorithm': 'logistic_regression',
                           'sklearn_obj': LogisticRegression(),
                           'gs_params': {'clf__C': loguniform(logistic_regression_c_min, logistic_regression_c_max)}})

    if xgb_classification:
        algorithms.append({'algorithm': 'xgb_classification',
                           'sklearn_obj': XGBClassifier(),
                           'gs_params': {'clf__n_estimators': randint(xgb_classification_n_estimators_min,xgb_classification_n_estimators_max),
                                         'clf__max_depth': randint(xgb_classification_max_depth_min,xgb_classification_max_depth_max),
                                         'clf__min_child_weight': uniform(xgb_classification_min_child_weight_min,xgb_classification_min_child_weight_max),
                                         'clf__learning_rate': loguniform(xgb_classification_learning_rate_min,xgb_classification_learning_rate_max)}})

    if lgbm_classification:
        algorithms.append({'algorithm': 'lgbm_classification',
                           'sklearn_obj': LGBMClassifier(),
                           'gs_params': {'clf__n_estimators': randint(lgbm_classification_n_estimators_min,lgbm_classification_n_estimators_max),
                                         'clf__max_depth': randint(lgbm_classification_max_depth_min,lgbm_classification_max_depth_max),
                                         'clf__min_child_weight': uniform(lgbm_classification_min_child_weight_min,lgbm_classification_min_child_weight_max),
                                         'clf__learning_rate': loguniform(lgbm_classification_learning_rate_min,lgbm_classification_learning_rate_max)}})

    if gb_classification:
        algorithms.append({'algorithm': 'gb_classification',
                           'sklearn_obj': GradientBoostingClassifier(),
                           'gs_params': {'clf__n_estimators': randint(gb_classification_n_estimators_min, gb_classification_n_estimators_max),
                                         'clf__max_depth': randint(gb_classification_max_depth_min,gb_classification_max_depth_max),
                                         'clf__min_samples_leaf': randint(gb_classification_min_samples_leaf_min,gb_classification_min_samples_leaf_max),
                                         'clf__learning_rate': loguniform(gb_classification_learning_rate_min,gb_classification_learning_rate_max)}})    

    if decision_tree_classification:
        algorithms.append({'algorithm': 'decision_tree_classification',
                           'sklearn_obj': DecisionTreeClassifier(),
                           'gs_params': {'clf__max_depth': randint(decision_tree_classification_max_depth_min,decision_tree_classification_max_depth_max),
                                         'clf__min_samples_leaf': randint(decision_tree_classification_min_samples_leaf_min,decision_tree_classification_min_samples_leaf_max)}})

else:
    if lasso_regression:
        algorithms.append({'algorithm': 'lasso_regression',
                           'sklearn_obj': Lasso(),
                           'gs_params': {'clf__alpha': loguniform(lasso_regression_alpha_min, lasso_regression_alpha_max)}})

    if random_forest_regression:
        algorithms.append({'algorithm': 'random_forest_regression',
                           'sklearn_obj': RandomForestRegressor(),
                           'gs_params': {'clf__n_estimators': randint(random_forest_regression_n_estimators_min,random_forest_regression_n_estimators_max),
                                         'clf__max_depth': randint(random_forest_regression_max_depth_min,random_forest_regression_max_depth_max),
                                         'clf__min_samples_leaf': randint(3,5)}})

    if xgb_regression:
        algorithms.append({'algorithm': 'xgb_regression',
                           'sklearn_obj': XGBRegressor(),
                           'gs_params': {'clf__n_estimators': randint(xgb_regression_n_estimators_min,xgb_regression_n_estimators_max),
                                         'clf__max_depth': randint(xgb_regression_max_depth_min,xgb_regression_max_depth_max),
                                         'clf__min_child_weight': uniform(xgb_regression_min_child_weight_min,xgb_regression_min_child_weight_max),
                                         'clf__learning_rate': loguniform(xgb_regression_learning_rate_min,xgb_regression_learning_rate_max)}})
    if lgbm_regression:
        algorithms.append({'algorithm': 'lgbm_regression',
                           'sklearn_obj': LGBMRegressor(),
                           'gs_params': {'clf__n_estimators': randint(lgbm_regression_n_estimators_min,lgbm_regression_n_estimators_max),
                                         'clf__max_depth': randint(lgbm_regression_max_depth_min,lgbm_regression_max_depth_max),
                                         'clf__min_child_weight': uniform(lgbm_regression_min_child_weight_min,lgbm_regression_min_child_weight_max),
                                         'clf__learning_rate': loguniform(lgbm_regression_learning_rate_min,lgbm_regression_learning_rate_max)}})

    if gb_regression:
        algorithms.append({'algorithm': 'gb_regression',
                           'sklearn_obj': GradientBoostingRegressor(),
                           'gs_params': {'clf__n_estimators': randint(gb_regression_n_estimators_min, gb_regression_n_estimators_max),
                                         'clf__max_depth': randint(gb_regression_max_depth_min,gb_regression_max_depth_max),
                                         'clf__min_samples_leaf': randint(gb_regression_min_samples_leaf_min,gb_regression_min_samples_leaf_max),
                                         'clf__learning_rate': loguniform(gb_regression_learning_rate_min,gb_regression_learning_rate_max)}})    

    if decision_tree_regression:
        algorithms.append({'algorithm': 'decision_tree_regression',
                           'sklearn_obj': DecisionTreeRegressor(),
                           'gs_params': {'clf__max_depth': randint(decision_tree_regression_max_depth_min,decision_tree_regression_max_depth_max),
                                         'clf__min_samples_leaf': randint(decision_tree_regression_min_samples_leaf_min,decision_tree_regression_min_samples_leaf_max)}})

### SECTION 10 - Train all models, do RandomSearch and hyperparameter tuning

# These ML algorithm wrappers will allow Dataiku's MLflow imported model to properly evaluate the model on another dataset 
# This solves the pandas column name (e.g. 'col_1' vs. Snowflake column name '"col_1"' issue)
class SnowparkMLClassifierWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from cloudpickle import load
        self.model = load(open(context.artifacts["grid_pipe_sklearn"], 'rb'))
        self.features_quotes_lookup = load(open(context.artifacts["features_quotes_lookup"], 'rb'))
        
    def predict(self, context, input_df):
        input_df_copy = input_df.copy()
        input_df_copy.columns = [self.features_quotes_lookup[col] for col in input_df_copy.columns]
        return self.model.predict_proba(input_df_copy)

class SnowparkMLRegressorWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from cloudpickle import load
        self.model = load(open(context.artifacts["grid_pipe_sklearn"], 'rb'))
        self.features_quotes_lookup = load(open(context.artifacts["features_quotes_lookup"], 'rb'))
        
    def predict(self, context, input_df):
        input_df_copy = input_df.copy()
        input_df_copy.columns = [self.features_quotes_lookup[col] for col in input_df_copy.columns]
        return self.model.predict(input_df_copy)    

# Function to run a RandomizedSearchCV hyperparameter tuning process, passing in the preprocessing Pipeline and and algorithm 
# Return the trained RandomizedSearchCV object and algorithm name
def train_model(algo, prepr, score_met, col_lab, samp_weight_col, feat_names, train_sp_df, num_iter):
    print(f"Training model... " + algo['algorithm'])
    pipe = Pipeline(steps=[
                        ('preprocessor', prepr),
                        ('clf', algo['sklearn_obj'])
                    ])

    if prediction_type == "two-class classification":
        rs_clf = RandomizedSearchCV(estimator = pipe,
                         param_distributions=algo['gs_params'],
                         n_iter=num_iter,
                         cv=DEFAULT_CROSS_VAL_FOLDS,
                         scoring=score_met,
                         n_jobs = -1,
                         verbose=1,
                         input_cols=feat_names,
                         label_cols=col_lab,
                         output_cols="PREDICTION",
                         sample_weight_col=samp_weight_col
                         )
    else:
        rs_clf = RandomizedSearchCV(estimator = pipe,
                         param_distributions=algo['gs_params'],
                         n_iter=num_iter,
                         cv=DEFAULT_CROSS_VAL_FOLDS,
                         scoring=score_met,
                         n_jobs = -1,
                         verbose=1,
                         input_cols=feat_names,
                         label_cols=col_lab,
                         output_cols="PREDICTION"
                         )
    
    rs_clf.fit(train_sp_df)

    return {'algorithm': algo['algorithm'],'sklearn_obj': rs_clf}

# Tune hyperparameters for all models chosen - store the trained RandomizedSearchCV objects and algorithm names in a list
trained_models = []
for alg in algorithms:
    trained_model = train_model(alg, preprocessor, scoring_metric, col_label_sf, sample_weight_col, included_feature_names, train_snowpark_df, n_iter)
    trained_models.append(trained_model)

### SECTION 11 - Log all trained model hyperparameters and performance metrics to MLflow
# Function to get the current time
def now_str() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H_%M")

# Loop through all trained models, log all hyperparameter tuning cross validation metrics, calculate holdout test set metrics on the best 
# model from each algorithm, then add these best models to a final_models list
final_models = []

for model in trained_models:
    rs_clf = model['sklearn_obj']
    model_algo = model['algorithm']

    grid_pipe_sklearn = rs_clf.to_sklearn()
    
    grid_pipe_sklearn_cv_results = pd.DataFrame(grid_pipe_sklearn.cv_results_)
    grid_pipe_sklearn_cv_results['algorithm'] = model_algo
    
    now = now_str()
    
    for index, row in grid_pipe_sklearn_cv_results.iterrows():
        cv_num = index + 1
        run_name = f"{MODEL_NAME}_{now}_cv_{cv_num}"
        run = mlflow.start_run(run_name=run_name)
        mlflow.log_param("algorithm", row['algorithm'])
        
        mlflow.log_metric("mean_fit_time", row["mean_fit_time"])
        score_name = "mean_test_" + scoring_metric
        mlflow.log_metric(score_name, row["mean_test_score"])
        mlflow.log_metric("rank_test_score", int(row["rank_test_score"]))

        for col in row.index:
            if "param_" in col:
                param_name = col.replace("param_clf__","")
                mlflow.log_param(param_name, row[col])

        mlflow.end_run()
    
    run_name = f"{MODEL_NAME}_{now}_final_model"

    run = mlflow.start_run(run_name=run_name)

    model_best_params = grid_pipe_sklearn.best_params_

    whole_dataset_refit_time = grid_pipe_sklearn.refit_time_
    mlflow.log_metric("whole_dataset_refit_time", whole_dataset_refit_time)

    for param in model_best_params.keys():
        if "clf" in param:
            param_name = param.replace("clf__","")
            mlflow.log_param(param_name, model_best_params[param])
    mlflow.log_param("algorithm", model_algo)

    test_predictions_df = rs_clf.predict(test_snowpark_df)

    test_metrics = {}
    
    if prediction_type == "two-class classification":
        model_classes = grid_pipe_sklearn.classes_

        test_prediction_probas_df = rs_clf.predict_proba(test_snowpark_df)

        target_col_value_cols = [col for col in test_prediction_probas_df.columns if "PREDICT_PROBA" in col]

        test_f1 = f1_score(df = test_predictions_df, y_true_col_names = col_label_sf, y_pred_col_names = '"PREDICTION"', pos_label=col_label_values[0])
        mlflow.log_metric("test_f1_score", test_f1)
        test_roc_auc = roc_auc_score(df = test_prediction_probas_df, y_true_col_names = col_label_sf, y_score_col_names = test_prediction_probas_df.columns[-1])
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        test_accuracy = accuracy_score(df = test_predictions_df, y_true_col_names = col_label_sf, y_pred_col_names = '"PREDICTION"')
        mlflow.log_metric("test_accuracy", test_accuracy)
        test_recall = recall_score(df = test_predictions_df, y_true_col_names = col_label_sf, y_pred_col_names = '"PREDICTION"', pos_label=col_label_values[0])
        mlflow.log_metric("test_recall", test_recall)
        test_precision = precision_score(df = test_predictions_df, y_true_col_names = col_label_sf, y_pred_col_names = '"PREDICTION"', pos_label=col_label_values[0])
        mlflow.log_metric("test_precision", test_precision)
        
        test_metrics["test_f1"] = test_f1
        test_metrics["test_roc_auc"] = test_roc_auc
        test_metrics["test_accuracy"] = test_accuracy
        test_metrics["test_recall"] = test_recall
        test_metrics["test_precision"] = test_precision
        
        print("F1 Score: " + str(test_f1))
        print("ROC AUC Score: " + str(test_roc_auc))
        print("Accuracy Score: " + str(test_accuracy))
        print("Recall Score: " + str(test_recall))
        print("Precision Score: " + str(test_precision))
    
    else:
        test_r2 = r2_score(df = test_predictions_df, y_true_col_name = col_label_sf, y_pred_col_name = '"PREDICTION"')
        mlflow.log_metric("test_r2_score", test_r2)
        test_mae = mean_absolute_error(df = test_predictions_df, y_true_col_names = col_label_sf, y_pred_col_names = '"PREDICTION"')
        mlflow.log_metric("test_mae_score", test_mae)
        test_mse = mean_squared_error(df = test_predictions_df, y_true_col_names = col_label_sf, y_pred_col_names = '"PREDICTION"')
        mlflow.log_metric("test_mse_score", test_mse)
        test_rmse = mean_squared_error(df = test_predictions_df, y_true_col_names = col_label_sf, y_pred_col_names = '"PREDICTION"', squared=False)
        mlflow.log_metric("test_rmse_score", test_rmse)

        test_metrics["test_r2"] = test_r2
        test_metrics["test_mae"] = test_mae
        test_metrics["test_mse"] = test_mse
        test_metrics["test_rmse"] = test_rmse

        print("R2 Score: " + str(test_r2))
        print("Mean Absolute Error: " + str(test_mae))
        print("Mean Squared Error: " + str(test_mse))
        print("Root Mean Squared Error: " + str(test_rmse))
    
    best_score = grid_pipe_sklearn.best_score_
    
    artifacts = {
        "grid_pipe_sklearn": "grid_pipe_sklearn.pkl",
        "features_quotes_lookup": "features_quotes_lookup.pkl"
    }

    dump(grid_pipe_sklearn, open(artifacts.get("grid_pipe_sklearn"), 'wb'))
    dump(features_quotes_lookup, open(artifacts.get("features_quotes_lookup"), 'wb'))
    
    if prediction_type == "two-class classification":
        logged_model = mlflow.pyfunc.log_model(artifact_path = "model", 
                                               python_model = SnowparkMLClassifierWrapper(),
                                               artifacts = artifacts)
    else:
        logged_model = mlflow.pyfunc.log_model(artifact_path = "model", 
                                               python_model = SnowparkMLRegressorWrapper(),
                                               artifacts = artifacts)
    
    mlflow.end_run()
    best_run_id = run.info.run_id
    final_models.append({'algorithm': model_algo,
                         'sklearn_obj': grid_pipe_sklearn,
                         'snowml_obj': rs_clf,
                         'mlflow_best_run_id': best_run_id,
                         'run_name': run_name,
                         'best_score': best_score,
                         'test_metrics': test_metrics})
    
### SECTION 12 - Pull the best model, import it into a SavedModel green diamond (it will create a new one if doesn't exist), and evaluate on the hold out Test dataset
# Get the final best model (of the best models of each algorithm type) based on the performance metric chosen
if scoring_metric in ['roc_auc','accuracy','f1','precision','recall','r2']:
    best_model = max(final_models, key=lambda x:x['best_score'])
else:
    best_model = min(final_models, key=lambda x:x['best_score'])
    
best_model_run_id = best_model['mlflow_best_run_id']

# If two-class classification, set the Dataiku MLflow imported model run inference info 
if prediction_type == "two-class classification":
    model_classes = best_model['sklearn_obj'].classes_
    if 'int' in str(type(model_classes[0])):
        model_classes = [int(model_class) for model_class in model_classes]
    mlflow_extension.set_run_inference_info(run_id = best_model_run_id, 
                                            prediction_type = 'BINARY_CLASSIFICATION',
                                            classes = list(model_classes),
                                            code_env_name = MLFLOW_CODE_ENV_NAME) 

# Get the managed folder subpath for the best trained model
model_artifact_first_directory = re.search(r'.*/(.+$)', mlflow_experiment.artifact_location).group(1)
model_path = f"{model_artifact_first_directory}/{best_model_run_id}/artifacts/model"

# If the Saved Model already exists in the flow (matching the user-inputted model name in the plugin), get it 
sm_id = None
for sm in project.list_saved_models():
    if sm["name"] != model_name:
        continue
    else:
        sm_id = sm["id"]
        print(f"Found Saved Model {sm['name']} with id {sm['id']}")
        break

if sm_id:
    sm = project.get_saved_model(sm_id)

# If the Saved Model does not exist, create a new placeholder
else:
    if prediction_type == "two-class classification":
        sm = project.create_mlflow_pyfunc_model(name = model_name,
                                                prediction_type = DSSPredictionMLTaskSettings.PredictionTypes.BINARY)
    else:
        sm = project.create_mlflow_pyfunc_model(name = model_name,
                                                prediction_type = DSSPredictionMLTaskSettings.PredictionTypes.REGRESSION)
    sm_id = sm.id
    print(f"Saved Model not found, created new one with id {sm_id}")
    
# Import the final trained model into the Dataiku Saved Model (Green Diamond)
mlflow_version = sm.import_mlflow_version_from_managed_folder(version_id = best_model["run_name"],
                                                              managed_folder = model_experiment_tracking_folder_id,
                                                              path = model_path,
                                                              code_env_name = MLFLOW_CODE_ENV_NAME,
                                                              container_exec_config_name = 'NONE')
# Make this Saved Model version the active one
sm.set_active_version(mlflow_version.version_id)

# Add the algorithm name as a label in the Saved Model version metadata (so you can see whether XGBoost, LogisticRegression, etc.)
active_version_details = sm.get_version_details(mlflow_version.version_id)
model_version_labels = active_version_details.details['userMeta']['labels']
model_version_labels.append({'key': 'model:algorithm', 'value': best_model['algorithm']})
active_version_details.details['userMeta']['labels'] = model_version_labels
active_version_details.save_user_meta()

# Get the output test dataset name
output_test_dataset_name = output_test_dataset_names[0].split('.')[1]

# Set the Saved Model metadata (target name, classes,...)
if prediction_type == "two-class classification":
    mlflow_version.set_core_metadata(target_column_name = col_label, class_labels = list(model_classes), get_features_from_dataset = output_test_dataset_name)
else:
    mlflow_version.set_core_metadata(target_column_name = col_label, get_features_from_dataset = output_test_dataset_name)

# Evaluate the performance of this new version, to populate the performance screens of the Saved Model version in Dataiku
mlflow_version.evaluate(output_test_dataset_name, container_exec_config_name='NONE')

# If selected, deploy the best trained model to a Snowpark ML Model Registry under the 'MODEL_REGISTRY' database
if deploy_to_snowflake_model_registry:
    try:
        model_registry_result = model_registry.create_model_registry(session = session, database_name = snowflake_model_registry)
        registry = model_registry.ModelRegistry(session = session, database_name = snowflake_model_registry)
        snowflake_registry_model_description = "Dataiku Project: " + project.project_key + ", Model: " + model_name
        snowflake_model_name = project.project_key + "_" + model_name
        model_id = registry.log_model(model = best_model["snowml_obj"],
                                      model_name = snowflake_model_name,
                                      model_version = best_model["run_name"],
                                      description = snowflake_registry_model_description,
                                      tags = {"application": "Dataiku",
                                              "dataiku_project_key": project.project_key,
                                              "dataiku_saved_model_id": sm_id})

        for test_metric in best_model["test_metrics"]:
            model_id.set_metric(metric_name = test_metric, metric_value = best_model["test_metrics"][test_metric])
        
        print("Successfully deployed model to Snowflake ML Model Registry: " + snowflake_model_registry)
    except:
        print("Failed to deploy model to Snowflake ML Model Registry")

# Get the current plugin recipe instance name
current_recipe_name = FLOW["currentActivityId"][:-3].replace('_NP', '')

# Get the Dataiku.Recipe object, and add the new trained Saved Model as an output of the recipe (if it isn't already)
recipe = project.get_recipe(current_recipe_name)
recipe_settings = recipe.get_settings()
saved_model_names = get_output_names_for_role('saved_model_name')

if len(saved_model_names)>0:
    prev_saved_model_name = saved_model_names[0].split('.')[-1]
    if prev_saved_model_name != sm_id:
        recipe_settings.replace_output(current_output_ref = prev_saved_model_name, new_output_ref = sm_id)
        recipe_settings.save()
else:
    recipe_settings.add_output(role="saved_model_name",ref=sm_id)
    recipe_settings.save()