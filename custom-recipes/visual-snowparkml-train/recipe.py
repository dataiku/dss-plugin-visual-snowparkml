# SECTION 1 - Package Imports
# Dataiku Imports

import dataiku
from dataiku.customrecipe import get_output_names_for_role
from dataiku.snowpark import DkuSnowpark
from dataikuapi.dss.ml import DSSPredictionMLTaskSettings
from dataiku.core.flow import FLOW

from visualsnowparkml.plugin_config_loading import load_train_config_snowpark_session_and_input_train_snowpark_df

# Other ML Imports
import pandas as pd
import numpy as np
import mlflow
from scipy.stats import uniform, randint, loguniform
from datetime import datetime
from cloudpickle import load, dump
import re

# Snowpark Imports
import snowflake.snowpark.types as T
from snowflake.snowpark.functions import col

# Snowpark-ML Imports
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.model_selection import RandomizedSearchCV
from snowflake.ml.modeling.compose import ColumnTransformer
from snowflake.ml.modeling.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from snowflake.ml.modeling.xgboost import XGBClassifier, XGBRegressor
from snowflake.ml.modeling.lightgbm import LGBMClassifier, LGBMRegressor
from snowflake.ml.modeling.tree import DecisionTreeClassifier, DecisionTreeRegressor
from snowflake.ml.modeling.linear_model import LogisticRegression, Lasso
from snowflake.ml.modeling.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from snowflake.ml.modeling.impute import SimpleImputer
from snowflake.ml.modeling.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, precision_score, r2_score, mean_absolute_error, mean_squared_error
import snowflake.snowpark.functions as F
from snowflake.ml.registry import Registry

# SECTION 2 - Load User-Inputted Config, Inputs, and Outputs
params, session, input_snowpark_df = load_train_config_snowpark_session_and_input_train_snowpark_df()

# Get recipe user-inputted parameters and print to the logs
print("-----------------------------")
print("Recipe Input Params")
attrs = dir(params)
for attr in attrs:
    if not attr.startswith('__'):
        print(str(attr) + ': ' + str(getattr(params, attr)))
print("-----------------------------")

DEFAULT_CROSS_VAL_FOLDS = 3

# Map metric name from dropdown to sklearn-compatible name
metric_to_sklearn_mapping = {
    'ROC AUC': 'roc_auc',
    'Accuracy': 'accuracy',
    'F1 Score': 'f1',
    'Precision': 'precision',
    'Recall': 'recall',
    'R2': 'r2',
    'MAE': 'neg_mean_absolute_error',
    'MSE': 'neg_mean_squared_error'
}

scoring_metric = metric_to_sklearn_mapping[params.model_metric]

# SECTION 3 - Set up MLflow Experiment Tracking
# MLFLOW Variables
MLFLOW_CODE_ENV_NAME = "py_39_snowpark"
MLFLOW_EXPERIMENT_NAME = f"{params.model_name}_exp"

# Get a Dataiku API client and the current project
client = dataiku.api_client()
client._session.verify = False

project = client.get_default_project()

# Set up the Dataiku MLflow extension and setup an experiment pointing to the output models folder
mlflow_extension = project.get_mlflow_extension()
mlflow_handle = project.setup_mlflow(managed_folder=params.model_experiment_tracking_folder)
mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
mlflow_experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

# SECTION 4 - Set up Snowpark
# Get a Snowpark session using the input dataset Snowflake connection
dku_snowpark = DkuSnowpark()

# SECTION 5 - Add a Target Class Weights Column if Two-Class Classification and do Train/Test Split

# Create a dictionary to store all dataset columns as read in by pandas vs. how they're stored on Snowflake.
# E.g. {'feat_1':'"feat_1"', 'feat_2':'"feat_2"', 'FEAT_1':'FEAT_1'}
# We use this lookup dictionary later on to map column names to their actual Snowflake names,
# where many have double quotes surrounding them to prevent Snowflake from auto-capitalizing
if params.disable_class_weights:
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


def sf_col_name(col_name):
    """
    This function will call the lookup dictionary and return the Snowflake column name
    """
    return features_quotes_lookup[col_name]


col_label_sf = sf_col_name(params.col_label)

if params.time_ordering_variable:
    time_ordering_variable_sf = sf_col_name(params.time_ordering_variable)

# Get a list of Target column values if two-class classification
if params.prediction_type == "two-class classification" or params.prediction_type == "multi-class classification":
    col_label_values = list(input_snowpark_df.select(sf_col_name(params.col_label)).distinct().to_pandas()[params.col_label])
else:
    col_label_values = None


def convert_snowpark_df_col_dtype(snowpark_df, col):
    """
    This function will retrieve the Snowflake data type from the corresponding pandas data type
    """
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


def add_sample_weights_col_to_snowpark_df(snowpark_df, col):
    """
    This function adds sample weights (inverse proportion of target class column) to the Snowpark df
    """
    sf_col = sf_col_name(col)
    y_collect = snowpark_df.select(sf_col).groupBy(sf_col).count().collect()
    unique_y = [x[col] for x in y_collect]
    total_y = sum([x["COUNT"] for x in y_collect])
    unique_y_count = len(y_collect)
    bin_count = [x["COUNT"] for x in y_collect]

    class_weights = {i: ii for i, ii in zip(unique_y, total_y / (unique_y_count * np.array(bin_count)))}

    res = []
    for key, val in class_weights.items():
        res.append([key, val])

    col_label_dtype = convert_snowpark_df_col_dtype(snowpark_df, sf_col)

    schema = T.StructType([T.StructField(sf_col, col_label_dtype), T.StructField("SAMPLE_WEIGHTS", T.DoubleType())])
    df_to_join = session.create_dataframe(res, schema)

    snowpark_df = snowpark_df.join(df_to_join, [sf_col], 'left')

    return snowpark_df

# Get an example of the input dataset to use as an input example for the model
input_example = input_snowpark_df.limit(10)

# Add sample weights column if two-class classification
if (params.prediction_type == "two-class classification" or params.prediction_type == "multi-class classification") and not params.disable_class_weights:
    input_snowpark_df = add_sample_weights_col_to_snowpark_df(input_snowpark_df, params.col_label)

# If chosen by the user, split train/test sets based on the time ordering column
if params.time_ordering:
    time_ordering_variable_unix = f"{time_ordering_variable_sf}_UNIX"
    input_snowpark_df = input_snowpark_df.withColumn(time_ordering_variable_unix, F.unix_timestamp(input_snowpark_df[time_ordering_variable_sf]))

    split_percentile_value = input_snowpark_df.approx_quantile(time_ordering_variable_unix, [params.train_ratio])[0]

    train_snowpark_df = input_snowpark_df.filter(col(time_ordering_variable_unix) < split_percentile_value)
    test_snowpark_df = input_snowpark_df.filter(col(time_ordering_variable_unix) >= split_percentile_value)

    train_snowpark_df = train_snowpark_df.drop(time_ordering_variable_unix)
    test_snowpark_df = test_snowpark_df.drop(time_ordering_variable_unix)

    print(f"train set nrecords: {train_snowpark_df.count()}")
    print(f"test set nrecords: {test_snowpark_df.count()}")

# Regular train/test split
else:
    test_ratio = 1 - params.train_ratio
    train_snowpark_df, test_snowpark_df = input_snowpark_df.random_split(weights=[params.train_ratio, test_ratio], seed=params.random_seed)

# SECTION 6 - Write Train/Test Datasets to Output Tables
dku_snowpark.write_with_schema(params.output_train_dataset, train_snowpark_df)
dku_snowpark.write_with_schema(params.output_test_dataset, test_snowpark_df)

# SECTION 7 - Create a feature preprocessing Pipeline for all selected input columns and the encoding/rescaling + imputation methods chosen
# List of numeric and categorical dtypes in order to auto-select a reasonable encoding/rescaling and missingness imputation method based on the column
numeric_dtypes_list = ['number', 'decimal', 'numeric', 'int', 'integer', 'bigint', 'smallint', 'tinyint', 'byteint',
                       'float', 'float4', 'float8', 'double', 'double precision', 'real']

categorical_dtypes_list = ['varchar', 'char', 'character', 'string', 'text', 'binary', 'varbinary', 'boolean', 'date',
                           'datetime', 'time', 'timestamp', 'timestamp_ltz', 'timestamp_ntz', 'timestamp_tz',
                           'variant', 'object', 'array', 'geography', 'geometry']

# Create list of input features and the encoding/rescaling and missingness imputation method chosen
included_features_handling_list = []

for feature_column in params.inputDatasetColumns:
    col_name = feature_column['name']
    col_name_sf = sf_col_name(col_name)
    feature_column['name'] = col_name_sf

    if col_name in params.selectedInputColumns:
        if params.selectedInputColumns[col_name]:
            feature_column['include'] = True
            if col_name in params.selectedOption1:
                feature_column["encoding_rescaling"] = params.selectedOption1[col_name]
            elif feature_column['type'] in numeric_dtypes_list:
                feature_column["encoding_rescaling"] = 'Standard rescaling'
            else:
                feature_column["encoding_rescaling"] = 'Dummy encoding'

            if col_name in params.selectedOption2:
                feature_column["missingness_impute"] = params.selectedOption2[col_name]
                if params.selectedOption2[col_name] == 'Constant':
                    if col_name in params.selectedConstantImpute:
                        feature_column["constant_impute"] = params.selectedConstantImpute[col_name]

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
    if feature_name.startswith('"') and feature_name.endswith('"'):
        transformer_name = f"{feature_name[1:-1]}_tform"
    else:
        transformer_name = f"{feature_name}_tform"

    feature_transformers = []

    if feature["missingness_impute"] == "Average":
        feature_transformers.append(('imputer', SimpleImputer(strategy='mean')))
    if feature["missingness_impute"] == "Median":
        feature_transformers.append(('imputer', SimpleImputer(strategy='median')))
    if feature["missingness_impute"] == "Constant":
        if "constant_impute" in feature:
            feature_transformers.append(('imputer', SimpleImputer(strategy='constant', fill_value=feature["constant_impute"], missing_values=pd.NA)))
        else:
            feature_transformers.append(('imputer', SimpleImputer(strategy='constant', missing_values=pd.NA)))
    if feature["missingness_impute"] == "Most frequent value":
        feature_transformers.append(('imputer', SimpleImputer(strategy='most_frequent', missing_values=pd.NA)))
    if feature["encoding_rescaling"] == "Standard rescaling":
        feature_transformers.append(('enc', StandardScaler()))
    if feature["encoding_rescaling"] == "Min-max rescaling":
        feature_transformers.append(('enc', MinMaxScaler()))
    if feature["encoding_rescaling"] == "Dummy encoding":
        feature_transformers.append(('enc', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                                          max_categories=10)))
    if feature["encoding_rescaling"] == "Ordinal encoding":
        feature_transformers.append(('enc', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                           unknown_value=-1,
                                                           encoded_missing_value=-1)))
    col_transformer_list.append((transformer_name, Pipeline(feature_transformers), [feature_name]))

preprocessor = ColumnTransformer(transformers=col_transformer_list)

# SECTION 9 - Initialize algorithms selected and hyperparameter spaces for the RandomSearch
algorithms = []

if params.prediction_type == "two-class classification" or params.prediction_type == "multi-class classification":
    if params.logistic_regression:
        algorithms.append({'algorithm': 'logistic_regression',
                           'sklearn_obj': LogisticRegression(),
                           'gs_params': {'clf__C': loguniform(params.logistic_regression_c_min, params.logistic_regression_c_max),
                                         'clf__multi_class': ['auto']}})

    if params.random_forest_classification:
        algorithms.append({'algorithm': 'random_forest_classification',
                           'sklearn_obj': RandomForestClassifier(),
                           'gs_params': {'clf__n_estimators': randint(params.random_forest_classification_n_estimators_min, params.random_forest_classification_n_estimators_max),
                                         'clf__max_depth': randint(params.random_forest_classification_max_depth_min, params.random_forest_classification_max_depth_max),
                                         'clf__min_samples_leaf': randint(params.random_forest_classification_min_samples_leaf_min, params.random_forest_classification_min_samples_leaf_max)}})

    if params.xgb_classification:
        algorithms.append({'algorithm': 'xgb_classification',
                           'sklearn_obj': XGBClassifier(),
                           'gs_params': {'clf__n_estimators': randint(params.xgb_classification_n_estimators_min, params.xgb_classification_n_estimators_max),
                                         'clf__max_depth': randint(params.xgb_classification_max_depth_min, params.xgb_classification_max_depth_max),
                                         'clf__min_child_weight': uniform(params.xgb_classification_min_child_weight_min, params.xgb_classification_min_child_weight_max),
                                         'clf__learning_rate': loguniform(params.xgb_classification_learning_rate_min, params.xgb_classification_learning_rate_max)}})

    if params.lgbm_classification:
        algorithms.append({'algorithm': 'lgbm_classification',
                           'sklearn_obj': LGBMClassifier(),
                           'gs_params': {'clf__n_estimators': randint(params.lgbm_classification_n_estimators_min, params.lgbm_classification_n_estimators_max),
                                         'clf__max_depth': randint(params.lgbm_classification_max_depth_min, params.lgbm_classification_max_depth_max),
                                         'clf__min_child_weight': uniform(params.lgbm_classification_min_child_weight_min, params.lgbm_classification_min_child_weight_max),
                                         'clf__learning_rate': loguniform(params.lgbm_classification_learning_rate_min, params.lgbm_classification_learning_rate_max)}})

    if params.gb_classification:
        algorithms.append({'algorithm': 'gb_classification',
                           'sklearn_obj': GradientBoostingClassifier(),
                           'gs_params': {'clf__n_estimators': randint(params.gb_classification_n_estimators_min, params.gb_classification_n_estimators_max),
                                         'clf__max_depth': randint(params.gb_classification_max_depth_min, params.gb_classification_max_depth_max),
                                         'clf__min_samples_leaf': randint(params.gb_classification_min_samples_leaf_min, params.gb_classification_min_samples_leaf_max),
                                         'clf__learning_rate': loguniform(params.gb_classification_learning_rate_min, params.gb_classification_learning_rate_max)}})

    if params.decision_tree_classification:
        algorithms.append({'algorithm': 'decision_tree_classification',
                           'sklearn_obj': DecisionTreeClassifier(),
                           'gs_params': {'clf__max_depth': randint(params.decision_tree_classification_max_depth_min, params.decision_tree_classification_max_depth_max),
                                         'clf__min_samples_leaf': randint(params.decision_tree_classification_min_samples_leaf_min, params.decision_tree_classification_min_samples_leaf_max)}})

else:
    if params.lasso_regression:
        algorithms.append({'algorithm': 'lasso_regression',
                           'sklearn_obj': Lasso(),
                           'gs_params': {'clf__alpha': loguniform(params.lasso_regression_alpha_min, params.lasso_regression_alpha_max)}})

    if params.random_forest_regression:
        algorithms.append({'algorithm': 'random_forest_regression',
                           'sklearn_obj': RandomForestRegressor(),
                           'gs_params': {'clf__n_estimators': randint(params.random_forest_regression_n_estimators_min, params.random_forest_regression_n_estimators_max),
                                         'clf__max_depth': randint(params.random_forest_regression_max_depth_min, params.random_forest_regression_max_depth_max),
                                         'clf__min_samples_leaf': randint(params.random_forest_regression_min_samples_leaf_min, params.random_forest_regression_min_samples_leaf_max)}})

    if params.xgb_regression:
        algorithms.append({'algorithm': 'xgb_regression',
                           'sklearn_obj': XGBRegressor(),
                           'gs_params': {'clf__n_estimators': randint(params.xgb_regression_n_estimators_min, params.xgb_regression_n_estimators_max),
                                         'clf__max_depth': randint(params.xgb_regression_max_depth_min, params.xgb_regression_max_depth_max),
                                         'clf__min_child_weight': uniform(params.xgb_regression_min_child_weight_min, params.xgb_regression_min_child_weight_max),
                                         'clf__learning_rate': loguniform(params.xgb_regression_learning_rate_min, params.xgb_regression_learning_rate_max)}})
    if params.lgbm_regression:
        algorithms.append({'algorithm': 'lgbm_regression',
                           'sklearn_obj': LGBMRegressor(),
                           'gs_params': {'clf__n_estimators': randint(params.lgbm_regression_n_estimators_min, params.lgbm_regression_n_estimators_max),
                                         'clf__max_depth': randint(params.lgbm_regression_max_depth_min, params.lgbm_regression_max_depth_max),
                                         'clf__min_child_weight': uniform(params.lgbm_regression_min_child_weight_min, params.lgbm_regression_min_child_weight_max),
                                         'clf__learning_rate': loguniform(params.lgbm_regression_learning_rate_min, params.lgbm_regression_learning_rate_max)}})

    if params.gb_regression:
        algorithms.append({'algorithm': 'gb_regression',
                           'sklearn_obj': GradientBoostingRegressor(),
                           'gs_params': {'clf__n_estimators': randint(params.gb_regression_n_estimators_min, params.gb_regression_n_estimators_max),
                                         'clf__max_depth': randint(params.gb_regression_max_depth_min, params.gb_regression_max_depth_max),
                                         'clf__min_samples_leaf': randint(params.gb_regression_min_samples_leaf_min, params.gb_regression_min_samples_leaf_max),
                                         'clf__learning_rate': loguniform(params.gb_regression_learning_rate_min, params.gb_regression_learning_rate_max)}})

    if params.decision_tree_regression:
        algorithms.append({'algorithm': 'decision_tree_regression',
                           'sklearn_obj': DecisionTreeRegressor(),
                           'gs_params': {'clf__max_depth': randint(params.decision_tree_regression_max_depth_min, params.decision_tree_regression_max_depth_max),
                                         'clf__min_samples_leaf': randint(params.decision_tree_regression_min_samples_leaf_min, params.decision_tree_regression_min_samples_leaf_max)}})

# SECTION 10 - Train all models, do RandomSearch and hyperparameter tuning

# These ML algorithm wrappers will allow Dataiku's MLflow imported model to properly evaluate the model on another dataset
# This solves the pandas column name (e.g. 'col_1' vs. Snowflake column name '"col_1"' issue)


class SnowparkMLClassifierWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = load(open(context.artifacts["grid_pipe_sklearn"], 'rb'))
        self.features_quotes_lookup = load(open(context.artifacts["features_quotes_lookup"], 'rb'))

    def predict(self, context, input_df):
        input_df_copy = input_df.copy()
        input_df_copy.columns = [self.features_quotes_lookup[col] for col in input_df_copy.columns]
        return self.model.predict_proba(input_df_copy)


class SnowparkMLRegressorWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = load(open(context.artifacts["grid_pipe_sklearn"], 'rb'))
        self.features_quotes_lookup = load(open(context.artifacts["features_quotes_lookup"], 'rb'))

    def predict(self, context, input_df):
        input_df_copy = input_df.copy()
        input_df_copy.columns = [self.features_quotes_lookup[col] for col in input_df_copy.columns]
        return self.model.predict(input_df_copy)


def train_model(algo, prepr, score_met, col_lab, samp_weight_col, feat_names, train_sp_df, num_iter):
    """
    This function runs a RandomizedSearchCV hyperparameter tuning process, passing in the preprocessing Pipeline and and algorithm
    Returns the trained RandomizedSearchCV object and algorithm name
    """
    print(f"Training model... {algo['algorithm']}")
    pipe = Pipeline(steps=[
                        ('preprocessor', prepr),
                        ('clf', algo['sklearn_obj'])
                    ])

    if params.prediction_type == "two-class classification" or params.prediction_type == "multi-class classification":
        rs_clf = RandomizedSearchCV(estimator=pipe,
                                    param_distributions=algo['gs_params'],
                                    n_iter=num_iter,
                                    cv=DEFAULT_CROSS_VAL_FOLDS,
                                    scoring=score_met,
                                    n_jobs=-1,
                                    verbose=1,
                                    input_cols=feat_names,
                                    label_cols=col_lab,
                                    output_cols="PREDICTION",
                                    sample_weight_col=samp_weight_col
                                    )
    else:
        rs_clf = RandomizedSearchCV(estimator=pipe,
                                    param_distributions=algo['gs_params'],
                                    n_iter=num_iter,
                                    cv=DEFAULT_CROSS_VAL_FOLDS,
                                    scoring=score_met,
                                    n_jobs=-1,
                                    verbose=1,
                                    input_cols=feat_names,
                                    label_cols=col_lab,
                                    output_cols="PREDICTION"
                                    )

    rs_clf.fit(train_sp_df)

    return {'algorithm': algo['algorithm'], 'sklearn_obj': rs_clf}


# Tune hyperparameters for all models chosen - store the trained RandomizedSearchCV objects and algorithm names in a list
trained_models = []
for alg in algorithms:
    trained_model = train_model(alg, preprocessor, scoring_metric, col_label_sf, sample_weight_col, included_feature_names, train_snowpark_df, params.n_iter)
    trained_models.append(trained_model)

# SECTION 11 - Log all trained model hyperparameters and performance metrics to MLflow

# Loop through all trained models, log all hyperparameter tuning cross validation metrics, calculate holdout test set metrics on the best
# model from each algorithm, then add these best models to a final_models list
final_models = []

for model in trained_models:
    rs_clf = model['sklearn_obj']
    model_algo = model['algorithm']

    grid_pipe_sklearn = rs_clf.to_sklearn()

    grid_pipe_sklearn_cv_results = pd.DataFrame(grid_pipe_sklearn.cv_results_)
    grid_pipe_sklearn_cv_results['algorithm'] = model_algo

    now = datetime.now().strftime("%Y_%m_%d_%H_%M")

    for index, row in grid_pipe_sklearn_cv_results.iterrows():
        cv_num = index + 1
        run_name = f"{params.model_name}_{now}_cv_{cv_num}"
        run = mlflow.start_run(run_name=run_name)
        mlflow.log_param("algorithm", row['algorithm'])

        mlflow.log_metric("mean_fit_time", row["mean_fit_time"])
        score_name = f"mean_test_{scoring_metric}"
        mlflow.log_metric(score_name, row["mean_test_score"])
        mlflow.log_metric("rank_test_score", int(row["rank_test_score"]))

        for column in row.index:
            if "param_" in column:
                param_name = column.replace("param_clf__", "")
                mlflow.log_param(param_name, row[column])

        mlflow.end_run()

    run_name = f"{params.model_name}_{now}_final_model"

    run = mlflow.start_run(run_name=run_name)

    model_best_params = grid_pipe_sklearn.best_params_

    whole_dataset_refit_time = grid_pipe_sklearn.refit_time_
    mlflow.log_metric("whole_dataset_refit_time", whole_dataset_refit_time)

    for param in model_best_params.keys():
        if "clf" in param:
            param_name = param.replace("clf__", "")
            mlflow.log_param(param_name, model_best_params[param])
    mlflow.log_param("algorithm", model_algo)

    test_predictions_df = rs_clf.predict(test_snowpark_df)
    # Sometimes, the PREDICTION column will be a string. Need to change it to be consistent with the target column
    col_label_dtype = convert_snowpark_df_col_dtype(test_predictions_df, col_label_sf)
    test_predictions_df = test_predictions_df.withColumn('"PREDICTION"', test_predictions_df['"PREDICTION"'].cast(col_label_dtype))

    test_metrics = {}

    if params.prediction_type == "two-class classification":
        model_classes = grid_pipe_sklearn.classes_

        test_prediction_probas_df = rs_clf.predict_proba(test_snowpark_df)

        target_col_value_cols = [col for col in test_prediction_probas_df.columns if "PREDICT_PROBA" in col]

        test_f1 = f1_score(df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"', pos_label=col_label_values[0])
        mlflow.log_metric("test_f1_score", test_f1)
        test_roc_auc = roc_auc_score(df=test_prediction_probas_df, y_true_col_names=col_label_sf, y_score_col_names=test_prediction_probas_df.columns[-1])
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        test_accuracy = accuracy_score(df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"')
        mlflow.log_metric("test_accuracy", test_accuracy)
        test_recall = recall_score(df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"', pos_label=col_label_values[0])
        mlflow.log_metric("test_recall", test_recall)
        test_precision = precision_score(df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"', pos_label=col_label_values[0])
        mlflow.log_metric("test_precision", test_precision)

        test_metrics["test_f1"] = test_f1
        test_metrics["test_roc_auc"] = test_roc_auc
        test_metrics["test_accuracy"] = test_accuracy
        test_metrics["test_recall"] = test_recall
        test_metrics["test_precision"] = test_precision

        print(f"F1 Score: {test_f1}")
        print(f"ROC AUC Score: {test_roc_auc}")
        print(f"Accuracy Score: {test_accuracy}")
        print(f"Recall Score: {test_recall}")
        print(f"Precision Score: {test_precision}")

    elif params.prediction_type == "multi-class classification":
        model_classes = grid_pipe_sklearn.classes_

        test_prediction_probas_df = rs_clf.predict_proba(test_snowpark_df)

        target_col_value_cols = [col for col in test_prediction_probas_df.columns if "PREDICT_PROBA" in col]

        test_f1 = f1_score(df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"', average="macro")
        mlflow.log_metric("test_f1_score", test_f1)
        test_roc_auc = roc_auc_score(df=test_prediction_probas_df, y_true_col_names=col_label_sf, y_score_col_names=target_col_value_cols, labels=model_classes, average="macro", multi_class="ovo")
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        test_accuracy = accuracy_score(df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"')
        mlflow.log_metric("test_accuracy", test_accuracy)
        test_recall = recall_score(df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"', average="macro")
        mlflow.log_metric("test_recall", test_recall)
        test_precision = precision_score(df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"', average="macro")
        mlflow.log_metric("test_precision", test_precision)

        test_metrics["test_f1"] = test_f1
        test_metrics["test_roc_auc"] = test_roc_auc
        test_metrics["test_accuracy"] = test_accuracy
        test_metrics["test_recall"] = test_recall
        test_metrics["test_precision"] = test_precision

        print(f"F1 Score: {test_f1}")
        print(f"ROC AUC Score: {test_roc_auc}")
        print(f"Accuracy Score: {test_accuracy}")
        print(f"Recall Score: {test_recall}")
        print(f"Precision Score: {test_precision}")

    else:
        test_r2 = r2_score(df=test_predictions_df, y_true_col_name=col_label_sf, y_pred_col_name='"PREDICTION"')
        mlflow.log_metric("test_r2_score", test_r2)
        test_mae = mean_absolute_error(df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"')
        mlflow.log_metric("test_mae_score", test_mae)
        test_mse = mean_squared_error(df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"')
        mlflow.log_metric("test_mse_score", test_mse)
        test_rmse = mean_squared_error(df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"', squared=False)
        mlflow.log_metric("test_rmse_score", test_rmse)

        test_metrics["test_r2"] = test_r2
        test_metrics["test_mae"] = test_mae
        test_metrics["test_mse"] = test_mse
        test_metrics["test_rmse"] = test_rmse

        print(f"R2 Score: {test_r2}")
        print(f"Mean Absolute Error: {test_mae}")
        print(f"Mean Squared Error: {test_mse}")
        print(f"Root Mean Squared Error: {test_rmse}")

    best_score = grid_pipe_sklearn.best_score_

    artifacts = {
        "grid_pipe_sklearn": "grid_pipe_sklearn.pkl",
        "features_quotes_lookup": "features_quotes_lookup.pkl"
    }

    dump(grid_pipe_sklearn, open(artifacts.get("grid_pipe_sklearn"), 'wb'))
    dump(features_quotes_lookup, open(artifacts.get("features_quotes_lookup"), 'wb'))

    if params.prediction_type == "two-class classification" or params.prediction_type == "multi-class classification":
        logged_model = mlflow.pyfunc.log_model(artifact_path="model",
                                               python_model=SnowparkMLClassifierWrapper(),
                                               artifacts=artifacts)
    else:
        logged_model = mlflow.pyfunc.log_model(artifact_path="model",
                                               python_model=SnowparkMLRegressorWrapper(),
                                               artifacts=artifacts)

    mlflow.end_run()
    best_run_id = run.info.run_id
    final_models.append({'algorithm': model_algo,
                         'sklearn_obj': grid_pipe_sklearn,
                         'snowml_obj': rs_clf,
                         'mlflow_best_run_id': best_run_id,
                         'run_name': run_name,
                         'best_score': best_score,
                         'test_metrics': test_metrics})

# SECTION 12 - Pull the best model, import it into a SavedModel green diamond (it will create a new one if doesn't exist), and evaluate on the hold out Test dataset
# Get the final best model (of the best models of each algorithm type) based on the performance metric chosen
if scoring_metric in ['roc_auc', 'accuracy', 'f1', 'precision', 'recall', 'r2']:
    best_model = max(final_models, key=lambda x: x['best_score'])
else:
    best_model = min(final_models, key=lambda x: x['best_score'])

best_model_run_id = best_model['mlflow_best_run_id']

# If two-class classification, set the Dataiku MLflow imported model run inference info
if params.prediction_type == "two-class classification":
    model_classes = best_model['sklearn_obj'].classes_
    # Deal with nasty numpy data types that are not json serializable
    if 'int' in str(type(model_classes[0])):
        model_classes = [int(model_class) for model_class in model_classes]
    if 'float' in str(type(model_classes[0])):
        model_classes = [np.float64(model_class) for model_class in model_classes]
    mlflow_extension.set_run_inference_info(run_id=best_model_run_id,
                                            prediction_type='BINARY_CLASSIFICATION',
                                            classes=list(model_classes),
                                            code_env_name=MLFLOW_CODE_ENV_NAME)

if params.prediction_type == "multi-class classification":
    model_classes = best_model['sklearn_obj'].classes_
    # Deal with nasty numpy data types that are not json serializable
    if 'int' in str(type(model_classes[0])):
        model_classes = [int(model_class) for model_class in model_classes]
    if 'float' in str(type(model_classes[0])):
        model_classes = [np.float64(model_class) for model_class in model_classes]
    mlflow_extension.set_run_inference_info(run_id=best_model_run_id,
                                            prediction_type='MULTICLASS',
                                            classes=list(model_classes),
                                            code_env_name=MLFLOW_CODE_ENV_NAME)

# Get the managed folder subpath for the best trained model
model_artifact_first_directory = re.search(r'.*/(.+$)', mlflow_experiment.artifact_location).group(1)
model_path = f"{model_artifact_first_directory}/{best_model_run_id}/artifacts/model"

# If the Saved Model already exists in the flow (matching the user-inputted model name in the plugin), get it
sm_id = None
for sm in project.list_saved_models():
    if sm["name"] != params.model_name:
        continue
    else:
        sm_id = sm["id"]
        print(f"Found Saved Model {sm['name']} with id {sm['id']}")
        break

if sm_id:
    sm = project.get_saved_model(sm_id)

# If the Saved Model does not exist, create a new placeholder
else:
    if params.prediction_type == "two-class classification":
        sm = project.create_mlflow_pyfunc_model(name=params.model_name,
                                                prediction_type=DSSPredictionMLTaskSettings.PredictionTypes.BINARY)
    elif params.prediction_type == "multi-class classification":
        sm = project.create_mlflow_pyfunc_model(name=params.model_name,
                                                prediction_type=DSSPredictionMLTaskSettings.PredictionTypes.MULTICLASS)
    else:
        sm = project.create_mlflow_pyfunc_model(name=params.model_name,
                                                prediction_type=DSSPredictionMLTaskSettings.PredictionTypes.REGRESSION)
    sm_id = sm.id
    print(f"Saved Model not found, created new one with id {sm_id}")

# Import the final trained model into the Dataiku Saved Model (Green Diamond)
mlflow_version = sm.import_mlflow_version_from_managed_folder(version_id=best_model["run_name"],
                                                              managed_folder=params.model_experiment_tracking_folder_id,
                                                              path=model_path,
                                                              code_env_name=MLFLOW_CODE_ENV_NAME,
                                                              container_exec_config_name='NONE')
# Make this Saved Model version the active one
sm.set_active_version(mlflow_version.version_id)

# Add the algorithm name as a label in the Saved Model version metadata (so you can see whether XGBoost, LogisticRegression, etc.)
active_version_details = sm.get_version_details(mlflow_version.version_id)
model_version_labels = active_version_details.details['userMeta']['labels']
model_version_labels.append({'key': 'model:algorithm', 'value': best_model['algorithm']})
active_version_details.details['userMeta']['labels'] = model_version_labels
active_version_details.save_user_meta()

# Get the output test dataset name
output_test_dataset_name = params.output_test_dataset.name.split('.')[1]

# Set the Saved Model metadata (target name, classes,...)
if params.prediction_type == "two-class classification":
    mlflow_version.set_core_metadata(target_column_name=params.col_label, class_labels=list(model_classes), get_features_from_dataset=output_test_dataset_name)
elif params.prediction_type == "multi-class classification":
    mlflow_version.set_core_metadata(target_column_name=params.col_label, class_labels=list(model_classes), get_features_from_dataset=output_test_dataset_name)
else:
    mlflow_version.set_core_metadata(target_column_name=params.col_label, get_features_from_dataset=output_test_dataset_name)

# Evaluate the performance of this new version, to populate the performance screens of the Saved Model version in Dataiku
mlflow_version.evaluate(output_test_dataset_name, container_exec_config_name='NONE')

# If selected, deploy the best trained model to a Snowpark ML Model Registry in the current working database and schema
if params.deploy_to_snowflake_model_registry:
    try:
        registry = Registry(session=session)
        snowflake_registry_model_description = f"Dataiku Project: {project.project_key}, Model: {params.model_name}"
        snowflake_model_name = f"{project.project_key}_{params.model_name}"
        model_ver = registry.log_model(model=best_model["snowml_obj"],
                                       model_name=snowflake_model_name,
                                       version_name=best_model["run_name"],
                                       sample_input_data=input_example,
                                       comment=snowflake_registry_model_description,
                                       options={"relax_version": False})

        for test_metric in best_model["test_metrics"]:
            model_ver.set_metric(metric_name=test_metric, value=best_model["test_metrics"][test_metric])

        # Need to set tags at the parent model level
        parent_model = registry.get_model(snowflake_model_name)
        # Update the defuault model version to the new version
        parent_model.default = best_model["run_name"]
        # Need to create the tag object in Snowflake if it doesn't exist
        session.sql("CREATE TAG IF NOT EXISTS APPLICATION;").show()
        session.sql("CREATE TAG IF NOT EXISTS DATAIKU_PROJECT_KEY;").show()
        session.sql("CREATE TAG IF NOT EXISTS DATAIKU_SAVED_MODEL_ID;").show()

        parent_model.set_tag("application", "Dataiku")
        parent_model.set_tag("dataiku_project_key", project.project_key)
        parent_model.set_tag("dataiku_saved_model_id", sm_id)

        print("Successfully deployed model to Snowflake ML Model Registry")
    except:
        print("Failed to deploy model to Snowflake ML Model Registry")

# Get the current plugin recipe instance name
current_recipe_name = FLOW["currentActivityId"][:-3].replace('_NP', '')

# Get the Dataiku.Recipe object, and add the new trained Saved Model as an output of the recipe (if it isn't already)
recipe = project.get_recipe(current_recipe_name)
recipe_settings = recipe.get_settings()
saved_model_names = get_output_names_for_role('saved_model_name')

if len(saved_model_names) > 0:
    prev_saved_model_name = saved_model_names[0].split('.')[-1]
    if prev_saved_model_name != sm_id:
        recipe_settings.replace_output(current_output_ref=prev_saved_model_name, new_output_ref=sm_id)
        recipe_settings.save()
else:
    recipe_settings.add_output(role="saved_model_name", ref=sm_id)
    recipe_settings.save()
