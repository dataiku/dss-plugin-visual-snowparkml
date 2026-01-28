# Package Imports - Dataiku
import dataiku
from dataiku.customrecipe import get_output_names_for_role
from dataiku.snowpark import DkuSnowpark
from dataikuapi.dss.ml import DSSPredictionMLTaskSettings
from dataiku.core.flow import FLOW
from visualsnowparkml.plugin_config_loading import load_train_config_snowpark_session_and_input_train_snowpark_df

# Package Imports - ML Libraries
import pandas as pd
import numpy as np
import mlflow
from scipy.stats import uniform, randint, loguniform
from datetime import datetime
from cloudpickle import load, dump
import re
from importlib import metadata
import logging

# Package Imports - Snowpark
import snowflake.snowpark.types as T
from snowflake.snowpark.functions import col, count, first_value
from snowflake.snowpark.window import Window

# Package Imports - Standard ML (for Compute Pool / ML Jobs)
import sklearn.pipeline as sk_pipeline
import sklearn.model_selection as sk_ms
import sklearn.compose as sk_compose
import sklearn.preprocessing as sk_prep
import sklearn.impute as sk_impute
import sklearn.ensemble as sk_ensemble
import sklearn.tree as sk_tree
import sklearn.linear_model as sk_linear
import sklearn.metrics as sk_metrics
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# Package Imports - Snowpark ML (for Warehouse)
import snowflake.ml.modeling.pipeline as snowpark_pipeline
import snowflake.ml.modeling.model_selection as snowpark_ms
import snowflake.ml.modeling.compose as snowpark_compose
import snowflake.ml.modeling.preprocessing as snowpark_prep
import snowflake.ml.modeling.impute as snowpark_impute
import snowflake.ml.modeling.ensemble as snowpark_ensemble
import snowflake.ml.modeling.xgboost as snowpark_xgboost
import snowflake.ml.modeling.lightgbm as snowpark_lightgbm
import snowflake.ml.modeling.tree as snowpark_tree
import snowflake.ml.modeling.linear_model as snowpark_linear
import snowflake.ml.modeling.metrics as snowpark_metrics
from snowflake.ml.jobs import remote
from snowflake.ml.registry import Registry
from snowflake.ml.model import model_signature
from snowflake.ml.experiment import ExperimentTracking

logger = logging.getLogger("visualsnowflakemlplugin")
logging.basicConfig(
    level=logging.INFO,
    format='Visual Snowflake ML plugin %(levelname)s - %(message)s'
)

# Configuration and Constants
params, session, input_snowpark_df = load_train_config_snowpark_session_and_input_train_snowpark_df()
is_snowpark_backend = (params.compute_backend == 'warehouse')

# Log recipe parameters
logger.info("Recipe Input Params")
for attr in dir(params):
    if not attr.startswith('__'):
        logger.info(f"{attr}: {getattr(params, attr)}")

METRIC_TO_SKLEARN = {
    'ROC AUC': 'roc_auc',
    'Accuracy': 'accuracy',
    'F1 Score': 'f1',
    'Precision': 'precision',
    'Recall': 'recall',
    'R2': 'r2',
    'MAE': 'neg_mean_absolute_error',
    'MSE': 'neg_mean_squared_error'
}

scoring_metric = METRIC_TO_SKLEARN[params.model_metric]

# MLflow Configuration
MLFLOW_CODE_ENV_NAME = "py_310_snowpark"
MLFLOW_EXPERIMENT_NAME = f"{params.model_name}_exp"
SNOWFLAKE_PYPI_EAI_NAME = "PYPI_EAI"

# Build list of ML package versions for Snowpark Container Services
REQUIRED_PACKAGES = ["scikit-learn", "xgboost", "lightgbm", "numpy", "pandas"]
snowpark_packages_versions = [
    f"{dist.name}=={dist.version}"
    for dist in metadata.distributions()
    if dist.name in REQUIRED_PACKAGES
]

# Initialize Dataiku client and project
client = dataiku.api_client()
client._session.verify = False
project = client.get_default_project()

# Set up MLflow experiment tracking
mlflow_extension = project.get_mlflow_extension()
mlflow_handle = project.setup_mlflow(
    managed_folder=params.model_experiment_tracking_folder)
mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
mlflow_experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

# Set up Snowflake experiment tracking
snowflake_experiment = ExperimentTracking(session=session)
snowflake_experiment.set_experiment(MLFLOW_EXPERIMENT_NAME)
logger.info(f"Initialized Snowflake Experiment: {MLFLOW_EXPERIMENT_NAME}")

# Initialize Snowpark
dku_snowpark = DkuSnowpark()

# Column Name Mapping
# Maps pandas column names to Snowflake column names (with quotes for case-sensitive columns)
is_classification = params.prediction_type in [
    "two-class classification", "multi-class classification"]
use_sample_weights = is_classification and params.enable_class_weights

if use_sample_weights:
    features_quotes_lookup = {'SAMPLE_WEIGHTS': 'SAMPLE_WEIGHTS'}
    sample_weight_col = 'SAMPLE_WEIGHTS'
else:
    features_quotes_lookup = {}
    sample_weight_col = None

for col_name in input_snowpark_df.columns:
    if col_name.startswith('"') and col_name.endswith('"'):
        features_quotes_lookup[col_name.replace('"', '')] = col_name
    else:
        features_quotes_lookup[col_name] = col_name


def sf_col_name(col_name):
    """Return the Snowflake column name (with quotes if needed)."""
    return features_quotes_lookup[col_name]


inverse_features_quotes_lookup = {
    v: k for k, v in features_quotes_lookup.items()}


def inverse_sf_col_name(col_name):
    """Return the column name without surrounding double quotes."""
    return inverse_features_quotes_lookup[col_name]


col_label_sf = sf_col_name(params.col_label)

if params.time_ordering_variable:
    time_ordering_variable_sf = sf_col_name(params.time_ordering_variable)

# Get distinct target column values for classification tasks
col_label_values = None
if is_classification:
    col_label_values = list(
        input_snowpark_df.select(col_label_sf).distinct().to_pandas()[
            params.col_label]
    )


SNOWFLAKE_DTYPE_MAPPINGS = {
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


def convert_snowpark_df_col_dtype(snowpark_df, col_name):
    """Get the Snowflake type for a column based on its current dtype."""
    for name, dtype in snowpark_df.dtypes:
        if name == col_name:
            return SNOWFLAKE_DTYPE_MAPPINGS[dtype]
    return None


def add_sample_weights_col_to_snowpark_df(snowpark_df, col):
    """Add sample weights (inverse proportion of target class) to the dataframe."""
    sf_col = sf_col_name(col)
    class_counts = snowpark_df.select(sf_col).groupBy(sf_col).count().collect()

    unique_classes = [row[col] for row in class_counts]
    counts = [row["COUNT"] for row in class_counts]
    total_samples = sum(counts)
    num_classes = len(class_counts)

    # Calculate inverse class weights
    weights_data = [
        [cls, total_samples / (num_classes * count)]
        for cls, count in zip(unique_classes, counts)
    ]

    col_dtype = convert_snowpark_df_col_dtype(snowpark_df, sf_col)
    schema = T.StructType([
        T.StructField(sf_col, col_dtype),
        T.StructField("SAMPLE_WEIGHTS", T.DoubleType())
    ])

    weights_df = session.create_dataframe(weights_data, schema)
    return snowpark_df.join(weights_df, [sf_col], 'left')


# Create input example for model signature
input_example = input_snowpark_df.sample(n=1000)
first_column_name = input_example.columns[0]
window_spec = Window.rows_between(
    Window.UNBOUNDED_PRECEDING, Window.UNBOUNDED_FOLLOWING).order_by(col(first_column_name))
input_example = input_example.select([
    first_value(col(c), ignore_nulls=True).over(window_spec).alias(c)
    for c in input_example.columns
]).limit(5)

# Add sample weights column for classification tasks
if use_sample_weights:
    input_snowpark_df = add_sample_weights_col_to_snowpark_df(
        input_snowpark_df, params.col_label)

# Train/Test Split
if params.time_ordering:
    # Time-based split
    time_ordering_variable_unix = f"{time_ordering_variable_sf}_UNIX"
    input_snowpark_df = input_snowpark_df.withColumn(
        time_ordering_variable_unix,
        F.unix_timestamp(input_snowpark_df[time_ordering_variable_sf])
    )

    split_percentile = input_snowpark_df.approx_quantile(
        time_ordering_variable_unix, [params.train_ratio])[0]

    train_snowpark_df = input_snowpark_df.filter(
        col(time_ordering_variable_unix) < split_percentile)
    test_snowpark_df = input_snowpark_df.filter(
        col(time_ordering_variable_unix) >= split_percentile)

    train_snowpark_df = train_snowpark_df.drop(time_ordering_variable_unix)
    test_snowpark_df = test_snowpark_df.drop(time_ordering_variable_unix)

    logger.info(f"train set nrecords: {train_snowpark_df.count()}")
    logger.info(f"test set nrecords: {test_snowpark_df.count()}")
else:
    # Random split
    train_snowpark_df, test_snowpark_df = input_snowpark_df.random_split(
        weights=[params.train_ratio, 1 - params.train_ratio],
        seed=params.random_seed
    )

# Write train/test datasets
dku_snowpark.write_with_schema(params.output_train_dataset, train_snowpark_df)
dku_snowpark.write_with_schema(params.output_test_dataset, test_snowpark_df)

# Feature Preprocessing Configuration
NUMERIC_DTYPES = {'number', 'decimal', 'numeric', 'int', 'integer', 'bigint', 'smallint', 'tinyint', 'byteint',
                  'float', 'float4', 'float8', 'double', 'double precision', 'real'}


def get_default_encoding(dtype):
    """Return default encoding based on column data type."""
    return 'Standard rescaling' if dtype in NUMERIC_DTYPES else 'Dummy encoding'


def get_default_imputation(dtype):
    """Return default imputation strategy based on column data type."""
    return 'Median' if dtype in NUMERIC_DTYPES else 'Most frequent value'


# Build list of features with their preprocessing configuration
included_features_handling_list = []

for feature_column in params.inputDatasetColumns:
    col_name = feature_column['name']
    if col_name not in params.selectedInputColumns or not params.selectedInputColumns[col_name]:
        continue

    feature_column['name_sf'] = sf_col_name(col_name)
    feature_column['include'] = True

    # Set encoding/rescaling
    if col_name in params.selectedOption1:
        feature_column["encoding_rescaling"] = params.selectedOption1[col_name]
    else:
        feature_column["encoding_rescaling"] = get_default_encoding(
            feature_column['type'])

    # Set imputation strategy
    if col_name in params.selectedOption2:
        feature_column["missingness_impute"] = params.selectedOption2[col_name]
        if params.selectedOption2[col_name] == 'Constant' and col_name in params.selectedConstantImpute:
            feature_column["constant_impute"] = params.selectedConstantImpute[col_name]
    else:
        feature_column["missingness_impute"] = get_default_imputation(
            feature_column['type'])

    if feature_column["encoding_rescaling"] == 'Dummy encoding':
        feature_column["max_categories"] = 20

    included_features_handling_list.append(feature_column)

# Build feature name lists
included_feature_names_non_sf = [f['name']
                                 for f in included_features_handling_list]
included_feature_names_sf = [f['name_sf']
                             for f in included_features_handling_list]

included_feature_names_plus_target_non_sf = included_feature_names_non_sf + \
    [params.col_label]
included_feature_names_plus_target_sf = included_feature_names_sf + \
    [col_label_sf]

# Filter input example to only include selected features and target
input_example = input_example.select(included_feature_names_plus_target_sf)

# Helper function to select transformer class based on backend


def get_transformer(snowpark_class, sklearn_class, **kwargs):
    """Return appropriate transformer class instance based on compute backend."""
    return snowpark_class(**kwargs) if is_snowpark_backend else sklearn_class(**kwargs)


def build_imputer(impute_strategy, constant_value=None):
    """Build an imputer transformer based on the strategy."""
    strategy_map = {
        "Average": "mean",
        "Median": "median",
        "Most frequent value": "most_frequent",
        "Constant": "constant"
    }

    strategy = strategy_map.get(impute_strategy)
    if strategy is None:
        return None

    if strategy == "constant":
        kwargs = {"strategy": "constant", "missing_values": pd.NA}
        if constant_value is not None:
            kwargs["fill_value"] = constant_value
        return get_transformer(snowpark_impute.SimpleImputer, sk_impute.SimpleImputer, **kwargs)

    return get_transformer(snowpark_impute.SimpleImputer, sk_impute.SimpleImputer, strategy=strategy)


def build_encoder(encoding_type, feature_name):
    """Build an encoder/scaler transformer based on the encoding type."""
    if encoding_type == "Standard rescaling":
        return get_transformer(snowpark_prep.StandardScaler, sk_prep.StandardScaler)
    elif encoding_type == "Min-max rescaling":
        return get_transformer(snowpark_prep.MinMaxScaler, sk_prep.MinMaxScaler)
    elif encoding_type == "No rescaling":
        logger.info(f"No rescaling for {feature_name}")
        return None
    elif encoding_type == "Dummy encoding":
        return get_transformer(
            snowpark_prep.OneHotEncoder, sk_prep.OneHotEncoder,
            handle_unknown='infrequent_if_exist', max_categories=10
        )
    elif encoding_type == "Ordinal encoding":
        return get_transformer(
            snowpark_prep.OrdinalEncoder, sk_prep.OrdinalEncoder,
            handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1
        )
    return None


# Build column transformers for each feature
col_transformer_list = []

for feature in included_features_handling_list:
    feature_name = feature['name_sf'] if is_snowpark_backend else feature['name']
    transformer_name = f"{feature_name[1:-1]}_tform" if feature_name.startswith(
        '"') else f"{feature_name}_tform"

    feature_transformers = []

    # Add imputer
    imputer = build_imputer(
        feature["missingness_impute"], feature.get("constant_impute"))
    if imputer is not None:
        feature_transformers.append(('imputer', imputer))

    # Add encoder/scaler
    encoder = build_encoder(feature["encoding_rescaling"], feature_name)
    if encoder is not None:
        feature_transformers.append(('enc', encoder))

    pipeline_class = snowpark_pipeline.Pipeline if is_snowpark_backend else sk_pipeline.Pipeline
    col_transformer_list.append(
        (transformer_name, pipeline_class(feature_transformers), [feature_name]))

# Create the final preprocessor
compose_class = snowpark_compose.ColumnTransformer if is_snowpark_backend else sk_compose.ColumnTransformer
preprocessor = compose_class(transformers=col_transformer_list)

# Algorithm Configuration
algorithms = []


def get_estimator(snowpark_cls, sklearn_cls):
    """Return appropriate estimator class instance based on compute backend."""
    return snowpark_cls() if is_snowpark_backend else sklearn_cls()


if is_classification:
    if params.logistic_regression:
        algorithms.append({'algorithm': 'logistic_regression',
                           'estimator': get_estimator(snowpark_linear.LogisticRegression, sk_linear.LogisticRegression),
                           'gs_params': {'clf__C': loguniform(params.logistic_regression_c_min, params.logistic_regression_c_max),
                                         'clf__multi_class': ['auto']}})

    if params.random_forest_classification:
        algorithms.append({'algorithm': 'random_forest_classification',
                           'estimator': get_estimator(snowpark_ensemble.RandomForestClassifier, sk_ensemble.RandomForestClassifier),
                           'gs_params': {'clf__n_estimators': randint(params.random_forest_classification_n_estimators_min, params.random_forest_classification_n_estimators_max),
                                         'clf__max_depth': randint(params.random_forest_classification_max_depth_min, params.random_forest_classification_max_depth_max),
                                         'clf__min_samples_leaf': randint(params.random_forest_classification_min_samples_leaf_min, params.random_forest_classification_min_samples_leaf_max)}})

    if params.xgb_classification:
        algorithms.append({'algorithm': 'xgb_classification',
                           'estimator': get_estimator(snowpark_xgboost.XGBClassifier, XGBClassifier),
                           'gs_params': {'clf__n_estimators': randint(params.xgb_classification_n_estimators_min, params.xgb_classification_n_estimators_max),
                                         'clf__max_depth': randint(params.xgb_classification_max_depth_min, params.xgb_classification_max_depth_max),
                                         'clf__min_child_weight': uniform(params.xgb_classification_min_child_weight_min, params.xgb_classification_min_child_weight_max),
                                         'clf__learning_rate': loguniform(params.xgb_classification_learning_rate_min, params.xgb_classification_learning_rate_max)}})

    if params.lgbm_classification:
        algorithms.append({'algorithm': 'lgbm_classification',
                           'estimator': get_estimator(snowpark_lightgbm.LGBMClassifier, LGBMClassifier),
                           'gs_params': {'clf__n_estimators': randint(params.lgbm_classification_n_estimators_min, params.lgbm_classification_n_estimators_max),
                                         'clf__max_depth': randint(params.lgbm_classification_max_depth_min, params.lgbm_classification_max_depth_max),
                                         'clf__min_child_weight': uniform(params.lgbm_classification_min_child_weight_min, params.lgbm_classification_min_child_weight_max),
                                         'clf__learning_rate': loguniform(params.lgbm_classification_learning_rate_min, params.lgbm_classification_learning_rate_max)}})

    if params.gb_classification:
        algorithms.append({'algorithm': 'gb_classification',
                           'estimator': get_estimator(snowpark_ensemble.GradientBoostingClassifier, sk_ensemble.GradientBoostingClassifier),
                           'gs_params': {'clf__n_estimators': randint(params.gb_classification_n_estimators_min, params.gb_classification_n_estimators_max),
                                         'clf__max_depth': randint(params.gb_classification_max_depth_min, params.gb_classification_max_depth_max),
                                         'clf__min_samples_leaf': randint(params.gb_classification_min_samples_leaf_min, params.gb_classification_min_samples_leaf_max),
                                         'clf__learning_rate': loguniform(params.gb_classification_learning_rate_min, params.gb_classification_learning_rate_max)}})

    if params.decision_tree_classification:
        algorithms.append({'algorithm': 'decision_tree_classification',
                           'estimator': get_estimator(snowpark_tree.DecisionTreeClassifier, sk_tree.DecisionTreeClassifier),
                           'gs_params': {'clf__max_depth': randint(params.decision_tree_classification_max_depth_min, params.decision_tree_classification_max_depth_max),
                                         'clf__min_samples_leaf': randint(params.decision_tree_classification_min_samples_leaf_min, params.decision_tree_classification_min_samples_leaf_max)}})

else:
    if params.lasso_regression:
        algorithms.append({'algorithm': 'lasso_regression',
                           'estimator': get_estimator(snowpark_linear.Lasso, sk_linear.Lasso),
                           'gs_params': {'clf__alpha': loguniform(params.lasso_regression_alpha_min, params.lasso_regression_alpha_max)}})

    if params.random_forest_regression:
        algorithms.append({'algorithm': 'random_forest_regression',
                           'estimator': get_estimator(snowpark_ensemble.RandomForestRegressor, sk_ensemble.RandomForestRegressor),
                           'gs_params': {'clf__n_estimators': randint(params.random_forest_regression_n_estimators_min, params.random_forest_regression_n_estimators_max),
                                         'clf__max_depth': randint(params.random_forest_regression_max_depth_min, params.random_forest_regression_max_depth_max),
                                         'clf__min_samples_leaf': randint(params.random_forest_regression_min_samples_leaf_min, params.random_forest_regression_min_samples_leaf_max)}})

    if params.xgb_regression:
        algorithms.append({'algorithm': 'xgb_regression',
                           'estimator': get_estimator(snowpark_xgboost.XGBRegressor, XGBRegressor),
                           'gs_params': {'clf__n_estimators': randint(params.xgb_regression_n_estimators_min, params.xgb_regression_n_estimators_max),
                                         'clf__max_depth': randint(params.xgb_regression_max_depth_min, params.xgb_regression_max_depth_max),
                                         'clf__min_child_weight': uniform(params.xgb_regression_min_child_weight_min, params.xgb_regression_min_child_weight_max),
                                         'clf__learning_rate': loguniform(params.xgb_regression_learning_rate_min, params.xgb_regression_learning_rate_max)}})
    if params.lgbm_regression:
        algorithms.append({'algorithm': 'lgbm_regression',
                           'estimator': get_estimator(snowpark_lightgbm.LGBMRegressor, LGBMRegressor),
                           'gs_params': {'clf__n_estimators': randint(params.lgbm_regression_n_estimators_min, params.lgbm_regression_n_estimators_max),
                                         'clf__max_depth': randint(params.lgbm_regression_max_depth_min, params.lgbm_regression_max_depth_max),
                                         'clf__min_child_weight': uniform(params.lgbm_regression_min_child_weight_min, params.lgbm_regression_min_child_weight_max),
                                         'clf__learning_rate': loguniform(params.lgbm_regression_learning_rate_min, params.lgbm_regression_learning_rate_max)}})

    if params.gb_regression:
        algorithms.append({'algorithm': 'gb_regression',
                           'estimator': get_estimator(snowpark_ensemble.GradientBoostingRegressor, sk_ensemble.GradientBoostingRegressor),
                           'gs_params': {'clf__n_estimators': randint(params.gb_regression_n_estimators_min, params.gb_regression_n_estimators_max),
                                         'clf__max_depth': randint(params.gb_regression_max_depth_min, params.gb_regression_max_depth_max),
                                         'clf__min_samples_leaf': randint(params.gb_regression_min_samples_leaf_min, params.gb_regression_min_samples_leaf_max),
                                         'clf__learning_rate': loguniform(params.gb_regression_learning_rate_min, params.gb_regression_learning_rate_max)}})

    if params.decision_tree_regression:
        algorithms.append({'algorithm': 'decision_tree_regression',
                           'estimator': get_estimator(snowpark_tree.DecisionTreeRegressor, sk_tree.DecisionTreeRegressor),
                           'gs_params': {'clf__max_depth': randint(params.decision_tree_regression_max_depth_min, params.decision_tree_regression_max_depth_max),
                                         'clf__min_samples_leaf': randint(params.decision_tree_regression_min_samples_leaf_min, params.decision_tree_regression_min_samples_leaf_max)}})

# MLflow Model Wrappers
# These wrappers handle column name mapping between pandas and Snowflake formats


class BaseModelWrapper(mlflow.pyfunc.PythonModel):
    """Base class for MLflow model wrappers."""

    def load_context(self, context):
        self.model = load(open(context.artifacts["grid_pipe_sklearn"], 'rb'))


class SnowparkMLClassifierWrapper(BaseModelWrapper):
    """Wrapper for Snowpark ML classifiers that maps column names."""

    def load_context(self, context):
        super().load_context(context)
        self.features_quotes_lookup = load(
            open(context.artifacts["features_quotes_lookup"], 'rb'))

    def predict(self, context, input_df):
        input_df_copy = input_df.copy()
        input_df_copy.columns = [self.features_quotes_lookup[col]
                                 for col in input_df_copy.columns]
        return self.model.predict_proba(input_df_copy)


class SnowparkMLRegressorWrapper(BaseModelWrapper):
    """Wrapper for Snowpark ML regressors that maps column names."""

    def load_context(self, context):
        super().load_context(context)
        self.features_quotes_lookup = load(
            open(context.artifacts["features_quotes_lookup"], 'rb'))

    def predict(self, context, input_df):
        input_df_copy = input_df.copy()
        input_df_copy.columns = [self.features_quotes_lookup[col]
                                 for col in input_df_copy.columns]
        return self.model.predict(input_df_copy)


class SklearnClassifierWrapper(BaseModelWrapper):
    """Wrapper for sklearn classifiers."""

    def predict(self, context, input_df):
        return self.model.predict_proba(input_df)


class SklearnRegressorWrapper(BaseModelWrapper):
    """Wrapper for sklearn regressors."""

    def predict(self, context, input_df):
        return self.model.predict(input_df)

# Training Implementation Functions


def train_snowpark_impl(algo, prepr, score_met, col_lab, weight_col, feat_names, train_df, num_iter):
    """Train a model using Snowpark ML (warehouse backend)."""
    logger.info(f"Training Snowpark ML model: {algo['algorithm']}")

    pipe = snowpark_pipeline.Pipeline(
        steps=[('preprocessor', prepr), ('clf', algo['estimator'])])

    rs_clf = snowpark_ms.RandomizedSearchCV(
        estimator=pipe,
        param_distributions=algo['gs_params'],
        n_iter=num_iter,
        cv=3,
        scoring=score_met,
        input_cols=feat_names,
        label_cols=col_lab,
        output_cols="PREDICTION",
        sample_weight_col=weight_col
    )
    rs_clf.fit(train_df)
    return {'algorithm': algo['algorithm'], 'model_obj': rs_clf, 'backend': 'snowpark'}


def train_sklearn_impl(algo, prepr, score_met, col_lab, weight_col, feat_names, train_df_pandas, num_iter):
    """Train a model using sklearn (compute pool backend)."""
    logger.info(f"Training sklearn model: {algo['algorithm']}")

    X = train_df_pandas[feat_names]
    y = train_df_pandas[col_lab]
    weights = train_df_pandas[weight_col] if weight_col else None

    pipe = sk_pipeline.Pipeline(
        steps=[('preprocessor', prepr), ('clf', algo['estimator'])])

    rs_clf = sk_ms.RandomizedSearchCV(
        estimator=pipe,
        param_distributions=algo['gs_params'],
        n_iter=num_iter,
        cv=3,
        scoring=score_met,
        n_jobs=-1
    )

    fit_params = {'clf__sample_weight': weights} if weights is not None else {}
    rs_clf.fit(X, y, **fit_params)
    return {'algorithm': algo['algorithm'], 'model_obj': rs_clf, 'backend': 'sklearn'}


def run_training_loop(backend, train_data, algos, prepr, metric, col_lab, weight_col, feats, n_iter):
    """Train all algorithms and return results."""
    train_func = train_snowpark_impl if backend == 'warehouse' else train_sklearn_impl
    return [train_func(alg, prepr, metric, col_lab, weight_col, feats, train_data, n_iter) for alg in algos]


# Execute Training
trained_models = []

if params.compute_backend == 'warehouse':
    trained_models = run_training_loop(
        'warehouse', train_snowpark_df, algorithms, preprocessor,
        scoring_metric, col_label_sf, sample_weight_col, included_feature_names_sf, params.n_iter
    )
else:
    full_table_name = params.output_train_dataset.get_location_info()[
        'info']['quotedResolvedTableName']

    def remote_wrapper(session, table_name, algos, prep, metric, col, weight, feats, iters):
        df_pandas = session.table(table_name).to_pandas()
        return run_training_loop('pool', df_pandas, algos, prep, metric, col, weight, feats, iters)

    remote_job = remote(
        compute_pool=params.compute_pool,
        stage_name=params.stage_name,
        session=session,
        pip_requirements=snowpark_packages_versions,
        external_access_integrations=[SNOWFLAKE_PYPI_EAI_NAME]
    )(remote_wrapper)
    train_models_job = remote_job(session, full_table_name, algorithms, preprocessor,
                                  scoring_metric, params.col_label, sample_weight_col, included_feature_names_non_sf, params.n_iter)
    logger.info("Starting Snowflake ML Training Job...")
    train_models_job.get_logs()
    trained_models = train_models_job.result()


# Model Evaluation and MLflow Logging
final_models = []

# Convert test set to pandas once for compute pool mode
test_df_pandas = None
if params.compute_backend != 'warehouse':
    logger.info("Converting test set to pandas for evaluation...")
    test_df_pandas = test_snowpark_df.to_pandas()

for model_info in trained_models:
    backend = model_info.get(
        'backend', 'warehouse' if params.compute_backend == 'warehouse' else 'pool')
    raw_model_obj = model_info['model_obj']
    model_algo = model_info['algorithm']

    # Extract sklearn object from Snowpark model if needed
    grid_pipe_sklearn = raw_model_obj.to_sklearn(
    ) if backend == 'snowpark' else raw_model_obj

    # Log cross-validation results
    cv_results = pd.DataFrame(grid_pipe_sklearn.cv_results_)
    cv_results['algorithm'] = model_algo
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    cv_results.reset_index(drop=True, inplace=True)

    for idx, row in cv_results.iterrows():
        run_name = f"{params.model_name}_{now}_cv_{idx + 1}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("algorithm", row['algorithm'])
            mlflow.log_metric("mean_fit_time", row["mean_fit_time"])
            mlflow.log_metric(
                f"mean_test_{scoring_metric}", row["mean_test_score"])
            mlflow.log_metric("rank_test_score", int(row["rank_test_score"]))

            for column in row.index:
                if "param_" in column:
                    mlflow.log_param(column.replace(
                        "param_clf__", ""), row[column])

        # Log CV results to Snowflake Experiment
        try:
            snowflake_run_name = run_name + "_sf"
            with snowflake_experiment.start_run(run_name=snowflake_run_name):

                snowflake_experiment.log_param("algorithm", row['algorithm'])
                snowflake_experiment.log_metric(
                    "mean_fit_time", row["mean_fit_time"])
                snowflake_experiment.log_metric(
                    f"mean_test_{scoring_metric}", row["mean_test_score"])
                snowflake_experiment.log_metric(
                    "rank_test_score", int(row["rank_test_score"]))

                for column in row.index:
                    if "param_" in column:
                        snowflake_experiment.log_param(column.replace(
                            "param_clf__", ""), row[column])
        except Exception as e:
            logger.info(
                f"Failed to log metrics with Snowflake Experiment Tracking, exception: {e}")

    # Log final model and evaluate on test set
    run_name = f"{params.model_name}_{now}_final_model"
    run = mlflow.start_run(run_name=run_name)

    # Log best hyperparameters
    mlflow.log_metric("whole_dataset_refit_time",
                      grid_pipe_sklearn.refit_time_)
    for param, value in grid_pipe_sklearn.best_params_.items():
        if "clf" in param:
            mlflow.log_param(param.replace("clf__", ""), value)
    mlflow.log_param("algorithm", model_algo)

    test_metrics = {}

    # --- BRANCH A: SNOWPARK ML EVALUATION (Warehouse) ---
    if backend == 'snowpark':
        # Predict on Snowpark DataFrame
        test_predictions_df = raw_model_obj.predict(test_snowpark_df)

        # Ensure PREDICTION column type matches target
        col_label_dtype = convert_snowpark_df_col_dtype(
            test_predictions_df, col_label_sf)
        test_predictions_df = test_predictions_df.withColumn(
            '"PREDICTION"', test_predictions_df['"PREDICTION"'].cast(
                col_label_dtype)
        )

        if params.prediction_type == "two-class classification":
            # Snowpark Metrics
            test_prediction_probas_df = raw_model_obj.predict_proba(
                test_snowpark_df)

            test_f1 = snowpark_metrics.f1_score(df=test_predictions_df, y_true_col_names=col_label_sf,
                                                y_pred_col_names='"PREDICTION"', pos_label=col_label_values[0])
            test_roc_auc = snowpark_metrics.roc_auc_score(df=test_prediction_probas_df, y_true_col_names=col_label_sf,
                                                          y_score_col_names=test_prediction_probas_df.columns[-1])
            test_accuracy = snowpark_metrics.accuracy_score(
                df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"')
            test_recall = snowpark_metrics.recall_score(df=test_predictions_df, y_true_col_names=col_label_sf,
                                                        y_pred_col_names='"PREDICTION"', pos_label=col_label_values[0])
            test_precision = snowpark_metrics.precision_score(df=test_predictions_df, y_true_col_names=col_label_sf,
                                                              y_pred_col_names='"PREDICTION"', pos_label=col_label_values[0])

            # Map for logging
            metric_vals = {"test_f1": test_f1, "test_roc_auc": test_roc_auc,
                           "test_accuracy": test_accuracy, "test_recall": test_recall, "test_precision": test_precision}

        elif params.prediction_type == "multi-class classification":
            model_classes = grid_pipe_sklearn.classes_
            test_prediction_probas_df = raw_model_obj.predict_proba(
                test_snowpark_df)
            target_col_value_cols = [
                col for col in test_prediction_probas_df.columns if "PREDICT_PROBA" in col]

            test_f1 = snowpark_metrics.f1_score(df=test_predictions_df, y_true_col_names=col_label_sf,
                                                y_pred_col_names='"PREDICTION"', average="macro")
            test_roc_auc = snowpark_metrics.roc_auc_score(df=test_prediction_probas_df, y_true_col_names=col_label_sf,
                                                          y_score_col_names=target_col_value_cols, labels=model_classes, average="macro", multi_class="ovo")
            test_accuracy = snowpark_metrics.accuracy_score(
                df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"')
            test_recall = snowpark_metrics.recall_score(
                df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"', average="macro")
            test_precision = snowpark_metrics.precision_score(
                df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"', average="macro")

            metric_vals = {"test_f1": test_f1, "test_roc_auc": test_roc_auc,
                           "test_accuracy": test_accuracy, "test_recall": test_recall, "test_precision": test_precision}

        else:  # Regression
            test_r2 = snowpark_metrics.r2_score(
                df=test_predictions_df, y_true_col_name=col_label_sf, y_pred_col_name='"PREDICTION"')
            test_mae = snowpark_metrics.mean_absolute_error(
                df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"')
            test_mse = snowpark_metrics.mean_squared_error(
                df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"')
            test_rmse = snowpark_metrics.mean_squared_error(
                df=test_predictions_df, y_true_col_names=col_label_sf, y_pred_col_names='"PREDICTION"', squared=False)

            metric_vals = {"test_r2": test_r2, "test_mae": test_mae,
                           "test_mse": test_mse, "test_rmse": test_rmse}

    # --- BRANCH B: STANDARD SKLEARN EVALUATION (Compute Pool) ---
    else:
        # Predict on Pandas DataFrame
        y_true = test_df_pandas[col_label_sf]
        y_pred = raw_model_obj.predict(test_df_pandas)

        if params.prediction_type == "two-class classification":
            # Get Probabilities (for AUC) - usually column 1 is the positive class
            y_proba = raw_model_obj.predict_proba(test_df_pandas)[:, 1]

            # Use Standard Sklearn Metrics (imported as aliases or directly)
            # Note: sk_metrics.* refers to sklearn.metrics imported as sk_metrics, or use direct names if imported
            test_f1 = sk_metrics.f1_score(
                y_true, y_pred, pos_label=col_label_values[0])
            test_roc_auc = sk_metrics.roc_auc_score(y_true, y_proba)
            test_accuracy = sk_metrics.accuracy_score(y_true, y_pred)
            test_recall = sk_metrics.recall_score(
                y_true, y_pred, pos_label=col_label_values[0])
            test_precision = sk_metrics.precision_score(
                y_true, y_pred, pos_label=col_label_values[0])

            metric_vals = {"test_f1": test_f1, "test_roc_auc": test_roc_auc,
                           "test_accuracy": test_accuracy, "test_recall": test_recall, "test_precision": test_precision}

        elif params.prediction_type == "multi-class classification":
            # Multi-class Probabilities
            y_proba = raw_model_obj.predict_proba(test_df_pandas)
            model_classes = raw_model_obj.classes_

            test_f1 = sk_metrics.f1_score(y_true, y_pred, average="macro")
            # roc_auc_score handles multi-class with 'ovo'/'ovr'
            test_roc_auc = sk_metrics.roc_auc_score(
                y_true, y_proba, labels=model_classes, average="macro", multi_class="ovo")
            test_accuracy = sk_metrics.accuracy_score(y_true, y_pred)
            test_recall = sk_metrics.recall_score(
                y_true, y_pred, average="macro")
            test_precision = sk_metrics.precision_score(
                y_true, y_pred, average="macro")

            metric_vals = {"test_f1": test_f1, "test_roc_auc": test_roc_auc,
                           "test_accuracy": test_accuracy, "test_recall": test_recall, "test_precision": test_precision}

        else:  # Regression
            test_r2 = sk_metrics.r2_score(y_true, y_pred)
            test_mae = sk_metrics.mean_absolute_error(y_true, y_pred)
            test_mse = sk_metrics.mean_squared_error(y_true, y_pred)
            test_rmse = sk_metrics.root_mean_squared_error(y_true, y_pred)

            metric_vals = {"test_r2": test_r2, "test_mae": test_mae,
                           "test_mse": test_mse, "test_rmse": test_rmse}

    # 4. Log Metrics to MLflow & Log to logger
    for metric_name, metric_value in metric_vals.items():
        mlflow.log_metric(f"{metric_name}_score", metric_value)
        test_metrics[metric_name] = metric_value  # store for final list
        logger.info(f"{metric_name}: {metric_value}")

    best_score = grid_pipe_sklearn.best_score_

    # 5. Serialize and Log Model
    artifacts = {
        "grid_pipe_sklearn": "grid_pipe_sklearn.pkl",
        "features_quotes_lookup": "features_quotes_lookup.pkl"
    }

    dump(grid_pipe_sklearn, open(artifacts.get("grid_pipe_sklearn"), 'wb'))
    dump(features_quotes_lookup, open(
        artifacts.get("features_quotes_lookup"), 'wb'))

    # We reuse the same Wrapper classes defined earlier.
    # Even for Standard Sklearn models, the wrapper is useful because it maps
    # Dataiku-friendly column names back to the specific (possibly quoted)
    # names the model was trained on.
    if is_snowpark_backend:
        if is_classification:
            logged_model = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=SnowparkMLClassifierWrapper(),
                artifacts=artifacts
            )
        else:
            logged_model = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=SnowparkMLRegressorWrapper(),
                artifacts=artifacts
            )
    else:
        if is_classification:
            logged_model = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=SklearnClassifierWrapper(),
                artifacts=artifacts
            )
        else:
            logged_model = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=SklearnRegressorWrapper(),
                artifacts=artifacts
            )

    mlflow.end_run()

    # Log final model to Snowflake Experiment
    try:
        snowflake_run_name = run_name + "_sf"
        with snowflake_experiment.start_run(run_name=snowflake_run_name):
            # Log best hyperparameters
            snowflake_experiment.log_metric("whole_dataset_refit_time",
                                            grid_pipe_sklearn.refit_time_)
            for param, value in grid_pipe_sklearn.best_params_.items():
                if "clf" in param:
                    snowflake_experiment.log_param(
                        param.replace("clf__", ""), value)
            snowflake_experiment.log_param("algorithm", model_algo)

            # Log test metrics
            for metric_name, metric_value in metric_vals.items():
                snowflake_experiment.log_metric(
                    f"{metric_name}_score", metric_value)
    except Exception as e:
        logger.info(
            f"Failed to log metrics with Snowflake Experiment Tracking, exception: {e}")

    # Store result for selection of "Best of the Best"
    best_run_id = run.info.run_id
    final_models.append({
        'algorithm': model_algo,
        'sklearn_obj': grid_pipe_sklearn,
        'snowml_obj': raw_model_obj,  # Might be standard sklearn obj in pool case
        'mlflow_best_run_id': best_run_id,
        'run_name': run_name,
        'best_score': best_score,
        'test_metrics': test_metrics
    })


# SECTION 12 - Pull the best model, import it into a SavedModel green diamond (it will create a new one if doesn't exist), and evaluate on the hold out Test dataset
# Get the final best model (of the best models of each algorithm type) based on the performance metric chosen
if scoring_metric in ['roc_auc', 'accuracy', 'f1', 'precision', 'recall', 'r2']:
    best_model = max(final_models, key=lambda x: x['best_score'])
else:
    best_model = min(final_models, key=lambda x: x['best_score'])

best_model_run_id = best_model['mlflow_best_run_id']

# If two-class classification, set the Dataiku MLflow imported model run inference info
if params.prediction_type == "two-class classification":
    # Use col_label_values from the actual Snowflake data instead of model.classes_
    # This ensures class labels match the type Dataiku sees in the evaluation dataset
    # (XGBoost converts float labels to integers internally, causing a mismatch)
    model_classes = sorted(col_label_values)
    # Convert numpy types to native Python types for JSON serialization
    if 'int' in str(type(model_classes[0])):
        model_classes = [int(c) for c in model_classes]
    elif 'float' in str(type(model_classes[0])):
        model_classes = [float(c) for c in model_classes]

    mlflow_extension.set_run_inference_info(run_id=best_model_run_id,
                                            prediction_type='BINARY_CLASSIFICATION',
                                            classes=list(model_classes),
                                            code_env_name=MLFLOW_CODE_ENV_NAME)

if params.prediction_type == "multi-class classification":
    # Use col_label_values from the actual Snowflake data instead of model.classes_
    model_classes = sorted(col_label_values)
    # Convert numpy types to native Python types for JSON serialization
    if 'int' in str(type(model_classes[0])):
        model_classes = [int(c) for c in model_classes]
    elif 'float' in str(type(model_classes[0])):
        model_classes = [float(c) for c in model_classes]

    mlflow_extension.set_run_inference_info(run_id=best_model_run_id,
                                            prediction_type='MULTICLASS',
                                            classes=list(model_classes),
                                            code_env_name=MLFLOW_CODE_ENV_NAME)

# Get the managed folder subpath for the best trained model
model_artifact_first_directory = re.search(
    r'.*/(.+$)', mlflow_experiment.artifact_location).group(1)
model_path = f"{model_artifact_first_directory}/{best_model_run_id}/artifacts/model"

# If the Saved Model already exists in the flow (matching the user-inputted model name in the plugin), get it
sm_id = None
for sm in project.list_saved_models():
    if sm["name"] != params.model_name:
        continue
    else:
        sm_id = sm["id"]
        logger.info(f"Found Saved Model {sm['name']} with id {sm['id']}")
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
    logger.info(f"Saved Model not found, created new one with id {sm_id}")

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
model_version_labels.append(
    {'key': 'model:algorithm', 'value': best_model['algorithm']})
active_version_details.details['userMeta']['labels'] = model_version_labels
active_version_details.save_user_meta()

# Get the output test dataset name
output_test_dataset_name = params.output_test_dataset.name.split('.')[1]

# Set the Saved Model metadata (target name, classes,...)
if is_classification:
    mlflow_version.set_core_metadata(target_column_name=params.col_label, class_labels=list(
        model_classes), get_features_from_dataset=output_test_dataset_name)
else:
    mlflow_version.set_core_metadata(target_column_name=params.col_label,
                                     get_features_from_dataset=output_test_dataset_name)

# Evaluate the performance of this new version, to populate the performance screens of the Saved Model version in Dataiku
mlflow_version.evaluate(output_test_dataset_name,
                        container_exec_config_name='NONE',
                        skip_expensive_reports=False)

# If selected, deploy the best trained model to a Snowpark ML Model Registry in the current working database and schema
if params.deploy_to_snowflake_model_registry:
    try:

        registry = Registry(session=session)
        snowflake_registry_model_description = f"Dataiku Project: {project.project_key}, Model: {params.model_name}"
        snowflake_model_name = f"{project.project_key}_{params.model_name}"

        if is_snowpark_backend:
            model_ver = registry.log_model(model=best_model["snowml_obj"],
                                           model_name=snowflake_model_name,
                                           version_name=best_model["run_name"],
                                           comment=snowflake_registry_model_description,
                                           options={"relax_version": False,
                                                    "method_options": {
                                                        "predict": {
                                                            "case_sensitive": True
                                                        },
                                                        "predict_proba": {
                                                            "case_sensitive": True
                                                        }
                                                    }
                                                    })
        else:
            input_example_pd = input_example.to_pandas()
            input_features_sample = input_example_pd[included_feature_names_non_sf]
            input_feature_names = included_feature_names_non_sf
            target_sample_list = input_example_pd[params.col_label].tolist()

            predict_sig = model_signature.infer_signature(
                input_data=input_features_sample,
                output_data=target_sample_list,
                input_feature_names=input_feature_names,
                output_feature_names=['PREDICTION'])

            if is_classification:
                # Use model's classes_ for signature - this matches what the model actually outputs
                # Note: XGBoost uses integers internally, so output columns will be PREDICT_PROBA_0, PREDICT_PROBA_1

                best_model_classes = best_model["sklearn_obj"].classes_
                predict_proba_output_names = [
                    f'"PREDICT_PROBA_{class_label}"' for class_label in best_model_classes]
                proba_output_sample = np.random.uniform(
                    0.0, 1.0, size=(10, len(best_model_classes)))

                predict_proba_sig = model_signature.infer_signature(
                    input_data=input_features_sample,
                    output_data=proba_output_sample,
                    input_feature_names=input_feature_names,
                    output_feature_names=predict_proba_output_names)

                model_ver = registry.log_model(model=best_model["snowml_obj"],
                                               model_name=snowflake_model_name,
                                               version_name=best_model["run_name"],
                                               comment=snowflake_registry_model_description,
                                               signatures={"predict": predict_sig,
                                                           "predict_proba": predict_proba_sig},
                                               options={"relax_version": False,
                                                        "method_options": {
                                                            "predict": {
                                                                "case_sensitive": True
                                                            },
                                                            "predict_proba": {
                                                                "case_sensitive": True
                                                            }
                                                        }
                                                        })
            else:
                model_ver = registry.log_model(model=best_model["snowml_obj"],
                                               model_name=snowflake_model_name,
                                               version_name=best_model["run_name"],
                                               comment=snowflake_registry_model_description,
                                               signatures={
                                                   "predict": predict_sig},
                                               options={"relax_version": False,
                                                        "method_options": {
                                                            "predict": {
                                                                "case_sensitive": True
                                                            }
                                                        }
                                                        })
        logger.info(
            f"Successfully deployed model name: {snowflake_model_name}, model version: {best_model['run_name']} to Snowflake ML Model Registry")

        for test_metric in best_model["test_metrics"]:
            model_ver.set_metric(metric_name=test_metric,
                                 value=best_model["test_metrics"][test_metric])

        # Need to set tags at the parent model level
        parent_model = registry.get_model(snowflake_model_name)

        # Update the defuault model version to the new version
        parent_model.default = best_model["run_name"]
        logger.info(
            f"Successfully updated the default model version to the new version: {best_model['run_name']}")

        # Need to create the tag object in Snowflake if it doesn't exist
        session.sql("CREATE TAG IF NOT EXISTS APPLICATION;").show()
        session.sql("CREATE TAG IF NOT EXISTS DATAIKU_PROJECT_KEY;").show()
        session.sql("CREATE TAG IF NOT EXISTS DATAIKU_SAVED_MODEL_ID;").show()

        parent_model.set_tag("application", "Dataiku")
        parent_model.set_tag("dataiku_project_key", project.project_key)
        parent_model.set_tag("dataiku_saved_model_id", sm_id)

        logger.info("Successfully set model tags")
    except Exception as e:
        logger.error(
            f"Failed to deploy model to Snowflake ML Model Registry, exception: {e}")

# Get the current plugin recipe instance name
current_recipe_name = FLOW["currentActivityId"][:-3].replace('_NP', '')

# Get the Dataiku.Recipe object, and add the new trained Saved Model as an output of the recipe (if it isn't already)
recipe = project.get_recipe(current_recipe_name)
recipe_settings = recipe.get_settings()
saved_model_names = get_output_names_for_role('saved_model_name')

if len(saved_model_names) > 0:
    prev_saved_model_name = saved_model_names[0].split('.')[-1]
    if prev_saved_model_name != sm_id:
        recipe_settings.replace_output(
            current_output_ref=prev_saved_model_name, new_output_ref=sm_id)
        recipe_settings.save()
else:
    recipe_settings.add_output(role="saved_model_name", ref=sm_id)
    recipe_settings.save()
