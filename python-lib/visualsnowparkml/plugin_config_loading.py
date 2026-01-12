# -*- coding: utf-8 -*-
"""Module with utility functions for loading, resolving and validating plugin parameters"""
from typing import Tuple
import re
import logging
import dataiku
from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role
)
from dataiku.snowpark import DkuSnowpark
from snowflake.snowpark.session import Session
from snowflake.snowpark.table import Table
from snowflake.ml.jobs import remote


# Custom Exceptions
class PluginParamValidationError(ValueError):
    """Plugin parameters chosen by the user are invalid."""


class CodeEnvSetupError(ValueError):
    """Code environment is not set up correctly."""


class InputTrainDatasetSetupError(ValueError):
    """Input training dataset is not set up correctly."""


class SnowflakeResourcesError(ValueError):
    """Snowflake resources are busy or unavailable."""


class TrainPluginParams:
    """Class to store train recipe parameters"""

    def __init__(self):
        # Loop through all slots and set them to None by default
        for slot in self.__slots__:
            setattr(self, slot, None)

    __slots__ = [
        "output_train_dataset",
        "output_test_dataset",
        "model_experiment_tracking_folder",
        "model_experiment_tracking_folder_id",
        "model_name",
        "col_label",
        "prediction_type",
        "disable_class_weights",
        "time_ordering",
        "time_ordering_variable",
        "train_ratio",
        "random_seed",
        "model_metric",
        "compute_backend",
        "compute_pool",
        "stage_name",
        "warehouse",
        "deploy_to_snowflake_model_registry",
        "inputDatasetColumns",
        "selectedInputColumns",
        "selectedOption1",
        "selectedOption2",
        "selectedConstantImpute",
        "logistic_regression",
        "logistic_regression_c_min",
        "logistic_regression_c_max",
        "random_forest_classification",
        "random_forest_classification_n_estimators_min",
        "random_forest_classification_n_estimators_max",
        "random_forest_classification_max_depth_min",
        "random_forest_classification_max_depth_max",
        "random_forest_classification_min_samples_leaf_min",
        "random_forest_classification_min_samples_leaf_max",
        "xgb_classification",
        "xgb_classification_n_estimators_min",
        "xgb_classification_n_estimators_max",
        "xgb_classification_max_depth_min",
        "xgb_classification_max_depth_max",
        "xgb_classification_min_child_weight_min",
        "xgb_classification_min_child_weight_max",
        "xgb_classification_learning_rate_min",
        "xgb_classification_learning_rate_max",
        "lgbm_classification",
        "lgbm_classification_n_estimators_min",
        "lgbm_classification_n_estimators_max",
        "lgbm_classification_max_depth_min",
        "lgbm_classification_max_depth_max",
        "lgbm_classification_min_child_weight_min",
        "lgbm_classification_min_child_weight_max",
        "lgbm_classification_learning_rate_min",
        "lgbm_classification_learning_rate_max",
        "gb_classification",
        "gb_classification_n_estimators_min",
        "gb_classification_n_estimators_max",
        "gb_classification_max_depth_min",
        "gb_classification_max_depth_max",
        "gb_classification_min_samples_leaf_min",
        "gb_classification_min_samples_leaf_max",
        "gb_classification_learning_rate_min",
        "gb_classification_learning_rate_max",
        "decision_tree_classification",
        "decision_tree_classification_max_depth_min",
        "decision_tree_classification_max_depth_max",
        "decision_tree_classification_min_samples_leaf_min",
        "decision_tree_classification_min_samples_leaf_max",
        "lasso_regression",
        "lasso_regression_alpha_min",
        "lasso_regression_alpha_max",
        "random_forest_regression",
        "random_forest_regression_n_estimators_min",
        "random_forest_regression_n_estimators_max",
        "random_forest_regression_max_depth_min",
        "random_forest_regression_max_depth_max",
        "random_forest_regression_min_samples_leaf_min",
        "random_forest_regression_min_samples_leaf_max",
        "xgb_regression",
        "xgb_regression_n_estimators_min",
        "xgb_regression_n_estimators_max",
        "xgb_regression_max_depth_min",
        "xgb_regression_max_depth_max",
        "xgb_regression_min_child_weight_min",
        "xgb_regression_min_child_weight_max",
        "xgb_regression_learning_rate_min",
        "xgb_regression_learning_rate_max",
        "lgbm_regression",
        "lgbm_regression_n_estimators_min",
        "lgbm_regression_n_estimators_max",
        "lgbm_regression_max_depth_min",
        "lgbm_regression_max_depth_max",
        "lgbm_regression_min_child_weight_min",
        "lgbm_regression_min_child_weight_max",
        "lgbm_regression_learning_rate_min",
        "lgbm_regression_learning_rate_max",
        "gb_regression",
        "gb_regression_n_estimators_min",
        "gb_regression_n_estimators_max",
        "gb_regression_max_depth_min",
        "gb_regression_max_depth_max",
        "gb_regression_min_samples_leaf_min",
        "gb_regression_min_samples_leaf_max",
        "gb_regression_learning_rate_min",
        "gb_regression_learning_rate_max",
        "decision_tree_regression",
        "decision_tree_regression_max_depth_min",
        "decision_tree_regression_max_depth_max",
        "decision_tree_regression_min_samples_leaf_min",
        "decision_tree_regression_min_samples_leaf_max",
        "n_iter"
    ]


class ScorePluginParams:
    """Class to store train recipe parameters"""

    def __init__(self):
        pass

    __slots__ = [
        "warehouse"
    ]


class ComputePoolBusyHandler(logging.Handler):
    def emit(self, record):
        if "Compute pool busy" in record.getMessage():
            raise SnowflakeResourcesError(
                "Snowflake compute pool busy. Please increase the MAX_NODES parameter of the compute pool, or choose a new one with available nodes")


def validate_hyperparameter_range(min_val, max_val, algo_name, param_name):
    """Validate that min <= max for a hyperparameter range."""
    if min_val is None or max_val is None:
        raise PluginParamValidationError(
            f"For the {algo_name} algorithm, please choose a min and max value for {param_name}")
    if min_val > max_val:
        raise PluginParamValidationError(
            f"The {algo_name} {param_name} min you selected: {min_val} is greater than max: {max_val}. "
            f"Choose a min that is less than max")


def load_and_validate_algorithm_params(params, recipe_config, algo_key, param_configs, algo_display_name):
    """Load algorithm parameters from config and validate hyperparameter ranges.

    Args:
        params: TrainPluginParams object to populate
        recipe_config: Recipe configuration dictionary
        algo_key: Base key for the algorithm (e.g., 'logistic_regression')
        param_configs: List of tuples (param_suffix, display_name) for each hyperparameter
                       e.g., [('c', 'C'), ('n_estimators', 'Number of Trees')]
        algo_display_name: Human-readable algorithm name for error messages

    Returns:
        True if algorithm is enabled, False otherwise
    """
    # Load algorithm enabled flag
    algo_enabled = recipe_config.get(algo_key, None)
    setattr(params, algo_key, algo_enabled)

    # Load all hyperparameter min/max values
    for param_suffix, _ in param_configs:
        min_key = f"{algo_key}_{param_suffix}_min"
        max_key = f"{algo_key}_{param_suffix}_max"
        setattr(params, min_key, recipe_config.get(min_key, None))
        setattr(params, max_key, recipe_config.get(max_key, None))

    # Validate if algorithm is enabled
    if algo_enabled:
        for param_suffix, display_name in param_configs:
            min_val = getattr(params, f"{algo_key}_{param_suffix}_min")
            max_val = getattr(params, f"{algo_key}_{param_suffix}_max")
            validate_hyperparameter_range(
                min_val, max_val, algo_display_name, display_name)

    return algo_enabled


def load_train_config_snowpark_session_and_input_train_snowpark_df() -> Tuple[TrainPluginParams, Session, Table]:
    """Utility function to:
        - Validate and load ml training parameters into a clean class

    Returns:
        - Class instance with parameter names as attributes and associated values
        - Snowpark session
        - Input training dataset as a Snowpark dataframe
    """

    params = TrainPluginParams()
    # Input dataset
    input_dataset_names = get_input_names_for_role("input_dataset_name")
    if len(input_dataset_names) != 1:
        raise PluginParamValidationError("Please specify one input dataset")
    input_dataset = dataiku.Dataset(input_dataset_names[0])
    input_dataset_column_types = {}
    for col in input_dataset.read_schema():
        input_dataset_column_types[col["name"]] = col["type"]

    # Output generated train and test sets
    output_train_dataset_names = get_output_names_for_role(
        'output_train_dataset_name')
    if len(output_train_dataset_names) != 1:
        raise PluginParamValidationError(
            "Please specify one output generated train dataset")
    else:
        output_train_dataset = dataiku.Dataset(output_train_dataset_names[0])
        params.output_train_dataset = output_train_dataset

    output_test_dataset_names = get_output_names_for_role(
        'output_test_dataset_name')
    if len(output_test_dataset_names) != 1:
        raise PluginParamValidationError(
            "Please specify one output generated test dataset")
    else:
        output_test_dataset = dataiku.Dataset(output_test_dataset_names[0])
        params.output_test_dataset = output_test_dataset

    # Output folder
    model_experiment_tracking_folder_names = get_output_names_for_role(
        'model_experiment_tracking_folder_name')
    if len(model_experiment_tracking_folder_names) != 1:
        raise PluginParamValidationError(
            "Please specify one output model folder")
    else:
        params.model_experiment_tracking_folder = dataiku.Folder(
            model_experiment_tracking_folder_names[0])
        params.model_experiment_tracking_folder_id = params.model_experiment_tracking_folder.get_id()

    # Recipe parameters
    recipe_config = get_recipe_config()

    # Model Name
    model_name = recipe_config.get('model_name', None)
    if not model_name:
        raise PluginParamValidationError("Empty model name")
    elif re.match(r'^[A-Za-z0-9_]+$', model_name):
        params.model_name = model_name
    else:
        raise PluginParamValidationError(
            f"Invalid model name: {model_name}. Alphanumeric and underscores only. No spaces, special characters (, . / \\ : ! @ # $ %, etc.)")

    # Target Column Label
    col_label = recipe_config.get('col_label', None)
    if not col_label:
        raise PluginParamValidationError("No target column selected")
    else:
        params.col_label = col_label

    # Prediction Type
    prediction_type = recipe_config.get('prediction_type', None)
    if not prediction_type:
        raise PluginParamValidationError(
            "No prediction type chosen. Choose Two-class classification, Multi-class classification, or Regression")
    else:
        params.prediction_type = prediction_type

    # Disable Class Weights (just a checkbox)
    params.disable_class_weights = recipe_config.get(
        'disable_class_weights', None)

    # Time Ordering (just a checkbox)
    params.time_ordering = recipe_config.get('time_ordering', False)

    # Time Ordering Column
    params.time_ordering_variable = recipe_config.get(
        'time_ordering_variable', None)
    if params.time_ordering:
        if not params.time_ordering_variable:
            raise PluginParamValidationError(
                "Selected time ordering but no time ordering column chosen. Choose a time ordering column")
        if input_dataset_column_types[params.time_ordering_variable] != "date":
            raise PluginParamValidationError(
                f"Time ordering column: {params.time_ordering_variable} is not a parsed date. Choose a parsed date")

        params.time_ordering_variable = params.time_ordering_variable

    # Train Ratio
    train_ratio = recipe_config.get('train_ratio', None)
    if not train_ratio:
        raise PluginParamValidationError(
            "No prediction train ratio chosen. Choose a train ratio between 0 and 1 (e.g. 0.8)")
    elif 0 < train_ratio < 1:
        params.train_ratio = train_ratio
    else:
        raise PluginParamValidationError(
            f"Train ratio: {train_ratio} is not between 0 and 1. Choose a train ratio between 0 and 1 (e.g. 0.8)")

    # Random Seed
    random_seed = recipe_config.get('random_seed', None)
    if not random_seed:
        raise PluginParamValidationError(
            "No random seed chosen. Choose a random seed that is an integer (e.g. 42)")
    elif isinstance(random_seed, int):
        params.random_seed = random_seed
    else:
        raise PluginParamValidationError(
            f"Random seed: {random_seed} is not an integer. Choose a random seed that is an integer (e.g. 42)")

    # Model Metric
    model_metric = recipe_config.get('model_metric', None)
    if not model_metric:
        raise PluginParamValidationError(
            "No model metric chosen. Choose a model metric")
    else:
        params.model_metric = model_metric

    # Compute Backend - Warehouse or Container Runtime with Compute Pool
    compute_backend = recipe_config.get('compute_backend', None)
    if not compute_backend:
        raise PluginParamValidationError(
            "No compute backend chosen. Choose Warehouse or Container Runtime")
    else:
        params.compute_backend = compute_backend

    client = dataiku.api_client()
    project = client.get_default_project()
    project_key = project.project_key

    dku_snowpark = DkuSnowpark()
    snowflake_connection_name = input_dataset.get_config()[
        'params']['connection']
    session = dku_snowpark.create_session(
        snowflake_connection_name, project_key=project_key)

    if compute_backend == 'warehouse':
        warehouse = recipe_config.get('warehouse', None)
        params.warehouse = f'"{warehouse}"'
        if params.warehouse:

            try:
                session.use_warehouse(warehouse)
                params.warehouse = warehouse
            except:
                raise PluginParamValidationError(
                    f"Snowflake Warehouse: {warehouse} does not exist or you do not have permission to use it")
    else:
        compute_pool = recipe_config.get('compute_pool', None)
        stage_name = recipe_config.get('stage_name', None)

        if not compute_pool:
            raise PluginParamValidationError("Empty compute pool name")
        else:
            params.compute_pool = compute_pool

        if not stage_name:
            raise PluginParamValidationError("Empty stage name")
        else:
            params.stage_name = stage_name

        try:
            @remote(compute_pool=compute_pool, stage_name=stage_name, external_access_integrations=["PYPI_EAI"], session=session)
            def test_compute_pool():
                return "Success"

            logger = logging.getLogger("snowflake.ml.jobs.job")
            logger.addHandler(ComputePoolBusyHandler())
            ml_job_test = test_compute_pool()
            ml_job_test.get_logs()
            ml_job_test_result = ml_job_test.result()
            if ml_job_test_result == 'Success':
                print(
                    f"Successfully ran test job on compute pool {compute_pool}")
            else:
                raise SnowflakeResourcesError(
                    f"Failed to run test job on compute pool {compute_pool}, job result: {ml_job_test_result}")

        except Exception as e:
            raise SnowflakeResourcesError(
                f"Failed to run test job on compute pool {compute_pool}, exception: {e}")

    # If the input dataset Snowflake connection doesn't have a default schema, pull the schema name from the input dataset settings
    connection_schema = session.get_current_schema()
    if not connection_schema:
        input_dataset_info = input_dataset.get_location_info()
        try:
            input_dataset_schema = input_dataset_info['info']['schema']
            session.use_schema(input_dataset_schema)
        except:
            raise PluginParamValidationError(
                f"Input dataset: {input_dataset_names[0]} has no schema. Please make sure it exists.")

    # Convert the input dataset into a Snowpark dataframe (we will return this df in the function outputs)
    input_snowpark_df = dku_snowpark.get_dataframe(
        input_dataset, session=session)

    # Snowflake Model Registry
    params.deploy_to_snowflake_model_registry = recipe_config.get(
        'deploy_to_snowflake_model_registry', False)

    # Selected Input Feature Columns
    params.inputDatasetColumns = recipe_config.get('inputDatasetColumns', None)
    params.selectedInputColumns = recipe_config.get(
        'selectedInputColumns', None)
    params.selectedOption1 = recipe_config.get('selectedOption1', None)
    params.selectedOption2 = recipe_config.get('selectedOption2', None)
    params.selectedConstantImpute = recipe_config.get(
        'selectedConstantImpute', None)

    if not params.selectedInputColumns:
        raise PluginParamValidationError(
            "No input features selected. Choose some features to include in the model")
    else:
        # Need to remove columns that were checked and then unchecked at run time - the dict values will be False
        for input_col, selected in list(params.selectedInputColumns.items()):
            if not selected:
                del params.selectedInputColumns[input_col]
        # Check if any features selected post check/uncheck pruning
        if len(params.selectedInputColumns) == 0:
            raise PluginParamValidationError(
                "No input features selected. Choose some features to include in the model")

        # Iterate through all selected input columns and check that an encoding/rescaling and missingness imputation method is chosen
        for selected_input_col in params.selectedInputColumns.keys():
            if selected_input_col not in params.selectedOption1.keys():
                raise PluginParamValidationError(
                    f"No Encoding / Rescaling option selected for input feature: {selected_input_col}. Choose an Encoding / Rescaling method")
            if selected_input_col not in params.selectedOption2.keys():
                raise PluginParamValidationError(
                    f"No 'Impute Missing Values With' option selected for input feature: {selected_input_col}. Choose an 'Impute Missing Values With' method")
            # If constant imputation selected, make sure a constant value was given
            if params.selectedOption2[selected_input_col] == 'Constant' and not params.selectedConstantImpute:
                raise PluginParamValidationError(
                    f"Constant imputation selected for input feature: {selected_input_col}, but no value chosen. Choose a value")
            elif params.selectedOption2[selected_input_col] == 'Constant' and selected_input_col not in params.selectedConstantImpute.keys():
                raise PluginParamValidationError(
                    f"Constant imputation selected for input feature: {selected_input_col}, but no value chosen. Choose a value")

    # Algorithm hyperparameter configurations
    # Each tuple is (param_suffix, display_name)
    RF_PARAMS = [('n_estimators', 'Number of Trees'), ('max_depth',
                                                       'Max Depth'), ('min_samples_leaf', 'Min Samples per Leaf')]
    XGB_LGBM_PARAMS = [('n_estimators', 'Number of Trees'), ('max_depth', 'Max Depth'),
                       ('min_child_weight', 'Min Child Weight'), ('learning_rate', 'Learning Rate')]
    GB_PARAMS = [('n_estimators', 'Number of Trees'), ('max_depth', 'Max Depth'),
                 ('min_samples_leaf', 'Min Samples per Leaf'), ('learning_rate', 'Learning Rate')]
    DT_PARAMS = [('max_depth', 'Max Depth'),
                 ('min_samples_leaf', 'Min Samples per Leaf')]

    # Count number of algorithms selected
    class_algos_selected = 0
    reg_algos_selected = 0

    # Classification algorithms
    if load_and_validate_algorithm_params(params, recipe_config, 'logistic_regression', [('c', 'C')], 'Logistic Regression'):
        class_algos_selected += 1

    if load_and_validate_algorithm_params(params, recipe_config, 'random_forest_classification', RF_PARAMS, 'Random Forest'):
        class_algos_selected += 1

    if load_and_validate_algorithm_params(params, recipe_config, 'xgb_classification', XGB_LGBM_PARAMS, 'XGBoost'):
        class_algos_selected += 1

    if load_and_validate_algorithm_params(params, recipe_config, 'lgbm_classification', XGB_LGBM_PARAMS, 'LightGBM'):
        class_algos_selected += 1

    if load_and_validate_algorithm_params(params, recipe_config, 'gb_classification', GB_PARAMS, 'Gradient Tree Boosting'):
        class_algos_selected += 1

    if load_and_validate_algorithm_params(params, recipe_config, 'decision_tree_classification', DT_PARAMS, 'Decision Tree'):
        class_algos_selected += 1

    # Regression algorithms
    if load_and_validate_algorithm_params(params, recipe_config, 'lasso_regression', [('alpha', 'Alpha')], 'Lasso Regression'):
        reg_algos_selected += 1

    if load_and_validate_algorithm_params(params, recipe_config, 'random_forest_regression', RF_PARAMS, 'Random Forest'):
        reg_algos_selected += 1

    if load_and_validate_algorithm_params(params, recipe_config, 'xgb_regression', XGB_LGBM_PARAMS, 'XGBoost'):
        reg_algos_selected += 1

    if load_and_validate_algorithm_params(params, recipe_config, 'lgbm_regression', XGB_LGBM_PARAMS, 'LightGBM'):
        reg_algos_selected += 1

    if load_and_validate_algorithm_params(params, recipe_config, 'gb_regression', GB_PARAMS, 'Gradient Tree Boosting'):
        reg_algos_selected += 1

    if load_and_validate_algorithm_params(params, recipe_config, 'decision_tree_regression', DT_PARAMS, 'Decision Tree'):
        reg_algos_selected += 1

    # If no algorithms selected, raise an error
    if (params.prediction_type == "two-class classification" or params.prediction_type == "multi-class classification") and class_algos_selected == 0:
        raise PluginParamValidationError(
            "You didn't select any algorithms. Please select at least one algorithm.")
    elif params.prediction_type == "regression" and reg_algos_selected == 0:
        raise PluginParamValidationError(
            "You didn't select any algorithms. Please select at least one algorithm.")

    # Search Space Limit
    n_iter = recipe_config.get('n_iter', None)
    if not n_iter:
        raise PluginParamValidationError(
            "No search space limit chosen. Choose a search space limit that is an integer (e.g. 4)")
    elif isinstance(random_seed, int):
        params.n_iter = n_iter
    else:
        raise PluginParamValidationError(
            f"Search space limit: {n_iter} is not an integer. Choose a search space limit that is an integer (e.g. 4)")

    # Check that a code env named "py_310_snowpark" exists
    client = dataiku.api_client()
    code_envs = [env["envName"] for env in client.list_code_envs()]
    if "py_310_snowpark" not in code_envs:
        raise CodeEnvSetupError(
            "You must create a python 3.10 code env named 'py_310_snowpark' with the packages listed here: https://github.com/dataiku/dss-plugin-visual-snowparkml")

    # Check that if user selected two-class or multi-class classification, that they converted the target column to numeric (0,1) - a current SnowML requirement
    if (prediction_type == "two-class classification" or prediction_type == "multi-class classification"):
        if input_dataset_column_types[col_label] not in ['int', 'bigint', 'smallint', 'tinyint', 'float', 'double']:
            raise InputTrainDatasetSetupError(
                f"Target column: {col_label} is of type: {input_dataset_column_types[col_label]}. When choosing two-class or multi-class classification, you must first convert the target column to one of type: int, bigint, smallint, tinyint, float, or double (e.g. (0, 1), (0.0, 1.0), (1, 2, 3, 4, 5))")

    return params, session, input_snowpark_df


def load_score_config_snowpark_session() -> Tuple[ScorePluginParams, Session]:
    """Utility function to:
        - Validate and load ml training parameters into a clean class

    Returns:
        - Class instance with parameter names as attributes and associated values
        - Snowpark session
        - Input training dataset as a Snowpark dataframe
    """

    params = ScorePluginParams()
    # Get input and output datasets
    input_dataset_names = get_input_names_for_role("input_dataset_name")
    if len(input_dataset_names) != 1:
        raise PluginParamValidationError("Please specify one input dataset")
    input_dataset = dataiku.Dataset(input_dataset_names[0])

    output_score_dataset_names = get_output_names_for_role(
        'output_score_dataset_name')
    if len(output_score_dataset_names) != 1:
        raise PluginParamValidationError(
            "Please specify one output scored dataset")
    else:
        output_score_dataset = dataiku.Dataset(output_score_dataset_names[0])

    # Get the input Dataiku Saved Model name
    saved_model_names = get_input_names_for_role('saved_model_name')
    if len(saved_model_names) != 1:
        raise PluginParamValidationError(
            "Please specify one input saved model (that has been trained with the Visual Snowpark ML train plugin recipe)")
    else:
        saved_model_id = saved_model_names[0].split(".")[1]

    # Check that there's an active version of the input model
    client = dataiku.api_client()
    project = client.get_default_project()
    project_key = project.project_key
    saved_model = project.get_saved_model(saved_model_id)
    active_model_version = saved_model.get_active_version()
    if not active_model_version:
        raise PluginParamValidationError(
            "There's something wrong with the input model. Please check that there is an active version that was trained with the Visual Snowpark ML train plugin recipe")

    # Recipe parameters
    recipe_config = get_recipe_config()

    # Snowflake Warehouse
    dku_snowpark = DkuSnowpark()
    snowflake_connection_name = input_dataset.get_config()[
        'params']['connection']
    session = dku_snowpark.create_session(
        snowflake_connection_name, project_key=project_key)

    params.warehouse = recipe_config.get('warehouse', None)
    if params.warehouse:
        warehouse = f'"{params.warehouse}"'
        try:
            session.use_warehouse(warehouse)
            params.warehouse = warehouse
        except:
            raise PluginParamValidationError(
                f"Snowflake Warehouse: {warehouse} does not exist or you do not have permission to use it")

    # If the input dataset Snowflake connection doesn't have a default schema, pull the schema name from the input dataset settings
    connection_schema = session.get_current_schema()
    if not connection_schema:
        input_dataset_info = input_dataset.get_location_info()
        input_dataset_schema = input_dataset_info['info']['schema']
        session.use_schema(input_dataset_schema)

    # Check that a code env named "py_310_snowpark" exists
    code_envs = [env["envName"] for env in client.list_code_envs()]
    if "py_310_snowpark" not in code_envs:
        raise CodeEnvSetupError(
            "You must create a python 3.10 code env named 'py_310_snowpark' with the packages listed here: https://github.com/dataiku/dss-plugin-visual-snowparkml")

    return params, session, input_dataset, saved_model_id, output_score_dataset
