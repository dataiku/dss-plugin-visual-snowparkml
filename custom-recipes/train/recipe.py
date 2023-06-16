# Code for custom code recipe train (imported from a Python recipe)

# To finish creating your custom recipe from your original PySpark recipe, you need to:
#  - Declare the input and output roles in recipe.json
#  - Replace the dataset names by roles access in your code
#  - Declare, if any, the params of your custom recipe in recipe.json
#  - Replace the hardcoded params values by acccess to the configuration map

# See sample code below for how to do that.
# The code of your original recipe is included afterwards for convenience.
# Please also see the "recipe.json" file for more information.

# import the classes for accessing DSS objects from the recipe
import dataiku
# Import the helpers for custom recipes
from dataiku.customrecipe import get_input_names_for_role
from dataiku.customrecipe import get_output_names_for_role
from dataiku.customrecipe import get_recipe_config

# Inputs and outputs are defined by roles. In the recipe's I/O tab, the user can associate one
# or more dataset to each input and output role.
# Roles need to be defined in recipe.json, in the inputRoles and outputRoles fields.

# To  retrieve the datasets of an input role named 'input_A' as an array of dataset names:
input_A_names = get_input_names_for_role('input_A_role')
# The dataset objects themselves can then be created like this:
input_A_datasets = [dataiku.Dataset(name) for name in input_A_names]

# For outputs, the process is the same:
output_A_names = get_output_names_for_role('main_output')
output_A_datasets = [dataiku.Dataset(name) for name in output_A_names]


# The configuration consists of the parameters set up by the user in the recipe Settings tab.

# Parameters must be added to the recipe.json file so that DSS can prompt the user for values in
# the Settings tab of the recipe. The field "params" holds a list of all the params for wich the
# user will be prompted for values.

# The configuration is simply a map of parameters, and retrieving the value of one of them is simply:
my_variable = get_recipe_config()['parameter_name']

# For optional parameters, you should provide a default value in case the parameter is not present:
my_variable = get_recipe_config().get('parameter_name', None)

# Note about typing:
# The configuration of the recipe is passed through a JSON object
# As such, INT parameters of the recipe are received in the get_recipe_config() dict as a Python float.
# If you absolutely require a Python int, use int(get_recipe_config()["my_int_param"])


#############################
# Your original recipe
#############################

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from dataiku.snowpark import DkuSnowpark
from snowflake.snowpark.functions import udf
import snowflake.snowpark
from snowflake.snowpark.functions import sproc
from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.functions import col

# Create the DSS wrapper around Snowpark
dku_snowpark = DkuSnowpark()

# Read inputs
loans_ds = dataiku.Dataset("LOAN_REQUESTS_CUST_INFO_JOINED")
loans_snowpark_df = dku_snowpark.get_dataframe(loans_ds)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#session = dku_snowpark.get_session("snowflake")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#session.sql("CREATE OR REPLACE STAGE FEATURE_ENGINEERING;").collect()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
@udf(name="divide_by_two",
     is_permanent=False,
     stage_location="@FEATURE_ENGINEERING",
     replace=True)
def divide_by_two(NUM: float) -> float:

    return NUM / 2.0

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
loans_snowpark_df = loans_snowpark_df.withColumn('"FICO_MED_DIV_TWO"', divide_by_two(col('"FICO_MED"')))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write outputs
output_dataset = dataiku.Dataset("LOAN_REQUESTS_FEATURES")
dku_snowpark.write_with_schema(output_dataset, loans_snowpark_df)