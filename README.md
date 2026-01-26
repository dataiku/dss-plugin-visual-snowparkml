# Visual Snowflake ML Plugin

<img width="658" height="291" alt="visual_snowflake_ml_diagram" src="https://github.com/user-attachments/assets/287396d5-7c6d-4b62-821a-7139c518d39b" />

With this plugin, you can train machine learning models and then use them to score new records, all within your Snowflake environment. This no-code UI allows data scientists and domain experts to quickly train models, track experiments, and visualize model performance.

# Capabilities

- No-code ML model training using Snowflake compute, either using a Snowpark optimized warehouse with Snowflake’s [snowflake-ml-python](https://pypi.org/project/snowflake-ml-python/) package, or using a Snowpark Container Services compute pool with [Snowflake ML Jobs](https://docs.snowflake.com/en/developer-guide/snowflake-ml/ml-jobs/overview#)
    - Two-class classification, multi-class classification, and regression tasks on tabular data stored in a Snowflake table
    - Hyperparameter tuning using Random Search on certain algorithm parameters
    - Track hyperparameter tuning and model performance through Dataiku’s Experiment Tracking MLflow integration
    - Output the best trained model to a Dataiku Saved Model in the flow, and deploy the model to a Snowflake Model Registry
- No-code ML batch scoring in Snowflake using a trained model (from the training recipe of this plugin)
- Macro to clean up models from the Snowflake Model Registry that have been deleted from the Dataiku project

# Limitations

- If doing two-class or multi-class classification, convert your target column to numeric (0,1) or (0, 1, 2, 3, 4) before using this plugin (this is a SnowparkML requirement)
- No int type columns can have missing values. If you have an int column with missing values, convert the type to double before this recipe (this is an MLflow requirement)
- If you want to treat a numeric column as categorical, change its storage type to string in a prior recipe

# Snowflake Resources and Permissions

- Must have a Snowflake connection. Plugin recipe Input + Output tables must be in the same Snowflake connection.
- The plugin uses the Snowflake role used for the Input + Output tables to access all other Snowflake resources
- The plugin uses a backend runtime environment of a Snowpark Container Services compute pool or Snowpark-optimized warehouse
    - Snowpark Container Services Compute Pool: Snowflake role must have the USAGE permission on the compute pool
    - Snowpark-optimized Warehouse: Snowflake role must have the USAGE permission on the warehouse
- Snowflake role must have the CREATE MODEL privilege on the schema used in the Snowflake connection for Input + Output tables

# Other Requirements

- Python 3.10 available on the instance

# Setup
## Build the plugin code environment
Right after installing the plugin, you will need to build its code environment. Note that this plugin requires Python version 3.10 and that conda is not supported.

## Build ANOTHER python 3.10 code environment
Name it “py_310_snowpark”
Under “Core packages versions”, choose Pandas 2.3
Add these packages, then update the environment:
```
snowflake-ml-python==1.20.0
numpy==1.26.4
mlflow==2.18.0
scikit-learn==1.7.2
xgboost==3.1.2
lightgbm==4.6.0
statsmodels==0.14.6
```

## 

# Usage
## Training models with Snowflake ML
### Create the train plugin recipe and outputs
Click once on the input dataset (with known, labeled target values), then find the Visual Snowflake ML plugin: 

<img width="965" height="427" alt="image" src="https://github.com/user-attachments/assets/49d5bb6d-d3f5-4a53-b7df-8847c8ceefd5" />

Click the train recipe:

<img width="797" height="381" alt="image" src="https://github.com/user-attachments/assets/d0c18e2a-67a4-4f03-8863-58e986ec6364" />

Create two output Snowflake tables to hold the generated output train/test sets, and one managed folder to hold saved models (connection doesn’t matter):

![image](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/1bf50274-8308-4d5f-8bb1-82bdbadb46b8)


### Design your ML training process and run the recipe
Make sure you fill out all required fields

**Target**
- Prediction type: two-class classification, multi-class classification, or regression
- Target: the name of your target column to predict
- Class weights: choose to enable (recommended) or disable class weights. Class weights are row weights that are inversely proportional to the cardinality of its target class and help with class imbalance issues
- Model name: the name of your model. This will be the name of the model created in your Dataiku project flow after running the train recipe. The best trained model will be registered in Snowflake model registry as DATAIKU_PROJECT_ID_MODEL_NAME.

**Train/Test Set**
- Time ordering: order the train and test sets by a datetime column (test set will be more recent timestamps than train set)
- Train ratio: train set / test set ratio. 0.8 is a good start
- Splitting random seed: Set this (to any integer) to maintain consistent train/test sets over multiple training runs

**Metrics**
- Optimize model hyperparameters for: metric to optimize model hyperparameters for while training

**Features handling**
For each column in the input training dataset, choose whether to include the column as an input features for the model training process, and how to handle this feature. Note: if you want to treat a numeric column as categorical, change its storage type to string in a prior recipe.
- Status: whether to include or exclude the feature
- Encoding / Rescaling: choose how to encode categorical features, and rescale numeric features
- Missing values: choose how to deal with missing values
- Constant: if “Constant” chosen for missingness imputation, the value to impute

**Algorithms**
- Select each algorithm you’d like to train
- For each algorithm, enter min and max values for each hyperparameter

**Hyperparameters**
This training recipe will kick off a Randomized Search process with 3-fold cross-validation in Snowflake to find the best hyperparameter combination for each algorithm selected
- Search space limit: the number of hyperparameter combinations to try for each algorithm within the min/max values chosen

**Runtime environment**
- Compute Backend: whether to use Snowflake Container Runtime (Snowpark Container Services) or a Snowflake warehouse for this ML training job
- (If Container Runtime) Compute Pool: the Snowpark Container Services compute pool to use for ML training. 
- (If Container Runtime) Snowflake Stage: the Snowflake Stage where model training functions will be uploaded prior to execution on the compute pool. 
- (If Warehouse) Snowflake Warehouse: the warehouse to use for ML training. You must use a Snowpark-optimized Snowflake warehouse. A multi-cluster warehouse will allow for parallelized hyperparameter tuning.
- Model Registry: deploy the best trained model to a Snowflake ML Model Registry (in the same database and schema as the input and output datasets. See Snowflake access requirements [here](https://docs.snowflake.com/en/developer-guide/snowpark-ml/snowpark-ml-mlops-model-registry#required-privileges). This is required in order to run a subsequent Visual Snowpark ML Score recipe, to run batch inference in Snowpark using the deployed model.

### Outputs
After running the train recipe successfully, you can find all model training and hyperparameter tuning information, including model performance metrics in Dataiku’s Experiment Tracking tab

![image](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/c48db527-d004-48dc-aab9-11d3570f3a21)

![Screenshot 2024-02-23 at 9 11 07 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/7e4d139e-a8e8-45a3-9bf6-10c9062478c8)

The best model will be deployed to the flow. If you selected “Deploy to Snowflake ML Model Registry”, the model will also be deployed to Snowflake’s Model Registry. 

<img width="443" alt="Screenshot 2024-05-15 at 8 09 47 AM" src="https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/5e7837da-60db-4814-afc6-6389fef71100">

## Scoring New Records with your Trained Model and Snowpark ML
**Note:** you can use a regular Dataiku Score recipe with the Snowpark ML trained model, however, the inference will happen in a local python kernel, and not in Snowflake. Note that for classification models where you did NOT disable class weights, you'll need to add a SAMPLE_WEIGHTS column in your input dataset before a regular Dataiku Score recipe (this column can have all empty values).

In order to run batch inference in Snowpark, use this plugin's Visual Snowpark ML Score recipe. You must have checked “Deploy to Snowflake ML Model Registry” when training the model for this to work.

### Create the score plugin recipe and outputs
Click once on the trained model and input dataset you’d like to make predictions for: 

![image](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/23a3a321-4832-414f-8347-50f45b2285e7)

Click the score recipe:

<img width="797" height="381" alt="image" src="https://github.com/user-attachments/assets/73714334-8356-4bba-90b9-c946e49c806d" />

Make sure you’ve selected the trained model and Snowflake table for scoring as inputs. Then create one output Snowflake table to hold the scored dataset. Then click “Create”:

![image](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/f9625a12-0bd2-471c-9151-dfb8675df4fa)

Optionally type the name of a Snowpark-optimized Snowflake warehouse to use for scoring. Leave empty to use the Snowflake connection’s default warehouse. Click “Run”.

![Screenshot 2024-02-23 at 9 30 05 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/c223ff1b-e054-4c35-981d-bd2066aa0515)

Your flow should look like this, and the output scored dataset should have prediction column(s):

<img width="698" alt="Screenshot 2024-05-14 at 4 21 17 PM" src="https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/a93839ac-490d-4bc8-bc6d-03ca841c80a8">

![Screenshot 2024-02-23 at 9 31 46 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/a5bf996b-cacd-463e-a807-1a0ee074e5c3)


## Clear SnowparkML Registry Models Macro 
When deploying trained models to a Snowflake Model Registry, we want to ensure that any trained models that we delete from the Dataiku UI (by deleting a full green diamond saved model, or a saved model version underneath), are also deleted from the Snowflake Model Registry. 

This macro checks for any models in the Snowflake Model Registry that have the Dataiku and the current project key tags; if the model has been deleted from the Dataiku project, this macro deletes the model from the Snowflake Model Registry.

You can list the models the macro would delete by un-checking the “Perform deletion” box.

The macro will show the model name and version deleted (or simulated) in the resulting list.

<img width="1166" alt="image" src="https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/0e5594ec-8bd2-41a2-8aa8-aba3386240d0">

<img width="631" alt="image" src="https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/7b857a57-fc0d-4408-8ed0-70598ac36e66">

<img width="313" alt="Screenshot 2024-02-23 at 12 51 50 PM" src="https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/7e2cd3bd-0561-424a-a68d-50608f3ed770">

<img width="893" alt="Screenshot 2024-02-23 at 12 52 30 PM" src="https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/d27a4ec8-9a52-430b-9c06-936ede71f33c">


# Release Notes

See the [changelog](CHANGELOG.md) for a history of notable changes to this plugin.

# License

This plugin is distributed under the [Apache License version 2.0](LICENSE).
