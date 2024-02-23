# Visual Snowpark ML Plugin

![image](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/4227975a-08aa-44af-8b3f-671dd29e4e5d)

With this plugin, you can train machine learning models and then use them to score new records, all within your Snowflake environment. This no-code UI allows data scientists and domain experts to quickly train models, track experiments, and visualize model performance.

# Capabilities

- No-code ML model training on Snowpark using Snowflake’s [snowflake-ml-python](https://pypi.org/project/snowflake-ml-python/) package
    - Two-class classification and regression tasks on tabular data stored in a Snowflake table
    - Hyperparameter tuning using Random Search on certain algorithm parameters
    - Track hyperparameter tuning and model performance through Dataiku’s Experiment Tracking MLflow integration
    - Output the best trained model to a Dataiku Saved Model in the flow, and deploy the model to a Snowflake Model Registry
- No-code ML batch scoring in Snowpark using a trained model (from the training recipe of this plugin)
- Macro to clean up models from the Snowflake Model Registry that have been deleted from the Dataiku project

# Limitations

- If doing two-class classification + XGBoost, convert your target column to numeric (0,1) before using this plugin (this is an XGBoost requirement)
- No int type columns can have missing values. If you have an int column with missing values, convert the type to double before this recipe (this is an MLflow requirement)
- If you want to treat a numeric column as categorical, change its storage type to string in a prior recipe

# Other Requirements

- Must have a Snowflake connection. Plugin recipe Input + Output tables should be in the same Snowflake connection
- Python 3.8 available on the instance
- (Strongly recommended) Snowpark-optimized Snowflake warehouse available for model training. Multi-cluster warehouse will allow for parallelized hyperparameter tuning

# Setup
## Build the plugin code environment
Right after installing the plugin, you will need to build its code environment. Note that this plugin requires Python version 3.8 and that conda is not supported.

## Build ANOTHER python 3.8 code environment
Name it “py_38_snowpark”
Under “Core packages versions”, choose Pandas 1.3
Add these packages, then update the environment:
```
scikit-learn==1.2.1
mlflow==2.5.0
mlflow[extras]==2.5.0
statsmodels==0.12.2
protobuf==3.16.0
xgboost==1.7.3
lightgbm==3.3.5
matplotlib==3.7.1
scipy==1.10.1
snowflake-snowpark-python==1.12.0
snowflake-snowpark-python[pandas]==1.12.0
snowflake-connector-python[pandas]==3.7.0
MarkupSafe==2.0.1
cloudpickle==2.0.0
flask==1.0.4
Jinja2==2.11.3
snowflake-ml-python==1.2.2
```

# Usage
## Training models with Snowpark ML
### Create the train plugin recipe and outputs
Click once on the input dataset (with known, labeled target values), then find the Visual SnowparkML plugin: 

![image](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/0a6b44f0-28eb-430a-a5cf-95a301cc67c3)

Click the train recipe:

![image](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/6a58dc0c-b0ee-4c85-a742-a2cd45c58abc)

Create two output Snowflake tables to hold the generated output train/test sets, and one managed folder to hold saved models (connection doesn’t matter):

![image](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/1bf50274-8308-4d5f-8bb1-82bdbadb46b8)

### Design your ML training process and run the recipe
Make sure you fill out all required fields
Below is an example for a two-class classification problem:
**Parameters**
- Final model name: the name of your model
- Target column: the name of your target column to predict
- Prediction type: either two-class classification or regression
- Disable class weights: choose to disable class weights (not recommended). Class weights are row weights that are inversely proportional to the cardinality of its target class and help with class imbalance issues
- Train ratio: train set / test set ratio. 0.8 is a good start
- Random seed: Set this (to any integer) to maintain consistent train/test sets over multiple training runs
- Enable time ordering: order the train and test sets by a datetime column (test set will be more recent timestamps than train set)
- Metrics: metric to optimize model performance

![Screenshot 2024-02-23 at 8 58 32 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/b0f627d0-b6c4-4974-ac6e-7312ad9be2e0)

**Features selection**
When you include a feature, don't leave the "Encoding / Rescaling" and "Impute Missing Values With" dropdowns empty!
- Type: underlying storage type. Note: if you want to treat a numeric column as categorical, change its storage type to string in a prior recipe
- Include: whether to include the column in the model
- Encoding / Rescaling: choose how to encode categorical features, and rescale numeric features.
- Impute Missing Values With: choose how to deal with missing values
- Constant Value (Impute): if “Constant” chosen for missingness imputation, the value to impute

![Screenshot 2024-02-23 at 9 00 12 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/b28204ce-acd4-4b43-8855-0a052a54c63f)

**Algorithms**
- Select each algorithm you’d like to train
- For each algorithm, enter min and max values for each hyperparameter 
- Enter a search space limit (randomly choose N hyperparameter combinations for each algorithm within the min/max values chosen) 
- A Snowpark ML Randomized Search with 3-fold cross-validation will kick off in Snowflake to find the best hyperparameter combination for each algorithm

![Screenshot 2024-02-23 at 9 00 51 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/f35bfe27-7ff6-47ba-838f-f5186275adb0)

![Screenshot 2024-02-23 at 9 01 36 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/aaaef879-3b30-4b29-9073-19e1621ebe20)

**Snowflake Resources and Deployment**
- Snowflake Warehouse: the warehouse to use for ML training. Strongly recommended to use a Snowpark-optimized Snowflake warehouse. A multi-cluster warehouse will allow for parallelized hyperparameter tuning.
- Deploy to Snowflake ML Model Registry: deploy the best trained model to a Snowflake ML Model Registry (in the MODEL_REGISTRY database. See Snowflake access requirements [here](https://docs.snowflake.com/LIMITEDACCESS/snowflake-ml-model-registry#required-privileges)). This is required in order to run a subsequent Visual Snowpark ML Score recipe, to run batch inference in Snowpark using the deployed model.

![Screenshot 2024-02-23 at 9 01 46 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/3c29084d-3a6e-455b-95c7-9c963d92b90e)

### Outputs
After running the train recipe successfully, you can find all model training and hyperparameter tuning information, including model performance metrics in Dataiku’s Experiment Tracking tab

![image](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/c48db527-d004-48dc-aab9-11d3570f3a21)

![Screenshot 2024-02-23 at 9 11 07 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/7e4d139e-a8e8-45a3-9bf6-10c9062478c8)

The best model will be deployed to the flow. If you selected “Deploy to Snowflake ML Model Registry”, the model will also be deployed to Snowflake’s Model Registry. 

![Screenshot 2024-02-23 at 9 12 34 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/4b21f000-bb76-4061-85c5-f4b07fdd6923)

## Scoring New Records with your Trained Model and Snowpark ML
**Note:** you can use a regular Dataiku Score recipe with the Snowpark ML trained model, however, the inference will happen in a local python kernel, and not in Snowflake. Note that for classification models where you did NOT disable class weights, you'll need to add a SAMPLE_WEIGHTS column in your input dataset before a regular Dataiku Score recipe (this column can have all empty values).

In order to run batch inference in Snowpark, use this plugin's Visual Snowpark ML Score recipe. You must have checked “Deploy to Snowflake ML Model Registry” when training the model for this to work.

### Create the score plugin recipe and outputs
Click once on the trained model and input dataset you’d like to make predictions for: 

![image](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/23a3a321-4832-414f-8347-50f45b2285e7)

Click the score recipe:

![image](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/a744b6eb-ecdf-4ef5-9e44-bbe364bea2f1)

Make sure you’ve selected the trained model and Snowflake table for scoring as inputs. Then create one output Snowflake table to hold the scored dataset. Then click “Create”:

![image](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/f9625a12-0bd2-471c-9151-dfb8675df4fa)

Optionally type the name of a Snowpark-optimized Snowflake warehouse to use for scoring. Leave empty to use the Snowflake connection’s default warehouse. Note: it’s less needed to use a Snowpark-optimized warehouse for inference compared to training. Click “Run”.

![Screenshot 2024-02-23 at 9 30 05 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/c223ff1b-e054-4c35-981d-bd2066aa0515)

Your flow should look like this, and the output scored dataset should have prediction column(s):

![Screenshot 2024-02-23 at 9 31 18 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/7c2a503d-775a-428c-83bf-130d8a1f5ae0)

![Screenshot 2024-02-23 at 9 31 46 AM](https://github.com/dataiku/dss-plugin-visual-snowparkml/assets/22987725/a5bf996b-cacd-463e-a807-1a0ee074e5c3)


## Clear SnowparkML Registry Models Macro 
When deploying trained models to a Snowflake Model Registry, we want to ensure that any trained models that we delete from the Dataiku UI (by deleting a full green diamond saved model, or a saved model version underneath), are also deleted from the Snowflake Model Registry. 

This macro checks for any models in the Snowflake Model Registry that have the Dataiku and the current project key tags; if the model has been deleted from the Dataiku project, this macro deletes the model from the Snowflake Model Registry.

You can list the models the macro would delete by un-checking the “Perform deletion” box.

The macro will show the model name and version deleted (or simulated) in the resulting list

# Release Notes

See the [changelog](CHANGELOG.md) for a history of notable changes to this plugin.

# License

This plugin is distributed under the [Apache License version 2.0](LICENSE).
