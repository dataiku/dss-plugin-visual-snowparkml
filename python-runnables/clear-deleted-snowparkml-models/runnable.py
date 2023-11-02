import dataiku
from dataiku.runnables import Runnable
from dataikuapi.utils import DataikuException
from dataiku.base.utils import safe_unicode_str

from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import json

from dataiku.snowpark import DkuSnowpark
from snowflake.ml.registry import model_registry
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from snowflake.snowpark.session import Session

class MyRunnable(Runnable):
    """The base interface for a Python runnable"""

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        super(MyRunnable, self).__init__(project_key, config, plugin_config)
        self.config = config
        self.client = dataiku.api_client()
        self.perform_deletion = self.config.get("perform_deletion", False)
        self.snowflake_connection_name = self.config.get("snowflake_connection", None)
        self.project = dataiku.api_client().get_project(project_key)
        
        
    def get_progress_target(self):
        return 100, 'NONE'

    def run(self, progress_callback):        
        # If user chooses a non-Snowflake connection, return an error message
        if self.client.get_connection(self.snowflake_connection_name).get_info()['type'] != 'Snowflake':
            return 'Please select a Snowflake connection'
        
        # Check the 'MODEL_REGISTRY' Snowflake database for models
        snowflake_model_registry = "MODEL_REGISTRY"
        
        # Get a Snowpark session
        dku_snowpark = DkuSnowpark()
        session = dku_snowpark.get_session(self.snowflake_connection_name)
        
        # Get the Snowflake Model Registry
        registry = model_registry.ModelRegistry(session = session, database_name = snowflake_model_registry)
        
        # List models in the registry
        registry_models = registry.list_models().to_pandas()
        
        # List Dataiku Saved Models in the current project
        dataiku_saved_model_ids = [model['id'] for model in self.project.list_saved_models()]
        
        # Create dictionary with key: Dataiku Saved Model IDs and values: Saved Model Versions
        dataiku_saved_model_ids_and_versions = {}

        for saved_model_id in dataiku_saved_model_ids:
            saved_model = self.project.get_saved_model(saved_model_id)
            saved_model_versions = [version['id'] for version in saved_model.list_versions()]
            dataiku_saved_model_ids_and_versions[saved_model_id] = saved_model_versions
        
        # Get tags from Snowflake Model Registry models
        registry_models['TAGS'] = registry_models['TAGS'].apply(json.loads)

        models_to_delete = []
        
        # Loop through all models in Snowflake Model Registry. If the model has the current Dataiku project key as a tag, and
        # if the model doesn't exist as a Dataiku Saved Model or version (meaning it was delete from Dataiku side), then 
        # simulate its deletion or actually delete it, depending on what the user selected
        for i, registry_model in registry_models.iterrows():
            try:
                if registry_model['TAGS']['dataiku_project_key'] == self.project.project_key:
                    registry_dataiku_saved_model_id = registry_model['TAGS']['dataiku_saved_model_id']
                    if registry_dataiku_saved_model_id not in dataiku_saved_model_ids:

                        models_to_delete.append({
                            "name": registry_model['NAME'],
                            "version": registry_model['VERSION']
                        })
                        if self.perform_deletion:
                            registry.delete_model(model_name = registry_model['NAME'],
                                                  model_version = registry_model['VERSION'])
                    elif registry_model['VERSION'] not in dataiku_saved_model_ids_and_versions[registry_dataiku_saved_model_id]:

                        models_to_delete.append({
                            "name": registry_model['NAME'],
                            "version": registry_model['VERSION']
                        })
                        if self.perform_deletion:
                            registry.delete_model(model_name = registry_model['NAME'],
                                                  model_version = registry_model['VERSION'])
                    else:
                        continue
                else:
                    continue
            except:
                continue
        
        # Return HTML with a table of deleted models from the Snowflake Model Registry (or simulated deletions)
        if self.perform_deletion:
            html = "<h4>Models Deleted</h4>"
        else:
            html = "<h4>Models to Delete (Simulation)</h4>"

        models_to_delete_df = pd.DataFrame(models_to_delete)
        if len(models_to_delete_df) == 0:
            html += "<span>No models to delete</span>"
        else:
            html += models_to_delete_df.to_html(justify="center")
        
        return html