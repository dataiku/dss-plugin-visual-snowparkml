import dataiku
from dataiku.runnables import Runnable
from dataiku.snowpark import DkuSnowpark

import pandas as pd
import json

from snowflake.ml.registry import Registry


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
        self.snowflake_connection_name = self.config.get(
            "snowflake_connection", None)
        self.project = dataiku.api_client().get_project(project_key)

    def get_progress_target(self):
        return 100, 'NONE'

    def run(self, progress_callback):
        # If user chooses a non-Snowflake connection, return an error message
        if self.client.get_connection(self.snowflake_connection_name).get_info()['type'] != 'Snowflake':
            return 'Please select a Snowflake connection'

        # Get a Snowpark session
        dku_snowpark = DkuSnowpark()
        session = dku_snowpark.create_session(
            self.snowflake_connection_name, project_key=self.project.project_key)
        current_database = session.get_current_database().replace('"', '')
        current_schema = session.get_current_schema().replace('"', '')

        # Get the Snowflake Model Registry
        registry = Registry(session=session)

        # List models in the registry
        registry_models = registry.models()

        # List Dataiku Saved Models in the current project
        dataiku_saved_model_ids = [model['id']
                                   for model in self.project.list_saved_models()]

        # Create dictionary with key: Dataiku Saved Model IDs and values: Saved Model Versions
        dataiku_saved_model_ids_and_versions = {}

        for saved_model_id in dataiku_saved_model_ids:
            saved_model = self.project.get_saved_model(saved_model_id)
            saved_model_versions = [version['id']
                                    for version in saved_model.list_versions()]
            dataiku_saved_model_ids_and_versions[saved_model_id] = saved_model_versions

        dataiku_saved_model_ids_and_versions_upper = {}

        for k, v in dataiku_saved_model_ids_and_versions.items():
            model_version_list_upper = [model_ver.upper() for model_ver in v]
            dataiku_saved_model_ids_and_versions_upper[k] = model_version_list_upper

        models_to_delete = []

        # Loop through all models in Snowflake Model Registry. If the model has the current Dataiku project key as a tag, and
        # if the model doesn't exist as a Dataiku Saved Model or version (meaning it was delete from Dataiku side), then
        # simulate its deletion or actually delete it, depending on what the user selected
        for registry_model in registry_models:
            try:
                registry_model_tags = registry_model.show_tags()
                dataiku_project_key_tag = current_database + "." + \
                    current_schema + "." + "DATAIKU_PROJECT_KEY"
                dataiku_saved_model_id_tag = current_database + "." + \
                    current_schema + "." + "DATAIKU_SAVED_MODEL_ID"

                if dataiku_project_key_tag in registry_model_tags and dataiku_saved_model_id_tag in registry_model_tags:
                    if registry_model_tags[dataiku_project_key_tag] == self.project.project_key:
                        registry_dataiku_saved_model_id = registry_model_tags[dataiku_saved_model_id_tag]
                        if registry_dataiku_saved_model_id not in dataiku_saved_model_ids:

                            models_to_delete.append({
                                "name": registry_model.name,
                                "version": "PARENT MODEL"
                            })

                            if self.perform_deletion:
                                registry.delete_model(registry_model.name)

                            continue

                        model_versions = registry_model.versions()

                        for model_version in model_versions:
                            if model_version.version_name not in dataiku_saved_model_ids_and_versions_upper[registry_dataiku_saved_model_id]:
                                models_to_delete.append({
                                    "name": registry_model.name,
                                    "version": model_version.version_name
                                })

                                if self.perform_deletion:
                                    registry_model.delete_version(
                                        model_version.version_name)

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
