import dataiku
import json


def do(payload, config, plugin_config, inputs):
    response = {}
    if "method" not in payload:
        return response

    if payload["method"] == "get-info-retrain":
        input_dataset_full_name = get_input_name_from_role(inputs, "input_dataset_name")
        input_dataset = dataiku.Dataset(input_dataset_full_name)
        input_dataset_columns = [c for c in input_dataset.read_schema()]
        input_dataset_columns_new = []
        i = 1
        for input_dataset_column in input_dataset_columns:
            input_dataset_column["id"] = i
                        
            i += 1
            input_dataset_columns_new.append(input_dataset_column)
            
        response['input_dataset_columns'] = input_dataset_columns_new

        #col_label = config.get('col_label', None)
        #response['col_label'] = col_label
        
    response['pluginId'] = 'visual-snowparkml'
    
    return response

def get_input_name_from_role(inputs, role):
    return [inp for inp in inputs if inp["role"] == role][0]["fullName"]

def add_plugin_id(response):
    response['pluginId'] = 'visual-snowparkml'
