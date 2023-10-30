import dataiku
from dataiku.runnables import Runnable
from dataikuapi.utils import DataikuException
from datetime import datetime as dt
from datetime import timedelta
from dataiku.base.utils import safe_unicode_str


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
        self.perform_deletion = self.config.get("perform_deletion", False)
        if not config.get('mes_id'):
            raise ValueError('No model evaluation store was selected.')
        self.project = dataiku.api_client().get_project(project_key)
        self.mes = dataiku.api_client().get_project(project_key).get_model_evaluation_store(config.get('mes_id'))
        if not config.get('min_days') >= 0:
            raise ValueError('Invalid number of days upon which older model evaluations will be deleted, minimum 0.')

    def get_progress_target(self):
        return 100, 'NONE'

    def run(self, progress_callback):
        """
        This method first identifies which model evaluations will be deleted and which will remain.
        It builds a summary of the actions for the user.
        If perform_deletion param is set to True, the model evaluations in model_evaluations_to_delete will be deleted.
        """

        min_days = int(self.config.get('min_days')) - 1
        model_evaluations_raw = self.project.client._perform_json("GET", "/projects/%s/modelevaluationstores/%s/evaluations/" % (self.project_key, self.mes.id))

        # Sorting between versions listed in dry runs and actual deletion is guaranteed to be stable.
        # See docs https://docs.python.org/3/howto/sorting.html#sort-stability-and-complex-sorts
        # See docstring under to_numeric explaining why this is required.

        model_evaluations = [{'name': me['userMeta']['name'],
                              'timestamp': me['created'],
                              'id': me['ref']['evaluationId']
                              } for me in model_evaluations_raw]

        model_evaluations = sorted(model_evaluations, key=lambda k: k['timestamp'], reverse=True)

        model_evaluations_to_delete = list(filter(filter_old_me(min_days), model_evaluations))
        model_evaluations_to_keep = [me for me in model_evaluations if me not in model_evaluations_to_delete]

        html = "<h4>Summary</h4>"
        html += "<span>{}</span><br>".format(summarize(model_evaluations_to_keep, 'to keep'))
        html += "<span>{}</span><br>".format(summarize(model_evaluations_to_delete, 'to delete'))

        if self.perform_deletion:
            try:
                ids_to_delete = [me['id'] for me in model_evaluations_to_delete]
                self.mes.delete_model_evaluations(ids_to_delete)
                html += "<br/><span><strong>{} model evaluation(s) deleted according to summary</strong></span>".format(len(model_evaluations_to_delete))
            except DataikuException as e:
                html += '<span>An error occurred while trying to delete versions.</span><br>'
                html += u'<span>{}</span>'.format(safe_unicode_str(e))
        return html


def filter_old_me(min_days):
    def ret(me):
        today = dt.now().date()
        date = dt.utcfromtimestamp(me['timestamp']/1000).date()
        return today - date > timedelta(days=min_days)
    return ret


def timestamp_to_date(timestamp):
    return dt.utcfromtimestamp(timestamp/1000).strftime("%Y-%m-%d")


def summarize(model_evaluations, action):
    if len(model_evaluations) == 0:
        text = "<strong>0 model evaluations {}</strong>".format(action)
    elif len(model_evaluations) == 1:
        me = model_evaluations[0]
        text = "<strong>1 model evaluation {}:</strong> {}, evaluated on {}".format(action, me['name'], timestamp_to_date(me['timestamp']))
    else:
        text = "<strong>{} model evaluations {}: </strong><br/>".format(len(model_evaluations), action)
        text += '<br/>'.join(list(map(lambda me:
                                      "- {}, evaluated on {}".format(me['name'], timestamp_to_date(me['timestamp'])),
                                      model_evaluations)))
    return text