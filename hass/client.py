import json
import requests

class HassClient:
    """
    The client is responsible for making calls to the Home Assistant REST API.
    """

    def __init__(self, base_url, token):
        self.base_url = base_url
        self.token = token

    def make_authorization_header(self):
        """
        Create the authorization header that's needed to make requests.
        """

        return {
            'Authorization' : 'Bearer %s' % self.token
        }

    def post(self, path, data):
        """
        Make a POST request.
        """
        try:
            r = requests.post(self.base_url + path, data=data, headers=self.make_authorization_header())
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            return "Error: " + str(e)

        return r

    def trigger_automation(self, entity_id):
        """
        Trigger specified automation.
        """

        path = '/api/services/automation/trigger'
        data = { 'entity_id': entity_id }
        response = self.post(path, json.dumps(data))
        print(response)
