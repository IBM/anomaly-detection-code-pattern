import requests

class ADService:
    
    def __init__(self):
        pass
    
    def _connection_check(self):

        headers = {
            'X-IBM-Client-Id': self.client_id_,
            'X-IBM-Client-Secret': self.client_secret_,
            'accept': "application/json",
            }
        
        post_response = requests.post("https://api.ibm.com/ai4industry/run/connection-check", 
                                    headers=headers)
        
        post_r_json = post_response.json()
        print (post_r_json)
        pass
    
    def connect(self, client_id, client_secret):
        """_summary_

        Args:
            client_id (_type_): _description_
            client_secret (_type_): _description_
        """
        self.client_id_ = client_id
        self.client_secret_ = client_secret
        self._connection_check()
    
    def execute(self):
        pass
    
    def status(self):
        pass
    
    def retrieve_logs(self):
        pass
    
    def clean_up(self):
        pass
    
    def fetch_results(self):
        pass
    
