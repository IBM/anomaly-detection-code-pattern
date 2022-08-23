import requests

class ADService:
    
    def __init__(self):
        pass
    
    def _connection_check(self):

        self.headers_ = {
            'X-IBM-Client-Id': self.client_id_,
            'X-IBM-Client-Secret': self.client_secret_,
            'accept': "application/json",
            }
        
        response = requests.get("https://api.ibm.com/ai4industry/run/connection-check", 
                                    headers=self.headers_)
        
        response_json = response.json()
        if 'message' in response_json.keys():
            return True
        else:
            return False
    
    def connect(self, client_id, client_secret):
        """_summary_

        Args:
            client_id (_type_): _description_
            client_secret (_type_): _description_
        """
        self.client_id_ = client_id
        self.client_secret_ = client_secret
        if self._connection_check():
            print ('Service is up')
        else:
            print ('Some issue')
    
    def execute(self, file_path, file_name, payload):
        complete_path =  file_path + file_name
        files = {'data_file': (file_name, open(complete_path, 'rb'))}

        post_response = requests.post("https://api.ibm.com/ai4industry/run/anomaly-detection/timeseries/univariate/batch", 
                                    data=payload,
                                    files=files,
                                    headers=self.headers_)

        post_r_json = post_response.json()
        self.anomaly_service_jobId_ = None
        if 'jobId' in post_r_json:
            self.anomaly_service_jobId_ = post_r_json['jobId']
            print ('submitted successfully job : ', post_r_json['jobId'])
        else:
            print (post_r_json)    
    
    def status(self):
        get_response = requests.get("https://api.ibm.com/ai4industry/run/result/" + self.anomaly_service_jobId_, headers=self.headers_)
        json_data = get_response.json()
        print("the status of job {} is {}.".format(self.anomaly_service_jobId_, json_data['status']))
    
    def retrieve_logs(self):
        pass
    
    def clean_up(self):
        pass
    
    def fetch_results(self):
        pass
    
