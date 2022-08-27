# anomaly-detection-code-pattern
Sample Jupyter Notebooks for playing around with the IBM Anomaly Detection service to be made available on [IBM API Hub](https://developer.ibm.com/apis/catalog/ai4industry--anomaly-detection-product/).

## Requirements and Installation
1. Python 3
2. Credentials to access the API service (Please follow the [instructions](https://developer.ibm.com/apis/catalog/ai4industry--anomaly-detection-product/Getting%20Started))
3. Clone the repository
```bash
git clone https://github.com/IBM/anomaly-detection-code-pattern.git
cd anomaly-detection-code-pattern/
```
4. (Optional) Create a virtual environment
```
virtualenv ad_env
source ad_env/bin/activate
```
5. Install required packages
  ```
  pip install -r requirements.txt
  ```
6. Open Jupyter notebook in current directory
```bash
python -m ipykernel install --user --name=ad_env  # optional: add virtual environment to jupyter notebook
jupyter notebook
```
## Notebooks

Here are the list of provided notebooks:
1. Univariate_AD_service_public_data.ipynb: Anomaly detection on univariate public data
2. Univariate_AD_service_sample_data.ipynb: Anomaly detection on univariate sample data
3. Multivariate_AD_service_sample_data.ipynb: Anomaly detection on multivariate sample data

## Additional Links
1. API Service in [IBM API Hub](https://developer.ibm.com/apis/catalog/ai4industry--anomaly-detection-product/)
2. API Service in [IBM Learning Path](https://developer.ibm.com/learningpaths/get-started-anomaly-detection-api/)

## Talk
1. MLSys 2022 half day tutorial
2. KDD 2022 3-hour tutorial
3. ICDE 2022 tutorial
4. DASFAA 2022 tutorial
