___

## Talks
1. AAAAI 2023 1.5-hour Lab based tutorial
1. MLSys 2022 half day tutorial
2. KDD 2022 3-hour tutorial
3. ICDE 2022 tutorial
4. DASFAA 2022 tutorial

___

## Anomaly Detection Service 
- [IBM Developer API Hub](https://developer.ibm.com/apis/catalog/ai4industry--anomaly-detection-product/)

___

## Example : Setting Local Juputer Environment
1. Python 3
2. Credentials to access the API service (Please follow the [instructions](https://developer.ibm.com/apis/catalog/ai4industry--anomaly-detection-product/Getting%20Started)
or [tutorial](./tutorials/ADTutorial_Registration.pdf))
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

___
## Notebooks

Here are the list of provided notebooks:
1. [Univariate_AD_service_public_data.ipynb](./notebooks/Univariate_AD_service_public_data.ipynb): Anomaly detection on univariate public data
2. [Univariate_AD_service_sample_data.ipynb](./notebooks/Univariate_AD_service_sample_data.ipynb): Anomaly detection on univariate sample data
3. [Multivariate_AD_service_sample_data.ipynb](./notebooks/Multivariate_AD_service_sample_data.ipynb): Anomaly detection on multivariate sample data
4. [Regression-aware_AD_service_sample_data.ipynb](./notebooks/Regression-aware_AD_service_sample_data.ipynb): Regression based anomaly detection
5. [MixtureModel-aware_AD_service_sample_data.ipynb](./notebooks/MixtureModel-aware_AD_service_sample_data.ipynb):
Mixture model based anomaly detection
___
## Additional Links

1. API Service in [IBM API Hub](https://developer.ibm.com/apis/catalog/ai4industry--anomaly-detection-product/)
2. API Service in [IBM Learning Path](https://developer.ibm.com/learningpaths/get-started-anomaly-detection-api/)

