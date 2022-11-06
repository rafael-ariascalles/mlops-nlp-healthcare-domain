Every Script need a enviroment file inside the folder

Usage for Huggingface Transformers and Datasets

* the MLFLOW_EXPERIMENT_NAME and
MLFLOW_FLATTEN_PARAMS are the name for Huggingface Transformers Trainer MLFlow Callbacks, it is recognize automatically by Trainer. for local tracking you can use mlflow ui with just this variables.

* for external tracking you need to set the MLFLOW_TRACKING_URI,MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD. (All this variables are compatible with DAGshub and Mlflow integration)

* the TOKEN_HF is the token for the huggingface hub. 


##### .env example

```.env
MLFLOW_EXPERIMENT_NAME=<CUSTOM_EXPERIMENT_TRACKING_NAME>
MLFLOW_FLATTEN_PARAMS=1
MLFLOW_TRACKING_URI=<MLFLOW_TRACKING_URI>
MLFLOW_TRACKING_USERNAME=<USERNAME>
MLFLOW_TRACKING_PASSWORD=<PASSWORD>
TOKEN_HF=<HUGGINGFACE_TOKEN_ACCESS>
PRETRAINED_MODEL_NAME=<PRETRAINED_MODEL_NAME>
FINETUNED_MODEL_NAME=<FINETUNED_MODEL_NAME>
```
