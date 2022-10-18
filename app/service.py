from fastapi import FastAPI, File, UploadFile
import requests
import os

app = FastAPI()

@app.get("/")
async def healthcheck():
    icd_classifier_healthcheck = requests.get(os.getenv('URL_ICD_SERVICE'))
    disease_token_classifier_healthcheck = requests.get(os.getenv('URl_DISEASE_SERVICE'))

    return {
        "icd_classifier_healthcheck": icd_classifier_healthcheck.json()
        , "disisease_token_classifier_healthcheck": disease_token_classifier_healthcheck.json()
    }
