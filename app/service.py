from fastapi import FastAPI, File, UploadFile
import requests
import os

app = FastAPI()

@app.get("/")
async def healthcheck():
    setfit_icd_classifier_healthcheck = requests.get(os.getenv('URL_SETFIT_ICD_SERVICE'))
    bert_icd_classifier_healthcheck = requests.get(os.getenv('URL_BERT_ICD_SERVICE'))
    #disease_token_classifier_healthcheck = requests.get(os.getenv('URl_DISEASE_SERVICE'))

    return {
         "setfit_icd_classifier_healthcheck": setfit_icd_classifier_healthcheck.json()
        ,"bert_icd_classifier_healthcheck": bert_icd_classifier_healthcheck.json() 
     #   ,"disease_token_classifier_healthcheck": disease_token_classifier_healthcheck.json()
    }
