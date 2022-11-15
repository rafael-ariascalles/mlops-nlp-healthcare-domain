from fastapi import FastAPI, File, UploadFile
import requests
import os

app = FastAPI()

@app.get("/")
async def healthcheck():
    #setfit_icd_classifier_healthcheck = requests.get(os.getenv('URL_SETFIT_ICD_SERVICE'))
    bert_icd_classifier_healthcheck = requests.get(os.getenv('ICD_SERVICE'))
    disease_token_classifier_healthcheck = requests.get(os.getenv('NER_SERVICE'))

    return {
        "bert_icd_classifier_healthcheck": bert_icd_classifier_healthcheck.json() 
        ,"disease_token_classifier_healthcheck": disease_token_classifier_healthcheck.json()
    }

@app.get("/analyze")
async def analyze(sentence: str):

    disease_token_classifier_response = requests.post(
        os.getenv('NER_SERVICE') + "/disease_extraction"
        , json={"sentences": [sentence]}
    )

    icd_classifier_response = requests.post(
        os.getenv('ICD_SERVICE') + "/icd_group_prediction"
        , json={"clinical_note": sentence}
    )

    return {
        "ICD10": icd_classifier_response.json()
        ,"Diseases": disease_token_classifier_response.json() 
    }
