from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from inference import ICD


app = FastAPI(title='SETFIT: Top 5 ICD Group Prediction')


class ServiceInput(BaseModel):
    clinical_note: str


class ServiceResponse(BaseModel):
    icd_group: List


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Service is online."}


@app.post("/icd_group_prediction", tags=['SETFIT: Top 5 ICD Group Prediction'])
async def icd_prediction(input: ServiceInput):
    
    # run model and get top 5 predictions 
    model = ICD(triton_url='triton:8000')
    response = model.predict(input.clinical_note, num_labels=5)
    response_object = ServiceResponse(icd_group=response)
    return response_object

