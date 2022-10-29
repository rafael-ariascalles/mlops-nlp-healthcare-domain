from fastapi import FastAPI
from transformers import pipeline,AutoTokenizer,AutoModelForTokenClassification
from pydantic import BaseModel
from typing import List

app = FastAPI(title='ICD Prediction')

tokenizer_final = AutoTokenizer.from_pretrained("models/")
model_final = AutoModelForTokenClassification.from_pretrained("models/")
service = pipeline("token-classification",model=model_final,tokenizer=tokenizer_final,aggregation_strategy="simple")


class ServiceInput(BaseModel):
    sentence: str

class ServiceResponse(BaseModel):
    diseases: List

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Service is online."}

@app.post("/icd_prediction", tags=['ICD Prediction'])
async def icd_prediction(input_text: ServiceInput):
    response = service(input_text.sentence)
    response = [d["word"] for d in response]
    response_object = ServiceResponse(diseases=response)
    return response_object