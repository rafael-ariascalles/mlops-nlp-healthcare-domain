from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel
from typing import List


app = FastAPI(title='Top 5 ICD Group Prediction')

tokenizer_final = AutoTokenizer.from_pretrained("models/")
model_final = AutoModelForSequenceClassification.from_pretrained("models/")

service = pipeline("text-classification", model=model_final, tokenizer=tokenizer_final, return_all_scores=True, function_to_apply='softmax')

class ServiceInput(BaseModel):
    clinical_note: str

class ServiceResponse(BaseModel):
    icd_group: List

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Service is online."}

@app.post("/icd_group_prediction", tags=['Top 5 ICD Group Prediction'])
async def icd_prediction(input: ServiceInput):
    response = service(input.clinical_note)
    response = sorted(response[0], key=lambda k: float(k['score']), reverse=True)[:5]
    
    response_list = []
    for pred in response:
        label = pred['label']
        score = round(pred['score']*100, 2)
        response_list.append(f'{label}: {score}% Confidence')
        
    response_object = ServiceResponse(icd_group=response_list)
    return response_object