from fastapi import FastAPI
from  icd import ICD
from clean_icd_output import prediction_json
import json

app = FastAPI(title='ICD9 Code Prediction')


@app.post("/", tags=['ICD Prediction'])
async def code_prediction(input_text: str = ''):
    
    model = ICD()
    prediction = model.predict(text=input_text)
    prediction_cleaned = prediction_json(prediction)
    
    return(prediction_cleaned)
    
  
@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Service is online."}



#uvicorn main:app --host 0.0.0.0 --port 8000