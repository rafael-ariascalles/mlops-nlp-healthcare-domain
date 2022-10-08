from fastapi import FastAPI
from  icd import ICD
import json

app = FastAPI(title='ICD9 Code Prediction')


@app.post("/", tags=['ICD Prediction'])
async def code_prediction(input_text: str = ''):
    
    model = ICD()
    prediction = model.predict(text=input_text)
    return json.dumps(str(prediction))
    
  
@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Service is online."}



#uvicorn main:app --host 0.0.0.0 --port 8000