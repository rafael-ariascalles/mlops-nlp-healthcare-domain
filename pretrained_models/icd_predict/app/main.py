from fastapi import FastAPI
from  icd import ICD
import json

app = FastAPI(title='ICD9 Code Prediction')


@app.post("/", tags=['ICD Prediction'])
async def code_prediction(input_text: str = ''):
    
    model = ICD()
    prediction = model.predict(text=input_text)
    return jsonß.dumps(str(prediction))
    
  
    #return StreamingResponse(io.BytesIO(png_img.tobytes()), media_type="image/png")

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Service is online."}



#uvicorn main:app --host 0.0.0.0 --port 8000