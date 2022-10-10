from fastapi import FastAPI

app = FastAPI(title='Disease NER')

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Service is online."}

@app.post("/disease_extraction", tags=['Disease NER'])
async def disease_extraction(input_text: str = ''):
        return {"message": "Model not Implemented"}
