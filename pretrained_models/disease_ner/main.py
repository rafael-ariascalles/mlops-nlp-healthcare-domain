from fastapi import FastAPI
from transformers import pipeline,AutoTokenizer,AutoModelForTokenClassification
from pydantic import BaseModel
from typing import List
from inference import TritonTokenClassificationPipeline

app = FastAPI(title='Disease NER')

MODEL_VERSION = "rjac/biobert-ner-diseases-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)
triton_url = "44.202.50.226:8000"
triton_model_name ="ner"
service = TritonTokenClassificationPipeline(triton_url,triton_model_name ="ner",tokenizer=tokenizer)

class ServiceInput(BaseModel):
    sentences: List[str]

class ServiceResponse(BaseModel):
    diseases: List

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Service is online."}

@app.post("/disease_extraction", tags=['Disease NER'])
async def disease_extraction(input_text: ServiceInput):
    response = service(input_text.sentences)
    print(response)
    response = [[d["word"] for d in list_ents] for list_ents in response]
    response_object = ServiceResponse(diseases=response)
    return response_object