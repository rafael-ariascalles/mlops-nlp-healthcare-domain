from turtle import st
from transformers import AutoModelForTokenClassification, AutoTokenizer

def get_model():
    model = AutoModelForTokenClassification.from_pretrained("rjac/biobert-ner-diseases-model")
    tokenizer = AutoTokenizer.from_pretrained("rjac/biobert-ner-diseases-model")
    model.save_pretrained("models/",state_dict=model.state_dict())
    tokenizer.save_pretrained("models/")

if __name__ == "__main__":
    get_model()