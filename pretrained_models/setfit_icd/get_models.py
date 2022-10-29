from transformers import AutoModelForTokenClassification, AutoTokenizer

def get_model():
    model = AutoModelForTokenClassification.from_pretrained("rjac/setfit-ICD10-L3")
    tokenizer = AutoTokenizer.from_pretrained("rjac/setfit-ICD10-L3")
    model.save_pretrained("models/",state_dict=model.state_dict())
    tokenizer.save_pretrained("models/")

if __name__ == "__main__":
    get_model()