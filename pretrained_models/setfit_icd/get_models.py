from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def get_model():
    model = AutoTokenizer.from_pretrained("rjac/setfit-ICD10-L3")
    tokenizer = AutoModelForSequenceClassification.from_pretrained("rjac/setfit-ICD10-L3")
    model.save_pretrained("models/")
    tokenizer.save_pretrained("models/")
    
    # save model for trinton server
    #torch.save(model.state_dict(), 'models/pytorch_model.pt')

if __name__ == "__main__":
    get_model()