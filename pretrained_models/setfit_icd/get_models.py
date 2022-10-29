from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

def get_model():
    model = AutoModelForTokenClassification.from_pretrained("rjac/setfit-ICD10-L3")
    tokenizer = AutoTokenizer.from_pretrained("rjac/setfit-ICD10-L3")
    model.save_pretrained("models/",state_dict=model.state_dict())
    tokenizer.save_pretrained("models/")
    
    # save model for trinton server
    torch.save(model_final.state_dict(), 'models/pytorch_model.pt')


if __name__ == "__main__":
    get_model()