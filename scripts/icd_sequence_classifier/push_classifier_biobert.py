import os
from transformers import BertForSequenceClassification, BertTokenizer

model_ = BertForSequenceClassification.from_pretrained("model/icd_biobert")
tokenizer_ = BertTokenizer.from_pretrained("model/icd_biobert")
model_.push_to_hub("biobert-ICD10-L3",use_auth_token=os.getenv("TOKEN_HF"))
tokenizer_.push_to_hub("biobert-ICD10-L3",use_auth_token=os.getenv("TOKEN_HF"))