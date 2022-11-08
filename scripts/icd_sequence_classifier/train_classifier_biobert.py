import json 
import pandas as pd
from datasets import load_dataset,Dataset
from sklearn import metrics
import numpy as np
import re
from transformers import (
    BertTokenizer
    ,BertForSequenceClassification
    ,DataCollatorWithPadding
    ,TrainingArguments
    ,Trainer
    ,pipeline
)
import os
import mlflow
import argparse

from dotenv import load_dotenv
load_dotenv()

TOKEN_HF = os.getenv("TOKEN_HF")
PRETRAINED_MODEL_NAME = os.getenv("PRETRAINED_MODEL_NAME")
FINETUNED_MODEL_NAME = os.getenv("FINETUNED_MODEL_NAME")

model_checkpoint = PRETRAINED_MODEL_NAME
tokenizer = BertTokenizer.from_pretrained(model_checkpoint)


def compute_metrics(eval_pred):
    logits_, labels_ = eval_pred
    predictions = np.argmax(logits_, axis=-1)

    accuracy = metrics.accuracy_score(labels_, predictions)
    f1_score_micro = metrics.f1_score(labels_, predictions, average='micro')
    f1_score_macro = metrics.f1_score(labels_, predictions, average='macro')

    return {"accuracy": accuracy, "f1_score_micro": f1_score_micro, "f1_score_macro": f1_score_macro}

def tokenize_text(batch):
    texts = batch["TEXT"]
    return tokenizer(texts,truncation=True)

def load_data(path):
    dataset = load_dataset(path,split="train")
    dataset = dataset.rename_column("ICD10_GROUP_DESCRIPTION","labels")
    dataset = dataset.class_encode_column("labels")
    dataset = dataset.train_test_split(test_size=0.2,seed=6654)
    return dataset

def main():
    
    parser = argparse.ArgumentParser(description="Indicate dataset to train biobert prediction model on.")
    parser.add_argument("-n", "--dataset_num", type=int, help="Indicate dataset cut to train on. Dataset 0 is the icd10 dataset, Dataset 1-5 is MIMIC III", required=True, choices=range(0, 6))
    args = parser.parse_args()
    
    path=f'rjac/biobert_pred_dataset_{args.dataset_num}'
    dataset = load_data(path=path)

    tokenized_dataset = dataset.map(tokenize_text,remove_columns=['TEXT', 'ICD10_L3_DESCRIPTION'])

    target_feature = tokenized_dataset["train"].features["labels"]
    num_classes = target_feature.num_classes
    label_names = target_feature.names
    label_names = [re.sub(r'[(),\[\]]','',i) for i in label_names]

    id2label = {str(i): label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}


    train_dataset = tokenized_dataset["train"].shuffle(seed=7854)
    validation_dataset = tokenized_dataset["test"]
    
    model = BertForSequenceClassification.from_pretrained(model_checkpoint,num_labels=num_classes,id2label=id2label,label2id=label2id)

    num_freeze_param = 180
    for i,p in enumerate(model.bert.parameters()):
        if i < num_freeze_param:
            p.requires_grad = False

    os.environ["MLFLOW_EXPERIMENT_NAME"] = f'icd_biobert_{args.dataset_num}'

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    batch_size=100
    
    args = TrainingArguments(
        'biobert-icd',
        evaluation_strategy="epoch",
        overwrite_output_dir=True,
        save_total_limit = 1,
        save_strategy="epoch",
        learning_rate=1e-3,
        num_train_epochs=10,
        weight_decay=0.001,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    mlflow.end_run()    
    
    trainer.save_model("model")
    tokenizer_final = BertTokenizer.from_pretrained("model")
    model_final = BertForSequenceClassification.from_pretrained("model")    
    
    model_final.push_to_hub(FINETUNED_MODEL_NAME,use_auth_token=TOKEN_HF)
    tokenizer_final.push_to_hub(FINETUNED_MODEL_NAME,use_auth_token=TOKEN_HF)

if __name__ == "__main__":
    
     main()

    
    