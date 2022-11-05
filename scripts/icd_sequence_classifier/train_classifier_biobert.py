import json 
import pandas as pd
from datasets import load_dataset,Dataset
import joblib
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

from dotenv import load_dotenv
load_dotenv()

tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

def compute_metrics(eval_pred):
    logits_, labels_ = eval_pred
    predictions = np.argmax(logits_, axis=-1)

    accuracy = metrics.accuracy_score(labels_, predictions)
    f1_score_micro = metrics.f1_score(labels_, predictions, average='micro')
    f1_score_macro = metrics.f1_score(labels_, predictions, average='macro')

    return {"accuracy": accuracy, "f1_score_micro": f1_score_micro, "f1_score_macro": f1_score_macro}

def tokenize_text(batch):
    texts = batch["text"]
    return tokenizer(texts,truncation=True)

def load_data(path):
    dataset = load_dataset("rjac/icd10-reference-cm",split="train")
    dataset = dataset.rename_column("icd10_tc_category_group","labels")
    dataset = dataset.class_encode_column("labels")
    dataset = dataset.train_test_split(test_size=0.2,seed=6654)
    return dataset

def main():
    dataset = load_data("rjac/icd10-reference-cm")
    tokenized_dataset = dataset.map(tokenize_text,remove_columns=['text', 'icd10_tc_category'])

    target_feature = tokenized_dataset["train"].features["labels"]
    num_classes = target_feature.num_classes
    label_names = target_feature.names
    label_names = [re.sub(r'[(),\[\]]','',i) for i in label_names]

    id2label = {str(i): label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}


    train_dataset = tokenized_dataset["train"].shuffle(7854)
    validation_dataset = tokenized_dataset["test"]
    

    model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1",num_labels=num_classes,id2label=id2label,label2id=label2id)

    num_freeze_param = 180
    for i,p in enumerate(model.bert.parameters()):
        if i < num_freeze_param:
            p.requires_grad = False

    os.environ["MLFLOW_EXPERIMENT_NAME"] = "icd_biobert"

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    batch_size=100
    
    args = TrainingArguments(
        "biobert-ner",
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
    trainer.save_model("model/icd_biobert")

if __name__ == "__main__":
    main()