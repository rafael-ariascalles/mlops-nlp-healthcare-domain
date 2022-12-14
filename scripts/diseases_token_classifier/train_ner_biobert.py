from transformers import AutoTokenizer,AutoModelForTokenClassification, TrainingArguments, Trainer,pipeline
from datasets import load_dataset,load_metric,DatasetDict
from transformers import DataCollatorForTokenClassification
from tqdm import tqdm
import numpy as np
import os
import mlflow
from dotenv import load_dotenv
load_dotenv()

TOKEN_HF = os.getenv("TOKEN_HF")
PRETRAINED_MODEL_NAME = os.getenv("PRETRAINED_MODEL_NAME")
FINETUNED_MODEL_NAME = os.getenv("FINETUNED_MODEL_NAME")

model_checkpoint = PRETRAINED_MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
metric = load_metric("seqeval")

raw_datasets = load_dataset("rjac/biobert-ner-diseases-dataset")
ner_feature = raw_datasets["train"].features["tags"]
label_names = ner_feature.feature.names



def align_labels_with_tokens(labels, word_ids):
    
    new_labels = []
    current_word = None
    
    for word_id in word_ids:
        
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
            
        elif word_id is None:
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(batch):
    
    tokenized_inputs = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
    all_labels = batch["tags"]
    
    new_labels = []
    
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def tokenize_text(batch):
    texts = batch["text"]
    return tokenizer(texts,truncation=True)

def main():

    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    id2label = {str(i): label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    batch_size=24

    args = TrainingArguments(
        "biobert-ner",
        evaluation_strategy="epoch",
        overwrite_output_dir=True,
        save_total_limit = 1,
        save_strategy="epoch",
        learning_rate=1e-5,
        num_train_epochs=10,
        weight_decay=0.005,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    mlflow.end_run()

    trainer.save_model("model")
    
    tokenizer_final = AutoTokenizer.from_pretrained("model")
    model_final = AutoModelForTokenClassification.from_pretrained("model")
    
    model_final.push_to_hub("biobert-ner-diseases-model",use_auth_token=TOKEN_HF)
    tokenizer_final.push_to_hub("biobert-ner-diseases-model",use_auth_token=TOKEN_HF)
    
if __name__ == "__main__":
    main()