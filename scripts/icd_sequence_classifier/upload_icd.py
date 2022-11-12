import json 
import pandas as pd
from datasets import Dataset,load_dataset,DatasetDict,concatenate_datasets, load_from_disk
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN_HF = os.getenv("TOKEN_HF")

def load_datasets():

    for dataset_num in range(0, 6):
        df = pd.read_csv(f"../../datasets/training_data/dataset_{dataset_num}.csv") 
        df = df[['TEXT', 'ICD10_L3_DESCRIPTION', 'ICD10_GROUP_DESCRIPTION']]
        df.to_json(f"../../datasets/training_data/train_dataset_{dataset_num}.jsonl",orient="records",lines=True)
        dataset = load_dataset("json",data_files=f"../../datasets/training_data/train_dataset_{dataset_num}.jsonl", split="train")
        dataset = dataset.train_test_split(test_size=0.2,seed=7524)
        dataset_eval = dataset['test']

        if dataset_num == 0:
            dataset_eval.save_to_disk(f"../../datasets/training_data/biobert_eval_dataset_{dataset_num}")
            
        else:
            dataset_eval_prior = load_from_disk(f"../../datasets/training_data/biobert_eval_dataset_{dataset_num-1}")
            dataset_eval_final = concatenate_datasets([dataset_eval, dataset_eval_prior])                                 
            dataset_eval_final.save_to_disk(f"../../datasets/training_data/biobert_eval_dataset_{dataset_num}")
    


if __name__ == "__main__":
    
    load_datasets()




    