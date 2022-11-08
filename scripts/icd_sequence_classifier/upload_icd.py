import json 
import pandas as pd
from datasets import Dataset,load_dataset
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN_HF = os.getenv("TOKEN_HF")

def load_datasets():

    for dataset_num in range(0, 6):
        df = pd.read_csv(f"../../datasets/training_data/dataset_{dataset_num}.csv") 
        df = df[['TEXT', 'ICD10_L3_DESCRIPTION', 'ICD10_GROUP_DESCRIPTION']]

        if dataset_num == 0:
            df.to_json(f"../../datasets/training_data/dataset_{dataset_num}.jsonl",orient="records",lines=True)
            dataset = load_dataset("json",data_files=f"../../datasets/training_data/dataset_{dataset_num}.jsonl", split="train")
            dataset.train_test_split(test_size=0.2,seed=7524)
        
        else:
            df_prior = pd.read_csv(f"../../datasets/training_data/dataset_{dataset_num-1}.csv") 
            df_prior = df_prior[['TEXT', 'ICD10_L3_DESCRIPTION', 'ICD10_GROUP_DESCRIPTION']]
            df_total = pd.concat([df_prior, df])
            df_total.to_json(f"../../datasets/training_data/dataset_{dataset_num}.jsonl",orient="records",lines=True)
            dataset = load_dataset("json",data_files=f"../../datasets/training_data/dataset_{dataset_num}.jsonl", split="train")
            dataset.train_test_split(test_size=0.2,seed=7524)
    
        dataset.push_to_hub(f"rjac/biobert_pred_dataset_{dataset_num}",max_shard_size="250MB",private=False,token=TOKEN_HF)


if __name__ == "__main__":
    
    load_datasets()




    