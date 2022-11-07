import json 
import pandas as pd
import simple_icd_10_cm as icd
from datasets import Dataset,load_dataset
import os
from dotenv import load_dotenv
load_dotenv()

def load_icd_data():
    
    df = pd.DataFrame(json.load(open("../../datasets/icd/icd_json.json","r"))) 

    df["text"] = df.text.apply(lambda i: i.get("codeDescription"))
    df["icd-l3"] = df.icd10Code.str[0:3]
    df["icd_group"] = df["icd-l3"].apply(icd.get_parent)
    df["icd_group_description"] = df["icd_group"].apply(icd.get_description)
    
    df["icd10_tc_category"] = df["icd-l3"].apply(icd.get_description)
    df["icd10_tc_category_group"] = df["icd_group_description"]

    df[["text","icd10_tc_category","icd10_tc_category_group"]].to_json("../../datasets/icd/icd_dataset.jsonl",orient="records",lines=True)
    dataset = load_dataset("json",data_files="../../datasets/icd/icd_dataset.jsonl")
    dataset.train_test_split(test_size=0.2,seed=7524)

    dataset.push_to_hub("rjac/icd10-reference-cm",max_shard_size="250MB",private=False,token=os.getenv("TOKEN_HF"))

def load_mimic_data(dataset_num=0):
    
    df = pd.read_csv(f"../../datasets/training_data/dataset_{dataset_num}.csv") 
    df.head()
    
    df[['SUBJECT_ID', 'HADM_ID', 'TEXT', 'ICD10_GROUP', 'ICD10_GROUP_DESCRIPTION']].to_json(f"../../datasets/training_data/dataset_{dataset_num}.jsonl",orient="records",lines=True)
    dataset = load_dataset("json",data_files=f"../../datasets/training_data/dataset_{dataset_num}.jsonl")
    dataset.train_test_split(test_size=0.2,seed=7524)
    
    dataset.push_to_hub(f"rjac/dataset_{dataset_num}",max_shard_size="250MB",private=False,token=os.getenv("TOKEN_HF"))

    

if __name__ == "__main__":
    
    load_icd_data()
    load_mimic_data()




    