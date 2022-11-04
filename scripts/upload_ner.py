import pandas as pd
import numpy as np
from datasets import Dataset,ClassLabel,DatasetDict
import os
from dotenv import load_dotenv
load_dotenv()

mapper = {"O":0,"B":1,"I":2}
labels_name = ["O","B-Disease","I-Disease"]

def dataframe2dataset(dataframe,labels_name):
    aggregator = lambda x: list(x)
    dataframe = dataframe.groupby(["sentence_id"]).agg({"tokens":aggregator,"tags":aggregator}).copy()
    ner_dataset = Dataset.from_pandas(dataframe)
    ner_dataset.features["tags"].feature = ClassLabel(num_classes=len(labels_name), names=labels_name, id=[0,1,2])
    #ner_feature = ner_dataset.features["tags"]
    return ner_dataset

def create_dataset(path,id_prefix,mapper = None):
    with open(path, 'r') as f:
        lines = f.readlines()
        corpus =[]
        sentences = []
        for line in lines:
            if line != '\n':
                sentences.append(line)
            else:
                corpus.append(sentences)
                sentences = []
        id = []
        word = []
        tag = []
        for i,lines in enumerate(corpus):
            for line in lines:
                word_tag = line.replace("\n",'').split('\t')
                id.append(id_prefix+str(i))
                word.append(word_tag[0])
                tag.append(word_tag[-1])

        df = pd.DataFrame({"sentence_id":id,"tokens":word,"tags":tag})
        if mapper:
            df['tags'] = df['tags'].map(mapper)
        return df

if __name__ == "__main__":   
    train_bc5cdr = create_dataset('../datasets/BC5CDR-disease-train_dev.tsv',"BC5CDR-",mapper)
    train_ncbi = create_dataset('../datasets/NCBI-disease-train_dev.tsv',"NCBI-",mapper)
    test_bc5cdr = create_dataset('../datasets/BC5CDR-disease-test.tsv',"BC5CDR-",mapper)
    test_ncbi = create_dataset('../datasets/NCBI-disease-test.tsv',"NCBI-",mapper)

    train_dataframe = pd.concat([train_bc5cdr,train_ncbi])
    test_dataframe = pd.concat([test_bc5cdr,test_ncbi])

    train_dataframe = dataframe2dataset(train_dataframe,labels_name)
    test_dataframe = dataframe2dataset(test_dataframe,labels_name)

    #save jsonl
    train_dataframe.to_json("../datasets/ner_train.jsonl",orient="records",lines=True)
    test_dataframe.to_json("../datasets/ner_test.jsonl",orient="records",lines=True)
    print(os.getenv("TOKEN_HF"))
    ds = DatasetDict({"train":train_dataframe,"test":test_dataframe})
    ds.push_to_hub("rjac/biobert-ner-diseases-dataset",max_shard_size="250MB",private=False,token=os.getenv("TOKEN_HF"))