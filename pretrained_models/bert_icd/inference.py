import tritonhttpclient
from transformers import AutoTokenizer
import numpy as np
import json
from scipy.special import softmax

#docker run --env-file ../../.env -p8000:8000 -p8001:8001 -p8002:8002 --rm --net=host nvcr.io/nvidia/tritonserver:22.06-py3 tritonserver --model-repository=s3://nlp-medical-biller-group-datasets-mlo4/models

class ICD():
    
    def __init__(self, triton_url):
        """
        Instantiate ICD model class with pretrained model paths
        """        
        
        # pretrained model params 
        self.input_name = ['input_ids', 'attention_mask']
        self.output_name = 'out_proj'
        self.model_version = '1'
        self.input_size = 512
        self.label = 275
        self.model_name = 'bert_icd'
        self.model_path = 'rjac/biobert-ICD10-L3'
        self.labels_path = 'labels.json'
        
        # trinton client 
        self.triton_client = tritonhttpclient.InferenceServerClient(url=triton_url, verbose=False)
        self.model_metadata = self.triton_client.get_model_metadata(model_name=self.model_name, model_version=self.model_version)
        self.model_config = self.triton_client.get_model_config(model_name=self.model_name, model_version=self.model_version)
    
    
    def predict(self, text, num_labels=5):
        
        # load model labels 
        with open(self.labels_path, 'r') as openfile:
            labels_object = json.load(openfile)
        
        # run inference and clean output 
        input_ids, mask = self.preprocess(text)
        logits = self.get_logits(input_ids, mask)
        position = [position.item() for position in logits.argsort()[0][-num_labels:]]
        top_predictions = [labels_object[str(pos)] for pos in position]
        probs = softmax(logits[0][position])
    
        response_list = []
        for label, prob in zip(top_predictions, probs):
            score = round(prob.item()*100, 2)
            response_list.append(f'{label}: {score}% Confidence')
    
        response_list.reverse()  
        return response_list  
    
    
    def get_logits(self, input_ids, mask):
        
        input0 = tritonhttpclient.InferInput(self.input_name[0], (1, self.input_size), 'INT32')
        input0.set_data_from_numpy(input_ids, binary_data=False)
        
        input1 = tritonhttpclient.InferInput(self.input_name[1], (1, self.input_size), 'INT32')
        input1.set_data_from_numpy(mask, binary_data=False)
        
        output = tritonhttpclient.InferRequestedOutput(self.output_name, binary_data=False)
        response = self.triton_client.infer(self.model_name,
                                            model_version=self.model_version,
                                            inputs=[input0, input1],
                                            outputs=[output])
        logits = response.as_numpy(self.output_name)
        logits = np.asarray(logits, dtype=np.float32)
             
        return logits


    def preprocess(self, text):
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=self.input_size, truncation=True, padding='max_length')
        input_ids = np.array(input_ids, dtype=np.int32)
        
        mask = input_ids != 0
        mask = np.array(mask, dtype=np.int32)
        
        input_ids = input_ids.reshape(1, self.input_size)
        mask = mask.reshape(1, self.input_size) 

        return (input_ids, mask)
    
    
        