import tritonclient.http as httpclient
from transformers import TokenClassificationPipeline,AutoTokenizer
import numpy as np
from typing import List, Optional, Tuple, Union
import torch

class TritonTokenClassificationPipeline(TokenClassificationPipeline):

    def __init__(self,triton_url=None,model_name=None,model_version="1",tokenizer=None):    
        #super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.framework = 'pt'
        self.id2label = {0: 'O', 1: 'B-Disease', 2: 'I-Disease'}
        self.aggregation_strategy = "simple"
        self.triton_client = httpclient.InferenceServerClient(url=triton_url)
        self.model_name = model_name
        self.model_version = model_version
        
    def cast_tritonhttpclient(self,model_inputs):
        input_tensor = model_inputs.get("input_ids").numpy().astype(np.int32)
        attention_tensor = model_inputs.get("attention_mask").numpy().astype(np.int32)
        inputs = httpclient.InferInput('input__0', input_tensor.shape, "INT32")
        attention = httpclient.InferInput('input__1',attention_tensor.shape, "INT32")
        inputs.set_data_from_numpy(input_tensor,binary_data=False)
        attention.set_data_from_numpy(attention_tensor,binary_data=False)
        outputs = httpclient.InferRequestedOutput("output__0",binary_data=False)
        return inputs,attention,outputs

    def preprocess(self,sentence):
        truncation = True if self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0 else False
        model_inputs = self.tokenizer(
            sentence,
            return_tensors=self.framework,
            truncation=truncation,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
        )
        model_inputs["sentence"] = sentence
        return model_inputs

    def forward(self,model_inputs):
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        triton_input, triton_attention,triton_outputs = self.cast_tritonhttpclient(model_inputs)
        logits = self.triton_client.infer(self.model_name,model_version=self.model_version,inputs=[triton_input,triton_attention],outputs=[triton_outputs])
        logits = torch.tensor(logits.as_numpy("output__0"))

        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            **model_inputs,
        }    

    def aggregate(self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        aggregation_strategy = self.aggregation_strategy
        if aggregation_strategy in {AggregationStrategy.NONE, AggregationStrategy.SIMPLE}:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.id2label[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            entities = self.aggregate_words(pre_entities, aggregation_strategy)

        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities(entities)

    def single_process(self,sentence):
        model_inputs = self.preprocess(sentence)
        model_outputs = self.forward(model_inputs)
        sentence_output = self.postprocess(model_outputs)
        return sentence_output

    def __call__(self,inputs: Union[str, List[str]]):
        outputs = [self.single_process(sentence) for sentence in inputs]
        return outputs    
