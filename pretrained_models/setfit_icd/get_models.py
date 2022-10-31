from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#def get_model_huggingface():
#    model = AutoTokenizer.from_pretrained("rjac/setfit-ICD10-L3")
#    tokenizer = AutoModelForSequenceClassification.from_pretrained("rjac/setfit-ICD10-L3")
#    model.save_pretrained("models/")
#    tokenizer.save_pretrained("models/")
s

class PyTorch_to_TorchScript(torch.nn.Module):
    def __init__(self):
        super(PyTorch_to_TorchScript, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('rjac/setfit-ICD10-L3', return_dict=False)
    def forward(self, data, attention_mask=None):
        return self.model(data, attention_mask)


def get_model_triton():
    
    tokenizer = AutoTokenizer.from_pretrained('rjac/setfit-ICD10-L3')
    text = "subarachnoid hemorrhage scalp laceration service: surgery major surgical or invasive"
    
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=768, padding='max_length', truncation=True)

    mask = input_ids != 1
    mask = mask.long()
    
    pt_model = PyTorch_to_TorchScript().eval()
    traced_script_module = torch.jit.trace(pt_model, (input_ids, mask))
    traced_script_module.save('models/1/model.pt')


def get_config():
    configuration = """
name: "setfit_icd"
platform: "pytorch_libtorch"
max_batch_size: 1024
input [
 {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ 768 ]
  },
{
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [ 768 ]
  }
]
output {
    name: "out_proj"
    data_type: TYPE_FP32
    dims: [ 275 ]
  }
"""

    with open('models/config.pbtxt', 'w') as file:
        file.write(configuration)      


if __name__ == "__main__":
    #get_model_huggingface()
    get_model_triton()
    get_config()
    


