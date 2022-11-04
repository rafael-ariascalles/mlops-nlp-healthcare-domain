from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from pathlib import Path

MODEL_VERSION = 'rjac/biobert-ner-diseases-model'
text = "Sample Text for the application"
Path("models/ner/1").mkdir(parents=True, exist_ok=True)


class PyTorch_to_TorchScript(torch.nn.Module):
    def __init__(self):
        super(PyTorch_to_TorchScript, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL_VERSION, return_dict=False)
    def forward(self, data, attention_mask=None):
        return self.model(data, attention_mask)

def get_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)
    MODEL_INPUT = min(tokenizer.max_len_single_sentence,254)
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=MODEL_INPUT, padding='max_length', truncation=True)
    mask = input_ids != 1
    mask = mask.long()

    pt_model = PyTorch_to_TorchScript().eval()
    traced_script_module = torch.jit.trace(pt_model, (input_ids, mask))
    traced_script_module.save('models/ner/1/model.pt')
    MODEL_OUTPUT = len(pt_model.model.config.label2id)
    
    configuration = """name: "ner"
platform: "pytorch_libtorch"
max_batch_size: 1024
input [
    {
        name: "input__0"
        data_type: TYPE_INT32
        dims: [ -1 ]
    },
    {
        name: "input__1"
        data_type: TYPE_INT32
        dims: [ -1 ]
    }
]
output {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ -1 , """+ str(MODEL_OUTPUT) +""" ]
}
    """
    with open('models/ner/config.pbtxt', 'w') as file:
        file.write(configuration)

if __name__ == "__main__":
    get_model()