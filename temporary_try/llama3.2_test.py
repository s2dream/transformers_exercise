import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizerFast
from huggingface_hub import login
import os

# model_id = "meta-llama/Llama-3.2-1B"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_id = "meta-llama/Llama-3.2-3B"

access_token = ""
login(token=access_token)
os.environ["HF_ACCESS_TOKEN"] = access_token

tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
# print(str(tokenizer))

model = LlamaForCausalLM.from_pretrained(model_id, token=access_token)
# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token)

# print(str(model))
# print(tokenizer)

input_string = "I want to ask you who is you?"
tokenized_string = tokenizer(input_string, return_tensors="pt")
ret = model(**tokenized_string)
output_ids = torch.argmax(ret["logits"], dim=-1).squeeze()
ret = tokenizer.decode(output_ids.tolist())
print(ret)

