import torch
from transformers import pipeline
from huggingface_hub import login
import os
access_token = ""
os.environ["HF_ACCESS_TOKEN"] = access_token

# model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_id = "meta-llama/Llama-3.2-3B"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token = access_token
)

input_string = "Paris is an amazing place to visit,"

ret= pipe(input_string)
print(ret)
