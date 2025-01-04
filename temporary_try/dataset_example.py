from datasets import Dataset
from transformers import AutoTokenizer

# list of dict 데이터
data = [
    {"text": "This is the first example.", "label": 0},
    {"text": "Here is another sentence.", "label": 1},
    {"text": "More data for testing.", "label": 0},
]

dict_data = {
    "text": [d["text"] for d in data],
    "label": [d["label"] for d in data],
}

# Dataset 객체 생성
dataset = Dataset.from_dict(dict_data)

# 토크나이저 준비
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 토크나이징 함수
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# 데이터셋에 토크나이징 적용
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 데이터 확인
print(tokenized_dataset[0])