import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


#### setting ####
dataset = load_dataset("yelp_review_full")
print(dataset["train"][100])
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


#### Train by Trainer in Transformer package
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

#### Hyperparameter
# training_args = TrainingArguments(output_dir='./model_save/05/test_trainer')

#### Evaluation
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
training_args = TrainingArguments(output_dir='./model_save/05/test_trainer', eval_strategy="epoch", num_train_epochs=3)

#### Start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
