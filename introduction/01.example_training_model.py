"""
Customized Training
Code from
https://huggingface.co/docs/transformers/ko/quicktour
"""

from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import AutoModel
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import Trainer


################################################
CUSTOM_MODEL = False
DIR_PATH_TO_SAVE_MODEL = "./model_save/01"
################################################


# model setting
if CUSTOM_MODEL:
    # Training Configuration Editing
    my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)
    # model load from configuration
    model = AutoModel.from_config(my_config)
else:
    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")

# training arguments setting
training_args = TrainingArguments(
    output_dir=DIR_PATH_TO_SAVE_MODEL,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5
)

# Tokenizer Setting
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Dataset Setting
dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT


def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])

dataset = dataset.map(tokenize_dataset, batched=True)


### a kind of the collate_fn() function
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # default
# Test -> Success
#data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length' , max_length=513) #
# padding ='max_length' : pad to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided.
# max_length = xxx : max length of the returned list and optionally padding length

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator
)

trainer.train()