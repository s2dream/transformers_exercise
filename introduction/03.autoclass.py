from transformers import AutoTokenizer
from transformers import pipeline

# AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
sequence = "In a hole in the ground there lived a hobbit."
print(tokenizer(sequence))

# AutoImageProcessor
from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# AutoFeatureExtractor
from transformers import AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

# AutoProcessor - for multi-modal processing

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")


#AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification

model_distilbert_seq_classification = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
model_distilbert_token_classification = AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

input_string= "my name is Jeonghoon Park"
# pipe = pipeline(model=model_distilbert_seq_classification)
# ret = pipe(input_string)

tokenized_string = tokenizer(input_string, return_tensors="pt")
ret = model_distilbert_token_classification(**tokenized_string)
print(ret)



