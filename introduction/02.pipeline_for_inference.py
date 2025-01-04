from transformers import pipeline

# try default model
generator = pipeline(task="automatic-speech-recognition")
ret = generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")    ## (??) error occured
print(ret)

# try better model
generator = pipeline(model="openai/whisper-large")
ret = generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(ret)


