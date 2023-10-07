import sys
sys.path.append('/transformers')
import transformers
import torch
from transformers import GPT2Tokenizer, GPT2Config,GPT2ForSequenceClassification
from datasets import load_dataset

sst2 = load_dataset("sst2",split = "validation") #change the datasets here
tokenizer = GPT2Tokenizer.from_pretrained("gpt2") #change the model size here
model = GPT2ForSequenceClassification.from_pretrained("gpt2", activation_function="relu") #change the model size here
print(model)

i = 0
for sentence in sst2["sentence"][:1]:
  inputs = tokenizer(sentence, return_tensors = "pt")
  outputs = model(**inputs, output_hidden_states = True)
  i += 1
  print ("======= example " + str(i) + "========")
