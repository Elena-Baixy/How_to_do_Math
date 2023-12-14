import sys
sys.path.append('/transformers')
import transformers
import torch
from transformers import GPT2Tokenizer, GPT2Config,GPT2ForSequenceClassification
from datasets import load_dataset

num_example = 2 #change the number of example here
sst2 = load_dataset("sst2",split = "validation") #load the dataset here
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium") #change the model
model = GPT2ForSequenceClassification.from_pretrained("gpt2-medium") #change the model
print(model)

i = 0
for sentence in sst2["sentence"][:num_example]: 
  inputs = tokenizer(sentence, return_tensors = "pt")
  outputs = model(**inputs, output_hidden_states = True) #,use_cache = True
  i += 1
  print ("======= example " + str(i) + "========")
