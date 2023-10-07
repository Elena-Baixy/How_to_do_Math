import sys
sys.path.append('/transformers')
import transformers
import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from datasets import load_dataset

num_example = 10
sst2 = load_dataset("sst2",split = "validation") #load dataset here
tokenizer = T5Tokenizer.from_pretrained("t5-large") #change the model size here
config = T5Config(hidden_act='relu')
model = T5ForConditionalGeneration.from_pretrained("t5-large") #change the model size here
print(model)

i = 0
for sentence in sst2["sentence"][:num_example]:
  inputs = tokenizer(sentence, return_tensors = "pt")
  outputs = model.generate(**inputs, max_length = 1) 
  i += 1
  print ("======= example " + str(i) + "========")
