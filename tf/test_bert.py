import sys
sys.path.append('/transformers')
import transformers
import torch
from transformers import BertTokenizer, BertConfig,BertForSequenceClassification
from datasets import load_dataset

num_example = 100 #change the number of example to test here
sst2 = load_dataset("sst2",split = "validation") #load datasets here
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased") # change the model size here
config = BertConfig(hidden_act='relu')
model = BertForSequenceClassification.from_pretrained("bert-large-uncased", hidden_act='relu') #change the model size here
print(model)

i = 0
for sentence in sst2["sentence"][:num_example]: 
  inputs = tokenizer(sentence, return_tensors = "pt")
  outputs = model(**inputs, output_hidden_states = True)
  i += 1
  print ("======= example " + str(i) + "========")
