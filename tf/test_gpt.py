import sys
# sys.path.append('/transformers')
import transformers
import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2ForSequenceClassification
from datasets import load_dataset

num_example = 2 #change the number of example here
# sst2 = load_dataset("sst2",split = "validation") #load the dataset here
dataset = load_dataset("math_qa", split = "validation") #load the math_qa dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium") #change the model
model = GPT2ForSequenceClassification.from_pretrained("gpt2-medium") #change the model
print(model)

# i = 0

# # Prompt: Problem
# for j, problem in enumerate(dataset["Problem"][:num_example]):
#     inputs = tokenizer(problem, return_tensors="pt")
#     print(f"======= Example {j + 1} =======")
#     outputs = model(**inputs, output_hidden_states=True)
#     # print(f"======= Example {j + 1} =======")


# # Prompt 1: Problem + Options
# for j in range(num_example):
#     prompt = dataset[j]["Problem"] + " " + dataset[j]["options"]
#     # print(prompt)
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model(**inputs, output_hidden_states=True)
#     print(f"======= Example {j + 1} with Problem + Options =======")

# Prompt 2: Problem + Rationale + Options
for j in range(num_example):
    prompt = dataset[j]["Problem"] + " " + dataset[j]["Rationale"] + " " + dataset[j]["options"]
    # print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    print(f"======= Example {j + 1} with Problem + Rationale + Options =======")

# # Prompt 3: Annotated_formula + Options
# for j in range(num_example):
#     prompt = dataset[j]["annotated_formula"] + " " + dataset[j]["options"]
#     print(prompt)
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model(**inputs, output_hidden_states=True)
#     print(f"======= Example {j + 1} with Annotated_formula + Options =======")




# for sentence in sst2["sentence"][:num_example]: 

#   inputs = tokenizer(sentence, return_tensors = "pt")
#   outputs = model(**inputs, output_hidden_states = True) #,use_cache = True
#   i += 1
#   print ("======= example " + str(i) + "========")


  # NOTE:
  # 1. can i try global varaible?
  # 2. now need to detect a round and append after each other / next time !
  # the list will keep in the memory until the whole process is over
  # --> the model will not initialized, only initialized when we load it (a training without loss backprop)
