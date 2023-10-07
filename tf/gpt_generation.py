import sys
sys.path.append('/transformers')
import transformers
import torch
from transformers import GPT2Tokenizer, GPT2Config,GPT2LMHeadModel
from datasets import load_dataset

num_example = 2 #change the number of example here
sst2 = load_dataset("sst2",split = "validation") #load the dataset here
tokenizer = GPT2Tokenizer.from_pretrained("gpt2") #change the model
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2", activation_function="relu") #change the model

def _generate(_prompt):
    # tokenized = tokenizer([_prompt], return_tensors="pt")
    # model=  GPT2LMHeadModel.from_pretrained('results/checkpoint-2000')
    tokenized = tokenizer.prepare_seq2seq_batch(src_texts=_prompt, return_tensors="pt")
    input_ids = tokenized['input_ids']
    output = model.generate(input_ids=input_ids, use_cache = True)

    decoded = tokenizer.batch_decode(output)
    return decoded

pred = _generate("hi")
print(pred)

# 一个block里的存在presents， 12个block结束以后存入layer——past， presnets --> None