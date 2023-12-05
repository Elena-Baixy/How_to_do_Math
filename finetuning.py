import argparse
import os
import math
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader


import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    GPT2LMHeadModel, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup, 
    set_seed, 
    TrainingArguments, 
    Trainer, 
    GPT2TokenizerFast, 
    AutoConfig, 
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    GPT2Tokenizer,
    TrainerCallback
)
from tqdm import tqdm

parser = argparse.ArgumentParser(description='')
# Dataset
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--mode', type=str)
parser.add_argument('--PLM', type=str, default='gpt2')
parser.add_argument('--task', type=str, default='e2e')

args = parser.parse_args()

# model_name_or_path = "roberta-large"
model_name_or_path = args.PLM
train_batch_size = 2
eval_batch_size = 2
lr = 2e-4
task = args.task
device = "cuda"
num_epochs = 5 #10
warmup_steps = 500

#################################### Prepare data####################################
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenized_datasets = torch.load(f'data/{args.task}/train.pt')
val_tok_dataset = torch.load(f'data/{args.task}/val.pt')
test_tok_dataset = torch.load(f'data/{args.task}/test.pt')
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)


#################################### Prepare data  END ####################################

# model = GPT2LMHeadModel.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained(f"../../../data/smallyan/gpt2_nlg_{args.task}/checkpoint-7500/")
model.to(device)
# model.print_trainable_parameters()
#load the model
# model = GPT2LMHeadModel.from_pretrained(f"../../../data/smallyan/gpt2_nlg_{args.task}/checkpoint-25000/")

training_args = TrainingArguments(
    output_dir=f'../../../data/smallyan/gpt2_nlg_{args.task}',
    evaluation_strategy="steps",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=4,
    warmup_steps=warmup_steps,
    do_predict=True,
    do_eval=True,
    learning_rate=lr,
    weight_decay=0.01 ,
    logging_dir='./logs',
    report_to="none",
    load_best_model_at_end = True,
)

def compute_metrics_cls(eval_preds):
    metric = evaluate.load("glue", task)
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def compute_metrics_nlg(eval_preds):
    bleu = evaluate.load("bleu")
    breakpoint()
    results = bleu.compute()
    print("bleu: ", results)
    breakpoint()
    return results

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        #print("Debug:",logs)
        #print(state.log_history)
        #breakpoint()
        print("eval_loss:",state.log_history[-1]['eval_loss'])
        print("Perplexity:", math.exp(state.log_history[-1]['eval_loss']))

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer = tokenizer,
    train_dataset=tokenized_datasets,
    eval_dataset=val_tok_dataset['input_ids'],
    data_collator=data_collator,
    callbacks=[PerplexityCallback()]
)

# Train the model
trainer.train()
# trainer.save_model(f"../../../data/smallyan/gpt2_nlg_{args.task}")




def _generate(_prompt):
    tokenized = tokenizer.prepare_seq2seq_batch(src_texts=_prompt, return_tensors="pt")
    model.to(device)
    input_ids = tokenized['input_ids'].to('cuda')
    if task == 'e2e':
        max_length = input_ids.shape[1] + 30
    elif task == 'cnn':
        max_length = input_ids.shape[1] + 30
    elif task == 'xsum':
        max_length = input_ids.shape[1] + 20
    else:
        max_length = input_ids.shape[1] + 20
    # breakpoint()
    output = model.generate(input_ids=input_ids, num_beams = 10, min_length = 20, max_length = max_length)

    decoded = tokenizer.batch_decode(output)
    return decoded

bleu_score = 0
rouge_score = 0
nist_score = 0
met_score = 0
count = 0
if task == 'cnn':
    prompt = 'article'
    conti = 'highlights'
elif task == 'e2e':
    prompt = 'meaning_representation'
    conti = 'human_reference'
elif task == 'xsum':
    prompt = 'document'
    conti = 'summary'
elif task == 'wmt':
    prompt = 'de'
    conti = 'en'
if task != 'wmt':
    for i in range(100):
        count += 1
        predictions = _generate(test_tok_dataset[i][prompt])
        print("The model generates: ", predictions)
        print("Human reference: ", test_tok_dataset[i][prompt]+test_tok_dataset[i][conti])
        reference =[]
        reference.append(test_tok_dataset[i][prompt]+test_tok_dataset[i][conti])

        references = []
        references.append(reference)
        # evaluate bleu
        bleu = evaluate.load("bleu")
        results_bleu = bleu.compute(predictions=predictions, references=references)
        bleu_score += results_bleu['bleu']
        # evaluate rouge
        rouge = evaluate.load("rouge")
        results_rouge = rouge.compute(predictions=predictions, references=references)
        rouge_score += results_rouge['rougeL']
        # evaluate nist
        nist_mt = evaluate.load("nist_mt")
        results_nist = nist_mt.compute(predictions = predictions, references = reference)
        nist_score = results_nist['nist_mt']
        # evaluate meteor
        meteor = evaluate.load('meteor')
        results_met = meteor.compute(predictions=predictions, references=references)
        met_score = results_met['meteor']
else:
    for i in range(100):
        count += 1
        predictions = _generate(test_tok_dataset[i]['translation'][prompt])
        print("The model generates: ", predictions)
        print("Human reference: ", test_tok_dataset[i]['translation'][prompt]+test_tok_dataset[i]['translation'][conti])
        reference =[]
        reference.append(test_tok_dataset[i]['translation'][prompt]+test_tok_dataset[i]['translation'][conti])

        references = []
        references.append(reference)
        # evaluate bleu
        bleu = evaluate.load("bleu")
        results_bleu = bleu.compute(predictions=predictions, references=references)
        bleu_score += results_bleu['bleu']
        # evaluate rouge
        rouge = evaluate.load("rouge")
        results_rouge = rouge.compute(predictions=predictions, references=references)
        rouge_score += results_rouge['rougeL']
        # evaluate nist
        nist_mt = evaluate.load("nist_mt")
        results_nist = nist_mt.compute(predictions = predictions, references = reference)
        nist_score = results_nist['nist_mt']
        # evaluate meteor
        meteor = evaluate.load('meteor')
        results_met = meteor.compute(predictions=predictions, references=references)
        met_score = results_met['meteor']

print("bleu: ", bleu_score/count)
print("rouge: ", rouge_score/count)
print("nist: ", nist_score/count)
print("met: ", met_score/count)

# evaluate ppl
eval_results = trainer.evaluate()





# RuntimeError: CUDA error: device-side assert triggered
# CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.






# if task == 'e2e_nlg':
#     predictions = _generate(val_tok_dataset[0]['meaning_representation'])
#     print("The model generates: ", predictions)
#     print("Human reference: ", val_tok_dataset[0]['meaning_representation']+val_tok_dataset[0]['human_reference'])
#     reference =[]
#     reference.append(val_tok_dataset[0]['meaning_representation']+val_tok_dataset[0]['human_reference'])
# elif task =='cnn':
#     predictions = _generate(val_tok_dataset[0]['article'])
#     print("The model generates: ", predictions)
#     print("Human reference: ", val_tok_dataset[0]['article']+val_tok_dataset[0]['highlights'])
#     reference =[]
#     reference.append(val_tok_dataset[0]['article']+val_tok_dataset[0]['highlights'])
# references = []
# references.append(reference)
# # evaluate bleu
# bleu = evaluate.load("bleu")
# results_bleu = bleu.compute(predictions=predictions, references=references)
# print(results_bleu)
# # evaluate rouge
# rouge = evaluate.load("rouge")
# results_rouge = rouge.compute(predictions=predictions, references=references)
# print(results_rouge)
# # evaluate nist
# breakpoint()
# nist_mt = evaluate.load("nist_mt")
# results_nist = nist_mt.compute(predictions = predictions, references = reference)
# print(results_nist)
# # evaluate meteor
# meteor = evaluate.load('meteor')
# results_met = meteor.compute(predictions=predictions, references=references)
# print(results_met)

# # evaluate ppl
# eval_results = trainer.evaluate()

# # 'bleu': 0.0, 'precisions': [0.26666666666666666, 0.10344827586206896, 0.03571428571428571, 0.0], 'brevity_penalty': 1.0, 'length_ratio': 1.2, 'translation_length': 30, 'reference_length': 25}
# # {'rouge1': 0.46153846153846156, 'rouge2': 0.21621621621621623, 'rougeL': 0.30769230769230765, 'rougeLsum': 0.30769230769230765}



# #try to  use auto model
