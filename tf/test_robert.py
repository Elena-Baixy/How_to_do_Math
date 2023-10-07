import argparse
import os

from transformers import GPT2ForSequenceClassification, GPT2TokenizerFast, GPT2LMHeadModel, RobertaForSequenceClassification, RobertaTokenizer
from datasets import load_dataset, concatenate_datasets
import evaluate
import numpy as np
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

import torch
import torch.nn as nn

import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='')
# Dataset
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--PLM', type=str, default='roberta-base')
parser.add_argument('--task', type=str, default='rte')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--ckpt_path', type=str)

args = parser.parse_args()

# Load the pre-trained tokenizer
tokenizer = RobertaTokenizer.from_pretrained(args.PLM)

#################################### Prepare Data ####################################
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", args.task)
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[args.task]


print(f'NLP task: {args.task}')
raw_dataset = load_dataset("glue", args.task)

def encode(examples):
    text_pattern = (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    return tokenizer(*text_pattern, truncation=True, padding=False)

if args.task == 'stsb':
    num_labels = 1
else:
    label_list = raw_dataset["train"].features["label"].names
    num_labels = len(label_list)
print(num_labels)

encoded_dataset = raw_dataset.map(encode, batched=True)
encoded_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
#################################### Prepare Data End ####################################

# Load the pre-trained GPT-2 model
model = RobertaForSequenceClassification.from_pretrained(args.PLM, num_labels=num_labels)



training_args = TrainingArguments(
    output_dir=f'./results/{args.task}',
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=32,
    warmup_ratio=0.06,
    weight_decay=0.1,
    learning_rate = args.lr,
    logging_dir='./logs',
    report_to="none",
    save_total_limit=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # load_best_model_at_end=True,
)
print(training_args)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

if args.task == 'mnli':
    val_dataset = concatenate_datasets([encoded_dataset["validation_matched"], encoded_dataset["validation_mismatched"]])
else:
    val_dataset = encoded_dataset["validation"]
print(val_dataset)
print(type(val_dataset))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

trainer.train()
torch.save(model.state_dict(), 'finetuend_model.pt')
