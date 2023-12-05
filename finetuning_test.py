import os
import torch
import logging
from datasets import load_dataset, concatenate_datasets 
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments, GPT2Config, DataCollatorWithPadding, DataCollatorForLanguageModeling
import transformers
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
transformers.logging.set_verbosity_info()

logs_dir = "/scratch/eecs487f23_class_root/eecs487f23_class/yanran/logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Load tokenizer and model
configuration = GPT2Config()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2ForSequenceClassification.from_pretrained("gpt2-medium", num_labels=5)
model.config.pad_token_id = model.config.eos_token_id

# Load dataset
# dataset = load_dataset("math_qa", split="validation")  # use train split when finetuning
# train_dataset = load_dataset("math_qa", split="train")
# eval_dataset = load_dataset("math_qa", split="validation")


# Load all splits of the MathQA dataset
train_dataset = load_dataset("math_qa", split="train")
validation_dataset = load_dataset("math_qa", split="validation")
eval_dataset = load_dataset("math_qa", split="test")
combined_train_dataset = concatenate_datasets([train_dataset, validation_dataset])

# # Load the dataset and split it into training and validation sets
# full_dataset = load_dataset("math_qa", split="validation") # change the split later
# train_size = 0.9
# train_dataset, eval_dataset = full_dataset.train_test_split(train_size=train_size).values()

# Tokenize and preprocess the dataset
def preprocess_data(examples):
    inputs = [problem + " Options: " + options for problem, options in zip(examples['Problem'], examples['options'])]
    # inputs = [problem + " Formula: " + formula + " Options: " + options
             # for problem, formula, options in zip(examples['Problem'], examples['annotated_formula'], examples['options'])]
    labels = [ord(correct_option.lower()) - ord('a') for correct_option in examples['correct']]  # 'a' -> 0, 'b' -> 1, etc.
    return {'input_ids': tokenizer(inputs, truncation=True)['input_ids'], 'labels': labels}

# tokenized_dataset = dataset.map(preprocess_data, batched=True)
tokenized_train_dataset = combined_train_dataset.map(preprocess_data, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_data, batched=True)
# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# # Tokenize and preprocess the dataset
# def preprocess_data(examples):
#     inputs = [problem + " Options: " + options for problem, options in zip(examples['Problem'], examples['options'])]
#     labels = [ord(correct_option.lower()) - ord('a') for correct_option in examples['correct']]
#     return {'input_ids': tokenizer(inputs, truncation=True)['input_ids'], 'labels': labels}
# tokenized_dataset = dataset.map(preprocess_data, batched=True)
# # Data collator
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False  # Masked Language Model not used in sequence classification
# )



# def optimizer_init(model):
#     return AdamW(model.parameters(), lr=5e-4)

# def scheduler_init(optimizer):
#     return ReduceLROnPlateau(optimizer, 'min')

# Training arguments
training_args = TrainingArguments(
    output_dir="/scratch/eecs487f23_class_root/eecs487f23_class/yanran/results",           # Output directory for model checkpoints
    num_train_epochs=3,               # Number of training epochs
    per_device_train_batch_size=2,   # Batch size for training
    per_device_eval_batch_size=2,    # Batch size for evaluation
    warmup_steps=500,                 # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                # Weight decay rate
    logging_dir=logs_dir,             # Directory for storing logs
    logging_steps=50,                   # Log every 50 steps
    evaluation_strategy="epoch",       # Evaluate each epoch
    learning_rate=5e-3,               # Learning rate
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    # optimizers=(optimizer_init, scheduler_init),
)

# check tokenized dataset
for i in range(5):
    sample = tokenized_train_dataset[i]
    print(f"Sample {i}:")
    print("Tokenized Input IDs:", sample['input_ids'])
    print("Length of Tokenized Input:", len(sample['input_ids']))
    print("Label:", sample['labels'])
    print("\n")
for i in range(5):
    sample = tokenized_eval_dataset[i]
    print(f"Sample {i}:")
    print("Tokenized Input IDs:", sample['input_ids'])
    print("Length of Tokenized Input:", len(sample['input_ids']))
    print("Label:", sample['labels'])
    print("\n")

# torch.cuda.empty_cache()

# Train the model
trainer.train()

# Save the model
model.save_pretrained('/scratch/eecs487f23_class_root/eecs487f23_class/yanran/results/fine_tuned_gpt2')
tokenizer.save_pretrained('/scratch/eecs487f23_class_root/eecs487f23_class/yanran/results/fine_tuned_gpt2')

# Run the baseline with fine-tuned model

# Load the dataset
# dataset_baseline = load_dataset("math_qa", split="validation")

# # Load the fine-tuned tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained('/scratch/eecs487f23_class_root/eecs487f23_class/yanran/results/fine_tuned_gpt2')
# model = GPT2ForSequenceClassification.from_pretrained('/scratch/eecs487f23_class_root/eecs487f23_class/yanran/results/fine_tuned_gpt2', num_labels=5)
# model.eval()
# tokenizer.pad_token = tokenizer.eos_token

# # Prediction function
# def predict(text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     return torch.argmax(logits, dim=1)

# answer_mapping = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

# # Evaluate predictions
# correct = 0
# num_example = 100
# for i in range(num_example):
#     prompt = dataset_baseline[i]["Problem"] + " " + dataset_baseline[i]["options"]
#     prediction_index = predict(prompt).item()
#     prediction_label = answer_mapping[prediction_index]
#     print(prediction_label)
#     correct_answer = dataset_baseline[i]['correct']
#     if prediction_label == correct_answer:
#         correct += 1

# accuracy = correct / num_example
# print(f"Accuracy: {accuracy:.2f}")