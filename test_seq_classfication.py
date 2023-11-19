import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2ForSequenceClassification.from_pretrained('gpt2')
model.eval()

tokenizer.pad_token = tokenizer.eos_token


def predict_proba(texts):
    outputs = []
    for text in texts:
        inputs = tokenizer.encode_plus(text, return_tensors='pt', truncation=True, padding='max_length', max_length=tokenizer.model_max_length)
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        outputs.append(probabilities.squeeze(0))
    return np.array(outputs)

explainer = LimeTextExplainer(class_names=["A", "B", "C", "D"])

text_to_explain = "Your example text here"

exp = explainer.explain_instance(text_to_explain, predict_proba, num_features=6)

print(exp.as_list())