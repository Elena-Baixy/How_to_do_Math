import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2', num_labels=4)
model = GPT2ForSequenceClassification(config)
model.eval()

tokenizer.pad_token = tokenizer.eos_token


def predict_proba(texts):
    outputs = []
    predicted_classes = []
    for text in texts:
        inputs = tokenizer.encode_plus(text, return_tensors='pt', truncation=True, padding='max_length', max_length=tokenizer.model_max_length)
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        outputs.append(probabilities.squeeze(0))
        predicted_class = np.argmax(probabilities, axis=1)
        predicted_classes.append(predicted_class[0])
    return np.array(outputs),predicted_classes

explainer = LimeTextExplainer(class_names=["A", "B", "C", "D"])

text_to_explain = "What's the sum of 1 and 2? A.3 B.4 C.5 D.6"

exp = explainer.explain_instance(text_to_explain, lambda x: predict_proba(x)[0], num_features=6, num_samples=50)

print(exp.as_list())