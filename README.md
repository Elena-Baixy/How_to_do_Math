# Sparsity and Activation

Download requirements through
```
pip install -r requirements.txt
```


## Test Activation
To change the number of example to test,
1. go to test_gpt.py to change the num_example
2. go to modeling_gpt2.py file, search "TODO: parameters for #layer and #example", change NUM_LAYER, NUM_EXAM according to the model and the examples you choose

To run the file, run (eg.gpt will be test_gpt.py)
```
python test_gpt.py
```
output for key will be in the directory named "visualized_key", while output for value will be in the directory named "visualized_value"

To change the head and the layer you want to print, search "TODO: print activation", change j will change the layer, change t will change the head.


## Test residual stream

To plot the residual stream
```
python reasoning_test.py
```

Then you can see the probability change of the next word prediction onto different layers. 

# Fine-tuning GPT2

To fine-tune GPT2 model for classification task on MathQA dataset

```
python finetuning_test.py
```

# Baseline

To get the baseline of GPT2 model running on MathQA dataset with different prompts before fine-tuning, run the notebook `baseline.ipynb`. Then you can see the prediction accuracy for a certain number of examples.

Each cell corresponds to a type of prompt. Change the value of "num_example" to adjust the number of examples you want to test. 

# Test with LIME

To test the performance of GPT2 on MathQA with LIME, go to the `lime_gpt2_mathqa.ipynb` and run the script.