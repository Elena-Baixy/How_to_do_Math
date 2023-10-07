# Sparsity and Activation

Download requirements through
```
pip install -r requirements.txt
```

## Test sparsity

To change the number of example to test,
1. go to test_* file to change the num_example
2. go to modeling_* file, search "TODO: parameters for #layer and #example", change NUM_LAYER, NUM_EXAM according to the model and the examples you choose

To run the file, run (eg.gpt will be test_gpt.py)
```
python test_bert.py
```

output will be the average of each layer's sparsity through #examples.

if you want to change the code for sparsity, search "TODO: sparsity"

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


## Sparsity GeLU pruning

To finetune the roberta model, run
```
CUDA_VISIBLE_DEVICES=1 python -u test_robert.py --lr 1e-5 --train_batch_size 16 --task mrpc| tee moe_result/mrpc-finetuned.log
```

It will save the model to the root repo, as finetuned_model_{task}.pt

To test different task, run
```
CUDA_VISIBLE_DEVICES=1 python roberta_evaluation.py --lr 1e-5 --train_batch_size 16 --task mrpc --sparsity 0.3 | tee moe_result/mrpc-sparsity.log 
```