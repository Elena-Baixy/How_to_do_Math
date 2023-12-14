'''
bias
'''
import plotly.io as pio
from matplotlib import pyplot as plt
import numpy as np
import json
import torch
import torch.nn.functional as F
import transformer_lens
import transformer_lens.utils as tlu
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from IPython.display import HTML
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import partial
from fancy_einsum import einsum
from emotions.emo_utils import aggregate_resid_stream
from repetitions.utils import GPT2_PAD_IDX
from utils import load_hooked_model, load_tokenizer
from constants import PROBES_DIR

TOPK=20
torch.set_grad_enabled(False)
## Helper and Visual Functions
def imshow(tensor, renderer=None, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_mimath2int=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show(renderer)


def line(tensor, renderer=None, **kwargs):
    px.line(y=tlu.to_numpy(tensor), **kwargs).show(renderer)


#def two_lines(math2, male, diff):
def two_lines(math2, male):
    assert math2.shape == male.shape
    fig = make_subplots(rows=1, cols=math2.shape[1])

    for idx in range(math2.shape[1]):
        print(math2[:, idx].shape)
        fig.add_trace(
            go.Scatter(x=list(np.arange(math2.shape[0])), y=math2[:, idx], name = 'Orignal'),
            row=1,
            col=idx + 1,
        )
        fig.add_trace(
            go.Scatter(x=list(np.arange(math2.shape[0])), y=male[:, idx], name = 'Combination'),
            row=1,
            col=idx + 1,
        )
        fig.update_layout(
            title='Probability of answer with different layers',  # Add a plot title if you like
            yaxis_title='Prob',  # X-axis label
            xaxis_title='Layers'   # Y-axis label
        )
    fig.show()


def main(model,config):
    # layer = config['layer']
    plain = 'Answer:'
    answer = ' a'
    prompt_male ="Problem: the banker ' s gain of a certain sum due 3 years hence at 10 % per annum is rs . 36 . what is the present worth ?  annotated_formula: divide(multiply(const_100, divide(multiply(36, const_100), multiply(3, 10))), multiply(3, 10)), options: a ) rs . 400 , b ) rs . 300 , c ) rs . 500 , d ) rs . 350 , e ) none of these Answer: "
    answer_male = ' a'
    prompt_math2="Problem: the banker ' s gain of a certain sum due 3 years hence at 10 % per annum is rs . 36 . what is the present worth ? options: 'a ) rs . 400 , b ) rs . 300 , c ) rs . 500 , d ) rs . 350 , e ) none of these Answer:"
    answer_math2 = ' a'
    # prompt_male="Problem: the banker ' s gain of a certain sum due 3 years hence at 10 % per annum is rs . 36 . what is the present worth ? Rationale:e xplanation : t = 3 years r = 10 % td = ( bg * 100 ) / tr = ( 36 * 100 ) / ( 3 * 10 ) = 12 * 10 = rs . 120 td = ( pw * tr ) / 100 ⇒ 120 = ( pw * 3 * 10 ) / 100 = 1200 = pw * 3 pw = 1200 / 3 = rs . 400 options: 'a ) rs . 400 , b ) rs . 300 , c ) rs . 500 , d ) rs . 350 , e ) none of these Answer:"
    # prompt_male="nnotated_formula: divide(multiply(const_100, divide(multiply(36, const_100), multiply(3, 10))), multiply(3, 10)), options: a ) rs . 400 , b ) rs . 300 , c ) rs . 500 , d ) rs . 350 , e ) none of these Answer: "

    tlu.test_prompt(plain,answer,model,prepend_bos=True)
    tlu.test_prompt(prompt_math2, answer_math2, model, prepend_bos=True)
    tlu.test_prompt(prompt_male, answer_male, model, prepend_bos=True)
    tokens_math2 = model.to_tokens(prompt_math2, prepend_bos=True)
    tokens_male = model.to_tokens(prompt_male, prepend_bos=True)
    idx_correct = model.to_tokens(' a', prepend_bos=True)[0][1].item() 
    idx_correct = model.to_tokens(' a', prepend_bos=True)[0][1].item() # 15849
    idx_wrong = model.to_tokens(' b', prepend_bos=True)[0][1].item()
    breakpoint()
    ###math2
    print("---------------------math2 Prompt---------------------")
    with torch.inference_mode():
        math2_logits, math2_cache = model.run_with_cache(tokens_math2)
        male_logits, male_cache = model.run_with_cache(tokens_male)
        
    math2_resid_accum = math2_cache.accumulated_resid(
        layer=-1, incl_mid=True, apply_ln=True
    )
    male_resid_accum, accum_labels = male_cache.accumulated_resid(
        layer=-1, incl_mid=True, apply_ln=True, return_labels=True
    )

    math2_resid_decomp = math2_cache.decompose_resid(layer=-1, apply_ln=True)
    male_resid_decomp, labels = male_cache.decompose_resid(
        layer=-1, apply_ln=True, return_labels=True
    )

    # Project each layer and each position onto vocab space
    math2_vocab_proj = einsum(
        "layer batch pos d_model, d_model d_vocab --> layer batch pos d_vocab",
        math2_resid_accum,
        model.W_U,
    )
    male_vocab_proj = einsum(
        "layer batch pos d_model, d_model d_vocab --> layer batch pos d_vocab",
        male_resid_accum,
        model.W_U,
    )


    # Project each layer and each position onto vocab space
    math2_vocab_proj_decomp = einsum(
        "layer batch pos d_model, d_model d_vocab --> layer batch pos d_vocab",
        math2_resid_decomp,
        model.W_U,
    )
    male_vocab_proj_decomp = einsum(
        "layer batch pos d_model, d_model d_vocab --> layer batch pos d_vocab",
        male_resid_decomp,
        model.W_U,
    )

    math2_probs = math2_vocab_proj.softmax(dim=-1)
    male_probs = male_vocab_proj.softmax(dim=-1)

    math2_probs_decomp = math2_vocab_proj_decomp.softmax(dim=-1)
    male_probs_decomp = male_vocab_proj_decomp.softmax(dim=-1)
    # braekpoint()
    two_lines(
        math2_probs_decomp[:,:,-1, idx_correct].cpu(),
        male_probs_decomp[:,:,-1,idx_correct].cpu()
    )
    two_lines( 
        math2_probs[:, :, -1, idx_correct].cpu(),
        male_probs[:, :, -1, idx_correct].cpu(),
    )
    # two_lines(
    #     math2_probs[:, :, -1, idx_wrong].cpu(),
    #     math2_probs[:, :, -1, idx_correct].cpu(),
    # )
    # two_lines(
    #     male_probs[:, :, -1, idx_wrong].cpu(),
    #     male_probs[:, :, -1, idx_correct].cpu(),
    # )
    # two_lines(
    #     math2_probs[:, :, -1, idx_correct].cpu(),
    #     male_probs[:, :, -1, idx_correct].cpu(),
    # )
    # two_lines( 
    #     math2_probs[:, :, -1, idx_correct].cpu(),
    #     math2_probs[:, :, -1, idx_correct].cpu(),
    # )
    # two_lines( 
    #     male_probs[:, :, -1, idx_correct].cpu(),
    #     male_probs[:, :, -1, idx_correct].cpu(),
    # )
    breakpoint()
    vocab_space = model.W_U
    rank_math2_correct = []
    rank_math2_correct = []
    for layer in range(24):
        resid_post = math2_cache["resid_post", layer]
        proj_vocab = einsum(
            "batch d_model, d_model vocab -> batch vocab",
            resid_post[:,-1,:].clone(),
            vocab_space,
        ).softmax(-1)
        # the rank of 'correct' in this layer
        sorted_pred = proj_vocab[0].argsort(-1, descending=True)
        rank_math2_correct.append(torch.nonzero(sorted_pred == idx_correct).item())
        rank_math2_correct.append(torch.nonzero(sorted_pred == idx_correct).item())
        print(f"--------layer {layer}----------")
        print(f"The rank of ' correct' is {torch.nonzero(sorted_pred == idx_correct).item()}")
        print(f"The rank of  ' correct' is {torch.nonzero(sorted_pred == idx_correct).item()}")
        # breakpoint()
        print("------------------")
    sorted_final = (math2_logits.softmax(-1)[0,-1]).argsort(-1, descending=True)
    print(f"The rank of ' correct' is {torch.nonzero(sorted_final == idx_correct).item()}")
    print(f"The rank of  ' correct' is {torch.nonzero(sorted_final == idx_correct).item()}")
    


    tlu.test_prompt(prompt_male, answer_male, model, prepend_bos=True)
    tlu.test_prompt(prompt_male, answer_math2, model, prepend_bos=True)


    print("---------------------Male Prompt---------------------")
    rank_male_correct = []
    rank_male_correct = []
    for layer in range(24):
        resid_post = male_cache["resid_post", layer]
        proj_vocab = einsum(
            "batch d_model, d_model vocab -> vocab",
            resid_post[:,-1,:].clone(),
            vocab_space,
        )
        # the rank of 'correct' in this layer
        sorted_pred = proj_vocab.argsort(-1, descending=True)
        rank_male_correct.append(torch.nonzero(sorted_pred == idx_correct).item())
        rank_male_correct.append(torch.nonzero(sorted_pred == idx_correct).item())
        print(f"--------layer {layer}----------")
        print(f"The rank of ' correct' is {torch.nonzero(sorted_pred == idx_correct).item()}")
        print(f"The rank of  ' correct' is {torch.nonzero(sorted_pred == idx_correct).item()}")
        print("------------------")
    sorted_final = (male_logits.softmax(-1)[0,-1]).argsort(-1, descending=True)
    print(f"The rank of ' correct' is {torch.nonzero(sorted_final == idx_correct).item()}")
    print(f"The rank of  ' correct' is {torch.nonzero(sorted_final == idx_correct).item()}")
    

# the difference for math2 and male



if __name__ == "__main__":
    config = {
        "device": "cuda",
        "layer":19,
    }
    model = HookedTransformer.from_pretrained(
        "gpt2-medium",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    model.tokenizer.padding_side = "left"
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    main(model, config)
