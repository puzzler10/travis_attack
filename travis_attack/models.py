# AUTOGENERATED! DO NOT EDIT! File to edit: 07_models.ipynb (unless otherwise specified).

__all__ = ['logger', 'prepare_models', 'pp_model_freeze_layers', 'save_pp_model', 'resume_pp_model', 'get_vm_probs',
           'get_optimizer', 'get_nli_probs', 'get_cola_probs']

# Cell
import numpy as np, torch
from transformers import (AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer)

from sentence_transformers import SentenceTransformer
from types import MethodType
from undecorated import undecorated
from .config import Config
from IPython.core.debugger import set_trace

import logging
logger = logging.getLogger("travis_attack.models")

# Cell
def _prepare_pp_tokenizer_and_model(cfg):
    """As well as preparing the pp model and tokenizer this function also adds a new method `generate_with_grad` to
    the pp model so that we can backprop when generating."""
    # PEGASUS takes about 3GB memory space up on the GPU
    # change the `local_files_only` argument if changing the model name
    pp_model = _load_pp_model(cfg)
    pp_model.train()
    pp_model_freeze_layers(cfg, pp_model)  # dictated by cfg.unfreeze_last_n_layers; set to "all" to do no freezing
    generate_with_grad = undecorated(pp_model.generate)      # removes the @no_grad decorator from generate so we can backprop
    pp_model.generate_with_grad = MethodType(generate_with_grad, pp_model)
    pp_tokenizer = AutoTokenizer.from_pretrained(cfg.pp_name)
    return pp_tokenizer, pp_model

def _load_pp_model(cfg):
    if cfg.using_t5():  pp_model = AutoModelForSeq2SeqLM.from_pretrained(cfg.pp_name, local_files_only=True).to(cfg.device)
    else:               pp_model = AutoModelForSeq2SeqLM.from_pretrained(cfg.pp_name, local_files_only=True, max_position_embeddings = cfg.orig_max_length + 10).to(cfg.device)
    return pp_model

def _prepare_vm_tokenizer_and_model(cfg):
    vm_tokenizer = AutoTokenizer.from_pretrained(cfg.vm_name)
    vm_model = AutoModelForSequenceClassification.from_pretrained(cfg.vm_name, local_files_only=True).to(cfg.device)
    vm_model.eval()
    return vm_tokenizer, vm_model

def _prepare_sts_model(cfg):
    sts_model = SentenceTransformer(cfg.sts_name).to(cfg.device)
    return sts_model

def _prepare_nli_tokenizer_and_model(cfg):
    nli_tokenizer = AutoTokenizer.from_pretrained(cfg.nli_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(cfg.nli_name, local_files_only=True).to(cfg.device)
    nli_model.eval()
    return nli_tokenizer, nli_model

def _prepare_cola_tokenizer_and_model(cfg):
    cola_tokenizer = AutoTokenizer.from_pretrained(cfg.cola_name)
    cola_model = AutoModelForSequenceClassification.from_pretrained(cfg.cola_name, local_files_only=True).to(cfg.device)
    cola_model.eval()
    return cola_tokenizer, cola_model

# def _pad_model_token_embeddings(cfg, pp_model, vm_model, sts_model):
#     """Resize first/embedding layer of all models to be a multiple of cfg.embedding_padding_multiple.
#     Good for tensor core efficiency when using fp16 (which we aren't...).
#     Makes changes to models in-place."""
#     def pad_token_embeddings_to_multiple_of_n(model, n):
#         def get_new_vocab_size(model): return int((np.floor(model.config.vocab_size / n) + 1) * n)
#         model.resize_token_embeddings(get_new_vocab_size(model))
#     pad_token_embeddings_to_multiple_of_n(pp_model, cfg.embedding_padding_multiple)
#     pad_token_embeddings_to_multiple_of_n(vm_model, cfg.embedding_padding_multiple)
#     # sts_model is from SentenceTransformers so needs a bit of unwrapping to access the base huggingface model
#     pad_token_embeddings_to_multiple_of_n(sts_model._first_module().auto_model, cfg.embedding_padding_multiple)

def _update_config(cfg, vm_model, pp_model):
    cfg.vm_num_labels = vm_model.num_labels
    cfg.vocab_size = pp_model.get_input_embeddings().num_embeddings   # unlike pp_tokenizer.vocab_size this includes the padding
    if   cfg.nli_name  == "microsoft/deberta-base-mnli"     :   cfg.contra_label = 0
    elif cfg.nli_name  == "howey/electra-small-mnli"        :   cfg.contra_label = 2
    if   cfg.cola_name == "textattack/albert-base-v2-CoLA"  :   cfg.cola_positive_label = 1
    return cfg

def prepare_models(cfg):
    """Load tokenizers and models for vm, pp, sts.
    Pad the first embedding layer if specified in the config.
    Update config with some model-specific variables.
    """
    vm_tokenizer, vm_model =  _prepare_vm_tokenizer_and_model(cfg)
    pp_tokenizer, pp_model =  _prepare_pp_tokenizer_and_model(cfg)
    ref_pp_model = _load_pp_model(cfg).eval()
    sts_model = _prepare_sts_model(cfg)
    nli_tokenizer, nli_model   = _prepare_nli_tokenizer_and_model(cfg)
    cola_tokenizer, cola_model = _prepare_cola_tokenizer_and_model(cfg)
 #   if cfg.pad_token_embeddings:  _pad_model_token_embeddings(cfg, pp_model, vm_model, sts_model)
    cfg = _update_config(cfg, vm_model, pp_model)
    return vm_tokenizer, vm_model, pp_tokenizer, pp_model, ref_pp_model, sts_model, nli_tokenizer, nli_model, cola_tokenizer, cola_model, cfg

def _get_layers_to_unfreeze(cfg):
    """Return a list that determines which layers should be kept unfrozen"""
    if cfg.pp_name   == "tuner007/pegasus_paraphrase":
        unfreeze_layer_list,last_layer_num = ['decoder.layer_norm'],          15
    elif cfg.pp_name == "tdopierre/ProtAugment-ParaphraseGenerator":
        unfreeze_layer_list,last_layer_num = ['decoder.layernorm_embedding'],  5
    elif cfg.pp_name == "eugenesiow/bart-paraphrase":
        unfreeze_layer_list,last_layer_num = ['decoder.layernorm_embedding'], 11
    for i in range(last_layer_num, last_layer_num-cfg.unfreeze_last_n_layers, -1):
        unfreeze_layer_list.append(f'decoder.layers.{i}')
    # self.lm_head is tied (the same parameter as) to self.encoder.embed_tokens and self.decoder.embed_tokens.
    # and this is given by shared.weight
    # From here: https://github.com/huggingface/transformers/issues/10479#issuecomment-788964822
    unfreeze_layer_list.append('shared.weight')
    return unfreeze_layer_list

def pp_model_freeze_layers(cfg, pp_model):
    """Freeze all layers of pp_model except the last few decoder layers (determined by cfg.unfreeze_last_n_layers),
    the final layer_norm layer, and the linear head (which is tied to the input embeddings). """
    if cfg.unfreeze_last_n_layers == "all":
        for i, (name, param) in enumerate(pp_model.named_parameters()): param.requires_grad = True
    else:
        unfreeze_layer_list = _get_layers_to_unfreeze(cfg)
        for i, (name, param) in enumerate(pp_model.base_model.named_parameters()):
            if np.any([o in name for o in unfreeze_layer_list]):   param.requires_grad = True
            else:                                                  param.requires_grad = False
    return pp_model


def save_pp_model(pp_model, optimizer, path):
    """Save training state (for both pp_model and optimiser) as a checkpoint at a given epoch. """
    torch.save({'pp_model_state_dict': pp_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, path)

def resume_pp_model(pp_model, optimizer, path):
    """Replace the training state with a saved checkpoint.. Reinitialises both pp_model and optimiser state. """
    state = torch.load(path)
    pp_model.load_state_dict( state['pp_model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    return pp_model, optimizer

def get_vm_probs(text, cfg, vm_tokenizer, vm_model, return_predclass=False):
    """Get victim model predictions for a batch of text.
    Used in data cleaning and by the reward_fn to get vm_score."""
    if vm_model.training: vm_model.eval()
    with torch.no_grad():
        tkns = vm_tokenizer(text, truncation=True, padding=True, pad_to_multiple_of=cfg.orig_padding_multiple,
                            return_tensors="pt").to(cfg.device)
        logits = vm_model(**tkns).logits
        probs = torch.softmax(logits,1)
        if return_predclass:    return probs, torch.argmax(probs,1)
        else:                   return probs

def get_optimizer(cfg, pp_model):  return torch.optim.AdamW(pp_model.parameters(), lr=cfg.lr)

# def get_start_end_special_token_ids(tokenizer):
#     """Return a dict indicating the token id's that input/output sequences should start and end with."""
#     d = {}
#     if tokenizer.name_or_path in ['eugenesiow/bart-paraphrase', 'tdopierre/ProtAugment-ParaphraseGenerator']:
#         d["input_start_id"] =  tokenizer.bos_token_id
#         d["input_end_id"] =  [tokenizer.pad_token_id, tokenizer.eos_token_id]
#         d["output_start_id"] =  tokenizer.eos_token_id
#         d["output_end_id"] =  [tokenizer.pad_token_id, tokenizer.eos_token_id]
#     elif tokenizer.name_or_path == "tuner007/pegasus_paraphrase":
#         d["input_start_id"] =  None
#         d["input_end_id"] =  [tokenizer.pad_token_id, tokenizer.eos_token_id]
#         d["output_start_id"] =  tokenizer.pad_token_id
#         d["output_end_id"] =  [tokenizer.pad_token_id, tokenizer.eos_token_id]
#     elif tokenizer.name_or_path in ["prithivida/parrot_paraphraser_on_T5", "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"]:
#         d["input_start_id"] =  None
#         d["input_end_id"] =  [tokenizer.pad_token_id, tokenizer.eos_token_id]
#         d["output_start_id"] =  tokenizer.pad_token_id
#         d["output_end_id"] =  [tokenizer.pad_token_id, tokenizer.eos_token_id]
#     else:
#         raise Exception("unrecognised tokenizer")
#     return d

def get_nli_probs(orig_l, pp_l, cfg, nli_tokenizer, nli_model):
    inputs = nli_tokenizer(orig_l, pp_l, return_tensors="pt", padding=True, truncation=True).to(cfg.device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = logits.softmax(1)
    return probs

def get_cola_probs(pp_l, cfg, cola_tokenizer, cola_model):
    inputs = cola_tokenizer(pp_l, return_tensors="pt", padding=True, truncation=True).to(cfg.device)
    with torch.no_grad():
        logits = cola_model(**inputs).logits
        probs = logits.softmax(1)
    return probs
