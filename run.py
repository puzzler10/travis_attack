#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Imports and environment variables 
import os
import torch
import wandb
from travis_attack.utils import set_seed, set_session_options, setup_logging, setup_parser, resume_wandb_run, display_all
from travis_attack.config import Config
from travis_attack.models import prepare_models, get_optimizer
from travis_attack.data import ProcessedDataset
from travis_attack.trainer import Trainer
from travis_attack.insights import (postprocess_df, create_and_log_wandb_postrun_plots, get_training_dfs)
from fastcore.basics import in_jupyter

import logging 
logger = logging.getLogger("run")


# In[ ]:


cfg = Config()  # default values
if not in_jupyter():  # override with any script options
    parser = setup_parser()
    newargs = vars(parser.parse_args())
    for k,v in newargs.items(): 
        if v is not None: setattr(cfg, k, v)
if cfg.use_small_ds:  cfg = cfg.small_ds()
set_seed(cfg.seed)
set_session_options()
setup_logging(cfg, disable_other_loggers=True)
vm_tokenizer, vm_model, pp_tokenizer, pp_model, ref_pp_model, sts_model, nli_tokenizer, nli_model, cfg = prepare_models(cfg)
optimizer = get_optimizer(cfg, pp_model)
ds = ProcessedDataset(cfg, vm_tokenizer, vm_model, pp_tokenizer, sts_model, load_processed_from_file=True)


# In[5]:


# orig_l = ['hello my name is tom', "i like this movie a lot"]
# pp = ['hi I am a fine guy called tom', "this movie is really good"]

# def get_ref_logprobs(self, orig_l, pp_l): 
#     #batch_size = len(orig_l) 
#     orig_input_ids = self.pp_tokenizer(orig_l, return_tensors='pt', padding=True, truncation=True).input_ids
#     pp_input_ids   = self.pp_tokenizer(pp_l,   return_tensors='pt', padding=True, truncation=True).input_ids
#     decoder_start_token_ids = torch.tensor([ref_pp_model.config.decoder_start_token_id]).repeat(batch_size,1)
#     pp_input_ids = torch.cat([decoder_start_token_ids, pp_input_ids], 1)
#     logprobs = []
#     for i in range(pp_input_ids.shape[1] - 1): 
#         decoder_input_ids = pp_input_ids[:, 0:(i+1)]
#         outputs = ref_pp_model(input_ids=orig_input_ids, decoder_input_ids=decoder_input_ids)
#         token_logprobs = outputs.logits[:,i,:].log_softmax(1)
#         pp_next_token_ids = pp_input_ids[:,i+1].unsqueeze(-1)
#         pp_next_token_logprobs = torch.gather(token_logprobs,1,pp_next_token_ids).detach().squeeze(-1)
#         logprobs.append(pp_next_token_logprobs)
#     logprobs = torch.stack(logprobs, 1)   
#     attention_mask = ref_pp_model._prepare_attention_mask_for_generation(
#                 pp_input_ids[:,1:], self.pp_tokenizer.pad_token_id, self.pp_tokenizer.eos_token_id)
#     logprobs = logprobs * attention_mask
#     logprobs_sum = logprobs.sum(1)
#     return logprobs_sum


# In[6]:


# from transformers import BartTokenizer, BartForConditionalGeneration
# import torch

# model_name = "sshleifer/distilbart-cnn-6-6"
# tokenizer = BartTokenizer.from_pretrained(model_name)
# model = BartForConditionalGeneration.from_pretrained(model_name)

# text = """The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."""

# input_ids = tokenizer(text, return_tensors="pt").input_ids

# decoder_input_ids = [model.config.decoder_start_token_id]
# predicted_ids = []
# for i in range(20): 
#     outputs = model(input_ids=input_ids, decoder_input_ids=torch.tensor([decoder_input_ids]))
#     logits = outputs.logits[:,i,:]
#     # perform argmax on the last dimension (i.e. greedy decoding)
#     predicted_id = logits.argmax(-1)
#     predicted_ids.append(predicted_id.item())
#     print(tokenizer.decode([predicted_id.squeeze()]))
#     # add predicted id to decoder_input_ids
#     decoder_input_ids = decoder_input_ids + [predicted_id]


# In[7]:


cfg.wandb['mode'] = 'online'
trainer = Trainer(cfg, vm_tokenizer, vm_model, pp_tokenizer, pp_model, ref_pp_model, sts_model, nli_tokenizer, nli_model, optimizer,
                  ds, initial_eval=False, use_cpu=False)
trainer.train()


# In[8]:


get_ipython().run_line_magic('debug', '')


# In[11]:


df_d = get_training_dfs(cfg.path_run, postprocessed=False)
for k, df in df_d.items(): 
    df_d[k] = postprocess_df(df, filter_idx=None, num_proc=1)
    df_d[k].to_pickle(f"{cfg.path_run}{k}_postprocessed.pkl")    
create_and_log_wandb_postrun_plots(df_d)
trainer.run.finish()


# In[ ]:




