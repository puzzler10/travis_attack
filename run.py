#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[108]:


cfg = Config()  # default values
if not in_jupyter():  # override with any -- options when running with command line
    parser = setup_parser()
    newargs = vars(parser.parse_args())
    for k,v in newargs.items(): 
        if v is not None: 
            if k in cfg.pp.keys():  cfg.pp[k] = v
            else:                   setattr(cfg, k, v)
if cfg.use_small_ds:  cfg = cfg.small_ds()
set_seed(cfg.seed)
set_session_options()
setup_logging(cfg, disable_other_loggers=True)
vm_tokenizer, vm_model, pp_tokenizer, pp_model, ref_pp_model, sts_model, nli_tokenizer, nli_model, cfg = prepare_models(cfg)
optimizer = get_optimizer(cfg, pp_model)
ds = ProcessedDataset(cfg, vm_tokenizer, vm_model, pp_tokenizer, sts_model, load_processed_from_file=False)


# In[100]:


cfg.eval_gen_params['sampling']


# In[101]:


split='valid'
data = next(iter(ds.dld_tkn[split])) 
raw  = next(iter(ds.dld_raw[split]))


# In[106]:


pp_output = pp_model.generate(
                    input_ids=data['input_ids'], attention_mask=data['attention_mask'], 
                    **cfg.eval_gen_params['sampling'],   remove_invalid_values=False, 
                    pad_token_id = pp_tokenizer.pad_token_id,eos_token_id = pp_tokenizer.eos_token_id)
pp_l = pp_tokenizer.batch_decode(pp_output, skip_special_tokens=True)


# In[107]:


pp_l


# In[77]:


raw


# In[5]:


cfg.wandb['mode'] = 'disabled'
trainer = Trainer(cfg, vm_tokenizer, vm_model, pp_tokenizer, pp_model, ref_pp_model, sts_model, nli_tokenizer, nli_model, optimizer,
                  ds, initial_eval=False, use_cpu=False)
print(vars(cfg))
trainer.train()


# In[7]:


df_d = get_training_dfs(cfg.path_run, postprocessed=False)
for k, df in df_d.items(): 
    df_d[k] = postprocess_df(df, filter_idx=None, num_proc=1)
    df_d[k].to_pickle(f"{cfg.path_run}{k}_postprocessed.pkl")    
create_and_log_wandb_postrun_plots(df_d)
trainer.run.finish()
#run.finish()

