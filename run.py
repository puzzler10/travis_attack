#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Imports and environment variables 
import os
import torch
import wandb
from travis_attack.utils import set_seed, set_session_options, setup_logging, setup_parser, resume_wandb_run, display_all, print_important_cfg_vars
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
cfg.wandb['mode'] = 'disabled'


# In[ ]:


cfg.wandb['mode'] = 'online'
trainer = Trainer(cfg, vm_tokenizer, vm_model, pp_tokenizer, pp_model, ref_pp_model, sts_model, nli_tokenizer, nli_model, optimizer,
                  ds, initial_eval=False, use_cpu=False)
print_important_cfg_vars(cfg)
trainer.train()


# In[ ]:


trainer.run.finish()


# In[ ]:


# from travis_attack.utils import unpack_nested_lists_in_df, merge_dicts
# from datasets import Dataset
# from sentence_transformers.util import pytorch_cos_sim
# import numpy as np, pandas as pd 
# from wandb.data_types import Histogram

# ## Params
# split='valid'

# ## Pre-epoch setup 
# pp_model = pp_model.to(cfg.device) # (not needed in trainer )
# # Data containers and data loaders
# eval_epoch_df_d = dict(train=[], valid=[], test=[]) # each eval epoch appended to here 


# for epoch in range(3): 
#     eval_batch_results = list()  # each eval batch appended to here, list of dicts
#     dl_key = "train_eval" if split == "train" else split
#     dl_raw = ds.dld_raw[dl_key]
#     dl_tkn = ds.dld_tkn[dl_key]
#     ## Loop through batches in eval set
#     for eval_batch_num, (data, raw) in enumerate(zip(dl_tkn, dl_raw)):
#         pp_output = pp_model.generate(
#                             input_ids=data['input_ids'].to(cfg.device), attention_mask=data['attention_mask'].to(cfg.device), 
#                             **cfg.eval_gen_params,   remove_invalid_values=False, 
#                             pad_token_id = pp_tokenizer.pad_token_id,eos_token_id = pp_tokenizer.eos_token_id)
#         pp_l = pp_tokenizer.batch_decode(pp_output, skip_special_tokens=True)
#         pp_l_nested = [pp_l[i:i+cfg.n_eval_seq] for i in range(0, len(pp_l), cfg.n_eval_seq)]
#         all([len(l) == cfg.n_eval_seq for l in pp_l_nested])  # make sure we generate the same number of paraphrases for each
#         eval_batch_results.append({'idx': raw['idx'], 'orig': raw['text'], 'pp_l':pp_l_nested, 'orig_n_letters': data['n_letters'].tolist(), 
#                               'label': raw['label'], 'orig_truelabel_probs': data['orig_truelabel_probs'].tolist(), 'orig_sts_embeddings': data['orig_sts_embeddings'] })

#     ## Convert eval batches to dataframes and create paraphrase identifier `pp_idx`
#     df = pd.DataFrame(eval_batch_results)    
#     df = df.apply(pd.Series.explode).reset_index(drop=True)  # This dataframe has one row per original example
#     def get_pp_idx(row): return ["orig_" + str(row['idx']) + "-epoch_" + str(epoch) +  "-pp_" +  str(pp_i) for pp_i in range(1, len(row['pp_l'])+1)]
#     df['pp_idx'] = df.apply(get_pp_idx, axis=1)

#     ## Create seperate dataframe for sts scores and expand original dataframe
#     df_sts = df[['pp_idx', 'pp_l', 'orig_sts_embeddings']] 
#     df1 = df.drop(columns='orig_sts_embeddings')
#     scalar_cols = [o for o in df1.columns if o not in ['pp_l', 'pp_idx']]
#     df_expanded = unpack_nested_lists_in_df(df1, scalar_cols=scalar_cols) # This dataframe has one row per paraphrase

#     ## Add vm_scores, sts_scores, pp_letter_diff, contradiction scores
#     ds_expanded = Dataset.from_pandas(df_expanded)
#     def add_vm_scores_eval(batch): 
#         output = trainer._get_vm_scores(pp_l=batch['pp_l'], labels=torch.tensor(batch['label'], device = cfg.device), 
#                                         orig_truelabel_probs=torch.tensor(batch['orig_truelabel_probs'], device=cfg.device))
#         for k, v in output.items(): batch[k] = v.cpu().tolist() 
#         return batch
#     def add_pp_letter_diff(batch): 
#         output = trainer._get_pp_letter_diff(pp_l=batch['pp_l'], orig_n_letters=batch['orig_n_letters'])
#         for k, v in output.items(): batch[k] = v.tolist() 
#         return batch
#     def add_contradiction_score(batch): 
#         batch['contradiction_scores'] = trainer._get_contradiction_scores(orig_l=batch['orig'], pp_l=batch['pp_l']).cpu().tolist()
#         return batch
#     ds_expanded = ds_expanded.map(add_vm_scores_eval,        batched=True)
#     ds_expanded = ds_expanded.map(add_pp_letter_diff,        batched=True)
#     ds_expanded = ds_expanded.map(add_contradiction_score,   batched=True)
#     def add_sts_scores_eval(row):  return trainer._get_sts_scores_one_to_many(row['pp_l'], row['orig_sts_embeddings'])[0]
#     df_sts['sts_scores'] = df_sts.apply(add_sts_scores_eval, axis=1)

#     ## Merge together results 
#     df_sts = df_sts.drop(columns = ['pp_l','orig_sts_embeddings'])
#     df_sts_expanded = df_sts.apply(pd.Series.explode).reset_index(drop=True)
#     ds_expanded = Dataset.from_pandas(ds_expanded.to_pandas().merge(df_sts_expanded, how='left', on='pp_idx').reset_index(drop=True))

#     ## Calculate rewards and identify adversarial examples 
#     def add_reward(batch): 
#         batch['reward'] = trainer._get_reward(vm_scores=batch['vm_scores'], sts_scores=batch['sts_scores'],
#                   pp_letter_diff=batch['pp_letter_diff'], contradiction_scores=batch['contradiction_scores']).cpu().tolist()
#         return batch
#     ds_expanded = ds_expanded.map(add_reward,   batched=True)
#     def add_is_valid_pp(example): 
#         example['is_valid_pp'] = trainer._is_valid_pp(sts_score=example['sts_scores'],
#              pp_letter_diff=example['pp_letter_diff'], contradiction_score=example['contradiction_scores'])*1
#         return example 
#     ds_expanded = ds_expanded.map(add_is_valid_pp,   batched=False)
#     def add_is_adv_example(batch): 
#         batch['is_adv_example'] = (np.array(batch['is_valid_pp']) * np.array(batch['label_flip'])).tolist()
#         return batch
#     ds_expanded = ds_expanded.map(add_is_adv_example,   batched=True)

#     ## Calculate summary statistics
#     df_expanded = ds_expanded.to_pandas()
#     eval_metric_cols = ['label_flip', 'is_valid_pp', 'is_adv_example', 'reward', 'vm_scores', 'sts_scores',  'pp_letter_diff', 'contradiction_scores']
#     agg_metrics = ['mean','std']  # not going to use the median 
#     # avg across each orig 
#     df_grp_stats = df_expanded[['idx'] + eval_metric_cols].groupby('idx').agg(agg_metrics)
#     df_grp_stats.columns = df_grp_stats.columns = ["-".join(a) for a in df_grp_stats.columns.to_flat_index()]
#     # avg across whole dataset 
#     df_overall_stats = df_expanded[eval_metric_cols].groupby(lambda _ : True).agg(agg_metrics).reset_index(drop=True)
#     df_overall_stats.columns = df_overall_stats.columns = ["-".join(a) + "-" + split for a in df_overall_stats.columns.to_flat_index()]
#     df_overall_metrics = df_overall_stats.iloc[0].to_dict()   ## WANDB this 
#     df_overall_metrics['any_adv_example_proportion' + "-" + split] = np.mean((df_grp_stats['is_adv_example-mean'] > 0 ) * 1)
#     # add epoch key
#     df_expanded['epoch'] = epoch
#     df_overall_metrics['epoch'] = epoch

#     ## Log results to wandb 
#     wandb_eval_d = dict()
#     mean_only = ['label_flip', 'is_valid_pp', 'is_adv_example']
#     mean_and_std = ['reward', 'vm_scores', 'sts_scores', 'pp_letter_diff', 'contradiction_scores']
#     for k in mean_only + mean_and_std: 
#         name = k + "-mean"
#         wandb_eval_d[name + "-"+ split + "-hist"] = Histogram(df_grp_stats[name].tolist())
#     for k in mean_and_std:
#         name = k + "-std"
#         wandb_eval_d[name + "-" + split + "-hist"] = Histogram(df_grp_stats[name].tolist())
#     wandb_eval_d = merge_dicts(df_overall_metrics, wandb_eval_d)

#     ## Save paraphrase-level dataframe 
#     eval_epoch_df_d[split].append(df_expanded)
    
# eval_final_dfs = dict()
# for k in ['train', 'valid', 'test']:   eval_final_dfs[k] =  pd.concat(eval_epoch_df_d[k]) if eval_epoch_df_d[k] != [] else []
    
    


# In[ ]:


df_d = get_training_dfs(cfg.path_run, postprocessed=False)
for k, df in df_d.items(): 
    df_d[k] = postprocess_df(df, filter_idx=None, num_proc=1)
    df_d[k].to_pickle(f"{cfg.path_run}{k}_postprocessed.pkl")    
create_and_log_wandb_postrun_plots(df_d)
trainer.run.finish()
#run.finish()

