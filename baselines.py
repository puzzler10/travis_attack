#!/usr/bin/env python
# coding: utf-8

# In[2]:


import transformers, nltk, pandas as pd, torch, string
from datasets import load_dataset, load_from_disk, DatasetDict, ClassLabel
from pprint import pprint
from datetime import datetime
import argparse
import functools


from textattack import Attack, AttackArgs,Attacker
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.loggers import CSVLogger # tracks a dataframe for us.
from textattack.attack_recipes import AttackRecipe
from textattack.search_methods import BeamSearch
from textattack.constraints import Constraint
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.transformations import WordSwapEmbedding, WordSwapMaskedLM
from textattack.goal_functions import UntargetedClassification
from textattack.metrics.attack_metrics.attack_success_rate import AttackSuccessRate
from textattack.metrics.attack_metrics.words_perturbed import WordsPerturbed
from textattack.metrics.attack_metrics.attack_queries import AttackQueries
from textattack.metrics.quality_metrics.perplexity import Perplexity
from textattack.metrics.quality_metrics.use import USEMetric
from sentence_transformers.util import pytorch_cos_sim

from travis_attack.utils import display_all, merge_dicts, append_df_to_csv, set_seed
from travis_attack.data import prep_dsd_rotten_tomatoes,prep_dsd_simple,prep_dsd_financial
from travis_attack.config import Config
from travis_attack.models import _prepare_vm_tokenizer_and_model, get_vm_probs, prepare_models, get_nli_probs
from travis_attack.baseline_attacks import AttackRecipes, setup_baselines_parser
from fastcore.basics import in_jupyter


import warnings
warnings.filterwarnings("ignore", message="FutureWarning: The frame.append method is deprecated") 

path_baselines = "./baselines/"

set_seed(1000)


# In[4]:


######### CONFIG (default values) #########
param_d = dict(
    ds_name = "financial",
    split='test',
    sts_threshold = 0.8,
    contradiction_threshold = 0.2,
    acceptability_threshold = 0.5,
    pp_letter_diff_threshold = 30
)
###########################################

if not in_jupyter():  # override with any script options
    parser = setup_baselines_parser()
    newargs = vars(parser.parse_args())
    for k,v in newargs.items(): 
        if v is not None: param_d[k] = v


# In[6]:


### Common attack components
attack_recipes = AttackRecipes(param_d)
attack_list = attack_recipes.get_attack_list()


# In[8]:


hf_dataset = HuggingFaceDataset(attack_recipes.ds.dsd_raw[param_d['split']], dataset_columns=(['text'], 'label'))
for attack_json in attack_list:
    print("Now doing attack recipe number", attack_json['attack_num'], "with code", attack_json['attack_code'])
    datetime_now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    param_d['datetime'] =  datetime_now
    param_d['attack_num'] = attack_json['attack_num']
    param_d['attack_code'] = attack_json['attack_code']
    filename = f"{path_baselines}{datetime_now}_{param_d['ds_name']}_{param_d['split']}_{attack_json['attack_code']}.csv"
    attack_args = AttackArgs(num_examples=-1, enable_advance_metrics=True,
                            log_to_csv=filename, csv_coloring_style='plain', disable_stdout=True)
    attacker = Attacker(attack_json['attack_recipe'], hf_dataset, attack_args)

    # print("Current config for attack:")
    # print(d)

    attack_results = attacker.attack_dataset()

    attack_result_metrics = {
        **AttackSuccessRate().calculate(attack_results), 
        **WordsPerturbed().calculate(attack_results),
        **AttackQueries().calculate(attack_results),
        **Perplexity().calculate(attack_results),
        **USEMetric().calculate(attack_results)
    }
    attack_result_metrics.pop('num_words_changed_until_success')
    d = merge_dicts(param_d, attack_result_metrics)
    summary_df = pd.Series(d).to_frame().T
    append_df_to_csv(summary_df, f"{path_baselines}results.csv")


# In[ ]:


def display_adv_example(df): 
    from IPython.core.display import display, HTML
    pd.options.display.max_colwidth = 480 # increase column width so we can actually read the examples
    #display(HTML(df[['original_text', 'perturbed_text']].to_html(escape=False)))
    display(df[['original_text', 'perturbed_text']])

# def add_vm_score_and_label_flip(df, dataset, cfg, vm_tokenizer, vm_model): 
#     truelabels = torch.tensor(dataset._dataset['label'], device =cfg.device)
#     orig_probs =  get_vm_probs(df['original_text'].tolist(), cfg, vm_tokenizer, vm_model, return_predclass=False)
#     pp_probs = get_vm_probs(df['perturbed_text'].tolist(), cfg, vm_tokenizer, vm_model, return_predclass=False)
#     orig_predclass = torch.argmax(orig_probs, axis=1)
#     pp_predclass = torch.argmax(pp_probs, axis=1)
#     orig_truelabel_probs = torch.gather(orig_probs, 1, truelabels[:,None]).squeeze()
#     pp_truelabel_probs   = torch.gather(pp_probs, 1,   truelabels[:,None]).squeeze()
#     pp_predclass_probs   = torch.gather(pp_probs, 1,   pp_predclass[ :,None]).squeeze()
    
#     df['truelabel'] = truelabels.cpu().tolist()
#     df['orig_predclass'] = orig_predclass.cpu().tolist()
#     df['pp_predclass'] = pp_predclass.cpu().tolist()
#     df['orig_truelabel_probs'] = orig_truelabel_probs.cpu().tolist()
#     df['pp_truelabel_probs'] = pp_truelabel_probs.cpu().tolist()
#     df['vm_scores'] = (orig_truelabel_probs - pp_truelabel_probs).cpu().tolist()
#     df['label_flip'] = ((pp_predclass != truelabels) * 1).cpu().tolist()
#     return df

# def add_sts_score(df, sts_model, cfg): 
#     orig_embeddings  = sts_model.encode(df['original_text'].tolist(),  convert_to_tensor=True, device=cfg.device)
#     pp_embeddings    = sts_model.encode(df['perturbed_text'].tolist(), convert_to_tensor=True, device=cfg.device)
#     df['sts_scores'] = pytorch_cos_sim(orig_embeddings, pp_embeddings).diagonal().cpu().tolist()
#     return df

# def add_contradiction_score(df, cfg, nli_tokenizer, nli_model): 
#     contradiction_scores = get_nli_probs(df['original_text'].tolist(), df['perturbed_text'].tolist(), cfg, nli_tokenizer, nli_model)
#     df['contradiction_scores'] =  contradiction_scores[:,cfg.contra_label].cpu().tolist()
#     return df 

# def get_df_mean_cols(df): 
#     cols = ['label_flip', 'vm_scores', 'sts_scores',
#             'contradiction_scores', 'sts_threshold_met', 'contradiction_threshold_met']
#     s = df[cols].mean()
#     s.index = [f"{o}_mean" for o in s.index]
#     return dict(s)

# def get_cts_summary_stats(df): 
#     cols = ['vm_scores', 'sts_scores', 'contradiction_scores']
#     df_summary = df[cols].describe(percentiles=[.1,.25,.5,.75,.9]).loc[['std','10%','25%','50%','75%','90%']]
#     tmp_d = dict()
#     for c in cols: 
#         s = df_summary[c]
#         s.index = [f"{c}_{o}" for o in s.index]
#         tmp_d = merge_dicts(tmp_d, dict(s))
#     return tmp_d


# In[ ]:


#filename1 = f"/data/tproth/travis_attack/baselines/2022-04-21_044443_rotten_tomatoes_valid_BeamSearchLMAttack_beam_sz=2_max_candidates=5.csv"
#filename = filename1
df = pd.read_csv(filename)
#display_adv_example(df)

#df = add_vm_score_and_label_flip(df, dataset, cfg, vm_tokenizer, vm_model)
#df = df.query("result_type != 'Skipped'")
#df = add_sts_score(df, sts_model, cfg)
#df = add_contradiction_score(df, cfg, nli_tokenizer, nli_model)

#df['sts_threshold_met'] = df['sts_scores'] > d['sts_threshold']
#df['contradiction_threshold_met'] = df['contradiction_scores'] < d['contradiction_threshold']
#df.to_csv(f"{filename[:-4]}_processed.csv", index=False)

#d = merge_dicts(d, get_df_mean_cols(df))
#d = merge_dicts(d, get_cts_summary_stats(df))


# In[ ]:


# df1 = df.sample(5)
# orig_l = df1['original_text'].tolist()
# pp_l = df1['perturbed_text'].tolist()
# print(orig_l)
# print(pp_l)


# In[ ]:


# for orig, adv in zip(df1['original_text'].tolist(), df1['perturbed_text'].tolist()): 
#     print(f"{orig}{adv}")
#     print()


# In[ ]:


#df.iloc[104][['original_text', 'perturbed_text']].values


# In[ ]:


#filename1 = f"/data/tproth/travis_attack/baselines/2022-04-20_133329_rotten_tomatoes_valid_BeamSearchCFEmbeddingAttack_beam_sz=1_max_candidates=1_processed.csv"
#df = pd.read_csv(filename1)
#display_all(df.sample(2))


# In[ ]:


#df_results = pd.read_csv(f"/data/tproth/travis_attack/baselines/results.csv")


# In[ ]:




