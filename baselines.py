#!/usr/bin/env python
# coding: utf-8

# In[2]:


import transformers, nltk, pandas as pd, torch
from datasets import load_dataset, load_from_disk, DatasetDict, ClassLabel
from pprint import pprint
from datetime import datetime
import argparse

from textattack import Attack, AttackArgs,Attacker
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.loggers import CSVLogger # tracks a dataframe for us.
from textattack.attack_recipes import AttackRecipe
from textattack.search_methods import BeamSearch
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.transformations import WordSwapEmbedding, WordSwapMaskedLM
from textattack.goal_functions import UntargetedClassification
from textattack.metrics.attack_metrics.attack_success_rate import AttackSuccessRate
from textattack.metrics.attack_metrics.words_perturbed import WordsPerturbed
from textattack.metrics.attack_metrics.attack_queries import AttackQueries
from textattack.metrics.quality_metrics.perplexity import Perplexity
from textattack.metrics.quality_metrics.use import USEMetric
from sentence_transformers.util import pytorch_cos_sim

from travis_attack.utils import display_all, merge_dicts, append_df_to_csv
from travis_attack.data import prep_dsd_rotten_tomatoes,prep_dsd_simple,prep_dsd_financial
from travis_attack.config import Config
from travis_attack.models import _prepare_vm_tokenizer_and_model, get_vm_probs, prepare_models, get_nli_probs
from fastcore.basics import in_jupyter

path_baselines = "./baselines/"
datetime_now = datetime.now().strftime("%Y-%m-%d_%H%M%S")


# In[4]:


class BeamSearchCFEmbeddingAttack(AttackRecipe):
    """Untarged classification + word embedding swap + [no repeat, no stopword] constraints + beam search"""
    @staticmethod
    def build(model_wrapper, beam_sz=2, max_candidates=5):
        goal_function = UntargetedClassification(model_wrapper)
        stopwords = nltk.corpus.stopwords.words("english") # The one used by default in textattack
        constraints = [RepeatModification(),
                       StopwordModification(stopwords)]
        transformation = WordSwapEmbedding(max_candidates=max_candidates)
        search_method = BeamSearch(beam_width=beam_sz)
        attack = Attack(goal_function, constraints, transformation, search_method)
        return attack

class BeamSearchLMAttack(AttackRecipe): 
    """"""
    @staticmethod
    def build(model_wrapper, beam_sz=2, max_candidates=5):
        stopwords = nltk.corpus.stopwords.words("english") # The one used by default in textattack
        goal_function = UntargetedClassification(model_wrapper)
        constraints = [RepeatModification(),
                       StopwordModification()]
        transformation = WordSwapMaskedLM(method='bae', masked_language_model='distilroberta-base', max_candidates=max_candidates)
        search_method = BeamSearch(beam_width=beam_sz)
        attack = Attack(goal_function, constraints, transformation, search_method)
        return attack

def setup_baselines_parser(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name")
    parser.add_argument("--split")
    parser.add_argument("--attack_recipe")
    parser.add_argument("--num_examples", type=int)
    parser.add_argument("--beam_sz", type=int)
    parser.add_argument("--max_candidates", type=int)
    parser.add_argument("--sts_threshold", type=float)
    parser.add_argument("--contradiction_threshold", type=float)
    #parser.add_argument('args', nargs=argparse.REMAINDER)  # activate to put keywords in kwargs.
    return parser


# In[5]:


######### CONFIG (default values) #########
d = dict(
    datetime=datetime_now,
    ds_name = "rotten_tomatoes",
    split = 'valid',
    attack_name = 'BeamSearchCFEmbeddingAttack',
    num_examples = -1,
    beam_sz = 1,
    max_candidates = 1,
    sts_threshold = 0.6,
    contradiction_threshold = 0.8
)
###########################################

if not in_jupyter():  # override with any script options
    parser = setup_baselines_parser()
    newargs = vars(parser.parse_args())
    for k,v in newargs.items(): 
        if v is not None: d[k] = v


# In[6]:


if   d['attack_name'] == 'BeamSearchLMAttack':          attack_recipe = BeamSearchLMAttack
elif d['attack_name'] == 'BeamSearchCFEmbeddingAttack': attack_recipe = BeamSearchCFEmbeddingAttack
filename = f"{path_baselines}{datetime_now}_{d['ds_name']}_{d['split']}_{d['attack_name']}_beam_sz={d['beam_sz']}_max_candidates={d['max_candidates']}.csv"


# In[7]:


if d['ds_name'] == "financial_phrasebank":
    cfg = Config().adjust_config_for_financial_dataset()
    dsd = prep_dsd_financial(cfg)
elif d['ds_name'] == "rotten_tomatoes":      
    cfg = Config().adjust_config_for_rotten_tomatoes_dataset()
    dsd = prep_dsd_rotten_tomatoes(cfg)
elif d['ds_name'] == "simple":      
    cfg = Config().adjust_config_for_simple_dataset()
    dsd = prep_dsd_simple(cfg)
    #dataset = ...
dataset = HuggingFaceDataset(dsd[d['split']])


# In[8]:


vm_tokenizer, vm_model, _,_, sts_model, nli_tokenizer, nli_model, cfg = prepare_models(cfg)
vm_model_wrapper = HuggingFaceModelWrapper(vm_model, vm_tokenizer)
attack = attack_recipe.build(vm_model_wrapper, d['beam_sz'], d['max_candidates'])
attack_args = AttackArgs(num_examples=d['num_examples'], enable_advance_metrics=True,
                        log_to_csv=filename, csv_coloring_style='plain', disable_stdout=True)
attacker = Attacker(attack, dataset, attack_args)


# In[ ]:


print("Current config for attack:")
print(d)


# In[9]:


attack_results = attacker.attack_dataset()


# In[10]:


attack_result_metrics = {
    **AttackSuccessRate().calculate(attack_results), 
    **WordsPerturbed().calculate(attack_results),
    **AttackQueries().calculate(attack_results),
    **Perplexity().calculate(attack_results),
    **USEMetric().calculate(attack_results)
}
attack_result_metrics.pop('num_words_changed_until_success')
d = merge_dicts(d, attack_result_metrics)


# In[11]:


def display_adv_example(df): 
    from IPython.core.display import display, HTML
    pd.options.display.max_colwidth = 480 # increase column width so we can actually read the examples
    #display(HTML(df[['original_text', 'perturbed_text']].to_html(escape=False)))
    display(df[['original_text', 'perturbed_text']])

def add_vm_score_and_label_flip(df, dataset, cfg, vm_tokenizer, vm_model): 
    truelabels = torch.tensor(dataset._dataset['label'], device =cfg.device)
    orig_probs =  get_vm_probs(df['original_text'].tolist(), cfg, vm_tokenizer, vm_model, return_predclass=False)
    pp_probs = get_vm_probs(df['perturbed_text'].tolist(), cfg, vm_tokenizer, vm_model, return_predclass=False)
    orig_predclass = torch.argmax(orig_probs, axis=1)
    pp_predclass = torch.argmax(pp_probs, axis=1)
    orig_truelabel_probs = torch.gather(orig_probs, 1, truelabels[:,None]).squeeze()
    pp_truelabel_probs   = torch.gather(pp_probs, 1,   truelabels[:,None]).squeeze()
    pp_predclass_probs   = torch.gather(pp_probs, 1,   pp_predclass[ :,None]).squeeze()
    
    df['truelabel'] = truelabels.cpu().tolist()
    df['orig_predclass'] = orig_predclass.cpu().tolist()
    df['pp_predclass'] = pp_predclass.cpu().tolist()
    df['orig_truelabel_probs'] = orig_truelabel_probs.cpu().tolist()
    df['pp_truelabel_probs'] = pp_truelabel_probs.cpu().tolist()
    df['vm_scores'] = (orig_truelabel_probs - pp_truelabel_probs).cpu().tolist()
    df['label_flip'] = ((pp_predclass != truelabels) * 1).cpu().tolist()
    return df

def add_sts_score(df, sts_model, cfg): 
    orig_embeddings  = sts_model.encode(df['original_text'].tolist(),  convert_to_tensor=True, device=cfg.device)
    pp_embeddings    = sts_model.encode(df['perturbed_text'].tolist(), convert_to_tensor=True, device=cfg.device)
    df['sts_scores'] = pytorch_cos_sim(orig_embeddings, pp_embeddings).diagonal().cpu().tolist()
    return df

def add_contradiction_score(df, cfg, nli_tokenizer, nli_model): 
    contradiction_scores = get_nli_probs(df['original_text'].tolist(), df['perturbed_text'].tolist(), cfg, nli_tokenizer, nli_model)
    df['contradiction_scores'] =  contradiction_scores[:,0].cpu().tolist()
    return df 

def get_df_mean_cols(df): 
    cols = ['label_flip', 'vm_scores', 'sts_scores',
            'contradiction_scores', 'sts_threshold_met', 'contradiction_threshold_met']
    s = df[cols].mean()
    s.index = [f"{o}_mean" for o in s.index]
    return dict(s)

def get_cts_summary_stats(df): 
    cols = ['vm_scores', 'sts_scores', 'contradiction_scores']
    df_summary = df[cols].describe(percentiles=[.1,.25,.5,.75,.9]).loc[['std','10%','25%','50%','75%','90%']]
    tmp_d = dict()
    for c in cols: 
        s = df_summary[c]
        s.index = [f"{c}_{o}" for o in s.index]
        tmp_d = merge_dicts(tmp_d, dict(s))
    return tmp_d


# In[14]:


#filename1 = f"/data/tproth/travis_attack/baselines/rotten_tomatoes_valid_BeamSearchLMAttack_beam_sz=5_max_candidates=25.csv"
#filename = filename1
df = pd.read_csv(filename)
#display_adv_example(df)
df = add_vm_score_and_label_flip(df,dataset, cfg, vm_tokenizer, vm_model)
df = df.query("result_type != 'Skipped'")
df = add_sts_score(df, sts_model, cfg)
df = add_contradiction_score(df, cfg, nli_tokenizer, nli_model)

df['sts_threshold_met'] = df['sts_scores'] > d['sts_threshold']
df['contradiction_threshold_met'] = df['contradiction_scores'] < d['contradiction_threshold']
df.to_csv(f"{filename[:-4]}_processed.csv", index=False)

d = merge_dicts(d, get_df_mean_cols(df))
d = merge_dicts(d, get_cts_summary_stats(df))

summary_df = pd.Series(d).to_frame().T
append_df_to_csv(summary_df, f"{path_baselines}results.csv")


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


#display_all(df.sample(2))


# In[ ]:




