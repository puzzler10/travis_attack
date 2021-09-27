import GPUtil
from threading import Thread
import time 
import torchsnooper



############### Misc useful ##############################
def show_random_elements(dataset, num_examples=10):
    """Print some elements in a nice format so you can take a look at them. Use for a dataset from the `datasets` package.  """
    import datasets
    import random
    import pandas as pd
    from IPython.display import display, HTML

    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

    
def round_t(t, dp=2):
    """Return rounded tensors for easy viewing. t is a tensor, dp=decimal places"""
    if t.device.type == "cuda": t=t.cpu()
    return t.detach().numpy().round(dp)                       
    
    
################# Data cleaning and prep #############################
def create_train_valid_test(df, frac_train=0.7, frac_valid=0.15,
                            temporal=False, date_col=None, shuffle=False,
                            drop_date=False, drop_other_cols=[]):
    """ Generates train, validation and test sets for data. Handles data with temporal components. 
    The test set will have (1 - frac_train - frac_valid) as a fraction of df 
    
    df: a Pandas dataframe
    frac_train: fraction of data to use in training set 
    frac_valid: fraction of data to use in validation set 
    temporal: does the data have a temporal aspect? Boolean
    date_col: the temporal column of df to order by. Required if temporal=True
    shuffle: shuffle the data in the training test sets (only valid for temporal=False)
    drop_date: drop the date column in the results or not 
    drop_other_cols: list with column names to drop 
    """
    import numpy as np
    import pandas as pd 
    if temporal and shuffle:      print("Shuffle = True is ignored for temporal data")
    if temporal and not date_col: raise ValueError("Need to pass in a value for date_col if temporal=True")
    if not temporal and date_col: print("Parameter for date_col ignored if temporal=False")
    frac_test = 1 - frac_train - frac_valid
    if temporal:
        # Sort the dataframe by the date column 
        inds = np.argsort(df[date_col])
        df = df.iloc[inds].copy()
    else: 
        if shuffle: 
            inds = np.random.permutation(df.shape[0])
            if type(df) == pd.DataFrame:   df = df.iloc[inds].copy()
            else:                          df = df[inds].copy()
    if drop_date: df.drop(date_col, axis=1, inplace=True)
    if drop_other_cols: df.drop(drop_other_cols, axis=1, inplace=True)
    train = df[0:int(df.shape[0] * frac_train)]
    valid = df[(train.shape[0]):(int(train.shape[0] + (df.shape[0] * frac_valid)))]
    test  = df[(train.shape[0] + valid.shape[0]):]
    return (train.copy(), valid.copy(), test.copy())
    
    
    
################### GPU functions ######################    
def print_device_info(): 
    """
    Prints some statistics around versions and the GPU's available for
    the host machine
    """
    import torch
    import sys
    print("######## Diagnostics and version information ######## ")
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION', )
    from subprocess import call
    # call(["nvcc", "--version"]) does not work
    #! nvcc --version
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print ('Available devices ', torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name())
    print ('Current cuda device ', torch.cuda.current_device())
    print("#################################################################")
    
    
def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector. 
    Useful when running into an out of memory error on the GPU. """
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__, 
                                            " GPU" if obj.is_cuda else "",
                                            " pinned" if obj.is_pinned else "",
                                            pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
                                                    type(obj.data).__name__, 
                                                    " GPU" if obj.is_cuda else "",
                                                    " pinned" if obj.data.is_pinned else "",
                                                    " grad" if obj.requires_grad else "", 
                                                    " volatile" if obj.volatile else "",
                                                    pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass        
    print("Total size:", total_size)   
    
    
       
class Monitor(Thread):
    """Use this to check that you are using the GPU during your pytorch functions and to track memory usage 
    of the GPU's as well.""" 
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        


        
        
############## A no_grad version of model.generate() ##################

# from transformers.models.pegasus import PegasusForConditionalGeneration
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
import torch


def generate_with_grad(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    early_stopping: Optional[bool] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    length_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    encoder_no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    max_time: Optional[float] = None,
    decoder_start_token_id: Optional[int] = None,
    use_cache: Optional[bool] = None,
    num_beam_groups: Optional[int] = None,
    diversity_penalty: Optional[float] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    forced_bos_token_id: Optional[int] = None,
    forced_eos_token_id: Optional[int] = None,
    remove_invalid_values: Optional[bool] = None,
    **model_kwargs):
    # set init values
    num_beams = num_beams if num_beams is not None else self.config.num_beams
    num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
    max_length = max_length if max_length is not None else self.config.max_length
    do_sample = do_sample if do_sample is not None else self.config.do_sample
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    )

    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states

    if input_ids is None:
        # init `input_ids` with bos_token_id
        input_ids = self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))

    if model_kwargs.get("attention_mask", None) is None:
        # init `attention_mask` depending on `pad_token_id`
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            input_ids, pad_token_id, eos_token_id
        )

    # special case if pad_token_id is not defined
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        pad_token_id = eos_token_id

    # Storing encoder_input_ids for logits_processor that could use them
    encoder_input_ids = input_ids if self.config.is_encoder_decoder else None

    if self.config.is_encoder_decoder:
        # add encoder_outputs to model_kwargs
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

        # set input_ids as decoder_input_ids
        if "decoder_input_ids" in model_kwargs:
            input_ids = model_kwargs.pop("decoder_input_ids")
        else:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
            )

#         if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
#             raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")
    if input_ids.shape[-1] >= max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
            "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
        )

    # determine generation mode
    is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
    is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
    is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
    is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
    is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
    if num_beam_groups > num_beams:
        raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
    if is_group_beam_gen_mode and do_sample is True:
        raise ValueError(
            "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        )

    # set model_kwargs
    model_kwargs["use_cache"] = use_cache

    # get distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        encoder_input_ids=encoder_input_ids,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        forced_eos_token_id=forced_eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=remove_invalid_values,
    )

    stopping_criteria = self._get_stopping_criteria(
        max_length=max_length,
        max_time=max_time,
    )

    if is_greedy_gen_mode:
        if num_return_sequences > 1:
            raise ValueError(
                f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
            )

        # greedy search
        return self.greedy_search(
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            **model_kwargs,
        )

    elif is_sample_gen_mode:
        # get probability distribution warper
        logits_warper = self._get_logits_warper(
            top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
        )

        # expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids,
            expand_size=num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # sample
        return self.sample(
            input_ids,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            **model_kwargs,
        )

    elif is_beam_gen_mode:
        batch_size = input_ids.shape[0]

        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
        )
        # interleave with `num_beams`
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
        )
        #with torchsnooper.snoop(depth=4, max_variable_length=200, normalize=True):
        return self.beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            **model_kwargs,
        )

    elif is_beam_sample_gen_mode:
        logits_warper = self._get_logits_warper(
            top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
        )

        batch_size = input_ids.shape[0] * num_return_sequences

        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
        )

        # interleave with `num_beams * num_return_sequences`
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids,
            expand_size=num_beams * num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        return self.beam_sample(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            **model_kwargs,
        )

    elif is_group_beam_gen_mode:
        batch_size = input_ids.shape[0]

        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if num_beams % num_beam_groups != 0:
            raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

        diverse_beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
            num_beam_groups=num_beam_groups,
        )
        # interleave with `num_beams`
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
        )
        return self.group_beam_search(
            input_ids,
            diverse_beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            **model_kwargs,
        )
