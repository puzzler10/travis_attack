import GPUtil
from threading import Thread
import time 
import torchsnooper
from timeit import default_timer as timer

######### for timing code #########
class timecode:
    def __enter__(self):
        self.t0 = timer()
        return self

    def __exit__(self, type, value, traceback):
        self.t = timer() - self.t0

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
    
    
################### Diagnostic functions ######################    

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    from matplotlib.lines import Line2D
    import numpy as np
    import matplotlib.pyplot as plt 
    plt.figure()
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return plt


def print_info_on_generated_text():
    """
        Prints a bunch of statistics around the generated text. Useful for debugging purposes.
        So far only works for greedy search.
    """
    logger.info("\n######################################################################\n")
    logger.info(f"Original text: {text}")
    tgt_text = pp_tokenizer.batch_decode(translated.sequences, skip_special_tokens=True)
    tgt_text_with_tokens = pp_tokenizer.batch_decode(translated.sequences, skip_special_tokens=False)
    logger.info(f"Generated text: {tgt_text}")
    logger.info(f"Generated text with special tokens: {tgt_text_with_tokens}")
    logger.info(f"Shape of translated.sequences:{translated.sequences.shape}")
    logger.info(f"translated.sequences:{translated.sequences}")
    logger.info(f"Scores is a tuple of length {len(translated.scores)} \
    and each score is a tensor of shape {translated.scores[0].shape}")
    scores_stacked = torch.stack(translated.scores, 1)
    logger.info(f"Stacking the scores into a tensor of shape {scores_stacked.shape}")
    scores_softmax = torch.softmax(scores_stacked, 2)
    logger.info(f"Now taking softmax. This shouldn't change the shape, but just to check,\
    its shape is {scores_softmax.shape}")
    probsums = scores_softmax.sum(axis=2)
    logger.info(f"These are probabilities now and so they should all sum to 1 (or close to it) in the axis \
    corresponding to each time step. We can check the sums here: {probsums}, but it's a long tensor \
    of shape {probsums.shape} and hard to see, so summing over all these values and removing 1 \
    from each gives {torch.sum(probsums - 1)} \
    which should be close to 0.")
    seq_without_first_tkn = translated.sequences[:, 1:]
    logger.info("Now calculating sequence probabilities")
    seq_token_probs = torch.gather(scores_softmax,2,seq_without_first_tkn[:,:,None]).squeeze(-1)
    seq_prob = seq_token_probs.prod(-1).item()
    logger.info(f"Sequence probability: {seq_prob}")

    # Get the 2nd and 3rd most likely tokens at each st
    topk_ids = torch.topk(scores_softmax,3,dim=2).indices[:,:,1:]
    topk_tokens_probs = torch.gather(scores_softmax,2,topk_ids).squeeze(-1)
    toks2 = pp_tokenizer.convert_ids_to_tokens(topk_ids[:,:,0].squeeze())
    toks3 = pp_tokenizer.convert_ids_to_tokens(topk_ids[:,:,1].squeeze())
    tok_probs2 = topk_tokens_probs[:,:,0].squeeze()
    tok_probs3 = topk_tokens_probs[:,:,1].squeeze()

    logger.info(f"Probabilities of getting the top 3 tokens at each step:")
    tokens = pp_tokenizer.convert_ids_to_tokens(seq_without_first_tkn.squeeze())
    for (p, t, p2,t2,p3,t3)  in zip(seq_token_probs.squeeze(), tokens, tok_probs2, toks2, tok_probs3, toks3): 
        logger.info(f"{t}: {round(p.item(),3)}  {t2}: {round(p2.item(),3)}  {t3}: {round(p3.item(),3)}")    
    
    
    
def check_parameters_update(dl): 
    """
    This checks which parameters are being updated. 
    We run one forward pass+backward pass (updating the parameters once) 
    and look at which ones change. 
    """
    # Check which parameters should be updated
    params_with_grad = [o for o in pp_model.named_parameters() if o[1].requires_grad]
    print("---- Parameters with 'requires_grad' and their sizes ------")
    for (name, p) in params_with_grad:  print(name, p.size())
        
    ## Take a step and see which weights update
    params_all = [o for o in pp_model.named_parameters()]  # this is updated by a training step    
    params_all_initial = [(name, p.clone()) for (name, p) in params_all]  # Initial values
        
    # take a step    
    loss, reward, pp_logp = training_step(data)
    
    print("\n---- Matrix norm of parameter update for one step ------\n")
    for (_,old_p), (name, new_p) in zip(params_all_initial, params_all): 
        print (name, torch.norm(new_p - old_p).item()) 
    
    
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
        

def show_gpu(msg):
    """
    ref: https://github.com/huggingface/transformers/issues/1742#issue-518262673
    put in logger.info()
    """
    import subprocess
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'], 
            encoding='utf-8'))
    def to_int(result):
        return int(result.strip().split('\n')[0])
    
    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    return f"{msg} {100*pct:2.1f}% ({used} out of {total})" 


