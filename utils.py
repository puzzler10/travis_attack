import GPUtil
from threading import Thread
import time 




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
        

