# AUTOGENERATED! DO NOT EDIT! File to edit: 03_config.ipynb (unless otherwise specified).

__all__ = ['Config']

# Cell
import torch
import datetime
import warnings


# Cell
class Config:
    def __init__(self):
        """Set up default parameters"""

        ### Models and datasets
        # options for the pp_model
        # 1. tuner007/pegasus_paraphrase
        # 2. tdopierre/ProtAugment-ParaphraseGenerator
        # 3. eugenesiow/bart-paraphrase
        self.pp_name = "tuner007/pegasus_paraphrase"
        self.sts_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.nli_name = "microsoft/deberta-base-mnli"
        self.dataset_name = "rotten_tomatoes"
        self._select_vm_model()


        ### Training hyperparameters
        self.seed = 420
        self.use_fp16 = False
        self.lr = 2e-5
        self.kl_coef = 0.01
        self.pin_memory = True
        self.zero_grad_with_none = False
        self.pad_token_embeddings = False
        self.embedding_padding_multiple = 8
        self.orig_padding_multiple = 8   # pad input to multiple of this
        self.bucket_by_length = True
        self.shuffle_train = False
        self.remove_misclassified_examples = True
        self.unfreeze_last_n_layers = "all"  #counting from the back. set to "all" to do no layer freezing, else set to an int
        self.reward_fn = "reward_fn_contradiction_and_letter_diff"
        # This makes the reward function easier to see in wandb
        # copy-paste this from reward function
        self.reward_strategy = ""

        ### Paraphrase parameters
        self.pp = {
            "num_beams": 1,
            "num_return_sequences": 1,
            "num_beam_groups": 1,
            "diversity_penalty": 0.,   # must be a float
            "temperature": 1,
            "length_penalty" : 1,
            "min_length" : 5,
        }

        ### Used for testing
        self.use_small_ds = False
        self.n_shards = None
        self.shard_contiguous = None

        ### Logging parameters
        self.save_model_while_training = False
        self.save_model_freq = 10

        ### These parameters don't do anything yet
        self.sampling_strategy = "greedy"  # doesn't do anything

        ### W&B parameters
        self.wandb = dict(
            project = "travis_attack",
            entity = "uts_nlp",
            mode = "disabled",  # set to "disabled" to turn off wandb, "online" to enable it
            log_grads = False,
            log_grads_freq = 1,  # no effect if wandb_log_grads is False
            log_token_entropy = True,
            log_token_probabilities = True,
            run_notes = f"Reward: {self.reward_strategy}\nDataset: {self.dataset_name}"
        )

        ### Devices and GPU settings
        #### TODO: do you need this with accelerator? does this handle the post-processing analytics too?
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.devicenum = torch.cuda.current_device() if self.device.type == 'cuda' else -1
        # When not using Accelerator
        #n_wkrs = 4 * torch.cuda.device_count()
        # When using Accelerator
        self.n_wkrs = 0

        ## Globals
        self.splits = ['train', 'valid', 'test']
        self.metrics = [ 'loss', 'pp_logp', 'ref_logp', 'kl_div', 'reward_with_kl', 'reward', 'vm_score', "sts_score", 'label_flip', 'contradiction_score', 'pp_letter_diff']
        self.path_data = "./data/"
        self.path_checkpoints = "../model_checkpoints/travis_attack/"
        self.path_run = None  # keep as None; this is automatically filled out by Trainer class
        self.path_data_cache = "/data/tproth/.cache/huggingface/datasets/"
        self.path_logs = f"./logs/"
        self.path_logfile = self.path_logs + f"run_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt"

        # Adjust config depending on dataset.
        if self.dataset_name   == "simple":           self.adjust_config_for_simple_dataset()
        elif self.dataset_name == "rotten_tomatoes":  self.adjust_config_for_rotten_tomatoes_dataset()
        elif self.dataset_name == "financial":        self.adjust_config_for_financial_dataset()


        # Checks
        self._validate_n_epochs()

    def _select_vm_model(self):
        if   self.dataset_name in ["rotten_tomatoes", "simple"]:  self.vm_name = "textattack/distilbert-base-uncased-rotten-tomatoes"
        elif self.dataset_name == "financial":                    self.vm_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"


    def adjust_config_for_simple_dataset(self):
        """Adjust config for the simple dataset."""
        self.dataset_name = "simple"
        self.orig_cname = "text"
        self.label_cname = 'label'
        self.orig_max_length = 20
        self.pp['max_length'] = 20
        self.batch_size_train = 2
        self.batch_size_eval = 4
        self.acc_steps = 2
        self.n_train_epochs = 10
        self.eval_freq = 1
        self._select_vm_model()
        return self

    def adjust_config_for_rotten_tomatoes_dataset(self):
        """Adjust config for the rotten_tomatoes dataset."""
        self.dataset_name = "rotten_tomatoes"
        self.orig_cname = "text"
        self.label_cname = 'label'
        self.orig_max_length = 60  # longest for pegasus
        self.pp['max_length'] = 60
        self.batch_size_train = 8
        self.batch_size_eval = 64
        self.acc_steps = 2
        self.n_train_epochs = 10
        self.eval_freq = 1
        self._select_vm_model()
        return self

    def adjust_config_for_financial_dataset(self):
        """Adjust config for the financial dataset."""
        self.dataset_name = "financial"
        self.orig_cname = "sentence"
        self.label_cname = 'label'
        self.orig_max_length = 60
        self.pp['max_length'] = 60
        self.batch_size_train = 8
        self.batch_size_eval = 64
        self.acc_steps = 2
        self.n_train_epochs = 10
        self.eval_freq = 1
        self._select_vm_model()
        return self

    def small_ds(self):
        """Adjust the config to use a small dataset (for testing purposes).
        Not possible when using the simple dataset. """
        if self.dataset_name == "simple":
            raise Exception("Don't shard when using the simple dataset (no need)")
        self.use_small_ds = True  # for testing purposes
        self.n_shards = 200
        self.shard_contiguous = False
        return self

    def _validate_n_epochs(self):
        if self.n_train_epochs % self.eval_freq != 0:
            raise Exception("Set n_train_epochs to a multiple of eval_freq so there are no leftover epochs.")
