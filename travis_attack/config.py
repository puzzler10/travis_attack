# AUTOGENERATED! DO NOT EDIT! File to edit: 03_config.ipynb (unless otherwise specified).

__all__ = ['Config']

# Cell
import torch

# Cell
class Config:
    def __init__(self):
        """Set up default parameters"""

        ### Models and datasets
        # options for the pp_model
        # 1. tuner007/pegasus_paraphrase
        # 2. tdopierre/ProtAugment-ParaphraseGenerator
        # 3. eugenesiow/bart-paraphrase
        self.pp_name = "eugenesiow/bart-paraphrase"
        self.vm_name = "textattack/distilbert-base-uncased-rotten-tomatoes"
        self.sts_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.dataset_name = "simple"

        ### Training hyperparameters
        self.seed = 420
        self.use_fp16 = True
        self.lr = 1e-5
        self.normalise_rewards = False
        self.pin_memory = True
        self.zero_grad_with_none = False
        self.pad_token_embeddings = True
        self.embedding_padding_multiple = 8
        self.orig_padding_multiple = 8   # pad input to multiple of this
        self.bucket_by_length = True
        self.shuffle_train = False
        self.remove_misclassified_examples = True
        self.unfreeze_last_n_layers = 2  # counting from the back. set to "all" to do no layer freezing.



        ### Paraphrase parameters
        self.pp = {
            "num_beams": 1,
            "num_return_sequences": 1,
            "num_beam_groups": 1,
            "diversity_penalty": 0.,   # must be a float
            "temperature": 1.5,
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
        self.sampling_strategy = "simple"  # doesn't do anything
        # This makes the reward function more visible
        # copy-paste this from reward function
        self.reward_strategy = "[-0.5 if sts < 0.5 else 0.5+v*sts for v,sts in zip(vm_scores, sts_scores)]"

        ### W&B parameters
        self.wandb = dict(
            project = "travis_attack",
            entity = "uts_nlp",
            mode = "online",  # set to "disabled" to turn off wandb, "online" to enable it
            log_grads = False,
            log_grads_freq = 1,  # no effect if wandb_log_grads is False
            plot_examples = False,
            n_examples_plot = 4,  # number of individual examples to plot curves for
            log_token_entropy=True,
            log_token_probabilities = True,
            run_notes = f"Reward: {self.reward_strategy}\nDataset: {self.dataset_name}"
        )

        ### Devices and GPU settings
        #### TODO: do you need this with accelerator? does this handle the post-processing analytics too?
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = accelerator.device
        self.devicenum = torch.cuda.current_device() if self.device.type == 'cuda' else -1
        # When not using Accelerator
        #n_wkrs = 4 * torch.cuda.device_count()
        # When using Accelerator
        self.n_wkrs = 0

        ## Globals
        self.splits = ['train', 'valid', 'test']
        self.metrics = ['loss', 'pp_logp', 'reward', 'vm_score', "sts_score", 'label_flip']
        self.path_data = "./data/"
        self.path_checkpoints = "../model_checkpoints/travis_attack/"
        self.path_run = None #keep as None; this is automatically filled out by Trainer class
        self.path_data_cache = "/data/tproth/.cache/huggingface/datasets/"


        # Adjust config depending on dataset.
        if self.dataset_name   == "simple":           self.adjust_config_for_simple_dataset()
        elif self.dataset_name == "rotten_tomatoes":  self.adjust_config_for_rotten_tomatoes_dataset()

        # Checks
        self._validate_n_epochs()


    def adjust_config_for_simple_dataset(self):
        """Adjust config for the simple dataset."""
        self.dataset_name = "simple"
        self.orig_cname = "text"
        self.label_cname = 'label'
        self.orig_max_length = 20
        self.pp['max_length'] = 20
        self.batch_size_train = 4
        self.batch_size_eval = 4
        self.acc_steps = 1  # gradient accumulation steps
        self.n_train_epochs = 6
        self.eval_freq = 2
        return self

    def adjust_config_for_rotten_tomatoes_dataset(self):
        """Adjust config for the rotten_tomatoes dataset."""
        self.dataset_name = "rotten_tomatoes"
        self.orig_cname = "text"
        self.label_cname = 'label'
        self.orig_max_length = 64
        self.pp['max_length'] = 64
        self.batch_size_train = 16
        self.batch_size_eval = 64
        self.acc_steps = 1  # gradient accumulation steps
        self.n_train_epochs = 8
        self.eval_freq = 1
        return self

    def small_ds(self):
        """Adjust the config to use a small dataset (for testing purposes).
        Not possible when using the simple dataset. """
        if self.dataset_name == "simple":
            raise Exception("Don't shard when using the simple dataset (no need)")
        self.use_small_ds = True  # for testing purposes
        self.n_shards = 40
        self.shard_contiguous = False
        return self

    def _validate_n_epochs(self):
        if self.n_train_epochs % self.eval_freq != 0:
            raise Exception("Set n_train_epochs to a multiple of eval_freq so there are no leftover epochs.")
