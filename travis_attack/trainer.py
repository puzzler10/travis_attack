# AUTOGENERATED! DO NOT EDIT! File to edit: 20_trainer.ipynb (unless otherwise specified).

__all__ = ['Trainer']

# Cell
import torch, wandb, gc, numpy as np, pandas as pd,os
from wandb.data_types import Histogram
from tqdm.auto import tqdm
from .utils import (timecode, show_gpu, merge_dicts, unpack_nested_lists_in_df,
                                 display_all, append_df_to_csv)
from .tests import check_no_nans_or_infs
from .models import save_pp_model, resume_pp_model, get_vm_probs, get_start_end_special_token_ids
from .charts import plot_grad_flow

# Cell
import torch, numpy as np, pandas as pd, gc,sys, logging, warnings
from torch.utils.data import DataLoader, RandomSampler
from datasets import load_dataset, load_metric, load_from_disk, DatasetDict
from transformers import (AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
                          AutoTokenizer, AdamW, SchedulerType, get_scheduler)
from torch.distributions import Categorical
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from collections import defaultdict
from accelerate import Accelerator, notebook_launcher
from cachetools import cached, LRUCache
from types import MethodType
from timeit import default_timer as timer
from tqdm.auto import tqdm
import itertools
import copy
import wandb
from undecorated import undecorated

# Cell
class Trainer:
    def __init__(self, cfg, vm_tokenizer, vm_model, pp_tokenizer, pp_model, sts_model, optimizer,
                ds, initial_eval=True, log_code=True):
        store_attr()
        self._cfg = self.cfg; del self.cfg;
        self.epoch,self.acc_num,self.global_step,self.eval_num = 0,0,0,0
        self._reset_batch_dicts()
        #resume_pp_model(f"{path_checkpoints}devout-durian-172_39")
        self._setup_data_stores()
        self._setup_gradient_accumulation_variables()
        self.start_end_token_d = get_start_end_special_token_ids(self.pp_tokenizer)

    def train(self):
        self._setup_wandb_run()
        ## we set num_processes=1 because we are running on 1 GPU only and we must specify the argument
        #%lprun -f _training_function -f  get_pp_logp -f training_step -f  reward_fn -f  loss_fn -f eval_dl  notebook_launcher(_training_function, args=(pp_model, vm_model, dld_tkn, dld_raw, optimizer), num_processes=1, use_fp16=use_fp16)
        notebook_launcher(self._training_function, args=(),
                           num_processes=1, use_fp16=self._cfg.use_fp16)

    def _reset_batch_dicts(self):
        # train_batch_d holds all info to write to csv, time_d has times, wandb_d has everything to log to wandb
        # there will be overlap between them.
        self.batch_d,self.batch_time_d,self.batch_wandb_d = dict(),dict(),dict()

    def _setup_wandb_run(self):
        """Init wandb run, set up paths, create dir for model artifacts if needed, """
        self.run = wandb.init(project=self._cfg.wandb['project'], entity=self._cfg.wandb['entity'],
                              config=vars(self._cfg), mode=self._cfg.wandb['mode'],
                              notes=self._cfg.wandb['run_notes'], save_code=True)
        if self._cfg.wandb['log_grads']:
            wandb.watch(self.pp_model, log='gradients', log_freq=self._cfg.wandb['log_grads_freq'])
        self._cfg.run_name,self._cfg.run_id = self.run.name, self.run.id
        self._cfg.path_run = f"{self._cfg.path_checkpoints}{self.run.name}/"
        if not os.path.exists(self._cfg.path_run): os.makedirs(self._cfg.path_run, exist_ok=True)
        if self.log_code: self.run.log_code(".")

    def _setup_data_stores(self):
        """Setup dict `self.data_d` to store observations. Setup column names for wandb tables. """
        # Raw observation data (lists of dicts, later becomes pandas df)
        self.data_d = dict()
        for split in self._cfg.splits + ['training_step']:   self.data_d[split] = []

    def _setup_gradient_accumulation_variables(self):
        """acc_global_l is a list of all batch sizes encountered during training.
            """
        self.acc_global_l = self._cfg.dl_batch_sizes['train'] * self._cfg.n_train_epochs
        assert len(self.acc_global_l) ==  self._cfg.n_train_steps
        # Check if there will be leftover batches
        self._cfg.acc_leftover_batches =  self._cfg.n_train_steps % self._cfg.acc_steps
        if self._cfg.acc_leftover_batches != 0:
            msg = f"Config set to do gradient accumulation every {self._cfg.acc_steps} batches, and there are \
            {self._cfg.n_train_steps} total training steps, so there will be {self._cfg.acc_leftover_batches} batches at \
            the end that will not be trained on."
            warnings.warn(msg)
        self._reset_acc_lists()

    def _reset_acc_lists(self):
        """call this at start and every time you call opt step"""
        # acc_current_l is a list of the batch sizes in the current accumulation batch.
        last_step = (self._cfg.n_train_steps - 1) - self._cfg.acc_leftover_batches
        if self.global_step == 0:   # at start of training
            self.acc_current_l = self.acc_global_l[self.global_step:self._cfg.acc_steps]
            assert len(self.acc_current_l) == self._cfg.acc_steps
        else:
            self.acc_current_l = self.acc_global_l[(self.global_step+1):(self.global_step+self._cfg.acc_steps+1)]
            if self.global_step == last_step:  assert len(self.acc_current_l) == self._cfg.acc_leftover_batches
            else:                              assert len(self.acc_current_l) == self._cfg.acc_steps
        self.acc_current_n_examples = sum(self.acc_current_l)

    def _eval_save_log_test_set(self):
        """Eval on test set, convert to df, save to file, and log to wandb summary"""
        self._eval_dl(split='test')
        self.data_d["test"] = self._convert_data_d_to_df("test")
        self._set_df_colorder("test")
        self.data_d["test"].to_csv(f"{self._cfg.path_run}test.csv", index=False)
        self._add_wandb_run_summary_statistics()

    def _training_function(self):
        self.accelerator = Accelerator()
        self._cfg.device = self.accelerator.device
        vm_model,pp_model,sts_model,optimizer,ds.dld_tkn['train'] = self.accelerator.prepare(
            self.vm_model,self.pp_model,self.sts_model,self.optimizer,self.ds.dld_tkn['train'])

        logger.debug(show_gpu(f'GPU memory usage after loading models:'))
        progress_bar = tqdm(range(self._cfg.n_train_steps))
        self.pp_model.zero_grad(set_to_none=self._cfg.zero_grad_with_none)

        # initial eval (at epoch 0)
        if self.initial_eval:
            logger.info("Launching initial eval run: train")
            self._eval_dl(split='train')
            logger.info("Launching initial eval run: valid")
            self._eval_dl(split='valid')
            self._compute_and_log_eval_metrics()

        for self.epoch in range(1, self._cfg.n_train_epochs+1):
            logger.info(f"Now on epoch {self.epoch} of {self._cfg.n_train_epochs}")
            if not self.pp_model.training: self.pp_model.train()
            with timecode() as time_train_one_epoch:
                for self.batch_num, (data, raw) in enumerate(zip(self.ds.dld_tkn['train'], self.ds.dld_raw['train'])):
                    self._reset_batch_dicts()
                    self._training_step(data, raw)
                    if self._batch_for_opt_step(): self._reset_acc_lists()
                    self.acc_num = (self.acc_num + 1) % self._cfg.acc_steps
                    self.global_step += 1
                    progress_bar.update(1)


            wandb.log({'time/train_one_epoch_time': time_train_one_epoch.t,
                       'time/train_one_epoch_thoroughput': len(self.ds.dsd_tkn['train']) / time_train_one_epoch.t,
                       'epoch': self.epoch}, commit=True)

            if self._cfg.wandb['log_grads'] and self.epoch % self._cfg.wandb_log_grads_freq == 0:
                plt = plot_grad_flow(self.pp_model.named_parameters())
                wandb.log({"gradient flow": wandb.Image(plt)})  # doesn't work as a non-image (i.e. plotly)
                del plt
            #gc.collect()
            #torch.cuda.empty_cache()

            if self._cfg.save_model_while_training and (self.epoch + 1) % self._cfg.save_model_freq == 0:  save_model(epoch)

            # Evaluation loop
            if self.epoch % self._cfg.eval_freq == 0:
                self.eval_num += 1
                with timecode() as time_eval_train:
                    self._eval_dl(split='train') # or train_eval?
                with timecode() as time_eval_valid:
                    self._eval_dl(split='valid')
                with timecode() as time_eval_compute_metrics:
                    self._compute_and_log_eval_metrics()
                with timecode() as time_eval_gc_collect:
                    gc.collect()
                with timecode() as time_eval_empty_cache:
                    torch.cuda.empty_cache()
                wandb.log({'time/eval_train_time': time_eval_train.t, 'time/eval_valid_time': time_eval_valid.t,
                           'time/eval_train_thoroughput': len(self.ds.dsd_tkn['train']) / time_eval_train.t,
                           'time/eval_valid_thoroughput': len(self.ds.dsd_tkn['valid']) / time_eval_valid.t,
                           'time/eval_gc_collect': time_eval_gc_collect.t,
                           'time/eval_empty_cache': time_eval_empty_cache.t,
                           'time/eval_compute_metrics': time_eval_compute_metrics.t,
                           'epoch': self.epoch}, commit=True)

        self._eval_save_log_test_set()
        self.run.finish()

    def _training_step(self, data, raw):
        """Forward pass, loss function, backwards pass, parameter update (with gradient accumulation optional),
        recording results, wandb logging.
        """
        if not self.pp_model.training: self.pp_model.train()
        if not self.vm_model.training: self.vm_model.train()
        with timecode() as self.batch_time_d['time_generate_pp']:
            pp_output, pp_l = self._pp_model_forward(data)

        logger.debug(show_gpu(f'TRAIN, epoch {self.epoch}, batch {self.batch_num}, GPU memory usage after forward pass: '))

        # autocast is used by accelerate to allow mixed-precision loss functions.
        # drop it if we deprecate fp16 support (because it isn't supported for models like PEGASUS)
        with self.accelerator.autocast():
            with timecode() as self.batch_time_d['time_loss_fn']:
                loss_batch = self._loss_fn(data, raw, pp_output, pp_l)

        with timecode() as self.batch_time_d['time_backwards']:
            self.accelerator.backward(loss_batch)

        logger.debug(show_gpu(f'TRAIN, epoch {self.epoch}, batch {self.batch_num}, GPU memory usage after backwards pass: '))
        with timecode() as self.batch_time_d['time_opt_step']:
            if self._batch_for_opt_step():
                self.optimizer.step()
                self.pp_model.zero_grad(set_to_none=self._cfg.zero_grad_with_none)

        self._prepare_train_batch_d(raw, data, pp_l)
        self.data_d['training_step'].append(self.batch_d)
        self._wandb_log_training_step()

    def _batch_for_opt_step(self): return self.acc_num == (self._cfg.acc_steps - 1)

    def _add_batch_vars_to_batch_d(self, raw, data, pp_l):
        # Add basics. (results are already added elsewhere)
        self.batch_d = merge_dicts(self.batch_d, { 'idx': raw['idx'],
            'epoch': self.epoch, 'batch_num': self.batch_num, 'global_step': self.global_step,
            'acc_num': self.acc_num, "acc_batch_n_examples": self.acc_current_n_examples,
            "orig_l": raw['text'],
            "orig_label": data['label'].cpu().tolist(),
            "orig_truelabel_probs": data['orig_truelabel_probs'].cpu().tolist(),
            'orig_length': self.orig_length, 'orig_batch_size': self.orig_batch_size,
            "pp_l": pp_l, 'pp_length': self.pp_length, 'pp_batch_size': self.pp_batch_size
        })

    def _prepare_train_batch_d(self, raw, data, pp_l):
        self._add_batch_vars_to_batch_d(raw, data, pp_l)
        # Add times (only for training, not eval)
        for k, v in self.batch_time_d.items(): self.batch_time_d[k] = v.t  # extract time from timecode object
        self.batch_d = merge_dicts(self.batch_d, self.batch_time_d)

    def _wandb_log_training_step(self):
        self.batch_wandb_d = merge_dicts(self.batch_wandb_d, {
            'vm_scores_hist':       Histogram(self.batch_d['vm_score']),
            'vm_scores_mean':       np.mean(  self.batch_d['vm_score']),
            'sts_scores_hist':      Histogram(self.batch_d['sts_score']),
            'sts_scores_mean':      np.mean(  self.batch_d['sts_score']),
            'rewards_hist':         Histogram(self.batch_d['reward']),
            'rewards_mean':         np.mean(  self.batch_d['reward']),
            'pp_logp_hist':         Histogram(self.batch_d['pp_logp']),
            'pp_logp_mean':         np.mean(  self.batch_d['pp_logp']),
            'loss_hist'   :         Histogram(self.batch_d['loss']),
            'acc_batch_sizes':      Histogram(self.acc_current_l)
        })
        self.batch_wandb_d = merge_dicts(self.batch_wandb_d, self.batch_d)
        not_for_wandb_keys = ['orig_l', 'orig_label','orig_truelabel_probs', 'pp_l', 'loss', 'pp_logp',
                              'reward', 'sts_score', 'vm_score',
                              'pp_predclass_probs', 'label_flip', 'pp_predclass', 'pp_truelabel_probs']
        for k in not_for_wandb_keys:  self.batch_wandb_d.pop(k, None)
        wandb.log(self.batch_wandb_d, commit=True)

    def _convert_data_d_to_df(self, data_d_key):
        df = pd.DataFrame(self.data_d[data_d_key])
        # check all lists have the same number of elements in their row
        # last batch will have different number of elements to the batch size
        nonscalar_cols = df.columns[[o == np.dtype('object') for o in df.head(1).dtypes]].tolist()
        df_lengths = df[nonscalar_cols].applymap(len)
        assert df_lengths.eq(df_lengths.iloc[:,0], axis=0).all(None)
        # expand lists and broadcast scalars
        scalar_cols = df.columns[[o != np.dtype('object') for o in df.head(1).dtypes]].tolist()
        df_expanded = unpack_nested_lists_in_df(df, scalar_cols)
        # check shape of new dataframe is correct
        if data_d_key == "training_step":
            if self.epoch == 0:
                df_shape = (self._cfg.ds_length["train"],                       df.shape[1])
            else:
                df_shape = (self._cfg.ds_length["train"] * self._cfg.eval_freq, df.shape[1])
        elif data_d_key in ["train", "valid", "test"]:
                 df_shape = (self._cfg.ds_length[data_d_key],                   df.shape[1])
        assert df_expanded.shape == df_shape
        return df_expanded

    def _pp_model_forward(self, data):
        pp_output, pp_l = self._get_paraphrases(data['input_ids'], data['attention_mask'])
        self._assert_start_and_end_tokens_are_correct(orig_ids=data['input_ids'], pp_ids=pp_output.sequences)
        # Keep the below line here because then both training and eval can access it
        self._update_batch_size_and_length_variables(orig_ids=data['input_ids'], pp_ids=pp_output.sequences)
        return pp_output, pp_l

    def _assert_start_and_end_tokens_are_correct(self, orig_ids, pp_ids):
        """Make sure input sequences (orig) and output sequences (pp) start and end with the
        right special tokens (depends on tokenizer)"""
        # Input
        if self.start_end_token_d['input_start_id'] is not None:
            assert torch.all(orig_ids[:,0] == self.start_end_token_d['input_start_id'])
        # can probs rewrite this to make it nicer but it's fine for now
        assert torch.all(torch.logical_or(orig_ids[:,-1] == self.start_end_token_d['input_end_id'][0],
                                          orig_ids[:,-1] == self.start_end_token_d['input_end_id'][1]))

        # Output
        assert torch.all(pp_ids[:,0] == self.start_end_token_d['output_start_id'])
        assert torch.all(torch.logical_or(pp_ids[:,-1] == self.start_end_token_d['output_end_id'][0],
                                          pp_ids[:,-1] == self.start_end_token_d['output_end_id'][1]))

    def _update_batch_size_and_length_variables(self, orig_ids, pp_ids):
        # Update variables
        # for greedy search self.pp_length is equal to self.orig_batch_size but this won't be for beam search
        self.orig_batch_size     = orig_ids.shape[0]
        self.orig_length         = orig_ids.shape[1]
        self.pp_batch_size       = pp_ids.shape[0]
        self.pp_length           = pp_ids.shape[1]

    def _get_paraphrases(self, orig_ids, attention_mask):
        """Wrapper for generating paraphrases (pp's).  Only greedy search supported at the moment"""
        pp_output = self.pp_model.generate_with_grad(input_ids=orig_ids,
                                                attention_mask=attention_mask,
                                                 **self._cfg.pp,
                                                 do_sample=False,
                                                 return_dict_in_generate=True,
                                                 output_scores=True,
                                                 remove_invalid_values=False,
                                                 pad_token_id = self.pp_tokenizer.pad_token_id,
                                                 eos_token_id = self.pp_tokenizer.eos_token_id)
        pp_l = self.pp_tokenizer.batch_decode(pp_output.sequences, skip_special_tokens=True)
        return pp_output, pp_l

    def _loss_fn(self, data, raw, pp_output, pp_l):
        with timecode() as self.batch_time_d['time_reward_fn']:
            reward = self._reward_fn(data, raw, pp_l)

        with timecode() as self.batch_time_d['time_pp_logp']:
            pp_logp = self._get_pp_logp(pp_output)

        with timecode() as self.batch_time_d['time_loss_fn_loss_calc']:
            loss       = -reward * pp_logp
            loss_sum   = torch.sum(loss)  # we scale it later
            loss_batch = loss_sum / self.acc_current_n_examples  # for gradient accumulation

        self.batch_d['pp_logp']    =        pp_logp.detach().cpu().tolist()
        self.batch_d['loss']       =           loss.detach().cpu().tolist()
        self.batch_d['loss_sum']   =       loss_sum.detach().cpu().tolist()
        self.batch_d['loss_batch']   =   loss_batch.detach().cpu().tolist()
        return loss_batch

    def _reward_fn(self, data, raw, pp_l):
        """"""
        # Victim model probability differences between orig and pp
        with timecode() as self.batch_time_d['time_vm_scores']:
            pp_probs = get_vm_probs(pp_l, self._cfg, self.vm_tokenizer, self.vm_model, return_predclass=False)
            pp_predclass = torch.argmax(pp_probs, axis=1)
            pp_truelabel_probs   = torch.gather(pp_probs, 1, data['label'][:,None]).squeeze()
            pp_predclass_probs   = torch.gather(pp_probs, 1, pp_predclass[ :,None]).squeeze()
            label_flip = ((pp_predclass != data['label']) * 1)
            vm_scores = (data['orig_truelabel_probs'] - pp_truelabel_probs)

        # STS scores
        with timecode() as self.batch_time_d['time_sts_scores']:
            pp_embeddings  = self.sts_model.encode(pp_l, batch_size=len(raw), convert_to_tensor=True, device=self._cfg.device)
            # This returns a cosine similarity matrix, of which we just want the diagonal
            sts_scores = pytorch_cos_sim(data['orig_sts_embeddings'], pp_embeddings).diagonal()

        # Reward calculation
        rewards = torch.tensor([0 if sts < 0.5 else 0.5+v*sts for v,sts in zip(vm_scores, sts_scores)],device=self._cfg.device)

        if self._cfg.normalise_rewards:
            self.batch_d['reward_unscaled'] = rewards.detach().cpu().tolist()
            rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)

        self.batch_d['pp_truelabel_probs']  = pp_truelabel_probs.detach().cpu().tolist()
        self.batch_d['pp_predclass']        = pp_predclass.detach().cpu().tolist()
        self.batch_d['pp_predclass_probs']  = pp_predclass_probs.detach().cpu().tolist()
        self.batch_d['label_flip']          = label_flip.detach().cpu().tolist()
        self.batch_d['label_flip_fraction'] = np.mean(self.batch_d['label_flip'])
        self.batch_d['reward']              = rewards.detach().cpu().tolist()
        self.batch_d['vm_score']            = vm_scores.detach().cpu().tolist()
        self.batch_d['sts_score']           = sts_scores.detach().cpu().tolist()

        return rewards

    def _get_pp_logp(self, pp_output):
        """log(p(pp|orig)) basically.
        works for greedy search, will need tweaking for other types probably"""
        ### We want to align tokens with token probabilities. The first token is given at the start
        # and has no probability attached to it, so we remove it.
        seq_without_first_tkn = pp_output.sequences[:, 1:]
        assert seq_without_first_tkn.shape == torch.Size([self.orig_batch_size, self.pp_length - 1])

        ### Convert from tuple of scores to one big tensor of scores
        scores_stacked = torch.stack(pp_output.scores, 1)
        ### TESTS
        # We check shape and that there is no +inf or nan in scores.
        # Scores can have -inf in them - see explanation in `exploring_generation`.
        assert scores_stacked.shape == torch.Size([self.orig_batch_size, (self.pp_length - 1), self._cfg.vocab_size])
        assert torch.all(~torch.isnan(scores_stacked))
        assert torch.all(~torch.isposinf(scores_stacked))
        # Rough check that all idx before min_length are -inf for all elements in batch
        # We do min_length - 1 because sequences are allowed to have length min_length so that idx
        # shouldn't be set to -inf
        # Not a 100% test but very likely to identify
        idx_neginf = torch.nonzero(torch.isneginf(scores_stacked))
        assert len(idx_neginf[idx_neginf[:,2] == self.pp_tokenizer.eos_token_id, :]) == \
                  (self._cfg.pp["min_length"] -1) * self.orig_batch_size
        del idx_neginf

        ### Take log softmax of scores and then extract those that correspond
        # to the generated sequences
        scores_log_softmax = scores_stacked.log_softmax(2)
        seq_token_log_probs = torch.gather(scores_log_softmax,2,seq_without_first_tkn[:,:,None]).squeeze(-1)
        ### TESTS
        # -inf is possible in scores_log_softmax and seq_token_log_probs before the attention mask is added.
        assert torch.all(~torch.isnan(   scores_log_softmax))
        assert torch.all(~torch.isposinf(scores_log_softmax))
        self._check_scores_log_softmax_sums(scores_log_softmax)
        # probs should be 1-1 with the filtered tkns: check shape to confirm
        assert seq_token_log_probs.shape == seq_without_first_tkn.shape
        # Check that the last token probability corresponds to a possible end token
        # this has to be tested before the attention mask is multiplied with it because if the
        # padding token is 0 then this will be 0 too (and not the same as scores_log_softmax)
        output_end_ids = self.start_end_token_d['output_end_id']
        assert all([o in scores_log_softmax[:, -1, output_end_ids] for o in seq_token_log_probs[:,-1]])
        del output_end_ids
        ## THIS ONE IS LONG - a test rather than assert
        # check_seq_token_log_prob_values_are_correct(seq_without_first_tkn, scores_log_softmax,
        #                                             seq_token_log_probs)

        ### Generate attention mask to identify padding tokens. Then apply it to the
        # sequence probabilities so that we don't consider probability of padding tokens
        # when getting sequence probabilities.
        # Also replace the -inf values in seq_token_log_probs with a large negative number because if we
        # leave them in we end up with nan's introduced after multiplying with attention_mask,
        # since  -inf * 0 = nan
        attention_mask = self.pp_model._prepare_attention_mask_for_generation(
            seq_without_first_tkn, self.pp_tokenizer.pad_token_id, self.pp_tokenizer.eos_token_id
        )
        seq_token_log_probs = torch.nan_to_num(seq_token_log_probs, nan=None, posinf=None, neginf=-20)
        seq_token_log_probs = seq_token_log_probs * attention_mask
        ### TESTS
        assert seq_token_log_probs.shape == attention_mask.shape == seq_token_log_probs.shape
        # check attention mask only has 0 for padding tokens and not eos tokens or anything else
        assert all(seq_without_first_tkn[attention_mask == 0] == self.pp_tokenizer.pad_token_id)
        check_no_nans_or_infs(seq_token_log_probs)
        # check that we aren't picking extrememly rare tokens
        assert torch.all(seq_token_log_probs  > -10)

        ### Get sequence probabilities by summing up token log probabilities
        seq_log_prob = seq_token_log_probs.sum(-1)
        ## TESTS
        assert seq_log_prob.shape == torch.Size([self.pp_batch_size])
        check_no_nans_or_infs(seq_log_prob)

        if self.pp_model.training:  # don't bother logging or calculate entropy, token_probs in eval mode
            if self._cfg.wandb['log_token_entropy']:
                with timecode() as self.batch_time_d['time_log_entropy']:
                    self.batch_wandb_d['ent_hist'] = self._get_entropy_hist(scores_stacked, attention_mask)
            if self._cfg.wandb['log_token_probabilities']:
                with timecode() as self.batch_time_d['time_log_token_probabilities']:
                    self.batch_wandb_d = merge_dicts(self.batch_wandb_d,
                        self._get_token_probability_metrics(scores_log_softmax, attention_mask, k=3))
        return seq_log_prob

    def _check_scores_log_softmax_sums(self, scores_log_softmax):
        sums = scores_log_softmax.exp().sum(2)
        # check that the axes is right
        # we want to sum over token probabilities at each generation step, so we
        # should end up with a shape [self.orig_batch_size, self.pp_length]
        assert sums.shape[0] == self.orig_batch_size
        assert sums.shape[1] == self.pp_length - 1
        # check that they sum to 1 along the self.pp_length axis
        assert torch.allclose(sums, torch.ones(sums.size(), device=self._cfg.device), atol = 1e-4)

    def _check_seq_token_log_prob_values_are_correct(self, seq_without_first_tkn, scores_log_softmax, seq_token_log_probs):
        """Just enumerates and checks values
        Quite slow for large batches so run as a test rather than an assert in every batch.
        """
        l = []
        for i_ex in range(self.orig_batch_size):
            for i_step in range(self.pp_length - 1):
                i_tkn = seq_without_first_tkn[i_ex][i_step].item()
                l.append(scores_log_softmax[i_ex,i_step, i_tkn] == seq_token_log_probs[i_ex,i_step])
        assert all(l)

    def _get_entropy_hist(self, scores_stacked, attention_mask):
        ent = Categorical(logits = scores_stacked).entropy().detach()
        assert ent.shape == attention_mask.shape == torch.Size([self.pp_batch_size, self.pp_length - 1])
        ent = ent * attention_mask  # stop values after eos token from contributing to ent score
        # first remove structure (otherwise we have ragged arrays), then remove corresponding attention mask values
        # we can't just filter by ent[ent != 0] because we might have zero tokens during the sequence
        att_flat= attention_mask.flatten()
        indices = torch.nonzero(att_flat)
        ent_flat = ent.flatten()[indices].flatten()
        assert ent_flat.shape[0] == (torch.sum(att_flat)*1).item()
        # check everything we filter out is zero
        torch.isclose(ent.flatten()[torch.nonzero(~(att_flat > 0))].sum(), torch.tensor(0.), 1e-3)
        return Histogram(ent_flat.detach().cpu().tolist())

    def _get_token_probability_metrics(self, scores_log_softmax, attention_mask, k=3):
        token_prob_d = dict()
        tkn_kmaxprob, _ = torch.topk(scores_log_softmax, largest=True, k=k, dim=2)
        tkn_kmaxprob = tkn_kmaxprob.detach()
        tkn_kmaxprob = torch.nan_to_num(tkn_kmaxprob, nan=None, posinf=None, neginf=-20)
        assert tkn_kmaxprob.shape == torch.Size([self.pp_batch_size, self.pp_length - 1, k])

        # % of first prob over 0.9, 0.75, 0.5, 0.3, 0.1
        top_probs = tkn_kmaxprob[:,:,0].exp()
        top_probs = (top_probs * attention_mask).flatten()
        top_probs = top_probs[top_probs != 0]
        prob_threshold_l = [0.99, 0.975, 0.95, 0.90, 0.75, 0.5, 0.3, 0.1]
        for p in prob_threshold_l:
            token_prob_d[f"top_token_prob_over_{str(p)}"] = (torch.sum(top_probs > p) / top_probs.shape[0]).item()

        # avg + median + lower + upper quartile of first, second, third choice probs
        tkn_kmaxprob_mask = tkn_kmaxprob * attention_mask[:,:,None]  # broadcasting over kth dim
        for i in range(k):
            probs = tkn_kmaxprob_mask[:,:, i].flatten()
            probs = probs[probs != 0]
            token_prob_d[f"rank_{i+1}_histogram"] = Histogram(probs.detach().cpu().tolist())
            token_prob_d[f"rank_{i+1}_token_prob_mean"] = probs.mean().item()
            token_prob_d[f"rank_{i+1}_token_prob_median"] = probs.median().item()
            token_prob_d[f"rank_{i+1}_token_prob_0.25_quantile"] = probs.quantile(0.25).item()
            token_prob_d[f"rank_{i+1}_token_prob_0.75_quantile"] = probs.quantile(0.75).item()

        # tokens over probs above 0.1, 0.01, 0.001, 0.0001, 1/vocab_size prob
        allprobs = (scores_log_softmax.detach().exp() * attention_mask[:,:,None]).flatten()
        allprobs = allprobs[allprobs != 0]
        for p in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
            token_prob_d[f"%_of_tokens_above_prob_{p}"] =  (torch.sum(allprobs > p) / allprobs.shape[0]).item()
        token_prob_d[f"%_of_tokens_above_prob_1/vocab_size"] = \
            (torch.sum(allprobs > (1/self._cfg.vocab_size)) / allprobs.shape[0]).item()
        return token_prob_d

    def _eval_dl(self, split):
        """Get evaluation metrics for a dataloader"""
        # Put models in eval mode and do the forward pass
        # Current logic: push all batches together into one big list.
        self._reset_batch_dicts()
        if self.pp_model.training: self.pp_model.eval()
        if self.vm_model.training: self.vm_model.eval()
        # The "train_eval" dataloader is the same as train but a bigger batch size and explicitly no shuffling
        dl_key = "train_eval" if split == "train" else split
        dl_raw = self.ds.dld_raw[dl_key]
        dl_tkn = self.ds.dld_tkn[dl_key]
        with torch.no_grad():
            for self.batch_num, (data, raw) in enumerate(zip(dl_tkn, dl_raw)):
                logger.debug(f"EVAL: {split} with dl_key {dl_key}")
                logger.debug(f"Elements in data_d[{split}]: {len(self.data_d[split])}")
                logger.debug(show_gpu(f'EVAL, epoch {self.epoch}, batch {self.batch_num}, GPU memory usage after loading data: '))
                assert data['input_ids'].shape[0] == len(raw['text'])
                self._reset_batch_dicts()
                assert len(self.batch_d) == len(self.batch_time_d) == len(self.batch_wandb_d) == 0
                for k, v in data.items():
                    # Eval data isn't loaded on GPU by default unlike train data. This is because train dataloader goes
                    # through accelerator `prepare` function, but eval dataloaders don't. So here we load the data onto GPU
                    if data[k].device != self._cfg.device: data[k] = data[k].to(self._cfg.device)
                pp_output, pp_l = self._pp_model_forward(data)
                _ = self._loss_fn(data, raw, pp_output, pp_l)
                self._add_batch_vars_to_batch_d(raw, data, pp_l)
                self.data_d[split].append(self.batch_d)
                logger.debug(show_gpu(f'EVAL, epoch {self.epoch}, batch {self.batch_num}, GPU memory usage after loss_fn pass: '))

    def _compute_and_log_eval_metrics(self):
        """Calculate eval metrics for each split and log to wandb, then empty data_d"""
        wandb_d = dict(epoch=self.epoch)
        eval_splits = ["training_step", 'train', "valid"] if self.epoch != 0 else ['train', 'valid']
        for split in eval_splits:
            # data d -> data frame
            self.data_d[split] = self._convert_data_d_to_df(split)
            self._set_df_colorder(split)
            # calc metrics
            df = self.data_d[split][['epoch'] + self._cfg.metrics]
            if split == "training_step": df = df.query("epoch == @self.epoch")
            d = df.mean()[self._cfg.metrics].to_dict()
            wandb_d = merge_dicts(wandb_d, {f"{k}_{split}": v for k, v in d.items()})
            # df append to file + empty data_d
            append_df_to_csv(self.data_d[split], path = f"{self._cfg.path_run}{split}.csv")
            self.data_d[split] = []
        wandb.log(wandb_d, commit=True)

    def _set_df_colorder(self, data_d_key):
        colorder_eval=['idx','epoch', 'orig_l',  'pp_l','orig_truelabel_probs','pp_truelabel_probs',
        'pp_predclass_probs','orig_label','pp_predclass','label_flip', 'vm_score','sts_score',
        'reward', 'pp_logp','loss','batch_num','global_step','acc_num','loss_sum', 'loss_batch', 'label_flip_fraction',
        'orig_length','orig_batch_size','pp_length','pp_batch_size']
        if data_d_key == "training_step":
            colorder_training_step = colorder_eval + [o for o in self.data_d['training_step'].columns if 'time_' in o]
            assert len(set(colorder_training_step).difference(set(self.data_d[data_d_key].columns))) == 0
            self.data_d[data_d_key] = self.data_d[data_d_key][colorder_training_step]
        else:
            assert len(set(colorder_eval).difference(set(self.data_d[data_d_key].columns))) == 0
            self.data_d[data_d_key] = self.data_d[data_d_key][colorder_eval]

    def _add_wandb_run_summary_statistics(self):
        """Compute test metrics for the run and log them to the wandb run summary pane. """
        ## Summary statistics of the test set
        # From the last epoch atm because we don't have early stopping
        test_metrics = self.data_d['test'].filter(self._cfg.metrics, axis=1).mean()
        for metric, val in zip(test_metrics.index, test_metrics):
            self.run.summary[f"{metric}_avg_test"] = val

    def get_training_dfs(self):
        """Return a dict of dataframes with all training and eval data"""
        df_d = dict()
        for key in self._cfg.splits + ['training_step']:
            df_d[key] = pd.read_csv(f"{self._cfg.path_run}{key}.csv")
        return df_d