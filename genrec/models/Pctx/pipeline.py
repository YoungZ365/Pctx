from logging import getLogger
from typing import Union
import torch
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
import shutil
import time
from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import get_config, init_seed, init_logger, init_device, \
    get_dataset, get_tokenizer, get_model, get_trainer, log,get_cluster
import numpy as np
import random
import datetime
from genrec.utils import list_to_str
import json
import pickle





class PctxPipeline:
    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        tokenizer: AbstractTokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
            accelerator=None ,
            Round=None
    ):


        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )
        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device()



        # Accelerator
        if accelerator is not None:
            self.accelerator = accelerator
        else:
            # Accelerator
            self.project_dir = os.path.join(
                self.config['tensorboard_log_dir'],
                self.config["dataset"],
                self.config["model"]
            )
            self.accelerator = Accelerator(log_with='tensorboard', project_dir=self.project_dir)


        self.config['accelerator'] = self.accelerator
        self.Round=Round


        # Seed and Logger
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(f'Device: {self.config["device"]}')



        self.run_mode=self.config['run_mode']



        if self.run_mode != 'train' and self.run_mode != 'test':
            raise ValueError('mode is out of range')


        # Dataset
        self.raw_dataset = get_dataset(dataset_name)(self.config)
        self.log(self.raw_dataset)
        self.split_datasets = self.raw_dataset.split()

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer(self.config, self.raw_dataset)
        else:
            assert isinstance(model_name, str), 'Tokenizer must be provided if model_name is not a string.'
            self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)
        self.log('finish getting tokenizer')



        if self.run_mode=='train':
            self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)
            self.log('finish self.tokenizer.tokenize(self.split_datasets)r')
        else:
            self.tokenized_datasets = self.tokenizer.tokenize_only_test(self.split_datasets)
            self.log('[Test mode] finish self.tokenizer.tokenize(self.split_datasets)r')



        if self.run_mode=='train':

          
            sem_ids_path = os.path.join(
                self.raw_dataset.cache_dir, "processed",
                f'{os.path.basename(self.config["sent_emb_model"])}_{list_to_str(self.tokenizer.codebook_sizes, remove_blank=True)}.sem_ids'
            )
            rng_path = sem_ids_path + ".rng"

            # ---------- (A) if it's round 0, no sid file, save rng ----------
            if self.accelerator.is_main_process and os.path.exists(sem_ids_path) and not os.path.exists(rng_path):
                rng_state = {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),  # list
                }
                torch.save(rng_state, rng_path)
                self.log(f"[PIPELINE] RNG snapshot saved → {rng_path}")

            # wait for the main rank to finish writing .rng file
            self.accelerator.wait_for_everyone()

            # ---------- (B) Round1-k：the sid has existed, load the rng to recover the seeds ----------
            if os.path.exists(rng_path):
                #rng_state = torch.load(rng_path, map_location="cpu")
                rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
                random.setstate(rng_state["python"])
                np.random.set_state(rng_state["numpy"])
                torch.set_rng_state(rng_state["torch"])
                torch.cuda.set_rng_state_all(rng_state["cuda"])
                self.log(f"[PIPELINE] RNG snapshot restored ← {rng_path}")
            else:
                #
                self.log(f"[PIPELINE] WARN: {rng_path} not found; this is the first run.")
            # ------------------------------------------------------------------
      

        else:

            self.log(f'This is in test mode.')





        # Model
        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.raw_dataset, self.tokenizer)
        self.log('finish getting model')


 
        self.log(self.model)
        self.log(self.model.n_parameters)

        # Trainer
        if trainer is not None:
            self.trainer = trainer
        else:
            self.trainer = get_trainer(model_name)(self.config, self.model, self.tokenizer,self.Round)

    def run(self):
        # DataLoader

        if self.run_mode=='train':

            train_dataloader = DataLoader(
                self.tokenized_datasets['train'],
                batch_size=self.config['train_batch_size'],
                shuffle=True,
                collate_fn=self.tokenizer.collate_fn['train']
            )


            val_dataloader = DataLoader(
                self.tokenized_datasets['val'],
                batch_size=self.config['eval_batch_size'],
                shuffle=False,
                collate_fn=self.tokenizer.collate_fn['val']
            )


        test_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )
        self.log(f'current augmentation_probability is {self.config["augmentation_probability"]}')
        self.log('start training')
        if self.run_mode == 'train':
            self.trainer.fit(train_dataloader, val_dataloader)

        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)

        if self.run_mode == 'train':
         
            ckpt_path = self.trainer.saved_model_ckpt
            ckpt_dir = os.path.dirname(ckpt_path)

            if os.path.exists(ckpt_path):
                # case1: There exists this file, just load it.
                self.model.load_state_dict(torch.load(ckpt_path))
                if self.accelerator.is_main_process:
                    self.log(f'Loaded best model checkpoint from {ckpt_path}')
            else:
                # case2: No such file in test, find the lasted .pth file and load.
                if os.path.exists(ckpt_dir):
                    pth_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
                    if pth_files:
                        # Sort by time, find the newest one.
                        pth_files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))
                        latest_ckpt = os.path.join(ckpt_dir, pth_files[-1])
                        self.model.load_state_dict(torch.load(latest_ckpt))
                        if self.accelerator.is_main_process:
                            self.log(f'WARNING: Original checkpoint not found at {ckpt_path}')
                            self.log(f'Loaded latest model checkpoint instead: {latest_ckpt}')
                    else:
                        raise FileNotFoundError(f'No .pth files found in {ckpt_dir}')
                else:
                    raise FileNotFoundError(f'Checkpoint directory not found: {ckpt_dir}')
           

        elif self.run_mode=='test':
            test_model_ckpt = os.path.join(
                self.config['test_file_dir'], self.config['dataset'], self.config['category'], self.config['test_ckpt']
            )

            if os.path.exists(test_model_ckpt):
                # case1: There exists this file, just load it.
                self.model.load_state_dict(torch.load(test_model_ckpt))
                if self.accelerator.is_main_process:
                    self.log(f'[Test mode] Loaded best model checkpoint from {test_model_ckpt}')
            else:
                raise ValueError(f'[Test mode] You must have the ckpt at {test_model_ckpt} for test mode')





        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )

        if self.run_mode=='test':
            self.trainer.model = self.model
        test_results = self.trainer.evaluate(test_dataloader)



        self.trainer.test_results = test_results



        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
        self.log(f'Test Results: {test_results}')





    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
