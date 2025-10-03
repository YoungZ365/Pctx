from logging import getLogger
from typing import Union
import torch
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import get_config, init_seed, init_logger, init_device, \
    get_dataset, get_tokenizer, get_model, get_trainer, log


class Pipeline:


    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        tokenizer: AbstractTokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )

        self.config['device'], self.config['use_ddp'] = init_device() 

        # Accelerator
        self.project_dir = os.path.join(
            self.config['tensorboard_log_dir'],
            self.config["dataset"],
            self.config["model"]
        )


        self.accelerator = Accelerator(log_with='tensorboard', project_dir=self.project_dir)
        self.config['accelerator'] = self.accelerator


        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(f'Device: {self.config["device"]}')

        self.raw_dataset = get_dataset(dataset_name)(self.config)

        self.log('231')
        self.log(self.raw_dataset)





        self.split_datasets = self.raw_dataset.split()

        if tokenizer is not None:
            self.tokenizer = tokenizer(self.config, self.raw_dataset)
        else:
            assert isinstance(model_name, str), 'Tokenizer must be provided if model_name is not a string.'
            self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)

        self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)



        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.raw_dataset, self.tokenizer)


        self.log(self.model)
        self.log(self.model.n_parameters)

        # Trainer
        if trainer is not None:
            self.trainer = trainer
        else:

            self.trainer = get_trainer(model_name)(self.config, self.model, self.tokenizer)

    def run(self):
        # DataLoader


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




        self.trainer.fit(train_dataloader, val_dataloader)

        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)

        self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))

        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )
        if self.accelerator.is_main_process:
            self.log(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')

        test_results = self.trainer.evaluate(test_dataloader)

        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
        self.log(f'Test Results: {test_results}')

        self.trainer.end()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
