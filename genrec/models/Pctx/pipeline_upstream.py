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
    get_dataset, get_tokenizer, get_model, get_trainer, log, get_config_upstream, get_model_upstream, get_trainer_upstream, get_tokenizer_upstream,get_pipeline_GR,get_cluster
#





def convert_split_to_dict(split_datasets):
    """
    Convert HuggingFace Dataset splits to a dict-of-dict format:
    {
        "train": {user_id: [item1, item2, ...], ...},
        "val":   {...},
        "test":  {...}
    }
    """
    result = {}
    for split_name, dataset in split_datasets.items():
        user_seq_dict = {}
        for row in dataset:
            user = row['user']
            item_seq = row['item_seq']
            user_seq_dict[user] = item_seq
        result[split_name] = user_seq_dict
    return result



class UpstreamPipeline:

    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        tokenizer: AbstractTokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
            Round=None,
    ):
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )

        self.config_upstream = get_config_upstream(
            model_name=model_name,

            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )
        self.model_name_para = model_name
        self.dataset_name_para = dataset_name
        self.tokenizer_para=tokenizer
        self.trainer_para=trainer
        self.config_dict_para=config_dict
        self.Round=Round

    def init_upstream_data_model_trainer(self):



        self.raw_dataset = get_dataset(self.dataset_name_para)(self.config_upstream)

        self.split_datasets = self.raw_dataset.split()

       


        self.split_datasets = convert_split_to_dict(self.split_datasets)



        train_seqs = self.split_datasets['train']
        min_seq_len = min(len(seq) for seq in train_seqs.values())

        self.log(f"Minimum sequence length in TRAIN split: {min_seq_len}")
        #
 


        if self.tokenizer_para is not None:
            self.tokenizer = self.tokenizer_para(self.config_upstream, self.raw_dataset)
        else:
            assert isinstance(self.config_upstream['pretrained_personalized_model_name'],
                              str), 'Tokenizer must be provided if model_name is not a string.'
            self.tokenizer = get_tokenizer_upstream(self.model_name_para, self.config_upstream['pretrained_personalized_model_name'])(
                self.config_upstream, self.raw_dataset)

        self.split_datasets['train'],len_train, self.appeared_items_sorted = self.tokenizer._generate_interaction2personalized_semantic_emb_pkl(self.split_datasets['train'], calculate_cold=True)

        self.split_datasets['test'],len_test = self.tokenizer._generate_interaction2personalized_semantic_emb_pkl(self.split_datasets['test'], calculate_cold=False)

        


        self.tokenized_datasets_upstream = self.tokenizer.tokenize_all_item_seqs_slided(self.split_datasets['train'])
        self.tokenized_datasets_upstream_test = self.tokenizer.tokenize_all_item_seqs_slided(self.split_datasets['test'])




        with self.accelerator.main_process_first():

            self.model_upstream = get_model_upstream(self.config_upstream['pretrained_personalized_model_name'])(
                self.config_upstream, self.raw_dataset, self.tokenizer,self.log,  0)

        # Trainer
        if self.trainer_para is not None:
            self.trainer = self.trainer_para
        else:

            self.trainer = get_trainer_upstream(self.model_name_para, self.config_upstream['pretrained_personalized_model_name'])(
                self.config_upstream, self.model_upstream, self.tokenizer)

    def init_upstream_basic(self):

        self.config_upstream['device'], self.config_upstream['use_ddp'] = init_device()

        # Accelerator
        self.project_dir = os.path.join(
            self.config_upstream['tensorboard_log_dir'],
            self.config_upstream["dataset"],
            self.config_upstream["model"]
        )

        self.accelerator = Accelerator(log_with='tensorboard', project_dir=self.project_dir)
        self.config_upstream['accelerator'] = self.accelerator

        init_seed(self.config_upstream['rand_seed'], self.config_upstream['reproducibility'])
        init_logger(self.config_upstream)
        self.logger = getLogger()
        self.log(f'Device: {self.config_upstream["device"]}')

    # tokenized_datasets_SASRec
    def run(self):

        self.init_upstream_basic()


        if self.config['run_GR_or_not']==False:



            # for train
            diff_filename = f"{self.config_upstream['diff_emb_name']}"
            diff_emb_path = os.path.join(self.config_upstream['dict_dir'], self.config_upstream['dataset']    , self.config_upstream['category'] ,diff_filename)
            if not os.path.exists(diff_emb_path):
                self.init_upstream_data_model_trainer()
                self.log('******Start to do the upstream task, get the personalied semantics of each item!*****')
                self.log("[important] Run the upstream only")
                self.log("[important] Run the upstream only")
                test_dataloader = DataLoader(
                    self.tokenized_datasets_upstream,
                    batch_size=self.config_upstream['eval_batch_size'],
                    shuffle=False,
                    collate_fn=self.tokenizer.collate_fn['test']
                )
                self.accelerator.wait_for_everyone()
                self.model_upstream = self.accelerator.unwrap_model(self.model_upstream)
                upstream_path = os.path.abspath(os.path.join(
                    os.path.dirname(__file__),
                    f'../../../{ self.config_upstream["pretrained_model_path"]  }/', f"{self.config_upstream['pretrained_model_name']}"
                ))
                self.model_upstream.load_state_dict(torch.load(upstream_path))
                self.model_upstream.compute_cold_item_emb(self.appeared_items_sorted)


                print(self.model_upstream)
                self.model_upstream, test_dataloader = self.accelerator.prepare(
                    self.model_upstream, test_dataloader
                )
                self.trainer.evaluate(test_dataloader,self.config_upstream['diff_emb_name'])
                self.log('[success] finish gerenating the pretrained_personalized_model_name.pkl')
                self.log('******The upstream task is finished*****')




            # for test
            diff_filename_test = f"{self.config_upstream['diff_emb_name_test']}"
            diff_emb_path = os.path.join(self.config_upstream['dict_dir'], self.config_upstream['dataset']    , self.config_upstream['category'] ,diff_filename_test)
            if not os.path.exists(diff_emb_path):
    
                self.log('******[test]Start to do the upstream task, get the personalied semantics of each item!*****')
                self.log("[important][test] Run the upstream only")
                self.log("[important][test] Run the upstream only")
                test_dataloader = DataLoader(
                    self.tokenized_datasets_upstream_test,
                    batch_size=self.config_upstream['eval_batch_size'],
                    shuffle=False,
                    collate_fn=self.tokenizer.collate_fn['test']
                )
                self.accelerator.wait_for_everyone()
                self.model_upstream = self.accelerator.unwrap_model(self.model_upstream)
                upstream_path = os.path.abspath(os.path.join(
                    os.path.dirname(__file__),
                    f'../../../{ self.config_upstream["pretrained_model_path"]  }/', f"{self.config_upstream['pretrained_model_name']}"
                ))
                self.model_upstream.load_state_dict(torch.load(upstream_path))
                print(self.model_upstream)
                self.model_upstream, test_dataloader = self.accelerator.prepare(
                    self.model_upstream, test_dataloader
                )
                self.trainer.evaluate(test_dataloader,self.config_upstream['diff_emb_name_test'])
                self.log('[success] finish gerenating the pretrained_personalized_model_name.pkl')
                self.log('******The upstream task is finished*****')

                #



            # strat to clustering

            cluster_file1 = os.path.join(self.config_upstream['dict_dir'], self.config_upstream['dataset']    , self.config_upstream['category'] ,f"{self.config_upstream['cluster_file1_ThreeEle']}")
            cluster_file2 = os.path.join(self.config_upstream['dict_dir'],self.config_upstream['dataset']    , self.config_upstream['category'] ,
                                         f"{self.config_upstream['cluster_file2_SetClusterIndexEmb']}")

            # if we have the two file, when refresh is true, we execute clustering, or else, we load
            # if we do not have the two files, we execute clustering.


            if ((os.path.exists(cluster_file1)) and (os.path.exists(cluster_file2))):
                if self.config_upstream['refresh_cluster_result']:

                    self.log(
                        '*****We have already prepared the cluster files, but we need to do it again, start to clustering now*****')
                    self.cluster_object = get_cluster(self.model_name_para)(self.config_upstream,self.log)
                    self.log('*****Finished clustering*****')
                else:
                    self.log('*****We have already prepared the cluster files and do not need to do it again*****')
            else:
                self.log('*****We do not have the cluster files, start to clustering now*****')
                self.cluster_object = get_cluster(self.model_name_para)(self.config_upstream,self.log)
                self.log('*****Finished clustering*****')

            #self.trainer.end()


        else:

            keys_to_merge = ['refresh_cluster_result', 'cluster_file1_ThreeEle', 'cluster_file2_SetClusterIndexEmb','dict_dir','diff_emb_name_test'] 
            for key in keys_to_merge:
                if key in self.config_upstream:
                    self.config_dict_para[key] = self.config_upstream[key]

            self.pipeline_GR = get_pipeline_GR(self.model_name_para) \
                    (
                    model_name=self.model_name_para,
                    dataset_name=self.dataset_name_para,
                    config_dict=self.config_dict_para,
                    accelerator=self.accelerator,
                    Round=self.Round ,
                )
            self.pipeline_GR.run()

    def log(self, message, level='info'):
        return log(message, self.accelerator, self.logger, level=level)
