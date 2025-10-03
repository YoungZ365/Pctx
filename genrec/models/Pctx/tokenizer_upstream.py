import os
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
from genrec.models.layers import RQVAEModel
from genrec.utils import list_to_str
#


class SASRecTokenizer(AbstractTokenizer):
    """
    Tokenizer for the TIGER model.

    An example when "rq_codebook_size == 256, rq_n_codebooks == 3, n_user_tokens == 2000":
        0: padding
        1-256: digit 1
        257-512: digit 2
        513-768: digit 3
        769-1024: digit 4 (used to avoid conflicts)
        1025-3024: user tokens
        3025: eos

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        item2tokens (dict): A dictionary mapping items to their semantic IDs.
        base_user_id (int): The base user ID.
        n_user_tokens (int): The number of user tokens.
        eos_token (int): The end-of-sequence token.
    """



    def __init__(self, config: dict, dataset: AbstractDataset):
        super(SASRecTokenizer, self).__init__(config, dataset)
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']

        self.item2id=dataset.id_mapping['item2id']

        self.max_item_seq_len = self.config['max_item_seq_len']
        self.all_item_seqs_slided = {}
        self.all_item_seqs = dataset.all_item_seqs




        self.n_items=dataset.n_items

   
        min_seq_len = min(len(seq) for seq in self.all_item_seqs.values())

        self.log(f"Minimum interaction sequence length across all users:{min_seq_len}")



        self.item2tokens = dataset.item2id
        self.eos_token = len(self.item2tokens) + 1

   



    def _generate_interaction2personalized_semantic_emb_pkl(self,item_seqs,calculate_cold):
        #
        self.log(f'the total number of items: {self.n_items}')
        self.log("excute _generate_interaction2personalized_semantic_emb_pkl")
        item_seqs_mapping2id={}
        for user, item in item_seqs.items():
            user_mapping2id_str=str(self.user2id[user])
            item_mapping2id = [self.item2id[i] for i in item]

            item_seqs_mapping2id[user_mapping2id_str]=item_mapping2id

        if calculate_cold==True:
        # 2) find all appered items
            appeared_items = set()
            for items in item_seqs_mapping2id.values():
                appeared_items.update(items)        

            all_items = set(range(1, self.n_items + 1))
            missing_items = sorted(list(all_items - appeared_items))

            if missing_items:
                self.log(f"[CHECK] Missing {len(missing_items)} items （cold-start item）: {missing_items[:50]}{'...' if len(missing_items) > 50 else ''}")
            else:
                self.log("[CHECK] All items appeared at least once (no cold-start item).")


        #start to slide
        item_seqs_slided={}
        # if the length of list of item <= max, no additional operation

        for user, item in item_seqs_mapping2id.items():
            if(len(item))<= self.max_item_seq_len:
                item_seqs_slided[user]=item
            else:
                # first, the former max item is inserted as normal
                item_seqs_slided[user]=item[:self.max_item_seq_len] #get the former max_item_seq_len item

                # for the reserve, change the name and insert
                num_slide=len(item)-self.max_item_seq_len
                for i in range(1,1+num_slide):
                    item_seqs_slided[user+f'-slided-{i}']=item[i:i+self.max_item_seq_len]


        if calculate_cold==True:
            # 4) make placeholder for cold-start item
           
            appeared_items_sorted = sorted(list(appeared_items))
            #we do not care the value of dummy_seq, just use it as the placeholder in case raising error.
            dummy_seq = appeared_items_sorted[:5] if len(appeared_items_sorted) >= 5 else appeared_items_sorted
            if not dummy_seq:
                dummy_seq = [0, 0, 0, 0, 0]  
            ###########
            for miss_item in missing_items:
                cold_key = f"0-{miss_item}-0-cold"
                item_seqs_slided[cold_key] = dummy_seq
            


        item_seqs_slided_len=len(item_seqs_slided)
        if calculate_cold==True:
            return item_seqs_slided,item_seqs_slided_len,appeared_items_sorted
        else:
            return item_seqs_slided,item_seqs_slided_len




    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """
        Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

        Args:
            dataset (AbstractDataset): The dataset containing the sentences to encode.
            output_path (str): The path to save the encoded sentence embeddings.

        Returns:
            numpy.ndarray: The encoded sentence embeddings.
        """
        assert self.config['metadata'] == 'sentence', \
            'TIGERTokenizer only supports sentence metadata.'

        sent_emb_model = SentenceTransformer(
            self.config['sent_emb_model']
        ).to(self.config['device'])


        meta_sentences = [] # 1-base, meta_sentences[0] -> item_id = 1


        for i in range(1, dataset.n_items+1):

            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])


        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device']
        )

        sent_embs.tofile(output_path)

        return sent_embs

    @property
    def n_digit(self):
        """
        Returns the number of digits for the tokenizer.

        The number of digits is determined by the value of `rq_n_codebooks` in the configuration.
        """
        return self.config['rq_n_codebooks'] + 1

    @property
    def codebook_sizes(self):
        """
        Returns the codebook size for the TIGER tokenizer.

        If `rq_codebook_size` is a list, it returns the list as is.
        If `rq_codebook_size` is an integer, it returns a list with `n_digit` elements,
        where each element is equal to `rq_codebook_size`.

        Returns:
            list: The codebook size for the TIGER tokenizer.
        """
        if isinstance(self.config['rq_codebook_size'], list):
            return self.config['rq_codebook_size']
        else:
            return [self.config['rq_codebook_size']] * self.n_digit

    def _token_single_user(self, user: str) -> int:
        """
        Tokenizes a single user.

        Args:
            user (str): The user to tokenize.

        Returns:
            int: The tokenized user ID.

        """
        user_id = self.user2id[user]
        return self.base_user_token + user_id % self.n_user_tokens

    def _token_single_item(self, item: str) -> int:
        """
        Tokenizes a single item.

        Args:
            item (str): The item to be tokenized.

        Returns:
            list: The tokens corresponding to the item.
        """
        return self.item2tokens[item]


    def _tokenize_once(self, example: dict) -> tuple:

        """
        Tokenizes a single example.

        Args:
            example (dict): A dictionary containing the example data.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, and labels.
        """
        max_item_seq_len = self.config['max_item_seq_len']

        # input_ids
        user_token = self._token_single_user(example['user'])


        input_ids = [user_token]

        for item in example['item_seq'][:-1][-max_item_seq_len:]:
            input_ids.extend(self._token_single_item(item))

        input_ids.append(self.eos_token)

        input_ids.extend([self.padding_token] * (self.max_token_seq_len - len(input_ids)))


        # attention_mask

        item_seq_len = min(len(example['item_seq'][:-1]), max_item_seq_len)

        attention_mask = [1] * (self.n_digit * item_seq_len + 2)

        attention_mask.extend([0] * (self.max_token_seq_len - len(attention_mask)))

        # labels
        labels = list(self._token_single_item(example['item_seq'][-1])) + [self.eos_token]


        return input_ids, attention_mask, labels


    def _tokenize_once_all_item_seqs_slided(self, example: dict) -> tuple:

        """
        Tokenizes a single example.

        Args:
            example (dict): A dictionary containing the example data.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, and labels.
        """
        max_item_seq_len_SASRec = self.config['max_item_seq_len_SASRec']

        if len(example['item_seq'])>max_item_seq_len_SASRec:
            raise ValueError('your all_item_seqs_slided is wrong, some item_seq len is over max')

        input_ids = []
        item_seq = example['item_seq']
        seq_lens = len(item_seq)

        for item in example['item_seq']:
            input_ids.append(item)
        attention_mask = [1] * seq_lens

        pad_lens = max_item_seq_len_SASRec - seq_lens
        #input_ids.append(self.eos_token)

        input_ids.extend([self.padding_token] * pad_lens  )

        attention_mask.extend([0] * pad_lens)

        # attention_mask

        return input_ids, attention_mask,seq_lens

    def tokenize_function_all_item_seqs_slided(self, example: dict) -> dict:
        """
        Tokenizes the input example based on the specified split.

        Args:
            example (dict): The input example containing user and item sequence.
            split (str): The split type, either 'train' or any other value.

        Returns:
            dict: A dictionary containing the tokenized input, attention mask, and labels.
                - If split is 'train', returns:
                    {
                        'input_ids': List[List[int]],
                        'attention_mask': List[List[int]],
                        'labels': List[List[int]]
                    }
                - If split is not 'train', returns:
                    {
                        'input_ids': List[int],
                        'attention_mask': List[int],
                        'labels': List[int]
                    }
        """

        input_ids, attention_mask,seq_len = self._tokenize_once_all_item_seqs_slided({k: v[0] for k, v in example.items()})

        seq_len_list=[seq_len]

        user=example['user']


        return {'input_ids': [input_ids], 'attention_mask': [attention_mask] ,'user':user  , 'seq_len': seq_len_list       }

    def tokenize_all_item_seqs_slided(self, datasets: dict) -> dict:
        """
        Tokenizes the given datasets.

        Args:
            datasets (dict): A dictionary of datasets to tokenize.

        Returns:
            dict: A dictionary of tokenized datasets.
        """

        sample_list = []
        for user_id, item_seq in datasets.items():
            sample_list.append({
                "user": str(user_id),
                "item_seq": item_seq
            })

        datasets = Dataset.from_list(sample_list)

        tokenized_datasets = {}

        tokenized_datasets = datasets.map(
            lambda t: self.tokenize_function_all_item_seqs_slided(t),
            batched=True,
            batch_size=1,
            remove_columns=datasets.column_names,
            num_proc=self.config['num_proc'],
            desc=f'Tokenizing tokenize_all_item_seqs_slided set: '
        )


        tokenized_datasets.set_format(type='torch')

        return tokenized_datasets



    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size for the TIGER tokenizer.
        """
        return self.eos_token + 1

    @property
    def max_token_seq_len(self) -> int:
        """
        Returns the maximum token sequence length for the TIGER tokenizer.
        """
        # +2 for user token and eos token
        #return self.config['max_item_seq_len'] * self.n_digit + 2
        return self.config['max_item_seq_len']

  