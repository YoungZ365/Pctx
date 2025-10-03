from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
import numpy as np
import torch
class DuoRecTokenizer(AbstractTokenizer):
    """
    Tokenizer for SASRec model.

    An example:
        0: padding
        1-n_items: item tokens
        n_items+1: eos token

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        item2tokens (dict): A dictionary mapping items to their internal IDs.
        eos_token (int): The end-of-sequence token.
        ignored_label (int): Should be -100. Used to ignore the loss for padding tokens in `transformers`.
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        super(DuoRecTokenizer, self).__init__(config, dataset)

        self.item2tokens = dataset.item2id
        self.user2id = dataset.user2id
        self.eos_token = len(self.item2tokens) + 1
        self.ignored_label = -100




        self.index_for_train=0

        self.collate_fn = {
            'train': self.collate_fn_train,
            'val': self.collate_fn_val,
            'test': self.collate_fn_test,
        }
    def _init_tokenizer(self):
        pass



    def _token_single_item(self, item: str) -> int:
        """
        Tokenizes a single item.

        Args:
            item (str): The item to be tokenized.

        Returns:
            list: The tokens corresponding to the item.
        """
        return [self.item2tokens[item]]

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

        input_ids = []
        for item in example['item_seq'][:-1][-max_item_seq_len:]:
            input_ids.extend([item])
        seq_lens=len(input_ids)
        input_ids.extend([self.padding_token] * (self.max_token_seq_len - len(input_ids)))

        # attention_mask
        item_seq_len = min(len(example['item_seq'][:-1]), max_item_seq_len)
        attention_mask = [1] * (1* item_seq_len )
        attention_mask.extend([0] * (self.max_token_seq_len - len(attention_mask)))

        # labels
        labels = example['item_seq'][-1]

        return input_ids, attention_mask, labels,seq_lens



    def tokenize_aug_once(self, aug_seq: list) -> tuple:
        """
        Tokenizes a single augmentation sequence (without returning label).

        Args:
            aug_seq (list): A list of item IDs representing the augmented sequence.

        Returns:
            tuple: A tuple containing:
                - input_ids (List[int])
                - attention_mask (List[int])
                - seq_lens (int)
        """
        max_item_seq_len = self.config['max_item_seq_len']

        # input_ids
        input_ids = []
        for item in aug_seq[:-1][-max_item_seq_len:]:
            input_ids.append(item)
        seq_lens = len(input_ids)
        input_ids.extend([self.padding_token] * (self.max_token_seq_len - len(input_ids)))

        # attention_mask
        item_seq_len = min(len(aug_seq[:-1]), max_item_seq_len)
        attention_mask = [1] * item_seq_len
        attention_mask.extend([0] * (self.max_token_seq_len - len(attention_mask)))

        return input_ids, attention_mask, seq_lens

    def collate_fn_train(self, batch):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_seq_lens = []
        all_index = []

        sem_aug = []
        sem_aug_attention_mask = []
        sem_aug_lengths = []

        for data in batch:
    
            user_cur = str(data['user'].item())
            item_seq_cur = [int(i) for i in data['item_seq']]
            single_example = {'user': user_cur, 'item_seq': item_seq_cur}

            input_ids, attention_mask, labels, seq_len = self._tokenize_once(single_example)
            index = int(data['index'])

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
            all_seq_lens.append(seq_len)
            all_index.append(index)

          
            same_indices = self.same_target_index_dict.get(index, [])
            if same_indices:
                sampled_index = np.random.choice(same_indices)
                sampled_seq = self.full_train_dataset[int(sampled_index)]['item_seq']  
            else:
                sampled_seq = item_seq_cur  
            aug_ids, aug_mask, aug_len = self.tokenize_aug_once(sampled_seq)

            sem_aug.append(aug_ids)
            sem_aug_attention_mask.append(aug_mask)
            sem_aug_lengths.append(aug_len)

        return {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention_mask, dtype=torch.long),
            'labels': torch.tensor(all_labels, dtype=torch.long),
            'seq_lens': torch.tensor(all_seq_lens, dtype=torch.long),
            'index': torch.tensor(all_index, dtype=torch.long),

            'sem_aug': torch.tensor(sem_aug, dtype=torch.long),
            'sem_aug_attention_mask': torch.tensor(sem_aug_attention_mask, dtype=torch.long),
            'sem_aug_lengths': torch.tensor(sem_aug_lengths, dtype=torch.long),
        }

    def collate_fn_val(self, batch):

        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_seq_lens = []

        for data in batch:
            # user_id
            user_cur = str(data['user'].item())

            # item_seq is a int list, consisting of itemIDs.
            # item_seq = data['item_seq']
            item_seq_cur = [int(i) for i in data['item_seq']]
            single_example = {'user': user_cur, 'item_seq': item_seq_cur}

            input_ids, attention_mask, labels, seq_len = self._tokenize_once(single_example)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append([labels])
            all_seq_lens.append(seq_len)
        return {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention_mask, dtype=torch.long),
            'labels': torch.tensor(all_labels, dtype=torch.long),
            'seq_lens': torch.tensor(all_seq_lens, dtype=torch.long)

        }
    def collate_fn_test(self, batch):

        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_seq_lens = []

        for data in batch:
            # user_id
            user_cur = str(data['user'].item())

            # item_seq is a int list, consisting of itemIDs.
            # item_seq = data['item_seq']
            item_seq_cur = [int(i) for i in data['item_seq']]
            single_example = {'user': user_cur, 'item_seq': item_seq_cur}

            input_ids, attention_mask, labels, seq_len = self._tokenize_once(single_example)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append([labels])
            all_seq_lens.append(seq_len)
        return {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention_mask, dtype=torch.long),
            'labels': torch.tensor(all_labels, dtype=torch.long),
            'seq_lens': torch.tensor(all_seq_lens, dtype=torch.long)

        }


    def tokenize_function(self, example: dict, split: str) -> dict:
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

        user = self.user2id[example['user'][0]]
        item_seq = [self.item2tokens[item] for item in example['item_seq'][0]]

    
        if split=='train':
            index = [example['index'][0]]
  
            return {
                'user': [user],
                'item_seq': [item_seq],
                'index': [index]
            }
        else:

            return {
                'user': [user],
                'item_seq': [item_seq]
            }

    def build_same_target_index_dict(self, dataset, cache_path=None):

        from collections import defaultdict
        import os
        import json

        label2indices = defaultdict(list)
        same_target_index_dict = {}
    
        for example in dataset:

            item_seq = example["item_seq"]
            index_list = example["index"]




            if isinstance(index_list, list):
                idx = int(index_list[0])
            else:
                idx = int(index_list)

        
            if isinstance(item_seq, (list, torch.Tensor)):
                label = int(item_seq[-1])
            else:
                raise ValueError(f"item_seq format unexpected: {item_seq}")

            label2indices[label].append(idx)


        for example in dataset:
            item_seq = example["item_seq"]
            index_list = example["index"]

            if isinstance(index_list, list):
                idx = int(index_list[0])
            else:
                idx = int(index_list)

            label = int(item_seq[-1])

            same_indices = [i for i in label2indices[label] if i != idx]
            same_target_index_dict[idx] = same_indices

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(same_target_index_dict, f)

        return same_target_index_dict

    def slide_and_index_for_train_dataset(self, dataset: dict, split: str) -> dict:



        if split != 'train':
            raise ValueError('must be train')

        from datasets import Dataset

        sliding_examples = []

        for example in dataset:
            item_seq = example['item_seq']
            user = example['user']
     
            n_return_examples = len(item_seq) - 1
            for i in range(n_return_examples):
                sliding_examples.append({
                    'user': user,
                    'item_seq': item_seq[:i + 2],
                    'index': self.index_for_train
                })
                self.index_for_train += 1



        dataset = Dataset.from_list(sliding_examples)
        return dataset




    def tokenize(self, datasets: dict) -> dict:
        """
        Tokenizes the datasets using the specified tokenizer function.

        Args:
            datasets (dict): A dictionary containing the datasets to be tokenized.

        Returns:
            dict: A dictionary containing the tokenized datasets.
        """
        tokenized_datasets = {}


        datasets['train'] = self.slide_and_index_for_train_dataset(datasets['train'], 'train')

        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')


        train_dataset = tokenized_datasets['train']
        self.same_target_index_dict = self.build_same_target_index_dict(train_dataset)


        self.full_train_dataset = tokenized_datasets['train'] 

        return tokenized_datasets

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return self.eos_token + 1

    @property
    def max_token_seq_len(self) -> int:
        """
        Returns the maximum token sequence length, including the EOS token.

        Returns:
            int: The maximum token sequence length.
        """
        return self.config['max_item_seq_len']
