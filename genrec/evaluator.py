import torch
import pickle
import os
class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])

        run_mode = self.config.get('run_mode', '-1')  
        if run_mode=='train':

            pkl_path = os.path.join(self.config['dict_dir'], self.config['dataset'], self.config['category'],
                                    self.config['sidStr2itemIDInt'])

            with open(pkl_path, "rb") as f:
                self.label_dict = pickle.load(f)
        elif run_mode=='test':
            test_model_sidStr2itemIDInt = os.path.join(
                self.config['test_file_dir'], self.config['dataset'], self.config['category'],
                self.config['sidStr2itemIDInt']
            )
            with open(test_model_sidStr2itemIDInt, "rb") as f:
                self.label_dict = pickle.load(f)

    def map_labels_and_preds(self, preds, labels):
        """
        Given predictions and labels, map them to their corresponding item IDs.

        Args:
            preds (Tensor): Predictions of shape [batch_size, maxk, ...].
            labels (Tensor): Ground truth labels of shape [batch_size, ...].

        Returns:
            Tuple[List[int], List[List[int]]]:
                - label_mapped: a list of int item ids for each label
                - preds_mapped: a list of list of int item ids for each prediction
        """
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        label_mapped = []
        preds_mapped = []

        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
            label_key = str(cur_label)
            int_cur_label_to_item = self.label_dict.get(label_key, -1)
            label_mapped.append(int_cur_label_to_item)

            preds_row = []
            for j in range(preds.shape[1]):
                cur_pred = preds[i, j].tolist()
                str_pred = str(cur_pred)
                int_cur_pred_to_item = self.label_dict.get(str_pred, -1)
                preds_row.append(int_cur_pred_to_item)
            preds_mapped.append(preds_row)

        return label_mapped, preds_mapped
    

    def calculate_pos_index(self, preds, labels):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()


            if self.eos_token in cur_label:
                # because labels is 5 tokens ending with eos token
                # but pres[2] only have 4 tokens meaning sids
                # we need to delete the last token of labels
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]

                int_cur_label_to_item= self.label_dict[str(cur_label)]

            for j in range(self.maxk):
                cur_pred = preds[i, j].tolist()
                str_pred = str(cur_pred)
                if str_pred in self.label_dict:
                    int_cur_pred_to_item = self.label_dict[str_pred]
                else:
                    int_cur_pred_to_item = -1

                if int_cur_label_to_item == int_cur_pred_to_item:
                    #print(f'int_cur_label_to_item={int_cur_label_to_item} == int_cur_pred_to_item={int_cur_pred_to_item}')
                    pos_index[i, j] = True
                    break


        return pos_index

    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

    def calculate_metrics(self, preds, labels):
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        return results
