import torch
from transformers import GPT2Config, GPT2LMHeadModel
import pickle
import os
from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from datetime import datetime  
import pickle
import os
###
class SASRec_upstream(AbstractModel):
    """
    SASRec model from Wang and McAuley, "Self-Attentive Sequential Recommendation." ICDM 2018.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (AbstractDataset): The dataset object.
        tokenizer (AbstractTokenizer): The tokenizer object.

    Attributes:
        gpt2 (GPT2LMHeadModel): The GPT-2 model used for the SASRec model.
    """
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
            log_func,length
    ):
        super(SASRec_upstream, self).__init__(config, dataset, tokenizer)

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )
        self.log=log_func
        self.gpt2 = GPT2LMHeadModel(gpt2config)
        #self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

        self.all_item_seqs_slided_len=length
        self.interaction2personalized_semantic_emb = {}
        self.num_operation_shift=0




    def compute_cold_item_emb(self, appeared_items: set):
        """
        Compute mean pooling embedding for appeared items, 
        and store it for later use in cold-start handling.
        """
        item_embeddings = self.gpt2.get_input_embeddings().weight  # [vocab_size, hidden_dim]


        appeared_ids = torch.tensor(sorted(list(appeared_items)), dtype=torch.long, device=item_embeddings.device)

        valid_item_embs = item_embeddings[appeared_ids]  # [num_appeared, hidden_dim]

        self.cold_item_emb = valid_item_embs.mean(dim=0).detach()
        self.log(f"[COLD] Precomputed cold embedding from {len(appeared_ids)} appeared items, "
                f"dim={self.cold_item_emb.shape}")





    @property
    def n_parameters(self) -> str:
        """
        Get the number of parameters in the model.

        Returns:
            str: A string representation of the number of parameters in the model.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        
        return f'#Here is SASRec_upstream.py in Pctx' \
            f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'



    def save_to_dict_file(self, file_model: str):
        """
        Save the interaction2personalized_semantic_emb dictionary to a .pkl file,
        ensuring all tensors are on CPU and detached from autograd to reduce size.

        Args:
            file_model (str): The base filename (without path), e.g., 'sasrec_diff_vectors'
        """
       

        filename = f"{file_model}"
        save_path = os.path.join(self.config['dict_dir'], self.config['dataset'],self.config['category'] ,filename)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        cpu_dict = {
            k: v.detach().cpu() if torch.is_tensor(v) else v
            for k, v in self.interaction2personalized_semantic_emb.items()
        }
        #
        with open(save_path, 'wb') as f:
            pickle.dump(cpu_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.log(f'[INFO] interaction2personalized_semantic_emb saved to {save_path}')


    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass of the model. Returns the logits and the loss.

        Args:
            batch (dict): The input batch.

        Returns:
            outputs (ModelOutput):
                The output of the model, which includes:
                - loss (torch.Tensor)
                - logits (torch.Tensor)
        """
        outputs = self.gpt2(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=True
        )

        return outputs


    def get_semantic_shift(self,batch):
        outputs = self.gpt2(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=True
        )

        last_hidden_states = outputs.hidden_states[-1]  # shape: (batch_size, seq_len, hidden_dim)

        batch_size = batch['input_ids'].size(0)
        for i in range(batch_size):
            user = batch['user'][i]
            input_ids = batch['input_ids'][i]
            seq_len = batch['seq_len'][i].item()
            hidden_seq = last_hidden_states[i, :seq_len, :]  # shape: [seq_len, hidden_dim]

            if 'slided' not in user and 'cold' not in user:

                preorder_tokens = []
                for t in range(seq_len):
                    item_token = input_ids[t].item()
                    key_user = user
                    key_item = str(item_token)
                    key_preorder = '0' if t == 0 else '_'.join(map(str, preorder_tokens))
                    key = f'{key_user}-{key_item}-{key_preorder}'

                    if t == 0:
                        value = hidden_seq[t]
                    else:
                     
                        value = hidden_seq[t]

                    self.interaction2personalized_semantic_emb[key] = value

                    preorder_tokens.append(item_token)
                self.num_operation_shift += 1
            elif 'slided' in user:

                if batch['attention_mask'][i][-1].item() != 1:
                    raise Exception(f"In user containing slided, User {user} \' s last token is invalid，attention_mask[-1] != 1")

                original_user = user.split('-')[0]
                item_token = input_ids[seq_len - 1].item()
                preorder = input_ids[:seq_len - 1].tolist()
                key_preorder = '_'.join(map(str, preorder))
                key = f'{original_user}-{item_token}-{key_preorder}'
              
                diff = hidden_seq[seq_len - 1]
                self.interaction2personalized_semantic_emb[key] = diff
                self.num_operation_shift += 1
            elif 'cold' in user:
          
                key = user.replace("-cold", "")

                if not hasattr(self, "cold_item_emb") or self.cold_item_emb is None:
                    raise RuntimeError("Cold embedding has not been precomputed! Call compute_cold_item_emb first.")

                self.interaction2personalized_semantic_emb[key] = self.cold_item_emb
                self.num_operation_shift += 1
                
                