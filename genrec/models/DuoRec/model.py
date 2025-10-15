import torch
from transformers import GPT2Config, GPT2LMHeadModel

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from torch import nn

class DuoRec(AbstractModel):
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
        tokenizer: AbstractTokenizer
    ):
        super(DuoRec, self).__init__(config, dataset, tokenizer)

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

        self.gpt2 = GPT2LMHeadModel(gpt2config)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

    
        self.aug_nce_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        self.sem_aug_nce_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

        self.ssl = config.get('contrast', None)  # 'us', 'su', 'us_x'
        self.tau = config.get('tau', 'Error')
        self.sim = config.get('sim', 'dot')
        self.lmd = config.get('lmd', 'Error')
        self.lmd_sem = config.get('lmd_sem', 'Error')
        self.batch_size = config['train_batch_size']
        self.mask_default = self.mask_correlated_samples(self.batch_size)
        self.loss_type = config['loss_type']


    @property
    def n_parameters(self) -> str:
        """
        Get the number of parameters in the model.

        Returns:
            str: A string representation of the number of parameters in the model.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'



    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask


    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """

   
        z_i = nn.functional.normalize(z_i, p=2, dim=1)
        z_j = nn.functional.normalize(z_j, p=2, dim=1)


        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels


    def forward(self, input_ids, attention_mask, seq_lens):

        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[-1]  # shape: [B, L, H]
        last_hidden = self.gather_index(hidden_states, seq_lens - 1)  # [B, H]

        return last_hidden



    def calculate_loss(self,batch,forward_fn):

        item_seq = batch['input_ids']
        item_seq_len = batch['seq_lens']
        attention_mask=batch['attention_mask']
        seq_output = forward_fn(item_seq,attention_mask, item_seq_len)
        pos_items = batch['labels']
        if self.loss_type == 'CE':
            all_item_emb = self.gpt2.get_input_embeddings().weight  # [V, H]
            logits = torch.matmul(seq_output, all_item_emb.T)  # [B, V]
            loss = self.loss_fct(logits, pos_items)
        else:
            raise ValueError('loss_fc is out of range')

        # Unsupervised NCE
        if self.ssl in ['us', 'un']:
            aug_seq_output = forward_fn(item_seq,attention_mask, item_seq_len)
            nce_logits, nce_labels = self.info_nce(seq_output, aug_seq_output, temp=self.tau,
                                                   batch_size=item_seq_len.shape[0], sim=self.sim)



            loss += self.lmd * self.aug_nce_fct(nce_logits, nce_labels)

        # Supervised NCE
        if self.ssl in ['us', 'su']:
            sem_aug, sem_aug_attention_mask, sem_aug_lengths = batch['sem_aug'],batch['sem_aug_attention_mask'], batch['sem_aug_lengths']
            sem_aug_seq_output = forward_fn(sem_aug, sem_aug_attention_mask, sem_aug_lengths)

            sem_nce_logits, sem_nce_labels = self.info_nce(seq_output, sem_aug_seq_output, temp=self.tau,
                                                           batch_size=item_seq_len.shape[0], sim=self.sim)


            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)

        if self.ssl == 'us_x':
            aug_seq_output = forward_fn(item_seq,attention_mask, item_seq_len)

            sem_aug, sem_aug_attention_mask, sem_aug_lengths = batch['sem_aug'], batch['sem_aug_attention_mask'], batch[
                'sem_aug_lengths']
            sem_aug_seq_output = forward_fn(sem_aug, sem_aug_attention_mask, sem_aug_lengths)

            sem_nce_logits, sem_nce_labels = self.info_nce(aug_seq_output, sem_aug_seq_output, temp=self.tau,
                                                           batch_size=item_seq_len.shape[0], sim=self.sim)

            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)


        return loss



    def gather_index(self, output, index):
        """
        Gather the output at a specific index.

        Args:
            output: The output tensor.
            index: The index tensor.

        Returns:
            torch.Tensor: The gathered output.
        """
        index = index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=index).squeeze(1)

    def generate(self, batch, n_return_sequences=1):
        """
        Generate sequences based on the input batch.

        Args:
            batch: The input batch.
            n_return_sequences (int): The number of sequences to generate.

        Returns:
            torch.Tensor: The generated sequences.
        """
        outputs = self.gpt2(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = self.gather_index(outputs.logits, batch['seq_lens'] - 1)
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return preds.unsqueeze(-1)
