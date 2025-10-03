import torch
from transformers import T5Config, T5ForConditionalGeneration
from collections import defaultdict
from genrec.model import AbstractModel
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
import numpy as np
import os
import pickle
class Pctx(AbstractModel):
    """
     model from Rajput et al. "Recommender Systems with Generative Retrieval." NeurIPS 2023.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (AbstractDataset): The dataset object.
        tokenizer (AbstractTokenizer): The tokenizer object.

    Attributes:
        t5 (T5ForConditionalGeneration): The T5 model for conditional generation.
    """
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(Pctx, self).__init__(config, dataset, tokenizer)

        t5config = T5Config(
            num_layers=config['num_layers'], 
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            activation_function=config['activation_function'],
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.padding_token,
            eos_token_id=tokenizer.eos_token,
            decoder_start_token_id=0,
            feed_forward_proj=config['feed_forward_proj'],
            n_positions=tokenizer.max_token_seq_len,
        )

        self.t5 = T5ForConditionalGeneration(config=t5config)
 
        self.n_inference_ensemble = config['n_inference_ensemble']
        
    @property
    def n_parameters(self) -> str:
        """
        Calculates the number of trainable parameters in the model.

        Returns:
            str: A string containing the number of embedding parameters, non-embedding parameters, and total trainable parameters.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.t5.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass of the model. Returns the output logits and the loss value.

        Args:
            batch (dict): A dictionary containing the input data for the model.

        Returns:
            outputs (ModelOutput): 
                The output of the model, which includes:
                - loss (torch.Tensor)
                - logits (torch.Tensor)
        """
        outputs = self.t5(**batch)
        return outputs



    # helper to load sid->itemID mapping
    def _load_sid_mapping_if_needed(self):
        """
        Lazily load the sidStr->itemIDInt mapping from the same paths used by Evaluator.
        The mapping key is a Python list serialized as str([...]), value is an int itemID.
        """
        if hasattr(self, "sidStr2itemIDInt") and isinstance(self.sidStr2itemIDInt, dict):
            return  # already loaded

        run_mode = self.config.get('run_mode', '-1')
        if run_mode == 'train':
            pkl_path = os.path.join(
                self.config['dict_dir'], self.config['dataset'], self.config['category'],
                self.config['sidStr2itemIDInt']
            )
        else:  # 'test'
            pkl_path = os.path.join(
                self.config['test_file_dir'], self.config['dataset'], self.config['category'],
                self.config['sidStr2itemIDInt']
            )

        with open(pkl_path, "rb") as f:
            self.sidStr2itemIDInt = pickle.load(f)

    # merge N beams that map to the same item
    @staticmethod
    def _merge_beams_for_one_group(sids_tensor, scores_tensor, n_return_sequences, n_digit, map_dict):
        """
        Merge beams that map to the same itemID using log-sum-exp over log-prob scores.

        Args:
            sids_tensor: Tensor [num_beams, n_digit], each row is a 4-token sid sequence
            scores_tensor: Tensor [num_beams], each is cumulative log-prob of that path
            n_return_sequences: how many sequences to keep after merging
            n_digit: sid length (e.g., 4)
            map_dict: dict[str(list_of_int)] -> int itemID

        Returns:
            kept_sids: Tensor [<=num_beams, n_digit] (merged representatives + unmerged)
            kept_scores: Tensor [<=num_beams] (merged scores + unmerged)
        """
        device = sids_tensor.device
        num_beams = sids_tensor.shape[0]

        # 1) Group indices by itemID (itemID == -1 means no mapping -> keep as its own group)
        itemid_to_indices = defaultdict(list)
        idx_to_itemid = []

        for j in range(num_beams):
            sid_list = sids_tensor[j].tolist()
            key = str(sid_list)
            itemid = map_dict.get(key, -1)
            idx_to_itemid.append(itemid)
            itemid_to_indices[itemid].append(j)

        # 2) For each itemID group, merge scores with log-sum-exp.
        #  The representative path is the member with the highest original score within that group.
        kept_sid_rows = []
        kept_score_vals = []

        for itemid, idx_list in itemid_to_indices.items():
            group_scores = scores_tensor[idx_list]  # [m]
            # log-sum-exp to merge probabilities in log-space
            merged_score = torch.logsumexp(group_scores, dim=0)

            # choose representative path: the one with the max original score in this group
            best_local_idx = idx_list[torch.argmax(group_scores).item()]
            rep_sid = sids_tensor[best_local_idx]  # [n_digit]

            kept_sid_rows.append(rep_sid.unsqueeze(0))
            kept_score_vals.append(merged_score.unsqueeze(0))

        kept_sids = torch.cat(kept_sid_rows, dim=0)          # [num_groups, n_digit]
        kept_scores = torch.cat(kept_score_vals, dim=0)       # [num_groups]
        # 3) Sort by merged scores (descending) and keep top n_return_sequences
        topk = min(n_return_sequences, kept_scores.shape[0])
        top_scores, top_idx = torch.topk(kept_scores, k=topk, largest=True, sorted=True)
        kept_sids = kept_sids[top_idx]                        # [topk, n_digit]
        kept_scores = top_scores                              # [topk]

        # 4) If we have fewer than n_return_sequences, pad with [-1, -1, -1, -1]
        if kept_sids.shape[0] < n_return_sequences:
            pad_rows = n_return_sequences - kept_sids.shape[0]
            pad_sid = torch.full((pad_rows, n_digit), -1, dtype=kept_sids.dtype, device=device)
            kept_sids = torch.cat([kept_sids, pad_sid], dim=0)
            # no need to pad scores for downstream (we only return sids to caller)
        return kept_sids, kept_scores





   
    # We use merging decoding tree, so we modify the original generate
    def generate(self, batch: dict, n_return_sequences: int = 1) -> torch.Tensor:
        """
        Generate sequences using beam search, but before returning the top-k,
        merge beam paths that map to the same itemID (via sid->itemID pkl), then re-rank.

        The merging is done per-ensemble-slice (if n_ensemble>1), and the final
        cross-ensemble fusion still uses the existing DCG-style voting in your code.
        """
        # Handle ensemble size (kept from your original code)
        n_ensemble = 1
        if self.n_inference_ensemble != -1:
            if batch['input_ids'].shape[0] != batch['labels'].shape[0]:
                assert batch['input_ids'].shape[0] % self.n_inference_ensemble == 0
                n_ensemble = self.n_inference_ensemble
        batch_size = batch['input_ids'].shape[0] // n_ensemble

        # sid length
        n_digit = self.tokenizer.n_digit

        # IMPORTANT: fetch ALL beams (num_return_sequences=num_beams) and also get scores
        num_beams = self.config['num_beams']
        # We set return_score=True to obtain per-beam scores.
        # Note beam_search (when return_score=True) returns *length-normalized* scores,
        # so we will multiply back by (seq_len-1) to recover the cumulative log-prob.
        sequences_all, scores_avg = self.beam_search(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=n_digit + 2,
            num_beams=num_beams,
            num_return_sequences=num_beams,   # fetch ALL beams first
            return_score=True
        )
        # sequences_all: [batch_size * num_beams, seq_len]
        # scores_avg:    [batch_size * num_beams]  (avg log-prob per token)

        # Recover cumulative log-prob from averaged scores.
        seq_len = sequences_all.shape[1]
        # Your beam_search divides by (decoder_input_ids.shape[1] - 1)
        # -> multiply back to get sum log prob (cumulative)
        scores_sum = scores_avg * (seq_len - 1)  # [B*num_beams]

        # Cut off the start token and eos to obtain 4-token sids
        sids_all = sequences_all[:, 1:1 + n_digit].long()  # [B*num_beams, n_digit]

        # Reshape into [B, n_ensemble, num_beams, ...]
        sids_all = sids_all.view(batch_size, n_ensemble, num_beams, n_digit)
        scores_sum = scores_sum.view(batch_size, n_ensemble, num_beams)

        # Load sid->itemID mapping once
        self._load_sid_mapping_if_needed()
        sid_map = self.sidStr2itemIDInt  # dict[str(list)] -> int

        # For each (batch, ensemble) slice: merge beams by item, then keep top n_return_sequences
        # Produce a tensor to feed the *existing* cross-ensemble voting code
        decoded_outputs_merged = torch.full(
            (batch_size, n_ensemble, n_return_sequences, n_digit),
            -1, dtype=torch.long, device=sids_all.device
        )

        for bid in range(batch_size):
            for ens in range(n_ensemble):
                sids_slice = sids_all[bid, ens, :, :]        # [num_beams, n_digit]
                scores_slice = scores_sum[bid, ens, :]       # [num_beams]

                kept_sids, _ = self._merge_beams_for_one_group(
                    sids_slice, scores_slice, n_return_sequences, n_digit, sid_map
                )  # kept_sids: [n_return_sequences, n_digit] (padded with -1 rows if needed)

                decoded_outputs_merged[bid, ens] = kept_sids  # fill for this ensemble slice

        # Now reuse your *existing* cross-ensemble fusion (pred2score + DCG weight).
        final_outputs = torch.full(
            (batch_size, n_return_sequences, n_digit), -1, dtype=torch.long, device=sids_all.device
        )

        for bid in range(batch_size):
            pred2score = defaultdict(float)
            for ens in range(n_ensemble):
                for j in range(n_return_sequences):
                    pred_tuple = tuple(decoded_outputs_merged[bid, ens, j].tolist())
                    if pred_tuple[0] != -1:
                        # keep the existing DCG-style ensemble weighting
                        pred2score[pred_tuple] += 1.0 / np.log2(j + 2)
            # sort and pick top-k
            all_scores = sorted(pred2score.items(), key=lambda x: x[1], reverse=True)
            for j in range(min(n_return_sequences, len(all_scores))):
                final_outputs[bid, j] = torch.tensor(all_scores[j][0], dtype=torch.long, device=sids_all.device)

        return final_outputs.to(batch['labels'].device)
     
     
    def beam_search(
        self,
        input_ids,
        attention_mask,
        max_length=6,
        num_beams=1,
        num_return_sequences=1,
        return_score=False
    ):
        """
        Adpated from huggingface's implementation
        https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

        Perform beam search to generate sequences using the specified model. 

        *** This implementation does not include stopping conditions based on end-of-sequence (EOS) tokens. Instead, the
        sequence generation is controlled solely by the `max_length` parameter. ***

        Note: In scenarios where the generation should explicitly detect and respond to EOS tokens 
        to terminate the sequence early, this function would need modifications. In the current setup,
        setting `max_length` to a suitable fixed value (e.g., 6) can serve the purpose by limiting
        the maximum sequence length.

        Parameters:
        - input_ids (torch.Tensor): Tensor of input ids.
        - attention_mask (torch.Tensor): Tensor representing the attention mask.
        - max_length (int): Maximum length of the sequence to be generated; controls when to stop extending the sequence.
        - num_beams (int): Number of beams for beam search.
        - num_return_sequences (int): Number of sequences to return.
        - return_score (bool): If True, returns a tuple of (sequences, scores) where 'scores' are the average log likelihood of the returned sequences.

        Returns:
        - torch.Tensor: The final decoder input ids from the beam search, or a tuple of (decoder_input_ids, scores) if 'return_score' is True.

        Example usage:
        # Assuming the model, input_ids, and attention_mask are predefined:
        sequences = beam_search(model, input_ids, attention_mask, max_length=6, num_beams=5, num_return_sequences=5)
        """

        batch_size = input_ids.shape[0]

        # Prepare beam search inputs
        input_ids, attention_mask, decoder_input_ids, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
                input_ids, attention_mask, batch_size, num_beams
            )
        # Store encoder_outputs to prevent running full forward path repeatedly
        with torch.no_grad():
            encoder_outputs = self.t5.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

        # Beam search loop
        while decoder_input_ids.shape[1] < max_length:
            with torch.no_grad():
                outputs = self.t5(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids
                )

            decoder_input_ids, beam_scores = self.beam_search_step(
                outputs.logits,
                decoder_input_ids,
                beam_scores,
                beam_idx_offset,
                batch_size,
                num_beams
            )

        # (batch_size * num_beams, ) -> (batch_size * num_return_sequences, )
        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
        selection_mask[:, :num_return_sequences] = True

        if return_score:
            return decoder_input_ids[selection_mask.view(-1), :], \
                beam_scores[selection_mask.view(-1)] / (decoder_input_ids.shape[1] - 1)

        return decoder_input_ids[selection_mask.view(-1), :]

    def prepare_beam_search_inputs(self, input_ids, attention_mask, batch_size, num_beams):
        """
        Adpated from huggingface's implementation
        https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

        Prepares and duplicates the input data for beam search decoding.

        This function initializes decoder input IDs and beam scores, creates an offset for beam indices, 
        and expands the input_ids and attention_mask tensors to accommodate the specified number of beams for each instance in the batch.

        Parameters:
        - input_ids (torch.Tensor): The input IDs tensor of shape (batch_size, sequence_length) used for the encoder part of the model.
        - attention_mask (torch.Tensor): The attention mask tensor of shape (batch_size, sequence_length) indicating to the model which tokens should be attended to.
        - batch_size (int): The number of instances per batch in the input data.
        - num_beams (int): The number of beams to use in beam search. This expands the input data and scores accordingly.

        Returns:
        - input_ids (torch.Tensor): The expanded input IDs tensor to match the number of beams, shape (batch_size * num_beams, sequence_length).
        - attention_mask (torch.Tensor): The expanded attention mask tensor to match the number of beams, shape (batch_size * num_beams, sequence_length).
        - initial_decoder_input_ids (torch.Tensor): The initialized decoder input IDs for each beam, shape (batch_size * num_beams, 1).
        - initial_beam_scores (torch.Tensor): The initialized scores for each beam, flattened to a single dimension, shape (batch_size * num_beams,).
        - beam_idx_offset (torch.Tensor): An offset for each beam index to assist in reordering beams during the search, shape (batch_size * num_beams,).

        Each input sequence is replicated 'num_beams' times to provide separate candidate paths in beam search. Beam scores are initialized with 0 for the first beam and a very low number (-1e9) for others to ensure the first token of each sequence is chosen from the first beam.
        """

        decoder_input_ids = torch.ones((batch_size * num_beams, 1), device=self.t5.device, dtype=torch.long)
        initial_decoder_input_ids = decoder_input_ids * self.t5.config.decoder_start_token_id

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9  # Set a low score for all but the first beam to ensure the first beam is selected initially


        #kind of flatten
        initial_beam_scores = beam_scores.view((batch_size * num_beams,))


        beam_idx_offset = torch.arange(batch_size, device=self.t5.device).repeat_interleave(num_beams) * num_beams

        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)


        return input_ids, attention_mask, initial_decoder_input_ids, initial_beam_scores, beam_idx_offset


    def beam_search_step(self, logits, decoder_input_ids, beam_scores, beam_idx_offset, batch_size, num_beams):
        """
        Adpated from huggingface's implementation
        https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

        Executes one step of beam search, calculating the next set of input IDs based on logits from a model.

        This function expands the current beam, calculates scores for all possible next tokens, selects the top tokens for each beam, and prepares the input IDs for the next iteration of the model. It utilizes logits output by the model to determine the most likely next tokens and updates the beam scores.

        Parameters:
        - logits (torch.Tensor): Logits returned from the model, shape (batch_size * num_beams, sequence_length, vocab_size).
        - decoder_input_ids (torch.Tensor): Current decoder input IDs, shape (batch_size * num_beams, current_sequence_length).
        - beam_scores (torch.Tensor): Current scores for each beam, shape (batch_size * num_beams,).
        - beam_idx_offset (torch.Tensor): Index offsets for each beam to handle batches correctly, shape (batch_size * num_beams,).
        - batch_size (int): Number of sequences being processed in a batch.
        - num_beams (int): Number of beams used in the beam search.

        Returns:
        - decoder_input_ids (torch.Tensor): Updated decoder input IDs after adding the next tokens, shape (batch_size * num_beams, current_sequence_length + 1).
        - beam_scores (torch.Tensor): Updated scores for each beam, shape (batch_size * num_beams,).

        The function selects the top `2 * num_beams` tokens from the logits based on their scores, reshapes and adjusts them based on the existing beam scores, and determines the next tokens to add to each beam path. The updated paths are then returned for use in the next iteration of the beam search.
        """

        assert batch_size * num_beams == logits.shape[0]


        vocab_size = logits.shape[-1]
        next_token_logits = logits[:, -1, :]

        next_token_scores = torch.log_softmax(next_token_logits, dim=-1)  # Calculate log softmax over the last dimension


        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)


        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)


        next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)


        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)

        # beam_idx_offset: beam_idx contains sequence indicies relative to each individual batch. We need to offset the indicies to retrieve the correct sequence in the corresponding batch
        # for example, when batch_size = 2, beam_size = 3, beam_idx_offset = [0, 0, 0, 3, 3, 3]
        decoder_input_ids = torch.cat([decoder_input_ids[beam_idx + beam_idx_offset, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        return decoder_input_ids, beam_scores
