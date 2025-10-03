import os
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
from genrec.models.layers import RQVAEModel
from genrec.utils import list_to_str
import pickle
import random
from collections import defaultdict, Counter
from collections import defaultdict, Counter
import copy

def _euclidean_misaligned(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Dim mismatch: {a.shape[0]} vs {b.shape[0]}")
    return float(np.linalg.norm(a - b))

#


class PctxTokenizer(AbstractTokenizer):
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
        super(PctxTokenizer, self).__init__(config, dataset)
        self.config = config

        self.accelerator = self.config['accelerator']

        self.num_clustered_interaction = 0

        self.user2id = dataset.user2id
        self.item2id = dataset.item2id
        self.id2item = dataset.id_mapping['id2item']

        self.run_mode=self.config['run_mode']
        if self.run_mode != 'train' and self.run_mode != 'test':
            raise ValueError('mode is out of range')


        # interactionKey2sidTokens
        # for more information, please refer to "interactionKey2sidTokens_sorted_by_item_id_before_merging_conflict.txt"
        if self.run_mode=='train':
            self.gene_sid_or_not=True
            sem_ids_path = os.path.join(
                dataset.cache_dir, 'processed',
                f'{os.path.basename(self.config["sent_emb_model"])}_{list_to_str(self.codebook_sizes, remove_blank=True)}.sem_ids'
            )

            if os.path.exists(sem_ids_path) and self.config['refresh_cluster_result'] == False:
                self.log('[[[ Do not need to generate sids. ]]]')
                self.log('[[[ Do not need to generate sids. ]]]')
                self.log('[[[ Do not need to generate sids. ]]]')
                self.gene_sid_or_not=False
                # Load or encode sentence embeddings
            else:

                self.interactionKey2sidTokens_sorted_by_item_id_before_merging_conflict = self._init_tokenizer(dataset)
        
    

                self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo, preview_dict = self.merge_interaction_and_embedding_dicts(
                    self.interactionKey2sidTokens_sorted_by_item_id_before_merging_conflict,
                    self.dict_2eles_interactionKey_To_clusterIndex_emb,
                    preview_dim=5
                )





                self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID, preview_itemID_dict = self.regroup_by_itemID(
                    self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo,
                    preview_dim=5
                )
                



                # if the sidToken has duplicated ones (not unique), raise error. 
                self.check_unique_sidtokens(self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID)



                # to check if the number of cluster centers is equal to that of sids, if not, raise errors.
                if self.run_mode == 'train':
                    self._checklength_nCluster_and_nSID(self.interactionKey2sidTokens_sorted_by_item_id_before_merging_conflict)

                # for more information, please refer to "itemID2sidTokenList_before_merging_conflict.txt"
                self.itemID2sidTokenList_before_merging_conflict = self._build_itemID2tokenList(
                    self.interactionKey2sidTokens_sorted_by_item_id_before_merging_conflict)

                self.itemID2sidTokenCountDict_before_merging_conflict = self.build_itemID2sidTokenCountDict(
                    self.itemID2sidTokenList_before_merging_conflict)
 
                # *********************start merge_conflict*********************************

                self.mergeConflictSearchingDict = self.get_merging_conflict_searching_dict(
                    self.itemID2sidTokenList_before_merging_conflict)
    
                self.interactionKey2sidTokens_sorted_by_item_id_after_merging_conflicts = self.replace_conflict_values(
                    self.mergeConflictSearchingDict, self.interactionKey2sidTokens_sorted_by_item_id_before_merging_conflict)
      
                self.itemID2sidTokenList_after_merging_conflicts = self._build_itemID2tokenList(
                    self.interactionKey2sidTokens_sorted_by_item_id_after_merging_conflicts)
        
                self.itemID2sidTokenCountDict_after_merging_conflicts = self.build_itemID2sidTokenCountDict(
                    self.itemID2sidTokenList_after_merging_conflicts)
           
                # To check when we use merging conflict, how many sids do we save.
                self.compare_cluster_centers_sum(self.itemID2sidTokenCountDict_before_merging_conflict,
                                                self.itemID2sidTokenCountDict_after_merging_conflicts)
                # *********************finish merge_conflict*********************************




                # [start] update the clusterIndex of the merge-conflict emb in the clustering map.


                self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID_updatedByMergeConflict, preview_updated = self.update_by_merge_conflict(
                    self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID,
                    self.mergeConflictSearchingDict,
                    preview_dim=5
                )

               
           


                # [end] update the clusterIndex of the merge-conflict emb in the clustering map.


                # *********************start merge_low_frequency_sid_strategy*********************************

                self.itemID2sidTokenCountDict_mergeLowFrequency, self.mergeLowFrequencySearchingDict = self.run_merge_all(
                    self.itemID2sidTokenCountDict_after_merging_conflicts)



                self.summarize_merge_result(
                    self.mergeLowFrequencySearchingDict,
                    self.itemID2sidTokenCountDict_after_merging_conflicts,
                    self.itemID2sidTokenCountDict_mergeLowFrequency
                )

                self.interactionKey2sidTokens_sorted_by_item_id_deepcopy = copy.deepcopy(
                    self.interactionKey2sidTokens_sorted_by_item_id_after_merging_conflicts)
                self.interactionKey2sidTokens_sorted_by_item_id_after_merge_low_frequency_sid = self.replace_conflict_values_for_merging_lowFrequencyStrategy(
                    self.mergeLowFrequencySearchingDict, self.interactionKey2sidTokens_sorted_by_item_id_deepcopy)
              
                self.itemID2sidTokenList_after_merge_low_frequency_sid = self._build_itemID2tokenList(
                    self.interactionKey2sidTokens_sorted_by_item_id_after_merge_low_frequency_sid)
          
                self.itemID2sidTokenCountDict_after_merge_low_frequency_sid = self.build_itemID2sidTokenCountDict(
                    self.itemID2sidTokenList_after_merge_low_frequency_sid)
          
       
                self.save_differences_between_sid_dicts(self.itemID2sidTokenCountDict_mergeLowFrequency,
                                                        self.itemID2sidTokenCountDict_after_merge_low_frequency_sid)

                self.check_sid_dicts(self.itemID2sidTokenCountDict_mergeLowFrequency,
                                    self.itemID2sidTokenCountDict_after_merge_low_frequency_sid)

                self.compare_cluster_centers_sum_after_merging_low_frequency(
                    self.itemID2sidTokenCountDict_before_merging_conflict,
                    self.itemID2sidTokenCountDict_after_merging_conflicts,
                    self.itemID2sidTokenCountDict_after_merge_low_frequency_sid)

                # *********************Finished merge_low_frequency_sid_strategy*********************************





                # [start] update the clusterIndex of the merge-infrequent emb in the clustering map.


                self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID_updatedByMergeConflict_updatedByMergeInfrequent, preview_updated_infrequent = self.update_by_merge_low_frequency(
                    self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID_updatedByMergeConflict,
                    self.mergeLowFrequencySearchingDict,
                    preview_dim=5
                )


       


                # [end] update the clusterIndex of the merge-infrequent emb in the clustering map.



                self.check_if_the_sid_num_in_sidTokenCountDict_is_equal_to_that_in_clusterIndex()





                # [final]*****************************************************
                # *****************************************************
                self.interactionKey2sidTokens_sorted_by_item_id = self.interactionKey2sidTokens_sorted_by_item_id_after_merge_low_frequency_sid


                self.interactionkey2sidTokens = dict(
                    sorted(self.interactionKey2sidTokens_sorted_by_item_id.items(), key=lambda x: self.parse_key(x[0])))
             

                self.update_the_interactionkey2sidTokens_by_val_and_test_set_for_cache_for_inference()




                # generate sem_ids
                if self.accelerator.is_main_process:
           
                    self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')

                with self.accelerator.main_process_first():
                    if not os.path.exists(sem_ids_path):
                        with open(sem_ids_path, "w") as f:
                            json.dump(self.interactionkey2sidTokens, f)
                        self.log(f" save interactionKey2sid to {sem_ids_path}")
            




            self.interactionkey2sidTokens = json.load(open(sem_ids_path, 'r'))

            for key, value in self.interactionkey2sidTokens.items():
                self.interactionkey2sidTokens[key] = tuple(value)


        elif self.run_mode=='test':


            sem_ids_path = os.path.join(
            self.config['test_file_dir'], self.config['dataset'], self.config['category'],
            f'{os.path.basename(self.config["sent_emb_model"])}_{list_to_str(self.codebook_sizes, remove_blank=True)}.sem_ids'
        )

            if not os.path.exists(sem_ids_path):
                raise ValueError('No .sid file for test mode')

            self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')


            self.interactionkey2sidTokens = json.load(open(sem_ids_path, 'r'))

         
            for key, value in self.interactionkey2sidTokens.items():
                self.interactionkey2sidTokens[key] = tuple(value)


   
   
    
        self.interactionkey2sidTokens_sorted_by_item_id = dict(
            sorted(self.interactionkey2sidTokens.items(), key=lambda x: self.parse_key(x[0])[1])
        )



        self.itemID2sidTokenList = self._build_itemID2tokenList(
            self.interactionkey2sidTokens_sorted_by_item_id)


        self.itemID2sidTokenCountDict = self.build_itemID2sidTokenCountDict(
            self.itemID2sidTokenList)
  


        # *****************************************************


        # is a mapping from sid(str) to item item_id, using for gathering the decoing tree during inference.
        self.sidStr2itemIDInt = self._build_and_save_sidStr2itemIDInt(self.interactionkey2sidTokens)

        # *****************************************************#
        # [final]*****************************************************

        # self.base_user_tokens= 256*4+1 = 1025
        self.base_user_token = sum(self.codebook_sizes) + 1
        # config: self.config['n_user_tokens']=1
        self.n_user_tokens = self.config['n_user_tokens']
        # so self.eos_token=1025+1=1026

        self.eos_token = self.base_user_token + self.n_user_tokens

      
        self.n_inference_ensemble = self.config['n_inference_ensemble']
    

        self.collate_fn = {
            'train': self.collate_fn_train,
            'val': self.collate_fn_val,
            'test': self.collate_fn_test,
        }
        if self.accelerator.is_main_process:
            self.log('Finish the init of Pctx')



    def update_the_interactionkey2sidTokens_by_val_and_test_set_for_cache_for_inference(self):
        diff_filename_test = f"{self.config['diff_emb_name_test']}"
        diff_emb_path_test = os.path.join(
            self.config['dict_dir'], 
            self.config['dataset'],
            self.config['category'],
            diff_filename_test
        )

        self.augment_tokenizer_interaction_sids_with_diff_emb(
            
                diff_emb_path_test=diff_emb_path_test,
                itemID_to_clusters=self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID_updatedByMergeConflict_updatedByMergeInfrequent,
            
            )



    def augment_tokenizer_interaction_sids_with_diff_emb(self,
     
        diff_emb_path_test: str, 
        itemID_to_clusters: dict,
       
        skipped_items_log_path="skipped_items_log.txt" 
    ):


    
        """
        Augments the tokenizer's interaction SID tokens by mapping new embeddings to existing interaction keys.
        This function processes each interaction key, finds the closest cluster based on the new embeddings, 
        and updates the SID tokens accordingly.

        Parameters:
            diff_emb_path_test (str): Path to the file containing new embeddings (in pickle format).
            itemID_to_clusters (dict): A dictionary where the key is itemID and the value is a list of clusters for that item.
            skipped_items_log_path (str): Path to the log file to record skipped items.

        Returns:
            None
        """

        inter2sid = self.return_interactionkey2sidTokens()  # dict: interactionKey -> sid token (tuple)
        #
 
        with open(diff_emb_path_test, "rb") as f:
            interactionKey2emb = pickle.load(f)

        added = 0
        skipped_exists = 0
        skipped_no_item = 0
        processed = 0

        skipped_item_count = {}  

       
   

        for interactionKey, emb1 in interactionKey2emb.items():
            processed += 1
            if interactionKey in inter2sid:
                skipped_exists += 1
                continue

            
            try:
                itemID = interactionKey.split("-")[1]
            except Exception as e:
                self.log(f"[augment] parse interactionKey failed: {interactionKey} ({e})")
                continue

        
            clusters = itemID_to_clusters.get(itemID)
            if clusters is None:
                skipped_no_item += 1
                


                skipped_item_count[itemID] = skipped_item_count.get(itemID, 0) + 1
                continue

        
            best_idx = None
            best_dist = float("inf")
            emb1 = np.asarray(emb1, dtype=np.float32)

            for clusterIndex, info in clusters.items():
                if "emb" not in info:
                
                    continue
                dist = _euclidean_misaligned(emb1, info["emb"])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = clusterIndex

            if best_idx is None:

            
            
                skipped_item_count[itemID] = skipped_item_count.get(itemID, 0) + 1
                continue

        
            final_idx = best_idx
            hop_guard = 0
            while True:
                tend_to = clusters[final_idx].get("TendTo", None)
                if tend_to is None or tend_to == "None":
                    break
                if tend_to not in clusters:
                
                    self.log(f"[augment] itemID={itemID} TendTo points to missing: {final_idx} -> {tend_to}, we use {final_idx}")
                    break
                final_idx = tend_to
                hop_guard += 1
                if hop_guard > 1024:  # 
                    self.log(f"[augment] itemID={itemID} TendTo may conduct a circle, stop at {final_idx}")
                    break

        
            final_sid = clusters[final_idx]["SIDToken"]  # tuple

    
            inter2sid[interactionKey] = tuple(final_sid)
            added += 1

         
       
        inter2sid_sorted = dict(
            sorted(inter2sid.items(), key=lambda x: self.parse_key(x[0]))
        )
        self.interactionkey2sidTokens = inter2sid_sorted

      
        self.log(f"[augment] processed={processed}, added={added}, "
                        f"skipped_exists={skipped_exists}, Number_Of_No_matched_items_when_tokenizing_Val_And_Test={skipped_no_item}")
  


    def return_clusterIndex(self):
        return self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID_updatedByMergeConflict_updatedByMergeInfrequent


    def return_interactionkey2sidTokens(self):
        return self.interactionkey2sidTokens



    def check_if_the_sid_num_in_sidTokenCountDict_is_equal_to_that_in_clusterIndex(self):
        """
        Check:
        Whether self.itemID2sidTokenCountDict_after_merge_low_frequency_sid[itemID]['cluster_centers']
        is equal to the number of TendTo=None entries in
        self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID_updatedByMergeConflict[itemID].

        """
        for itemID, info in self.itemID2sidTokenCountDict_after_merge_low_frequency_sid.items():
            expected_centers = info["cluster_centers"]

            if itemID not in self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID_updatedByMergeConflict_updatedByMergeInfrequent:
                raise KeyError(f"[ERROR] itemID={itemID} does not exist in the structure of TendTo ")

            clusters = self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID_updatedByMergeConflict_updatedByMergeInfrequent[itemID]
            actual_none_count = sum(
                1 for record in clusters.values()
                if record["TendTo"] is None or record["TendTo"] == "None"
            )

            if expected_centers != actual_none_count:
                raise ValueError(
                    f"[ERROR] itemID={itemID}: cluster_centers={expected_centers} "
                    f"≠ TendTo=None count={actual_none_count}"
                )
        if self.accelerator.is_main_process:
            self.log("[CHECK PASSED] The number of all the cluster_centers of itemID is equal to the number of TendTo=None.")






    def update_by_merge_low_frequency(
        self,itemID_clusterIndex_dict_updatedByConflict,
        mergeLowFrequencySearchingDict,
        preview_dim=5
    ):
        """

        Update TendTo based on mergeLowFrequencySearchingDict.

        Args:

            itemID_clusterIndex_dict_updatedByConflict (dict):
            self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID_updatedByMergeConflict

            mergeLowFrequencySearchingDict (dict): { itemID: [ {origin, new}, ... ] }

            preview_dim (int): The dimension for truncating embeddings, used for generating preview files.

        Returns:

            updated (dict): The updated structure.

            preview (dict): A preview version where embeddings are replaced by their first preview_dim elements.




        """
        updated = copy.deepcopy(itemID_clusterIndex_dict_updatedByConflict)

        for itemID, conflict_list in mergeLowFrequencySearchingDict.items():
            if itemID not in updated:
                raise ValueError(f"[ERROR] itemID={itemID} does not show up in dict！")

            clusters = updated[itemID]
      
            sid_to_cluster = {v["SIDToken"]: idx for idx, v in clusters.items()}

            for conflict in conflict_list:
                origin = conflict["origin"]
                new = conflict["new"]

               
                if origin not in sid_to_cluster:
                    raise ValueError(f"[ERROR] itemID={itemID} lacks origin SIDToken={origin}")
                if new not in sid_to_cluster:
                    raise ValueError(f"[ERROR] itemID={itemID} lacks new SIDToken={new}")

                origin_clusterIndex = sid_to_cluster[origin]
                new_clusterIndex = sid_to_cluster[new]

               
                if clusters[origin_clusterIndex]["TendTo"] != "None":
                    raise ValueError(
                        f"[ERROR] itemID={itemID} origin={origin} has already possessed TendTo={clusters[origin_clusterIndex]['TendTo']}"
                    )

               
                if clusters[new_clusterIndex]["TendTo"] == "None":
                    clusters[origin_clusterIndex]["TendTo"] = new_clusterIndex
                else:
                  
                    clusters[origin_clusterIndex]["TendTo"] = clusters[new_clusterIndex]["TendTo"]

     
        preview = copy.deepcopy(updated)
        for itemID, clusters in preview.items():
            for clusterIndex, v in clusters.items():
                if "emb" in v:
                    v["emb_first_five"] = np.array(v["emb"][:preview_dim])
                    del v["emb"]

        return updated, preview







    def update_by_merge_conflict(
        self,itemID_clusterIndex_dict,
        mergeConflictSearchingDict,
        preview_dim=5
    ):
        """
        Update TendTo based on mergeConflictSearchingDict
        
        Args:

            itemID_clusterIndex_dict (dict): self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID

            mergeConflictSearchingDict (dict): { itemID: [ {origin, new}, ... ] }

            preview_dim (int): The dimension for truncating embeddings, used for generating preview files.

        Returns:

            updated (dict): The updated structure.

            preview (dict): A preview version where embeddings are replaced by their first preview_dim elements.
        """
        updated = copy.deepcopy(itemID_clusterIndex_dict)

        for itemID, conflict_list in mergeConflictSearchingDict.items():
            if itemID not in updated:
                raise ValueError(f"[ERROR] itemID={itemID} does not show up in itemID_clusterIndex_dict!")

            clusters = updated[itemID]

         
            sidtokens_available = {v["SIDToken"]: idx for idx, v in clusters.items()}

            for conflict in conflict_list:
                origin = conflict["origin"]
                new = conflict["new"]

          
                if origin not in sidtokens_available:
                    raise ValueError(f"[ERROR] itemID={itemID} lacks origin SIDToken={origin}")
                if new not in sidtokens_available:
                    raise ValueError(f"[ERROR] itemID={itemID} lacks new SIDToken={new}")

                
                origin_clusterIndex = sidtokens_available[origin]
                new_clusterIndex = sidtokens_available[new]

              
                clusters[origin_clusterIndex]["TendTo"] = new_clusterIndex

        
        preview = copy.deepcopy(updated)
        for itemID, clusters in preview.items():
            for clusterIndex, v in clusters.items():
                if "emb" in v:
                    v["emb_first_five"] = np.array(v["emb"][:preview_dim])
                    del v["emb"]

        return updated, preview






    def check_unique_sidtokens(self,itemID_dict):
        """
       Check if all SIDTokens in itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID are unique.
        If any duplicates are found, an error will be raised.
        """
        seen = {}
        for itemID, clusters in itemID_dict.items():
            for clusterIndex, value in clusters.items():
                sidtoken = value["SIDToken"]
                if sidtoken in seen:
                    raise ValueError(
                        f"[ERROR] SIDToken {sidtoken} repeated！"
                        f"\nFirst in itemID={seen[sidtoken][0]}, clusterIndex={seen[sidtoken][1]}"
                        f"\nSecond in itemID={itemID}, clusterIndex={clusterIndex}"
                    )
                seen[sidtoken] = (itemID, clusterIndex)
        if self.accelerator.is_main_process:
            self.log("[CHECK PASSED] All SIDToken are unique.")





    def regroup_by_itemID(self,itemID_To_clusterIndex_dict, preview_dim=5):
        """
        Restructure itemID_To_clusterIndex_To_SIDToken_Emb_TendTo from the interactionKey perspective  
        to the itemID perspective, and check whether duplicate clusterIndex keys have completely identical values.  

        Args:
            itemID_To_clusterIndex_dict (dict): The original dictionary, where the key is interactionKey (userID-itemID-preordered).
            preview_dim (int): The dimension length for truncating embeddings in the preview.

        Returns:
            regrouped (dict): A new structure regrouped by itemID.
            preview (dict): A preview version with embeddings replaced by their first preview_dim elements.

        """
        regrouped = {}

        for interactionKey, cluster_dict in itemID_To_clusterIndex_dict.items():
            #  interactionKey = userID-itemID-preordered
            parts = interactionKey.split("-")
            if len(parts) < 2:
                raise ValueError(f"[ERROR] interactionKey {interactionKey} is wrong with format itemID")
            itemID = parts[1]  

            if itemID not in regrouped:
                regrouped[itemID] = {}

            for clusterIndex, value in cluster_dict.items():
                if clusterIndex in regrouped[itemID]:
                    
                    existing_value = regrouped[itemID][clusterIndex]
                    if not np.array_equal(existing_value["SIDToken"], value["SIDToken"]) or \
                    not np.array_equal(existing_value["emb"], value["emb"]) or \
                    existing_value["TendTo"] != value["TendTo"]:
                        raise ValueError(
                            f"[ERROR] itemID={itemID} repeat clusterIndex={clusterIndex} (interactionKey={interactionKey}) "
                            f"but the value is not the same！"
                        )
                  
                else:
                   
                    regrouped[itemID][clusterIndex] = {
                        "SIDToken": value["SIDToken"],
                        "emb": value["emb"],
                        "TendTo": value["TendTo"]
                    }

        
        preview = copy.deepcopy(regrouped)
        for itemID, clusters in preview.items():
            for clusterIndex, v in clusters.items():
                if "emb" in v:
                    v["emb_first_five"] = np.array(v["emb"][:preview_dim])
                    del v["emb"]

        return regrouped, preview





    def merge_interaction_and_embedding_dicts(self, interaction_dict, emb_dict, preview_dim=5):
        """
        merge interactionKey -> sidtoken 和 interactionKey -> {clusterIndex, emb}
        output itemID_To_clusterIndex_To_SIDToken_Emb_TendTo
        """
        # 1. check
        keys1 = list(interaction_dict.keys())
        keys2 = list(emb_dict.keys())
        if keys1 != keys2:
            raise ValueError("[ERROR] the interactionKey in two dicts are not equal totally！")

        result = {}

     
        for key in keys1:
            sidtoken = interaction_dict[key]  # tuple
            clusterIndex = emb_dict[key]["clusterIndex"]
            emb = emb_dict[key]["emb"]  # np.array 256 d

            result[key] = {
                clusterIndex: {
                    "SIDToken": sidtoken,
                    "emb": emb,
                    "TendTo": "None"
                }
            }

     
        preview_result = copy.deepcopy(result)
        for key, cluster_dict in preview_result.items():
            for clusterIndex, v in cluster_dict.items():
                if "emb" in v:
                    v["emb_first_five"] = np.array(v["emb"][:preview_dim])
                    del v["emb"]

        return result, preview_result








    # ***************************The function for merge_low_frequency_sid_strategy********************************
    def save_differences_between_sid_dicts(self, dict1, dict2, output_path='differences_between_sid_dicts.txt'):
  
        differences = []

        for item_id in dict1:
            if item_id not in dict2:
                differences.append(f" ItemID {item_id} missing in dict2.\n")
                continue

            sids1 = {k for k in dict1[item_id] if isinstance(k, tuple)}
            sids2 = {k for k in dict2[item_id] if isinstance(k, tuple)}

            if sids1 != sids2:
                diff = []
                diff.append(f" ItemID: {item_id}")
                diff.append(f"  - SIDs in dict1 (mergeLowFreq): {sorted(sids1)}")
                diff.append(f"  - SIDs in dict2 (after_merge):  {sorted(sids2)}\n")
                differences.append('\n'.join(diff))

       
        if differences:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(differences))

            if self.accelerator.is_main_process:
                self.log(f"[CHECK] Saved {len(differences)} mismatched itemIDs to: {output_path}")
        else:
            if self.accelerator.is_main_process:
                self.log("[CHECK PASSED] No mismatched itemIDs found in merge_infrequent strategy")

    # -------------------------------
    # Merge_low_frequency_sid----Part1
    # Calculate the similarity of two sids
    # -------------------------------
 
    def calculate_sid_similarity(self, item_id, sid1, sid2):
     
        clusters = self.itemID_To_clusterIndex_To_SIDToken_Emb_TendTo_sorted_by_itemID_updatedByMergeConflict.get(item_id)
        if clusters is None:
            raise KeyError(f"[ERROR] item_id={item_id} does not exist")

        def resolve_emb(target_sid):

            for clusterIndex, record in clusters.items():
                if record["SIDToken"] == target_sid:
                    
                    if record["TendTo"] is None or record["TendTo"] == "None":
                        return record["emb"]
                  
                    if record["TendTo"] not in clusters:
                        raise KeyError(f"[ERROR] TendTo={record['TendTo']} is not in item_id={item_id} 's clusters ")
                    return clusters[record["TendTo"]]["emb"]
            raise KeyError(f"[ERROR] SIDToken={target_sid} is not in item_id={item_id} 's clusters")

        emb1 = np.array(resolve_emb(sid1))
        emb2 = np.array(resolve_emb(sid2))

        return float(np.linalg.norm(emb1 - emb2))

    # -------------------------------
    # Merge_low_frequency_sid----Part2
    # To judge if we should merge the sid
    # -------------------------------

    # itemid: sid1: 10 sid2: :40
    # 0.2
    def should_merge_sid(self, sid, count, total, cluster_centers, frequency_threshold=0):
        if (1.0 * count / total) < frequency_threshold:
            return True

        return False

    # -------------------------------
    # Merge_low_frequency_sid----Part3
    # Find the most similar sid in this sample (except itself)
    # -------------------------------
    def find_most_similar_sid(self, item_id, target_sid, sid_list):
        min_sim = float('inf')
        most_similar = None
        for sid in sid_list:
            if sid == target_sid:
                continue  # this line of code makes sure the sid won't calculate similarity with itself
            sim = self.calculate_sid_similarity(item_id,target_sid, sid)
            if sim < min_sim:
                min_sim = sim
                most_similar = sid
        return most_similar

    # -------------------------------
    # Merge_low_frequency_sid----Part4
    # merge function
    # -------------------------------
    def merge_sids_for_item(self, item_id, entry, merge_dict):
        entry = copy.deepcopy(entry)
        while True:
            sids = [sid for sid in entry if isinstance(sid, tuple)]
            total = sum(entry[sid] for sid in sids)
            cluster_centers = entry['cluster_centers']
            marked_sids = []
            # sid, count, total, cluster_centers
            for sid in sids:
                if self.should_merge_sid(sid=sid, count=entry[sid], total=total, cluster_centers=cluster_centers,frequency_threshold=self.config['frequency_threshold']):
                    marked_sids.append(sid)

            if not marked_sids:
                break

            for sid in marked_sids:
                target_sid = self.find_most_similar_sid(item_id,sid, [s for s in entry if isinstance(s, tuple)])
                if not target_sid:
                    continue
                # update the sample, merge in it
                entry[target_sid] += entry[sid]
                del entry[sid]
                entry['cluster_centers'] -= 1
                merge_dict[item_id].append({'origin': sid, 'new': target_sid})

        return entry

    # -------------------------------
    # # Merge_low_frequency_sid----Part5
    # Operate on the whole dict
    # -------------------------------
    def run_merge_all(self, data_dict):
        merge_dict = defaultdict(list)
        new_data = copy.deepcopy(data_dict)

        for item_id, entry in data_dict.items():
            if entry['cluster_centers'] >= 2:
                new_entry = self.merge_sids_for_item(item_id, entry, merge_dict)
                new_data[item_id] = new_entry

        #
        merge_dict = {str(k): v for k, v in merge_dict.items()}

        return new_data, merge_dict

    # -------------------------------
    # # Merge_low_frequency_sid----Part5
    # statics information
    # -------------------------------
    def summarize_merge_result(self, merge_dict, original_dict, merged_dict):
        merge_stats = []

        for str_item_id, merge_list in merge_dict.items():
            item_id = str_item_id  #
            original_cc = original_dict[item_id]['cluster_centers']
            new_cc = merged_dict[item_id]['cluster_centers']
            total_tokens = sum(count for key, count in merged_dict[item_id].items() if isinstance(key, tuple))
            merge_stats.append({
                'item_id': item_id,
                'merge_times': len(merge_list),
                'original_cluster_centers': original_cc,
                'new_cluster_centers': new_cc,
                'total_token_count': total_tokens
            })

        merge_stats.sort(key=lambda x: x['merge_times'], reverse=True)

        total_before = sum(v['cluster_centers'] for v in original_dict.values())
        total_after = sum(v['cluster_centers'] for v in merged_dict.values())
        total_removed = total_before - total_after



    def check_sid_dicts(self, dict1, dict2):
        def compute_stats(d):
            total_cluster_centers = sum(entry['cluster_centers'] for entry in d.values())
            total_token_count = sum(
                count for entry in d.values()
                for key, count in entry.items()
                if isinstance(key, tuple)
            )
            return total_cluster_centers, total_token_count

        cc1, tokens1 = compute_stats(dict1)
        cc2, tokens2 = compute_stats(dict2)

        are_equal = (cc1 == cc2) and (tokens1 == tokens2)
       
        #self.log(f"Dict1 - cluster_centers sum: {cc1}, sid token count: {tokens1}")
        #self.log(f"Dict2 - cluster_centers sum: {cc2}, sid token count: {tokens2}")

        if are_equal == True:
            self.log('[CHECK PASSED] Merge infrequent pass the check')
        else:
            raise ValueError('Merge_infrequent_strategy FAIL to pass the check')

        return are_equal

    # **********************The function for merge_low_frequency_sid_strategy********************************

    def compare_cluster_centers_sum(self, dict1, dict2):
        """
        Compare the total sum of 'cluster_centers' values in two dictionaries,
        and calculate what percentage the second sum is of the first.

        Args:
            dict1 (dict): The first dictionary (e.g., itemID2sidTokenCountDict_before_merging_conflict)
            dict2 (dict): The second dictionary (e.g., itemID2sidTokenCountDict)

        Returns:
            tuple: (sum of dict1, sum of dict2, percentage of dict2 relative to dict1)
        """

        def sum_cluster_centers(d):
            # Sum the 'cluster_centers' value for each entry
            return sum(item.get('cluster_centers', 0) for item in d.values())

        sum1 = sum_cluster_centers(dict1)
        sum2 = sum_cluster_centers(dict2)
        percent = (sum2 / sum1 * 100) if sum1 != 0 else float('inf')  # Avoid division by zero
        if self.accelerator.is_main_process:
            self.log(f"The original number of sids is {sum1}")
            self.log(f'When we use merging conflict, the number of sids is {sum2}')
            self.log(f'We use only {percent:.2f} % of the original sids thanks to merging conflict operations.')

    def compare_cluster_centers_sum_after_merging_low_frequency(self, dict_before_merge, dict_after_mergeConflict,
                                                                dict_after_mergeLowFre):
        """
        Compare the total sum of 'cluster_centers' values in two dictionaries,
        and calculate what percentage the second sum is of the first.

        Args:
            dict1 (dict): The first dictionary (e.g., itemID2sidTokenCountDict_before_merging_conflict)
            dict2 (dict): The second dictionary (e.g., itemID2sidTokenCountDict)

        Returns:
            tuple: (sum of dict1, sum of dict2, percentage of dict2 relative to dict1)
        """

        def sum_cluster_centers(d):
            # Sum the 'cluster_centers' value for each entry
            return sum(item.get('cluster_centers', 0) for item in d.values())

        sum0 = len(self.item2id)

        sum_inter = len(self.interactionKey2sidTokens_sorted_by_item_id_before_merging_conflict)

        sum1 = sum_cluster_centers(dict_before_merge)
        sum2 = sum_cluster_centers(dict_after_mergeConflict)
        sum3 = sum_cluster_centers(dict_after_mergeLowFre)
        percent1 = (sum1 / sum0 * 100) if sum0 != 0 else float('inf')  # Avoid division by zero
        percent2 = (sum2 / sum0 * 100) if sum0 != 0 else float('inf')  # Avoid division by zero
        percent3 = (sum3 / sum0 * 100) if sum0 != 0 else float('inf')  # Avoid division by zero

        percent4 = (sum2 / sum1 * 100) if sum1 != 0 else float('inf')  # Avoid division by zero
        percent5 = (sum3 / sum1 * 100) if sum1 != 0 else float('inf')  # Avoid division by zero

        percent6 = (sum3 / sum2 * 100) if sum2 != 0 else float('inf')  # Avoid division by zero
        if self.accelerator.is_main_process:
            self.log(f"[statics1-origin] The original number of item is {sum0}")
            self.log(f"[statics2-inter] The original number of item is {sum_inter}")
            self.log(
                f"[statics3-cluster] The original number of sids after clustering is {sum1} ({percent1} % of the origin)   ")
            self.log(
                f'[statics4-mergeDuplicated] When we use merging duplicated, the number of sids is {sum2} ({percent2} % of the origin) , the reduced number compared to cluster: {sum1 - sum2}')
            self.log(
                f'[statics5-mergeInfrequent] When we use merging infrequent strategy, the number of sids is {sum3} ({percent3} % of the origin), the reduced number compared to merge_duplicated: {sum2 - sum3}')

            self.log(
                f'We use only {percent4:.2f} % of the original sids (after clustering) thanks to merging duplicated operations.')
            self.log(
                f'We use only {percent5:.2f} % of the original sids (after clustering) thanks to merging infrequent operations.')
            self.log(
                f'We use only {percent6:.2f} % of the after_merging duplicated sids thanks to merging infrequent operations.')

            self.log(
                f'[Summary] We have {sum0} items, with a static tokenizer, we will use {sum0} sid to represent {sum0} items, i.e. one-to one.'
                f'\n Every item exhibits personalized interactive pattern at its every interaction, and we have totally {sum_inter} interactions'
                f'\n it means that we will use at most {sum_inter} sids to represent {sum0} items  ({(sum_inter / sum0) * 100} %) '
                f'\n But with a cluster strategy, we will employ {sum1} sids to represent {sum0} items ({percent1} %, the reduced number compared to everyInter-a-sid: {sum_inter - sum1})'
                f'\n After merge_duplicated, we will apply {sum2} sids to represent {sum0} items ({percent2} %, the reduced number compared to cluster: {sum1 - sum2})'
                f'\n After merge_infrequent, we will utilize {sum3} sids to represent {sum0} items ({percent3} %, the reduced number compared to merge_duplicated: {sum2 - sum3})')

    def get_merging_conflict_searching_dict(self, itemID2sidTokenList):
        mergeConflictSearchingDict = {}
        removed_count = 0
        unmatched_count = 0

 
        # rq_n_codebooks: 3
        # rq_codebook_size: 256
        self.conflict_starts = self.config['rq_n_codebooks'] * self.config['rq_codebook_size'] + 2
   
        for item_key, tuple_list in itemID2sidTokenList.items():
            # Step 1: Deduplication + sort by the conflict number in ascending order
            deduped_sorted = sorted(set(tuple_list), key=lambda x: x[3])

            i = 0
            while i < len(deduped_sorted):
                tuplej = deduped_sorted[i]
          
                if tuplej[3] != self.conflict_starts:
               
                    found_match = False
                    for k in range(i):
                        tuplek = deduped_sorted[k]
                        if tuplej[:3] == tuplek[:3]:
                            # this item may have many conflicts that need to be merged
                            mergeConflictSearchingDict.setdefault(item_key, []).append({
                                'origin': tuplej,
                                'new': tuplek
                            })
                            # delete tuplej, cause we have already merged it into tuplek
                            deduped_sorted.pop(i)
                            removed_count += 1
                            found_match = True
                            break
                    if not found_match:
                        unmatched_count += 1
                        i += 1
                else:
                    i += 1

        # to check if all the tuple in mergeConflictSearchingDict have their conflict number be 770
        invalid_news = []
        for vlist in mergeConflictSearchingDict.values():
            for d in vlist:
      
                if d['new'][3] != self.conflict_starts:
                   
                    invalid_news.append(d['new'])

        # the number of merging
        total_conflict_count = sum(len(vlist) for vlist in mergeConflictSearchingDict.values())
        # statics
        if self.accelerator.is_main_process:
            self.log(f"the number of samples in mergeConflictSearchingDict : {len(mergeConflictSearchingDict)}")
            self.log(f"all the origin-new pairs in mergeConflictSearchingDict: {total_conflict_count}")
            #self.log(f"Merged sids: {removed_count}")
            self.log(f"The number of shared former 3 sid in different items: {unmatched_count}")
            self.log(f"The number of merged at least to 771: {len(invalid_news)}")

        # To check if the operations are right
        if total_conflict_count != removed_count:
            raise ValueError(f"wrong, {total_conflict_count} is not equal to {removed_count}.")

  
        return mergeConflictSearchingDict

    def replace_conflict_values(self, mergeConflictSearchingDict, interaction_dict):
        replaced_count = 0
        new_dict = {}

        for key, value in interaction_dict.items():
            parts = key.split("-")
            if len(parts) != 3:
                raise ValueError(f"[invalid]：{key} should in format: A-B-C")

            mid_key = parts[1]  # get the B part, i.e., itemID
            conflict_entries = mergeConflictSearchingDict.get(mid_key, [])

            # To find if there are any origin matching
            replaced = False
            for entry in conflict_entries:
                if entry['origin'] == value:
                    new_dict[key] = entry['new']
                    replaced_count += 1
                    replaced = True
                    break

            if not replaced:
                new_dict[key] = value  # there is no this key in mergeConflictSearchingDict, do nothing
        if self.accelerator.is_main_process:
            self.log(
                f" The total merge times in interactionKey2sidTokens_sorted_by_item_id_before_merging_duplicated: {replaced_count}")
        return new_dict

    def replace_conflict_values_for_merging_lowFrequencyStrategy(self, mergeConflictSearchingDict, interaction_dict):
        replaced_count = 0

        for key, value in interaction_dict.items():
            parts = key.split("-")
            if len(parts) != 3:
                raise ValueError(f"[invalid]：{key} should in format: A-B-C")

            mid_key = parts[1]  # get the B part, i.e., itemID
            conflict_entries = mergeConflictSearchingDict.get(mid_key, [])

            # To find if there are any origin matching
            replaced = False
            for entry in conflict_entries:
                if entry['origin'] == value:
                    interaction_dict[key] = entry['new']
                    value = entry['new']
                    replaced_count += 1
                    replaced = True

        if self.accelerator.is_main_process:
            self.log(
                f" The total merge times in interactionKey2sidTokens_sorted_by_item_id_before_merging_duplicated: {replaced_count}")
        return interaction_dict

    def build_itemID2sidTokenCountDict(self, itemID2sidTokenList):
        """


        input：
            itemID2sidTokenList: { Bi: [token_tuple1, token_tuple2, ...] }

        output：
            itemID2tokenCountDict: { Bi: { token_tuple: count, ... } }
        """
        itemID2tokenCountDict = dict()

        for itemID, tokenList in itemID2sidTokenList.items():
            counter = Counter(tokenList)
            token_count_dict = dict(counter)
            cluster_count = len(token_count_dict)

            token_count_dict_with_center = {"cluster_centers": cluster_count}
            token_count_dict_with_center.update(token_count_dict)

            itemID2tokenCountDict[itemID] = token_count_dict_with_center

        return itemID2tokenCountDict

    def _checklength_nCluster_and_nSID(self, interactionKey2sidTokens_sorted_by_item_id):
        # to check if the number of cluster centers is equal to that of sids, if not, raise errors.
        n_sid = len(set(interactionKey2sidTokens_sorted_by_item_id.values()))

        setClusterEmb_path_ = os.path.join(self.config['dict_dir'], self.config['dataset'], self.config['category'],
                                           self.config['cluster_file2_SetClusterIndexEmb'])

        with open(f"{setClusterEmb_path_}", 'rb') as f:
            self.SetClusterEmb_ThreeEle_clustered_sorted_data_ = pickle.load(f)

        n_cluster = len(self.SetClusterEmb_ThreeEle_clustered_sorted_data_)

        if n_sid == n_cluster:
            self.log('[CHECK PASSED]The number of clusters is equal to the number of sids')
        else:
            raise ValueError(
                f'The number of clusters should be equal to the number of sids, now n_cluster is {n_cluster}, and n_sid is {n_sid}')

    def _statics_to_txt(self, file, filename):
        if self.accelerator.is_main_process:
            # Generate statistical information for the specified dictionary and save it to a text file.
            output_path = f"{filename}.txt"
            filename = 'self.' + filename
            with open(output_path, "w") as f:
               
                if not file:
                    f.write("The dictionary is empty. No key-value pairs to display.\n")
                    return
               
                f.write(f"type({filename}): {type(file)}\n")
                f.write(f"len({filename}): {len(file)}\n\n")

                first_key = next(iter(file))
                first_value = file[first_key]
                f.write("Example key-value pair:\n")
                f.write(f"{first_key} (key type: {type(first_key)}): {first_value} (value type: {type(first_value)})\n")

                f.write("\nAll key-value pairs:\n")
                for k, v in file.items():
                    f.write(f"{k}: {v}\n")
            if self.accelerator.is_main_process:
                self.log(f"Statics Information saved to {output_path}")






    def _build_and_save_sidStr2itemIDInt(self, interactionkey2sidTokens):


        sidStr2itemIDInt = {}
        for item_key, token_tuple in interactionkey2sidTokens.items():
            token_key = str(list(token_tuple))

            middle_b = int(item_key.split("-")[1])
            sidStr2itemIDInt[token_key] = middle_b

        pkl_path = os.path.join(self.config['dict_dir'], self.config['dataset'], self.config['category'],
                                self.config['sidStr2itemIDInt'])

        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

        with self.accelerator.main_process_first():
            # if the sidStr2itemIDInt exists, we still need to generate a new one, for test mode.
            if self.run_mode=='train':
                if self.gene_sid_or_not==True:
                    with open(pkl_path, "wb") as f:
                        pickle.dump(sidStr2itemIDInt, f)
                    self.log(f" save sidStr2itemIDInt to {pkl_path}")
                else:
                    if not os.path.exists(pkl_path):
                        with open(pkl_path, "wb") as f:
                            pickle.dump(sidStr2itemIDInt, f)
                        self.log(f" save sidStr2itemIDInt to {pkl_path}")
            else: #test
                path_test_model_sidStr2itemIDInt = os.path.join(
                    self.config['test_file_dir'], self.config['dataset'], self.config['category'],
                    self.config['sidStr2itemIDInt']
                )
                with open(path_test_model_sidStr2itemIDInt, "wb") as f:
                    pickle.dump(sidStr2itemIDInt, f)
                self.log(f"[Test mode] save test_model_sidStr2itemIDInt to {path_test_model_sidStr2itemIDInt}")

        return sidStr2itemIDInt

    def collate_fn_train(self, batch):

        input_ids = []
        attention_mask = []
        labels = []

        for data in batch:
            # user_id
            user = str(data['user'].item())

            # item_seq is a int list, consisting of itemIDs.
            # item_seq = data['item_seq']
            item_seq = [int(i) for i in data['item_seq']]

           
            if self.config['augmentation_probability'] == 0:
                ids, mask, lbl = self._tokenize_once_accurate(user, item_seq)
            elif 0 < self.config['augmentation_probability'] <1 :
                ids, mask, lbl = self._tokenize_once_accAndRand(user, item_seq)
            elif self.config['augmentation_probability'] == 1:
                ids, mask, lbl = self._tokenize_once_random(user, item_seq)
            else:
                raise ValueError("config ['augmentation_probability'] must be at [0,1]!!!!!!!")

            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lbl)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

    def collate_fn_val(self, batch):

        input_ids = []
        attention_mask = []
        labels = []

        for data in batch:
            user = str(data['user'].item())
            # item_seq = data['item_seq']
            item_seq = [int(i) for i in data['item_seq']]

            if self.config['augmentation_probability'] == 0:
                ids, mask, lbl = self._tokenize_once_accurate(user, item_seq)
            elif 0 <self.config['augmentation_probability'] < 1:
                ids, mask, lbl = self._tokenize_once_accAndRand(user, item_seq)
            elif self.config['augmentation_probability'] == 1:
                ids, mask, lbl = self._tokenize_once_random(user, item_seq)
            else:
                raise ValueError("config ['augmentation_probability'] must be at [0,1]!!!!!!!")

            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lbl)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

    def collate_fn_test(self, batch):

        input_ids = []
        attention_mask = []
        labels = []

        for data in batch:
            user = str(data['user'].item())
            # item_seq = data['item_seq']
            item_seq = [int(i) for i in data['item_seq']]

           
            if self.config['augmentation_probability'] == 0:
                ids, mask, lbl = self._tokenize_once_accurate(user, item_seq)
                input_ids.append(ids)
                attention_mask.append(mask)
                labels.append(lbl)
            elif 0<self.config['augmentation_probability'] <1:
                if self.n_inference_ensemble == -1:
                    ids, mask, lbl = self._tokenize_once_accAndRand(user, item_seq)
                    input_ids.append(ids)
                    attention_mask.append(mask)
                    labels.append(lbl)
                elif isinstance(self.n_inference_ensemble, int) and self.n_inference_ensemble > 0:
                    for i in range(self.n_inference_ensemble):
                        ids, mask, lbl = self._tokenize_once_accAndRand(user, item_seq)
                        input_ids.append(ids)
                        attention_mask.append(mask)
                    labels.append(lbl)  #
                else:
                    raise ValueError(
                        f'n_inference_ensemble must be positive integer!, now n_inference_ensemble= {self.n_inference_ensemble}')

            elif self.config['augmentation_probability'] == 1:
                if self.n_inference_ensemble == -1:
                    ids, mask, lbl = self._tokenize_once_random(user, item_seq)
                    input_ids.append(ids)
                    attention_mask.append(mask)
                    labels.append(lbl)
                elif isinstance(self.n_inference_ensemble, int) and self.n_inference_ensemble > 0:
                    for i in range(self.n_inference_ensemble):
                        ids, mask, lbl = self._tokenize_once_random(user, item_seq)
                        input_ids.append(ids)
                        attention_mask.append(mask)
                    labels.append(lbl)
                else:
                    raise ValueError(
                        f'n_inference_ensemble must be positive integer!, now n_inference_ensemble= {self.n_inference_ensemble}')

            else:
                raise ValueError("config ['augmentation_probability'] must be at [0,1]!!!!!!!")
      
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

    def get_random_value_by_bi(self, bi_key):
        """
        return a value randomly from item2tokenlist with a key of bi_key

        given a itemID (int) as a key, to find the corresponding value: sid token list,
        return a sid token randomly

        params:
            item2tokenlist: is a dict, {Bi (itemID): [value1, value2, ...]  (sid token list)}
            bi_key: key (itemID)
        """


        bi_key = str(bi_key)
        if bi_key not in self.itemID2sidTokenList:
            raise KeyError(f"Key '{bi_key}' not found in self.itemID2sidTokenList")

        values = self.itemID2sidTokenList[bi_key]
        if not values:
            raise ValueError(f"Key '{bi_key}' has an empty list of values")

        return random.choice(values)

    def _build_itemID2tokenList(self, interactionKey2tokens_dict):
        """
        fuction：original format: Ai-Bi-Ci : values, group by Bi, all the values are appended into a list.
        params:
            item2tokens_dict:  { "Ai-Bi-Ci": (value1, value2,...) }
        return：
            a new dict { Bi: [all the corresponding values] }
        """

        all_bi = set()
        for key in interactionKey2tokens_dict:
            try:
                bi = key.split("-")[1]
                all_bi.add(bi)
            except IndexError:
                continue  # skip

        itemID2tokenlist = defaultdict(list)
        append_count = 0

        for key, value in interactionKey2tokens_dict.items():
            try:
                bi = key.split("-")[1]
                itemID2tokenlist[bi].append(value)
                append_count += 1
            except IndexError:
                continue


        return itemID2tokenlist

    def parse_key(self, key_str):
        # user_id-item_id-preordered
        parts = key_str.split('-')
        A = int(parts[0])
        B = int(parts[1])
        C = '-'.join(parts[2:])
        return (A, B, C)

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

        meta_sentences = []  # 1-base, meta_sentences[0] -> item_id = 1

      
        for i in range(1, dataset.n_items+1):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device']
        )

        # this line is for debug runhyper, without this 2 lines of code, round 1 will raise error
        # cause when round 0 is deleting, round 1 is try to delete.
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        sent_embs.tofile(output_path)

        return sent_embs

    def _extend_semantic_ids(self, sem_ids: np.ndarray):
        """
        Extends the semantic IDs from k digits to (k + 1) digits to avoid conflict.

        Args:
            sem_ids (np.ndarray): The input array of semantic IDs.

        Returns:
            dict: A dictionary mapping item IDs to semantic IDs.
        """

        sem_id2key = defaultdict(list)
      
        clusterIndex2semids_dict = {}
        max_conflict = 0
        for i in range(sem_ids.shape[0]):
            str_id = ' '.join(map(str, sem_ids[i].tolist()))
         

            # self.clusterIndex_keys is a list, len is N= the number of cluster centers.
            # the value is cluster_index. format: item_token-index_of_cluster

            sem_id2key[str_id].append(self.clusterIndex_keys[i])

            # sem_ids[i] is the semantic ids (three ints) from the self.clusterIndex_keys[i] (cluster_index)
            # self.clusterIndex_keys[i] (a cluster_index) corresponds to a 256 diff emb, and then is pca to 64, concat 128 text emb
            # the 192 emb is related to the cluster_index, and then it's fed into the faiss, generating a sid (three ints).
            # so the cluster_index corresponds to a sid (three ints)
            # different cluster_index may have the same sid, so we need to deal with the conflict.

         
            key = self.clusterIndex_keys[i]
 
            clusterIndex2semids_dict[key] = (*tuple(sem_ids[i].tolist()), len(sem_id2key[str_id]))
            max_conflict = max(max_conflict, len(sem_id2key[str_id]))
        if self.accelerator.is_main_process:
            self.log(f'[TOKENIZER] RQ-VAE semantic IDs, maximum conflict: {max_conflict}')
        if max_conflict > self.codebook_sizes[-1]:
            raise ValueError(
                f'[TOKENIZER] RQ-VAE semantic IDs conflict with codebook size: '
                f'{max_conflict} > {self.codebook_sizes[-1]}. Please increase the codebook size.'
            )

        return clusterIndex2semids_dict

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """
        Get a boolean mask indicating which items are used for training.

        Args:
            dataset (AbstractDataset): The dataset containing the item sequences.

        Returns:
            np.ndarray: A boolean mask indicating which items are used for training.
        """
        items_for_training = set()

        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        if self.accelerator.is_main_process:
            self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items}, the remainings are cold-start items')

        mask = np.zeros(dataset.n_items, dtype=bool)

        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask

    def get_interactionKey2sid(self, clusterIndex2semids_dict):

        threeele_path = os.path.join(self.config['dict_dir'], self.config['dataset'], self.config['category'],
                                     self.config['cluster_file1_ThreeEle'])

        with open(threeele_path, 'rb') as f:
            ThreeEle_clustered_sorted_data = pickle.load(f)

        interactionKey2sem_ids = {}

        missing_keys = []
        duplicate_check_set = set()

        for key1, value in ThreeEle_clustered_sorted_data.items():
            cluster_index = value['cluster_index']

            if cluster_index in clusterIndex2semids_dict:
                value1 = clusterIndex2semids_dict[cluster_index]
                interactionKey2sem_ids[key1] = value1
                duplicate_check_set.add(cluster_index)
            else:
                missing_keys.append(cluster_index)
                raise KeyError(f"[ERROR] cluster_index '{cluster_index}' is not in key2sem_ids, key1 = {key1}")

 
        return interactionKey2sem_ids





    def get_2eles_interactionKey_To_clusterIndex_emb(self, clusterIndex2semids_dict):
        threeele_path = os.path.join(
            self.config['dict_dir'],
            self.config['dataset'],
            self.config['category'],
            self.config['cluster_file1_ThreeEle']
        )

        with open(threeele_path, 'rb') as f:
            ThreeEle_clustered_sorted_data = pickle.load(f)

        result_dict = {}
        missing_keys = []

        for key1, value in ThreeEle_clustered_sorted_data.items():
            cluster_index = value['cluster_index']

            if cluster_index in clusterIndex2semids_dict:
         
                emb = value['emb']
                emb_array = np.array(emb)
                '''
                emb_info = {
                    'norm': float(np.linalg.norm(emb_array)),
                    'mean': float(np.mean(emb_array)),
                    'std': float(np.std(emb_array)),
                    'max_min': float(np.max(emb_array) - np.min(emb_array)),
                }
                '''

                result_dict[key1] = {
                
                    'clusterIndex': cluster_index,

                    'emb': emb_array
                }

            else:
                missing_keys.append(cluster_index)
                raise KeyError(
                    f"[ERROR] cluster_index '{cluster_index}' not in clusterIndex2semids_dict, key1 = {key1}")

        return result_dict











    def make_preview_dict(self,full_dict, dim=5):
   
        preview_dict = copy.deepcopy(full_dict)

        for k, v in preview_dict.items():
            if "emb" in v:
              
                v["emb_first_five"] = np.array(v["emb"][:dim])
                del v["emb"]  

        return preview_dict



 

    def _generate_semantic_id_faiss(
            self,
            fused_emb: np.ndarray,
         
            train_mask: np.ndarray
    ) -> None:
        """
        Generates semantic IDs using the Faiss library and saves them to a file.

        Args:
            fused_emb (np.ndarray): The fused embeddings.
            sem_ids_path (str): The path to save the semantic IDs.
            train_mask (np.ndarray): A boolean mask indicating which items are used for training.

        Returns:
            None
        """

        n_bits = int(np.log2(self.config['rq_codebook_size']))

        import faiss
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])

        '''
        notice that
        fused_emb is (N,192)
        N means the number of clusters, usually XX (100-500) percents of the number of items, 
        192 is 64 concat 128
        '''
        train_mask=train_mask

        index = faiss.IndexResidualQuantizer(
            fused_emb.shape[-1],
            self.config['rq_n_codebooks'],
            n_bits,
            faiss.METRIC_INNER_PRODUCT
        )

        if self.accelerator.is_main_process:
            self.log(f'[TOKENIZER] Training index...')
        index.train(fused_emb)

        index.add(fused_emb)

        uint8_code = index.rq.compute_codes(fused_emb)

        n_bytes = uint8_code.shape[1]

        faiss_sem_ids = []
        if self.accelerator.is_main_process:
            self.logger.info(f'[TOKENIZER] Generating semantic IDs...')
        for u8_code in uint8_code:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), n_bytes)
            code = []
            for i in range(self.config['rq_n_codebooks']):
                code.append(bs.read(n_bits))
            faiss_sem_ids.append(code)

        faiss_sem_ids = np.array(faiss_sem_ids)
        # clusterIndex2semids_dict is a dict, key is the clusterIndex, format: itemID-clusterIndex
        # e.g. 5-3, means the 3rd cluster centers of item 5.
        # value is the sids, with 4 ints.
        clusterIndex2semids_dict = self._extend_semantic_ids(faiss_sem_ids)

        # interactionKey2sid is a dict, key is the interactionkey, format: A-B-C, userID-itemID-preorderedSeq
        # value is the sids, with 4 ints.
        interactionKey2sid = self.get_interactionKey2sid(clusterIndex2semids_dict)

    
    



        self.dict_2eles_interactionKey_To_clusterIndex_emb = self.get_2eles_interactionKey_To_clusterIndex_emb(clusterIndex2semids_dict)
       
        dict_preview = self.make_preview_dict(self.dict_2eles_interactionKey_To_clusterIndex_emb, dim=5)
      



        value_types = set(type(v) for v in interactionKey2sid.values())
        if self.accelerator.is_main_process:
            self.log(f"[analysis] The python type of all values: {value_types}")

        non_int_list_items = [
            item for item, sem_id in interactionKey2sid.items()
            if not isinstance(sem_id, list) or not all(isinstance(x, int) for x in sem_id)
        ]
        if self.accelerator.is_main_process:
            self.log(f"[analysis] The number of seqs with no int elements: {len(non_int_list_items)}")

        from collections import Counter

        length_counter = Counter(len(v) for v in interactionKey2sid.values())
        for length, count in sorted(length_counter.items()):
            self.log(f"[analysis] The number of sids of length {length} : {count}")

        all_last_values = [v[-1] for v in interactionKey2sid.values()]
        max_last = max(all_last_values)
        if self.accelerator.is_main_process:
            self.log(f"[analysis] the max value of the last position (for conflict) all over the sids: {max_last}")

        suffix_values = [v[-1] for v in interactionKey2sid.values() if len(v) == 4]
        if suffix_values:
            max_suffix = max(suffix_values)

            self.log(f"[analysis] Maximum value of conflict markers in a semantic ID of length 4: {max_suffix}")
        else:
            self.log(f"[analysis] There are no sids of length 4")

        sem_id_matrix = np.array(list(interactionKey2sid.values()), dtype=int)

        if self.accelerator.is_main_process:
            self.log("[analysis] the range of every dimension (min, max):")
            for dim in range(sem_id_matrix.shape[1]):
                col_min = sem_id_matrix[:, dim].min()
                col_max = sem_id_matrix[:, dim].max()
                self.log(f"  No. {dim} dimension: min = {col_min}, max = {col_max}")
        return interactionKey2sid



    def _interactionKey_2_sidTokens_add_offest(self, interactionKey2sid: dict) -> dict:
        """
        Converts semantic IDs to tokens.

        Args:
            interactionKey2sid (dict): A dictionary mapping items to their corresponding semantic IDs.
            key is A-B-C interaction keys, value is sid tokens

        Returns:
            dict: A dictionary mapping items to their corresponding tokens.
        """

        sem_id_offsets = [0]

        for digit in range(1, self.n_digit):
            sem_id_offsets.append(sem_id_offsets[-1] + self.codebook_sizes[digit - 1])

        for key in interactionKey2sid:
            # tokens= list ((2, 220, 52, 1))= [2,220,52,1]
            tokens = list(interactionKey2sid[key])
            for digit in range(self.n_digit):  # digit 取值是0 1 2 3
                # "+ 1" as 0 is reserved for padding
                # tokens=[2,220,52,1]
                # sem_id_offsets = [0, 256, 512, 768]
                # tokens[0]+=sem_id_offsets[0] + 1  -> 2+=0+1 -> 3
                # tokens[1]+=sem_id_offsets[1] + 1  -> 220+=256+1 -> 477
                # tokens[2]+=sem_id_offsets[2] + 1  -> 52+=512+1 -> 565
                # tokens[3]+=sem_id_offsets[3] + 1  -> 1+=768+1 -> 770
                tokens[digit] += sem_id_offsets[digit] + 1
                # [2,220,52,1] to [3,477,565,770]
            interactionKey2sid[key] = tuple(tokens)
            # (2,220,52,1) to (3,477,565,770)
        return interactionKey2sid

    def get_fused_clustred_sorted_interaction2personalized_semantic_emb(self, sent_embs):
        """
       om:
            sent_embs: numpy.ndarray, shape (17591, 768)
            alpha: float, concat wegiht, default = 0.5
        """
        alpha = self.config['alpha_fused_for_diff']
        pca_diff_dimension = self.config['pca_diff_dimension']
        pca_sentence_dimension = self.config['pca_sentence_dimension']

        setClusterEmb_path = os.path.join(self.config['dict_dir'], self.config['dataset'], self.config['category'],
                                          self.config['cluster_file2_SetClusterIndexEmb'])

        with open(f"{setClusterEmb_path}", 'rb') as f:
            self.SetClusterEmb_ThreeEle_clustered_sorted_data = pickle.load(f)

        # Step 2: parse item_token from key and record the order
        self.clusterIndex_keys = list(self.SetClusterEmb_ThreeEle_clustered_sorted_data.keys())
        item_tokens = [int(k.split('-')[0]) for k in self.clusterIndex_keys]  # e.g., '1-3' -> 1

        # Step 3: pca diff emb to 64 with whiten
        emb_matrix = np.array([
            self.SetClusterEmb_ThreeEle_clustered_sorted_data[k].cpu().numpy()
            for k in self.clusterIndex_keys
        ])
        pca_item_emb = PCA(n_components=pca_diff_dimension, whiten=True)
        pca_64_item_emb = pca_item_emb.fit_transform(emb_matrix)

        # Step 4: pca text emb to 128 with whiten
        pca_sent = PCA(n_components=pca_sentence_dimension, whiten=True)
        pca_128_sent_emb = pca_sent.fit_transform(sent_embs)  # shape: (17591, 128)

        # Step 5: concat the two
        fused_embs = []
        self.fused_embs_dict = {}

        for i, item_token in enumerate(item_tokens):
            item_emb = pca_64_item_emb[i] * alpha
            sent_vec = pca_128_sent_emb[item_token - 1] * (1 - alpha)
            fused = np.concatenate([item_emb, sent_vec])  # shape: (192,)
            fused_embs.append(fused)

            cluster_index = self.clusterIndex_keys[i]
            if cluster_index in self.fused_embs_dict:
                raise KeyError(f"Key '{cluster_index}' already exists")
            else:
                self.fused_embs_dict[cluster_index] = fused
        '''
            self.fused_embs_dict is a dict, key is cluster_index, i.e., itemToken-cluster_index
            value is the fused emb.
        '''

        # Step 6:  (N, 192)
        fused_embs_matrix = np.array(fused_embs)  # shape: (N, 192)
        if self.accelerator.is_main_process:
            self.log("\n[Step 6] Fused embedding matrix created.")
            self.log(f"  Shape: {fused_embs_matrix.shape}")
            self.log(f"  Type: {type(fused_embs_matrix)}")

        # N is the number of cluster centers.

        return fused_embs_matrix  # shape (N, 192), have the same order with the key in SetClusterEmb

    def get_gene_sid_or_not(self):
        return self.gene_sid_or_not

    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Initialize the tokenizer.

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """

 

        sent_emb_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
        )

        if os.path.exists(sent_emb_path):
            self.log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
            sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
        else:

            self.log(f'[TOKENIZER] Encoding sentence embeddings...')
            sent_embs = self._encode_sent_emb(dataset, sent_emb_path)

        fused_emb = self.get_fused_clustred_sorted_interaction2personalized_semantic_emb(sent_embs)

        # PCA

        self.log(f'[TOKENIZER] Sentence embeddings shape: {fused_emb.shape}')
        self.num_clustered_interaction = fused_emb.shape[0]
        self.log(f'[TOKENIZER] self.num_clustered_interaction = {self.num_clustered_interaction}')

        # Generate semantic IDs
        training_item_mask = self._get_items_for_training(dataset)
        if self.config['rq_faiss']:
            self.log(f'[TOKENIZER] Semantic IDs not found. Training index using Faiss...')
            interactionKey2sid=self._generate_semantic_id_faiss(fused_emb, training_item_mask)

        else:

            self.log(f'[TOKENIZER] Semantic IDs not found. Training RQ-VAE model...')
          
            embs_for_training = torch.FloatTensor(fused_emb).to(self.config['device'])
            fused_emb = torch.FloatTensor(fused_emb).to(self.config['device'])
            model_path = os.path.join(dataset.cache_dir, 'processed/rqvae.pth')
            rqvae_model = self._train_rqvae(embs_for_training, model_path)
            interactionKey2sid=self._generate_semantic_id(rqvae_model, fused_emb)

     

        interactionKey2sidTokens = self._interactionKey_2_sidTokens_add_offest(interactionKey2sid)
        #


        return interactionKey2sidTokens



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

    def _token_single_item(self, interactionKey: str) -> int:
        """
        Tokenizes a single item.

        Args:
            interactionKey (str): is the interactionKey, A-B-C

        Returns:
            list: The tokens corresponding to the item.
        """
        interactionKey = str(interactionKey)
        return self.interactionkey2sidTokens[interactionKey]

    '''
    for better understanding of this function, please refer to 'test_tokenize_once.ipynb in genrec/models/Pctx'
    '''

    def return_one_item_interactionKey_ABC_in_a_seq(self, lst, idx, userID, max_len):
        '''
        Args:
            lst:  list, a list containing interactive items (seq)
            idx:   int, the index of an item in this list
            userID:  int, user_id
            max_len: int, max_length

        Returns:
            a string, indicating the interactionKey,
            whose format is A-B-C, i.e., userID-itemID-preorderedSeq, corresponding to the index in this seq list.
        '''
        '''
        for better understanding of this function, please refer to 'test_tokenize_once.ipynb in genrec/models/Pctx'
        '''

        A = userID
        B = lst[idx]
        start_idx = max(0, idx - (max_len - 1))
        if idx == 0:
            preorderedSeq = '0'
        else:
            preorderedSeq = '_'.join(str(x) for x in lst[start_idx:idx])
        C = preorderedSeq
        interactionKey = f"{A}-{B}-{C}"
        return interactionKey

    '''
    for better understanding of this function, please refer to 'test_tokenize_once.ipynb in genrec/models/Pctx'
    '''

    def _tokenize_once_accAndRand(self, user_para, item_seq_para) -> tuple:
        '''
        for better understanding of this function, please refer to 'test_tokenize_once.ipynb in genrec/models/Pctx'
        '''
        example = {'user': user_para, 'item_seq': item_seq_para}

        user_id = example['user']

        max_item_seq_len = self.config['max_item_seq_len']

        input_ids = []

        lst = example['item_seq']
        max_len = self.config['max_item_seq_len']

        a = 1-self.config['augmentation_probability']

        N = len(lst)
        M = len(lst) - max(0, len(lst) - (max_len + 1))

        label_str_flag = -1

        proba_lst = [-1] * N
        start = max(0, len(lst) - (max_len + 1))
        for i in range(start, N):
            aaaa = random.random()

            proba_lst[i] = 0 if aaaa < a else 1

        for i in range(start, N):
            if proba_lst[i] == -1:
                raise ValueError(f"index {i} could not be -1")
            elif proba_lst[i] == 0:
                # this means tokenizing accurately
                result = self.return_one_item_interactionKey_ABC_in_a_seq(lst, i, user_id, max_len)
                if i != len(lst) - 1:
                    input_ids.extend(self._token_single_item(result))
                else:
                    label_str = result
                    label_str_flag = 0
            else:
                # this means augmentation, tokenizing one item randomly with its multiple personalized semantic IDs
                if i != len(lst) - 1:
                    input_ids.extend(self.get_random_value_by_bi(lst[i]))
                else:
                    label_str = lst[i]
                    label_str_flag = 1

        input_ids.append(self.eos_token)

        input_ids.extend([self.padding_token] * (self.max_token_seq_len - len(input_ids)))

        # attention_mask

        item_seq_len = min(len(example['item_seq'][:-1]), max_item_seq_len)

        # use +1 instead of +2, because we have the <eos> token and no user token.
        attention_mask = [1] * (self.n_digit * item_seq_len + 1)

        attention_mask.extend([0] * (self.max_token_seq_len - len(attention_mask)))

        if label_str_flag == -1:
            raise ValueError('!!!!fail to get label,-1')
        elif label_str_flag == 0:  # accurate, augmentation_probability=0
            labels = list(self._token_single_item(label_str)) + [self.eos_token]
        elif label_str_flag == 1:  # random,augmentation_probability=1
            labels = list(self.get_random_value_by_bi(label_str)) + [self.eos_token]
        else:
            raise ValueError(f'!!!!fail to get label,-1')
        # labels = list(self._token_single_item(label_str)) + [self.eos_token]
        # labels = list(self.get_random_value_by_bi(label_str)) + [self.eos_token]
        return input_ids, attention_mask, labels

    '''
    for better understanding of this function, please refer to 'test_tokenize_once.ipynb in genrec/models/Pctx'
    '''

    def _tokenize_once_random(self, user_para, item_seq_para) -> tuple:
        '''
        for better understanding of this function, please refer to 'test_tokenize_once.ipynb in genrec/models/Pctx'
        '''

        example = {'user': user_para, 'item_seq': item_seq_para}

        user_id = example['user']

        max_item_seq_len = self.config['max_item_seq_len']

        input_ids = []

        lst = example['item_seq']
        max_len = self.config['max_item_seq_len']

        # get_random_value_by_bi
        for i in range(max(0, len(lst) - (max_len + 1)), len(lst)):
            if i != len(lst) - 1:
                input_ids.extend(self.get_random_value_by_bi(lst[i]))
            else:
                label_str = lst[i]

        input_ids.append(self.eos_token)

        input_ids.extend([self.padding_token] * (self.max_token_seq_len - len(input_ids)))

        # attention_mask

        item_seq_len = min(len(example['item_seq'][:-1]), max_item_seq_len)

        # use +1 instead of +2, because we have the <eos> token and no user token.
        attention_mask = [1] * (self.n_digit * item_seq_len + 1)

        attention_mask.extend([0] * (self.max_token_seq_len - len(attention_mask)))

        labels = list(self.get_random_value_by_bi(label_str)) + [self.eos_token]

        return input_ids, attention_mask, labels

    '''
    for better understanding of this function, please refer to 'test_tokenize_once.ipynb in genrec/models/Pctx'
    '''

    def _tokenize_once_accurate(self, user_para, item_seq_para) -> tuple:

        """
        Tokenizes a single example.

        Args:
            user_para: user_id, string.
            item_seq_para: a list containing the interactive itemID, list.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, and labels.
        """

        '''
        for better understanding of this function, please refer to 'test_tokenize_once.ipynb in genrec/models/Pctx'
        '''

        example = {'user': user_para, 'item_seq': item_seq_para}

        # example['user'] =  self.user2id [  example['user'] ]

        user_id = example['user']

        #
        # example['item_seq'] = [self.item2id[item] for item in example['item_seq']]

        max_item_seq_len = self.config['max_item_seq_len']

        input_ids = []

        lst = example['item_seq']
        max_len = self.config['max_item_seq_len']

        for i in range(max(0, len(lst) - (max_len + 1)), len(lst)):
            interactionKey = self.return_one_item_interactionKey_ABC_in_a_seq(lst, i, user_id, max_len)
            if i == len(lst) - 1:
                label_interactionKey_str = interactionKey
            else:
                input_ids.extend(self._token_single_item(interactionKey))

        input_ids.append(self.eos_token)

        input_ids.extend([self.padding_token] * (self.max_token_seq_len - len(input_ids)))

        # attention_mask
        item_seq_len = min(len(example['item_seq'][:-1]), max_item_seq_len)

        # use +1 instead of +2, because we have the <eos> token and no user token.
        attention_mask = [1] * (self.n_digit * item_seq_len + 1)
        attention_mask.extend([0] * (self.max_token_seq_len - len(attention_mask)))
        # labels
        # label_str=self.format_str(example['item_seq'][-max_item_seq_len:], example['item_seq'][-1], user_id)
        labels = list(self._token_single_item(label_interactionKey_str)) + [self.eos_token]

        return input_ids, attention_mask, labels

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

        if split == 'train':
            n_return_examples = len(example['item_seq'][0]) - 1
            # n_return_examples=6
            all_input_ids, all_attention_mask, all_labels = [], [], []
            user_list = []
            item_seq_list = []
            for i in range(n_return_examples):  # for i in range(0,6) ,get 0 1 2 3 4 5
                cur_example = {
                    'user': example['user'][0],
                    'item_seq': example['item_seq'][0][:i + 2]
                }

                cur_example = {
                    'user': self.user2id[cur_example['user']],
                    'item_seq': [self.item2id[item] for item in cur_example['item_seq']]
                }

                user_list.append(cur_example['user'])
                item_seq_list.append(cur_example['item_seq'])

            return {
                'user': user_list,
                'item_seq': item_seq_list
            }
        else:

            user = self.user2id[example['user'][0]]
            item_seq = [self.item2id[item] for item in example['item_seq'][0]]

            return {
                'user': [user],
                'item_seq': [item_seq]
            }
        #

    def tokenize(self, datasets: dict) -> dict:
        """
        Tokenizes the given datasets.

        Args:
            datasets (dict): A dictionary of datasets to tokenize.

        Returns:
            dict: A dictionary of tokenized datasets.
        """
        '''
        split_datasets = {'train': {'user': [], 'item_seq': []},
                            'val': {'user': [], 'item_seq': []},
                           'test': {'user': [], 'item_seq': []}}
        '''

        tokenized_datasets = {}
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

        return tokenized_datasets



    def tokenize_only_test(self, datasets: dict) -> dict:
        """
        Tokenizes the given datasets.

        Args:
            datasets (dict): A dictionary of datasets to tokenize.

        Returns:
            dict: A dictionary of tokenized datasets.
        """
        '''
        split_datasets = {'train': {'user': [], 'item_seq': []},
                            'val': {'user': [], 'item_seq': []},
                           'test': {'user': [], 'item_seq': []}}
        '''

        tokenized_datasets = {}

        split='test'

        tokenized_datasets[split] = datasets[split].map(
            lambda t: self.tokenize_function(t, split),
            batched=True,
            batch_size=1,
            remove_columns=datasets[split].column_names,
            num_proc=self.config['num_proc'],
            desc=f'Tokenizing {split} set: '
        )


        tokenized_datasets[split].set_format(type='torch')

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
        # now we do not use user token, so here is +1
        return self.config['max_item_seq_len'] * self.n_digit + 1

    # This function is for RQ-VAE NN training
    def _train_rqvae(self, sent_embs: torch.Tensor, model_path: str) -> RQVAEModel:
        """
        Trains the RQ-VAE model using the given sentence embeddings.

        Args:
            sent_embs (torch.Tensor): Array of sentence embeddings.
            model_path (str): Path to save the trained model.

        Returns:
            rqvae_model: Trained RQ-VAE model.
        """
        device = self.config['device']

        # Initialize RQ-VAE model
        all_hidden_sizes = [sent_embs.shape[1]] + self.config['rqvae_hidden_sizes']
        rqvae_model = RQVAEModel(
            hidden_sizes=all_hidden_sizes,
            n_codebooks=self.config['rq_n_codebooks'],
            codebook_size=self.config['rq_codebook_size'],
            dropout=self.config['rqvae_dropout'],
            low_usage_threshold=self.config['rqvae_low_usage_threshold']
        ).to(device)
        self.log(rqvae_model)
        if os.path.exists(model_path):
            self.log(f"[TOKENIZER] Loading RQ-VAE model from {model_path}...")
            rqvae_model.load_state_dict(torch.load(model_path))
            return rqvae_model

        # Model training
        batch_size = self.config['ravae_batch_size']
        num_epochs = self.config['rqvae_epoch']
        beta = self.config['rqvae_beta']
        verbose = self.config['rqvae_verbose']

        rqvae_model.generate_codebook(sent_embs, device)
        optimizer = torch.optim.Adagrad(rqvae_model.parameters(), lr=self.config['rqvae_lr'])
        train_dataset = TensorDataset(sent_embs)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.log("[TOKENIZER] Training RQ-VAE model...")
        rqvae_model.train()
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0.0
            total_rec_loss = 0.0
            total_quant_loss = 0.0
            total_count = 0
            for batch in dataloader:
                x_batch = batch[0]
                optimizer.zero_grad()
                recon_x, quant_loss, count = rqvae_model(x_batch)
                reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction='mean')
                loss = reconstruction_mse_loss + beta * quant_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().cpu().item()
                total_rec_loss += reconstruction_mse_loss.detach().cpu().item()
                total_quant_loss += quant_loss.detach().cpu().item()
                total_count += count

            if (epoch + 1) % verbose == 0:
                self.log(
                    f"[TOKENIZER] RQ-VAE training\n"
                    f"\tEpoch [{epoch + 1}/{num_epochs}]\n"
                    f"\t  Training loss: {total_loss / len(dataloader)}\n"
                    f"\t  Unused codebook:{total_count / len(dataloader)}\n"
                    f"\t  Recosntruction loss: {total_rec_loss / len(dataloader)}\n"
                    f"\t  Quantization loss: {total_quant_loss / len(dataloader)}\n")
        self.log("[TOKENIZER] RQ-VAE training complete.")

        # Save model
        torch.save(rqvae_model.state_dict(), model_path, pickle_protocol=4)
        return rqvae_model

    # This function is for RQ-VAE NN training
    def _generate_semantic_id(
            self,
            rqvae_model: RQVAEModel,
            sent_embs: torch.Tensor,
       
    ) -> None:
        """
        Generates semantic IDs using the given RQVAE model and saves them to a file.

        Args:
            rqvae_model (RQVAEModel): The RQVAE model used for encoding sentence embeddings.
            sent_embs (torch.Tensor): The sentence embeddings to be encoded.
            sem_ids_path (str): The path to save the generated semantic IDs.

        Returns:
            None
        """
        rqvae_model.eval()
        rqvae_sem_ids = rqvae_model.encode(sent_embs)

        clusterIndex2semids_dict = self._extend_semantic_ids(rqvae_sem_ids)

        # interactionKey2sid is a dict, key is the interactionkey, format: A-B-C, userID-itemID-preorderedSeq
        # value is the sids, with 4 ints.
        interactionKey2sid = self.get_interactionKey2sid(clusterIndex2semids_dict)

 

   
        self.log(f"[analysis] len(interactionKey2sid): {len(interactionKey2sid)}")

        self.log("[analysis] sid of first 5 items:")
        for i, (item, sem_id) in enumerate(interactionKey2sid.items()):
            if i >= 5:
                break
            self.log(f"  {item}: {sem_id}")

        value_types = set(type(v) for v in interactionKey2sid.values())
        self.log(f"[analysis] The python type of all values: {value_types}")

        non_int_list_items = [
            item for item, sem_id in interactionKey2sid.items()
            if not isinstance(sem_id, list) or not all(isinstance(x, int) for x in sem_id)
        ]
        self.log(f"[analysis] The number of seqs with no int elements: {len(non_int_list_items)}")

        from collections import Counter

        length_counter = Counter(len(v) for v in interactionKey2sid.values())
        for length, count in sorted(length_counter.items()):
            self.log(f"[analysis] The number of sids of length {length} : {count}")

        all_last_values = [v[-1] for v in interactionKey2sid.values()]
        max_last = max(all_last_values)
        self.log(f"[analysis] the max value of the last position (for conflict) all over the sids: {max_last}")

        suffix_values = [v[-1] for v in interactionKey2sid.values() if len(v) == 4]
        if suffix_values:
            max_suffix = max(suffix_values)
            self.log(f"[analysis] Maximum value of conflict markers in a semantic ID of length 4: {max_suffix}")
        else:
            self.log(f"[analysis] There are no sids of length 4")

        sem_id_matrix = np.array(list(interactionKey2sid.values()), dtype=int)

        self.log("[analysis] the range of every dimension (min, max):")
        for dim in range(sem_id_matrix.shape[1]):
            col_min = sem_id_matrix[:, dim].min()
            col_max = sem_id_matrix[:, dim].max()
            self.log(f"  No. {dim} dimension: min = {col_min}, max = {col_max}")

        return interactionKey2sid