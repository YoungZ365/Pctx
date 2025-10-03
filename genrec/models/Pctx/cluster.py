import pickle
import torch
import os

import pickle
import torch
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import KMeans, DBSCAN
import random

import math
import csv



from scipy.stats import gamma


def arithmetic_sequence(n, start, d):
    return [start + i * d for i in range(n)]

def extract_and_normalize(k, x_range_start, x_range_end):
    # Set the scale parameter to 1 (normalized)
    theta = 1

    # Generate the range of x values
    x_points = np.arange(x_range_start, x_range_end + 1)  # Generate integer x values within range
    # Compute the corresponding y values from the Gamma PDF
    y_points = gamma.pdf(x_points, k, scale=theta)

    # Normalize the y values so their sum equals 1
    y_normalized_sum1 = y_points / np.sum(y_points)

    # Round to two decimal places
    y_rounded = np.round(y_normalized_sum1, 2)

    # Calculate the difference between the sum and 1
    diff = 1 - np.sum(y_rounded)

    # Find the indices of the two largest values
    largest_two_indices = np.argsort(y_rounded)[-2:]

    # Evenly distribute the difference across the two largest values
    y_rounded[largest_two_indices[0]] += diff / 2
    y_rounded[largest_two_indices[1]] += diff / 2

    # Return the adjusted list of y values rounded to two decimals and summing to 1
    return list(y_rounded)

class PctxCluster():
    def __init__(
        self,
        config: dict,
        log_fuc
    ):
        self.config_upstream=config

        self.log=log_fuc

        '''
        key: A-B-C, userID-itemID-preorderedSeq
        value: 256 d emb
        '''
        self.data=self.read_diff_emb()

        '''
          key: A-B-C, userID-itemID-preorderedSeq
          value: 256 d emb
          
          the difference from self.data is that self.sorted_data is sorted by itemID, i.e. B.
        '''
        self.sorted_data=self.sorted_by_itemID_return_dict(self.data)


        self.T=self.config_upstream['n_groups']
        self.d=self.config_upstream['distance']
        self.start=self.config_upstream['start']
        self.k_gamma=self.config_upstream['k_gamma']



        self.T=self.T
        self.K_list=arithmetic_sequence(self.T,self.start,self.d)
        self.split_ratio=extract_and_normalize(self.k_gamma,1,self.T)





        if not isinstance(self.K_list, list):
            raise TypeError("para_list must be a list.")




        assert len(self.K_list) == self.T



        self.itemID_count,self.itemID_to_interactionKey= self.count_itemID_with_interactions()




        self.token_groups, self.quantile_bounds=self.split_itemID_by_sample_count_quantiles(self.itemID_count, self.T, self.split_ratio)

        self.statics_of_token_group()


        self.cluster_file1_ThreeEle,group_stats =self.clustering()

     
        self.export_per_item_cluster_statics(group_stats)

   

        # sort by itemID
        self.cluster_file1_ThreeEle = self.sorted_by_itemID_return_dict(self.cluster_file1_ThreeEle)
        '''
        Notice that cluster_file1_ThreeEle is a dict, key is interactionKey (A-B-C, i.e., userID-itemID-preorderedSeq)
        
        Value is a dict, Value's key is cluster_index (itemID-clusterIndex), Value'value is emb (256 dimension)
        '''

        self.save_to_pkl(self.cluster_file1_ThreeEle, self.config_upstream['cluster_file1_ThreeEle'])


        self.cluster_file2_SetClusterIndexEmb=self.get_cluster_file2_SetClusterIndexEmb(self.cluster_file1_ThreeEle)

        self.save_to_pkl(self.cluster_file2_SetClusterIndexEmb, self.config_upstream['cluster_file2_SetClusterIndexEmb'])

   
    def export_per_item_cluster_statics(self, group_stats):
        """
        output file in cluster/cluster_statics.csv, containing detailed information for every itemID
        """
        import os, csv

        statics_list = []
        for g in group_stats:
            group_index = g['group_index']
            group_param = g['param']
            group_range = f"[{int(self.quantile_bounds[group_index])}, {int(self.quantile_bounds[group_index + 1])}]"
            group_n_itemID = g['token_count']
            group_n_interactions = sum(
                len(self.itemID_to_interactionKey[token_id]) for token_id in self.token_groups[group_index]
            )

            algorithm_used = g['algorithm']
            params_used = g['params']
            group_ideal_clusters = g['total_clusters']
            group_used_clusters = g['used_clusters']
            group_clusters_in_use_per_item = g['avg_clusters_per_token']

            tokens_in_group = self.token_groups[group_index]
            for token_id in tokens_in_group:
                n_interactions = len(self.itemID_to_interactionKey[token_id])

                if algorithm_used == "DBSCAN":
                    n_clusters = round(group_clusters_in_use_per_item)
                else:
                    if isinstance(group_param, dict):
                        n_clusters = -1
                    elif group_param < 1:
                        n_clusters = math.ceil(group_param * n_interactions)
                    else:
                        n_clusters = group_param

                statics_list.append({
                    "itemID": token_id,
                    "n_interactions": n_interactions,
                    "n_clusters": n_clusters,
                    "algorithm": algorithm_used,
                    "algorithm_params": params_used,
                    "groupID": group_index + 1,
                    "group_para": group_param,
                    "group_range": group_range,
                    "group_n_itemID": group_n_itemID,
                    "group_n_interactions": group_n_interactions,
                    "group_ideal_clusters": group_ideal_clusters,
                    "group_used_clusters": group_used_clusters,
                    "group_clusters_in_use_per_item": round(group_clusters_in_use_per_item, 2)
                })

        # sorted by itemID
        statics_list_sorted = sorted(statics_list, key=lambda x: x["itemID"])

        output_dir = "cluster"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "cluster_statics.csv")

        fieldnames = [
            "itemID",
            "n_interactions",
            "n_clusters",
            "algorithm",
            "algorithm_params",
            "groupID",
            "group_para",
            "group_range",
            "group_n_itemID",
            "group_n_interactions",
            "group_ideal_clusters",
            "group_used_clusters",
            "group_clusters_in_use_per_item"
        ]

        with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(statics_list_sorted)

        self.log(f"[INFO] Cluster statics saved to {output_file}")




    def _statics_to_txt(self,file,filename,num_of_pairs):
        # Generate statistical information for the specified dictionary and save it to a text file.
        output_path = f"{filename}.txt"
        filename='self.'+filename
        with open(output_path, "w") as f:

            f.write(f"type({filename}): {type(file)}\n")
            f.write(f"len({filename}): {len(file)}\n\n")


            first_key = next(iter(file))
            first_value = file[first_key]
            f.write("Example key-value pair:\n")
            f.write(f"{first_key} (key type: {type(first_key)}): {first_value} (value type: {type(first_value)})\n")


            f.write(f"\nThe first {num_of_pairs} key-value pairs:\n")
            for i, (k, v) in enumerate(file.items()):
                if i >= num_of_pairs:
                    break
                f.write(f"{k}: {v}\n")

        self.log(f"Statics Information saved to {output_path}")



    def get_cluster_file2_SetClusterIndexEmb(self,ThreeEle_clustered_sorted_data):
        SetClusterEmb_ThreeEle_clustered_sorted_data = {}
        self.duplicate_check_count = 0

        for k, v in ThreeEle_clustered_sorted_data.items():
            emb = v['emb']
            cluster_index = v['cluster_index']

            if cluster_index in SetClusterEmb_ThreeEle_clustered_sorted_data:
                self.duplicate_check_count += 1
                existing_emb = SetClusterEmb_ThreeEle_clustered_sorted_data[cluster_index]

                # check
                if not torch.allclose(existing_emb, emb, atol=1e-6):
                    raise ValueError(f"[ERROR] cluster_index={cluster_index} is repeated but the corresponding value is not the same！")

            else:
                SetClusterEmb_ThreeEle_clustered_sorted_data[cluster_index] = emb
        return SetClusterEmb_ThreeEle_clustered_sorted_data


        a = len(ThreeEle_clustered_sorted_data)
        b = self.duplicate_check_count + len(SetClusterEmb_ThreeEle_clustered_sorted_data)
        if (a == b):
            print()
        else:
            raise ValueError('something went wrong in SetClusterEmb_ThreeEle_clustered_sorted_data')

    def save_to_pkl(self,save_file,filename):


        save_path = os.path.join(self.config_upstream['dict_dir'], self.config_upstream['dataset'],
                                     self.config_upstream['category'], filename)
        with open(save_path, 'wb') as f:
            pickle.dump(save_file, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.log(f'[INFO] saved to {save_path}')

    def clustering(self):
        self.log('[important] start to clustering...')
        ThreeEle_clustered_sorted_data = {}
        total_keys_updated = 0
        all_cluster_index_set = set()


        total_cluster_count = 0
        unused_cluster_count = 0
        token_with_unused_clusters = []

        group_stats = []
        for group_index, tokens_in_group in enumerate(self.token_groups):
            group_param = self.K_list[group_index]


            group_total_clusters = 0
            group_used_clusters = 0
            algorithm_used = ""
            params_used = ""

            for token_id in tokens_in_group:
                keys = self.itemID_to_interactionKey[token_id]

   
                n_interactions=len(keys)
        

                vectors = [self.sorted_data[k].cpu().numpy() for k in keys]

                if isinstance(group_param, dict):
                    raise ValueError('Now we do not support DBSCAN, pleasu input int.')



                else:
                    algorithm_used = "KMeans"
                    params_used = f"n_clusters={group_param}, random_state=42, n_init='auto'"
                    Ki = group_param


                   
                    if group_param<1: #is a ratio
                        Ki=math.ceil(group_param * n_interactions)
                    else:
                        Ki = group_param
           
                    if len(vectors) < Ki:
                        Ki = 1
                 
                    kmeans = KMeans(n_clusters=Ki, random_state=42, n_init='auto')
                    labels = kmeans.fit_predict(vectors)
                    total_cluster_count += Ki
                    centers = kmeans.cluster_centers_
                    used_labels = set(labels)

                    current_used = len(used_labels)
                    group_total_clusters += Ki
                    group_used_clusters += current_used

                    if len(used_labels) < Ki:
                        unused = Ki - len(used_labels)
                        unused_cluster_count += unused
                        token_with_unused_clusters.append((token_id, Ki, len(used_labels), unused))

                    for key, label in zip(keys, labels):
                       
                        emb = torch.tensor(centers[label], dtype=torch.float64)
                        emb = torch.round(emb * 1e8) / 1e8  #
                        emb = emb.to(torch.float32)



                        ThreeEle_clustered_sorted_data[key] = {
                            "emb": emb,
                            "cluster_index": f"{token_id}-{label}"
                        }
                        all_cluster_index_set.add(f"{token_id}-{label}")



                total_keys_updated += len(vectors)


            avg_clusters_per_token = group_used_clusters / len(tokens_in_group) if tokens_in_group else 0
            group_stats.append({
                "group_index": group_index,
                "param": group_param,
                "token_count": len(tokens_in_group),
                "total_clusters": group_total_clusters,
                "used_clusters": group_used_clusters,
                "avg_clusters_per_token": avg_clusters_per_token,
                "algorithm": algorithm_used,
                "params": params_used
            })



        #statics
        self.log(f"\n[Clustering-Stage2] cluster_file1_ThreeEle:")
        self.log(f"  The number of itemID to be handled: {sum(len(g) for g in self.token_groups)}")
        self.log(f"  The number of updated k-v pairs: {total_keys_updated}")
        self.log(f"  len of cluster_file1_ThreeEle : {len(ThreeEle_clustered_sorted_data)}")

        self.log(f"\n[Clustering-Stage3] Result of clustering-1 (plan):")

        kmeans_clusters = sum(s['total_clusters'] for s in group_stats if s['algorithm'] == 'KMeans')
        dbscan_clusters = sum(s['total_clusters'] for s in group_stats if s['algorithm'] == 'DBSCAN')
        total_cluster_count = kmeans_clusters + dbscan_clusters

        self.log(f"  The number of clusters during clustering (The set cluster of Kmeans + The found cluster of DBSCAN): {total_cluster_count}")


        cluster_index_values = [v['cluster_index'] for v in ThreeEle_clustered_sorted_data.values()]
        unique_cluster_indices = set(cluster_index_values)
        self.log(f"  The number of final used clusters (some clusters in K-means may not be used, cause only few clusters could cover the data) : {len(unique_cluster_indices)}")

        self.log(f"  The number of set clusters in Kmeans: {kmeans_clusters}")
        self.log(f"  The number of found clusters in DBSCAN: {dbscan_clusters}")

        self.log(f"\n[Clustering-Stage4] Result of clustering-2 (reality):")
        self.log(f"  The number of set clusters of Kmeans: {kmeans_clusters}")
        self.log(f"  The number of found cluster of DBSCAN: {dbscan_clusters}")
        self.log(f"  [important] The number of cluster_index in use: {len(unique_cluster_indices)}")
        self.log(f"  The unused clusters from Kmeans: {unused_cluster_count}")
        self.log(f"  The number of ideal clusters (set in Kmeans + found in DBSCAN): {total_cluster_count}")
        self.log(f"  The ratio of used cluster_index all over the ideal clusters {(len(unique_cluster_indices) / total_cluster_count * 100):.2f}%")

        if token_with_unused_clusters:
            self.log(f"\n  The situation of unused clusters from Kmeans:")
            self.log(f"    There are {len(token_with_unused_clusters)}  ItemID(token_id) with clusters unused:")
            for tid, ki, used, miss in token_with_unused_clusters[:10]:
                self.log(f"    ItemID(token_id)={tid}: Ki={ki}, unsed={used}, unsed={miss}")
        else:
            self.log("\n  The kmeans algorithm leaves no clusters unused.")

     
        self.log("\n[Clustering-Stage5]  The whole information:")
        self.log("\n[Final result]:")
        self.log("{:<8} {:<9}  {:<9} {:<10} {:<12} {:<14} {:<31}".format(
            "|Group|", "|n_ItemID(token_id)|", "|algorithm|", "|ideal(set)_clusters|", "|used_clusters|", "|clusters_in_use_per_item|", "|parameters_of_algorithm|"))
        for stat in group_stats:
            self.log("{:<8} {:<13} {:<20} {:<25} {:<26} {:<30.2f} {:<31}".format(
                f"No {stat['group_index'] + 1}",
                stat['token_count'],
                stat['algorithm'],
                stat['total_clusters'],
                stat['used_clusters'],
                stat['avg_clusters_per_token'],
                stat['params']))

        self.log(f"\nTotally: {len(self.token_groups)} groups, {sum(s['token_count'] for s in group_stats)} itemID(token_id), "
              f"ideal clusters: {sum(s['total_clusters'] for s in group_stats)}, "
              f"clusters in use: {sum(s['used_clusters'] for s in group_stats)}, "
              f"unused clusters: {unused_cluster_count}")
        self.log("")
        n_items = sum(s['token_count'] for s in group_stats)
        n_cluster_inuse = sum(s['used_clusters'] for s in group_stats)
        n_inter = len(ThreeEle_clustered_sorted_data)
        self.log(f"We have totally {n_items} items, if we use static tokenizer, we "
                 f"will have {n_items} sids, but now, we will use {n_cluster_inuse} sids to represent {n_items} items."
                 f"The number of sid per item has is {round((n_cluster_inuse * 1.0) / n_items, 2)}. "
                 f"And if we do not cluster, we will assign totally {n_inter} sids to "
                 f"{n_items}, The number of sid per item has in this situation is {(n_inter * 1.0) / n_items}. "
                 f"We reduce {round((((n_inter - n_cluster_inuse) * 1.0) / n_inter) * 100, 2)} % of the full-personalized tokenizer, "
                 f"And we use {round(((n_cluster_inuse * 1.0) / n_items) * 100, 2)} % of sids than static tokenizer.")
        #
        #
        #
        return ThreeEle_clustered_sorted_data,group_stats



    def statics_of_token_group(self):
        self.log(f"[Clustering-Stage1] Statics of group( totally {sum(self.itemID_count.values())}  samples):")
        for i, group in enumerate(self.token_groups):
            total_samples = sum(self.itemID_count[t] for t in group)
            self.log(
                f"  No. {i + 1} group: n_itemID={len(group)}, n_interactions={total_samples}, para={self.K_list[i]}, range=[{int(self.quantile_bounds[i])}, {int(self.quantile_bounds[i + 1])}]")

    def count_itemID_with_interactions(self):
        itemID_count = defaultdict(int)
        itemID_to_interactionKey = defaultdict(list)
        # key in self.sorted_data is A-B-C, userID-itemID-preorderedSeq
        for key in self.sorted_data:
            _, itemID_str, _ = key.split('-')
            itemID = int(itemID_str)
            itemID_count[itemID] += 1
            itemID_to_interactionKey[itemID].append(key)
        return itemID_count,itemID_to_interactionKey



    def split_itemID_by_sample_count_quantiles(self, itemID_count, T, split_ratio):
   
        if not isinstance(split_ratio, list) or len(split_ratio) != T:
            raise ValueError(f"split_ratio must be a list of length T (T={T}).")
        if not np.isclose(sum(split_ratio), 1.0):
            raise ValueError(f"sum of split_ratio must be 1.0, but got {sum(split_ratio)}.")

     
        interaction_counts = sorted(set(itemID_count.values()))

        self.n_items=len(itemID_count.keys())

        total_unique_interactions = len(interaction_counts) 
        self.log(f'total_unique_interactions: {total_unique_interactions}')

        group_boundaries = []
        current_index = 0  

    
        for i in range(T - 1):  
            ratio = split_ratio[i]

            boundary_index = int(ratio * total_unique_interactions)  
            real_index=current_index+boundary_index
            current_index=real_index
   
            group_boundaries.append(interaction_counts[real_index - 1])

      
        group_boundaries.append(interaction_counts[-1])  

     
        self.log(f"Group boundaries: {group_boundaries}")

      
        if len(group_boundaries) != T:
            raise ValueError("The number of group boundaries does not match T.")

   
        groups = [[] for _ in range(T)]

        num_append_to_groupd_interactions=0
   
        for itemID, interaction_count in itemID_count.items():
            for i in range(T):
                if interaction_count <= group_boundaries[i]:
                    groups[i].append(itemID)
                    num_append_to_groupd_interactions=num_append_to_groupd_interactions+1
                    break
        # to check if the times of append interaction is equal to n_items
        self.log(f'self.n_items = {self.n_items}')
        self.log(f'num_append_to_groupd_interactions = {num_append_to_groupd_interactions}')

        if self.n_items==num_append_to_groupd_interactions:
            self.log('self.n_items=num_append_to_groupd_interactions')
        else:
            raise  ValueError("self.n_items    must     =    num_append_to_groupd_interactions!!!!!!!")

     
        for i, group in enumerate(groups):
            self.log(f"Group {i + 1}: {len(group)} items")


        if any(len(group) == 0 for group in groups):
            self.log("DEBUG: empty groups encountered, reviewing group boundaries.")
            raise ValueError("Some groups have no items assigned.")


        min_counts = [float('inf')] * T
        max_counts = [0] * T

        for i, group in enumerate(groups):
            group_interactions = [itemID_count[t] for t in group]
            if len(group_interactions) > 0:  
                min_counts[i] = min(group_interactions)
                max_counts[i] = max(group_interactions)

        quantile_bounds = [min_counts[0]] + [max_counts[i] for i in range(T)]

        return groups, quantile_bounds


    def read_diff_emb(self):

        diff_filename = f"{self.config_upstream['diff_emb_name']}"

        diff_emb_path = os.path.join(self.config_upstream['dict_dir'], self.config_upstream['dataset'],
                                     self.config_upstream['category'], diff_filename)

        with open(diff_emb_path, 'rb') as f:
            data = pickle.load(f)

        self.log(f"successfully load {diff_filename} from {diff_emb_path}")
        self.log(f"the number of k-v pairs of {diff_filename} is {len(data)}")

        return data

    def sorted_by_itemID(self,key: str):
        """
        parse key='user-item_token-preordered' is (item_token:int, user:int, preordered:str)
        used for sorting, but do not change the key itself.
        """

        user_str, item_token_str, preorder_str = key.split('-')
        return (int(item_token_str), int(user_str), preorder_str)


    def sorted_by_itemID_return_dict(self,data):

        sorted_items = sorted(data.items(), key=lambda kv: self.sorted_by_itemID(kv[0]))

        return dict(sorted_items)


