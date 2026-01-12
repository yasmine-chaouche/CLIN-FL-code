import os
import random
import shutil
import importlib
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cudf
import tensorflow as tf
from numba import cuda
from sklearn.cluster import KMeans
import matplotlib.pyplot as pltinterp1d
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys
import utils_5G3Ev1
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
warnings.filterwarnings('ignore')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Reload utils in case it's updated
importlib.reload(utils_5G3Ev1)

# -------------------- Load and Process Latency Data --------------------
df_latency_filtered = utils_5G3Ev1.load_latency_data("/etudiants/siscol/c/chaou_ya/Yasmine/MyThesisCode/TCP-traffic-flow/donnees_topologiques_brutes.csv")
inverse_cdf_backhaul = utils_5G3Ev1.get_inverse_cdf(df_latency_filtered["back-hauling-delay"])
inverse_cdf_synack = utils_5G3Ev1.get_inverse_cdf(df_latency_filtered["delay_syn_synack"])

# -------------------- Load and Preprocess Dataset --------------------
df_train_preprocessed, df_test_preprocessed, y_test = utils.load_and_preprocess_5G3E_data()
x_train = utils.apply_pca_with_cuml(df_train_preprocessed, n_components=20)
x_test = utils.apply_pca_with_cuml(df_test_preprocessed, n_components=20)
train_df, val_df = train_test_split(x_train, test_size=0.2, random_state=42, shuffle=True)
# val_df.to_csv("validation.csv", index=False)
print("Training shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print(f"\n Dataset Shapes after preprocessing: \n Train: {train_df.shape} \n Test: {x_test.shape}")


B, N, S = 5, 5, 1  # Brokers, AIFs per Broker, DS per Broker
deact_ratio=0.8
seeds = [254,74, 145,541,325,852,954,222]#[12,23, 42, 49, 55, 102,78,99] #,254,74, 145,541,325,852,954,222,333,741,5498,79, 66,24,8542,96587,54876,1422,11,43,757,6842]
bitrate=0.00424959/1000 # bitrate conversion from Mbps to Mbpms (because delays are in ms)
jsd_matrices = []

generate_Conv_FL=False

if generate_Conv_FL==False:
    for seed in seeds:
        utils_5G3Ev1.set_random_seeds(seed)
        ds_list = [f"DS_{i}" for i in range(B * S)]
        labels = KMeans(n_clusters=len(ds_list),init='k-means++',random_state=12).fit_predict(train_df)
        ds_data = [train_df[labels == i] for i in range(len(ds_list))]
        print("============")
        broker_ds_map = {f"B{b}": ds_list[b * S:(b + 1) * S] for b in range(B)}
        broker_aif_map = {f"B{b}": [f"AIF_{i}" for i in range(N)] for b in range(B)}

        # -------------------- Generate Latencies --------------------
        latency_broker_to_aif = {}

        for broker, aifs in broker_aif_map.items():
            for aif_idx, aif in enumerate(aifs):
                ds_idx = aif_idx % S
                if ds_idx < len(broker_ds_map[broker]):
                    ds = broker_ds_map[broker][ds_idx]
                    u = np.random.rand()
                    latency_broker_to_aif[(broker, aif)] = float(inverse_cdf_backhaul(u))

        print("broker_ds_map:", broker_ds_map)
        print("broker_aif_map:", broker_aif_map)
        print("latency_broker_to_aif:", latency_broker_to_aif)

        # -------------------- Deactivate AIFs --------------------
        updated_broker_aif_map, updated_latency_broker_to_aif = utils_5G3Ev1.deactivate_per_broker2(
            latency_broker_to_aif, broker_aif_map, deactivation_ratio=deact_ratio
        )

        print("**********************************************************")
        # -------------------- Assign Data to AIFs --------------------
        aif_data_map = defaultdict(list)
        comm_cost_broker_to_aif = 0

        for broker, ds_ids in broker_ds_map.items():
            aif_choices = updated_broker_aif_map[broker]
            print("Broker", broker, "has AIF choices:", aif_choices)

            for ds in ds_ids:
                ds_idx = int(ds.split('_')[1]) - 1
                data_chunk = ds_data[ds_idx]
                n_rows = data_chunk.shape[0]
                selected_aifs = np.random.choice(aif_choices, size=n_rows)
                unique, counts = np.unique(selected_aifs, return_counts=True)
                comm_cost_broker_to_aif += sum(counts[i] * updated_latency_broker_to_aif[(broker, unique[i])] for i in range(len(unique)))

                data_chunk['aif'] = selected_aifs
                for aif, group in data_chunk.groupby('aif'):
                    aif_data_map[aif].append(group.drop(columns='aif'))

        print("**********************************************************")
        # -------------------- Compute and Save Communication Cost --------------------
        total_cost = (comm_cost_broker_to_aif) *bitrate

        print("Total communication cost:", total_cost)
        print("**********************************************************")
        # -------------------- Save Final Dataset --------------------
        split_type=f"brokerBased-{int(deact_ratio*100)}"
        save_dir = f'./seed_{seed}_results/{split_type}_split'

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        aif_partitions = {aif: pd.concat(chunks, ignore_index=True) for aif, chunks in aif_data_map.items()}
        total_dataframe = pd.DataFrame()

        for aif, data in aif_partitions.items():
            print(f"{aif} has {len(data)} samples.")
            data = data.copy()
            data.to_csv(f'./seed_{seed}_results/{split_type}_split/{aif}_{split_type}.csv', index=False)  
            data['AIFID'] = np.int32(aif.split('_')[1])
            total_dataframe = pd.concat([total_dataframe, data], ignore_index=True)

        # -------------------- Compute JSD Matrix --------------------
        jsd_df_broker = utils_5G3Ev1.compute_jsd_matrix(total_dataframe)
        plt.figure(figsize=(10, 8))
        sns.heatmap(jsd_df_broker, cmap="Reds", annot=True, fmt=".3f")
        plt.xlabel("Clients")
        plt.ylabel("Features")
        plt.title("Jensen-Shannon Divergence Matrix: The divergence of client i from the global state")
        plt.savefig(f'./JSD_Matrixes/JSD_matrix_seed_{seed}_Deact-{int(deact_ratio*100)}_5G3E.pdf', bbox_inches='tight')
        # jsd_matrices.append(jsd_df_broker)
else:
    utils_5G3Ev1.Conv_split(train_df, B,12)  # Here B should be the number of client 

