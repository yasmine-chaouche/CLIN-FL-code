import argparse
import utils
import pandas as pd
import numpy as np
import cudf as cdf
from sklearn.cluster import KMeans
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from sklearn.preprocessing import MinMaxScaler


def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--nbclients',type=int,default=5,help="Number of clients")
    parser.add_argument('--nbBrokers',type=int,default=5,help="Number of brokers")
    parser.add_argument('--nbDSPerBroker',type=int,default=1,help="Number of datasources per broker")
    parser.add_argument('--dataset',type=str,default='CICIDS2017',help="dataset used for FL simulation")
    parser.add_argument('--model',type=str,default='autoencoder',help="model used for FL simulation: autoencoder or DASVDD")
    parser.add_argument('--threshold',type=float,default=0.09,help="Variance threshold for feature selection")
    parser.add_argument('--deactRatio',type=float,default=0.8,help="Ratio of deactivated links per broker")
    args=parser.parse_args()
    return args


if __name__=='__main__':
    args=get_args()
    seeds = [12]#[12,23, 42, 49, 55, 102,78,99,254,74, 145,541,325,852,954,222,333,741,5498,79, 66,24,8542,96587,54876,1422,11,43,757,6842]
    bitrate_CICIDS2017 = 0.000444928 / 1000 # bitrate conversion from Mbps to Mbpms (because delays are in ms)
    df_latency_filtered = utils.load_latency_data("./datasets/donnees_topologiques_brutes.csv")
    inverse_cdf_backhaul = utils.get_inverse_cdf(df_latency_filtered["back-hauling-delay"])
    inverse_cdf_synack = utils.get_inverse_cdf(df_latency_filtered["delay_syn_synack"])
    jsd_matrices = []
    if args.dataset=='CICIDS2017':
        df_train = cdf.read_csv('./datasets/CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX.csv').to_pandas()
        df_test=cdf.read_csv('./datasets/CIC-IDS-2017/Tuesday-WorkingHours.pcap_ISCX.csv').to_pandas()
        df_train_preprocessed=utils.data_preprocessing_CICIDS2017(df_train)
        df_test_preprocessed=utils.data_preprocessing_CICIDS2017(df_test)
        print(df_train_preprocessed.columns.equals(df_test_preprocessed.columns))

        y_test = df_test_preprocessed[' Label']

        df_train_features = df_train_preprocessed.drop(columns=[' Label'], errors='ignore')
        df_test_features = df_test_preprocessed.drop(columns=[' Label'],errors='ignore')

        # Concatenate
        combined_df = pd.concat([df_train_features, df_test_features], axis=0)

        pca_combined_data=utils.apply_pca_with_cuml(combined_df, n_components=11)

        #Datasets ready to use
        x_train = pca_combined_data[:len(df_train)]

        x_test = pca_combined_data[len(df_train):]

        scaler = MinMaxScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
        x_test  = pd.DataFrame(scaler.transform(x_test),  columns=x_test.columns)


        print(f"\n Dataset Shapes after preprocessing: \n Train: {x_train.shape} \n Test: {x_test.shape}")
        print(y_test.value_counts())

    for seed in seeds:
        utils.set_random_seeds(seed)
        ds_list = [f"DS_{i}" for i in range(args.nbBrokers * args.nbDSPerBroker)]
        labels = KMeans(n_clusters=len(ds_list),init='k-means++',random_state=12).fit_predict(x_train)
        ds_data = [x_train[labels == i] for i in range(len(ds_list))]
        print("============")
        broker_ds_map = {f"B{b}": ds_list[b * args.nbDSPerBroker:(b + 1) * args.nbDSPerBroker] for b in range(args.nbBrokers)}
        broker_aif_map = {f"B{b}": [f"AIF_{i}" for i in range(args.nbclients)] for b in range(args.nbBrokers)}

        # -------------------- Generate Latencies --------------------
        latency_broker_to_aif = {}

        for broker, aifs in broker_aif_map.items():
            for aif_idx, aif in enumerate(aifs):
                ds_idx = aif_idx % args.nbDSPerBroker
                if ds_idx < len(broker_ds_map[broker]):
                    ds = broker_ds_map[broker][ds_idx]
                    u = np.random.rand()
                    latency_broker_to_aif[(broker, aif)] = float(inverse_cdf_backhaul(u))

        print("broker_ds_map:", broker_ds_map)
        print("broker_aif_map:", broker_aif_map)
        print("latency_broker_to_aif:", latency_broker_to_aif)

        # -------------------- Deactivate AIFs --------------------
        updated_broker_aif_map, updated_latency_broker_to_aif = utils.deactivate_per_broker(
            latency_broker_to_aif, broker_aif_map, deactivation_ratio=args.deactRatio
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

        print("*"*20)
        # -------------------- Compute and Save Communication Cost --------------------
        total_cost = (comm_cost_broker_to_aif) *bitrate_CICIDS2017

        print("Total communication cost:", total_cost)
        print("*"*20)
        # -------------------- Save Final Dataset --------------------
        split_type=f"brokerBased-{int(args.deactRatio*100)}"
        save_dir = f'./FLSplit_CIC-IDS-2017/seed_{seed}_results/{split_type}_split'

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        aif_partitions = {aif: pd.concat(chunks, ignore_index=True) for aif, chunks in aif_data_map.items()}
        total_dataframe = pd.DataFrame()

        for aif, data in aif_partitions.items():
            print(f"{aif} has {len(data)} samples.")
            data = data.copy()
            data.to_csv(f'./FLSplit_CIC-IDS-2017/seed_{seed}_results/{split_type}_split/{aif}_{split_type}.csv', index=False)  # Optional saving
            data['AIFID'] = np.int32(aif.split('_')[1])
            total_dataframe = pd.concat([total_dataframe, data], ignore_index=True)

        # -------------------- Compute JSD Matrix --------------------
        jsd_df_broker = utils.compute_jsd_matrix(total_dataframe)
        jsd_matrices.append(jsd_df_broker)


    # -------------------- Aggregate JSD Results --------------------
    jsd_array = np.stack([df.values for df in jsd_matrices], axis=2)  # shape: (n_features, n_clients+1, n_seeds)
    mean_jsd = np.mean(jsd_array, axis=2)
    h_jsd = np.zeros_like(mean_jsd)


    # Convert to DataFrame with column names
    mean_jsd_df = pd.DataFrame(mean_jsd, columns=[f"AIF {c+1}" for c in range(args.nbclients)] + ["Avg"])

    # Compute confidence intervals
    for i in range(mean_jsd.shape[0]):
        for j in range(mean_jsd.shape[1]):
            _, h = utils.confidence_interval(jsd_array[i, j, :])
            h_jsd[i, j] = h

    # Create annotations with mean ± h
    annotations = np.empty_like(mean_jsd, dtype=object)
    for i in range(mean_jsd.shape[0]):
        for j in range(mean_jsd.shape[1]):
            annotations[i, j] = f"{mean_jsd[i, j]:.3f}\n±{h_jsd[i, j]:.3f}"

    # -------------------- Plot Final Annotated JSD Heatmap --------------------
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_jsd_df, annot=annotations, fmt="", cmap="Blues", cbar=False,vmin=0, vmax=0.6, annot_kws={"size": 17, "va": "center", "ha": "center"})
    plt.xlabel("", fontsize=17)
    plt.ylabel("Features", fontsize=17)
    plt.yticks(fontsize=17)
    plt.xticks(fontsize=17)
    plt.savefig(f'JSD_matrix_{int(args.deactRatio*100)}_CICIDS2027.pdf', bbox_inches='tight')