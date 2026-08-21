import os
import shutil
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cudf
# import tensorflow as tf
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path
import data_utils
import utils
warnings.filterwarnings('ignore')
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

def load_data_delays():
    df_latency_filtered = data_utils.load_latency_data("./datasets/donnees_topologiques_brutes.csv")
    inverse_cdf_backhaul = data_utils.get_inverse_cdf(df_latency_filtered["back-hauling-delay"])
    inverse_cdf_synack = data_utils.get_inverse_cdf(df_latency_filtered["delay_syn_synack"])
    return inverse_cdf_backhaul, inverse_cdf_synack

def generate_FL_data(dataset, seeds,nbBrokers, nbclients, nbDSPerBroker, deactRatio,generateConvFL,bitrate):
    # -------------------- Load and Process Latency Data --------------------
    inverse_cdf_backhaul, inverse_cdf_synack = load_data_delays()
    # -------------------- Load and Preprocess Dataset --------------------
    if dataset == '5G3E':
        # 1. Load and Transform Data
        df_train, df_test, y_test = data_utils.load_and_preprocess_5G3E_data()
        
        x_train, pca_model = data_utils.apply_pca_with_cuml(df_train, n_components=20)
        test_df = data_utils.apply_pca_with_cuml(df_test, n_components=20, pca_model=pca_model)
        test_df["label"] = y_test

        train_df, val_df = train_test_split(x_train, test_size=0.2, random_state=42, shuffle=True)

        Path("FLSplit_5G3E/JSD_Matrixes").mkdir(parents=True, exist_ok=True)
        os.chdir("FLSplit_5G3E")

        files_to_save = {
            "5G3E_validation.csv": val_df,
            "5G3E_test.csv": test_df
        }

    elif dataset=='CICIDS2017':
        df_train = cudf.read_csv('./datasets/CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX.csv').to_pandas()
        df_test = cudf.read_csv('./datasets/CIC-IDS-2017/Wednesday-workingHours.pcap_ISCX.csv').to_pandas()


        df_train_features, y_train, scaler = data_utils.data_preprocessing_CICIDS2017(df_train, fit_scaler=True)
        df_test_features, y_test,scaler = data_utils.data_preprocessing_CICIDS2017(df_test, scaler=scaler, fit_scaler=True)  
        x_train_df, pca_model = data_utils.apply_pca_with_cuml(df_train_features, n_components=11)
        test_df = data_utils.apply_pca_with_cuml(df_test_features, n_components=11)

        train_df, val_df = train_test_split(x_train_df, test_size=0.2, random_state=42, shuffle=True)
    #     test_df["label"] = y_test.to_numpy()

    #     Path("FLSplit_CICIDS2017/JSD_Matrixes").mkdir(parents=True, exist_ok=True)
    #     os.chdir("FLSplit_CICIDS2017")

    #     files_to_save = {
    #         "CICIDS_validation.csv": val_df,
    #         "CICIDS_test.csv": test_df
    #     }

    # for filename, df in files_to_save.items():
    #     if not Path(filename).exists():
    #         df.to_csv(filename, index=False)    

    # print("Dataset Shapes after preprocessing ")
    # print("Training shape:", train_df.shape)
    # print("Validation shape:", val_df.shape)
    # print("Test shape: ",test_df.shape)

    #############################################################################################################################
    broker_range = [5, 8, 12, 16,32, 64, 128, 256]
    T_values = []
    c_values = []
    N=5
    S = 1
    deactivation_ratios = [0, 0.2, 0.4, 0.6, 0.8]
    bitrate=0.000444928/1000 # bitrate conversion from Mbps to Mbpms (because delays are in ms)

    global_plot_data = []  # <-- stocke tous les résultats

    for ratio in deactivation_ratios:
        for B in broker_range:
            print(f"\n\n======= Running simulation with deactivation_ratio ={ratio}, {B} Brokers, {N} AIFs, and {S} DS per Broker ============================")
            utils.set_random_seeds()

            ds_list = [f"DS_{i}" for i in range(B * S)]

            # Cluster and split the data
            labels = KMeans(n_clusters=len(ds_list), random_state=42).fit_predict(train_df)
            ds_data = [train_df[labels == i] for i in range(len(ds_list))]

            broker_ds_map = {f"B{b}": ds_list[b * S:(b + 1) * S] for b in range(B)}
            broker_aif_map = {f"B{b}": [f"AIF_{i}" for i in range(N)] for b in range(B)}

            latency_ds_to_broker = {}
            latency_broker_to_aif = {}
            for broker, aifs in broker_aif_map.items():
                for aif_idx, aif in enumerate(aifs):
                    # For simplicity, assign DS round-robin to AIFs
                    ds_idx = aif_idx % S
                    if ds_idx < len(broker_ds_map[broker]):
                        ds = broker_ds_map[broker][ds_idx]
                        u = np.random.rand()
                        latency_broker_to_aif[(broker, aif)] = float(inverse_cdf_backhaul(u))
                        latency_ds_to_broker[(ds, broker)] = float(inverse_cdf_synack(u))

            updated_broker_aif_map, updated_latency_broker_to_aif = utils.deactivate_per_broker2(
                latency_broker_to_aif, broker_aif_map, ratio)

            max_ds_to_broker_latency = max(latency_ds_to_broker.values())
            max_broker_to_aif_latency = max(updated_latency_broker_to_aif.values())
            T = max_ds_to_broker_latency + max_broker_to_aif_latency
            T_values.append(T)

            aif_data_map = defaultdict(list)
            c1, c2 = 0, 0

            for b, ds_list in broker_ds_map.items():
                ds_data_copy = ds_data.copy()
                utils.set_random_seeds()
                aif_choices = updated_broker_aif_map[b]
                print("Broker ", b, " has AIF choices: ", aif_choices)

                for ds in ds_list:
                    data_chunk = ds_data_copy[int(ds.split('_')[1]) - 1]
                    n_rows = data_chunk.shape[0]
                    c1 += n_rows * latency_ds_to_broker[(ds, b)]

                    selected_aifs = np.random.choice(aif_choices, size=n_rows)
                    unique, counts = np.unique(selected_aifs, return_counts=True)
                    c2 += sum(counts[i] * updated_latency_broker_to_aif[(b, unique[i])] for i in range(len(unique)))

                    data_chunk['aif'] = selected_aifs
                    for aif, group in data_chunk.groupby('aif'):
                        aif_data_map[aif].append(group.drop(columns='aif'))

            C = (c1 + c2) * bitrate
            c_values.append(C)

            aif_partitions = {aif: pd.concat(chunks, ignore_index=True) for aif, chunks in aif_data_map.items()}

            total_dataframe = pd.DataFrame()
            for aif, data in aif_partitions.items():
                print(f"{aif} has {len(data)} samples.")
                data = data.copy()
                data['AIFID'] = np.int32(aif.split('_')[1])
                total_dataframe = pd.concat([total_dataframe, data], ignore_index=True)

            jsd_df_broker = utils.compute_jsd_matrix(total_dataframe)
            print(jsd_df_broker)
            jsd_avg_uniform = jsd_df_broker.iloc[:, -1].tolist()

            # Ajoute les résultats à la liste globale avec ratio
            global_plot_data.extend([
                {'Brokers': B, 'JSD Avg Uniform': v, 'Deactivation Ratio': f"{int(ratio * 100)}%"}
                for v in jsd_avg_uniform
            ])
    
    # ========== TRACÉ GLOBAL ==========
    df_all = pd.DataFrame(global_plot_data)
    points = (df_all
          .groupby(['Brokers', 'Deactivation Ratio'])['JSD Avg Uniform']
          .mean()
          .reset_index())

    points.to_csv("vary_number_of_brokers_points.csv", index=False)
    df_all.to_csv("vary_number_of_brokers_data.csv", index=False)

    plt.figure(figsize=(12, 7))
    # sns.boxplot(x='Brokers', y='JSD Avg Uniform', hue='Deactivation Ratio', data=df_all, palette='tab10')
    sns.pointplot(x='Brokers', y='JSD Avg Uniform', hue='Deactivation Ratio', data=df_all,
                dodge=0, markers='D', linestyles='-', errorbar=None, palette='tab10', alpha=0.9)

    plt.title("JSD Avg vs Number of Brokers for Different Deactivation Ratios", fontsize=16)
    plt.xlabel("Number of Brokers", fontsize=14)
    plt.ylabel("JSD Avg", fontsize=14)
    plt.legend(title="Deactivation Ratio", title_fontsize=12, fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("vary_number_of_brokers_figure.pdf",format="pdf")







    jsd_matrices = []
    if generateConvFL==False:
        for seed in seeds:
            utils.set_random_seeds(seed)
            ds_list = [f"DS_{i}" for i in range(nbBrokers * nbDSPerBroker)]
            labels = KMeans(n_clusters=len(ds_list),init='k-means++',random_state=12).fit_predict(train_df)
            ds_data = [train_df[labels == i] for i in range(len(ds_list))]
            print("============")
            broker_ds_map = {f"B{b}": ds_list[b * nbDSPerBroker:(b + 1) * nbDSPerBroker] for b in range(nbBrokers)}
            broker_aif_map = {f"B{b}": [f"AIF_{i}" for i in range(nbclients)] for b in range(nbBrokers)}

            # -------------------- Generate Latencies --------------------
            latency_broker_to_aif = {}

            for broker, aifs in broker_aif_map.items():
                for aif_idx, aif in enumerate(aifs):
                    ds_idx = aif_idx % nbDSPerBroker
                    if ds_idx < len(broker_ds_map[broker]):
                        ds = broker_ds_map[broker][ds_idx]
                        u = np.random.rand()
                        latency_broker_to_aif[(broker, aif)] = float(inverse_cdf_backhaul(u))

            print("broker_ds_map:", broker_ds_map)
            print("broker_aif_map:", broker_aif_map)
            print("latency_broker_to_aif:", latency_broker_to_aif)

            # -------------------- Deactivate AIFs --------------------
            updated_broker_aif_map, updated_latency_broker_to_aif = utils.deactivate_per_broker2(
                latency_broker_to_aif, broker_aif_map, deactivation_ratio=deactRatio
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

            print("**"*12)
            # -------------------- Compute and Save Communication Cost --------------------
            total_cost = (comm_cost_broker_to_aif) *bitrate

            print("Total communication cost:", total_cost)
            print("**"*12)
            # -------------------- Save Final Dataset --------------------
            split_type=f"brokerBased-{int(deactRatio*100)}"
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
            jsd_df_broker = utils.compute_jsd_matrix(total_dataframe)
            plt.figure(figsize=(10, 8))
            sns.heatmap(jsd_df_broker, cmap="Reds", annot=True, fmt=".3f")
            plt.xlabel("Clients")
            plt.ylabel("Features")
            plt.title("Jensen-Shannon Divergence Matrix: The divergence of client i from the global state")
            plt.savefig(f'./JSD_Matrixes/JSD_matrix_seed_{seed}_Deact-{int(deactRatio*100)}_{dataset}.pdf', bbox_inches='tight')
            jsd_matrices.append(jsd_df_broker)

        # -------------------- Aggregate JSD Results --------------------
        jsd_array = np.stack([df.values for df in jsd_matrices], axis=2) 
        mean_jsd = np.mean(jsd_array, axis=2)
        h_jsd = np.zeros_like(mean_jsd)

        # Convert to DataFrame with column names
        mean_jsd_df = pd.DataFrame(mean_jsd, columns=[f"AIF {c+1}" for c in range(nbclients)] + ["Avg"])

        # Compute confidence intervals
        for i in range(mean_jsd.shape[0]):
            for j in range(mean_jsd.shape[1]):
                _, h = utils.confidence_interval(jsd_array[i, j, :])
                h_jsd[i, j] = h

        annotations = np.empty_like(mean_jsd, dtype=object)
        for i in range(mean_jsd.shape[0]):
            for j in range(mean_jsd.shape[1]):
                annotations[i, j] = f"{mean_jsd[i, j]:.3f}\n±{h_jsd[i, j]:.3f}"

        plt.figure(figsize=(10, 8))
        sns.heatmap(mean_jsd_df, annot=annotations, fmt="", cmap="Blues", cbar=False,vmin=0, vmax=0.6, annot_kws={"size": 17, "va": "center", "ha": "center"})
        plt.xlabel("", fontsize=17)
        plt.ylabel("Features", fontsize=17)
        plt.yticks(fontsize=17) 
        plt.xticks(fontsize=17)
        plt.savefig(f'JSD_matrix_{int(deactRatio*100)}_{dataset}.pdf', bbox_inches='tight')

    else:
        utils.Conv_split(train_df, nbBrokers,12)  # Here nbBrokers should be the same as the number of clients



def main():
    """
    Main entry point for Federated Learning data generation.
    
    Configuration Parameters:
    - dataset: Choose between '5G3E' or 'CICIDS2017'
    - seeds: List of random seeds for reproducibility
    - nbBrokers: Number of brokers in the network
    - nbclients: Number of clients per broker
    - nbDSPerBroker: Number of data sources per broker
    - deactRatio: Deactivation ratio for AIFs (0.0 to 1.0)
    - generateConvFL: If True, use conventional split; if False, use broker-based split
    """
    
    # -------------------- Configuration --------------------
    dataset = 'CICIDS2017'  # 'CICIDS2017' or '5G3E'
    nbBrokers = 5
    nbclients = 5
    nbDSPerBroker = 1
    deactRatio = 0.6
    generateConvFL = True  # Set to True for conventional FL split
    seeds = [12]

    if dataset == 'CICIDS2017':
        bitrate = 0.000444928 / 1000  # bitrate conversion from Mbps to Mbpms (because delays are in ms)
    elif dataset == '5G3E':
        bitrate = 0.00424959 / 1000  # bitrate conversion from Mbps to Mbpms (because delays are in ms)
    
    # -------------------- Run FL Data Generation --------------------
    print("=" * 60)
    print("Starting Federated Learning Data Generation")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Number of Brokers: {nbBrokers}")
    print(f"Number of Clients per Broker: {nbclients}")
    print(f"Data Sources per Broker: {nbDSPerBroker}")
    print(f"Deactivation Ratio: {deactRatio}")
    print(f"Generate Conventional FL: {generateConvFL}")
    print("=" * 60)
    
    try:
        generate_FL_data(
            dataset=dataset,
            seeds=seeds,
            nbBrokers=nbBrokers,
            nbclients=nbclients,
            nbDSPerBroker=nbDSPerBroker,
            deactRatio=deactRatio,
            generateConvFL=generateConvFL,
            bitrate=bitrate
        )
        print("\n" + "=" * 60)
        print("FL Data Generation Completed Successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError during FL data generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()